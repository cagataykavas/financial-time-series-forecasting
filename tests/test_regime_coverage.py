from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.regime_coverage import (
    REGIME_COVERAGE_SCHEMA,
    RegimeCoverageError,
    RegimeCoveragePolicy,
    RegimeIntervalObservation,
    audit_regime_coverage,
    require_regime_coverage,
)


def _policy(**overrides: object) -> RegimeCoveragePolicy:
    values = {
        "regimes": ("high-volatility", "low-volatility", "normal-volatility"),
        "target_coverage": 0.9,
        "tolerated_shortfall": 0.1,
        "confidence_level": 0.95,
        "min_observations_per_regime": 20,
    }
    values.update(overrides)
    return RegimeCoveragePolicy(**values)


def _observations(
    per_regime: int = 60,
    misses: dict[str, set[int]] | None = None,
) -> list[RegimeIntervalObservation]:
    misses = misses or {}
    issued_at = datetime(2026, 1, 1, tzinfo=UTC)
    result: list[RegimeIntervalObservation] = []
    regimes = ("high-volatility", "low-volatility", "normal-volatility")
    for position in range(per_regime * len(regimes)):
        regime = regimes[position % len(regimes)]
        regime_position = position // len(regimes)
        issued = issued_at + timedelta(hours=position)
        missed = regime_position in misses.get(regime, set())
        result.append(
            RegimeIntervalObservation(
                issued_at=issued,
                observed_at=issued + timedelta(days=1),
                regime=regime,
                regime_as_of=issued - timedelta(minutes=1),
                lower=-1.0,
                upper=1.0,
                actual=2.0 if missed else 0.0,
            )
        )
    return result


def test_accepts_well_covered_predeclared_regimes() -> None:
    report = require_regime_coverage(_observations(), _policy())

    assert report["schema_version"] == REGIME_COVERAGE_SCHEMA
    assert report["accepted"] is True
    assert report["finding_codes"] == []
    assert report["aggregate_observed_coverage"] == 1.0
    assert [row["regime"] for row in report["regimes"]] == list(_policy().regimes)
    assert all(row["simultaneous_lower_bound"] >= 0.8 for row in report["regimes"])


def test_rejects_hidden_high_volatility_undercoverage() -> None:
    misses = {"high-volatility": set(range(24))}

    report = audit_regime_coverage(_observations(misses=misses), _policy())

    assert report["aggregate_observed_coverage"] > 0.85
    assert report["accepted"] is False
    assert report["finding_codes"] == ["coverage_lower_bound_below_policy:high-volatility"]
    high = report["regimes"][0]
    assert high["observed_coverage"] == 0.6
    assert high["upper_misses"] == 24
    assert high["lower_misses"] == 0


def test_missing_and_undersized_regimes_fail_closed() -> None:
    observations = [
        row for row in _observations(per_regime=25) if row.regime != "normal-volatility"
    ]

    report = audit_regime_coverage(
        observations,
        _policy(min_observations_per_regime=30),
    )

    assert report["finding_codes"] == [
        "insufficient_observations:high-volatility",
        "insufficient_observations:low-volatility",
        "insufficient_observations:normal-volatility",
    ]
    assert report["regimes"][2]["observed_coverage"] is None
    assert report["regimes"][2]["simultaneous_lower_bound"] is None


def test_policy_digest_and_json_report_are_deterministic() -> None:
    first = audit_regime_coverage(_observations(), _policy())
    second = audit_regime_coverage(_observations(), _policy())

    assert first == second
    assert len(first["policy_sha256"]) == 64
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["per_regime_alpha"] == pytest.approx((1 - 0.95) / 3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"regimes": ("normal", "high")}, "sorted"),
        ({"regimes": ("normal", "normal")}, "unique"),
        ({"regimes": ("bad label",)}, "canonical"),
        ({"target_coverage": 1.0}, "strictly between"),
        ({"confidence_level": float("nan")}, "strictly between"),
        ({"tolerated_shortfall": 0.9}, "below target"),
        ({"min_observations_per_regime": True}, "integer"),
    ],
)
def test_policy_rejects_ambiguous_or_unbounded_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RegimeCoverageError, match=message):
        _policy(**overrides)


def test_rejects_regime_labels_not_available_at_issuance() -> None:
    rows = _observations()
    row = rows[0]
    rows[0] = RegimeIntervalObservation(
        issued_at=row.issued_at,
        observed_at=row.observed_at,
        regime=row.regime,
        regime_as_of=row.issued_at + timedelta(seconds=1),
        lower=row.lower,
        upper=row.upper,
        actual=row.actual,
    )

    with pytest.raises(RegimeCoverageError, match="not available at issuance"):
        audit_regime_coverage(rows, _policy())


def test_rejects_undeclared_regime_and_unordered_issuance() -> None:
    rows = _observations()
    row = rows[0]
    rows[0] = RegimeIntervalObservation(
        issued_at=row.issued_at,
        observed_at=row.observed_at,
        regime="crisis",
        regime_as_of=row.regime_as_of,
        lower=row.lower,
        upper=row.upper,
        actual=row.actual,
    )
    with pytest.raises(RegimeCoverageError, match="undeclared regime"):
        audit_regime_coverage(rows, _policy())

    rows = _observations()
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(RegimeCoverageError, match="strictly increasing"):
        audit_regime_coverage(rows, _policy())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lower", float("nan"), "lower must be finite"),
        ("upper", float("inf"), "upper must be finite"),
        ("actual", True, "actual must be finite"),
        ("lower", 2.0, "inverted bounds"),
    ],
)
def test_rejects_malformed_interval_values(field: str, value: object, message: str) -> None:
    rows = _observations()
    values = {
        "issued_at": rows[0].issued_at,
        "observed_at": rows[0].observed_at,
        "regime": rows[0].regime,
        "regime_as_of": rows[0].regime_as_of,
        "lower": rows[0].lower,
        "upper": rows[0].upper,
        "actual": rows[0].actual,
    }
    values[field] = value
    rows[0] = RegimeIntervalObservation(**values)

    with pytest.raises(RegimeCoverageError, match=message):
        audit_regime_coverage(rows, _policy())


def test_rejects_naive_or_invalid_event_times() -> None:
    rows = _observations()
    row = rows[0]
    rows[0] = RegimeIntervalObservation(
        issued_at=row.issued_at.replace(tzinfo=None),
        observed_at=row.observed_at,
        regime=row.regime,
        regime_as_of=row.regime_as_of,
        lower=row.lower,
        upper=row.upper,
        actual=row.actual,
    )
    with pytest.raises(RegimeCoverageError, match="timezone-aware"):
        audit_regime_coverage(rows, _policy())

    rows = _observations()
    row = rows[0]
    rows[0] = RegimeIntervalObservation(
        issued_at=row.issued_at,
        observed_at=row.issued_at,
        regime=row.regime,
        regime_as_of=row.regime_as_of,
        lower=row.lower,
        upper=row.upper,
        actual=row.actual,
    )
    with pytest.raises(RegimeCoverageError, match="mature after issuance"):
        audit_regime_coverage(rows, _policy())


def test_interval_boundaries_count_as_covered_and_tail_misses_are_distinct() -> None:
    rows = _observations(per_regime=2)
    rows[0] = RegimeIntervalObservation(
        issued_at=rows[0].issued_at,
        observed_at=rows[0].observed_at,
        regime=rows[0].regime,
        regime_as_of=rows[0].regime_as_of,
        lower=-1.0,
        upper=1.0,
        actual=-1.0,
    )
    rows[3] = RegimeIntervalObservation(
        issued_at=rows[3].issued_at,
        observed_at=rows[3].observed_at,
        regime=rows[3].regime,
        regime_as_of=rows[3].regime_as_of,
        lower=-1.0,
        upper=1.0,
        actual=-2.0,
    )

    report = audit_regime_coverage(rows, _policy(min_observations_per_regime=2))

    high = report["regimes"][0]
    assert high["covered"] == 1
    assert high["lower_misses"] == 1
    assert high["upper_misses"] == 0


def test_require_raises_stable_findings_for_policy_rejection() -> None:
    with pytest.raises(
        RegimeCoverageError,
        match="coverage policy failed: coverage_lower_bound_below_policy:high-volatility",
    ):
        require_regime_coverage(
            _observations(misses={"high-volatility": set(range(30))}),
            _policy(),
        )
