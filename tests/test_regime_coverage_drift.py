from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.regime_coverage import (
    REGIME_COVERAGE_DRIFT_SCHEMA,
    RegimeCoverageDriftPolicy,
    RegimeCoverageError,
    RegimeIntervalObservation,
    audit_regime_coverage_drift,
    require_regime_coverage_drift,
)


def _policy(**overrides: object) -> RegimeCoverageDriftPolicy:
    values = {
        "regimes": ("high-volatility", "normal-volatility"),
        "max_coverage_drop": 0.1,
        "confidence_level": 0.95,
        "min_observations_per_window": 100,
    }
    values.update(overrides)
    return RegimeCoverageDriftPolicy(**values)


def _window(
    start: datetime,
    *,
    per_regime: int = 200,
    misses: dict[str, set[int]] | None = None,
) -> list[RegimeIntervalObservation]:
    misses = misses or {}
    regimes = ("high-volatility", "normal-volatility")
    result: list[RegimeIntervalObservation] = []
    for position in range(per_regime * len(regimes)):
        regime = regimes[position % len(regimes)]
        regime_position = position // len(regimes)
        issued_at = start + timedelta(hours=position)
        result.append(
            RegimeIntervalObservation(
                issued_at=issued_at,
                observed_at=issued_at + timedelta(days=1),
                regime=regime,
                regime_as_of=issued_at - timedelta(minutes=1),
                lower=-1.0,
                upper=1.0,
                actual=2.0 if regime_position in misses.get(regime, set()) else 0.0,
            )
        )
    return result


def test_accepts_stable_regime_coverage() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(
        datetime(2026, 2, 1, tzinfo=UTC),
        misses={"high-volatility": set(range(10))},
    )

    report = require_regime_coverage_drift(reference, current, _policy())

    assert report["schema_version"] == REGIME_COVERAGE_DRIFT_SCHEMA
    assert report["accepted"] is True
    assert report["finding_codes"] == []
    high = report["regimes"][0]
    assert high["observed_coverage_drop"] == pytest.approx(0.05)
    assert high["simultaneous_drop_lower_bound"] < 0.1


def test_detects_hidden_high_volatility_coverage_drop() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(
        datetime(2026, 2, 1, tzinfo=UTC),
        misses={"high-volatility": set(range(80))},
    )

    report = audit_regime_coverage_drift(reference, current, _policy())

    assert report["current_coverage"] == 0.8
    assert report["accepted"] is False
    assert report["finding_codes"] == ["coverage_drop_above_policy:high-volatility"]
    high = report["regimes"][0]
    assert high["reference_coverage"] == 1.0
    assert high["current_coverage"] == 0.6
    assert high["simultaneous_drop_lower_bound"] > 0.1


def test_undersized_or_missing_windows_fail_closed() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=4)
    current = [
        row
        for row in _window(datetime(2026, 2, 1, tzinfo=UTC), per_regime=4)
        if row.regime == "high-volatility"
    ]

    report = audit_regime_coverage_drift(
        reference,
        current,
        _policy(min_observations_per_window=5),
    )

    assert report["finding_codes"] == [
        "insufficient_reference_observations:high-volatility",
        "insufficient_current_observations:high-volatility",
        "insufficient_reference_observations:normal-volatility",
        "insufficient_current_observations:normal-volatility",
    ]
    assert report["regimes"][1]["current_coverage"] is None
    assert report["regimes"][1]["simultaneous_drop_lower_bound"] is None


def test_rejects_overlapping_or_reversed_windows() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=10)
    current = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=10)

    with pytest.raises(RegimeCoverageError, match="strictly after"):
        audit_regime_coverage_drift(
            reference,
            current,
            _policy(min_observations_per_window=10),
        )


def test_rejects_reference_labels_that_mature_during_current_window() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=10)
    current = _window(datetime(2026, 1, 2, tzinfo=UTC), per_regime=10)
    last = reference[-1]
    reference[-1] = RegimeIntervalObservation(
        issued_at=last.issued_at,
        observed_at=current[0].issued_at + timedelta(hours=1),
        regime=last.regime,
        regime_as_of=last.regime_as_of,
        lower=last.lower,
        upper=last.upper,
        actual=last.actual,
    )

    with pytest.raises(RegimeCoverageError, match="fully mature"):
        audit_regime_coverage_drift(
            reference,
            current,
            _policy(min_observations_per_window=10),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"regimes": ("normal", "high")}, "sorted"),
        ({"max_coverage_drop": 1.0}, "one exclusive"),
        ({"max_coverage_drop": True}, "finite"),
        ({"confidence_level": 0.0}, "strictly between"),
        ({"min_observations_per_window": True}, "integer"),
    ],
)
def test_policy_rejects_unsafe_configuration(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(RegimeCoverageError, match=message):
        _policy(**overrides)


def test_report_is_deterministic_self_digested_and_privacy_reduced() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(datetime(2026, 2, 1, tzinfo=UTC))

    first = audit_regime_coverage_drift(reference, current, _policy())
    second = audit_regime_coverage_drift(reference, current, _policy())

    assert first == second
    assert len(first["policy_sha256"]) == 64
    assert len(first["report_sha256"]) == 64
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    serialized = json.dumps(first)
    assert "2026-" not in serialized
    assert '"actual"' not in serialized


def test_require_raises_stable_drift_findings() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(
        datetime(2026, 2, 1, tzinfo=UTC),
        misses={"high-volatility": set(range(80))},
    )

    with pytest.raises(
        RegimeCoverageError,
        match="coverage drift policy failed: coverage_drop_above_policy:high-volatility",
    ):
        require_regime_coverage_drift(reference, current, _policy())
