from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.regime_coverage import RegimeCoverageError, RegimeIntervalObservation
from finforecast.regime_coverage_sensitivity import (
    BLOCK_SIZE_SENSITIVITY_METHOD,
    REGIME_COVERAGE_BLOCK_SENSITIVITY_SCHEMA,
    RegimeCoverageBlockSensitivityPolicy,
    audit_regime_coverage_block_sensitivity,
    require_regime_coverage_block_sensitivity,
)


def _policy(**overrides: object) -> RegimeCoverageBlockSensitivityPolicy:
    values = {
        "regimes": ("high-volatility", "normal-volatility"),
        "block_sizes": (5, 10, 20),
        "target_coverage": 0.8,
        "tolerated_shortfall": 0.15,
        "max_coverage_drop": 0.1,
        "confidence_level": 0.95,
        "min_observations_per_window": 50,
        "resamples_per_block_size": 2_000,
        "random_seed": 17,
    }
    values.update(overrides)
    return RegimeCoverageBlockSensitivityPolicy(**values)


def _window(
    start: datetime,
    *,
    per_regime: int = 100,
    misses: dict[str, set[int]] | None = None,
) -> list[RegimeIntervalObservation]:
    misses = misses or {}
    regimes = ("high-volatility", "normal-volatility")
    rows: list[RegimeIntervalObservation] = []
    for position in range(per_regime * len(regimes)):
        regime = regimes[position % len(regimes)]
        regime_position = position // len(regimes)
        issued_at = start + timedelta(hours=position)
        rows.append(
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
    return rows


def test_accepts_only_when_absolute_and_all_block_size_gates_accept() -> None:
    report = require_regime_coverage_block_sensitivity(
        _window(datetime(2026, 1, 1, tzinfo=UTC)),
        _window(
            datetime(2026, 2, 1, tzinfo=UTC),
            misses={"high-volatility": set(range(10, 20))},
        ),
        _policy(),
    )

    assert report["schema_version"] == REGIME_COVERAGE_BLOCK_SENSITIVITY_SCHEMA
    assert report["method"] == BLOCK_SIZE_SENSITIVITY_METHOD
    assert report["accepted"] is True
    assert report["absolute_current_coverage"]["accepted"] is True
    assert [child["block_size"] for child in report["block_size_reports"]] == [5, 10, 20]
    assert all(child["accepted"] for child in report["block_size_reports"])
    assert report["regimes"][0]["most_conservative_drop_lower_bound"] == max(
        row["simultaneous_drop_lower_bound"] for row in report["regimes"][0]["drop_lower_bounds"]
    )


def test_absolute_current_coverage_failure_cannot_be_masked_by_stable_drift() -> None:
    misses = {"high-volatility": set(range(35)), "normal-volatility": set(range(35))}
    report = audit_regime_coverage_block_sensitivity(
        _window(datetime(2026, 1, 1, tzinfo=UTC), misses=misses),
        _window(datetime(2026, 2, 1, tzinfo=UTC), misses=misses),
        _policy(),
    )

    assert report["accepted"] is False
    assert all(child["accepted"] for child in report["block_size_reports"])
    assert report["finding_codes"] == [
        "absolute_current_coverage:coverage_lower_bound_below_policy:high-volatility",
        "absolute_current_coverage:coverage_lower_bound_below_policy:normal-volatility",
    ]


def test_rejects_and_flags_instability_when_block_size_decisions_differ() -> None:
    report = audit_regime_coverage_block_sensitivity(
        _window(datetime(2026, 1, 1, tzinfo=UTC)),
        _window(
            datetime(2026, 2, 1, tzinfo=UTC),
            misses={"high-volatility": set(range(20, 60))},
        ),
        _policy(target_coverage=0.6, tolerated_shortfall=0.1),
    )

    assert report["accepted"] is False
    assert [child["accepted"] for child in report["block_size_reports"]] == [
        False,
        False,
        True,
    ]
    assert report["finding_codes"] == [
        "block_size_5:coverage_drop_above_policy:high-volatility",
        "block_size_10:coverage_drop_above_policy:high-volatility",
        "block_size_decision_instability",
    ]


def test_report_is_deterministic_content_bound_and_privacy_reduced() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(datetime(2026, 2, 1, tzinfo=UTC))
    first = audit_regime_coverage_block_sensitivity(reference, current, _policy())
    second = audit_regime_coverage_block_sensitivity(reference, current, _policy())

    assert first == second
    assert len(first["reference_population_sha256"]) == 64
    assert len(first["current_population_sha256"]) == 64
    assert len(first["report_sha256"]) == 64
    rendered = json.dumps(first, allow_nan=False)
    assert "2026-" not in rendered
    assert '"actual"' not in rendered

    changed = list(current)
    row = changed[0]
    changed[0] = RegimeIntervalObservation(
        issued_at=row.issued_at,
        observed_at=row.observed_at,
        regime=row.regime,
        regime_as_of=row.regime_as_of,
        lower=row.lower,
        upper=row.upper,
        actual=0.5,
    )
    changed_report = audit_regime_coverage_block_sensitivity(reference, changed, _policy())
    assert changed_report["current_population_sha256"] != first["current_population_sha256"]


def test_policy_and_total_work_budget_are_bounded() -> None:
    with pytest.raises(RegimeCoverageError, match="2 through 8"):
        _policy(block_sizes=(10,))
    with pytest.raises(RegimeCoverageError, match="unique and strictly increasing"):
        _policy(block_sizes=(10, 5))
    with pytest.raises(RegimeCoverageError, match="unique and strictly increasing"):
        _policy(block_sizes=(5, 5))

    with pytest.raises(RegimeCoverageError, match="sensitivity work budget"):
        audit_regime_coverage_block_sensitivity(
            _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=500),
            _window(datetime(2026, 3, 1, tzinfo=UTC), per_regime=500),
            _policy(block_sizes=(5, 10, 20, 40), resamples_per_block_size=50_000),
        )


def test_require_raises_stable_sensitivity_findings() -> None:
    with pytest.raises(
        RegimeCoverageError,
        match="block-size sensitivity policy failed: block_size_5:coverage_drop_above_policy",
    ):
        require_regime_coverage_block_sensitivity(
            _window(datetime(2026, 1, 1, tzinfo=UTC)),
            _window(
                datetime(2026, 2, 1, tzinfo=UTC),
                misses={"high-volatility": set(range(20, 60))},
            ),
            _policy(target_coverage=0.6, tolerated_shortfall=0.1),
        )
