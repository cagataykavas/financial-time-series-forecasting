from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.regime_coverage import RegimeCoverageError, RegimeIntervalObservation
from finforecast.regime_coverage_blocks import (
    BLOCK_BOOTSTRAP_METHOD,
    REGIME_COVERAGE_BLOCK_DRIFT_SCHEMA,
    RegimeCoverageBlockBootstrapPolicy,
    audit_blocked_regime_coverage_drift,
    require_blocked_regime_coverage_drift,
)


def _policy(**overrides: object) -> RegimeCoverageBlockBootstrapPolicy:
    values = {
        "regimes": ("high-volatility", "normal-volatility"),
        "max_coverage_drop": 0.1,
        "confidence_level": 0.95,
        "min_observations_per_window": 50,
        "block_size": 10,
        "resamples": 2_000,
        "random_seed": 17,
    }
    values.update(overrides)
    return RegimeCoverageBlockBootstrapPolicy(**values)


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


def test_accepts_stable_coverage_with_block_dependence_policy() -> None:
    report = require_blocked_regime_coverage_drift(
        _window(datetime(2026, 1, 1, tzinfo=UTC)),
        _window(
            datetime(2026, 2, 1, tzinfo=UTC),
            misses={"high-volatility": set(range(10, 20))},
        ),
        _policy(),
    )

    assert report["schema_version"] == REGIME_COVERAGE_BLOCK_DRIFT_SCHEMA
    assert report["method"] == BLOCK_BOOTSTRAP_METHOD
    assert report["accepted"] is True
    assert report["finding_codes"] == []
    assert report["regimes"][0]["observed_coverage_drop"] == pytest.approx(0.1)
    assert report["regimes"][0]["simultaneous_drop_lower_bound"] < 0.1


def test_detects_clustered_regime_coverage_deterioration() -> None:
    report = audit_blocked_regime_coverage_drift(
        _window(datetime(2026, 1, 1, tzinfo=UTC)),
        _window(
            datetime(2026, 2, 1, tzinfo=UTC),
            misses={"high-volatility": set(range(20, 60))},
        ),
        _policy(),
    )

    assert report["accepted"] is False
    assert report["finding_codes"] == ["coverage_drop_above_policy:high-volatility"]
    high = report["regimes"][0]
    assert high["current_coverage"] == pytest.approx(0.6)
    assert high["simultaneous_drop_lower_bound"] > 0.1
    assert len(high["resampling_seed_sha256"]) == 64


def test_report_is_deterministic_content_bound_and_privacy_reduced() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC))
    current = _window(datetime(2026, 2, 1, tzinfo=UTC))
    first = audit_blocked_regime_coverage_drift(reference, current, _policy())
    second = audit_blocked_regime_coverage_drift(reference, current, _policy())

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
    changed_report = audit_blocked_regime_coverage_drift(reference, changed, _policy())
    assert changed_report["current_population_sha256"] != first["current_population_sha256"]


def test_insufficient_block_evidence_fails_closed_without_sampling() -> None:
    report = audit_blocked_regime_coverage_drift(
        _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=15),
        _window(datetime(2026, 2, 1, tzinfo=UTC), per_regime=15),
        _policy(min_observations_per_window=10, block_size=10),
    )

    assert report["finding_codes"] == [
        "insufficient_reference_blocks:high-volatility",
        "insufficient_current_blocks:high-volatility",
        "insufficient_reference_blocks:normal-volatility",
        "insufficient_current_blocks:normal-volatility",
    ]
    assert report["regimes"][0]["simultaneous_drop_lower_bound"] is None
    assert report["regimes"][0]["resampling_seed_sha256"] is None


def test_rejects_overlapping_or_immature_windows() -> None:
    reference = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=20)
    current = _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=20)
    with pytest.raises(RegimeCoverageError, match="strictly after"):
        audit_blocked_regime_coverage_drift(
            reference,
            current,
            _policy(min_observations_per_window=20, block_size=5),
        )

    current = _window(datetime(2026, 1, 3, tzinfo=UTC), per_regime=20)
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
        audit_blocked_regime_coverage_drift(
            reference,
            current,
            _policy(min_observations_per_window=20, block_size=5),
        )


def test_policy_and_work_budget_are_bounded_before_sampling() -> None:
    with pytest.raises(RegimeCoverageError, match="block_size"):
        _policy(block_size=1)
    with pytest.raises(RegimeCoverageError, match="resamples"):
        _policy(resamples=999)
    with pytest.raises(RegimeCoverageError, match="random_seed"):
        _policy(random_seed=True)

    with pytest.raises(RegimeCoverageError, match="work budget"):
        audit_blocked_regime_coverage_drift(
            _window(datetime(2026, 1, 1, tzinfo=UTC), per_regime=500),
            _window(datetime(2026, 3, 1, tzinfo=UTC), per_regime=500),
            _policy(resamples=20_000),
        )


def test_require_raises_stable_block_drift_findings() -> None:
    with pytest.raises(
        RegimeCoverageError,
        match="blocked coverage drift policy failed: coverage_drop_above_policy:high-volatility",
    ):
        require_blocked_regime_coverage_drift(
            _window(datetime(2026, 1, 1, tzinfo=UTC)),
            _window(
                datetime(2026, 2, 1, tzinfo=UTC),
                misses={"high-volatility": set(range(20, 60))},
            ),
            _policy(),
        )
