"""Serial-dependence-aware regime coverage drift validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC

import numpy as np

from .regime_coverage import (
    RegimeCoverageDriftPolicy,
    RegimeCoverageError,
    RegimeIntervalObservation,
    _window_counts,
)

REGIME_COVERAGE_BLOCK_DRIFT_SCHEMA = "finforecast-regime-coverage-block-drift/v1"
BLOCK_BOOTSTRAP_METHOD = "circular-moving-block-percentile/v1"
_MAX_BOOTSTRAP_DRAWS = 25_000_000


@dataclass(frozen=True, slots=True)
class RegimeCoverageBlockBootstrapPolicy:
    """Prespecified block-bootstrap policy for regime coverage deterioration."""

    regimes: tuple[str, ...]
    max_coverage_drop: float = 0.05
    confidence_level: float = 0.95
    min_observations_per_window: int = 100
    block_size: int = 5
    resamples: int = 20_000
    random_seed: int = 42

    def __post_init__(self) -> None:
        RegimeCoverageDriftPolicy(
            regimes=self.regimes,
            max_coverage_drop=self.max_coverage_drop,
            confidence_level=self.confidence_level,
            min_observations_per_window=self.min_observations_per_window,
        )
        if (
            not isinstance(self.block_size, int)
            or isinstance(self.block_size, bool)
            or not 2 <= self.block_size <= 10_000
        ):
            raise RegimeCoverageError("block_size must be an integer from 2 through 10000")
        if (
            not isinstance(self.resamples, int)
            or isinstance(self.resamples, bool)
            or not 1_000 <= self.resamples <= 100_000
        ):
            raise RegimeCoverageError("resamples must be an integer from 1000 through 100000")
        if (
            not isinstance(self.random_seed, int)
            or isinstance(self.random_seed, bool)
            or not 0 <= self.random_seed <= 2**32 - 1
        ):
            raise RegimeCoverageError("random_seed must be an integer from 0 through 4294967295")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["regimes"] = list(self.regimes)
        return result


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _population_sha256(observations: Sequence[RegimeIntervalObservation]) -> str:
    rows = []
    for row in observations:
        rows.append(
            {
                "issued_at": row.issued_at.astimezone(UTC).isoformat(timespec="microseconds"),
                "observed_at": row.observed_at.astimezone(UTC).isoformat(timespec="microseconds"),
                "regime": row.regime,
                "regime_as_of": row.regime_as_of.astimezone(UTC).isoformat(timespec="microseconds"),
                "lower": float(row.lower).hex(),
                "upper": float(row.upper).hex(),
                "actual": float(row.actual).hex(),
            }
        )
    return _digest({"schema_version": 1, "observations": rows})


def _coverage_by_regime(
    observations: Sequence[RegimeIntervalObservation], regimes: tuple[str, ...]
) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = {regime: [] for regime in regimes}
    for row in observations:
        values[row.regime].append(float(row.lower <= row.actual <= row.upper))
    return {regime: np.asarray(rows, dtype=np.float64) for regime, rows in values.items()}


def _regime_seed(policy: RegimeCoverageBlockBootstrapPolicy, regime: str) -> tuple[int, str]:
    seed_sha256 = _digest(
        {
            "method": BLOCK_BOOTSTRAP_METHOD,
            "random_seed": policy.random_seed,
            "regime": regime,
        }
    )
    return int(seed_sha256[:16], 16), seed_sha256


def _block_bootstrap_means(
    values: np.ndarray,
    *,
    block_size: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    size = int(values.size)
    block_count = math.ceil(size / block_size)
    sample_width = block_count * block_size
    offsets = np.arange(block_size)
    means = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(resamples, 1_000_000 // sample_width))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        block_starts = rng.integers(0, size, size=(stop - start, block_count))
        indices = (block_starts[:, :, None] + offsets) % size
        samples = values[indices.reshape(stop - start, -1)[:, :size]]
        means[start:stop] = samples.mean(axis=1)
    return means


def audit_blocked_regime_coverage_drift(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageBlockBootstrapPolicy,
) -> dict[str, object]:
    """Audit regime coverage drift while preserving short-range breach dependence."""

    if not isinstance(policy, RegimeCoverageBlockBootstrapPolicy):
        raise TypeError("policy must be a RegimeCoverageBlockBootstrapPolicy")
    reference_counts, _, reference_last, reference_latest_observed = _window_counts(
        reference, policy.regimes, "reference"
    )
    current_counts, current_first, _, _ = _window_counts(current, policy.regimes, "current")
    if current_first <= reference_last:
        raise RegimeCoverageError(
            "current window must begin strictly after the reference window ends"
        )
    if reference_latest_observed >= current_first:
        raise RegimeCoverageError(
            "reference window must be fully mature before the current window begins"
        )

    resampled_observations_per_iteration = 0
    for regime in policy.regimes:
        for counts in (reference_counts, current_counts):
            size = int(counts[regime]["observations"])
            if size >= policy.min_observations_per_window and size >= 2 * policy.block_size:
                resampled_observations_per_iteration += (
                    math.ceil(size / policy.block_size) * policy.block_size
                )
    total_draws = policy.resamples * resampled_observations_per_iteration
    if total_draws > _MAX_BOOTSTRAP_DRAWS:
        raise RegimeCoverageError("block bootstrap work budget exceeded")
    reference_series = _coverage_by_regime(reference, policy.regimes)
    current_series = _coverage_by_regime(current, policy.regimes)
    per_regime_alpha = (1.0 - policy.confidence_level) / len(policy.regimes)
    findings: list[str] = []
    reports: list[dict[str, object]] = []

    for regime in policy.regimes:
        reference_values = reference_series[regime]
        current_values = current_series[regime]
        reference_size = int(reference_counts[regime]["observations"])
        current_size = int(current_counts[regime]["observations"])
        reference_sufficient = reference_size >= policy.min_observations_per_window
        current_sufficient = current_size >= policy.min_observations_per_window
        reference_blocks_sufficient = reference_size >= 2 * policy.block_size
        current_blocks_sufficient = current_size >= 2 * policy.block_size
        if not reference_sufficient:
            findings.append(f"insufficient_reference_observations:{regime}")
        if not current_sufficient:
            findings.append(f"insufficient_current_observations:{regime}")
        if reference_sufficient and not reference_blocks_sufficient:
            findings.append(f"insufficient_reference_blocks:{regime}")
        if current_sufficient and not current_blocks_sufficient:
            findings.append(f"insufficient_current_blocks:{regime}")

        lower_bound: float | None = None
        seed_sha256: str | None = None
        if (
            reference_sufficient
            and current_sufficient
            and reference_blocks_sufficient
            and current_blocks_sufficient
        ):
            seed, seed_sha256 = _regime_seed(policy, regime)
            rng = np.random.Generator(np.random.PCG64(seed))
            reference_means = _block_bootstrap_means(
                reference_values,
                block_size=policy.block_size,
                resamples=policy.resamples,
                rng=rng,
            )
            current_means = _block_bootstrap_means(
                current_values,
                block_size=policy.block_size,
                resamples=policy.resamples,
                rng=rng,
            )
            bootstrap_drops = reference_means - current_means
            lower_bound = float(np.quantile(bootstrap_drops, per_regime_alpha))
            if lower_bound > policy.max_coverage_drop:
                findings.append(f"coverage_drop_above_policy:{regime}")

        reference_coverage = float(reference_values.mean()) if reference_size else None
        current_coverage = float(current_values.mean()) if current_size else None
        reports.append(
            {
                "regime": regime,
                "reference_observations": reference_size,
                "reference_coverage": reference_coverage,
                "current_observations": current_size,
                "current_coverage": current_coverage,
                "observed_coverage_drop": (
                    reference_coverage - current_coverage
                    if reference_coverage is not None and current_coverage is not None
                    else None
                ),
                "simultaneous_drop_lower_bound": lower_bound,
                "resampling_seed_sha256": seed_sha256,
            }
        )

    policy_document = policy.as_dict()
    report: dict[str, object] = {
        "schema_version": REGIME_COVERAGE_BLOCK_DRIFT_SCHEMA,
        "accepted": not findings,
        "finding_codes": findings,
        "method": BLOCK_BOOTSTRAP_METHOD,
        "policy": policy_document,
        "policy_sha256": _digest(policy_document),
        "reference_population_sha256": _population_sha256(reference),
        "current_population_sha256": _population_sha256(current),
        "family_wise_confidence_level": policy.confidence_level,
        "per_regime_alpha": per_regime_alpha,
        "bootstrap_resamples": policy.resamples,
        "block_size": policy.block_size,
        "total_resampled_observations": total_draws,
        "regimes": reports,
    }
    report["report_sha256"] = _digest(report)
    return report


def require_blocked_regime_coverage_drift(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageBlockBootstrapPolicy,
) -> dict[str, object]:
    """Return an accepted block-drift report or raise stable policy findings."""

    report = audit_blocked_regime_coverage_drift(reference, current, policy)
    if not report["accepted"]:
        findings = ", ".join(report["finding_codes"])
        raise RegimeCoverageError(f"blocked coverage drift policy failed: {findings}")
    return report
