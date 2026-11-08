"""Reviewed block-size sensitivity for regime coverage promotion gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .regime_coverage import (
    RegimeCoverageError,
    RegimeCoveragePolicy,
    RegimeIntervalObservation,
    _window_counts,
    audit_regime_coverage,
)
from .regime_coverage_blocks import (
    RegimeCoverageBlockBootstrapPolicy,
    audit_blocked_regime_coverage_drift,
)

REGIME_COVERAGE_BLOCK_SENSITIVITY_SCHEMA = "finforecast-regime-coverage-block-sensitivity/v1"
BLOCK_SIZE_SENSITIVITY_METHOD = "reviewed-block-size-envelope/v1"
_MAX_BLOCK_SIZES = 8
_MAX_SENSITIVITY_DRAWS = 50_000_000


@dataclass(frozen=True, slots=True)
class RegimeCoverageBlockSensitivityPolicy:
    """Prespecified absolute-coverage and block-size robustness policy."""

    regimes: tuple[str, ...]
    block_sizes: tuple[int, ...]
    target_coverage: float = 0.9
    tolerated_shortfall: float = 0.05
    max_coverage_drop: float = 0.05
    confidence_level: float = 0.95
    min_observations_per_window: int = 100
    resamples_per_block_size: int = 20_000
    random_seed: int = 42

    def __post_init__(self) -> None:
        RegimeCoveragePolicy(
            regimes=self.regimes,
            target_coverage=self.target_coverage,
            tolerated_shortfall=self.tolerated_shortfall,
            confidence_level=self.confidence_level,
            min_observations_per_regime=self.min_observations_per_window,
        )
        RegimeCoverageBlockBootstrapPolicy(
            regimes=self.regimes,
            max_coverage_drop=self.max_coverage_drop,
            confidence_level=self.confidence_level,
            min_observations_per_window=self.min_observations_per_window,
            block_size=2,
            resamples=self.resamples_per_block_size,
            random_seed=self.random_seed,
        )
        if (
            not isinstance(self.block_sizes, tuple)
            or not 2 <= len(self.block_sizes) <= _MAX_BLOCK_SIZES
        ):
            raise RegimeCoverageError("block_sizes must declare from 2 through 8 values")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or not 2 <= value <= 10_000
            for value in self.block_sizes
        ):
            raise RegimeCoverageError("block_sizes values must be integers from 2 through 10000")
        if tuple(sorted(set(self.block_sizes))) != self.block_sizes:
            raise RegimeCoverageError("block_sizes must be unique and strictly increasing")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["regimes"] = list(self.regimes)
        result["block_sizes"] = list(self.block_sizes)
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


def _sensitivity_draws(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageBlockSensitivityPolicy,
) -> int:
    reference_counts, _, _, _ = _window_counts(reference, policy.regimes, "reference")
    current_counts, _, _, _ = _window_counts(current, policy.regimes, "current")
    observations_per_iteration = 0
    for block_size in policy.block_sizes:
        for regime in policy.regimes:
            for counts in (reference_counts, current_counts):
                size = int(counts[regime]["observations"])
                if size >= policy.min_observations_per_window and size >= 2 * block_size:
                    observations_per_iteration += math.ceil(size / block_size) * block_size
    return policy.resamples_per_block_size * observations_per_iteration


def audit_regime_coverage_block_sensitivity(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageBlockSensitivityPolicy,
) -> dict[str, object]:
    """Require absolute current coverage and drift acceptance at every block size."""

    if not isinstance(policy, RegimeCoverageBlockSensitivityPolicy):
        raise TypeError("policy must be a RegimeCoverageBlockSensitivityPolicy")
    total_draws = _sensitivity_draws(reference, current, policy)
    if total_draws > _MAX_SENSITIVITY_DRAWS:
        raise RegimeCoverageError("block-size sensitivity work budget exceeded")

    absolute_report = audit_regime_coverage(
        current,
        RegimeCoveragePolicy(
            regimes=policy.regimes,
            target_coverage=policy.target_coverage,
            tolerated_shortfall=policy.tolerated_shortfall,
            confidence_level=policy.confidence_level,
            min_observations_per_regime=policy.min_observations_per_window,
        ),
    )
    findings = [f"absolute_current_coverage:{code}" for code in absolute_report["finding_codes"]]
    block_reports: list[dict[str, object]] = []
    for block_size in policy.block_sizes:
        block_report = audit_blocked_regime_coverage_drift(
            reference,
            current,
            RegimeCoverageBlockBootstrapPolicy(
                regimes=policy.regimes,
                max_coverage_drop=policy.max_coverage_drop,
                confidence_level=policy.confidence_level,
                min_observations_per_window=policy.min_observations_per_window,
                block_size=block_size,
                resamples=policy.resamples_per_block_size,
                random_seed=policy.random_seed,
            ),
        )
        block_reports.append(block_report)
        findings.extend(f"block_size_{block_size}:{code}" for code in block_report["finding_codes"])

    block_decisions = {bool(report["accepted"]) for report in block_reports}
    if len(block_decisions) > 1:
        findings.append("block_size_decision_instability")

    regime_reports: list[dict[str, object]] = []
    for position, regime in enumerate(policy.regimes):
        lower_bounds = [
            {
                "block_size": block_size,
                "simultaneous_drop_lower_bound": block_report["regimes"][position][
                    "simultaneous_drop_lower_bound"
                ],
            }
            for block_size, block_report in zip(policy.block_sizes, block_reports, strict=True)
        ]
        available_bounds = [
            float(item["simultaneous_drop_lower_bound"])
            for item in lower_bounds
            if item["simultaneous_drop_lower_bound"] is not None
        ]
        first_regime = block_reports[0]["regimes"][position]
        regime_reports.append(
            {
                "regime": regime,
                "reference_observations": first_regime["reference_observations"],
                "reference_coverage": first_regime["reference_coverage"],
                "current_observations": first_regime["current_observations"],
                "current_coverage": first_regime["current_coverage"],
                "observed_coverage_drop": first_regime["observed_coverage_drop"],
                "drop_lower_bounds": lower_bounds,
                "most_conservative_drop_lower_bound": (
                    max(available_bounds) if available_bounds else None
                ),
            }
        )

    policy_document = policy.as_dict()
    report: dict[str, object] = {
        "schema_version": REGIME_COVERAGE_BLOCK_SENSITIVITY_SCHEMA,
        "accepted": not findings,
        "finding_codes": findings,
        "method": BLOCK_SIZE_SENSITIVITY_METHOD,
        "policy": policy_document,
        "policy_sha256": _digest(policy_document),
        "reference_population_sha256": block_reports[0]["reference_population_sha256"],
        "current_population_sha256": block_reports[0]["current_population_sha256"],
        "absolute_current_coverage": absolute_report,
        "block_size_reports": block_reports,
        "total_resampled_observations": total_draws,
        "regimes": regime_reports,
    }
    report["report_sha256"] = _digest(report)
    return report


def require_regime_coverage_block_sensitivity(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageBlockSensitivityPolicy,
) -> dict[str, object]:
    """Return an accepted sensitivity report or raise stable policy findings."""

    report = audit_regime_coverage_block_sensitivity(reference, current, policy)
    if not report["accepted"]:
        findings = ", ".join(report["finding_codes"])
        raise RegimeCoverageError(f"block-size sensitivity policy failed: {findings}")
    return report
