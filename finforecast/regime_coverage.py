"""Fail-closed coverage validation across predeclared market regimes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import NormalDist

REGIME_COVERAGE_SCHEMA = "finforecast-regime-coverage/v1"
REGIME_COVERAGE_DRIFT_SCHEMA = "finforecast-regime-coverage-drift/v1"
_REGIME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_MAX_REGIMES = 16
_MAX_OBSERVATIONS = 1_000_000


class RegimeCoverageError(ValueError):
    """The regime policy or supplied interval evidence is malformed."""

    code = "REGIME_COVERAGE_INVALID"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RegimeCoveragePolicy:
    """A prespecified family-wise coverage policy."""

    regimes: tuple[str, ...]
    target_coverage: float = 0.9
    tolerated_shortfall: float = 0.05
    confidence_level: float = 0.95
    min_observations_per_regime: int = 50

    def __post_init__(self) -> None:
        _validate_regimes(self.regimes)
        _unit_interval(self.target_coverage, "target_coverage")
        _unit_interval(self.confidence_level, "confidence_level")
        if not isinstance(self.tolerated_shortfall, int | float) or isinstance(
            self.tolerated_shortfall, bool
        ):
            raise RegimeCoverageError("tolerated_shortfall must be finite")
        if not math.isfinite(float(self.tolerated_shortfall)) or not (
            0 <= self.tolerated_shortfall < self.target_coverage
        ):
            raise RegimeCoverageError(
                "tolerated_shortfall must be non-negative and below target_coverage"
            )
        if (
            not isinstance(self.min_observations_per_regime, int)
            or isinstance(self.min_observations_per_regime, bool)
            or not 2 <= self.min_observations_per_regime <= _MAX_OBSERVATIONS
        ):
            raise RegimeCoverageError(
                "min_observations_per_regime must be an integer from 2 through 1000000"
            )


@dataclass(frozen=True, slots=True)
class RegimeCoverageDriftPolicy:
    """A prespecified simultaneous coverage-deterioration policy."""

    regimes: tuple[str, ...]
    max_coverage_drop: float = 0.05
    confidence_level: float = 0.95
    min_observations_per_window: int = 50

    def __post_init__(self) -> None:
        _validate_regimes(self.regimes)
        _unit_interval(self.confidence_level, "confidence_level")
        if not isinstance(self.max_coverage_drop, int | float) or isinstance(
            self.max_coverage_drop, bool
        ):
            raise RegimeCoverageError("max_coverage_drop must be finite")
        if not math.isfinite(float(self.max_coverage_drop)) or not (
            0 <= self.max_coverage_drop < 1
        ):
            raise RegimeCoverageError(
                "max_coverage_drop must be finite and between zero inclusive and one exclusive"
            )
        if (
            not isinstance(self.min_observations_per_window, int)
            or isinstance(self.min_observations_per_window, bool)
            or not 2 <= self.min_observations_per_window <= _MAX_OBSERVATIONS
        ):
            raise RegimeCoverageError(
                "min_observations_per_window must be an integer from 2 through 1000000"
            )


@dataclass(frozen=True, slots=True)
class RegimeIntervalObservation:
    """One matured interval and its point-in-time regime assignment."""

    issued_at: datetime
    observed_at: datetime
    regime: str
    regime_as_of: datetime
    lower: float
    upper: float
    actual: float


def _validate_regimes(regimes: object) -> None:
    if not isinstance(regimes, tuple) or not 1 <= len(regimes) <= _MAX_REGIMES:
        raise RegimeCoverageError("policy must declare between 1 and 16 regimes")
    if len(set(regimes)) != len(regimes):
        raise RegimeCoverageError("policy regimes must be unique")
    if tuple(sorted(regimes)) != regimes:
        raise RegimeCoverageError("policy regimes must be sorted")
    if any(not isinstance(value, str) or _REGIME_RE.fullmatch(value) is None for value in regimes):
        raise RegimeCoverageError("regime names must be canonical identifiers")


def _unit_interval(value: object, field: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RegimeCoverageError(f"{field} must be finite and strictly between zero and one")
    if not math.isfinite(float(value)) or not 0 < float(value) < 1:
        raise RegimeCoverageError(f"{field} must be finite and strictly between zero and one")


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RegimeCoverageError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _finite(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RegimeCoverageError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RegimeCoverageError(f"{field} must be finite")
    return result


def _wilson_lower_bound(successes: int, observations: int, z_score: float) -> float:
    if observations == 0:
        return 0.0
    proportion = successes / observations
    z_squared = z_score**2
    numerator = proportion + z_squared / (2 * observations)
    numerator -= z_score * math.sqrt(
        proportion * (1 - proportion) / observations + z_squared / (4 * observations**2)
    )
    return max(0.0, numerator / (1 + z_squared / observations))


def _wilson_upper_bound(successes: int, observations: int, z_score: float) -> float:
    if observations == 0:
        return 1.0
    proportion = successes / observations
    z_squared = z_score**2
    numerator = proportion + z_squared / (2 * observations)
    numerator += z_score * math.sqrt(
        proportion * (1 - proportion) / observations + z_squared / (4 * observations**2)
    )
    return min(1.0, numerator / (1 + z_squared / observations))


def _policy_document(policy: RegimeCoveragePolicy) -> dict[str, object]:
    return {
        "confidence_level": policy.confidence_level,
        "min_observations_per_regime": policy.min_observations_per_regime,
        "regimes": list(policy.regimes),
        "target_coverage": policy.target_coverage,
        "tolerated_shortfall": policy.tolerated_shortfall,
    }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _window_counts(
    observations: Sequence[RegimeIntervalObservation],
    regimes: tuple[str, ...],
    field: str,
) -> tuple[dict[str, dict[str, float | int]], datetime, datetime, datetime]:
    if not isinstance(observations, Sequence) or isinstance(observations, str | bytes):
        raise RegimeCoverageError(f"{field} must be a bounded sequence")
    if not 1 <= len(observations) <= _MAX_OBSERVATIONS:
        raise RegimeCoverageError(f"{field} must contain from 1 through 1000000 rows")

    row_label = "observation" if field == "observations" else f"{field} observation"
    counts: dict[str, dict[str, float | int]] = {
        regime: {
            "observations": 0,
            "covered": 0,
            "lower_misses": 0,
            "upper_misses": 0,
            "width_sum": 0.0,
        }
        for regime in regimes
    }
    first_issued_at: datetime | None = None
    previous_issued_at: datetime | None = None
    latest_observed_at: datetime | None = None
    for position, row in enumerate(observations):
        if not isinstance(row, RegimeIntervalObservation):
            raise RegimeCoverageError(f"{row_label} {position} has an invalid type")
        issued_at = _utc(row.issued_at, f"{row_label} {position} issued_at")
        observed_at = _utc(row.observed_at, f"{row_label} {position} observed_at")
        regime_as_of = _utc(row.regime_as_of, f"{row_label} {position} regime_as_of")
        if previous_issued_at is not None and issued_at <= previous_issued_at:
            raise RegimeCoverageError(f"{field} issued_at values must be strictly increasing")
        if observed_at <= issued_at:
            raise RegimeCoverageError(f"{row_label} {position} must mature after issuance")
        if regime_as_of > issued_at:
            raise RegimeCoverageError(
                f"{row_label} {position} regime label was not available at issuance"
            )
        if row.regime not in counts:
            raise RegimeCoverageError(
                f"{row_label} {position} uses undeclared regime {row.regime!r}"
            )
        lower = _finite(row.lower, f"{row_label} {position} lower")
        upper = _finite(row.upper, f"{row_label} {position} upper")
        actual = _finite(row.actual, f"{row_label} {position} actual")
        if lower > upper:
            raise RegimeCoverageError(f"{row_label} {position} has inverted bounds")

        bucket = counts[row.regime]
        bucket["observations"] += 1
        bucket["width_sum"] += upper - lower
        if actual < lower:
            bucket["lower_misses"] += 1
        elif actual > upper:
            bucket["upper_misses"] += 1
        else:
            bucket["covered"] += 1
        if first_issued_at is None:
            first_issued_at = issued_at
        previous_issued_at = issued_at
        if latest_observed_at is None or observed_at > latest_observed_at:
            latest_observed_at = observed_at

    assert first_issued_at is not None
    assert previous_issued_at is not None
    assert latest_observed_at is not None
    return counts, first_issued_at, previous_issued_at, latest_observed_at


def audit_regime_coverage(
    observations: Sequence[RegimeIntervalObservation],
    policy: RegimeCoveragePolicy,
) -> dict[str, object]:
    """Audit simultaneous interval coverage across point-in-time regime labels.

    Regime membership must have been available no later than interval issuance. The
    report contains aggregate counts only; raw forecasts, outcomes, and timestamps are
    deliberately omitted.
    """

    if not isinstance(policy, RegimeCoveragePolicy):
        raise TypeError("policy must be a RegimeCoveragePolicy")
    counts, _, _, _ = _window_counts(observations, policy.regimes, "observations")

    family_alpha = 1 - policy.confidence_level
    per_regime_alpha = family_alpha / len(policy.regimes)
    z_score = NormalDist().inv_cdf(1 - per_regime_alpha)
    required_lower_bound = policy.target_coverage - policy.tolerated_shortfall
    reasons: list[str] = []
    regime_reports: list[dict[str, object]] = []
    total_covered = 0
    total_width = 0.0
    for regime in policy.regimes:
        bucket = counts[regime]
        size = int(bucket["observations"])
        covered = int(bucket["covered"])
        total_covered += covered
        total_width += float(bucket["width_sum"])
        lower_bound = _wilson_lower_bound(covered, size, z_score)
        if size < policy.min_observations_per_regime:
            reasons.append(f"insufficient_observations:{regime}")
        elif lower_bound < required_lower_bound:
            reasons.append(f"coverage_lower_bound_below_policy:{regime}")
        regime_reports.append(
            {
                "regime": regime,
                "observations": size,
                "covered": covered,
                "breaches": size - covered,
                "lower_misses": int(bucket["lower_misses"]),
                "upper_misses": int(bucket["upper_misses"]),
                "observed_coverage": covered / size if size else None,
                "simultaneous_lower_bound": lower_bound if size else None,
                "mean_interval_width": float(bucket["width_sum"]) / size if size else None,
            }
        )

    policy_document = _policy_document(policy)
    return {
        "schema_version": REGIME_COVERAGE_SCHEMA,
        "accepted": not reasons,
        "finding_codes": reasons,
        "policy": policy_document,
        "policy_sha256": _digest(policy_document),
        "family_wise_confidence_level": policy.confidence_level,
        "per_regime_alpha": per_regime_alpha,
        "required_lower_bound": required_lower_bound,
        "total_observations": len(observations),
        "aggregate_observed_coverage": total_covered / len(observations),
        "aggregate_mean_interval_width": total_width / len(observations),
        "regimes": regime_reports,
    }


def require_regime_coverage(
    observations: Sequence[RegimeIntervalObservation],
    policy: RegimeCoveragePolicy,
) -> dict[str, object]:
    """Return an accepted report or raise with its stable policy findings."""

    report = audit_regime_coverage(observations, policy)
    if not report["accepted"]:
        findings = ", ".join(report["finding_codes"])
        raise RegimeCoverageError(f"coverage policy failed: {findings}")
    return report


def _drift_policy_document(policy: RegimeCoverageDriftPolicy) -> dict[str, object]:
    return {
        "confidence_level": policy.confidence_level,
        "max_coverage_drop": policy.max_coverage_drop,
        "min_observations_per_window": policy.min_observations_per_window,
        "regimes": list(policy.regimes),
    }


def audit_regime_coverage_drift(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageDriftPolicy,
) -> dict[str, object]:
    """Audit deterioration between disjoint, ordered regime-coverage windows.

    The one-sided lower confidence bound for each coverage drop is formed from a
    Wilson lower bound on the reference coverage and a Wilson upper bound on the
    current coverage. Bonferroni allocation covers both bounds for every predeclared
    regime. The reference must be fully mature before the current window begins. A drift
    finding is emitted only when the simultaneous lower bound exceeds the tolerated drop;
    undersized windows fail closed with separate findings.
    """

    if not isinstance(policy, RegimeCoverageDriftPolicy):
        raise TypeError("policy must be a RegimeCoverageDriftPolicy")
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

    family_alpha = 1 - policy.confidence_level
    component_alpha = family_alpha / (2 * len(policy.regimes))
    z_score = NormalDist().inv_cdf(1 - component_alpha)
    findings: list[str] = []
    regime_reports: list[dict[str, object]] = []
    total_reference_covered = 0
    total_current_covered = 0

    for regime in policy.regimes:
        reference_bucket = reference_counts[regime]
        current_bucket = current_counts[regime]
        reference_size = int(reference_bucket["observations"])
        current_size = int(current_bucket["observations"])
        reference_covered = int(reference_bucket["covered"])
        current_covered = int(current_bucket["covered"])
        total_reference_covered += reference_covered
        total_current_covered += current_covered

        reference_coverage = reference_covered / reference_size if reference_size else None
        current_coverage = current_covered / current_size if current_size else None
        observed_drop = (
            reference_coverage - current_coverage
            if reference_coverage is not None and current_coverage is not None
            else None
        )
        drop_lower_bound = (
            _wilson_lower_bound(reference_covered, reference_size, z_score)
            - _wilson_upper_bound(current_covered, current_size, z_score)
            if reference_size and current_size
            else None
        )

        reference_sufficient = reference_size >= policy.min_observations_per_window
        current_sufficient = current_size >= policy.min_observations_per_window
        if not reference_sufficient:
            findings.append(f"insufficient_reference_observations:{regime}")
        if not current_sufficient:
            findings.append(f"insufficient_current_observations:{regime}")
        if (
            reference_sufficient
            and current_sufficient
            and drop_lower_bound is not None
            and drop_lower_bound > policy.max_coverage_drop
        ):
            findings.append(f"coverage_drop_above_policy:{regime}")

        regime_reports.append(
            {
                "regime": regime,
                "reference_observations": reference_size,
                "reference_covered": reference_covered,
                "reference_coverage": reference_coverage,
                "current_observations": current_size,
                "current_covered": current_covered,
                "current_coverage": current_coverage,
                "observed_coverage_drop": observed_drop,
                "simultaneous_drop_lower_bound": drop_lower_bound,
            }
        )

    policy_document = _drift_policy_document(policy)
    report: dict[str, object] = {
        "schema_version": REGIME_COVERAGE_DRIFT_SCHEMA,
        "accepted": not findings,
        "finding_codes": findings,
        "policy": policy_document,
        "policy_sha256": _digest(policy_document),
        "family_wise_confidence_level": policy.confidence_level,
        "component_alpha": component_alpha,
        "reference_observations": len(reference),
        "reference_coverage": total_reference_covered / len(reference),
        "current_observations": len(current),
        "current_coverage": total_current_covered / len(current),
        "regimes": regime_reports,
    }
    report["report_sha256"] = _digest(report)
    return report


def require_regime_coverage_drift(
    reference: Sequence[RegimeIntervalObservation],
    current: Sequence[RegimeIntervalObservation],
    policy: RegimeCoverageDriftPolicy,
) -> dict[str, object]:
    """Return an accepted drift report or raise with stable policy findings."""

    report = audit_regime_coverage_drift(reference, current, policy)
    if not report["accepted"]:
        findings = ", ".join(report["finding_codes"])
        raise RegimeCoverageError(f"coverage drift policy failed: {findings}")
    return report
