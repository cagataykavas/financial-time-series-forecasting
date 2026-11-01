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
        if not 1 <= len(self.regimes) <= _MAX_REGIMES:
            raise RegimeCoverageError("policy must declare between 1 and 16 regimes")
        if len(set(self.regimes)) != len(self.regimes):
            raise RegimeCoverageError("policy regimes must be unique")
        if tuple(sorted(self.regimes)) != self.regimes:
            raise RegimeCoverageError("policy regimes must be sorted")
        if any(
            not isinstance(value, str) or _REGIME_RE.fullmatch(value) is None
            for value in self.regimes
        ):
            raise RegimeCoverageError("regime names must be canonical identifiers")
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
class RegimeIntervalObservation:
    """One matured interval and its point-in-time regime assignment."""

    issued_at: datetime
    observed_at: datetime
    regime: str
    regime_as_of: datetime
    lower: float
    upper: float
    actual: float


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
    if not isinstance(observations, Sequence) or isinstance(observations, str | bytes):
        raise RegimeCoverageError("observations must be a bounded sequence")
    if not 1 <= len(observations) <= _MAX_OBSERVATIONS:
        raise RegimeCoverageError("observations must contain from 1 through 1000000 rows")

    counts = {
        regime: {
            "observations": 0,
            "covered": 0,
            "lower_misses": 0,
            "upper_misses": 0,
            "width_sum": 0.0,
        }
        for regime in policy.regimes
    }
    previous_issued_at: datetime | None = None
    for position, row in enumerate(observations):
        if not isinstance(row, RegimeIntervalObservation):
            raise RegimeCoverageError(f"observation {position} has an invalid type")
        issued_at = _utc(row.issued_at, f"observation {position} issued_at")
        observed_at = _utc(row.observed_at, f"observation {position} observed_at")
        regime_as_of = _utc(row.regime_as_of, f"observation {position} regime_as_of")
        if previous_issued_at is not None and issued_at <= previous_issued_at:
            raise RegimeCoverageError("issued_at values must be strictly increasing")
        if observed_at <= issued_at:
            raise RegimeCoverageError(f"observation {position} must mature after issuance")
        if regime_as_of > issued_at:
            raise RegimeCoverageError(
                f"observation {position} regime label was not available at issuance"
            )
        if row.regime not in counts:
            raise RegimeCoverageError(
                f"observation {position} uses undeclared regime {row.regime!r}"
            )
        lower = _finite(row.lower, f"observation {position} lower")
        upper = _finite(row.upper, f"observation {position} upper")
        actual = _finite(row.actual, f"observation {position} actual")
        if lower > upper:
            raise RegimeCoverageError(f"observation {position} has inverted bounds")

        bucket = counts[row.regime]
        bucket["observations"] += 1
        bucket["width_sum"] += upper - lower
        if actual < lower:
            bucket["lower_misses"] += 1
        elif actual > upper:
            bucket["upper_misses"] += 1
        else:
            bucket["covered"] += 1
        previous_issued_at = issued_at

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
