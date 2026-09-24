"""Fail-closed coverage and breach-independence audit for forecast intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BreachAuditInputError(ValueError):
    """Raised when interval evidence cannot be audited safely."""


@dataclass(frozen=True)
class BreachAuditPolicy:
    min_observations: int = 100
    min_expected_breaches: float = 5.0
    significance_level: float = 0.05
    max_consecutive_breaches: int = 3
    max_observations: int = 1_000_000
    max_evidence_timestamps: int = 100


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BreachAuditInputError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BreachAuditInputError(f"{field} must be a finite number")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BreachAuditInputError(f"{field} must be a positive integer")
    return value


def _validate_policy(policy: BreachAuditPolicy) -> None:
    for field in (
        "min_observations",
        "max_consecutive_breaches",
        "max_observations",
        "max_evidence_timestamps",
    ):
        _positive_int(getattr(policy, field), f"policy.{field}")
    if policy.min_observations > policy.max_observations:
        raise BreachAuditInputError("policy.min_observations cannot exceed policy.max_observations")
    expected = _finite_number(policy.min_expected_breaches, "policy.min_expected_breaches")
    if expected <= 0:
        raise BreachAuditInputError("policy.min_expected_breaches must be greater than zero")
    significance = _finite_number(policy.significance_level, "policy.significance_level")
    if not 0.0 < significance < 1.0:
        raise BreachAuditInputError("policy.significance_level must be between zero and one")


def _utc_timestamp(value: Any, field: str) -> tuple[datetime, str]:
    if not isinstance(value, str) or not value.strip():
        raise BreachAuditInputError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BreachAuditInputError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BreachAuditInputError(f"{field} must include a UTC offset")
    normalized = parsed.astimezone(UTC)
    return normalized, normalized.isoformat().replace("+00:00", "Z")


def _xlogy(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -math.inf
    return count * math.log(probability)


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    return _xlogy(successes, probability) + _xlogy(failures, 1.0 - probability)


def _chi_square_survival_df1(statistic: float) -> float:
    return math.erfc(math.sqrt(statistic / 2.0))


def _likelihood_ratio(null_log_likelihood: float, alternative_log_likelihood: float) -> float:
    statistic = -2.0 * (null_log_likelihood - alternative_log_likelihood)
    if not math.isfinite(statistic):
        raise BreachAuditInputError("likelihood-ratio statistic is not finite")
    return max(0.0, statistic)


def _coverage_test(breaches: list[bool], expected_rate: float) -> dict[str, float]:
    count = sum(breaches)
    total = len(breaches)
    observed_rate = count / total
    null = _bernoulli_log_likelihood(count, total - count, expected_rate)
    alternative = _bernoulli_log_likelihood(count, total - count, observed_rate)
    statistic = _likelihood_ratio(null, alternative)
    return {
        "expected_breach_rate": expected_rate,
        "observed_breach_rate": observed_rate,
        "likelihood_ratio": statistic,
        "p_value": _chi_square_survival_df1(statistic),
    }


def _independence_test(breaches: list[bool]) -> dict[str, Any]:
    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(breaches, breaches[1:], strict=False):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1

    transitions = n00 + n01 + n10 + n11
    breach_rate = (n01 + n11) / transitions
    rate_after_covered = n01 / (n00 + n01) if n00 + n01 else 0.0
    rate_after_breach = n11 / (n10 + n11) if n10 + n11 else 0.0
    independent = _bernoulli_log_likelihood(n01 + n11, n00 + n10, breach_rate)
    markov = _bernoulli_log_likelihood(n01, n00, rate_after_covered) + (
        _bernoulli_log_likelihood(n11, n10, rate_after_breach)
    )
    statistic = _likelihood_ratio(independent, markov)
    return {
        "transition_counts": {
            "covered_covered": n00,
            "covered_breach": n01,
            "breach_covered": n10,
            "breach_breach": n11,
        },
        "breach_rate_after_covered": rate_after_covered,
        "breach_rate_after_breach": rate_after_breach,
        "likelihood_ratio": statistic,
        "p_value": _chi_square_survival_df1(statistic),
    }


def _longest_streak(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def audit_interval_breaches(
    artifact: dict[str, Any], policy: BreachAuditPolicy | None = None
) -> dict[str, Any]:
    """Audit marginal coverage and serial independence of interval breaches."""

    active_policy = policy or BreachAuditPolicy()
    _validate_policy(active_policy)
    if not isinstance(artifact, dict):
        raise BreachAuditInputError("artifact must be an object")
    if artifact.get("schema_version") != 1:
        raise BreachAuditInputError("artifact.schema_version must equal 1")
    target_coverage = _finite_number(artifact.get("target_coverage"), "target_coverage")
    if not 0.0 < target_coverage < 1.0:
        raise BreachAuditInputError("target_coverage must be between zero and one")
    rows = artifact.get("observations")
    if not isinstance(rows, list):
        raise BreachAuditInputError("artifact.observations must be a list")
    if len(rows) < active_policy.min_observations:
        raise BreachAuditInputError(
            f"artifact requires at least {active_policy.min_observations} observations"
        )
    if len(rows) > active_policy.max_observations:
        raise BreachAuditInputError(
            f"artifact exceeds policy.max_observations={active_policy.max_observations}"
        )
    expected_breaches = len(rows) * (1.0 - target_coverage)
    if expected_breaches < active_policy.min_expected_breaches:
        raise BreachAuditInputError(
            "expected breach count is below policy.min_expected_breaches; "
            "increase the evidence window"
        )

    normalized: list[dict[str, float | str]] = []
    breaches: list[bool] = []
    lower_breaches = upper_breaches = 0
    previous_timestamp: datetime | None = None
    breach_timestamps: list[str] = []
    for index, row in enumerate(rows):
        field = f"observations[{index}]"
        if not isinstance(row, dict):
            raise BreachAuditInputError(f"{field} must be an object")
        timestamp, timestamp_text = _utc_timestamp(row.get("timestamp"), f"{field}.timestamp")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise BreachAuditInputError("observation timestamps must be unique and increasing")
        previous_timestamp = timestamp
        actual = _finite_number(row.get("actual"), f"{field}.actual")
        lower = _finite_number(row.get("lower"), f"{field}.lower")
        upper = _finite_number(row.get("upper"), f"{field}.upper")
        if lower > upper:
            raise BreachAuditInputError(f"{field}.lower cannot exceed upper")
        lower_miss = actual < lower
        upper_miss = actual > upper
        breach = lower_miss or upper_miss
        lower_breaches += lower_miss
        upper_breaches += upper_miss
        breaches.append(breach)
        if breach:
            breach_timestamps.append(timestamp_text)
        normalized.append(
            {"timestamp": timestamp_text, "actual": actual, "lower": lower, "upper": upper}
        )

    coverage = _coverage_test(breaches, 1.0 - target_coverage)
    independence = _independence_test(breaches)
    longest_streak = _longest_streak(breaches)
    coverage_rejected = coverage["p_value"] < active_policy.significance_level
    independence_rejected = independence["p_value"] < active_policy.significance_level
    findings: list[str] = []
    if coverage_rejected:
        findings.append("UNCONDITIONAL_COVERAGE_REJECTED")
    if independence_rejected:
        findings.append("BREACH_INDEPENDENCE_REJECTED")
    if longest_streak > active_policy.max_consecutive_breaches:
        findings.append("BREACH_STREAK_EXCEEDS_POLICY")

    conditional_statistic = coverage["likelihood_ratio"] + independence["likelihood_ratio"]
    canonical = json.dumps(
        {"schema_version": 1, "target_coverage": target_coverage, "observations": normalized},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    timestamp_limit = active_policy.max_evidence_timestamps
    return {
        "audit": "forecast_interval_breach_independence",
        "schema_version": 1,
        "accepted": not findings,
        "findings": findings or ["ALL_BREACH_GATES_PASSED"],
        "policy": asdict(active_policy),
        "evidence": {
            "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
            "observation_count": len(rows),
            "breach_count": sum(breaches),
            "lower_breach_count": lower_breaches,
            "upper_breach_count": upper_breaches,
            "breach_timestamps": breach_timestamps[:timestamp_limit],
            "breach_timestamps_truncated": len(breach_timestamps) > timestamp_limit,
        },
        "metrics": {
            "coverage": coverage,
            "independence": independence,
            "conditional_coverage": {
                "likelihood_ratio": conditional_statistic,
                "p_value": math.exp(-conditional_statistic / 2.0),
                "degrees_of_freedom": 2,
            },
            "longest_consecutive_breach_streak": longest_streak,
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BreachAuditInputError(f"JSON contains duplicate field {key!r}")
        value[key] = item
    return value


def load_artifact(path: Path, max_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    if path.stat().st_size > max_bytes:
        raise BreachAuditInputError(f"artifact exceeds {max_bytes} bytes")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BreachAuditInputError(f"cannot read artifact: {exc}") from exc
    if not isinstance(value, dict):
        raise BreachAuditInputError("artifact must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit forecast-interval coverage and breach clustering."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--max-consecutive-breaches", type=int, default=3)
    parser.add_argument("--min-expected-breaches", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = BreachAuditPolicy(
            min_observations=args.min_observations,
            significance_level=args.significance_level,
            max_consecutive_breaches=args.max_consecutive_breaches,
            min_expected_breaches=args.min_expected_breaches,
        )
        report = audit_interval_breaches(load_artifact(args.artifact), policy)
    except (BreachAuditInputError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "invalid_input",
                    "error_code": "INVALID_INTERVAL_EVIDENCE",
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
