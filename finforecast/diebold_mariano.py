"""Horizon-aware Diebold-Mariano release audit for paired forecast losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

ARTIFACT_SCHEMA = "finforecast-diebold-mariano/v1"
REPORT_SCHEMA = "finforecast-diebold-mariano-report/v1"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuditInputError(ValueError):
    """Raised when an audit artifact or policy is invalid."""


@dataclass(frozen=True)
class DieboldMarianoPolicy:
    min_observations: int = 60
    significance_level: float = 0.05
    min_relative_loss_improvement: float = 0.01
    max_failed_case_fraction: float = 0.0
    max_cases: int = 64
    max_observations_per_case: int = 100_000
    max_hac_lag: int = 252
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES
    max_age_seconds: int = 7 * 24 * 60 * 60
    max_future_skew_seconds: int = 300
    variance_epsilon: float = 1e-15

    def validate(self) -> None:
        max_observations = _integer(
            "max_observations_per_case", self.max_observations_per_case, 3, 1_000_000
        )
        _integer("min_observations", self.min_observations, 3, max_observations)
        _number("significance_level", self.significance_level, 1e-6, 0.5)
        _number("min_relative_loss_improvement", self.min_relative_loss_improvement, 0.0, 1.0)
        _number("max_failed_case_fraction", self.max_failed_case_fraction, 0.0, 1.0)
        _integer("max_cases", self.max_cases, 1, 10_000)
        _integer("max_hac_lag", self.max_hac_lag, 0, 10_000)
        _integer("max_artifact_bytes", self.max_artifact_bytes, 1, 32 * 1024 * 1024)
        _integer("max_age_seconds", self.max_age_seconds, 1, 365 * 24 * 60 * 60)
        _integer("max_future_skew_seconds", self.max_future_skew_seconds, 0, 86_400)
        _number("variance_epsilon", self.variance_epsilon, 0.0, 1.0, lower_inclusive=False)


def _fail(message: str) -> NoReturn:
    raise AuditInputError(message)


def _integer(name: str, value: object, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        _fail(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _number(
    name: str,
    value: object,
    lower: float,
    upper: float,
    *,
    lower_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= lower if lower_inclusive else result > lower
    if not math.isfinite(result) or not lower_ok or result > upper:
        opening = "[" if lower_inclusive else "("
        _fail(f"{name} must be finite and in {opening}{lower}, {upper}]")
    return result


def _object(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"{name} must contain exactly: {', '.join(sorted(fields))}")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{name} keys must be strings")
    return value


def _identifier(value: object, name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        _fail(f"{name} must be a non-empty string of at most {max_length} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        _fail(f"{name} must not contain control characters")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _fail(f"{name} must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AuditInputError(f"{name} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditInputError("artifact must be canonical JSON with finite values") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hac_long_run_variance(values: list[float], lag: int) -> float:
    """Newey-West long-run variance with Bartlett weights and n denominator."""
    sample_size = len(values)
    average = math.fsum(values) / sample_size
    centered = [value - average for value in values]
    gamma_zero = math.fsum(value * value for value in centered) / sample_size
    result = gamma_zero
    for offset in range(1, lag + 1):
        covariance = (
            math.fsum(
                centered[index] * centered[index - offset] for index in range(offset, sample_size)
            )
            / sample_size
        )
        bartlett_weight = 1.0 - offset / (lag + 1.0)
        result += 2.0 * bartlett_weight * covariance
    return result


def _normal_upper_tail(value: float) -> float:
    return 0.5 * math.erfc(value / math.sqrt(2.0))


def _parse_observations(
    value: object,
    *,
    case_index: int,
    policy: DieboldMarianoPolicy,
) -> tuple[list[float], list[float]]:
    if not isinstance(value, list):
        _fail(f"cases[{case_index}].observations must be a list")
    if not policy.min_observations <= len(value) <= policy.max_observations_per_case:
        _fail(f"cases[{case_index}].observations is outside the configured bounds")

    previous_timestamp: datetime | None = None
    baseline_losses: list[float] = []
    candidate_losses: list[float] = []
    for observation_index, raw_observation in enumerate(value):
        prefix = f"cases[{case_index}].observations[{observation_index}]"
        observation = _object(
            raw_observation,
            prefix,
            {"timestamp", "baseline_loss", "candidate_loss"},
        )
        timestamp = _timestamp(observation["timestamp"], f"{prefix}.timestamp")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            _fail(f"cases[{case_index}] timestamps must be strictly increasing")
        previous_timestamp = timestamp
        baseline_losses.append(
            _number(f"{prefix}.baseline_loss", observation["baseline_loss"], 0.0, 1e12)
        )
        candidate_losses.append(
            _number(f"{prefix}.candidate_loss", observation["candidate_loss"], 0.0, 1e12)
        )
    return baseline_losses, candidate_losses


def _audit_case(
    value: object,
    *,
    case_index: int,
    policy: DieboldMarianoPolicy,
) -> dict[str, object]:
    case = _object(
        value,
        f"cases[{case_index}]",
        {"case_id", "forecast_horizon", "hac_lag", "observations"},
    )
    case_id = _identifier(case["case_id"], f"cases[{case_index}].case_id")
    horizon = _integer(
        f"cases[{case_index}].forecast_horizon",
        case["forecast_horizon"],
        1,
        policy.max_observations_per_case - 1,
    )
    hac_lag = _integer(
        f"cases[{case_index}].hac_lag",
        case["hac_lag"],
        0,
        policy.max_hac_lag,
    )
    if hac_lag < horizon - 1:
        _fail(f"cases[{case_index}].hac_lag must be at least forecast_horizon - 1")

    baseline, candidate = _parse_observations(
        case["observations"], case_index=case_index, policy=policy
    )
    sample_size = len(baseline)
    if horizon >= sample_size:
        _fail(f"cases[{case_index}].forecast_horizon must be smaller than the sample")
    if hac_lag >= sample_size:
        _fail(f"cases[{case_index}].hac_lag must be smaller than the sample")

    baseline_mean = math.fsum(baseline) / sample_size
    candidate_mean = math.fsum(candidate) / sample_size
    if baseline_mean <= policy.variance_epsilon:
        _fail(f"cases[{case_index}] baseline mean loss must be positive")
    differentials = [
        baseline_loss - candidate_loss
        for baseline_loss, candidate_loss in zip(baseline, candidate, strict=True)
    ]
    mean_improvement = math.fsum(differentials) / sample_size
    relative_improvement = mean_improvement / baseline_mean
    long_run_variance = _hac_long_run_variance(differentials, hac_lag)
    if not math.isfinite(long_run_variance) or long_run_variance <= policy.variance_epsilon:
        _fail(f"cases[{case_index}] HAC variance is not positive; the DM test is undefined")
    standard_error = math.sqrt(long_run_variance / sample_size)
    dm_statistic = mean_improvement / standard_error

    harvey_numerator = sample_size + 1 - 2 * horizon + (horizon * (horizon - 1)) / sample_size
    if harvey_numerator <= 0:
        _fail(f"cases[{case_index}] cannot apply the Harvey correction")
    harvey_factor = math.sqrt(harvey_numerator / sample_size)
    adjusted_statistic = dm_statistic * harvey_factor
    one_sided_p_value = _normal_upper_tail(adjusted_statistic)
    two_sided_p_value = min(1.0, 2.0 * _normal_upper_tail(abs(adjusted_statistic)))

    reasons: list[str] = []
    if mean_improvement <= 0.0:
        reasons.append("candidate_mean_loss_not_better")
    if relative_improvement < policy.min_relative_loss_improvement:
        reasons.append("relative_loss_improvement_below_threshold")
    if one_sided_p_value > policy.significance_level:
        reasons.append("dm_significance_not_reached")

    return {
        "case_sha256": _sha256_text(case_id),
        "passed": not reasons,
        "reasons": reasons,
        "observation_count": sample_size,
        "forecast_horizon": horizon,
        "hac_lag": hac_lag,
        "metrics": {
            "baseline_mean_loss": baseline_mean,
            "candidate_mean_loss": candidate_mean,
            "mean_loss_improvement": mean_improvement,
            "relative_loss_improvement": relative_improvement,
            "hac_long_run_variance": long_run_variance,
            "standard_error": standard_error,
            "dm_statistic": dm_statistic,
            "harvey_correction_factor": harvey_factor,
            "harvey_adjusted_statistic": adjusted_statistic,
            "one_sided_p_value": one_sided_p_value,
            "two_sided_p_value": two_sided_p_value,
        },
    }


def audit_artifact(
    payload: object,
    *,
    policy: DieboldMarianoPolicy | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate paired loss evidence and apply the configured release policy."""
    active_policy = policy or DieboldMarianoPolicy()
    active_policy.validate()
    artifact_bytes = _canonical_bytes(payload)
    if len(artifact_bytes) > active_policy.max_artifact_bytes:
        _fail("artifact exceeds max_artifact_bytes")

    root = _object(
        payload,
        "artifact",
        {
            "schema",
            "generated_at",
            "benchmark_id",
            "baseline_model_sha256",
            "candidate_model_sha256",
            "loss_name",
            "loss_config_sha256",
            "cases",
        },
    )
    if root["schema"] != ARTIFACT_SCHEMA:
        _fail(f"schema must equal {ARTIFACT_SCHEMA}")
    generated_at = _timestamp(root["generated_at"], "generated_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    age_seconds = (current - generated_at).total_seconds()
    if age_seconds > active_policy.max_age_seconds:
        _fail("artifact is stale")
    if age_seconds < -active_policy.max_future_skew_seconds:
        _fail("artifact timestamp is too far in the future")

    benchmark_id = _identifier(root["benchmark_id"], "benchmark_id")
    baseline_digest = _digest(root["baseline_model_sha256"], "baseline_model_sha256")
    candidate_digest = _digest(root["candidate_model_sha256"], "candidate_model_sha256")
    if baseline_digest == candidate_digest:
        _fail("baseline and candidate model digests must differ")
    loss_name = _identifier(root["loss_name"], "loss_name")
    loss_config_digest = _digest(root["loss_config_sha256"], "loss_config_sha256")
    cases = root["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= active_policy.max_cases:
        _fail("cases must be a non-empty list within max_cases")

    case_results: list[dict[str, object]] = []
    seen_case_hashes: set[str] = set()
    total_observations = 0
    for index, case in enumerate(cases):
        result = _audit_case(case, case_index=index, policy=active_policy)
        case_hash = str(result["case_sha256"])
        if case_hash in seen_case_hashes:
            _fail("case_id values must be unique")
        seen_case_hashes.add(case_hash)
        total_observations += int(result["observation_count"])
        case_results.append(result)

    failed_count = sum(not bool(result["passed"]) for result in case_results)
    failed_fraction = failed_count / len(case_results)
    accepted = failed_fraction <= active_policy.max_failed_case_fraction
    policy_payload = asdict(active_policy)
    artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
    policy_digest = _sha256_json(policy_payload)
    evidence = {
        "artifact_sha256": artifact_digest,
        "policy_sha256": policy_digest,
        "cases": case_results,
    }
    return {
        "schema": REPORT_SCHEMA,
        "accepted": accepted,
        "reason": "accepted" if accepted else "forecast_comparison_policy_rejected",
        "identity": {
            "benchmark_sha256": _sha256_text(benchmark_id),
            "baseline_model_sha256": baseline_digest,
            "candidate_model_sha256": candidate_digest,
            "loss_name_sha256": _sha256_text(loss_name),
            "loss_config_sha256": loss_config_digest,
        },
        "summary": {
            "case_count": len(case_results),
            "failed_case_count": failed_count,
            "failed_case_fraction": failed_fraction,
            "total_observations": total_observations,
        },
        "policy": policy_payload,
        "cases": case_results,
        "artifact_sha256": artifact_digest,
        "policy_sha256": policy_digest,
        "evidence_sha256": _sha256_json(evidence),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuditInputError("unable to read artifact") from exc
    if len(raw) > max_bytes:
        _fail("artifact exceeds the input byte budget")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: _fail(f"non-finite JSON value: {value}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditInputError("artifact must be valid UTF-8 JSON") from exc


def write_report_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--max-failed-case-fraction", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = DieboldMarianoPolicy(
            min_observations=args.min_observations,
            significance_level=args.significance_level,
            min_relative_loss_improvement=args.min_relative_improvement,
            max_failed_case_fraction=args.max_failed_case_fraction,
        )
        artifact = load_artifact(args.artifact, max_bytes=policy.max_artifact_bytes)
        report = audit_artifact(artifact, policy=policy)
    except AuditInputError as exc:
        print(json.dumps({"accepted": False, "reason": "malformed_artifact", "error": str(exc)}))
        return 3

    if args.output is not None:
        write_report_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
