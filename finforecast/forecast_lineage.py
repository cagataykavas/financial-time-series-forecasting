from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_RECORDS = 50_000
MAX_REASON_CODES = 64
GENESIS_HASH = "0" * 64
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,127}$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "created_at",
    "dataset_digest",
    "config_digest",
    "records",
}
_RECORD_FIELDS = {
    "forecast_id",
    "series_id",
    "model_revision",
    "horizon_steps",
    "feature_cutoff_at",
    "issued_at",
    "target_at",
    "realized_at",
    "prediction_digest",
    "actual_digest",
    "previous_hash",
    "record_hash",
}


class ArtifactError(ValueError):
    """Raised when evidence cannot be safely interpreted."""


@dataclass(frozen=True)
class LineagePolicy:
    cadence_seconds: int
    max_issue_delay_seconds: int = 3_600
    min_lead_seconds: int = 1
    max_realization_delay_seconds: int = 86_400
    max_artifact_age_seconds: int = 604_800
    max_future_skew_seconds: int = 300

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ArtifactError(f"{name} must be a non-negative integer")
        if self.cadence_seconds == 0:
            raise ArtifactError("cadence_seconds must be a positive integer")
        if self.min_lead_seconds >= self.cadence_seconds:
            raise ArtifactError("min_lead_seconds must be smaller than cadence_seconds")


def _reject_constant(value: str) -> None:
    raise ArtifactError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ArtifactError("artifact cannot be read") from exc
    if size > MAX_ARTIFACT_BYTES:
        raise ArtifactError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("artifact root must be an object")
    return payload


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def chain_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with a canonical, order-sensitive lineage chain."""
    previous = GENESIS_HASH
    chained: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        record["previous_hash"] = previous
        record["record_hash"] = record_hash(record)
        previous = record["record_hash"]
        chained.append(record)
    return chained


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ArtifactError(f"{field} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ArtifactError(f"{field} is not a valid timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ArtifactError(f"{field} must use UTC")
    return parsed


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ArtifactError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ArtifactError(f"{field} is not a bounded identifier")
    return value


def _reason(reasons: list[str], code: str) -> None:
    if code not in reasons and len(reasons) < MAX_REASON_CODES:
        reasons.append(code)


def audit_lineage(
    artifact: dict[str, Any],
    policy: LineagePolicy,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate forecast chronology and return a privacy-preserving release decision."""
    policy.validate()
    if set(artifact) != _TOP_LEVEL_FIELDS:
        raise ArtifactError("artifact fields do not match schema version 1")
    if artifact["schema_version"] != 1:
        raise ArtifactError("unsupported schema_version")
    created_at = _timestamp(artifact["created_at"], "created_at")
    dataset_digest = _digest(artifact["dataset_digest"], "dataset_digest")
    config_digest = _digest(artifact["config_digest"], "config_digest")
    records = artifact["records"]
    if not isinstance(records, list) or not records:
        raise ArtifactError("records must be a non-empty array")
    if len(records) > MAX_RECORDS:
        raise ArtifactError(f"records exceeds {MAX_RECORDS} entries")

    current = now or datetime.now(UTC)
    if current.tzinfo != UTC:
        raise ArtifactError("now must be timezone-aware UTC")
    reasons: list[str] = []
    if created_at > current + timedelta(seconds=policy.max_future_skew_seconds):
        _reason(reasons, "artifact_from_future")
    if current - created_at > timedelta(seconds=policy.max_artifact_age_seconds):
        _reason(reasons, "artifact_stale")

    seen_forecast_ids: set[str] = set()
    seen_targets: set[tuple[str, str, int, datetime]] = set()
    group_latest: dict[tuple[str, str, int], tuple[datetime, datetime, datetime]] = {}
    previous_hash = GENESIS_HASH
    max_issue_delay = 0.0
    max_realization_delay = 0.0
    min_lead = math.inf

    for index, raw in enumerate(records):
        if not isinstance(raw, dict) or set(raw) != _RECORD_FIELDS:
            raise ArtifactError(f"records[{index}] fields do not match schema version 1")
        forecast_id = _identifier(raw["forecast_id"], f"records[{index}].forecast_id")
        series_id = _identifier(raw["series_id"], f"records[{index}].series_id")
        model_revision = _digest(raw["model_revision"], f"records[{index}].model_revision")
        _digest(raw["prediction_digest"], f"records[{index}].prediction_digest")
        _digest(raw["actual_digest"], f"records[{index}].actual_digest")
        claimed_previous = _digest(raw["previous_hash"], f"records[{index}].previous_hash")
        claimed_hash = _digest(raw["record_hash"], f"records[{index}].record_hash")
        horizon = raw["horizon_steps"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 512:
            raise ArtifactError(f"records[{index}].horizon_steps must be in [1, 512]")

        cutoff = _timestamp(raw["feature_cutoff_at"], f"records[{index}].feature_cutoff_at")
        issued = _timestamp(raw["issued_at"], f"records[{index}].issued_at")
        target = _timestamp(raw["target_at"], f"records[{index}].target_at")
        realized = _timestamp(raw["realized_at"], f"records[{index}].realized_at")

        if forecast_id in seen_forecast_ids:
            _reason(reasons, "duplicate_forecast_id")
        seen_forecast_ids.add(forecast_id)
        target_key = (series_id, model_revision, horizon, target)
        if target_key in seen_targets:
            _reason(reasons, "forecast_revision_detected")
        seen_targets.add(target_key)

        if claimed_previous != previous_hash:
            _reason(reasons, "lineage_chain_broken")
        if claimed_hash != record_hash(raw):
            _reason(reasons, "record_hash_mismatch")
        previous_hash = claimed_hash

        if not cutoff <= issued < target <= realized <= created_at:
            _reason(reasons, "invalid_event_chronology")
        expected_target = cutoff + timedelta(seconds=policy.cadence_seconds * horizon)
        if target != expected_target:
            _reason(reasons, "horizon_mismatch")

        issue_delay = (issued - cutoff).total_seconds()
        lead = (target - issued).total_seconds()
        realization_delay = (realized - target).total_seconds()
        max_issue_delay = max(max_issue_delay, issue_delay)
        min_lead = min(min_lead, lead)
        max_realization_delay = max(max_realization_delay, realization_delay)
        if issue_delay < 0 or issue_delay > policy.max_issue_delay_seconds:
            _reason(reasons, "issue_delay_exceeded")
        if lead < policy.min_lead_seconds:
            _reason(reasons, "insufficient_forecast_lead")
        if realization_delay < 0 or realization_delay > policy.max_realization_delay_seconds:
            _reason(reasons, "realization_delay_exceeded")

        group_key = (series_id, model_revision, horizon)
        latest = group_latest.get(group_key)
        if latest is not None and not (
            latest[0] < cutoff and latest[1] < issued and latest[2] < target
        ):
            _reason(reasons, "non_monotonic_forecast_sequence")
        group_latest[group_key] = (cutoff, issued, target)

    evidence = {
        "schema_version": 1,
        "artifact_digest": hashlib.sha256(canonical_json(artifact)).hexdigest(),
        "dataset_digest": dataset_digest,
        "config_digest": config_digest,
        "terminal_chain_hash": previous_hash,
        "record_count": len(records),
        "group_count": len(group_latest),
        "max_issue_delay_seconds": max_issue_delay,
        "min_lead_seconds": min_lead,
        "max_realization_delay_seconds": max_realization_delay,
        "policy": asdict(policy),
        "accepted": not reasons,
        "reason_codes": reasons,
    }
    evidence["evidence_digest"] = hashlib.sha256(canonical_json(evidence)).hexdigest()
    return evidence


def _atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit point-in-time forecast lineage evidence")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--cadence-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-issue-delay-seconds", type=int, default=3_600)
    parser.add_argument("--min-lead-seconds", type=int, default=1)
    parser.add_argument("--max-realization-delay-seconds", type=int, default=86_400)
    parser.add_argument("--max-artifact-age-seconds", type=int, default=604_800)
    parser.add_argument("--max-future-skew-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        policy = LineagePolicy(
            cadence_seconds=args.cadence_seconds,
            max_issue_delay_seconds=args.max_issue_delay_seconds,
            min_lead_seconds=args.min_lead_seconds,
            max_realization_delay_seconds=args.max_realization_delay_seconds,
            max_artifact_age_seconds=args.max_artifact_age_seconds,
            max_future_skew_seconds=args.max_future_skew_seconds,
        )
        report = audit_lineage(load_artifact(args.artifact), policy)
    except ArtifactError as exc:
        print(json.dumps({"accepted": False, "error": str(exc), "kind": "malformed_artifact"}))
        return 2
    if args.output:
        _atomic_write(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["accepted"] else 3


if __name__ == "__main__":
    sys.exit(main())
