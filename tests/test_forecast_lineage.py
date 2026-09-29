from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.forecast_lineage import (
    GENESIS_HASH,
    ArtifactError,
    LineagePolicy,
    audit_lineage,
    chain_records,
    load_artifact,
    main,
)

NOW = datetime(2026, 9, 29, 3, 0, tzinfo=UTC)
DIGEST = hashlib.sha256(b"test").hexdigest()
REVISION = hashlib.sha256(b"model-v1").hexdigest()


def _record(step: int, *, horizon: int = 1) -> dict[str, object]:
    cutoff = NOW - timedelta(hours=5 - step)
    target = cutoff + timedelta(hours=horizon)
    return {
        "forecast_id": f"forecast-{step}",
        "series_id": "BTC-USD",
        "model_revision": REVISION,
        "horizon_steps": horizon,
        "feature_cutoff_at": cutoff.isoformat().replace("+00:00", "Z"),
        "issued_at": (cutoff + timedelta(minutes=2)).isoformat().replace("+00:00", "Z"),
        "target_at": target.isoformat().replace("+00:00", "Z"),
        "realized_at": (target + timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        "prediction_digest": hashlib.sha256(f"prediction-{step}".encode()).hexdigest(),
        "actual_digest": hashlib.sha256(f"actual-{step}".encode()).hexdigest(),
    }


def _artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "dataset_digest": DIGEST,
        "config_digest": DIGEST,
        "records": chain_records([_record(0), _record(1), _record(2)]),
    }


def _audit(artifact: dict[str, object]) -> dict[str, object]:
    return audit_lineage(artifact, LineagePolicy(cadence_seconds=3600), now=NOW)


def test_accepts_valid_chained_forecasts() -> None:
    report = _audit(_artifact())
    assert report["accepted"] is True
    assert report["record_count"] == 3
    assert report["group_count"] == 1
    assert report["reason_codes"] == []
    assert report["terminal_chain_hash"] != GENESIS_HASH


def test_report_is_deterministic_and_does_not_leak_identifiers() -> None:
    first = _audit(_artifact())
    second = _audit(_artifact())
    assert first == second
    encoded = json.dumps(first)
    assert "BTC-USD" not in encoded
    assert "forecast-0" not in encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_cutoff_at", "2026-09-28T22:00:00+00:00"),
        ("issued_at", "not-a-time"),
        ("target_at", "2026-09-28T23:00:00+03:00"),
    ],
)
def test_rejects_non_utc_or_invalid_timestamps(field: str, value: str) -> None:
    artifact = _artifact()
    artifact["records"][0][field] = value  # type: ignore[index]
    with pytest.raises(ArtifactError):
        _audit(artifact)


def test_detects_post_target_issuance_even_with_rehashed_chain() -> None:
    artifact = _artifact()
    records = artifact["records"]
    records[0]["issued_at"] = records[0]["realized_at"]  # type: ignore[index]
    artifact["records"] = chain_records(records)  # type: ignore[arg-type]
    report = _audit(artifact)
    assert report["accepted"] is False
    assert "invalid_event_chronology" in report["reason_codes"]
    assert "insufficient_forecast_lead" in report["reason_codes"]


def test_detects_feature_target_horizon_mismatch() -> None:
    artifact = _artifact()
    artifact["records"][0]["horizon_steps"] = 2  # type: ignore[index]
    artifact["records"] = chain_records(artifact["records"])  # type: ignore[arg-type]
    assert "horizon_mismatch" in _audit(artifact)["reason_codes"]


def test_detects_duplicate_forecast_id() -> None:
    artifact = _artifact()
    artifact["records"][1]["forecast_id"] = "forecast-0"  # type: ignore[index]
    artifact["records"] = chain_records(artifact["records"])  # type: ignore[arg-type]
    assert "duplicate_forecast_id" in _audit(artifact)["reason_codes"]


def test_detects_post_hoc_revision_for_same_target() -> None:
    artifact = _artifact()
    duplicate = copy.deepcopy(artifact["records"][0])  # type: ignore[index]
    duplicate["forecast_id"] = "forecast-revision"
    artifact["records"] = chain_records([artifact["records"][0], duplicate])  # type: ignore[index]
    assert "forecast_revision_detected" in _audit(artifact)["reason_codes"]


def test_detects_hash_tampering() -> None:
    artifact = _artifact()
    artifact["records"][1]["prediction_digest"] = DIGEST  # type: ignore[index]
    report = _audit(artifact)
    assert "record_hash_mismatch" in report["reason_codes"]


def test_detects_chain_reordering() -> None:
    artifact = _artifact()
    artifact["records"][0], artifact["records"][1] = (  # type: ignore[index]
        artifact["records"][1],
        artifact["records"][0],
    )
    report = _audit(artifact)
    assert "lineage_chain_broken" in report["reason_codes"]
    assert "non_monotonic_forecast_sequence" in report["reason_codes"]


def test_detects_issue_delay_budget() -> None:
    artifact = _artifact()
    artifact["records"][0]["issued_at"] = "2026-09-28T22:59:00Z"  # type: ignore[index]
    artifact["records"] = chain_records(artifact["records"])  # type: ignore[arg-type]
    policy = LineagePolicy(cadence_seconds=3600, max_issue_delay_seconds=30)
    assert "issue_delay_exceeded" in audit_lineage(artifact, policy, now=NOW)["reason_codes"]


def test_detects_realization_delay_budget() -> None:
    artifact = _artifact()
    artifact["records"][0]["realized_at"] = "2026-09-29T01:00:00Z"  # type: ignore[index]
    artifact["records"] = chain_records(artifact["records"])  # type: ignore[arg-type]
    policy = LineagePolicy(cadence_seconds=3600, max_realization_delay_seconds=60)
    assert "realization_delay_exceeded" in audit_lineage(artifact, policy, now=NOW)["reason_codes"]


def test_detects_stale_and_future_artifacts() -> None:
    stale = _artifact()
    stale["created_at"] = "2026-09-20T00:00:00Z"
    assert "artifact_stale" in _audit(stale)["reason_codes"]
    future = _artifact()
    future["created_at"] = "2026-09-29T04:00:00Z"
    assert "artifact_from_future" in _audit(future)["reason_codes"]


@pytest.mark.parametrize("horizon", [0, 513, True, 1.5])
def test_rejects_invalid_horizon(horizon: object) -> None:
    artifact = _artifact()
    artifact["records"][0]["horizon_steps"] = horizon  # type: ignore[index]
    with pytest.raises(ArtifactError):
        _audit(artifact)


def test_rejects_unexpected_fields() -> None:
    artifact = _artifact()
    artifact["secret"] = "must-not-be-accepted"
    with pytest.raises(ArtifactError):
        _audit(artifact)


def test_rejects_bad_digest_and_identifier() -> None:
    artifact = _artifact()
    artifact["dataset_digest"] = "ABC"
    with pytest.raises(ArtifactError):
        _audit(artifact)
    artifact = _artifact()
    artifact["records"][0]["series_id"] = "contains a space"  # type: ignore[index]
    with pytest.raises(ArtifactError):
        _audit(artifact)


def test_policy_validation_is_fail_closed() -> None:
    with pytest.raises(ArtifactError):
        audit_lineage(_artifact(), LineagePolicy(cadence_seconds=0), now=NOW)
    with pytest.raises(ArtifactError):
        audit_lineage(_artifact(), LineagePolicy(cadence_seconds="60"), now=NOW)  # type: ignore[arg-type]
    with pytest.raises(ArtifactError):
        audit_lineage(_artifact(), LineagePolicy(cadence_seconds=60, min_lead_seconds=60), now=NOW)


def test_loader_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate JSON key"):
        load_artifact(path)


def test_loader_rejects_non_finite_numbers(tmp_path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ArtifactError, match="non-finite"):
        load_artifact(path)


def test_cli_writes_atomic_report_and_uses_stable_exit_codes(tmp_path, monkeypatch) -> None:
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    monkeypatch.setattr("finforecast.forecast_lineage.datetime", _FrozenDatetime)
    arguments = [
        str(artifact_path),
        "--cadence-seconds",
        "3600",
        "--output",
        str(output_path),
    ]
    assert main(arguments) == 0
    assert json.loads(output_path.read_text())["accepted"] is True

    tampered = _artifact()
    tampered["records"][0]["prediction_digest"] = DIGEST  # type: ignore[index]
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert main([str(artifact_path), "--cadence-seconds", "3600"]) == 3

    artifact_path.write_text("{", encoding="utf-8")
    assert main([str(artifact_path), "--cadence-seconds", "3600"]) == 2


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001, ANN206
        return NOW
