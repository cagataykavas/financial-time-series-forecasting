from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.breach_audit import (
    BreachAuditInputError,
    BreachAuditPolicy,
    audit_interval_breaches,
    load_artifact,
)


def _artifact(breach_indices: set[int], count: int = 200) -> dict:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    observations = []
    for index in range(count):
        actual = 2.0 if index in breach_indices else 0.0
        observations.append(
            {
                "timestamp": (start + timedelta(hours=index)).isoformat(),
                "actual": actual,
                "lower": -1.0,
                "upper": 1.0,
            }
        )
    return {"schema_version": 1, "target_coverage": 0.9, "observations": observations}


def _dispersed_breaches() -> set[int]:
    # Deterministic but irregular spacing avoids encoding a periodic dependence pattern.
    return {3, 11, 27, 36, 52, 68, 75, 89, 104, 117, 126, 143, 151, 166, 179, 193}


def test_accepts_well_calibrated_dispersed_breaches() -> None:
    report = audit_interval_breaches(_artifact(_dispersed_breaches()))

    assert report["accepted"] is True
    assert report["findings"] == ["ALL_BREACH_GATES_PASSED"]
    assert report["evidence"]["breach_count"] == 16
    assert report["metrics"]["coverage"]["observed_breach_rate"] == pytest.approx(0.08)
    assert report["metrics"]["longest_consecutive_breach_streak"] == 1


def test_rejects_excess_marginal_breaches() -> None:
    report = audit_interval_breaches(_artifact(set(range(0, 200, 3))))

    assert report["accepted"] is False
    assert "UNCONDITIONAL_COVERAGE_REJECTED" in report["findings"]


def test_rejects_clustered_breaches_with_acceptable_total_count() -> None:
    report = audit_interval_breaches(_artifact(set(range(80, 100))))

    assert report["evidence"]["breach_count"] == 20
    assert "BREACH_INDEPENDENCE_REJECTED" in report["findings"]
    assert "BREACH_STREAK_EXCEEDS_POLICY" in report["findings"]
    assert report["metrics"]["independence"]["breach_rate_after_breach"] == pytest.approx(0.95)


def test_records_lower_and_upper_breaches_separately() -> None:
    artifact = _artifact(_dispersed_breaches())
    for index in sorted(_dispersed_breaches())[:5]:
        artifact["observations"][index]["actual"] = -2.0

    report = audit_interval_breaches(artifact)

    assert report["evidence"]["lower_breach_count"] == 5
    assert report["evidence"]["upper_breach_count"] == 11


def test_interval_boundaries_are_covered() -> None:
    artifact = _artifact(_dispersed_breaches())
    artifact["observations"][3]["actual"] = 1.0

    report = audit_interval_breaches(artifact)

    assert report["evidence"]["breach_count"] == 15


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value.update(target_coverage=1.0), "target_coverage"),
        (lambda value: value["observations"][0].update(actual=float("nan")), "finite number"),
        (lambda value: value["observations"][0].update(lower=2.0), "cannot exceed"),
        (
            lambda value: value["observations"][0].update(timestamp="2025-01-01T00:00:00"),
            "UTC offset",
        ),
        (
            lambda value: value["observations"][1].update(
                timestamp=value["observations"][0]["timestamp"]
            ),
            "unique and increasing",
        ),
    ],
)
def test_malformed_evidence_fails_closed(mutation, message: str) -> None:
    artifact = _artifact(_dispersed_breaches())
    mutation(artifact)

    with pytest.raises(BreachAuditInputError, match=message):
        audit_interval_breaches(artifact)


def test_insufficient_expected_breaches_fails_closed() -> None:
    artifact = _artifact(set())
    artifact["target_coverage"] = 0.999

    with pytest.raises(BreachAuditInputError, match="expected breach count"):
        audit_interval_breaches(artifact)


@pytest.mark.parametrize(
    "policy",
    [
        BreachAuditPolicy(min_observations=0),
        BreachAuditPolicy(min_observations=201, max_observations=200),
        BreachAuditPolicy(significance_level=0.0),
        BreachAuditPolicy(min_expected_breaches=float("nan")),
        BreachAuditPolicy(max_consecutive_breaches=-1),
    ],
)
def test_invalid_policy_fails_closed(policy: BreachAuditPolicy) -> None:
    with pytest.raises(BreachAuditInputError, match="policy"):
        audit_interval_breaches(_artifact(_dispersed_breaches()), policy)


def test_digest_normalizes_equivalent_utc_timestamps() -> None:
    left = _artifact(_dispersed_breaches())
    right = deepcopy(left)
    for row in right["observations"]:
        row["timestamp"] = row["timestamp"].replace("+00:00", "Z")

    left_report = audit_interval_breaches(left)
    right_report = audit_interval_breaches(right)

    assert left_report["evidence"]["artifact_sha256"] == right_report["evidence"]["artifact_sha256"]
    assert left_report["metrics"] == right_report["metrics"]


def test_breach_timestamp_evidence_is_bounded() -> None:
    report = audit_interval_breaches(
        _artifact(set(range(20))), BreachAuditPolicy(max_evidence_timestamps=5)
    )

    assert len(report["evidence"]["breach_timestamps"]) == 5
    assert report["evidence"]["breach_timestamps_truncated"] is True


def test_all_covered_and_all_breached_sequences_do_not_break_statistics() -> None:
    covered = audit_interval_breaches(_artifact(set()))
    breached = audit_interval_breaches(_artifact(set(range(200))))

    assert covered["accepted"] is False
    assert breached["accepted"] is False
    assert math_is_finite_report(covered)
    assert math_is_finite_report(breached)


def math_is_finite_report(report: dict) -> bool:
    rendered = json.dumps(report, allow_nan=False)
    return bool(rendered)


def test_loader_rejects_duplicate_json_fields(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":1,"target_coverage":0.9,"target_coverage":0.8,"observations":[]}',
        encoding="utf-8",
    )

    with pytest.raises(BreachAuditInputError, match="duplicate field"):
        load_artifact(path)


def test_cli_distinguishes_acceptance_rejection_and_invalid_input(tmp_path) -> None:
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(_artifact(_dispersed_breaches())), encoding="utf-8")
    accepted = subprocess.run(
        [sys.executable, "-m", "finforecast.breach_audit", str(accepted_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["accepted"] is True

    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(_artifact(set(range(80, 100)))), encoding="utf-8")
    output_path = tmp_path / "report.json"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "finforecast.breach_audit",
            str(rejected_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is False

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, "-m", "finforecast.breach_audit", str(invalid_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 3
    assert json.loads(invalid.stderr)["error_code"] == "INVALID_INTERVAL_EVIDENCE"
