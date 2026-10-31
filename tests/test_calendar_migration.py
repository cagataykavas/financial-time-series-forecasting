from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from finforecast.business_calendar import PINNED_TZDATA_VERSION, load_business_calendar
from finforecast.calendar_migration import (
    CalendarMigrationError,
    audit_calendar_migration,
    build_calendar_behavior_manifest,
    dump_calendar_behavior_manifest,
    load_calendar_behavior_manifest,
    require_calendar_migration,
)


def _calendar_artifact() -> dict[str, object]:
    return {
        "schema_version": 2,
        "calendar_id": "USD-LOCAL-SETTLEMENT",
        "revision_id": "2026f-r1",
        "timezone": "America/New_York",
        "tzdata_version": PINNED_TZDATA_VERSION,
        "valid_from": "2024-03-07",
        "valid_through": "2024-03-13",
        "weekend_days": [5, 6],
        "holidays": ["2024-03-08"],
    }


def _write_calendar(tmp_path: Path) -> Path:
    path = tmp_path / "calendar.json"
    path.write_text(json.dumps(_calendar_artifact()), encoding="utf-8")
    return path


def _resign(manifest: dict[str, object]) -> None:
    document = dict(manifest)
    document.pop("manifest_sha256", None)
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()


def _old_runtime_manifest(candidate: dict[str, object]) -> dict[str, object]:
    previous = copy.deepcopy(candidate)
    previous["revision_id"] = "2026e-r1"
    previous["tzdata_version"] = "2026.5"
    _resign(previous)
    return previous


def test_behavior_manifest_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    calendar = load_business_calendar(_write_calendar(tmp_path))

    first = build_calendar_behavior_manifest(calendar)
    second = build_calendar_behavior_manifest(calendar)
    path = tmp_path / "baseline.json"
    path.write_bytes(dump_calendar_behavior_manifest(first))

    assert first == second
    assert load_calendar_behavior_manifest(path) == first
    assert len(first["day_digests"]) == 7
    assert first["transition_scan_step_seconds"] == 900
    assert first["transition_probe_offsets_seconds"] == [-1, 0, 1]
    assert [row["transition_utc"] for row in first["transition_digests"]] == [
        "2024-03-10T07:00:00+00:00"
    ]
    assert first["transition_digests"][0]["local_dates"] == ["2024-03-10"]
    assert len(first["manifest_sha256"]) == 64


def test_migration_accepts_a_new_release_with_identical_behavior(tmp_path: Path) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    previous = _old_runtime_manifest(candidate)

    report = audit_calendar_migration(previous, candidate)

    assert report["accepted"] is True
    assert report["observed_changed_dates"] == []
    assert report["finding_codes"] == []
    assert require_calendar_migration(previous, candidate) == report


def test_migration_requires_an_exact_reviewed_change_set(tmp_path: Path) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    previous = _old_runtime_manifest(candidate)
    changed_date = candidate["day_digests"][2]["date"]
    candidate["day_digests"][2]["sha256"] = "f" * 64
    _resign(candidate)

    rejected = audit_calendar_migration(previous, candidate)
    accepted = audit_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=[changed_date],
    )

    assert rejected["accepted"] is False
    assert rejected["finding_codes"] == ["UNAPPROVED_BEHAVIOR_CHANGE"]
    assert rejected["unexpected_changed_dates"] == [changed_date]
    assert accepted["accepted"] is True
    assert accepted["observed_changed_dates"] == [changed_date]
    with pytest.raises(CalendarMigrationError, match="policy failed"):
        require_calendar_migration(previous, candidate)


def test_migration_detects_transition_change_between_daily_probes(tmp_path: Path) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    previous = _old_runtime_manifest(candidate)
    transition = candidate["transition_digests"][0]["transition_utc"]
    candidate["transition_digests"][0]["sha256"] = "f" * 64
    _resign(candidate)

    rejected = audit_calendar_migration(previous, candidate)
    accepted = audit_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=["2024-03-10"],
    )

    assert rejected["observed_changed_dates"] == ["2024-03-10"]
    assert rejected["observed_changed_transition_utc"] == [transition]
    assert rejected["finding_codes"] == ["UNAPPROVED_BEHAVIOR_CHANGE"]
    assert accepted["accepted"] is True
    assert accepted["observed_changed_transition_utc"] == [transition]


def test_transition_changes_are_reviewed_by_calendar_local_date(tmp_path: Path) -> None:
    artifact = _calendar_artifact()
    artifact.update(
        timezone="Pacific/Auckland",
        valid_from="2024-09-28",
        valid_through="2024-09-30",
        holidays=[],
    )
    path = tmp_path / "new-zealand-calendar.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    candidate = build_calendar_behavior_manifest(load_business_calendar(path))
    previous = _old_runtime_manifest(candidate)
    candidate["transition_digests"][0]["sha256"] = "f" * 64
    _resign(candidate)

    report = audit_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=["2024-09-29"],
    )

    assert candidate["transition_digests"][0]["transition_utc"] == "2024-09-28T14:00:00+00:00"
    assert candidate["transition_digests"][0]["local_dates"] == ["2024-09-29"]
    assert report["observed_changed_dates"] == ["2024-09-29"]
    assert report["accepted"] is True


def test_migration_rejects_missing_expected_change_and_rule_drift(tmp_path: Path) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    previous = _old_runtime_manifest(candidate)
    expected_date = candidate["day_digests"][0]["date"]

    missing = audit_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=[expected_date],
    )
    candidate["calendar_rules_sha256"] = "e" * 64
    _resign(candidate)
    changed_rules = audit_calendar_migration(previous, candidate)

    assert missing["finding_codes"] == ["APPROVED_BEHAVIOR_CHANGE_NOT_OBSERVED"]
    assert changed_rules["accepted"] is False
    assert "CALENDAR_RULES_CHANGED" in changed_rules["finding_codes"]


def test_manifest_rejects_tampering_duplicate_keys_and_noncanonical_approvals(
    tmp_path: Path,
) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    previous = _old_runtime_manifest(candidate)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps({**candidate, "revision_id": "tampered"}), encoding="utf-8")

    with pytest.raises(CalendarMigrationError, match="digest mismatch"):
        load_calendar_behavior_manifest(path)
    path.write_text('{"schema_version":"x","schema_version":"x"}', encoding="utf-8")
    with pytest.raises(CalendarMigrationError, match="duplicate keys"):
        load_calendar_behavior_manifest(path)
    with pytest.raises(CalendarMigrationError, match="sorted and unique"):
        audit_calendar_migration(
            previous,
            candidate,
            approved_changed_dates=["2024-03-08", "2024-03-08"],
        )


def test_manifest_rejects_duplicate_or_noncanonical_transition_rows(tmp_path: Path) -> None:
    candidate = build_calendar_behavior_manifest(load_business_calendar(_write_calendar(tmp_path)))
    duplicate = copy.deepcopy(candidate)
    duplicate["transition_digests"].append(copy.deepcopy(duplicate["transition_digests"][0]))
    _resign(duplicate)

    with pytest.raises(CalendarMigrationError, match="not canonical"):
        dump_calendar_behavior_manifest(duplicate)

    malformed = copy.deepcopy(candidate)
    malformed["transition_digests"][0]["transition_utc"] = "2024-03-10T07:00:00Z"
    _resign(malformed)
    with pytest.raises(CalendarMigrationError, match="not canonical"):
        dump_calendar_behavior_manifest(malformed)


def test_manifest_records_empty_transition_set_for_fixed_offset_zone(tmp_path: Path) -> None:
    artifact = _calendar_artifact()
    artifact["timezone"] = "Asia/Tokyo"
    path = tmp_path / "fixed-zone-calendar.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    manifest = build_calendar_behavior_manifest(load_business_calendar(path))

    assert manifest["transition_digests"] == []


def test_manifest_builder_bounds_calendar_coverage(tmp_path: Path) -> None:
    artifact = _calendar_artifact()
    artifact["valid_from"] = "2000-01-01"
    artifact["valid_through"] = "2011-01-01"
    artifact["holidays"] = []
    path = tmp_path / "wide-calendar.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    calendar = load_business_calendar(path)

    with pytest.raises(CalendarMigrationError, match="3660 days"):
        build_calendar_behavior_manifest(calendar)
