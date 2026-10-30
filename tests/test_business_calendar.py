from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from finforecast.business_calendar import load_business_calendar


def _artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "calendar_id": "USD-SETTLEMENT",
        "revision_id": "2024-r1",
        "timezone": "UTC",
        "valid_from": "2024-01-01",
        "valid_through": "2024-12-31",
        "weekend_days": [5, 6],
        "holidays": ["2024-01-01", "2024-07-04"],
    }


def _write(tmp_path: Path, artifact: object) -> Path:
    path = tmp_path / "settlement-calendar.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path


def test_calendar_digest_is_canonical_and_metadata_bound(tmp_path: Path) -> None:
    artifact = _artifact()
    calendar = load_business_calendar(_write(tmp_path, artifact))
    expected = hashlib.sha256(
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

    assert calendar.calendar_id == "USD-SETTLEMENT"
    assert calendar.revision_id == "2024-r1"
    assert calendar.sha256 == expected


@pytest.mark.parametrize(
    ("timestamp", "convention", "expected"),
    [
        ("2024-07-04T14:30:00Z", "FOLLOWING", "2024-07-05T14:30:00Z"),
        ("2024-06-30T14:30:00Z", "MODIFIED_FOLLOWING", "2024-06-28T14:30:00Z"),
        ("2024-07-07T14:30:00Z", "PRECEDING", "2024-07-05T14:30:00Z"),
        ("2024-07-03T14:30:00Z", "FOLLOWING", "2024-07-03T14:30:00Z"),
    ],
)
def test_calendar_applies_reviewed_roll_conventions(
    tmp_path: Path,
    timestamp: str,
    convention: str,
    expected: str,
) -> None:
    calendar = load_business_calendar(_write(tmp_path, _artifact()))
    assert calendar.adjust(pd.Timestamp(timestamp), convention) == pd.Timestamp(expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"timezone": "America/New_York"}, "timezone must be UTC"),
        ({"weekend_days": [6, 5]}, "weekend_days must be sorted"),
        ({"holidays": ["2024-07-04", "2024-01-01"]}, "holidays must be sorted"),
        ({"valid_from": "2025-01-01"}, "coverage is reversed"),
    ],
)
def test_calendar_rejects_ambiguous_or_nondeterministic_artifacts(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    artifact = _artifact()
    artifact.update(mutation)
    with pytest.raises(ValueError, match=message):
        load_business_calendar(_write(tmp_path, artifact))


def test_calendar_rejects_unknown_fields_and_out_of_coverage_adjustment(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["unreviewed_policy"] = True
    with pytest.raises(ValueError, match="does not match schema"):
        load_business_calendar(_write(tmp_path, artifact))

    calendar = load_business_calendar(_write(tmp_path, _artifact()))
    with pytest.raises(ValueError, match="outside business-calendar coverage"):
        calendar.adjust(pd.Timestamp("2025-01-01T00:00:00Z"), "FOLLOWING")


def test_calendar_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-calendar.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"calendar_id":"USD",'
        '"revision_id":"r1","timezone":"UTC","valid_from":"2024-01-01",'
        '"valid_through":"2024-12-31","weekend_days":[5,6],"holidays":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        load_business_calendar(path)
