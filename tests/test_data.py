from __future__ import annotations

import pandas as pd
import pytest

from finforecast.data import load_market_csv, load_point_in_time_market_csv


def write_csv(tmp_path, rows: list[dict[str, object]], name: str = "market.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def observations(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "timestamp": f"2024-01-{day:02d}",
            "close": 100.0 + day,
            "volume": 1_000 + day,
        }
        for day in range(1, count + 1)
    ]


def revision_history() -> list[dict[str, object]]:
    return [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "available_at": "2024-01-01T01:00:00Z",
            "revision_id": "vendor-r1",
            "close": 100.0,
            "volume": 1_000,
        },
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "vendor-r2",
            "close": 101.0,
            "volume": 1_010,
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "available_at": "2024-01-02T01:00:00Z",
            "revision_id": "vendor-r1",
            "close": 102.0,
            "volume": 1_020,
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "available_at": "2024-01-04T01:00:00Z",
            "revision_id": "vendor-r2",
            "close": 103.0,
            "volume": 1_030,
        },
    ]


def test_load_market_csv_normalizes_data_and_builds_stable_fingerprint(tmp_path):
    path = write_csv(tmp_path, observations())
    first = load_market_csv(path)
    second = load_market_csv(path)

    assert first.close.index.tz is not None
    assert first.close.tolist() == [101.0, 102.0, 103.0, 104.0]
    assert first.volume is not None
    assert first.volume.tolist() == [1001.0, 1002.0, 1003.0, 1004.0]
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.metadata()["source_name"] == "market.csv"


def test_fingerprint_is_semantic_not_based_on_filename(tmp_path):
    rows = observations()
    first = load_market_csv(write_csv(tmp_path, rows, "a.csv"))
    second = load_market_csv(write_csv(tmp_path, rows, "b.csv"))
    changed_rows = [*rows]
    changed_rows[-1] = {**changed_rows[-1], "close": 999.0}
    changed = load_market_csv(write_csv(tmp_path, changed_rows, "changed.csv"))

    assert first.sha256 == second.sha256
    assert changed.sha256 != first.sha256


def test_load_market_csv_can_omit_volume(tmp_path):
    rows = [{"date": row["timestamp"], "price": row["close"]} for row in observations()]
    market = load_market_csv(
        write_csv(tmp_path, rows),
        timestamp_column="date",
        close_column="price",
        volume_column=None,
    )
    assert market.volume is None
    assert not market.metadata()["has_volume"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[::-1], "increasing"),
        (lambda rows: [*rows, rows[-1]], "unique"),
        (lambda rows: [{**rows[0], "timestamp": "bad-date"}, *rows[1:]], "timestamps"),
        (lambda rows: [{**rows[0], "close": 0.0}, *rows[1:]], "positive"),
        (lambda rows: [{**rows[0], "volume": -1.0}, *rows[1:]], "negative"),
    ],
)
def test_load_market_csv_rejects_invalid_market_data(tmp_path, mutate, message: str):
    path = write_csv(tmp_path, mutate(observations()))
    with pytest.raises(ValueError, match=message):
        load_market_csv(path)


def test_load_market_csv_rejects_missing_columns_and_paths(tmp_path):
    path = write_csv(tmp_path, [{"timestamp": "2024-01-01", "close": 100.0}])
    with pytest.raises(ValueError, match="missing columns"):
        load_market_csv(path)
    with pytest.raises(ValueError, match="does not exist"):
        load_market_csv(tmp_path / "missing.csv")


def test_point_in_time_loader_selects_latest_revision_available_at_cutoff(tmp_path):
    market = load_point_in_time_market_csv(
        write_csv(tmp_path, revision_history()),
        as_of="2024-01-02T12:00:00+00:00",
    )

    assert market.close.tolist() == [100.0, 102.0]
    assert market.volume is not None
    assert market.volume.tolist() == [1_000.0, 1_020.0]
    metadata = market.metadata()
    assert metadata["kind"] == "point_in_time_market_csv"
    assert metadata["snapshot"] == {
        "as_of": "2024-01-02T12:00:00+00:00",
        "available_at_column": "available_at",
        "revision_column": "revision_id",
        "raw_rows": 4,
        "eligible_rows": 2,
        "selected_observations": 2,
        "excluded_future_revisions": 2,
        "superseded_revisions": 0,
        "selected_revisions_sha256": metadata["snapshot"]["selected_revisions_sha256"],
        "lineage_sha256": metadata["snapshot"]["lineage_sha256"],
    }
    assert len(metadata["snapshot"]["selected_revisions_sha256"]) == 64
    assert len(metadata["snapshot"]["lineage_sha256"]) == 64


def test_point_in_time_cutoff_changes_selected_values_and_identity(tmp_path):
    path = write_csv(tmp_path, revision_history())
    early = load_point_in_time_market_csv(path, as_of="2024-01-02T12:00:00Z")
    late = load_point_in_time_market_csv(path, as_of="2024-01-05T00:00:00Z")

    assert early.close.tolist() == [100.0, 102.0]
    assert late.close.tolist() == [101.0, 103.0]
    assert early.sha256 != late.sha256
    assert early.snapshot is not None
    assert late.snapshot is not None
    assert early.snapshot.selected_revisions_sha256 != late.snapshot.selected_revisions_sha256
    assert early.snapshot.lineage_sha256 != late.snapshot.lineage_sha256
    assert late.snapshot.superseded_revisions == 2


@pytest.mark.parametrize(
    ("mutate", "as_of", "message"),
    [
        (lambda rows: rows, "2024-01-02", "explicit timezone"),
        (lambda rows: rows, "2023-12-31T00:00:00Z", "no market revisions"),
        (
            lambda rows: [
                {**rows[0], "available_at": "2023-12-31T23:00:00Z"},
                *rows[1:],
            ],
            "2024-01-05T00:00:00Z",
            "before observation",
        ),
        (
            lambda rows: [{**rows[0], "revision_id": "bad revision"}, *rows[1:]],
            "2024-01-05T00:00:00Z",
            "invalid revision IDs",
        ),
        (
            lambda rows: [rows[1], rows[0], *rows[2:]],
            "2024-01-05T00:00:00Z",
            "ordered",
        ),
        (
            lambda rows: [{**rows[0]}, {**rows[0]}, *rows[2:]],
            "2024-01-05T00:00:00Z",
            "ambiguous availability",
        ),
    ],
)
def test_point_in_time_loader_rejects_ambiguous_lineage(tmp_path, mutate, as_of, message):
    path = write_csv(tmp_path, mutate(revision_history()))
    with pytest.raises(ValueError, match=message):
        load_point_in_time_market_csv(path, as_of=as_of)


def test_revision_columns_require_an_as_of_cutoff(tmp_path):
    with pytest.raises(ValueError, match="require an as_of"):
        load_market_csv(
            write_csv(tmp_path, revision_history()),
            available_at_column="available_at",
            revision_column="revision_id",
        )
