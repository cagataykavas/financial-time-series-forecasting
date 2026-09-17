from __future__ import annotations

import pandas as pd
import pytest

from finforecast.data import load_market_csv, market_data_from_frame


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


def test_frame_and_csv_ingestion_have_identical_semantic_identity(tmp_path):
    frame = pd.DataFrame(observations())
    path = tmp_path / "market.csv"
    frame.to_csv(path, index=False)

    from_csv = load_market_csv(path)
    from_frame = market_data_from_frame(frame, source_name="market.csv")

    assert from_frame.sha256 == from_csv.sha256
    assert from_frame.metadata() == from_csv.metadata()
