from pathlib import Path

import pandas as pd
import pytest

from finforecast.corporate_actions import (
    apply_point_in_time_corporate_actions,
    apply_point_in_time_splits,
)
from finforecast.data import load_market_csv


def market(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "close": [100.0, 102.0, 104.0, 53.0, 54.0],
            "volume": [10.0, 11.0, 12.0, 24.0, 25.0],
        }
    ).to_csv(path, index=False)
    return load_market_csv(path)


def actions(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "actions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def dividend_market(tmp_path: Path):
    path = tmp_path / "dividend-prices.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "close": [100.0, 102.0, 104.0, 101.0, 102.0],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    ).to_csv(path, index=False)
    return load_market_csv(path)


def dividend_rows() -> list[dict[str, object]]:
    return [
        {
            "action_id": "dividend-2024-01",
            "symbol": "ACME",
            "action_type": "cash_dividend",
            "effective_at": "2024-01-04T00:00:00Z",
            "available_at": "2024-01-02T10:00:00Z",
            "revision_id": "r1",
            "cash_amount": 2.0,
            "currency": "USD",
        },
        {
            "action_id": "dividend-2024-01",
            "symbol": "ACME",
            "action_type": "cash_dividend",
            "effective_at": "2024-01-04T00:00:00Z",
            "available_at": "2024-01-03T10:00:00Z",
            "revision_id": "r2",
            "cash_amount": 3.0,
            "currency": "USD",
        },
    ]


def action_rows() -> list[dict[str, object]]:
    return [
        {
            "action_id": "split-2024-01",
            "symbol": "ACME",
            "action_type": "split",
            "effective_at": "2024-01-04T00:00:00Z",
            "available_at": "2024-01-02T10:00:00Z",
            "revision_id": "r1",
            "split_ratio": 4.0,
        },
        {
            "action_id": "split-2024-01",
            "symbol": "ACME",
            "action_type": "split",
            "effective_at": "2024-01-04T00:00:00Z",
            "available_at": "2024-01-03T10:00:00Z",
            "revision_id": "r2",
            "split_ratio": 2.0,
        },
        {
            "action_id": "split-2024-02",
            "symbol": "ACME",
            "action_type": "split",
            "effective_at": "2024-02-01T00:00:00Z",
            "available_at": "2024-01-05T10:00:00Z",
            "revision_id": "r1",
            "split_ratio": 3.0,
        },
    ]


def test_applies_latest_known_effective_split_and_binds_lineage(tmp_path: Path):
    adjusted = apply_point_in_time_splits(
        market(tmp_path),
        actions(tmp_path, action_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
    )

    assert adjusted.close.tolist() == [50.0, 51.0, 52.0, 53.0, 54.0]
    assert adjusted.volume.tolist() == [20.0, 22.0, 24.0, 24.0, 25.0]
    assert adjusted.corporate_actions is not None
    snapshot = adjusted.corporate_actions
    assert snapshot.raw_rows == 3
    assert snapshot.eligible_revisions == 3
    assert snapshot.selected_actions == 2
    assert snapshot.applied_splits == 1
    assert snapshot.excluded_not_effective_actions == 1
    assert snapshot.superseded_revisions == 1
    assert snapshot.unadjusted_market_sha256 != snapshot.adjusted_market_sha256
    assert snapshot.adjusted_market_sha256 == adjusted.sha256
    assert len(snapshot.selected_actions_sha256) == len(snapshot.lineage_sha256) == 64


def test_future_revision_does_not_change_historical_adjustment(tmp_path: Path):
    rows = action_rows()
    rows[1]["available_at"] = "2024-01-07T10:00:00Z"
    adjusted = apply_point_in_time_splits(
        market(tmp_path),
        actions(tmp_path, rows),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
    )

    assert adjusted.close.iloc[:3].tolist() == [25.0, 25.5, 26.0]
    assert adjusted.corporate_actions.excluded_unavailable_revisions == 1
    assert adjusted.corporate_actions.superseded_revisions == 0


def test_applies_latest_known_cash_dividend_as_total_return_factor(tmp_path: Path):
    source = dividend_market(tmp_path)
    original_volume = source.volume.copy()
    adjusted = apply_point_in_time_corporate_actions(
        source,
        actions(tmp_path, dividend_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
    )

    factor = (104.0 - 3.0) / 104.0
    assert adjusted.close.tolist() == pytest.approx(
        [100.0 * factor, 102.0 * factor, 101.0, 101.0, 102.0]
    )
    assert adjusted.volume.equals(original_volume)
    snapshot = adjusted.corporate_actions
    assert snapshot.applied_cash_dividends == 1
    assert snapshot.applied_splits == 0
    assert snapshot.price_currency == "USD"
    assert snapshot.superseded_revisions == 1
    assert len(snapshot.selected_actions_sha256) == 64


def test_future_dividend_revision_cannot_enter_historical_snapshot(tmp_path: Path):
    rows = dividend_rows()
    rows[1]["available_at"] = "2024-01-07T10:00:00Z"
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, rows),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
    )

    factor = (104.0 - 2.0) / 104.0
    assert adjusted.close.iloc[:2].tolist() == pytest.approx([100.0 * factor, 102.0 * factor])
    assert adjusted.corporate_actions.excluded_unavailable_revisions == 1
    assert adjusted.corporate_actions.superseded_revisions == 0


@pytest.mark.parametrize(
    ("mutate", "currency", "message"),
    [
        (lambda rows: rows, None, "declared price_currency"),
        (lambda rows: rows, "TRY", "currency must match"),
        (
            lambda rows: [{**rows[0], "cash_amount": 104.0}],
            "USD",
            "smaller than its reference close",
        ),
        (
            lambda rows: [{**rows[0], "split_ratio": 2.0}],
            "USD",
            "cannot declare a split ratio",
        ),
    ],
)
def test_invalid_cash_dividend_contract_fails_closed(
    tmp_path: Path, mutate, currency: str | None, message: str
):
    with pytest.raises(ValueError, match=message):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, mutate([dividend_rows()[0]])),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency=currency,
        )


def test_action_type_cannot_change_across_revisions(tmp_path: Path):
    rows = dividend_rows()
    rows[1] = {
        **rows[1],
        "action_type": "split",
        "split_ratio": 2.0,
        "cash_amount": None,
        "currency": None,
    }
    with pytest.raises(ValueError, match="cannot change type"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, rows),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
        )


def test_multiple_splits_apply_cumulative_factors_by_effective_time(tmp_path: Path):
    rows = [
        {
            "action_id": "split-a",
            "symbol": "ACME",
            "action_type": "split",
            "effective_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-02T00:00:00Z",
            "revision_id": "r1",
            "split_ratio": 2.0,
        },
        {
            "action_id": "split-b",
            "symbol": "ACME",
            "action_type": "split",
            "effective_at": "2024-01-05T00:00:00Z",
            "available_at": "2024-01-04T00:00:00Z",
            "revision_id": "r1",
            "split_ratio": 3.0,
        },
    ]

    adjusted = apply_point_in_time_splits(
        market(tmp_path),
        actions(tmp_path, rows),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
    )

    assert adjusted.close.tolist() == pytest.approx(
        [100.0 / 6.0, 102.0 / 6.0, 104.0 / 3.0, 53.0 / 3.0, 54.0]
    )
    assert adjusted.volume.tolist() == pytest.approx([60.0, 66.0, 36.0, 72.0, 25.0])
    assert adjusted.corporate_actions.applied_splits == 2


def test_snapshot_is_deterministic_and_sensitive_to_action_revision(tmp_path: Path):
    source_market = market(tmp_path)
    path = actions(tmp_path, action_rows())
    first = apply_point_in_time_splits(
        source_market, path, as_of="2024-01-06T00:00:00Z", symbol="ACME"
    )
    second = apply_point_in_time_splits(
        source_market, path, as_of="2024-01-06T00:00:00Z", symbol="ACME"
    )
    rows = action_rows()
    rows[1]["split_ratio"] = 2.5
    changed = apply_point_in_time_splits(
        source_market,
        actions(tmp_path, rows),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
    )

    assert first.metadata() == second.metadata()
    assert first.corporate_actions.lineage_sha256 != changed.corporate_actions.lineage_sha256
    assert first.sha256 != changed.sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: [{**rows[0], "symbol": "OTHER"}, *rows[1:]], "requested symbol"),
        (
            lambda rows: [{**rows[0], "action_type": "dividend"}, *rows[1:]],
            "types are: split",
        ),
        (lambda rows: [{**rows[0], "action_id": "bad id"}, *rows[1:]], "invalid identifiers"),
        (lambda rows: [{**rows[0], "split_ratio": 1.0}, *rows[1:]], "different from one"),
        (
            lambda rows: [{**rows[0], "available_at": "2024-01-02"}, *rows[1:]],
            "timezone-aware",
        ),
        (lambda rows: [rows[1], rows[0], *rows[2:]], "must be ordered"),
        (lambda rows: [{**rows[0]}, {**rows[0]}, *rows[2:]], "ambiguous availability"),
    ],
)
def test_invalid_action_lineage_fails_closed(tmp_path: Path, mutate, message: str):
    with pytest.raises(ValueError, match=message):
        apply_point_in_time_splits(
            market(tmp_path),
            actions(tmp_path, mutate(action_rows())),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
        )


def test_market_and_action_cutoffs_must_match(tmp_path: Path):
    source = tmp_path / "vintages.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z"],
            "available_at": ["2024-01-01T01:00:00Z"],
            "revision_id": ["r1"],
            "close": [100.0],
            "volume": [10.0],
        }
    ).to_csv(source, index=False)
    from finforecast.data import load_point_in_time_market_csv

    point_in_time = load_point_in_time_market_csv(source, as_of="2024-01-02T00:00:00Z")
    with pytest.raises(ValueError, match="same as_of cutoff"):
        apply_point_in_time_splits(
            point_in_time,
            actions(tmp_path, action_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
        )


def test_already_adjusted_market_cannot_be_adjusted_twice(tmp_path: Path):
    path = actions(tmp_path, action_rows())
    adjusted = apply_point_in_time_splits(
        market(tmp_path),
        path,
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
    )

    with pytest.raises(ValueError, match="already contains"):
        apply_point_in_time_splits(
            adjusted,
            path,
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
        )
