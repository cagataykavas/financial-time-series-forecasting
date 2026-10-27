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


def foreign_dividend_rows() -> list[dict[str, object]]:
    rows = dividend_rows()
    for row in rows:
        row["currency"] = "EUR"
    return rows


def delisting_rows(currency: str = "USD") -> list[dict[str, object]]:
    return [
        {
            "action_id": "delisting-2024-01",
            "symbol": "ACME",
            "action_type": "cash_delisting",
            "effective_at": "2024-01-06T00:00:00Z",
            "available_at": "2024-01-03T10:00:00Z",
            "revision_id": "r1",
            "cash_amount": 40.0,
            "currency": currency,
        },
        {
            "action_id": "delisting-2024-01",
            "symbol": "ACME",
            "action_type": "cash_delisting",
            "effective_at": "2024-01-06T00:00:00Z",
            "available_at": "2024-01-05T10:00:00Z",
            "revision_id": "r2",
            "cash_amount": 45.0,
            "currency": currency,
        },
    ]


def fx_rates(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    path = tmp_path / "fx-rates.csv"
    pd.DataFrame(
        rows
        or [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "observed_at": "2024-01-03T00:00:00Z",
                "available_at": "2024-01-03T01:00:00Z",
                "revision_id": "fx-r1",
                "rate": 1.1,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "observed_at": "2024-01-03T00:00:00Z",
                "available_at": "2024-01-05T01:00:00Z",
                "revision_id": "fx-r2",
                "rate": 1.2,
            },
        ]
    ).to_csv(path, index=False)
    return path


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


def test_cash_delisting_adds_point_in_time_terminal_payout(tmp_path: Path):
    source = dividend_market(tmp_path)
    adjusted = apply_point_in_time_corporate_actions(
        source,
        actions(tmp_path, delisting_rows()),
        as_of="2024-01-07T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
    )

    assert adjusted.close.iloc[:-1].equals(source.close)
    assert adjusted.close.index[-1] == pd.Timestamp("2024-01-06T00:00:00Z")
    assert adjusted.close.iloc[-1] == 45.0
    assert adjusted.volume.iloc[-1] == 0.0
    snapshot = adjusted.corporate_actions
    assert snapshot.applied_cash_delistings == 1
    assert snapshot.terminal_zero_recovery is False
    assert snapshot.applied_cash_dividends == 0
    assert snapshot.superseded_revisions == 1
    assert adjusted.metadata()["kind"] == "point_in_time_total_return_adjusted_market"
    assert snapshot.unadjusted_market_sha256 != snapshot.adjusted_market_sha256


def test_cash_delisting_can_bind_explicit_zero_recovery_without_fx(tmp_path: Path):
    rows = delisting_rows("EUR")
    rows[1]["cash_amount"] = 0.0
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, rows),
        as_of="2024-01-07T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
    )

    assert adjusted.close.iloc[-1] == 0.0
    assert adjusted.volume.iloc[-1] == 0.0
    snapshot = adjusted.corporate_actions
    assert snapshot.terminal_zero_recovery is True
    assert snapshot.fx_routes == 0
    assert snapshot.fx_source_name is None


def test_cash_delisting_rejects_negative_recovery(tmp_path: Path):
    rows = delisting_rows()
    rows[1]["cash_amount"] = -0.01

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, rows),
            as_of="2024-01-07T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
        )


def test_future_delisting_revision_cannot_rewrite_terminal_payout(tmp_path: Path):
    rows = delisting_rows()
    rows[1]["available_at"] = "2024-01-08T10:00:00Z"

    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, rows),
        as_of="2024-01-07T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
    )

    assert adjusted.close.iloc[-1] == 40.0
    assert adjusted.corporate_actions.excluded_unavailable_revisions == 1
    assert adjusted.corporate_actions.superseded_revisions == 0


def test_foreign_cash_delisting_uses_effective_time_fx(tmp_path: Path):
    rates = [
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-06T00:00:00Z",
            "available_at": "2024-01-06T01:00:00Z",
            "revision_id": "fx-r1",
            "rate": 1.1,
        }
    ]
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, delisting_rows("EUR")),
        as_of="2024-01-07T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
        fx_rates_path=fx_rates(tmp_path, rates),
        max_fx_age="24 hours",
    )

    assert adjusted.close.iloc[-1] == pytest.approx(45.0 * 1.1)
    assert adjusted.corporate_actions.fx_routes == 1
    assert adjusted.corporate_actions.fx_selected_observations == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{**delisting_rows()[0], "effective_at": "2024-01-05T00:00:00Z"}],
            "must end strictly before",
        ),
        (
            [
                delisting_rows()[0],
                {
                    **delisting_rows()[0],
                    "action_id": "delisting-2024-02",
                    "effective_at": "2024-01-07T00:00:00Z",
                },
            ],
            "multiple effective cash delistings",
        ),
        (
            [
                {**delisting_rows()[0], "action_id": "a-delisting"},
                {
                    **dividend_rows()[0],
                    "action_id": "z-dividend",
                    "effective_at": "2024-01-07T00:00:00Z",
                },
            ],
            "must be the terminal",
        ),
    ],
)
def test_cash_delisting_must_be_unique_terminal_and_post_market(tmp_path: Path, rows, message: str):
    with pytest.raises(ValueError, match=message):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, rows),
            as_of="2024-01-08T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
        )


def test_converts_foreign_dividend_with_latest_known_direct_fx_revision(tmp_path: Path):
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
        fx_rates_path=fx_rates(tmp_path),
        max_fx_age="24 hours",
    )

    factor = (104.0 - (3.0 * 1.2)) / 104.0
    assert adjusted.close.iloc[:2].tolist() == pytest.approx([100.0 * factor, 102.0 * factor])
    snapshot = adjusted.corporate_actions
    assert snapshot.fx_source_name == "fx-rates.csv"
    assert snapshot.fx_max_age_seconds == 86_400.0
    assert snapshot.fx_raw_rows == 2
    assert snapshot.fx_eligible_revisions == 2
    assert snapshot.fx_selected_observations == 1
    assert snapshot.fx_superseded_revisions == 1
    assert len(snapshot.fx_lineage_sha256) == 64


def test_future_fx_revision_cannot_enter_historical_snapshot(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "fx-r1",
            "rate": 1.1,
        },
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-07T01:00:00Z",
            "revision_id": "fx-r2",
            "rate": 1.9,
        },
    ]
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
        fx_rates_path=fx_rates(tmp_path, rows),
    )

    factor = (104.0 - (3.0 * 1.1)) / 104.0
    assert adjusted.close.iloc[:2].tolist() == pytest.approx([100.0 * factor, 102.0 * factor])
    assert adjusted.corporate_actions.fx_excluded_unavailable_revisions == 1
    assert adjusted.corporate_actions.fx_superseded_revisions == 0


def test_converts_foreign_dividend_through_declared_point_in_time_bridge(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "GBP",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "eur-gbp-r1",
            "rate": 0.8,
        },
        {
            "base_currency": "GBP",
            "quote_currency": "USD",
            "observed_at": "2024-01-02T23:30:00Z",
            "available_at": "2024-01-03T01:30:00Z",
            "revision_id": "gbp-usd-r1",
            "rate": 1.25,
        },
    ]
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
        fx_rates_path=fx_rates(tmp_path, rows),
        max_fx_age="24 hours",
        max_fx_leg_skew="30 minutes",
        fx_bridge_currencies=("GBP",),
    )

    factor = (104.0 - (3.0 * 0.8 * 1.25)) / 104.0
    assert adjusted.close.iloc[:2].tolist() == pytest.approx([100.0 * factor, 102.0 * factor])
    snapshot = adjusted.corporate_actions
    assert snapshot.fx_bridge_currencies == ("GBP",)
    assert snapshot.fx_routes == 1
    assert snapshot.fx_max_leg_skew_seconds == 1_800.0
    assert snapshot.fx_observed_max_leg_skew_seconds == 1_800.0
    assert snapshot.metadata()["fx_bridge_currencies"] == ["GBP"]


def test_bridge_route_rejects_fresh_but_asynchronous_legs(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "GBP",
            "observed_at": "2024-01-02T04:00:00Z",
            "available_at": "2024-01-02T05:00:00Z",
            "revision_id": "eur-gbp-r1",
            "rate": 0.8,
        },
        {
            "base_currency": "GBP",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "gbp-usd-r1",
            "rate": 1.25,
        },
    ]

    with pytest.raises(ValueError, match="maximum timestamp skew"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path, rows),
            max_fx_age="24 hours",
            max_fx_leg_skew="1 hour",
            fx_bridge_currencies=("GBP",),
        )


def test_fx_leg_skew_policy_changes_lineage_without_changing_prices(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "GBP",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "eur-gbp-r1",
            "rate": 0.8,
        },
        {
            "base_currency": "GBP",
            "quote_currency": "USD",
            "observed_at": "2024-01-02T23:30:00Z",
            "available_at": "2024-01-03T01:30:00Z",
            "revision_id": "gbp-usd-r1",
            "rate": 1.25,
        },
    ]
    common = {
        "as_of": "2024-01-06T00:00:00Z",
        "symbol": "ACME",
        "price_currency": "USD",
        "fx_rates_path": fx_rates(tmp_path, rows),
        "max_fx_age": "24 hours",
        "fx_bridge_currencies": ("GBP",),
    }

    strict = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        max_fx_leg_skew="30 minutes",
        **common,
    )
    relaxed = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        max_fx_leg_skew="2 hours",
        **common,
    )

    assert strict.close.equals(relaxed.close)
    assert strict.corporate_actions.selected_actions_sha256 == (
        relaxed.corporate_actions.selected_actions_sha256
    )
    assert strict.corporate_actions.lineage_sha256 != relaxed.corporate_actions.lineage_sha256


def test_direct_fx_route_takes_precedence_over_declared_bridge(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "direct-r1",
            "rate": 1.1,
        }
    ]
    adjusted = apply_point_in_time_corporate_actions(
        dividend_market(tmp_path),
        actions(tmp_path, foreign_dividend_rows()),
        as_of="2024-01-06T00:00:00Z",
        symbol="ACME",
        price_currency="USD",
        fx_rates_path=fx_rates(tmp_path, rows),
        fx_bridge_currencies=("GBP",),
    )

    factor = (104.0 - (3.0 * 1.1)) / 104.0
    assert adjusted.close.iloc[:2].tolist() == pytest.approx([100.0 * factor, 102.0 * factor])


def test_multiple_fresh_bridge_routes_fail_closed(tmp_path: Path):
    rows = []
    for base, quote, rate in [
        ("EUR", "GBP", 0.8),
        ("EUR", "JPY", 160.0),
        ("GBP", "USD", 1.25),
        ("JPY", "USD", 0.00625),
    ]:
        rows.append(
            {
                "base_currency": base,
                "quote_currency": quote,
                "observed_at": "2024-01-03T00:00:00Z",
                "available_at": "2024-01-03T01:00:00Z",
                "revision_id": f"{base}-{quote}-r1",
                "rate": rate,
            }
        )
    rows.sort(key=lambda row: (row["base_currency"], row["quote_currency"]))

    with pytest.raises(ValueError, match="ambiguous bridge routes"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path, rows),
            fx_bridge_currencies=("GBP", "JPY"),
        )


@pytest.mark.parametrize(
    ("rows", "bridges", "message"),
    [
        (
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "CAD",
                    "observed_at": "2024-01-03T00:00:00Z",
                    "available_at": "2024-01-03T01:00:00Z",
                    "revision_id": "undeclared-r1",
                    "rate": 1.5,
                }
            ],
            ("GBP",),
            "outside the declared FX routes",
        ),
        (
            [
                {
                    "base_currency": "EUR",
                    "quote_currency": "GBP",
                    "observed_at": "2024-01-01T00:00:00Z",
                    "available_at": "2024-01-01T01:00:00Z",
                    "revision_id": "stale-r1",
                    "rate": 0.8,
                },
                {
                    "base_currency": "GBP",
                    "quote_currency": "USD",
                    "observed_at": "2024-01-03T00:00:00Z",
                    "available_at": "2024-01-03T01:00:00Z",
                    "revision_id": "fresh-r1",
                    "rate": 1.25,
                },
            ],
            ("GBP",),
            "fresh point-in-time",
        ),
    ],
)
def test_bridge_route_policy_fails_closed(tmp_path: Path, rows, bridges, message: str):
    with pytest.raises(ValueError, match=message):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path, rows),
            max_fx_age="24 hours",
            fx_bridge_currencies=bridges,
        )


def test_unused_declared_fx_pair_fails_closed(tmp_path: Path):
    rows = [
        {
            "base_currency": "EUR",
            "quote_currency": "GBP",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "unused-r1",
            "rate": 0.8,
        },
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "direct-r1",
            "rate": 1.1,
        },
    ]

    with pytest.raises(ValueError, match="unused or incomplete"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path, rows),
            fx_bridge_currencies=("GBP",),
        )


@pytest.mark.parametrize(
    ("mutate", "max_age", "message"),
    [
        (
            lambda rows: [{**rows[0], "observed_at": "2024-01-03T00:00:01Z"}],
            "24 hours",
            "at or before",
        ),
        (
            lambda rows: [
                {
                    **rows[0],
                    "observed_at": "2024-01-01T00:00:00Z",
                    "available_at": "2024-01-01T01:00:00Z",
                }
            ],
            "24 hours",
            "maximum age",
        ),
        (
            lambda rows: [{**rows[0], "base_currency": "USD", "quote_currency": "EUR"}],
            "24 hours",
            "required direct currency pairs",
        ),
        (lambda rows: [{**rows[0], "rate": 0.0}], "24 hours", "finite and positive"),
    ],
)
def test_invalid_fx_contract_fails_closed(tmp_path: Path, mutate, max_age, message: str):
    base = [
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-03T00:00:00Z",
            "available_at": "2024-01-03T01:00:00Z",
            "revision_id": "fx-r1",
            "rate": 1.1,
        }
    ]
    with pytest.raises(ValueError, match=message):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path, mutate(base)),
            max_fx_age=max_age,
        )


def test_fx_input_is_rejected_when_no_foreign_dividend_is_applied(tmp_path: Path):
    with pytest.raises(ValueError, match="not allowed"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path),
        )


@pytest.mark.parametrize("value", [0, "-1 hour", "NaT"])
def test_invalid_fx_leg_skew_policy_fails_closed(tmp_path: Path, value):
    with pytest.raises(ValueError, match="max_fx_leg_skew must be a positive duration"):
        apply_point_in_time_corporate_actions(
            dividend_market(tmp_path),
            actions(tmp_path, foreign_dividend_rows()),
            as_of="2024-01-06T00:00:00Z",
            symbol="ACME",
            price_currency="USD",
            fx_rates_path=fx_rates(tmp_path),
            max_fx_leg_skew=value,
        )


@pytest.mark.parametrize(
    ("mutate", "currency", "message"),
    [
        (lambda rows: rows, None, "declared price_currency"),
        (lambda rows: rows, "TRY", "require an FX-rate CSV"),
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
