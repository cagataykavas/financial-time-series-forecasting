import json
import math
from pathlib import Path

import pandas as pd
import pytest

from finforecast.corporate_actions import apply_point_in_time_corporate_actions
from finforecast.data import load_market_csv
from finforecast.rate_curves import year_fraction
from finforecast.settlements import load_point_in_time_settlement

AS_OF = "2024-01-07T00:00:00Z"
MARKET_END = "2024-01-05T00:00:00Z"
CLOSEOUT = "2024-01-06T00:00:00Z"


def settlement_rows() -> list[dict[str, object]]:
    return [
        {
            "event_id": "cash-a",
            "symbol": "ACME",
            "event_type": "cash_flow",
            "effective_at": "2024-01-05T06:00:00Z",
            "available_at": "2024-01-05T07:00:00Z",
            "revision_id": "r1",
            "cash_amount": 10.0,
            "currency": "USD",
        },
        {
            "event_id": "cash-a",
            "symbol": "ACME",
            "event_type": "cash_flow",
            "effective_at": "2024-01-05T06:00:00Z",
            "available_at": "2024-01-08T00:00:00Z",
            "revision_id": "r2",
            "cash_amount": 12.0,
            "currency": "USD",
        },
        {
            "event_id": "cash-b",
            "symbol": "ACME",
            "event_type": "cash_flow",
            "effective_at": "2024-01-05T18:00:00Z",
            "available_at": "2024-01-05T19:00:00Z",
            "revision_id": "r1",
            "cash_amount": 15.0,
            "currency": "USD",
        },
        {
            "event_id": "closeout",
            "symbol": "ACME",
            "event_type": "closeout",
            "effective_at": CLOSEOUT,
            "available_at": "2024-01-06T01:00:00Z",
            "revision_id": "r1",
            "cash_amount": 0.0,
            "currency": "",
        },
    ]


def write_settlement(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "settlement.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_fx(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "settlement-fx.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_curve(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "settlement-rates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def rate_curve_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for observed_at, available_at, revision_id, rates in [
        ("2024-01-05T05:00:00Z", "2024-01-05T05:30:00Z", "morning-r1", (0.20, 0.365)),
        ("2024-01-05T05:00:00Z", "2024-01-08T00:00:00Z", "morning-r2", (9.0, 9.0)),
        ("2024-01-05T17:00:00Z", "2024-01-05T17:30:00Z", "evening-r1", (0.73, 0.90)),
    ]:
        for tenor_days, zero_rate in zip((0.25, 0.75), rates, strict=True):
            rows.append(
                {
                    "curve_id": "USD-OIS",
                    "currency": "USD",
                    "calendar_id": "NONE",
                    "day_count": "ACT/365F",
                    "observed_at": observed_at,
                    "available_at": available_at,
                    "revision_id": revision_id,
                    "tenor_days": tenor_days,
                    "zero_rate": zero_rate,
                }
            )
    return rows


def settlement_fx_rows() -> list[dict[str, object]]:
    return [
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-05T05:30:00Z",
            "available_at": "2024-01-05T05:45:00Z",
            "revision_id": "morning-r1",
            "rate": 1.1,
        },
        {
            "base_currency": "EUR",
            "quote_currency": "USD",
            "observed_at": "2024-01-05T17:30:00Z",
            "available_at": "2024-01-05T17:45:00Z",
            "revision_id": "evening-r1",
            "rate": 1.2,
        },
    ]


def market(tmp_path: Path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC"),
            "close": [100.0, 102.0, 104.0, 101.0, 102.0],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    ).to_csv(path, index=False)
    return load_market_csv(path)


def delisting(tmp_path: Path, cash_amount: float = 0.0) -> Path:
    path = tmp_path / "actions.csv"
    pd.DataFrame(
        [
            {
                "action_id": "delisting-1",
                "symbol": "ACME",
                "action_type": "cash_delisting",
                "effective_at": CLOSEOUT,
                "available_at": "2024-01-05T01:00:00Z",
                "revision_id": "r1",
                "cash_amount": cash_amount,
                "currency": "USD",
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_selects_only_known_revisions_and_binds_closeout(tmp_path: Path) -> None:
    snapshot = load_point_in_time_settlement(
        write_settlement(tmp_path, settlement_rows()),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        market_end=MARKET_END,
        expected_closeout_at=CLOSEOUT,
    )

    assert snapshot.total_cash_amount == 25.0
    assert snapshot.applied_cash_flows == 2
    assert snapshot.excluded_unavailable_revisions == 1
    assert snapshot.superseded_revisions == 0
    assert len(snapshot.selected_events_sha256) == len(snapshot.lineage_sha256) == 64


def test_settlement_drives_terminal_total_recovery(tmp_path: Path) -> None:
    adjusted = apply_point_in_time_corporate_actions(
        market(tmp_path),
        delisting(tmp_path),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        settlement_ledger_path=write_settlement(tmp_path, settlement_rows()),
    )

    assert adjusted.close.iloc[-1] == 25.0
    assert adjusted.close.index[-1] == pd.Timestamp(CLOSEOUT)
    assert adjusted.volume.iloc[-1] == 0.0
    snapshot = adjusted.corporate_actions
    assert snapshot.terminal_zero_recovery is False
    assert snapshot.settlement is not None
    assert snapshot.settlement.total_cash_amount == 25.0
    assert snapshot.metadata()["settlement"]["applied_cash_flows"] == 2


def test_converts_each_foreign_cash_flow_at_its_effective_timestamp(tmp_path: Path) -> None:
    rows = settlement_rows()
    rows[0]["currency"] = "EUR"
    rows[1]["currency"] = "EUR"
    rows[2]["currency"] = "EUR"

    adjusted = apply_point_in_time_corporate_actions(
        market(tmp_path),
        delisting(tmp_path),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        settlement_ledger_path=write_settlement(tmp_path, rows),
        settlement_fx_rates_path=write_fx(tmp_path, settlement_fx_rows()),
        settlement_max_fx_age="2 hours",
    )

    settlement = adjusted.corporate_actions.settlement
    assert settlement is not None
    assert settlement.total_cash_amount == pytest.approx(29.0)
    assert settlement.fx_routes == 2
    assert settlement.fx_selected_observations == 2
    assert settlement.fx_source_name == "settlement-fx.csv"
    assert settlement.fx_lineage_sha256 is not None
    assert adjusted.close.iloc[-1] == pytest.approx(29.0)


def test_carries_each_cash_flow_to_closeout_with_point_in_time_zero_curve(
    tmp_path: Path,
) -> None:
    settlement = settlement_rows()
    settlement[0]["effective_at"] = "2024-01-05T12:00:00Z"
    snapshot = load_point_in_time_settlement(
        write_settlement(tmp_path, settlement),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        market_end=MARKET_END,
        expected_closeout_at=CLOSEOUT,
        rate_curve_path=write_curve(tmp_path, rate_curve_rows()),
        max_rate_age="8 hours",
    )

    interpolated_rate = 0.20 + 0.5 * (0.365 - 0.20)
    expected = 10.0 * math.exp(interpolated_rate * 0.5 / 365.0) + 15.0 * math.exp(
        0.73 * 0.25 / 365.0
    )
    assert snapshot.nominal_cash_amount == 25.0
    assert snapshot.total_cash_amount == pytest.approx(expected)
    assert snapshot.rate_curve_id == "USD-OIS"
    assert snapshot.rate_calendar_id == "NONE"
    assert snapshot.rate_day_count == "ACT/365F"
    assert snapshot.rate_selected_curves == 2
    assert snapshot.rate_excluded_unavailable_rows == 2
    assert snapshot.rate_superseded_revisions == 0
    assert snapshot.rate_lineage_sha256 is not None


def test_rate_curve_interpolates_but_never_extrapolates(tmp_path: Path) -> None:
    rows = rate_curve_rows()
    for row in rows:
        row["tenor_days"] = float(row["tenor_days"]) + 0.25

    with pytest.raises(ValueError, match="forbidden extrapolation"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, rows),
        )


def test_rate_curve_enforces_currency_freshness_and_revision_shape(tmp_path: Path) -> None:
    wrong_currency = rate_curve_rows()
    wrong_currency[0]["currency"] = "EUR"
    with pytest.raises(ValueError, match="price currency"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, wrong_currency),
        )

    with pytest.raises(ValueError, match="maximum age"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, rate_curve_rows()),
            max_rate_age="30 minutes",
        )

    changed_shape = rate_curve_rows()
    changed_shape[3]["tenor_days"] = 0.5
    with pytest.raises(ValueError, match="change tenor shape"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, changed_shape),
        )


def test_rate_curve_day_counts_match_high_precision_golden_vectors() -> None:
    fixture = Path(__file__).parent / "fixtures" / "settlement_day_count_vectors.json"
    vectors = json.loads(fixture.read_text(encoding="utf-8"))

    for vector in vectors:
        start = pd.Timestamp(vector["start"])
        end = pd.Timestamp(vector["end"])
        fraction = year_fraction(start, end, vector["day_count"])
        factor = math.exp(float(vector["zero_rate"]) * fraction)
        assert fraction == pytest.approx(float(vector["year_fraction"]), abs=1e-15)
        assert factor == pytest.approx(float(vector["accumulation_factor"]), abs=1e-15)


def test_settlement_uses_curve_declared_act_360(tmp_path: Path) -> None:
    rows = rate_curve_rows()
    for row in rows:
        row["day_count"] = "ACT/360"
    settlements = settlement_rows()
    settlements[0]["effective_at"] = "2024-01-05T12:00:00Z"

    snapshot = load_point_in_time_settlement(
        write_settlement(tmp_path, settlements),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        market_end=MARKET_END,
        expected_closeout_at=CLOSEOUT,
        rate_curve_path=write_curve(tmp_path, rows),
        max_rate_age="8 hours",
    )

    interpolated_rate = 0.20 + 0.5 * (0.365 - 0.20)
    expected = 10.0 * math.exp(interpolated_rate * 0.5 / 360.0) + 15.0 * math.exp(
        0.73 * 0.25 / 360.0
    )
    assert snapshot.total_cash_amount == pytest.approx(expected)
    assert snapshot.rate_day_count == "ACT/360"


def test_rate_curve_rejects_ungoverned_calendar_and_mixed_day_counts(tmp_path: Path) -> None:
    calendar_adjusted = rate_curve_rows()
    for row in calendar_adjusted:
        row["calendar_id"] = "NYC"
    with pytest.raises(ValueError, match="calendar_id must be NONE"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, calendar_adjusted),
        )

    mixed_day_count = rate_curve_rows()
    mixed_day_count[0]["day_count"] = "ACT/360"
    with pytest.raises(ValueError, match="exactly one day_count"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, mixed_day_count),
        )

    unsupported = rate_curve_rows()
    for row in unsupported:
        row["day_count"] = "30E/360"
    with pytest.raises(ValueError, match="ACT/360 or ACT/365F"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, settlement_rows()),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            rate_curve_path=write_curve(tmp_path, unsupported),
        )


def test_does_not_reuse_a_later_quote_for_an_earlier_cash_flow(tmp_path: Path) -> None:
    rows = settlement_rows()
    rows[0]["currency"] = "EUR"
    rows[1]["currency"] = "EUR"
    rows[2]["currency"] = "EUR"

    with pytest.raises(ValueError, match="at or before its governed reference timestamp"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, rows),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
            fx_rates_path=write_fx(tmp_path, settlement_fx_rows()[1:]),
        )


def test_settlement_fx_uses_only_revisions_known_at_cutoff(tmp_path: Path) -> None:
    rows = settlement_rows()
    rows[0]["currency"] = "EUR"
    rows[1]["currency"] = "EUR"
    rows[2]["currency"] = "EUR"
    rates = settlement_fx_rows()
    rates.insert(
        1,
        {
            **rates[0],
            "available_at": "2024-01-08T00:00:00Z",
            "revision_id": "morning-r2",
            "rate": 9.9,
        },
    )

    snapshot = load_point_in_time_settlement(
        write_settlement(tmp_path, rows),
        as_of=AS_OF,
        symbol="ACME",
        price_currency="USD",
        market_end=MARKET_END,
        expected_closeout_at=CLOSEOUT,
        fx_rates_path=write_fx(tmp_path, rates),
    )

    assert snapshot.total_cash_amount == pytest.approx(29.0)
    assert snapshot.fx_excluded_unavailable_revisions == 1


def test_foreign_settlement_requires_separate_fx_feed(tmp_path: Path) -> None:
    rows = settlement_rows()
    rows[0]["currency"] = "EUR"

    with pytest.raises(ValueError, match="require a settlement FX-rate CSV"):
        load_point_in_time_settlement(
            write_settlement(tmp_path, rows),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
        )


def test_settlement_cannot_double_count_direct_delisting_payout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must declare zero cash"):
        apply_point_in_time_corporate_actions(
            market(tmp_path),
            delisting(tmp_path, cash_amount=5.0),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            settlement_ledger_path=write_settlement(tmp_path, settlement_rows()),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "exactly one effective closeout"),
        (
            lambda rows: rows[0].update({"effective_at": "2024-01-04T12:00:00Z"}),
            "after market data",
        ),
        (
            lambda rows: rows[0].update({"currency": "eur"}),
            "uppercase ISO 4217",
        ),
        (
            lambda rows: rows[-1].update({"effective_at": "2024-01-06T12:00:00Z"}),
            "match the cash-delisting",
        ),
    ],
)
def test_settlement_fails_closed_on_ambiguous_boundaries(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    rows = settlement_rows()
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        load_point_in_time_settlement(
            write_settlement(tmp_path, rows),
            as_of=AS_OF,
            symbol="ACME",
            price_currency="USD",
            market_end=MARKET_END,
            expected_closeout_at=CLOSEOUT,
        )
