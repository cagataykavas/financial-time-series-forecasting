from __future__ import annotations

import json

import pandas as pd
import pytest

from finforecast.cli import main
from finforecast.synthetic import synthetic_market


def market_csv(tmp_path, rows: int = 340):
    close, volume = synthetic_market(rows=rows, seed=11)
    path = tmp_path / "market.csv"
    pd.DataFrame(
        {"timestamp": close.index, "close": close.to_numpy(), "volume": volume.to_numpy()}
    ).to_csv(path, index=False)
    return path


def vintage_market_csv(tmp_path, rows: int = 340):
    close, volume = synthetic_market(rows=rows, seed=11)
    timestamps = close.index.tz_localize("UTC")
    path = tmp_path / "market-vintages.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_at": timestamps + pd.Timedelta(hours=1),
            "revision_id": "vendor-r1",
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
        }
    ).to_csv(path, index=False)
    return path, (timestamps[-1] + pd.Timedelta(hours=2)).isoformat()


def split_actions_csv(tmp_path, market_path):
    timestamps = pd.to_datetime(pd.read_csv(market_path)["timestamp"], utc=True)
    path = tmp_path / "split-vintages.csv"
    pd.DataFrame(
        [
            {
                "action_id": "split-1",
                "symbol": "ACME",
                "action_type": "split",
                "effective_at": timestamps.iloc[220],
                "available_at": timestamps.iloc[180],
                "revision_id": "vendor-r1",
                "split_ratio": 2.0,
            }
        ]
    ).to_csv(path, index=False)
    return path


def dividend_actions_csv(tmp_path, market_path, currency="USD"):
    frame = pd.read_csv(market_path)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    reference_close = float(frame.loc[219, "close"])
    path = tmp_path / "dividend-vintages.csv"
    pd.DataFrame(
        [
            {
                "action_id": "dividend-1",
                "symbol": "ACME",
                "action_type": "cash_dividend",
                "effective_at": timestamps.iloc[220],
                "available_at": timestamps.iloc[180],
                "revision_id": "vendor-r1",
                "cash_amount": reference_close * 0.01,
                "currency": currency,
            }
        ]
    ).to_csv(path, index=False)
    return path


def delisting_actions_csv(tmp_path, market_path, currency="USD", recovery_ratio=0.75):
    frame = pd.read_csv(market_path)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    path = tmp_path / "delisting-vintages.csv"
    pd.DataFrame(
        [
            {
                "action_id": "delisting-1",
                "symbol": "ACME",
                "action_type": "cash_delisting",
                "effective_at": timestamps.iloc[-1] + pd.Timedelta(hours=1),
                "available_at": timestamps.iloc[-2],
                "revision_id": "vendor-r1",
                "cash_amount": float(frame.loc[len(frame) - 1, "close"]) * recovery_ratio,
                "currency": currency,
            }
        ]
    ).to_csv(path, index=False)
    return path


def settlement_ledger_csv(tmp_path, market_path):
    timestamps = pd.to_datetime(pd.read_csv(market_path)["timestamp"], utc=True)
    path = tmp_path / "settlement-vintages.csv"
    closeout = timestamps.iloc[-1] + pd.Timedelta(hours=1)
    pd.DataFrame(
        [
            {
                "event_id": "cash-a",
                "symbol": "ACME",
                "event_type": "cash_flow",
                "effective_at": timestamps.iloc[-1] + pd.Timedelta(minutes=15),
                "available_at": timestamps.iloc[-1] + pd.Timedelta(minutes=20),
                "revision_id": "vendor-r1",
                "cash_amount": 20.0,
                "currency": "USD",
            },
            {
                "event_id": "cash-b",
                "symbol": "ACME",
                "event_type": "cash_flow",
                "effective_at": timestamps.iloc[-1] + pd.Timedelta(minutes=30),
                "available_at": timestamps.iloc[-1] + pd.Timedelta(minutes=35),
                "revision_id": "vendor-r1",
                "cash_amount": 10.0,
                "currency": "USD",
            },
            {
                "event_id": "closeout",
                "symbol": "ACME",
                "event_type": "closeout",
                "effective_at": closeout,
                "available_at": closeout,
                "revision_id": "vendor-r1",
                "cash_amount": 0.0,
                "currency": "",
            },
        ]
    ).to_csv(path, index=False)
    return path


def foreign_settlement_fx_csv(tmp_path, settlement_path):
    settlements = pd.read_csv(settlement_path)
    cash_flows = settlements.loc[settlements["event_type"] == "cash_flow"].copy()
    cash_flows["currency"] = "EUR"
    settlements.loc[cash_flows.index, "currency"] = "EUR"
    settlements.to_csv(settlement_path, index=False)
    path = tmp_path / "settlement-fx-vintages.csv"
    pd.DataFrame(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "observed_at": pd.Timestamp(row.effective_at) - pd.Timedelta(minutes=5),
                "available_at": pd.Timestamp(row.effective_at) - pd.Timedelta(minutes=1),
                "revision_id": f"settlement-fx-{index}",
                "rate": rate,
            }
            for index, (row, rate) in enumerate(
                zip(cash_flows.itertuples(index=False), (1.1, 1.2), strict=True), start=1
            )
        ]
    ).to_csv(path, index=False)
    return path


def settlement_rate_curve_csv(tmp_path, settlement_path):
    settlements = pd.read_csv(settlement_path)
    cash_flows = settlements.loc[settlements["event_type"] == "cash_flow"]
    path = tmp_path / "settlement-rate-curves.csv"
    rows = []
    for index, row in enumerate(cash_flows.itertuples(index=False), start=1):
        observed_at = pd.Timestamp(row.effective_at) - pd.Timedelta(minutes=5)
        available_at = pd.Timestamp(row.effective_at) - pd.Timedelta(minutes=1)
        for tenor_days in (0.01, 0.05):
            rows.append(
                {
                    "curve_id": "USD-OIS",
                    "currency": "USD",
                    "observed_at": observed_at,
                    "available_at": available_at,
                    "revision_id": f"curve-{index}",
                    "tenor_days": tenor_days,
                    "zero_rate": 0.10,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def fx_rates_csv(tmp_path, market_path):
    timestamps = pd.to_datetime(pd.read_csv(market_path)["timestamp"], utc=True)
    path = tmp_path / "fx-vintages.csv"
    pd.DataFrame(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "observed_at": timestamps.iloc[219],
                "available_at": timestamps.iloc[219] + pd.Timedelta(minutes=1),
                "revision_id": "fx-r1",
                "rate": 1.1,
            }
        ]
    ).to_csv(path, index=False)
    return path


def bridged_fx_rates_csv(tmp_path, market_path):
    timestamps = pd.to_datetime(pd.read_csv(market_path)["timestamp"], utc=True)
    path = tmp_path / "bridged-fx-vintages.csv"
    pd.DataFrame(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "observed_at": timestamps.iloc[219],
                "available_at": timestamps.iloc[219] + pd.Timedelta(minutes=1),
                "revision_id": "eur-gbp-r1",
                "rate": 0.8,
            },
            {
                "base_currency": "GBP",
                "quote_currency": "USD",
                "observed_at": timestamps.iloc[219] - pd.Timedelta(minutes=15),
                "available_at": timestamps.iloc[219] + pd.Timedelta(minutes=1),
                "revision_id": "gbp-usd-r1",
                "rate": 1.25,
            },
        ]
    ).to_csv(path, index=False)
    return path


def test_cli_runs_standard_evaluation_from_csv(tmp_path):
    output = tmp_path / "standard"
    assert main(["--csv", str(market_csv(tmp_path)), "--output", str(output)]) == 0

    result = json.loads((output / "forecast_results.json").read_text())
    assert result["dataset"]["kind"] == "external_market_csv"
    assert result["dataset"]["source_name"] == "market.csv"
    assert result["observations"] > 0
    assert (output / "forecast_report.html").is_file()


def test_cli_writes_governed_evidence_from_csv(tmp_path):
    output = tmp_path / "evidence"
    assert (
        main(
            [
                "--csv",
                str(market_csv(tmp_path)),
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert evidence["dataset"]["kind"] == "external_market_csv"
    assert manifest["payload"]["dataset"]["sha256"] == evidence["dataset"]["sha256"]


def test_cli_rejects_undersized_csv(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--csv", str(market_csv(tmp_path, rows=100))])
    assert error.value.code == 2


def test_cli_writes_point_in_time_selection_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    output = tmp_path / "point-in-time-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    snapshot = evidence["dataset"]["snapshot"]
    assert evidence["dataset"]["kind"] == "point_in_time_market_csv"
    assert snapshot["selected_observations"] == 340
    assert snapshot["eligible_rows"] == snapshot["raw_rows"] == 340
    assert len(snapshot["lineage_sha256"]) == 64
    assert manifest["payload"]["dataset"]["snapshot"] == snapshot
    assert any("as-of availability cutoff" in claim for claim in evidence["claims"])


def test_cli_binds_point_in_time_split_snapshot_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = split_actions_csv(tmp_path, source)
    output = tmp_path / "split-adjusted-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    action_snapshot = evidence["dataset"]["corporate_actions"]
    assert evidence["dataset"]["kind"] == "point_in_time_split_adjusted_market"
    assert action_snapshot["applied_splits"] == 1
    assert action_snapshot["adjusted_market_sha256"] == evidence["dataset"]["sha256"]
    assert manifest["payload"]["dataset"]["corporate_actions"] == action_snapshot
    assert any("corporate-action revisions" in claim for claim in evidence["claims"])


def test_cli_binds_cash_dividend_total_return_snapshot_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = dividend_actions_csv(tmp_path, source)
    output = tmp_path / "dividend-adjusted-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    snapshot = evidence["dataset"]["corporate_actions"]
    assert evidence["schema_version"] == 13
    assert evidence["dataset"]["kind"] == "point_in_time_total_return_adjusted_market"
    assert snapshot["applied_cash_dividends"] == 1
    assert snapshot["price_currency"] == "USD"
    assert any("cash-dividend adjustments" in claim for claim in evidence["claims"])


def test_cli_binds_cash_delisting_terminal_payout_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = delisting_actions_csv(tmp_path, source)
    output = tmp_path / "delisting-adjusted-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    snapshot = evidence["dataset"]["corporate_actions"]
    assert evidence["schema_version"] == 13
    assert evidence["dataset"]["kind"] == "point_in_time_total_return_adjusted_market"
    assert evidence["dataset"]["raw_observations"] == 340
    assert evidence["dataset"]["adjusted_observations"] == 341
    assert snapshot["applied_cash_delistings"] == 1
    assert snapshot["applied_cash_dividends"] == 0
    assert any("Cash-delisting terminal observations" in claim for claim in evidence["claims"])


def test_cli_binds_zero_recovery_without_accepting_ordinary_zero_prices(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = delisting_actions_csv(tmp_path, source, currency="EUR", recovery_ratio=0.0)
    output = tmp_path / "zero-recovery-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    snapshot = evidence["dataset"]["corporate_actions"]
    assert evidence["schema_version"] == 13
    assert snapshot["terminal_zero_recovery"] is True
    assert snapshot["fx_source_name"] is None
    assert any("zero terminal close" in claim for claim in evidence["claims"])


def test_cli_binds_revision_aware_settlement_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = delisting_actions_csv(tmp_path, source, recovery_ratio=0.0)
    settlements = settlement_ledger_csv(tmp_path, source)
    output = tmp_path / "settlement-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--settlement-ledger-csv",
                str(settlements),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    settlement = evidence["dataset"]["corporate_actions"]["settlement"]
    assert evidence["schema_version"] == 13
    assert settlement["total_cash_amount"] == 30.0
    assert settlement["applied_cash_flows"] == 2
    assert len(settlement["lineage_sha256"]) == 64
    assert any("settlement cash-flow revisions" in claim for claim in evidence["claims"])


def test_cli_binds_per_cash_flow_settlement_fx_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = delisting_actions_csv(tmp_path, source, recovery_ratio=0.0)
    settlements = settlement_ledger_csv(tmp_path, source)
    rates = foreign_settlement_fx_csv(tmp_path, settlements)
    output = tmp_path / "foreign-settlement-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--settlement-ledger-csv",
                str(settlements),
                "--settlement-fx-rates-csv",
                str(rates),
                "--settlement-max-fx-age-hours",
                "1",
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    settlement = evidence["dataset"]["corporate_actions"]["settlement"]
    assert settlement["total_cash_amount"] == 34.0
    assert settlement["fx_routes"] == 2
    assert len(settlement["fx_lineage_sha256"]) == 64
    assert any("each foreign-currency settlement" in claim for claim in evidence["claims"])


def test_cli_binds_revision_aware_settlement_curve_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = delisting_actions_csv(tmp_path, source, recovery_ratio=0.0)
    settlements = settlement_ledger_csv(tmp_path, source)
    curves = settlement_rate_curve_csv(tmp_path, settlements)
    output = tmp_path / "rate-adjusted-settlement-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--settlement-ledger-csv",
                str(settlements),
                "--settlement-rate-curve-csv",
                str(curves),
                "--settlement-max-rate-age-hours",
                "1",
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    settlement = evidence["dataset"]["corporate_actions"]["settlement"]
    assert evidence["schema_version"] == 13
    assert settlement["nominal_cash_amount"] == 30.0
    assert settlement["total_cash_amount"] > 30.0
    assert settlement["rate_curve_id"] == "USD-OIS"
    assert settlement["rate_selected_curves"] == 2
    assert len(settlement["rate_lineage_sha256"]) == 64
    assert any("revision-aware price-currency zero curve" in claim for claim in evidence["claims"])


def test_cli_binds_point_in_time_fx_conversion_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = dividend_actions_csv(tmp_path, source, currency="EUR")
    rates = fx_rates_csv(tmp_path, source)
    output = tmp_path / "fx-adjusted-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--fx-rates-csv",
                str(rates),
                "--max-fx-age-hours",
                "24",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = json.loads((output / "forecast_evidence.json").read_text())
    snapshot = evidence["dataset"]["corporate_actions"]
    assert snapshot["fx_source_name"] == "fx-vintages.csv"
    assert snapshot["fx_max_age_seconds"] == 86_400.0
    assert snapshot["fx_selected_observations"] == 1
    assert len(snapshot["fx_lineage_sha256"]) == 64
    assert any("FX revisions known" in claim for claim in evidence["claims"])


def test_cli_binds_declared_fx_bridge_route_to_evidence(tmp_path):
    source, as_of = vintage_market_csv(tmp_path)
    actions = dividend_actions_csv(tmp_path, source, currency="EUR")
    rates = bridged_fx_rates_csv(tmp_path, source)
    output = tmp_path / "bridged-fx-evidence"

    assert (
        main(
            [
                "--csv",
                str(source),
                "--as-of",
                as_of,
                "--corporate-actions-csv",
                str(actions),
                "--symbol",
                "ACME",
                "--price-currency",
                "USD",
                "--fx-rates-csv",
                str(rates),
                "--fx-bridge-currency",
                "GBP",
                "--max-fx-age-hours",
                "24",
                "--max-fx-leg-skew-hours",
                "0.5",
                "--evidence",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    snapshot = json.loads((output / "forecast_evidence.json").read_text())["dataset"][
        "corporate_actions"
    ]
    assert snapshot["fx_bridge_currencies"] == ["GBP"]
    assert snapshot["fx_routes"] == 1
    assert snapshot["fx_max_leg_skew_seconds"] == 1_800.0
    assert snapshot["fx_observed_max_leg_skew_seconds"] == 900.0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--corporate-actions-csv", "actions.csv"],
        ["--symbol", "ACME"],
        ["--corporate-actions-csv", "actions.csv", "--symbol", "ACME"],
        ["--price-currency", "USD"],
        ["--fx-rates-csv", "fx.csv"],
        ["--fx-bridge-currency", "GBP"],
        ["--settlement-ledger-csv", "settlement.csv"],
        ["--settlement-rate-curve-csv", "rates.csv"],
    ],
)
def test_cli_requires_complete_point_in_time_action_configuration(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
