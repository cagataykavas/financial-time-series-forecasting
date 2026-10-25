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


def dividend_actions_csv(tmp_path, market_path):
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
                "currency": "USD",
            }
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
    assert evidence["schema_version"] == 5
    assert evidence["dataset"]["kind"] == "point_in_time_total_return_adjusted_market"
    assert snapshot["applied_cash_dividends"] == 1
    assert snapshot["price_currency"] == "USD"
    assert any("cash-dividend adjustments" in claim for claim in evidence["claims"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--corporate-actions-csv", "actions.csv"],
        ["--symbol", "ACME"],
        ["--corporate-actions-csv", "actions.csv", "--symbol", "ACME"],
        ["--price-currency", "USD"],
    ],
)
def test_cli_requires_complete_point_in_time_action_configuration(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
