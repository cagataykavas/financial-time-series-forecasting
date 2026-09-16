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
