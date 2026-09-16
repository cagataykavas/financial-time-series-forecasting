from __future__ import annotations

import json

import pandas as pd

from finforecast.data import load_market_csv
from finforecast.evidence import build_evidence, build_market_evidence, write_evidence
from finforecast.synthetic import synthetic_market


def test_evidence_is_deterministic_and_explicit_about_claim_scope():
    first = build_evidence(rows=520, seed=9)
    second = build_evidence(rows=520, seed=9)
    assert first == second
    assert first["validation"]["fold_count"] > 1
    assert first["uncertainty"]["evaluation_size"] > 0
    assert any("not evidence of market profitability" in claim for claim in first["claims"])


def test_evidence_writes_hashed_candidate_manifest(tmp_path):
    payload = write_evidence(tmp_path, rows=520, seed=3)
    evidence = json.loads((tmp_path / "forecast_evidence.json").read_text())
    manifest = json.loads((tmp_path / "candidate_manifest.json").read_text())
    assert evidence == payload
    assert len(manifest["sha256"]) == 64


def test_market_evidence_carries_source_identity_and_honest_claim(tmp_path):
    close, volume = synthetic_market(rows=340, seed=5)
    source = tmp_path / "prices.csv"
    pd.DataFrame(
        {"timestamp": close.index, "close": close.to_numpy(), "volume": volume.to_numpy()}
    ).to_csv(source, index=False)

    payload = build_market_evidence(load_market_csv(source), seed=5)
    assert payload["dataset"]["kind"] == "external_market_csv"
    assert payload["dataset"]["source_name"] == "prices.csv"
    assert len(payload["dataset"]["sha256"]) == 64
    assert any("future market profitability" in claim for claim in payload["claims"])
