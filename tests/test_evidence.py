from __future__ import annotations

import json

from finforecast.evidence import build_evidence, write_evidence


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
