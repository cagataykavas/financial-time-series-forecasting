from __future__ import annotations

import json

import pytest

from finforecast.domain import ModelStatus
from finforecast.registry import PromotionPolicy, write_model_manifest


def folds(candidate_mae: float, baseline_mae: float):
    return (
        {
            "models": {
                "candidate": {"mae": candidate_mae},
                "baseline": {"mae": baseline_mae},
            }
        },
        {
            "models": {
                "candidate": {"mae": candidate_mae},
                "baseline": {"mae": baseline_mae},
            }
        },
    )


def test_candidate_must_pass_every_promotion_gate():
    aggregate = {
        "candidate": {"mae": 0.8, "mean_business_cost": 1.2},
        "baseline": {"mae": 1.0, "mean_business_cost": 1.0},
    }
    decision = PromotionPolicy(0.1, 1.0, 0.5).decide(
        "candidate", "baseline", aggregate, folds(0.8, 1.0)
    )
    assert decision.status is ModelStatus.REJECTED
    assert decision.reasons == ("business_cost_regression",)


def test_candidate_is_promoted_when_all_gates_pass():
    aggregate = {
        "candidate": {"mae": 0.8, "mean_business_cost": 0.8},
        "baseline": {"mae": 1.0, "mean_business_cost": 1.0},
    }
    decision = PromotionPolicy(0.1, 1.0, 1.0).decide(
        "candidate", "baseline", aggregate, folds(0.8, 1.0)
    )
    assert decision.promoted


def test_manifest_hash_is_stable_and_payload_sensitive(tmp_path):
    first = write_model_manifest(tmp_path / "a.json", {"model": "x", "metric": 1})
    second = write_model_manifest(tmp_path / "b.json", {"metric": 1, "model": "x"})
    changed = write_model_manifest(tmp_path / "c.json", {"model": "x", "metric": 2})
    assert first["sha256"] == second["sha256"]
    assert changed["sha256"] != first["sha256"]
    assert json.loads((tmp_path / "a.json").read_text())["schema_version"] == 1


def test_promotion_handles_zero_baseline_without_non_finite_metrics():
    aggregate = {
        "candidate": {"mae": 1.0, "mean_business_cost": 1.0},
        "baseline": {"mae": 0.0, "mean_business_cost": 0.0},
    }
    decision = PromotionPolicy().decide("candidate", "baseline", aggregate, folds(1.0, 0.0))
    assert not decision.promoted
    assert decision.metrics["relative_mae_improvement"] == -1.0


def test_promotion_rejects_missing_fold_evidence():
    aggregate = {
        "candidate": {"mae": 0.8},
        "baseline": {"mae": 1.0},
    }
    with pytest.raises(ValueError, match="at least one fold"):
        PromotionPolicy().decide("candidate", "baseline", aggregate, ())
