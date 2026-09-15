from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import ModelStatus, PromotionDecision


@dataclass(frozen=True)
class PromotionPolicy:
    min_relative_mae_improvement: float = 0.0
    max_business_cost_ratio: float = 1.0
    min_fold_win_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.min_relative_mae_improvement < 0:
            raise ValueError("minimum improvement cannot be negative")
        if self.max_business_cost_ratio <= 0:
            raise ValueError("business-cost ratio must be positive")
        if not 0.0 <= self.min_fold_win_rate <= 1.0:
            raise ValueError("fold win rate must be between zero and one")

    def decide(
        self,
        candidate: str,
        baseline: str,
        aggregate: dict[str, dict[str, float]],
        folds: tuple[dict[str, object], ...],
    ) -> PromotionDecision:
        if candidate == baseline:
            raise ValueError("candidate and baseline must be different models")
        missing = {candidate, baseline}.difference(aggregate)
        if missing:
            raise ValueError(f"aggregate metrics are missing models: {sorted(missing)}")
        if not folds:
            raise ValueError("at least one fold is required for a promotion decision")
        candidate_metrics = aggregate[candidate]
        baseline_metrics = aggregate[baseline]
        candidate_mae = candidate_metrics["mae"]
        baseline_mae = baseline_metrics["mae"]
        if not all(math.isfinite(value) and value >= 0 for value in (candidate_mae, baseline_mae)):
            raise ValueError("aggregate MAE values must be finite and non-negative")
        if baseline_mae == 0:
            relative_improvement = 0.0 if candidate_mae == 0 else -1.0
        else:
            relative_improvement = (baseline_mae - candidate_mae) / baseline_mae
        wins = 0
        for fold in folds:
            models = fold.get("models")
            if not isinstance(models, dict) or candidate not in models or baseline not in models:
                raise ValueError("each fold must contain candidate and baseline metrics")
            candidate_fold = models[candidate]
            baseline_fold = models[baseline]
            if not isinstance(candidate_fold, dict) or not isinstance(baseline_fold, dict):
                raise ValueError("fold model metrics must be mappings")
            fold_maes = (candidate_fold.get("mae"), baseline_fold.get("mae"))
            if not all(
                isinstance(value, int | float) and math.isfinite(value) and value >= 0
                for value in fold_maes
            ):
                raise ValueError("fold MAE values must be finite and non-negative")
            if candidate_fold["mae"] < baseline_fold["mae"]:
                wins += 1
        fold_win_rate = wins / len(folds)
        candidate_cost = candidate_metrics.get("mean_business_cost", candidate_metrics["mae"])
        baseline_cost = baseline_metrics.get("mean_business_cost", baseline_metrics["mae"])
        if not all(
            math.isfinite(value) and value >= 0 for value in (candidate_cost, baseline_cost)
        ):
            raise ValueError("business-cost values must be finite and non-negative")
        if baseline_cost == 0:
            cost_ratio = 1.0 if candidate_cost == 0 else float.fromhex("0x1.fffffffffffffp+1023")
        else:
            cost_ratio = candidate_cost / baseline_cost
        reasons: list[str] = []
        if relative_improvement < self.min_relative_mae_improvement:
            reasons.append("relative_mae_improvement_below_threshold")
        if fold_win_rate < self.min_fold_win_rate:
            reasons.append("insufficient_fold_win_rate")
        if cost_ratio > self.max_business_cost_ratio:
            reasons.append("business_cost_regression")
        status = ModelStatus.CHAMPION if not reasons else ModelStatus.REJECTED
        return PromotionDecision(
            candidate,
            baseline,
            status,
            tuple(reasons) or ("all_promotion_gates_passed",),
            {
                "relative_mae_improvement": relative_improvement,
                "fold_win_rate": fold_win_rate,
                "business_cost_ratio": cost_ratio,
            },
        )


def write_model_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = {
        "schema_version": 1,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "payload": payload,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
