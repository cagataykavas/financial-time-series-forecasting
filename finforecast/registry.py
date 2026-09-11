from __future__ import annotations

import hashlib
import json
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
        candidate_metrics = aggregate[candidate]
        baseline_metrics = aggregate[baseline]
        baseline_mae = baseline_metrics["mae"]
        relative_improvement = (
            (baseline_mae - candidate_metrics["mae"]) / baseline_mae if baseline_mae else 0.0
        )
        wins = 0
        for fold in folds:
            models = fold["models"]
            assert isinstance(models, dict)
            if models[candidate]["mae"] < models[baseline]["mae"]:  # type: ignore[index]
                wins += 1
        fold_win_rate = wins / len(folds) if folds else 0.0
        candidate_cost = candidate_metrics.get("mean_business_cost", candidate_metrics["mae"])
        baseline_cost = baseline_metrics.get("mean_business_cost", baseline_metrics["mae"])
        cost_ratio = candidate_cost / baseline_cost if baseline_cost else float("inf")
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
