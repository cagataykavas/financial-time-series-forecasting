from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backtest import BacktestRunner
from .features import make_supervised_features
from .intervals import SplitConformalCalibrator
from .metrics import BusinessCost, interval_metrics, pinball_loss
from .models import default_model_factories
from .registry import PromotionPolicy, write_model_manifest
from .splits import WalkForwardConfig, WalkForwardSplitter
from .synthetic import synthetic_market


def build_evidence(rows: int = 900, seed: int = 42) -> dict[str, Any]:
    close, volume = synthetic_market(rows, seed)
    supervised = make_supervised_features(close, volume)
    splitter = WalkForwardSplitter(WalkForwardConfig(min_train_size=250, test_size=25, gap=1))
    runner = BacktestRunner(splitter, default_model_factories(seed))
    backtest = runner.run(supervised)
    cost = BusinessCost(
        underforecast_weight=1.5,
        overforecast_weight=1.0,
        action_threshold=0.002,
        action_cost=0.00005,
    )
    aggregate = runner.aggregate(backtest, cost)

    predictions = backtest.predictions
    calibration_end = max(2, int(len(predictions) * 0.4))
    calibration = predictions.iloc[:calibration_end]
    evaluation = predictions.iloc[calibration_end:]
    calibrator = SplitConformalCalibrator(coverage=0.9).fit(
        calibration["actual"], calibration["gradient_boosting"]
    )
    lower, upper = calibrator.predict(evaluation["gradient_boosting"])
    uncertainty = {
        **calibrator.metadata(),
        **interval_metrics(evaluation["actual"], lower, upper),
        "lower_pinball_loss": pinball_loss(evaluation["actual"], lower, 0.05),
        "upper_pinball_loss": pinball_loss(evaluation["actual"], upper, 0.95),
        "evaluation_size": len(evaluation),
    }
    policy = PromotionPolicy(
        min_relative_mae_improvement=0.01,
        max_business_cost_ratio=1.0,
        min_fold_win_rate=0.5,
    )
    decision = policy.decide("gradient_boosting", "naive_zero", aggregate, backtest.folds)
    return {
        "schema_version": 1,
        "dataset": {
            "kind": "deterministic_synthetic_regime_shift",
            "seed": seed,
            "raw_observations": rows,
            "supervised_observations": len(supervised),
        },
        "validation": {
            "strategy": "expanding_walk_forward",
            "gap": 1,
            "fold_count": len(backtest.folds),
            "out_of_sample_observations": len(predictions),
            "folds": [fold["boundary"] for fold in backtest.folds],
        },
        "aggregate_metrics": aggregate,
        "uncertainty": uncertainty,
        "promotion_decision": decision.as_dict(),
        "claims": [
            "All reported forecasts are out of sample within the synthetic experiment.",
            (
                "The conformal interval is calibrated on an earlier prediction segment "
                "and evaluated later."
            ),
            "Synthetic results are engineering evidence, not evidence of market profitability.",
        ],
    }


def write_evidence(output: str | Path, rows: int = 900, seed: int = 42) -> dict[str, Any]:
    payload = build_evidence(rows, seed)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "forecast_evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_model_manifest(
        destination / "candidate_manifest.json",
        {
            "model": "gradient_boosting",
            "dataset": payload["dataset"],
            "validation": payload["validation"],
            "metrics": payload["aggregate_metrics"]["gradient_boosting"],
            "decision": payload["promotion_decision"],
        },
    )
    return payload
