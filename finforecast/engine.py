from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestRunner
from .metrics import mae, rmse
from .models import default_model_factories
from .splits import WalkForwardConfig, WalkForwardSplitter


class ForecastExperiment:
    def __init__(self, min_train: int = 250, test_size: int = 20, seed: int = 42) -> None:
        WalkForwardConfig(min_train_size=min_train, test_size=test_size)
        self.min_train = min_train
        self.test_size = test_size
        self.seed = seed

    def walk_forward(self, data: pd.DataFrame) -> pd.DataFrame:
        splitter = WalkForwardSplitter(
            WalkForwardConfig(min_train_size=self.min_train, test_size=self.test_size)
        )
        factories = default_model_factories(self.seed)
        selected = {
            name: factory
            for name, factory in factories.items()
            if name in {"naive_zero", "ridge", "gradient_boosting"}
        }
        return BacktestRunner(splitter, selected).run(data).predictions.drop(columns="fold")

    @staticmethod
    def _strategy_metrics(
        actual: pd.Series, prediction: pd.Series, cost_bps: float
    ) -> dict[str, float]:
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError("cost_bps must be finite and non-negative")
        position = np.sign(prediction).astype(float)
        turnover = position.diff().abs().fillna(position.abs())
        gross = position * actual
        net = gross - turnover * (cost_bps / 10_000.0)
        ann_return = float(net.mean() * 252)
        ann_vol = float(net.std(ddof=1) * np.sqrt(252))
        wealth = (1.0 + net).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        return {
            "gross_cumulative_return": float((1.0 + gross).prod() - 1.0),
            "net_cumulative_return": float(wealth.iloc[-1] - 1.0),
            "average_turnover": float(turnover.mean()),
            "strategy_sharpe": ann_return / ann_vol if ann_vol > 0 else 0.0,
            "max_drawdown": float(drawdown.min()),
        }

    @staticmethod
    def paired_bootstrap_mae_difference(
        actual: pd.Series,
        prediction_a: pd.Series,
        prediction_b: pd.Series,
        *,
        samples: int = 3000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Bootstrap MAE(A)-MAE(B); negative values favor model A.

        This is not a Diebold-Mariano test and is labelled accordingly. It is a simple
        resampling diagnostic for this public reference project.
        """
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        loss_a = np.abs(actual.to_numpy() - prediction_a.to_numpy())
        loss_b = np.abs(actual.to_numpy() - prediction_b.to_numpy())
        if (
            loss_a.ndim != 1
            or loss_a.shape != loss_b.shape
            or loss_a.size == 0
            or not np.isfinite(loss_a).all()
            or not np.isfinite(loss_b).all()
        ):
            raise ValueError("bootstrap inputs must be finite, non-empty paired vectors")
        diff = loss_a - loss_b
        rng = np.random.default_rng(seed)
        means = np.empty(samples)
        for i in range(samples):
            means[i] = rng.choice(diff, size=len(diff), replace=True).mean()
        return {
            "observed_mae_difference": float(diff.mean()),
            "bootstrap_p05": float(np.quantile(means, 0.05)),
            "bootstrap_p95": float(np.quantile(means, 0.95)),
        }

    def evaluate(self, predictions: pd.DataFrame, cost_bps: float = 5.0) -> dict[str, Any]:
        required = {"actual", "naive_zero", "gradient_boosting"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"prediction columns are missing: {sorted(missing)}")
        if predictions.empty:
            raise ValueError("predictions cannot be empty")
        if not predictions.index.is_unique or not predictions.index.is_monotonic_increasing:
            raise ValueError("predictions must have a unique, increasing time index")
        try:
            numeric = predictions.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("predictions must be numeric") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("predictions must be finite")
        if not np.isfinite(cost_bps) or cost_bps < 0:
            raise ValueError("cost_bps must be finite and non-negative")

        actual = predictions["actual"]
        model_metrics: dict[str, dict[str, float]] = {}
        model_columns = [
            column for column in predictions.columns if column not in {"actual", "fold"}
        ]
        for name in model_columns:
            pred = predictions[name]
            model_metrics[name] = {
                "mae": mae(actual, pred),
                "rmse": rmse(actual, pred),
                "directional_accuracy": float((np.sign(actual) == np.sign(pred)).mean()),
                **self._strategy_metrics(actual, pred, cost_bps),
            }
        comparison = self.paired_bootstrap_mae_difference(
            actual,
            predictions["gradient_boosting"],
            predictions["naive_zero"],
            seed=self.seed,
        )
        return {
            "observations": len(predictions),
            "cost_bps": cost_bps,
            "models": model_metrics,
            "gradient_boosting_vs_naive_mae_bootstrap": comparison,
            "predictions": [
                {
                    "timestamp": str(index),
                    **{column: float(value) for column, value in row.items()},
                }
                for index, row in predictions.iterrows()
            ],
        }
