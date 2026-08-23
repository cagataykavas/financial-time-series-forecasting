from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import assert_no_target_leakage


@dataclass(frozen=True)
class FoldPrediction:
    timestamp: str
    actual: float
    naive_zero: float
    ridge: float
    gradient_boosting: float


class ForecastExperiment:
    def __init__(self, min_train: int = 250, test_size: int = 20, seed: int = 42) -> None:
        self.min_train = min_train
        self.test_size = test_size
        self.seed = seed

    def _models(self) -> dict[str, Callable[[], object]]:
        return {
            "ridge": lambda: Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=3.0)),
            ]),
            "gradient_boosting": lambda: HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=180,
                l2_regularization=0.5,
                random_state=self.seed,
            ),
        }

    def walk_forward(self, data: pd.DataFrame) -> pd.DataFrame:
        assert_no_target_leakage(data)
        features = [column for column in data.columns if column != "target_next_return"]
        rows: list[pd.DataFrame] = []
        for start in range(self.min_train, len(data), self.test_size):
            train = data.iloc[:start]
            test = data.iloc[start : start + self.test_size]
            if test.empty:
                break
            fold = pd.DataFrame(index=test.index)
            fold["actual"] = test["target_next_return"]
            fold["naive_zero"] = 0.0
            for name, factory in self._models().items():
                model = factory()
                model.fit(train[features], train["target_next_return"])
                fold[name] = model.predict(test[features])
            rows.append(fold)
        if not rows:
            raise ValueError("not enough observations for a walk-forward fold")
        return pd.concat(rows).sort_index()

    @staticmethod
    def _strategy_metrics(actual: pd.Series, prediction: pd.Series, cost_bps: float) -> dict[str, float]:
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
        loss_a = np.abs(actual.to_numpy() - prediction_a.to_numpy())
        loss_b = np.abs(actual.to_numpy() - prediction_b.to_numpy())
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
        actual = predictions["actual"]
        model_metrics: dict[str, dict[str, float]] = {}
        for name in [column for column in predictions.columns if column != "actual"]:
            pred = predictions[name]
            model_metrics[name] = {
                "mae": float(mean_absolute_error(actual, pred)),
                "rmse": float(mean_squared_error(actual, pred) ** 0.5),
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
                {"timestamp": str(index), **{column: float(value) for column, value in row.items()}}
                for index, row in predictions.iterrows()
            ],
        }
