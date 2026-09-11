from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _paired(actual: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(predicted, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("actual and predicted must be equally sized one-dimensional arrays")
    if left.size == 0 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("metric inputs must be finite and non-empty")
    return left, right


def mae(actual: object, predicted: object) -> float:
    left, right = _paired(actual, predicted)
    return float(np.mean(np.abs(left - right)))


def rmse(actual: object, predicted: object) -> float:
    left, right = _paired(actual, predicted)
    return float(np.sqrt(np.mean(np.square(left - right))))


def smape(actual: object, predicted: object) -> float:
    left, right = _paired(actual, predicted)
    denominator = np.abs(left) + np.abs(right)
    terms = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(left),
        where=denominator > 0,
    )
    return float(np.mean(terms))


def mase(actual: object, predicted: object, training_target: object, seasonality: int = 1) -> float:
    left, right = _paired(actual, predicted)
    history = np.asarray(training_target, dtype=float)
    if seasonality < 1 or history.size <= seasonality:
        raise ValueError("training history must exceed seasonality")
    scale = float(np.mean(np.abs(history[seasonality:] - history[:-seasonality])))
    if scale <= np.finfo(float).eps:
        raise ValueError("MASE is undefined for a constant training series")
    return float(np.mean(np.abs(left - right)) / scale)


def pinball_loss(actual: object, quantile_forecast: object, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    left, right = _paired(actual, quantile_forecast)
    error = left - right
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def interval_metrics(actual: object, lower: object, upper: object) -> dict[str, float]:
    values, lows = _paired(actual, lower)
    _, highs = _paired(actual, upper)
    if np.any(lows > highs):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    covered = (values >= lows) & (values <= highs)
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(highs - lows)),
    }


@dataclass(frozen=True)
class BusinessCost:
    underforecast_weight: float = 1.0
    overforecast_weight: float = 1.0
    action_threshold: float = 0.0
    action_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.underforecast_weight < 0 or self.overforecast_weight < 0:
            raise ValueError("forecast error weights cannot be negative")
        if self.action_threshold < 0 or self.action_cost < 0:
            raise ValueError("action threshold and cost cannot be negative")

    def evaluate(self, actual: object, predicted: object) -> dict[str, float]:
        left, right = _paired(actual, predicted)
        error = left - right
        weighted = np.where(
            error >= 0,
            error * self.underforecast_weight,
            -error * self.overforecast_weight,
        )
        actions = np.abs(right) >= self.action_threshold
        costs = weighted + actions.astype(float) * self.action_cost
        return {
            "mean_business_cost": float(costs.mean()),
            "total_business_cost": float(costs.sum()),
            "action_rate": float(actions.mean()),
        }
