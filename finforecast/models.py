from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import (
    HistoricalMeanRegressor,
    Regressor,
    SeasonalNaiveRegressor,
    ZeroReturnRegressor,
)


def default_model_factories(seed: int = 42) -> dict[str, Callable[[], Regressor]]:
    return {
        "naive_zero": ZeroReturnRegressor,
        "historical_mean": HistoricalMeanRegressor,
        "seasonal_naive_5": lambda: SeasonalNaiveRegressor(5),
        "ridge": lambda: Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=3.0))]),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=100,
            l2_regularization=0.5,
            random_state=seed,
        ),
    }
