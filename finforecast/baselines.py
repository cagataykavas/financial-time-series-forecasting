from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Regressor(Protocol):
    def fit(self, features: object, target: object) -> Regressor: ...

    def predict(self, features: object) -> np.ndarray: ...


@dataclass
class ZeroReturnRegressor:
    value: float = 0.0

    def fit(self, features: object, target: object) -> ZeroReturnRegressor:
        return self

    def predict(self, features: object) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)  # type: ignore[arg-type]


@dataclass
class HistoricalMeanRegressor:
    mean_: float | None = None

    def fit(self, features: object, target: object) -> HistoricalMeanRegressor:
        values = np.asarray(target, dtype=float)
        if values.size == 0:
            raise ValueError("cannot fit a baseline without target history")
        self.mean_ = float(values.mean())
        return self

    def predict(self, features: object) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("baseline must be fitted before prediction")
        return np.full(len(features), self.mean_, dtype=float)  # type: ignore[arg-type]


@dataclass
class SeasonalNaiveRegressor:
    season_length: int = 5
    tail_: np.ndarray | None = None

    def fit(self, features: object, target: object) -> SeasonalNaiveRegressor:
        values = np.asarray(target, dtype=float)
        if self.season_length < 1 or values.size < self.season_length:
            raise ValueError("season_length must fit inside target history")
        self.tail_ = values[-self.season_length :].copy()
        return self

    def predict(self, features: object) -> np.ndarray:
        if self.tail_ is None:
            raise RuntimeError("baseline must be fitted before prediction")
        count = len(features)  # type: ignore[arg-type]
        return np.resize(self.tail_, count).astype(float)
