from __future__ import annotations

import numpy as np
import pytest

from finforecast.baselines import (
    HistoricalMeanRegressor,
    SeasonalNaiveRegressor,
    ZeroReturnRegressor,
)


def test_zero_return_baseline_has_no_fit_state():
    baseline = ZeroReturnRegressor().fit([[1.0]], [99.0])
    assert baseline.predict([[1.0], [2.0]]).tolist() == [0.0, 0.0]


def test_historical_mean_uses_training_target_only():
    baseline = HistoricalMeanRegressor().fit(np.zeros((3, 1)), [1.0, 2.0, 3.0])
    assert baseline.predict(np.zeros((2, 1))).tolist() == [2.0, 2.0]


def test_seasonal_naive_repeats_only_the_training_tail():
    baseline = SeasonalNaiveRegressor(season_length=3).fit(
        np.zeros((5, 1)), [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert baseline.predict(np.zeros((5, 1))).tolist() == [3.0, 4.0, 5.0, 3.0, 4.0]


def test_unfitted_or_undersized_baselines_fail_closed():
    with pytest.raises(RuntimeError, match="fitted"):
        HistoricalMeanRegressor().predict([[1.0]])
    with pytest.raises(ValueError, match="season_length"):
        SeasonalNaiveRegressor(4).fit([[1.0]], [1.0])
