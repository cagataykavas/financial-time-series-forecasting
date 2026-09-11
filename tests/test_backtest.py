from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finforecast.backtest import BacktestRunner
from finforecast.baselines import HistoricalMeanRegressor, ZeroReturnRegressor
from finforecast.metrics import BusinessCost
from finforecast.splits import WalkForwardConfig, WalkForwardSplitter


def dataset(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": np.arange(rows), "target_next_return": np.sin(np.arange(rows) / 4)},
        index=pd.date_range("2024-01-01", periods=rows, freq="D"),
    )


def test_backtest_produces_unique_out_of_sample_rows_and_fold_reports():
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5, gap=1)),
        {"zero": ZeroReturnRegressor, "mean": HistoricalMeanRegressor},
    )
    result = runner.run(dataset())
    assert result.predictions.index.is_unique
    assert set(result.predictions) == {"actual", "fold", "zero", "mean"}
    assert len(result.folds) == 5
    assert result.folds[0]["boundary"]["train_end"] == 15


def test_aggregate_includes_asymmetric_business_cost():
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5)),
        {"zero": ZeroReturnRegressor},
    )
    metrics = runner.aggregate(runner.run(dataset()), BusinessCost(2.0, 1.0))
    assert metrics["zero"]["mean_business_cost"] > 0
    assert 0 <= metrics["zero"]["directional_accuracy"] <= 1


def test_backtest_rejects_non_monotonic_time_index():
    frame = dataset().sort_index(ascending=False)
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5)),
        {"zero": ZeroReturnRegressor},
    )
    with pytest.raises(ValueError, match="increasing"):
        runner.run(frame)


def test_overlapping_test_windows_are_rejected():
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5, step_size=2)),
        {"zero": ZeroReturnRegressor},
    )
    with pytest.raises(ValueError, match="overlapping"):
        runner.run(dataset())
