from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finforecast.backtest import BacktestRunner
from finforecast.baselines import HistoricalMeanRegressor, ZeroReturnRegressor
from finforecast.features import ForecastTargetSpec, make_supervised_features
from finforecast.metrics import BusinessCost
from finforecast.splits import WalkForwardConfig, WalkForwardSplitter
from finforecast.synthetic import synthetic_market


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


def test_custom_target_column_is_supported():
    frame = dataset().rename(columns={"target_next_return": "label"})
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5)),
        {"zero": ZeroReturnRegressor},
        target_column="label",
    )
    assert len(runner.run(frame).predictions) == 24


def test_declared_target_horizon_must_match_split_policy_before_fitting():
    spec = ForecastTargetSpec(horizon=3)
    with pytest.raises(ValueError, match="label_horizon must match"):
        BacktestRunner(
            WalkForwardSplitter(
                WalkForwardConfig(min_train_size=15, test_size=5, gap=3, label_horizon=2)
            ),
            {"zero": ZeroReturnRegressor},
            target_column=spec.column,
            target_spec=spec,
        )


def test_backtest_rejects_target_built_under_a_different_spec():
    close, volume = synthetic_market(rows=100, seed=3)
    built_spec = ForecastTargetSpec(horizon=2)
    policy_spec = ForecastTargetSpec(horizon=3)
    data = make_supervised_features(close, volume, target_spec=built_spec)
    runner = BacktestRunner(
        WalkForwardSplitter(
            WalkForwardConfig(min_train_size=20, test_size=5, gap=3, label_horizon=3)
        ),
        {"zero": ZeroReturnRegressor},
        target_column=policy_spec.column,
        target_spec=policy_spec,
    )

    with pytest.raises(ValueError, match="does not match"):
        runner.run(data)


def test_backtest_rejects_bad_data_before_model_fit():
    runner = BacktestRunner(
        WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5)),
        {"zero": ZeroReturnRegressor},
    )
    with pytest.raises(ValueError, match="finite"):
        runner.run(dataset().assign(feature=float("nan")))
    with pytest.raises(ValueError, match="reserved"):
        BacktestRunner(
            WalkForwardSplitter(WalkForwardConfig(min_train_size=15, test_size=5)),
            {"actual": ZeroReturnRegressor},
        )
