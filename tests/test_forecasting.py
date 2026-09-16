from __future__ import annotations

import asyncio

import httpx
import pandas as pd
import pytest

from app.api import app
from finforecast.engine import ForecastExperiment
from finforecast.features import make_supervised_features
from finforecast.synthetic import synthetic_market


def api_get(path: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(request())


def test_walk_forward_produces_out_of_sample_predictions():
    close, volume = synthetic_market(rows=650, seed=42)
    data = make_supervised_features(close, volume)
    experiment = ForecastExperiment(min_train=200, test_size=25, seed=42)
    predictions = experiment.walk_forward(data)
    assert len(predictions) > 100
    assert {"actual", "naive_zero", "ridge", "gradient_boosting"}.issubset(predictions.columns)


def test_evaluation_includes_baselines_costs_and_bootstrap():
    close, volume = synthetic_market(rows=620, seed=7)
    data = make_supervised_features(close, volume)
    experiment = ForecastExperiment(min_train=200, test_size=30, seed=7)
    result = experiment.evaluate(experiment.walk_forward(data), cost_bps=5.0)
    assert "naive_zero" in result["models"]
    assert "gradient_boosting" in result["models"]
    assert "gradient_boosting_vs_naive_mae_bootstrap" in result
    bootstrap = result["gradient_boosting_vs_naive_mae_bootstrap"]
    assert bootstrap["method"] == "circular_moving_block_bootstrap"
    assert bootstrap["block_size"] > 0
    assert result["models"]["gradient_boosting"]["average_turnover"] >= 0


def test_demo_api():
    response = api_get("/demo?rows=650&seed=3&cost_bps=5")
    assert response.status_code == 200
    assert response.json()["observations"] > 100


@pytest.mark.parametrize(
    "query",
    ["rows=319", "rows=2001", "cost_bps=-1", "cost_bps=1001"],
)
def test_demo_api_rejects_unsafe_parameter_ranges(query: str):
    response = api_get(f"/demo?{query}")
    assert response.status_code == 422


def test_evaluation_rejects_invalid_prediction_frames():
    experiment = ForecastExperiment(min_train=2, test_size=1)
    valid = pd.DataFrame(
        {
            "actual": [0.1, -0.2],
            "naive_zero": [0.0, 0.0],
            "gradient_boosting": [0.1, -0.1],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    with pytest.raises(ValueError, match="non-negative"):
        experiment.evaluate(valid, cost_bps=-1)
    with pytest.raises(ValueError, match="finite"):
        experiment.evaluate(valid.assign(actual=[float("nan"), 0.1]))


def test_evaluation_ignores_backtest_fold_metadata():
    experiment = ForecastExperiment(min_train=2, test_size=1)
    predictions = pd.DataFrame(
        {
            "actual": [0.1, -0.2],
            "fold": [0, 1],
            "naive_zero": [0.0, 0.0],
            "gradient_boosting": [0.1, -0.1],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    assert "fold" not in experiment.evaluate(predictions)["models"]


def test_block_bootstrap_is_deterministic_and_validates_block_size():
    actual = pd.Series([0.0, 1.0, -1.0, 2.0, -2.0, 1.0])
    prediction_a = pd.Series([0.1, 0.8, -0.7, 1.7, -1.8, 0.8])
    prediction_b = pd.Series([0.0, 0.4, -0.2, 1.0, -1.0, 0.2])

    first = ForecastExperiment.paired_bootstrap_mae_difference(
        actual, prediction_a, prediction_b, samples=200, block_size=3, seed=9
    )
    second = ForecastExperiment.paired_bootstrap_mae_difference(
        actual, prediction_a, prediction_b, samples=200, block_size=3, seed=9
    )
    assert first == second
    assert first["samples"] == 200
    assert first["block_size"] == 3
    assert first["observed_mae_difference"] < 0

    with pytest.raises(ValueError, match="block_size"):
        ForecastExperiment.paired_bootstrap_mae_difference(
            actual, prediction_a, prediction_b, block_size=7
        )

    with pytest.raises(ValueError, match="paired vectors"):
        ForecastExperiment.paired_bootstrap_mae_difference(
            actual,
            prediction_a.set_axis(pd.RangeIndex(10, 16)),
            prediction_b,
        )
