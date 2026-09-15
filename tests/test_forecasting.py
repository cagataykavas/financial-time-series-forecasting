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
