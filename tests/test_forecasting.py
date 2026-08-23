from __future__ import annotations

from fastapi.testclient import TestClient

from app.api import app
from finforecast.engine import ForecastExperiment
from finforecast.features import make_supervised_features
from finforecast.synthetic import synthetic_market


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
    response = TestClient(app).get("/demo?rows=650&seed=3&cost_bps=5")
    assert response.status_code == 200
    assert response.json()["observations"] > 100
