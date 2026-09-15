from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Query

from finforecast import __version__
from finforecast.engine import ForecastExperiment
from finforecast.features import make_supervised_features
from finforecast.synthetic import synthetic_market

app = FastAPI(title="Financial Time-Series Forecasting", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo(
    seed: int = 42,
    rows: Annotated[int, Query(ge=320, le=2_000)] = 900,
    cost_bps: Annotated[float, Query(ge=0.0, le=1_000.0)] = 5.0,
) -> dict[str, object]:
    close, volume = synthetic_market(rows=rows, seed=seed)
    data = make_supervised_features(close, volume)
    experiment = ForecastExperiment(min_train=250, test_size=20, seed=seed)
    predictions = experiment.walk_forward(data)
    return experiment.evaluate(predictions, cost_bps=cost_bps)
