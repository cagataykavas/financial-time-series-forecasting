from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finforecast import __version__
from finforecast.data import market_data_from_frame
from finforecast.engine import ForecastExperiment
from finforecast.evidence import build_market_evidence
from finforecast.features import make_supervised_features
from finforecast.synthetic import synthetic_market

app = FastAPI(title="Financial Time-Series Forecasting", version=__version__)


class MarketObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    close: float = Field(gt=0, allow_inf_nan=False)
    volume: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class MarketEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1, max_length=128)
    seed: int = 42
    observations: list[MarketObservation] = Field(min_length=320, max_length=5_000)

    @model_validator(mode="after")
    def consistent_volume_schema(self) -> MarketEvaluationRequest:
        supplied = [row.volume is not None for row in self.observations]
        if any(supplied) and not all(supplied):
            raise ValueError("volume must be supplied for every observation or omitted entirely")
        return self


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


@app.post("/v1/evaluations", response_model=None)
def evaluate_market(request: MarketEvaluationRequest) -> dict[str, object]:
    """Run the governed evaluation contract on a bounded market-data payload."""
    include_volume = request.observations[0].volume is not None
    frame = pd.DataFrame.from_records(
        [row.model_dump(mode="python") for row in request.observations]
    )
    try:
        market = market_data_from_frame(
            frame,
            source_name=request.source_name,
            volume_column="volume" if include_volume else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return build_market_evidence(market, seed=request.seed)
