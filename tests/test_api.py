from __future__ import annotations

import httpx
import pytest

from app.api import app
from finforecast.synthetic import synthetic_market

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as session:
        yield session


def _payload(rows: int = 320) -> dict[str, object]:
    close, volume = synthetic_market(rows, seed=9)
    return {
        "source_name": "integration-fixture",
        "seed": 9,
        "observations": [
            {
                "timestamp": timestamp.isoformat(),
                "close": float(close_value),
                "volume": float(volume_value),
            }
            for timestamp, close_value, volume_value in zip(
                close.index, close.to_numpy(), volume.to_numpy(), strict=True
            )
        ],
    }


async def test_market_evaluation_exposes_governed_real_data_path(client) -> None:
    response = await client.post("/v1/evaluations", json=_payload())

    assert response.status_code == 200
    result = response.json()
    assert result["dataset"]["kind"] == "external_market_csv"
    assert result["dataset"]["source_name"] == "integration-fixture"
    assert len(result["dataset"]["sha256"]) == 64
    assert result["validation"]["fold_count"] > 0
    assert "promotion_decision" in result


async def test_market_evaluation_rejects_unordered_observations(client) -> None:
    payload = _payload()
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations[10], observations[11] = observations[11], observations[10]

    response = await client.post("/v1/evaluations", json=payload)

    assert response.status_code == 422
    assert "timestamps must be increasing" in response.text


async def test_market_evaluation_rejects_mixed_volume_schema(client) -> None:
    payload = _payload()
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations[0]["volume"] = None

    response = await client.post("/v1/evaluations", json=payload)

    assert response.status_code == 422
    assert "volume must be supplied" in response.text


async def test_market_evaluation_rejects_non_finite_close(client) -> None:
    payload = _payload()
    observations = payload["observations"]
    assert isinstance(observations, list)
    observations[0]["close"] = "NaN"

    response = await client.post("/v1/evaluations", json=payload)

    assert response.status_code == 422


async def test_market_evaluation_rejects_payloads_outside_work_bound(client) -> None:
    response = await client.post("/v1/evaluations", json=_payload(319))

    assert response.status_code == 422
