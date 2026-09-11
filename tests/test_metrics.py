from __future__ import annotations

import pytest

from finforecast.metrics import (
    BusinessCost,
    interval_metrics,
    mae,
    mase,
    pinball_loss,
    rmse,
    smape,
)


def test_point_metrics_have_known_values():
    actual = [0.0, 1.0, 2.0]
    predicted = [0.0, 2.0, 1.0]
    assert mae(actual, predicted) == pytest.approx(2 / 3)
    assert rmse(actual, predicted) == pytest.approx((2 / 3) ** 0.5)
    assert smape([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_mase_rejects_constant_training_scale():
    with pytest.raises(ValueError, match="constant"):
        mase([1.0], [0.0], [2.0, 2.0, 2.0])


def test_pinball_penalizes_underforecast_by_quantile():
    assert pinball_loss([2.0], [1.0], 0.9) == pytest.approx(0.9)
    assert pinball_loss([2.0], [1.0], 0.1) == pytest.approx(0.1)


def test_interval_metrics_validate_bounds():
    result = interval_metrics([0.0, 2.0], [-1.0, 0.0], [1.0, 3.0])
    assert result == {"coverage": 1.0, "mean_width": 2.5}
    with pytest.raises(ValueError, match="lower"):
        interval_metrics([0.0], [1.0], [-1.0])


def test_business_cost_can_value_underforecast_more_than_overforecast():
    cost = BusinessCost(underforecast_weight=3.0, overforecast_weight=1.0)
    under = cost.evaluate([2.0], [1.0])
    over = cost.evaluate([1.0], [2.0])
    assert under["mean_business_cost"] == 3 * over["mean_business_cost"]
