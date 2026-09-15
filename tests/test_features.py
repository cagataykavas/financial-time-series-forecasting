from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finforecast.features import make_supervised_features


def prices(rows: int = 80) -> pd.Series:
    return pd.Series(
        np.linspace(100.0, 120.0, rows),
        index=pd.date_range("2024-01-01", periods=rows, freq="D"),
    )


def test_features_keep_a_unique_increasing_time_index():
    result = make_supervised_features(prices())
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert np.isfinite(result.to_numpy()).all()


def test_features_reject_malformed_market_series():
    with pytest.raises(ValueError, match="increasing"):
        make_supervised_features(prices().sort_index(ascending=False))
    with pytest.raises(ValueError, match="strictly positive"):
        make_supervised_features(prices().mask(lambda value: value == value.iloc[3], 0.0))
    with pytest.raises(ValueError, match="every close timestamp"):
        make_supervised_features(prices(), prices().iloc[:-1])


def test_features_require_enough_history_for_rolling_windows():
    with pytest.raises(ValueError, match="63"):
        make_supervised_features(prices(20))
