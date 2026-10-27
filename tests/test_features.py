from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finforecast.features import ForecastTargetSpec, assert_target_spec, make_supervised_features


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


def test_features_admit_only_an_explicit_terminal_zero_recovery():
    close = prices()
    close.iloc[-1] = 0.0

    result = make_supervised_features(close, allow_terminal_zero=True)

    assert result.index[-1] == close.index[-2]
    assert result.iloc[-1]["target_next_return"] == -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        make_supervised_features(close)
    early_zero = prices()
    early_zero.iloc[3] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        make_supervised_features(early_zero, allow_terminal_zero=True)


def test_features_require_enough_history_for_rolling_windows():
    with pytest.raises(ValueError, match="not enough"):
        make_supervised_features(prices(20))


def test_target_spec_builds_exact_forward_return_for_declared_horizon():
    close = prices()
    spec = ForecastTargetSpec(horizon=3)
    result = make_supervised_features(close, target_spec=spec)
    timestamp = result.index[0]
    position = close.index.get_loc(timestamp)

    expected = close.iloc[position + 3] / close.iloc[position] - 1.0
    assert result.loc[timestamp, spec.column] == pytest.approx(expected)
    assert result.attrs["target_spec"] == spec.as_dict()
    assert result.attrs["target_spec_sha256"] == spec.sha256
    assert len(result.attrs["target_values_sha256"]) == 64
    assert_target_spec(result, spec)


@pytest.mark.parametrize("horizon", [0, 253, -1])
def test_target_spec_rejects_unbounded_horizons(horizon):
    with pytest.raises(ValueError, match="between 1 and 252"):
        ForecastTargetSpec(horizon=horizon)


def test_target_spec_rejects_boolean_horizon():
    with pytest.raises(TypeError, match="integer"):
        ForecastTargetSpec(horizon=True)


def test_target_evidence_is_deterministic_and_tamper_evident():
    first = ForecastTargetSpec(horizon=5)
    second = ForecastTargetSpec(horizon=5)
    other = ForecastTargetSpec(horizon=6)
    frame = make_supervised_features(prices(90), target_spec=first)

    assert first.sha256 == second.sha256
    assert first.sha256 != other.sha256
    frame.attrs["target_spec"]["horizon"] = 6
    with pytest.raises(ValueError, match="digest is invalid"):
        assert_target_spec(frame, first)

    frame = make_supervised_features(prices(90), target_spec=first)
    frame.loc[frame.index[0], first.column] += 0.01
    with pytest.raises(ValueError, match="target values digest is invalid"):
        assert_target_spec(frame, first)
