from __future__ import annotations

import numpy as np
import pandas as pd


def make_supervised_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Create features using only information available before the prediction timestamp."""
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.empty:
        raise ValueError("close must contain observations")
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("close must have a unique, increasing time index")
    close = close.astype(float)
    if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError("close values must be finite and strictly positive")

    returns = close.pct_change()
    frame = pd.DataFrame(index=close.index)
    for lag in (1, 2, 3, 5, 10, 20):
        frame[f"return_lag_{lag}"] = returns.shift(lag)
    for window in (5, 10, 20, 60):
        history = returns.shift(1)
        frame[f"mean_{window}"] = history.rolling(window).mean()
        frame[f"vol_{window}"] = history.rolling(window).std()
        frame[f"momentum_{window}"] = close.shift(1).pct_change(window)
    if volume is not None:
        if not isinstance(volume, pd.Series):
            raise TypeError("volume must be a pandas Series")
        if not volume.index.is_unique:
            raise ValueError("volume must have a unique index")
        volume = volume.astype(float).reindex(close.index)
        if volume.isna().any():
            raise ValueError("volume must contain every close timestamp")
        if not np.isfinite(volume.to_numpy()).all() or (volume < 0).any():
            raise ValueError("volume values must be finite and non-negative")
        volume_change = volume.pct_change()
        frame["volume_change_lag_1"] = volume_change.shift(1)
        frame["volume_z_20"] = (
            volume.shift(1) - volume.shift(1).rolling(20).mean()
        ) / volume.shift(1).rolling(20).std()
    frame["target_next_return"] = returns.shift(-1)
    supervised = frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if supervised.empty:
        raise ValueError("at least 63 valid price observations are required")
    return supervised


def assert_no_target_leakage(
    frame: pd.DataFrame, target_column: str = "target_next_return"
) -> None:
    """Validate the structural contract expected by the backtest runner.

    Column names cannot prove that arbitrary user features are point-in-time safe, but
    this catches known target aliases and malformed matrices before a model is fitted.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("supervised data must be a pandas DataFrame")
    forbidden = {"future_return", "next_close", "target_lag_0"}
    overlap = forbidden.intersection(column for column in frame.columns if column != target_column)
    if overlap:
        raise ValueError(f"forbidden leakage-prone columns present: {sorted(overlap)}")
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} is required")
    feature_columns = [column for column in frame.columns if column != target_column]
    if not feature_columns:
        raise ValueError("at least one feature column is required")
    try:
        values = frame[[*feature_columns, target_column]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("features and target must be numeric") from exc
    if frame.empty or not np.isfinite(values).all():
        raise ValueError("features and target must be finite and non-empty")
