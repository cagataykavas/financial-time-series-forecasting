from __future__ import annotations

import pandas as pd


def make_supervised_features(close: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    """Create features using only information available before the prediction timestamp."""
    close = close.astype(float).sort_index()
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
        volume = volume.astype(float).reindex(close.index)
        volume_change = volume.pct_change()
        frame["volume_change_lag_1"] = volume_change.shift(1)
        frame["volume_z_20"] = (
            volume.shift(1) - volume.shift(1).rolling(20).mean()
        ) / volume.shift(1).rolling(20).std()
    frame["target_next_return"] = returns.shift(-1)
    return frame.replace([float("inf"), float("-inf")], pd.NA).dropna()


def assert_no_target_leakage(frame: pd.DataFrame) -> None:
    forbidden = {"future_return", "next_close", "target_lag_0"}
    overlap = forbidden.intersection(frame.columns)
    if overlap:
        raise ValueError(f"forbidden leakage-prone columns present: {sorted(overlap)}")
    if "target_next_return" not in frame.columns:
        raise ValueError("target_next_return column is required")
