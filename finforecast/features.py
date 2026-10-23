from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastTargetSpec:
    """Immutable contract for constructing a forward simple-return target."""

    horizon: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("target horizon must be an integer")
        if not 1 <= self.horizon <= 252:
            raise ValueError("target horizon must be between 1 and 252 observations")

    @property
    def column(self) -> str:
        return "target_next_return" if self.horizon == 1 else f"target_return_h{self.horizon}"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "forward_simple_return",
            "horizon": self.horizon,
            "target_column": self.column,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _target_values_sha256(target: pd.Series) -> str:
    canonical = pd.DataFrame(
        {
            "timestamp": [str(value) for value in target.index],
            "target": target.to_numpy(dtype=float),
        }
    )
    return hashlib.sha256(canonical.to_csv(index=False).encode("utf-8")).hexdigest()


def make_supervised_features(
    close: pd.Series,
    volume: pd.Series | None = None,
    *,
    target_spec: ForecastTargetSpec | None = None,
) -> pd.DataFrame:
    """Create features using only information available before the prediction timestamp."""
    spec = target_spec or ForecastTargetSpec()
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
    frame[spec.column] = close.shift(-spec.horizon) / close - 1.0
    supervised = frame.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if supervised.empty:
        raise ValueError(
            f"not enough valid price observations for horizon {spec.horizon} and rolling features"
        )
    supervised.attrs["target_spec"] = spec.as_dict()
    supervised.attrs["target_spec_sha256"] = spec.sha256
    supervised.attrs["target_values_sha256"] = _target_values_sha256(supervised[spec.column])
    return supervised


def assert_target_spec(frame: pd.DataFrame, expected: ForecastTargetSpec) -> None:
    """Fail closed when generated target evidence is absent, changed or mismatched."""
    metadata = frame.attrs.get("target_spec")
    digest = frame.attrs.get("target_spec_sha256")
    values_digest = frame.attrs.get("target_values_sha256")
    if metadata is None or digest is None or values_digest is None:
        raise ValueError("supervised data is missing target specification evidence")
    if (
        not isinstance(metadata, dict)
        or not isinstance(digest, str)
        or not isinstance(values_digest, str)
    ):
        raise ValueError("target specification evidence is malformed")
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    observed_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if observed_digest != digest:
        raise ValueError("target specification evidence digest is invalid")
    if metadata != expected.as_dict() or digest != expected.sha256:
        raise ValueError("supervised target specification does not match the evaluation policy")
    if expected.column not in frame:
        raise ValueError("supervised data is missing the declared target column")
    if _target_values_sha256(frame[expected.column]) != values_digest:
        raise ValueError("supervised target values digest is invalid")


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
