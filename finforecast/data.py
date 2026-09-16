from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    """Validated, normalized market observations and their reproducible identity."""

    close: pd.Series
    volume: pd.Series | None
    source_name: str
    sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "external_market_csv",
            "source_name": self.source_name,
            "sha256": self.sha256,
            "raw_observations": len(self.close),
            "start": self.close.index[0].isoformat(),
            "end": self.close.index[-1].isoformat(),
            "has_volume": self.volume is not None,
        }


def _numeric_column(frame: pd.DataFrame, column: str, *, non_negative: bool) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"column {column!r} must contain only finite numeric values")
    if non_negative:
        if (values < 0).any():
            raise ValueError(f"column {column!r} cannot contain negative values")
    elif (values <= 0).any():
        raise ValueError(f"column {column!r} must contain strictly positive values")
    return values


def _fingerprint(close: pd.Series, volume: pd.Series | None) -> str:
    canonical = pd.DataFrame({"timestamp": close.index, "close": close.to_numpy()})
    if volume is not None:
        canonical["volume"] = volume.to_numpy()
    payload = canonical.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_market_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    close_column: str = "close",
    volume_column: str | None = "volume",
) -> MarketData:
    """Load a point-in-time ordered CSV and fail closed on ambiguous market data."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"market CSV does not exist or is not a file: {source}")
    selected = [timestamp_column, close_column]
    if volume_column is not None:
        selected.append(volume_column)
    if any(not column for column in selected) or len(set(selected)) != len(selected):
        raise ValueError("timestamp, close and volume column names must be non-empty and distinct")

    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read market CSV: {exc}") from exc
    missing = set(selected).difference(frame.columns)
    if missing:
        raise ValueError(f"market CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("market CSV cannot be empty")

    timestamps = pd.to_datetime(frame[timestamp_column], format="mixed", errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(f"column {timestamp_column!r} contains invalid timestamps")
    index = pd.DatetimeIndex(timestamps, name="timestamp")
    if not index.is_unique:
        raise ValueError("market timestamps must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("market timestamps must be increasing; sort the source explicitly")

    close_values = _numeric_column(frame, close_column, non_negative=False)
    close = pd.Series(close_values.to_numpy(), index=index, name="close")
    volume: pd.Series | None = None
    if volume_column is not None:
        volume_values = _numeric_column(frame, volume_column, non_negative=True)
        volume = pd.Series(volume_values.to_numpy(), index=index, name="volume")
    return MarketData(close, volume, source.name, _fingerprint(close, volume))
