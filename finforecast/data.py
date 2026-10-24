from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .corporate_actions import CorporateActionSnapshot

REVISION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Point-in-time selection evidence for a revision-bearing market feed."""

    as_of: str
    available_at_column: str
    revision_column: str
    raw_rows: int
    eligible_rows: int
    selected_observations: int
    excluded_future_revisions: int
    superseded_revisions: int
    selected_revisions_sha256: str
    lineage_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "available_at_column": self.available_at_column,
            "revision_column": self.revision_column,
            "raw_rows": self.raw_rows,
            "eligible_rows": self.eligible_rows,
            "selected_observations": self.selected_observations,
            "excluded_future_revisions": self.excluded_future_revisions,
            "superseded_revisions": self.superseded_revisions,
            "selected_revisions_sha256": self.selected_revisions_sha256,
            "lineage_sha256": self.lineage_sha256,
        }


@dataclass(frozen=True)
class MarketData:
    """Validated, normalized market observations and their reproducible identity."""

    close: pd.Series
    volume: pd.Series | None
    source_name: str
    sha256: str
    snapshot: MarketDataSnapshot | None = None
    corporate_actions: CorporateActionSnapshot | None = None

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "kind": (
                "point_in_time_split_adjusted_market"
                if self.corporate_actions is not None
                else (
                    "point_in_time_market_csv"
                    if self.snapshot is not None
                    else "external_market_csv"
                )
            ),
            "source_name": self.source_name,
            "sha256": self.sha256,
            "raw_observations": len(self.close),
            "start": self.close.index[0].isoformat(),
            "end": self.close.index[-1].isoformat(),
            "has_volume": self.volume is not None,
        }
        if self.snapshot is not None:
            metadata["snapshot"] = self.snapshot.metadata()
        if self.corporate_actions is not None:
            metadata["corporate_actions"] = self.corporate_actions.metadata()
        return metadata


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


def _utc_cutoff(value: object) -> pd.Timestamp:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("as_of must be a valid timezone-aware timestamp") from exc
    if cutoff.tzinfo is None:
        raise ValueError("as_of must include an explicit timezone")
    return cutoff.tz_convert("UTC")


def _snapshot_lineage_sha256(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    timestamp_column: str,
    close_column: str,
    volume_column: str | None,
    available_at_column: str,
    revision_column: str,
) -> str:
    columns = [
        timestamp_column,
        available_at_column,
        revision_column,
        close_column,
    ]
    if volume_column is not None:
        columns.append(volume_column)
    rows = frame[columns].copy()
    rows[timestamp_column] = pd.to_datetime(rows[timestamp_column], utc=True).map(
        lambda value: value.isoformat()
    )
    rows[available_at_column] = pd.to_datetime(rows[available_at_column], utc=True).map(
        lambda value: value.isoformat()
    )
    payload = {
        "schema_version": 1,
        "as_of": as_of.isoformat(),
        "selection": "latest_available_revision_per_observation",
        "columns": columns,
        "rows": rows.to_dict(orient="records"),
    }
    canonical = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_market_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    close_column: str = "close",
    volume_column: str | None = "volume",
    as_of: object | None = None,
    available_at_column: str | None = None,
    revision_column: str | None = None,
) -> MarketData:
    """Load a point-in-time ordered CSV and fail closed on ambiguous market data."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"market CSV does not exist or is not a file: {source}")
    selected = [timestamp_column, close_column]
    if volume_column is not None:
        selected.append(volume_column)
    point_in_time = as_of is not None
    if point_in_time:
        if not available_at_column or not revision_column:
            raise ValueError("point-in-time loading requires available_at and revision columns")
        selected.extend((available_at_column, revision_column))
    elif available_at_column is not None or revision_column is not None:
        raise ValueError("available_at and revision columns require an as_of cutoff")
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
    close_values = _numeric_column(frame, close_column, non_negative=False)
    volume_values: pd.Series | None = None
    if volume_column is not None:
        volume_values = _numeric_column(frame, volume_column, non_negative=True)

    snapshot: MarketDataSnapshot | None = None
    if point_in_time:
        cutoff = _utc_cutoff(as_of)
        availability = pd.to_datetime(
            frame[available_at_column], format="mixed", errors="coerce", utc=True
        )
        if availability.isna().any():
            raise ValueError(f"column {available_at_column!r} contains invalid timestamps")
        revisions = frame[revision_column]
        if revisions.isna().any() or any(
            not isinstance(value, str) or REVISION_ID_RE.fullmatch(value) is None
            for value in revisions
        ):
            raise ValueError(f"column {revision_column!r} contains invalid revision IDs")
        order = list(zip(timestamps, availability, strict=True))
        if order != sorted(order):
            raise ValueError("point-in-time rows must be ordered by timestamp and availability")
        identity = pd.DataFrame(
            {
                "timestamp": timestamps,
                "available_at": availability,
                "revision": revisions,
            }
        )
        if identity.duplicated(["timestamp", "available_at"]).any():
            raise ValueError("market revisions have ambiguous availability timestamps")
        if identity.duplicated(["timestamp", "revision"]).any():
            raise ValueError("market revision IDs must be unique per observation")
        if (availability < timestamps).any():
            raise ValueError("market revisions cannot be available before observation time")

        eligible_mask = availability <= cutoff
        eligible = identity.loc[eligible_mask].copy()
        if eligible.empty:
            raise ValueError("no market revisions were available at the as_of cutoff")
        eligible["source_row"] = eligible.index
        selected_rows = eligible.groupby("timestamp", sort=False).tail(1)
        row_indices = selected_rows["source_row"].to_numpy(dtype=int)
        index = pd.DatetimeIndex(selected_rows["timestamp"], name="timestamp")
        close = pd.Series(close_values.iloc[row_indices].to_numpy(), index=index, name="close")
        volume = (
            pd.Series(volume_values.iloc[row_indices].to_numpy(), index=index, name="volume")
            if volume_values is not None
            else None
        )
        selected_revision_payload = [
            {
                "timestamp": timestamp.isoformat(),
                "available_at": available.isoformat(),
                "revision": revision,
            }
            for timestamp, available, revision in zip(
                selected_rows["timestamp"],
                selected_rows["available_at"],
                selected_rows["revision"],
                strict=True,
            )
        ]
        selected_revision_digest = hashlib.sha256(
            json.dumps(
                selected_revision_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        snapshot = MarketDataSnapshot(
            as_of=cutoff.isoformat(),
            available_at_column=available_at_column,
            revision_column=revision_column,
            raw_rows=len(frame),
            eligible_rows=int(eligible_mask.sum()),
            selected_observations=len(selected_rows),
            excluded_future_revisions=int((~eligible_mask).sum()),
            superseded_revisions=len(eligible) - len(selected_rows),
            selected_revisions_sha256=selected_revision_digest,
            lineage_sha256=_snapshot_lineage_sha256(
                frame,
                as_of=cutoff,
                timestamp_column=timestamp_column,
                close_column=close_column,
                volume_column=volume_column,
                available_at_column=available_at_column,
                revision_column=revision_column,
            ),
        )
    else:
        index = pd.DatetimeIndex(timestamps, name="timestamp")
        if not index.is_unique:
            raise ValueError("market timestamps must be unique")
        if not index.is_monotonic_increasing:
            raise ValueError("market timestamps must be increasing; sort the source explicitly")
        close = pd.Series(close_values.to_numpy(), index=index, name="close")
        volume = (
            pd.Series(volume_values.to_numpy(), index=index, name="volume")
            if volume_values is not None
            else None
        )
    return MarketData(
        close,
        volume,
        source.name,
        _fingerprint(close, volume),
        snapshot,
    )


def load_point_in_time_market_csv(
    path: str | Path,
    *,
    as_of: object,
    timestamp_column: str = "timestamp",
    close_column: str = "close",
    volume_column: str | None = "volume",
    available_at_column: str = "available_at",
    revision_column: str = "revision_id",
) -> MarketData:
    """Select the latest vendor revision available at a declared historical cutoff."""

    return load_market_csv(
        path,
        timestamp_column=timestamp_column,
        close_column=close_column,
        volume_column=volume_column,
        as_of=as_of,
        available_at_column=available_at_column,
        revision_column=revision_column,
    )
