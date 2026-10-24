from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import MarketData, _fingerprint, _utc_cutoff

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{0,31}$")


@dataclass(frozen=True)
class CorporateActionSnapshot:
    """Point-in-time lineage for split adjustments applied to raw observations."""

    as_of: str
    symbol: str
    source_name: str
    raw_rows: int
    eligible_revisions: int
    selected_actions: int
    applied_splits: int
    excluded_unavailable_revisions: int
    excluded_not_effective_actions: int
    superseded_revisions: int
    unadjusted_market_sha256: str
    adjusted_market_sha256: str
    selected_actions_sha256: str
    lineage_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "symbol": self.symbol,
            "source_name": self.source_name,
            "raw_rows": self.raw_rows,
            "eligible_revisions": self.eligible_revisions,
            "selected_actions": self.selected_actions,
            "applied_splits": self.applied_splits,
            "excluded_unavailable_revisions": self.excluded_unavailable_revisions,
            "excluded_not_effective_actions": self.excluded_not_effective_actions,
            "superseded_revisions": self.superseded_revisions,
            "unadjusted_market_sha256": self.unadjusted_market_sha256,
            "adjusted_market_sha256": self.adjusted_market_sha256,
            "selected_actions_sha256": self.selected_actions_sha256,
            "lineage_sha256": self.lineage_sha256,
        }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _strict_utc_timestamps(values: pd.Series, column: str) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp] = []
    for raw in values:
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {column!r} contains invalid timestamps") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"column {column!r} must contain timezone-aware timestamps")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(parsed)


def _validate_identifiers(values: pd.Series, column: str) -> list[str]:
    identifiers = values.tolist()
    if any(
        not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None
        for value in identifiers
    ):
        raise ValueError(f"column {column!r} contains invalid identifiers")
    return identifiers


def apply_point_in_time_splits(
    market: MarketData,
    actions_path: str | Path,
    *,
    as_of: object,
    symbol: str,
    action_id_column: str = "action_id",
    symbol_column: str = "symbol",
    action_type_column: str = "action_type",
    effective_at_column: str = "effective_at",
    available_at_column: str = "available_at",
    revision_column: str = "revision_id",
    split_ratio_column: str = "split_ratio",
) -> MarketData:
    """Apply only split revisions known and effective at the declared cutoff."""

    if not isinstance(market, MarketData):
        raise TypeError("market must be a MarketData instance")
    if market.corporate_actions is not None:
        raise ValueError("market data already contains corporate-action adjustments")
    if not isinstance(symbol, str) or SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("symbol must be an uppercase stable identifier")
    cutoff = _utc_cutoff(as_of)
    if market.close.index.tz is None:
        raise ValueError("market observations must be timezone-aware")
    market_index = market.close.index.tz_convert("UTC")
    if market_index[-1] > cutoff:
        raise ValueError("market observations cannot extend beyond the as_of cutoff")
    if market.snapshot is not None and pd.Timestamp(market.snapshot.as_of) != cutoff:
        raise ValueError("market and corporate-action snapshots must use the same as_of cutoff")

    source = Path(actions_path)
    if not source.is_file():
        raise ValueError(f"corporate-action CSV does not exist or is not a file: {source}")
    columns = [
        action_id_column,
        symbol_column,
        action_type_column,
        effective_at_column,
        available_at_column,
        revision_column,
        split_ratio_column,
    ]
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("corporate-action column names must be non-empty and distinct")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read corporate-action CSV: {exc}") from exc
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"corporate-action CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("corporate-action CSV cannot be empty")

    action_ids = _validate_identifiers(frame[action_id_column], action_id_column)
    revisions = _validate_identifiers(frame[revision_column], revision_column)
    symbols = frame[symbol_column].tolist()
    if any(value != symbol for value in symbols):
        raise ValueError("corporate-action rows must match the requested symbol")
    action_types = frame[action_type_column].tolist()
    if any(value != "split" for value in action_types):
        raise ValueError("only split corporate actions are supported")
    effective_at = _strict_utc_timestamps(frame[effective_at_column], effective_at_column)
    available_at = _strict_utc_timestamps(frame[available_at_column], available_at_column)
    ratios = pd.to_numeric(frame[split_ratio_column], errors="coerce").astype(float)
    if any(not math.isfinite(value) or value <= 0 or value == 1 for value in ratios):
        raise ValueError("split ratios must be finite, positive and different from one")

    order = list(zip(action_ids, available_at, strict=True))
    if order != sorted(order):
        raise ValueError("corporate-action rows must be ordered by action_id and availability")
    identity = pd.DataFrame(
        {
            "action_id": action_ids,
            "effective_at": effective_at,
            "available_at": available_at,
            "revision_id": revisions,
            "split_ratio": ratios,
        }
    )
    if identity.duplicated(["action_id", "available_at"]).any():
        raise ValueError("corporate-action revisions have ambiguous availability timestamps")
    if identity.duplicated(["action_id", "revision_id"]).any():
        raise ValueError("revision IDs must be unique per corporate action")
    identity["source_row"] = identity.index

    eligible_mask = identity["available_at"] <= cutoff
    eligible = identity.loc[eligible_mask].copy()
    selected = eligible.groupby("action_id", sort=False).tail(1)
    effective_mask = selected["effective_at"] <= cutoff
    applied = selected.loc[effective_mask].sort_values(["effective_at", "action_id"], kind="stable")

    adjusted_close = market.close.copy(deep=True)
    adjusted_close.index = market_index
    adjusted_volume = market.volume.copy(deep=True) if market.volume is not None else None
    if adjusted_volume is not None:
        adjusted_volume.index = market_index
    for row in applied.itertuples(index=False):
        historical = market_index < row.effective_at
        adjusted_close.loc[historical] = adjusted_close.loc[historical] / row.split_ratio
        if adjusted_volume is not None:
            adjusted_volume.loc[historical] = adjusted_volume.loc[historical] * row.split_ratio

    adjusted_sha256 = _fingerprint(adjusted_close, adjusted_volume)
    selected_payload = [
        {
            "action_id": row.action_id,
            "effective_at": row.effective_at.isoformat(),
            "available_at": row.available_at.isoformat(),
            "revision_id": row.revision_id,
            "split_ratio": row.split_ratio,
        }
        for row in applied.itertuples(index=False)
    ]
    lineage_payload = [
        {
            "action_id": action_id,
            "symbol": action_symbol,
            "action_type": action_type,
            "effective_at": effective.isoformat(),
            "available_at": available.isoformat(),
            "revision_id": revision,
            "split_ratio": ratio,
        }
        for action_id, action_symbol, action_type, effective, available, revision, ratio in zip(
            action_ids,
            symbols,
            action_types,
            effective_at,
            available_at,
            revisions,
            ratios,
            strict=True,
        )
    ]
    snapshot = CorporateActionSnapshot(
        as_of=cutoff.isoformat(),
        symbol=symbol,
        source_name=source.name,
        raw_rows=len(identity),
        eligible_revisions=len(eligible),
        selected_actions=len(selected),
        applied_splits=len(applied),
        excluded_unavailable_revisions=int((~eligible_mask).sum()),
        excluded_not_effective_actions=int((~effective_mask).sum()),
        superseded_revisions=len(eligible) - len(selected),
        unadjusted_market_sha256=market.sha256,
        adjusted_market_sha256=adjusted_sha256,
        selected_actions_sha256=_digest(selected_payload),
        lineage_sha256=_digest(
            {
                "schema_version": 1,
                "as_of": cutoff.isoformat(),
                "symbol": symbol,
                "selection": "latest_available_revision_then_effective_split",
                "rows": lineage_payload,
            }
        ),
    )
    return MarketData(
        close=adjusted_close,
        volume=adjusted_volume,
        source_name=market.source_name,
        sha256=adjusted_sha256,
        snapshot=market.snapshot,
        corporate_actions=snapshot,
    )
