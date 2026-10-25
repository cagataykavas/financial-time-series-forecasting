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
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
SUPPORTED_ACTION_TYPES = frozenset({"split", "cash_dividend"})


@dataclass(frozen=True)
class CorporateActionSnapshot:
    """Point-in-time lineage for corporate-action adjusted observations."""

    as_of: str
    symbol: str
    price_currency: str | None
    source_name: str
    raw_rows: int
    eligible_revisions: int
    selected_actions: int
    applied_splits: int
    applied_cash_dividends: int
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
            "price_currency": self.price_currency,
            "source_name": self.source_name,
            "raw_rows": self.raw_rows,
            "eligible_revisions": self.eligible_revisions,
            "selected_actions": self.selected_actions,
            "applied_splits": self.applied_splits,
            "applied_cash_dividends": self.applied_cash_dividends,
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


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([float("nan")] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _optional_text(frame: pd.DataFrame, column: str) -> list[str | None]:
    if column not in frame:
        return [None] * len(frame)
    return [None if pd.isna(value) or value == "" else value for value in frame[column]]


def _apply_point_in_time_actions(
    market: MarketData,
    actions_path: str | Path,
    *,
    as_of: object,
    symbol: str,
    price_currency: str | None,
    allowed_action_types: frozenset[str],
    action_id_column: str = "action_id",
    symbol_column: str = "symbol",
    action_type_column: str = "action_type",
    effective_at_column: str = "effective_at",
    available_at_column: str = "available_at",
    revision_column: str = "revision_id",
    split_ratio_column: str = "split_ratio",
    cash_amount_column: str = "cash_amount",
    currency_column: str = "currency",
) -> MarketData:
    if not isinstance(market, MarketData):
        raise TypeError("market must be a MarketData instance")
    if market.corporate_actions is not None:
        raise ValueError("market data already contains corporate-action adjustments")
    if not isinstance(symbol, str) or SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("symbol must be an uppercase stable identifier")
    if price_currency is not None and (
        not isinstance(price_currency, str) or CURRENCY_RE.fullmatch(price_currency) is None
    ):
        raise ValueError("price_currency must be an uppercase ISO 4217 code")
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
    all_column_names = [
        action_id_column,
        symbol_column,
        action_type_column,
        effective_at_column,
        available_at_column,
        revision_column,
        split_ratio_column,
        cash_amount_column,
        currency_column,
    ]
    if any(not column for column in all_column_names) or len(set(all_column_names)) != len(
        all_column_names
    ):
        raise ValueError("corporate-action column names must be non-empty and distinct")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read corporate-action CSV: {exc}") from exc
    required = {
        action_id_column,
        symbol_column,
        action_type_column,
        effective_at_column,
        available_at_column,
        revision_column,
    }
    missing = required.difference(frame.columns)
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
    if any(value not in allowed_action_types for value in action_types):
        supported = ", ".join(sorted(allowed_action_types))
        raise ValueError(f"supported corporate-action types are: {supported}")
    if "split" in action_types and split_ratio_column not in frame:
        raise ValueError(f"corporate-action CSV is missing columns: ['{split_ratio_column}']")
    if "cash_dividend" in action_types:
        dividend_columns = {cash_amount_column, currency_column}
        missing_dividend = dividend_columns.difference(frame.columns)
        if missing_dividend:
            raise ValueError(f"corporate-action CSV is missing columns: {sorted(missing_dividend)}")
        if price_currency is None:
            raise ValueError("cash dividends require a declared price_currency")

    effective_at = _strict_utc_timestamps(frame[effective_at_column], effective_at_column)
    available_at = _strict_utc_timestamps(frame[available_at_column], available_at_column)
    ratios = _optional_numeric(frame, split_ratio_column)
    cash_amounts = _optional_numeric(frame, cash_amount_column)
    currencies = _optional_text(frame, currency_column)
    for action_type, ratio, cash_amount, currency in zip(
        action_types, ratios, cash_amounts, currencies, strict=True
    ):
        if action_type == "split":
            if not math.isfinite(ratio) or ratio <= 0 or ratio == 1:
                raise ValueError("split ratios must be finite, positive and different from one")
            if math.isfinite(cash_amount) or currency is not None:
                raise ValueError("split rows cannot declare cash dividend fields")
        else:
            if not math.isfinite(cash_amount) or cash_amount <= 0:
                raise ValueError("cash dividend amounts must be finite and positive")
            if math.isfinite(ratio):
                raise ValueError("cash dividend rows cannot declare a split ratio")
            if currency != price_currency:
                raise ValueError("cash dividend currency must match price_currency")

    order = list(zip(action_ids, available_at, strict=True))
    if order != sorted(order):
        raise ValueError("corporate-action rows must be ordered by action_id and availability")
    identity = pd.DataFrame(
        {
            "action_id": action_ids,
            "action_type": action_types,
            "effective_at": effective_at,
            "available_at": available_at,
            "revision_id": revisions,
            "split_ratio": ratios,
            "cash_amount": cash_amounts,
            "currency": currencies,
        }
    )
    if identity.duplicated(["action_id", "available_at"]).any():
        raise ValueError("corporate-action revisions have ambiguous availability timestamps")
    if identity.duplicated(["action_id", "revision_id"]).any():
        raise ValueError("revision IDs must be unique per corporate action")
    if identity.groupby("action_id")["action_type"].nunique().gt(1).any():
        raise ValueError("a corporate action cannot change type across revisions")
    identity["source_row"] = identity.index

    eligible_mask = identity["available_at"] <= cutoff
    eligible = identity.loc[eligible_mask].copy()
    selected = eligible.groupby("action_id", sort=False).tail(1)
    effective_mask = selected["effective_at"] <= cutoff
    applied = selected.loc[effective_mask].sort_values(["effective_at", "action_id"], kind="stable")
    if applied.duplicated(["effective_at"]).any():
        raise ValueError("multiple actions at the same effective timestamp are ambiguous")

    adjusted_close = market.close.copy(deep=True)
    adjusted_close.index = market_index
    adjusted_volume = market.volume.copy(deep=True) if market.volume is not None else None
    if adjusted_volume is not None:
        adjusted_volume.index = market_index
    selected_payload: list[dict[str, object]] = []
    for row in applied.itertuples(index=False):
        historical = market_index < row.effective_at
        payload: dict[str, object] = {
            "action_id": row.action_id,
            "action_type": row.action_type,
            "effective_at": row.effective_at.isoformat(),
            "available_at": row.available_at.isoformat(),
            "revision_id": row.revision_id,
        }
        if row.action_type == "split":
            adjusted_close.loc[historical] = adjusted_close.loc[historical] / row.split_ratio
            if adjusted_volume is not None:
                adjusted_volume.loc[historical] = adjusted_volume.loc[historical] * row.split_ratio
            payload["split_ratio"] = row.split_ratio
        else:
            reference_positions = market_index[historical]
            if reference_positions.empty:
                raise ValueError("cash dividend requires a pre-effective market observation")
            reference_at = reference_positions[-1]
            reference_close = float(adjusted_close.loc[reference_at])
            if row.cash_amount >= reference_close:
                raise ValueError("cash dividend must be smaller than its reference close")
            factor = (reference_close - row.cash_amount) / reference_close
            adjusted_close.loc[historical] = adjusted_close.loc[historical] * factor
            payload.update(
                {
                    "cash_amount": row.cash_amount,
                    "currency": row.currency,
                    "reference_at": reference_at.isoformat(),
                    "reference_close": reference_close,
                    "back_adjustment_factor": factor,
                }
            )
        selected_payload.append(payload)

    adjusted_sha256 = _fingerprint(adjusted_close, adjusted_volume)
    lineage_payload: list[dict[str, object]] = []
    lineage_rows = zip(
        action_ids,
        symbols,
        action_types,
        effective_at,
        available_at,
        revisions,
        ratios,
        cash_amounts,
        currencies,
        strict=True,
    )
    for row in lineage_rows:
        action_id, action_symbol, action_type, effective, available, revision = row[:6]
        ratio, cash_amount, currency = row[6:]
        lineage_payload.append(
            {
                "action_id": action_id,
                "symbol": action_symbol,
                "action_type": action_type,
                "effective_at": effective.isoformat(),
                "available_at": available.isoformat(),
                "revision_id": revision,
                "split_ratio": ratio if math.isfinite(ratio) else None,
                "cash_amount": cash_amount if math.isfinite(cash_amount) else None,
                "currency": currency,
            }
        )
    snapshot = CorporateActionSnapshot(
        as_of=cutoff.isoformat(),
        symbol=symbol,
        price_currency=price_currency,
        source_name=source.name,
        raw_rows=len(identity),
        eligible_revisions=len(eligible),
        selected_actions=len(selected),
        applied_splits=int((applied["action_type"] == "split").sum()),
        applied_cash_dividends=int((applied["action_type"] == "cash_dividend").sum()),
        excluded_unavailable_revisions=int((~eligible_mask).sum()),
        excluded_not_effective_actions=int((~effective_mask).sum()),
        superseded_revisions=len(eligible) - len(selected),
        unadjusted_market_sha256=market.sha256,
        adjusted_market_sha256=adjusted_sha256,
        selected_actions_sha256=_digest(selected_payload),
        lineage_sha256=_digest(
            {
                "schema_version": 2,
                "as_of": cutoff.isoformat(),
                "symbol": symbol,
                "price_currency": price_currency,
                "selection": "latest_available_revision_then_effective_action",
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


def apply_point_in_time_corporate_actions(
    market: MarketData,
    actions_path: str | Path,
    *,
    as_of: object,
    symbol: str,
    price_currency: str | None = None,
) -> MarketData:
    """Apply split and cash-dividend revisions known and effective at the cutoff."""

    return _apply_point_in_time_actions(
        market,
        actions_path,
        as_of=as_of,
        symbol=symbol,
        price_currency=price_currency,
        allowed_action_types=SUPPORTED_ACTION_TYPES,
    )


def apply_point_in_time_splits(
    market: MarketData,
    actions_path: str | Path,
    *,
    as_of: object,
    symbol: str,
) -> MarketData:
    """Backward-compatible split-only corporate-action boundary."""

    return _apply_point_in_time_actions(
        market,
        actions_path,
        as_of=as_of,
        symbol=symbol,
        price_currency=None,
        allowed_action_types=frozenset({"split"}),
    )
