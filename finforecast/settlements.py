from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data import _utc_cutoff

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._:-]{0,31}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
SETTLEMENT_EVENT_TYPES = frozenset({"cash_flow", "closeout"})
SETTLEMENT_COLUMNS = frozenset(
    {
        "event_id",
        "symbol",
        "event_type",
        "effective_at",
        "available_at",
        "revision_id",
        "cash_amount",
        "currency",
    }
)


@dataclass(frozen=True)
class SettlementSnapshot:
    """Point-in-time evidence for a closed, revision-bearing cash settlement."""

    as_of: str
    symbol: str
    price_currency: str
    source_name: str
    raw_rows: int
    eligible_revisions: int
    selected_events: int
    applied_cash_flows: int
    excluded_unavailable_revisions: int
    excluded_not_effective_events: int
    superseded_revisions: int
    closeout_at: str
    total_cash_amount: float
    selected_events_sha256: str
    lineage_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "symbol": self.symbol,
            "price_currency": self.price_currency,
            "source_name": self.source_name,
            "raw_rows": self.raw_rows,
            "eligible_revisions": self.eligible_revisions,
            "selected_events": self.selected_events,
            "applied_cash_flows": self.applied_cash_flows,
            "excluded_unavailable_revisions": self.excluded_unavailable_revisions,
            "excluded_not_effective_events": self.excluded_not_effective_events,
            "superseded_revisions": self.superseded_revisions,
            "closeout_at": self.closeout_at,
            "total_cash_amount": self.total_cash_amount,
            "selected_events_sha256": self.selected_events_sha256,
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


def _timestamps(values: pd.Series, column: str) -> pd.DatetimeIndex:
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


def _identifiers(values: pd.Series, column: str) -> list[str]:
    result = values.tolist()
    if any(
        not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None for value in result
    ):
        raise ValueError(f"column {column!r} contains invalid identifiers")
    return result


def load_point_in_time_settlement(
    path: str | Path,
    *,
    as_of: object,
    symbol: str,
    price_currency: str,
    market_end: object,
    expected_closeout_at: object,
) -> SettlementSnapshot:
    """Select the latest known settlement revisions and require an explicit closeout.

    Cash flows are summed without discounting or reinvestment. Foreign-currency
    entries are rejected rather than silently sharing the corporate-action FX route.
    """

    if not isinstance(symbol, str) or SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("symbol must be an uppercase stable identifier")
    if not isinstance(price_currency, str) or CURRENCY_RE.fullmatch(price_currency) is None:
        raise ValueError("price_currency must be an uppercase ISO 4217 code")
    cutoff = _utc_cutoff(as_of)
    market_timestamp = _utc_cutoff(market_end)
    closeout_timestamp = _utc_cutoff(expected_closeout_at)
    if market_timestamp >= closeout_timestamp:
        raise ValueError("settlement closeout must be strictly after the final market observation")

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"settlement CSV does not exist or is not a file: {source}")
    try:
        frame = pd.read_csv(source, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read settlement CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("settlement CSV cannot be empty")
    if set(frame.columns) != SETTLEMENT_COLUMNS:
        missing = sorted(SETTLEMENT_COLUMNS.difference(frame.columns))
        unexpected = sorted(set(frame.columns).difference(SETTLEMENT_COLUMNS))
        raise ValueError(
            "settlement CSV columns do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )

    event_ids = _identifiers(frame["event_id"], "event_id")
    revisions = _identifiers(frame["revision_id"], "revision_id")
    symbols = frame["symbol"].tolist()
    if any(value != symbol for value in symbols):
        raise ValueError("settlement rows must match the requested symbol")
    event_types = frame["event_type"].tolist()
    if any(value not in SETTLEMENT_EVENT_TYPES for value in event_types):
        raise ValueError("settlement event_type must be cash_flow or closeout")
    effective_at = _timestamps(frame["effective_at"], "effective_at")
    available_at = _timestamps(frame["available_at"], "available_at")
    cash_amounts = pd.to_numeric(frame["cash_amount"], errors="coerce").astype(float)
    currencies = [None if value == "" else value for value in frame["currency"]]
    for event_type, cash_amount, currency in zip(
        event_types, cash_amounts, currencies, strict=True
    ):
        if event_type == "cash_flow":
            if not math.isfinite(cash_amount) or cash_amount <= 0:
                raise ValueError("settlement cash flows must be finite and positive")
            if currency != price_currency:
                raise ValueError("settlement cash flows must use the declared price_currency")
        elif cash_amount != 0 or currency is not None:
            raise ValueError("settlement closeout must declare zero cash and no currency")

    order = list(zip(event_ids, available_at, strict=True))
    if order != sorted(order):
        raise ValueError("settlement rows must be ordered by event_id and availability")
    identity = pd.DataFrame(
        {
            "event_id": event_ids,
            "symbol": symbols,
            "event_type": event_types,
            "effective_at": effective_at,
            "available_at": available_at,
            "revision_id": revisions,
            "cash_amount": cash_amounts,
            "currency": currencies,
        }
    )
    if identity.duplicated(["event_id", "available_at"]).any():
        raise ValueError("settlement revisions have ambiguous availability timestamps")
    if identity.duplicated(["event_id", "revision_id"]).any():
        raise ValueError("revision IDs must be unique per settlement event")
    if identity.groupby("event_id")["event_type"].nunique().gt(1).any():
        raise ValueError("a settlement event cannot change type across revisions")

    eligible_mask = identity["available_at"] <= cutoff
    eligible = identity.loc[eligible_mask].copy()
    selected = eligible.groupby("event_id", sort=False).tail(1)
    effective_mask = selected["effective_at"] <= cutoff
    applied = selected.loc[effective_mask].sort_values(["effective_at", "event_id"], kind="stable")
    closeouts = applied.loc[applied["event_type"] == "closeout"]
    if len(closeouts) != 1:
        raise ValueError("settlement requires exactly one effective closeout")
    closeout = closeouts.iloc[0]
    if closeout["effective_at"] != closeout_timestamp:
        raise ValueError("settlement closeout must match the cash-delisting effective timestamp")
    if applied.iloc[-1]["event_type"] != "closeout":
        raise ValueError("settlement closeout must be the terminal effective event")
    cash_flows = applied.loc[applied["event_type"] == "cash_flow"]
    if not cash_flows.empty and (
        (cash_flows["effective_at"] <= market_timestamp).any()
        or (cash_flows["effective_at"] > closeout_timestamp).any()
    ):
        raise ValueError("settlement cash flows must occur after market data and by closeout")
    total_cash_amount = math.fsum(float(value) for value in cash_flows["cash_amount"])
    if not math.isfinite(total_cash_amount):
        raise ValueError("settlement total cash amount must be finite")

    def record(row: Any) -> dict[str, object]:
        return {
            "event_id": row.event_id,
            "symbol": row.symbol,
            "event_type": row.event_type,
            "effective_at": row.effective_at.isoformat(),
            "available_at": row.available_at.isoformat(),
            "revision_id": row.revision_id,
            "cash_amount": row.cash_amount,
            "currency": row.currency if isinstance(row.currency, str) else None,
        }

    selected_payload = [record(row) for row in selected.itertuples(index=False)]
    lineage_payload = [record(row) for row in identity.itertuples(index=False)]
    return SettlementSnapshot(
        as_of=cutoff.isoformat(),
        symbol=symbol,
        price_currency=price_currency,
        source_name=source.name,
        raw_rows=len(identity),
        eligible_revisions=len(eligible),
        selected_events=len(selected),
        applied_cash_flows=len(cash_flows),
        excluded_unavailable_revisions=int((~eligible_mask).sum()),
        excluded_not_effective_events=int((~effective_mask).sum()),
        superseded_revisions=len(eligible) - len(selected),
        closeout_at=closeout_timestamp.isoformat(),
        total_cash_amount=total_cash_amount,
        selected_events_sha256=_digest(
            {
                "schema_version": 1,
                "as_of": cutoff.isoformat(),
                "selection": "latest_available_revision_then_effective_closeout",
                "events": selected_payload,
            }
        ),
        lineage_sha256=_digest(
            {
                "schema_version": 1,
                "as_of": cutoff.isoformat(),
                "rows": lineage_payload,
            }
        ),
    )
