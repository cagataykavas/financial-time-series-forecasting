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
from .fx import load_fx_rate_book

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
    fx_source_name: str | None
    fx_max_age_seconds: float | None
    fx_max_leg_skew_seconds: float | None
    fx_observed_max_leg_skew_seconds: float | None
    fx_bridge_currencies: tuple[str, ...]
    fx_routes: int
    fx_raw_rows: int
    fx_eligible_revisions: int
    fx_selected_observations: int
    fx_excluded_unavailable_revisions: int
    fx_superseded_revisions: int
    fx_lineage_sha256: str | None
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
            "fx_source_name": self.fx_source_name,
            "fx_max_age_seconds": self.fx_max_age_seconds,
            "fx_max_leg_skew_seconds": self.fx_max_leg_skew_seconds,
            "fx_observed_max_leg_skew_seconds": self.fx_observed_max_leg_skew_seconds,
            "fx_bridge_currencies": list(self.fx_bridge_currencies),
            "fx_routes": self.fx_routes,
            "fx_raw_rows": self.fx_raw_rows,
            "fx_eligible_revisions": self.fx_eligible_revisions,
            "fx_selected_observations": self.fx_selected_observations,
            "fx_excluded_unavailable_revisions": self.fx_excluded_unavailable_revisions,
            "fx_superseded_revisions": self.fx_superseded_revisions,
            "fx_lineage_sha256": self.fx_lineage_sha256,
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


def _positive_duration(value: object, name: str) -> pd.Timedelta:
    try:
        duration = pd.Timedelta(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive duration") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive duration")
    return duration


def load_point_in_time_settlement(
    path: str | Path,
    *,
    as_of: object,
    symbol: str,
    price_currency: str,
    market_end: object,
    expected_closeout_at: object,
    fx_rates_path: str | Path | None = None,
    max_fx_age: object = "7 days",
    max_fx_leg_skew: object = "1 hour",
    fx_bridge_currencies: tuple[str, ...] = (),
) -> SettlementSnapshot:
    """Select the latest known settlement revisions and require an explicit closeout.

    Cash flows are summed without discounting or reinvestment. Each foreign-currency
    payment is converted at its own effective timestamp under a separate FX policy.
    """

    if not isinstance(symbol, str) or SYMBOL_RE.fullmatch(symbol) is None:
        raise ValueError("symbol must be an uppercase stable identifier")
    if not isinstance(price_currency, str) or CURRENCY_RE.fullmatch(price_currency) is None:
        raise ValueError("price_currency must be an uppercase ISO 4217 code")
    if not isinstance(fx_bridge_currencies, tuple) or any(
        not isinstance(currency, str) or CURRENCY_RE.fullmatch(currency) is None
        for currency in fx_bridge_currencies
    ):
        raise ValueError("settlement fx_bridge_currencies must contain uppercase ISO 4217 codes")
    if len(set(fx_bridge_currencies)) != len(fx_bridge_currencies):
        raise ValueError("settlement fx_bridge_currencies must be unique")
    fx_bridge_currencies = tuple(sorted(fx_bridge_currencies))
    if price_currency in fx_bridge_currencies:
        raise ValueError("price_currency cannot also be a settlement FX bridge currency")
    fx_age = _positive_duration(max_fx_age, "settlement max_fx_age")
    fx_leg_skew = _positive_duration(max_fx_leg_skew, "settlement max_fx_leg_skew")
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
            if not isinstance(currency, str) or CURRENCY_RE.fullmatch(currency) is None:
                raise ValueError("settlement cash-flow currency must be an uppercase ISO 4217 code")
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
    required_fx_pairs = {
        (row.currency, price_currency)
        for row in cash_flows.itertuples(index=False)
        if row.currency != price_currency
    }
    cash_currencies = {base for base, _ in required_fx_pairs}
    if cash_currencies.intersection(fx_bridge_currencies):
        raise ValueError("a settlement cash-flow currency cannot also be an FX bridge currency")
    allowed_fx_pairs = set(required_fx_pairs)
    for base, quote in required_fx_pairs:
        for bridge in fx_bridge_currencies:
            allowed_fx_pairs.add((base, bridge))
            allowed_fx_pairs.add((bridge, quote))
    if required_fx_pairs and fx_rates_path is None:
        raise ValueError("foreign-currency settlement cash flows require a settlement FX-rate CSV")
    if not required_fx_pairs and fx_rates_path is not None:
        raise ValueError(
            "settlement FX-rate CSV is not allowed without a foreign-currency cash flow"
        )
    fx_book = (
        load_fx_rate_book(fx_rates_path, cutoff=cutoff, allowed_pairs=allowed_fx_pairs)
        if fx_rates_path is not None
        else None
    )
    converted_by_event: dict[str, dict[str, object]] = {}
    converted_cash_amounts: list[float] = []
    used_fx_pairs: set[tuple[str, str]] = set()
    observed_leg_skews: list[float] = []
    for row in cash_flows.itertuples(index=False):
        converted_cash_amount = float(row.cash_amount)
        if row.currency != price_currency:
            if fx_book is None:  # guarded above; retained as a fail-closed invariant
                raise ValueError("foreign-currency settlement cash flow has no FX-rate book")
            fx_rate, fx_legs, observed_leg_skew = fx_book.resolve(
                row.currency,
                price_currency,
                row.effective_at,
                fx_age,
                fx_leg_skew,
                fx_bridge_currencies,
                context="settlement cash flow",
            )
            used_fx_pairs.update((leg["base_currency"], leg["quote_currency"]) for leg in fx_legs)
            observed_leg_skews.append(observed_leg_skew)
            converted_cash_amount *= fx_rate
            converted_by_event[row.event_id] = {
                "fx_rate": fx_rate,
                "fx_leg_skew_seconds": observed_leg_skew,
                "fx_route": [
                    {
                        "base_currency": leg["base_currency"],
                        "quote_currency": leg["quote_currency"],
                        "observed_at": leg["observed_at"].isoformat(),
                        "available_at": leg["available_at"].isoformat(),
                        "revision_id": leg["revision_id"],
                        "rate": leg["rate"],
                    }
                    for leg in fx_legs
                ],
                "converted_cash_amount": converted_cash_amount,
            }
        converted_cash_amounts.append(converted_cash_amount)
    if fx_book is not None and fx_book.supplied_pairs != used_fx_pairs:
        raise ValueError("settlement FX-rate CSV contains unused or incomplete route pairs")
    total_cash_amount = math.fsum(converted_cash_amounts)
    if not math.isfinite(total_cash_amount):
        raise ValueError("settlement total cash amount must be finite")

    def record(row: Any, *, include_conversion: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "event_id": row.event_id,
            "symbol": row.symbol,
            "event_type": row.event_type,
            "effective_at": row.effective_at.isoformat(),
            "available_at": row.available_at.isoformat(),
            "revision_id": row.revision_id,
            "cash_amount": row.cash_amount,
            "currency": row.currency if isinstance(row.currency, str) else None,
        }
        if include_conversion:
            payload.update(converted_by_event.get(row.event_id, {}))
        return payload

    selected_payload = [
        record(row, include_conversion=True) for row in selected.itertuples(index=False)
    ]
    lineage_payload = [
        record(row, include_conversion=False) for row in identity.itertuples(index=False)
    ]
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
        fx_source_name=fx_book.source_name if fx_book else None,
        fx_max_age_seconds=fx_age.total_seconds() if fx_book else None,
        fx_max_leg_skew_seconds=fx_leg_skew.total_seconds() if fx_book else None,
        fx_observed_max_leg_skew_seconds=max(observed_leg_skews, default=0.0) if fx_book else None,
        fx_bridge_currencies=fx_bridge_currencies if fx_book else (),
        fx_routes=len(converted_by_event),
        fx_raw_rows=fx_book.raw_rows if fx_book else 0,
        fx_eligible_revisions=fx_book.eligible_revisions if fx_book else 0,
        fx_selected_observations=len(fx_book.selected) if fx_book else 0,
        fx_excluded_unavailable_revisions=(
            fx_book.excluded_unavailable_revisions if fx_book else 0
        ),
        fx_superseded_revisions=fx_book.superseded_revisions if fx_book else 0,
        fx_lineage_sha256=fx_book.lineage_sha256 if fx_book else None,
        selected_events_sha256=_digest(
            {
                "schema_version": 2,
                "as_of": cutoff.isoformat(),
                "price_currency": price_currency,
                "fx_lineage_sha256": fx_book.lineage_sha256 if fx_book else None,
                "fx_max_age_seconds": fx_age.total_seconds() if fx_book else None,
                "fx_max_leg_skew_seconds": fx_leg_skew.total_seconds() if fx_book else None,
                "fx_bridge_currencies": list(fx_bridge_currencies) if fx_book else [],
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
