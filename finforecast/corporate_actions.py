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
    fx_source_name: str | None
    fx_max_age_seconds: float | None
    fx_bridge_currencies: tuple[str, ...]
    fx_routes: int
    fx_raw_rows: int
    fx_eligible_revisions: int
    fx_selected_observations: int
    fx_excluded_unavailable_revisions: int
    fx_superseded_revisions: int
    fx_lineage_sha256: str | None
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
            "fx_source_name": self.fx_source_name,
            "fx_max_age_seconds": self.fx_max_age_seconds,
            "fx_bridge_currencies": list(self.fx_bridge_currencies),
            "fx_routes": self.fx_routes,
            "fx_raw_rows": self.fx_raw_rows,
            "fx_eligible_revisions": self.fx_eligible_revisions,
            "fx_selected_observations": self.fx_selected_observations,
            "fx_excluded_unavailable_revisions": self.fx_excluded_unavailable_revisions,
            "fx_superseded_revisions": self.fx_superseded_revisions,
            "fx_lineage_sha256": self.fx_lineage_sha256,
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


@dataclass(frozen=True)
class _FxRateBook:
    source_name: str
    selected: pd.DataFrame
    raw_rows: int
    eligible_revisions: int
    excluded_unavailable_revisions: int
    superseded_revisions: int
    lineage_sha256: str
    supplied_pairs: frozenset[tuple[str, str]]

    def resolve(
        self,
        base_currency: str,
        quote_currency: str,
        reference_at: pd.Timestamp,
        max_age: pd.Timedelta,
        bridge_currencies: tuple[str, ...],
    ) -> tuple[float, list[pd.Series]]:
        def latest(base: str, quote: str) -> tuple[pd.Series | None, str | None]:
            pair = self.selected.loc[
                (self.selected["base_currency"] == base)
                & (self.selected["quote_currency"] == quote)
            ]
            candidates = self.selected.loc[
                (self.selected["base_currency"] == base)
                & (self.selected["quote_currency"] == quote)
                & (self.selected["observed_at"] <= reference_at)
            ]
            if candidates.empty:
                return None, "future" if not pair.empty else "missing"
            resolved = candidates.sort_values("observed_at", kind="stable").iloc[-1]
            if reference_at - resolved["observed_at"] > max_age:
                return None, "stale"
            return resolved, None

        direct, direct_error = latest(base_currency, quote_currency)
        if direct is not None:
            return float(direct["rate"]), [direct]
        if not bridge_currencies:
            if direct_error == "stale":
                raise ValueError("cash dividend FX observation exceeds the maximum age")
            raise ValueError(
                "cash dividend requires a point-in-time FX observation at or before "
                "its reference close"
            )

        routes: list[tuple[float, list[pd.Series]]] = []
        for bridge in bridge_currencies:
            first, _ = latest(base_currency, bridge)
            second, _ = latest(bridge, quote_currency)
            if first is not None and second is not None:
                routes.append((float(first["rate"] * second["rate"]), [first, second]))
        if len(routes) > 1:
            raise ValueError("cash dividend FX conversion has ambiguous bridge routes")
        if routes:
            return routes[0]
        raise ValueError(
            "cash dividend requires a fresh point-in-time direct or declared bridge FX route"
        )


def _load_fx_rate_book(
    path: str | Path,
    *,
    cutoff: pd.Timestamp,
    allowed_pairs: set[tuple[str, str]],
) -> _FxRateBook:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"FX-rate CSV does not exist or is not a file: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read FX-rate CSV: {exc}") from exc
    required = {
        "base_currency",
        "quote_currency",
        "observed_at",
        "available_at",
        "revision_id",
        "rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FX-rate CSV is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("FX-rate CSV cannot be empty")

    bases = frame["base_currency"].tolist()
    quotes = frame["quote_currency"].tolist()
    if any(
        not isinstance(value, str) or CURRENCY_RE.fullmatch(value) is None
        for value in [*bases, *quotes]
    ):
        raise ValueError("FX currencies must be uppercase ISO 4217 codes")
    pairs = set(zip(bases, quotes, strict=True))
    if not pairs.issubset(allowed_pairs):
        raise ValueError(
            "FX-rate CSV contains a currency pair outside the declared FX routes; "
            "only required direct currency pairs or declared bridge legs are allowed"
        )
    revisions = _validate_identifiers(frame["revision_id"], "revision_id")
    observed_at = _strict_utc_timestamps(frame["observed_at"], "observed_at")
    available_at = _strict_utc_timestamps(frame["available_at"], "available_at")
    observation_availability = zip(observed_at, available_at, strict=True)
    if any(observed > available for observed, available in observation_availability):
        raise ValueError("FX observations cannot be available before they were observed")
    rates = pd.to_numeric(frame["rate"], errors="coerce").astype(float)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("FX rates must be finite and positive")

    order = list(zip(bases, quotes, observed_at, available_at, strict=True))
    if order != sorted(order):
        raise ValueError("FX-rate rows must be ordered by pair, observation time and availability")
    identity = pd.DataFrame(
        {
            "base_currency": bases,
            "quote_currency": quotes,
            "observed_at": observed_at,
            "available_at": available_at,
            "revision_id": revisions,
            "rate": rates,
        }
    )
    observation_key = ["base_currency", "quote_currency", "observed_at"]
    if identity.duplicated([*observation_key, "available_at"]).any():
        raise ValueError("FX revisions have ambiguous availability timestamps")
    if identity.duplicated([*observation_key, "revision_id"]).any():
        raise ValueError("FX revision IDs must be unique per observation")

    eligible_mask = identity["available_at"] <= cutoff
    eligible = identity.loc[eligible_mask].copy()
    selected = eligible.groupby(observation_key, sort=False).tail(1)
    lineage_rows = [
        {
            "base_currency": row.base_currency,
            "quote_currency": row.quote_currency,
            "observed_at": row.observed_at.isoformat(),
            "available_at": row.available_at.isoformat(),
            "revision_id": row.revision_id,
            "rate": row.rate,
        }
        for row in identity.itertuples(index=False)
    ]
    return _FxRateBook(
        source_name=source.name,
        selected=selected,
        raw_rows=len(identity),
        eligible_revisions=len(eligible),
        excluded_unavailable_revisions=int((~eligible_mask).sum()),
        superseded_revisions=len(eligible) - len(selected),
        lineage_sha256=_digest(
            {
                "schema_version": 1,
                "as_of": cutoff.isoformat(),
                "selection": "latest_available_revision_per_observed_pair",
                "rows": lineage_rows,
            }
        ),
        supplied_pairs=frozenset(pairs),
    )


def _apply_point_in_time_actions(
    market: MarketData,
    actions_path: str | Path,
    *,
    as_of: object,
    symbol: str,
    price_currency: str | None,
    fx_rates_path: str | Path | None,
    max_fx_age: object,
    fx_bridge_currencies: tuple[str, ...],
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
    if not isinstance(fx_bridge_currencies, tuple) or any(
        not isinstance(currency, str) or CURRENCY_RE.fullmatch(currency) is None
        for currency in fx_bridge_currencies
    ):
        raise ValueError("fx_bridge_currencies must contain uppercase ISO 4217 codes")
    if len(set(fx_bridge_currencies)) != len(fx_bridge_currencies):
        raise ValueError("fx_bridge_currencies must be unique")
    fx_bridge_currencies = tuple(sorted(fx_bridge_currencies))
    if price_currency in fx_bridge_currencies:
        raise ValueError("price_currency cannot also be an FX bridge currency")
    try:
        fx_age = pd.Timedelta(max_fx_age)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("max_fx_age must be a positive duration") from exc
    if fx_age <= pd.Timedelta(0):
        raise ValueError("max_fx_age must be a positive duration")
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
            if not isinstance(currency, str) or CURRENCY_RE.fullmatch(currency) is None:
                raise ValueError("cash dividend currency must be an uppercase ISO 4217 code")

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
    required_fx_pairs = {
        (row.currency, price_currency)
        for row in applied.itertuples(index=False)
        if row.action_type == "cash_dividend" and row.currency != price_currency
    }
    dividend_currencies = {base for base, _ in required_fx_pairs}
    if dividend_currencies.intersection(fx_bridge_currencies):
        raise ValueError("a dividend currency cannot also be an FX bridge currency")
    allowed_fx_pairs = set(required_fx_pairs)
    for base, quote in required_fx_pairs:
        for bridge in fx_bridge_currencies:
            allowed_fx_pairs.add((base, bridge))
            allowed_fx_pairs.add((bridge, quote))
    if required_fx_pairs and fx_rates_path is None:
        raise ValueError("foreign-currency cash dividends require an FX-rate CSV")
    if not required_fx_pairs and fx_rates_path is not None:
        raise ValueError("FX-rate CSV is not allowed without a foreign-currency dividend")
    fx_book = (
        _load_fx_rate_book(
            fx_rates_path,
            cutoff=cutoff,
            allowed_pairs=allowed_fx_pairs,
        )
        if fx_rates_path is not None
        else None
    )

    adjusted_close = market.close.copy(deep=True)
    adjusted_close.index = market_index
    adjusted_volume = market.volume.copy(deep=True) if market.volume is not None else None
    if adjusted_volume is not None:
        adjusted_volume.index = market_index
    selected_payload: list[dict[str, object]] = []
    used_fx_pairs: set[tuple[str, str]] = set()
    fx_routes = 0
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
            converted_cash_amount = row.cash_amount
            fx_payload: dict[str, object] = {}
            if row.currency != price_currency:
                if fx_book is None:  # guarded above; retained as a fail-closed invariant
                    raise ValueError("foreign-currency cash dividend has no FX-rate book")
                fx_rate, fx_legs = fx_book.resolve(
                    row.currency,
                    price_currency,
                    reference_at,
                    fx_age,
                    fx_bridge_currencies,
                )
                fx_routes += 1
                used_fx_pairs.update(
                    (leg["base_currency"], leg["quote_currency"]) for leg in fx_legs
                )
                converted_cash_amount = row.cash_amount * fx_rate
                fx_payload = {
                    "fx_rate": fx_rate,
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
            if converted_cash_amount >= reference_close:
                raise ValueError("cash dividend must be smaller than its reference close")
            factor = (reference_close - converted_cash_amount) / reference_close
            adjusted_close.loc[historical] = adjusted_close.loc[historical] * factor
            payload.update(
                {
                    "cash_amount": row.cash_amount,
                    "currency": row.currency,
                    "reference_at": reference_at.isoformat(),
                    "reference_close": reference_close,
                    "back_adjustment_factor": factor,
                    **fx_payload,
                }
            )
        selected_payload.append(payload)

    if fx_book is not None and fx_book.supplied_pairs != used_fx_pairs:
        raise ValueError("FX-rate CSV contains unused or incomplete declared route pairs")

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
        fx_source_name=fx_book.source_name if fx_book else None,
        fx_max_age_seconds=fx_age.total_seconds() if fx_book else None,
        fx_bridge_currencies=fx_bridge_currencies if fx_book else (),
        fx_routes=fx_routes,
        fx_raw_rows=fx_book.raw_rows if fx_book else 0,
        fx_eligible_revisions=fx_book.eligible_revisions if fx_book else 0,
        fx_selected_observations=len(fx_book.selected) if fx_book else 0,
        fx_excluded_unavailable_revisions=(
            fx_book.excluded_unavailable_revisions if fx_book else 0
        ),
        fx_superseded_revisions=fx_book.superseded_revisions if fx_book else 0,
        fx_lineage_sha256=fx_book.lineage_sha256 if fx_book else None,
        excluded_unavailable_revisions=int((~eligible_mask).sum()),
        excluded_not_effective_actions=int((~effective_mask).sum()),
        superseded_revisions=len(eligible) - len(selected),
        unadjusted_market_sha256=market.sha256,
        adjusted_market_sha256=adjusted_sha256,
        selected_actions_sha256=_digest(selected_payload),
        lineage_sha256=_digest(
            {
                "schema_version": 4,
                "as_of": cutoff.isoformat(),
                "symbol": symbol,
                "price_currency": price_currency,
                "fx_max_age_seconds": fx_age.total_seconds() if fx_book else None,
                "fx_bridge_currencies": list(fx_bridge_currencies) if fx_book else [],
                "fx_lineage_sha256": fx_book.lineage_sha256 if fx_book else None,
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
    fx_rates_path: str | Path | None = None,
    max_fx_age: object = "7 days",
    fx_bridge_currencies: tuple[str, ...] = (),
) -> MarketData:
    """Apply point-in-time splits and optionally FX-converted cash dividends."""

    return _apply_point_in_time_actions(
        market,
        actions_path,
        as_of=as_of,
        symbol=symbol,
        price_currency=price_currency,
        fx_rates_path=fx_rates_path,
        max_fx_age=max_fx_age,
        fx_bridge_currencies=fx_bridge_currencies,
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
        fx_rates_path=None,
        max_fx_age="7 days",
        fx_bridge_currencies=(),
        allowed_action_types=frozenset({"split"}),
    )
