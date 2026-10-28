from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


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
    identifiers = values.tolist()
    if any(
        not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None
        for value in identifiers
    ):
        raise ValueError(f"column {column!r} contains invalid identifiers")
    return identifiers


@dataclass(frozen=True)
class FxRateBook:
    """Point-in-time FX revisions constrained to explicitly governed routes."""

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
        max_leg_skew: pd.Timedelta,
        bridge_currencies: tuple[str, ...],
        *,
        context: str = "cash action",
    ) -> tuple[float, list[pd.Series], float]:
        def latest(base: str, quote: str) -> tuple[pd.Series | None, str | None]:
            pair = self.selected.loc[
                (self.selected["base_currency"] == base)
                & (self.selected["quote_currency"] == quote)
            ]
            candidates = pair.loc[pair["observed_at"] <= reference_at]
            if candidates.empty:
                return None, "future" if not pair.empty else "missing"
            resolved = candidates.sort_values("observed_at", kind="stable").iloc[-1]
            if reference_at - resolved["observed_at"] > max_age:
                return None, "stale"
            return resolved, None

        direct, direct_error = latest(base_currency, quote_currency)
        if direct is not None:
            return float(direct["rate"]), [direct], 0.0
        if not bridge_currencies:
            if direct_error == "stale":
                raise ValueError(f"{context} FX observation exceeds the maximum age")
            raise ValueError(
                f"{context} requires a point-in-time FX observation at or before "
                "its governed reference timestamp"
            )

        routes: list[tuple[float, list[pd.Series]]] = []
        for bridge in bridge_currencies:
            first, _ = latest(base_currency, bridge)
            second, _ = latest(bridge, quote_currency)
            if first is not None and second is not None:
                routes.append((float(first["rate"] * second["rate"]), [first, second]))
        if len(routes) > 1:
            raise ValueError(f"{context} FX conversion has ambiguous bridge routes")
        if routes:
            rate, legs = routes[0]
            leg_skew = abs(legs[0]["observed_at"] - legs[1]["observed_at"])
            if leg_skew > max_leg_skew:
                raise ValueError(f"{context} FX route legs exceed the maximum timestamp skew")
            return rate, legs, leg_skew.total_seconds()
        raise ValueError(
            f"{context} requires a fresh point-in-time direct or declared bridge FX route"
        )


def load_fx_rate_book(
    path: str | Path,
    *,
    cutoff: pd.Timestamp,
    allowed_pairs: set[tuple[str, str]],
) -> FxRateBook:
    """Load latest-known FX revisions after validating the complete supplied lineage."""

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
    revisions = _identifiers(frame["revision_id"], "revision_id")
    observed_at = _timestamps(frame["observed_at"], "observed_at")
    available_at = _timestamps(frame["available_at"], "available_at")
    if any(
        observed > available for observed, available in zip(observed_at, available_at, strict=True)
    ):
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
    return FxRateBook(
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
