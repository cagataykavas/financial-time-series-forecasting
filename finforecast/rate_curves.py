from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .business_calendar import (
    SUPPORTED_ROLL_CONVENTIONS,
    BusinessCalendar,
    load_business_calendar,
)

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
SUPPORTED_DAY_COUNTS = frozenset({"ACT/360", "ACT/365F"})
UNADJUSTED_CALENDAR_ID = "NONE"
UNADJUSTED_ROLL_CONVENTION = "UNADJUSTED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RATE_CURVE_COLUMNS = frozenset(
    {
        "curve_id",
        "currency",
        "calendar_id",
        "calendar_sha256",
        "roll_convention",
        "day_count",
        "observed_at",
        "available_at",
        "revision_id",
        "tenor_days",
        "zero_rate",
    }
)


def year_fraction(start: pd.Timestamp, end: pd.Timestamp, day_count: str) -> float:
    """Return an actual-time year fraction for a reviewed curve convention."""

    if day_count not in SUPPORTED_DAY_COUNTS:
        raise ValueError(f"unsupported settlement rate-curve day count: {day_count!r}")
    elapsed_seconds = (end - start).total_seconds()
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("settlement rate-curve accrual interval must be finite and positive")
    denominator_days = 360.0 if day_count == "ACT/360" else 365.0
    return elapsed_seconds / (denominator_days * 86_400.0)


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
    result: list[pd.Timestamp] = []
    for raw in values:
        try:
            timestamp = pd.Timestamp(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {column!r} contains invalid timestamps") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"column {column!r} must contain timezone-aware timestamps")
        result.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(result)


@dataclass(frozen=True)
class RateCurveSelection:
    curve_id: str
    calendar_id: str
    calendar_sha256: str | None
    roll_convention: str
    day_count: str
    observed_at: pd.Timestamp
    available_at: pd.Timestamp
    revision_id: str
    lower_tenor_days: float
    upper_tenor_days: float
    zero_rate: float
    observation_age_seconds: float

    def metadata(self) -> dict[str, object]:
        return {
            "curve_id": self.curve_id,
            "calendar_id": self.calendar_id,
            "calendar_sha256": self.calendar_sha256,
            "roll_convention": self.roll_convention,
            "day_count": self.day_count,
            "observed_at": self.observed_at.isoformat(),
            "available_at": self.available_at.isoformat(),
            "revision_id": self.revision_id,
            "lower_tenor_days": self.lower_tenor_days,
            "upper_tenor_days": self.upper_tenor_days,
            "zero_rate": self.zero_rate,
            "observation_age_seconds": self.observation_age_seconds,
        }


@dataclass(frozen=True)
class RateCurveBook:
    source_name: str
    curve_id: str
    currency: str
    calendar_id: str
    calendar_sha256: str | None
    roll_convention: str
    business_calendar: BusinessCalendar | None
    day_count: str
    rows: pd.DataFrame
    cutoff: pd.Timestamp
    raw_rows: int
    eligible_rows: int
    excluded_unavailable_rows: int
    superseded_revisions: int
    lineage_sha256: str

    def resolve(
        self,
        reference_at: pd.Timestamp,
        tenor_days: float,
        max_age: pd.Timedelta,
    ) -> RateCurveSelection:
        if not math.isfinite(tenor_days) or tenor_days <= 0:
            raise ValueError("settlement rate-curve tenor must be finite and positive")
        candidates = self.rows.loc[self.rows["observed_at"] <= reference_at]
        if candidates.empty:
            raise ValueError(
                "settlement rate curve has no observation at or before the cash-flow timestamp"
            )
        observed_at = candidates["observed_at"].max()
        revision_rows = candidates.loc[candidates["observed_at"] == observed_at]
        available_at = revision_rows["available_at"].max()
        curve = revision_rows.loc[revision_rows["available_at"] == available_at].sort_values(
            "tenor_days", kind="stable"
        )
        age = reference_at - observed_at
        if age > max_age:
            raise ValueError("settlement rate-curve observation exceeds the maximum age")

        tenors = curve["tenor_days"].to_numpy(dtype=float)
        rates = curve["zero_rate"].to_numpy(dtype=float)
        if tenor_days < tenors[0] or tenor_days > tenors[-1]:
            raise ValueError("settlement rate-curve tenor requires forbidden extrapolation")
        upper_index = int(tenors.searchsorted(tenor_days, side="left"))
        if tenors[upper_index] == tenor_days:
            lower_index = upper_index
            zero_rate = rates[upper_index]
        else:
            lower_index = upper_index - 1
            weight = (tenor_days - tenors[lower_index]) / (
                tenors[upper_index] - tenors[lower_index]
            )
            zero_rate = rates[lower_index] + weight * (rates[upper_index] - rates[lower_index])
        return RateCurveSelection(
            curve_id=self.curve_id,
            calendar_id=self.calendar_id,
            calendar_sha256=self.calendar_sha256,
            roll_convention=self.roll_convention,
            day_count=self.day_count,
            observed_at=observed_at,
            available_at=available_at,
            revision_id=str(curve.iloc[0]["revision_id"]),
            lower_tenor_days=float(tenors[lower_index]),
            upper_tenor_days=float(tenors[upper_index]),
            zero_rate=float(zero_rate),
            observation_age_seconds=age.total_seconds(),
        )


def load_rate_curve_book(
    path: str | Path,
    *,
    cutoff: pd.Timestamp,
    expected_currency: str,
    calendar_path: str | Path | None = None,
) -> RateCurveBook:
    if not isinstance(expected_currency, str) or CURRENCY_RE.fullmatch(expected_currency) is None:
        raise ValueError("settlement rate-curve currency must be an uppercase ISO 4217 code")
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"settlement rate-curve CSV does not exist or is not a file: {source}")
    try:
        frame = pd.read_csv(source, keep_default_na=False, dtype={"calendar_sha256": str})
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"cannot read settlement rate-curve CSV: {exc}") from exc
    if frame.empty:
        raise ValueError("settlement rate-curve CSV cannot be empty")
    if set(frame.columns) != RATE_CURVE_COLUMNS:
        missing = sorted(RATE_CURVE_COLUMNS.difference(frame.columns))
        unexpected = sorted(set(frame.columns).difference(RATE_CURVE_COLUMNS))
        raise ValueError(
            "settlement rate-curve CSV columns do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )

    curve_ids = frame["curve_id"].tolist()
    revisions = frame["revision_id"].tolist()
    if any(
        not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None
        for value in curve_ids + revisions
    ):
        raise ValueError("settlement rate-curve identifiers are invalid")
    if len(set(curve_ids)) != 1:
        raise ValueError("settlement rate-curve CSV must contain exactly one curve_id")
    currencies = frame["currency"].tolist()
    if any(value != expected_currency for value in currencies):
        raise ValueError("settlement rate curve must use the price currency")
    calendar_ids = frame["calendar_id"].tolist()
    if (
        any(
            not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None
            for value in calendar_ids
        )
        or len(set(calendar_ids)) != 1
    ):
        raise ValueError("settlement rate-curve CSV must contain one valid calendar_id")
    calendar_hashes = frame["calendar_sha256"].tolist()
    roll_conventions = frame["roll_convention"].tolist()
    if len(set(calendar_hashes)) != 1 or len(set(roll_conventions)) != 1:
        raise ValueError("settlement rate-curve CSV must contain one calendar policy")
    calendar_id = calendar_ids[0]
    calendar_sha256 = calendar_hashes[0]
    roll_convention = roll_conventions[0]
    business_calendar: BusinessCalendar | None = None
    if calendar_id == UNADJUSTED_CALENDAR_ID:
        if calendar_sha256 != "" or roll_convention != UNADJUSTED_ROLL_CONVENTION:
            raise ValueError("unadjusted settlement curves require an empty calendar digest")
        if calendar_path is not None:
            raise ValueError(
                "settlement business-calendar JSON is not allowed for calendar_id NONE"
            )
    else:
        if SHA256_RE.fullmatch(calendar_sha256) is None:
            raise ValueError("settlement rate-curve calendar_sha256 must be lowercase SHA-256")
        if roll_convention not in SUPPORTED_ROLL_CONVENTIONS:
            raise ValueError("settlement rate-curve roll_convention is unsupported")
        if calendar_path is None:
            raise ValueError("adjusted settlement curve requires a business-calendar JSON")
        business_calendar = load_business_calendar(calendar_path)
        if business_calendar.calendar_id != calendar_id:
            raise ValueError("settlement business-calendar identifier does not match rate curve")
        if business_calendar.sha256 != calendar_sha256:
            raise ValueError("settlement business-calendar digest does not match rate curve")
    day_counts = frame["day_count"].tolist()
    if any(value not in SUPPORTED_DAY_COUNTS for value in day_counts):
        raise ValueError("settlement rate-curve day_count must be ACT/360 or ACT/365F")
    if len(set(day_counts)) != 1:
        raise ValueError("settlement rate-curve CSV must contain exactly one day_count")

    observed_at = _timestamps(frame["observed_at"], "observed_at")
    available_at = _timestamps(frame["available_at"], "available_at")
    tenor_days = pd.to_numeric(frame["tenor_days"], errors="coerce").astype(float)
    zero_rates = pd.to_numeric(frame["zero_rate"], errors="coerce").astype(float)
    if any(not math.isfinite(value) or value <= 0 for value in tenor_days):
        raise ValueError("settlement rate-curve tenors must be finite and positive")
    if any(not math.isfinite(value) for value in zero_rates):
        raise ValueError("settlement rate-curve zero rates must be finite")
    if (available_at < observed_at).any():
        raise ValueError("settlement rate-curve revisions cannot be available before observation")

    identity = pd.DataFrame(
        {
            "curve_id": curve_ids,
            "currency": currencies,
            "calendar_id": calendar_ids,
            "calendar_sha256": calendar_hashes,
            "roll_convention": roll_conventions,
            "day_count": day_counts,
            "observed_at": observed_at,
            "available_at": available_at,
            "revision_id": revisions,
            "tenor_days": tenor_days,
            "zero_rate": zero_rates,
        }
    )
    order = list(
        zip(
            identity["observed_at"],
            identity["available_at"],
            identity["tenor_days"],
            strict=True,
        )
    )
    if order != sorted(order):
        raise ValueError(
            "settlement rate-curve rows must be ordered by observation, availability and tenor"
        )
    if identity.duplicated(["observed_at", "available_at", "tenor_days"]).any():
        raise ValueError("settlement rate-curve revisions contain duplicate tenors")
    grouped = identity.groupby(["observed_at", "available_at"], sort=False)
    if grouped["revision_id"].nunique().gt(1).any():
        raise ValueError("one settlement curve revision must have one revision_id")
    if grouped.size().lt(2).any():
        raise ValueError("each settlement rate-curve revision requires at least two tenors")
    if identity.groupby(["observed_at", "revision_id"])["available_at"].nunique().gt(1).any():
        raise ValueError("settlement rate-curve revision IDs must identify one availability")
    revision_shapes = grouped["tenor_days"].apply(tuple)
    for observed, shapes in revision_shapes.groupby(level=0):
        if len(set(shapes.tolist())) != 1:
            raise ValueError(
                f"settlement rate-curve revisions change tenor shape at {observed.isoformat()}"
            )

    eligible_mask = identity["available_at"] <= cutoff
    eligible = identity.loc[eligible_mask].copy()
    if eligible.empty:
        raise ValueError("settlement rate curve has no revision known at the as_of cutoff")
    revision_keys = eligible[["observed_at", "available_at"]].drop_duplicates()
    selected_keys = revision_keys.groupby("observed_at", sort=False).tail(1)
    selected = eligible.merge(selected_keys, on=["observed_at", "available_at"], how="inner")
    superseded = len(revision_keys) - len(selected_keys)
    lineage_rows: list[dict[str, Any]] = []
    for row in identity.itertuples(index=False):
        lineage_rows.append(
            {
                "curve_id": row.curve_id,
                "currency": row.currency,
                "calendar_id": row.calendar_id,
                "calendar_sha256": row.calendar_sha256 or None,
                "roll_convention": row.roll_convention,
                "day_count": row.day_count,
                "observed_at": row.observed_at.isoformat(),
                "available_at": row.available_at.isoformat(),
                "revision_id": row.revision_id,
                "tenor_days": row.tenor_days,
                "zero_rate": row.zero_rate,
            }
        )
    return RateCurveBook(
        source_name=source.name,
        curve_id=curve_ids[0],
        currency=expected_currency,
        calendar_id=calendar_ids[0],
        calendar_sha256=calendar_sha256 or None,
        roll_convention=roll_convention,
        business_calendar=business_calendar,
        day_count=day_counts[0],
        rows=selected,
        cutoff=cutoff,
        raw_rows=len(identity),
        eligible_rows=len(eligible),
        excluded_unavailable_rows=int((~eligible_mask).sum()),
        superseded_revisions=superseded,
        lineage_sha256=_digest(
            {"schema_version": 3, "as_of": cutoff.isoformat(), "rows": lineage_rows}
        ),
    )
