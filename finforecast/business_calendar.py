from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

CALENDAR_SCHEMA_VERSION = 1
LOCAL_CALENDAR_SCHEMA_VERSION = 2
PINNED_TZDATA_VERSION = "2026.6"
SUPPORTED_ROLL_CONVENTIONS = frozenset({"FOLLOWING", "MODIFIED_FOLLOWING", "PRECEDING"})
_UTC_CALENDAR_KEYS = frozenset(
    {
        "schema_version",
        "calendar_id",
        "revision_id",
        "timezone",
        "valid_from",
        "valid_through",
        "weekend_days",
        "holidays",
    }
)
_LOCAL_CALENDAR_KEYS = _UTC_CALENDAR_KEYS | {"tzdata_version"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_ZONE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._+-]+$")
_MAX_ADJUSTMENT_DAYS = 366


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("settlement business-calendar JSON contains duplicate keys")
        result[key] = value
    return result


def _date(value: object, field: str) -> date:
    if not isinstance(value, str) or _DATE_RE.fullmatch(value) is None:
        raise ValueError(f"business-calendar {field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"business-calendar {field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"business-calendar {field} must be a canonical ISO date")
    return parsed


@dataclass(frozen=True)
class BusinessCalendar:
    calendar_id: str
    revision_id: str
    timezone: str
    tzdata_version: str | None
    valid_from: date
    valid_through: date
    weekend_days: frozenset[int]
    holidays: frozenset[date]
    sha256: str
    _zone: ZoneInfo | None = field(default=None, repr=False, compare=False)

    def _covered(self, value: date) -> None:
        if not self.valid_from <= value <= self.valid_through:
            raise ValueError("settlement date falls outside business-calendar coverage")

    def _is_business_day(self, value: date) -> bool:
        return value.weekday() not in self.weekend_days and value not in self.holidays

    def _search(self, value: date, step: int) -> date:
        candidate = value
        for _ in range(_MAX_ADJUSTMENT_DAYS):
            self._covered(candidate)
            if self._is_business_day(candidate):
                return candidate
            candidate += timedelta(days=step)
        raise ValueError("business-calendar adjustment exceeds safety bound")

    def adjust(self, timestamp: pd.Timestamp, roll_convention: str) -> pd.Timestamp:
        """Adjust by the governed calendar date while preserving its wall-clock time."""

        if roll_convention not in SUPPORTED_ROLL_CONVENTIONS:
            raise ValueError("unsupported settlement business-day roll convention")
        if not isinstance(timestamp, pd.Timestamp) or timestamp.tzinfo is None:
            raise ValueError("business-calendar timestamps must be timezone-aware")
        value = timestamp.tz_convert("UTC")
        calendar_value = value if self._zone is None else value.tz_convert(self._zone)
        original = calendar_value.date()
        self._covered(original)
        if self._is_business_day(original):
            return value

        if roll_convention == "PRECEDING":
            adjusted = self._search(original, -1)
        else:
            adjusted = self._search(original, 1)
            if roll_convention == "MODIFIED_FOLLOWING" and adjusted.month != original.month:
                adjusted = self._search(original, -1)
        if self._zone is None:
            return value + pd.Timedelta(days=(adjusted - original).days)

        target = calendar_value.tz_localize(None) + pd.Timedelta(days=(adjusted - original).days)
        try:
            localized = target.tz_localize(
                self._zone,
                ambiguous="raise",
                nonexistent="raise",
            )
        except ValueError as exc:
            raise ValueError(
                "business-calendar adjustment produces ambiguous or nonexistent local time"
            ) from exc
        return localized.tz_convert("UTC")


def _load_pinned_zone(timezone: object, tzdata_version: object) -> tuple[str, ZoneInfo]:
    if not isinstance(timezone, str) or not 1 <= len(timezone) <= 255:
        raise ValueError("business-calendar timezone must be a canonical IANA name")
    segments = timezone.split("/")
    if (
        len(segments) < 2
        or any(segment in {"", ".", ".."} for segment in segments)
        or any(_ZONE_SEGMENT_RE.fullmatch(segment) is None for segment in segments)
    ):
        raise ValueError("business-calendar timezone must be a canonical IANA name")
    if tzdata_version != PINNED_TZDATA_VERSION:
        raise ValueError(f"business-calendar tzdata_version must be {PINNED_TZDATA_VERSION}")
    try:
        installed_version = importlib.metadata.version("tzdata")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError("pinned tzdata package is not installed") from exc
    if installed_version != PINNED_TZDATA_VERSION:
        raise ValueError("installed tzdata package does not match the governed version")
    resource = importlib.resources.files("tzdata.zoneinfo").joinpath(*segments)
    try:
        with resource.open("rb") as handle:
            zone = ZoneInfo.from_file(handle, key=timezone)
    except (FileNotFoundError, IsADirectoryError, OSError, ZoneInfoNotFoundError) as exc:
        raise ValueError("business-calendar timezone is not present in pinned tzdata") from exc
    return timezone, zone


def load_business_calendar(path: str | Path) -> BusinessCalendar:
    source = Path(path)
    if not source.is_file():
        raise ValueError(
            f"settlement business-calendar JSON does not exist or is not a file: {source}"
        )
    try:
        artifact = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read settlement business-calendar JSON") from exc
    if not isinstance(artifact, dict):
        raise ValueError("settlement business-calendar JSON does not match schema")
    schema_version = artifact.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("unsupported settlement business-calendar schema version")
    expected_keys = {
        CALENDAR_SCHEMA_VERSION: _UTC_CALENDAR_KEYS,
        LOCAL_CALENDAR_SCHEMA_VERSION: _LOCAL_CALENDAR_KEYS,
    }.get(schema_version)
    if expected_keys is None:
        raise ValueError("unsupported settlement business-calendar schema version")
    if set(artifact) != expected_keys:
        raise ValueError("settlement business-calendar JSON does not match schema")
    calendar_id = artifact["calendar_id"]
    revision_id = artifact["revision_id"]
    if any(
        not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None
        for value in (calendar_id, revision_id)
    ):
        raise ValueError("settlement business-calendar identifiers are invalid")
    timezone: str
    tzdata_version: str | None
    zone: ZoneInfo | None
    if schema_version == CALENDAR_SCHEMA_VERSION:
        if artifact["timezone"] != "UTC":
            raise ValueError("schema-v1 settlement business-calendar timezone must be UTC")
        timezone = "UTC"
        tzdata_version = None
        zone = None
    else:
        timezone, zone = _load_pinned_zone(artifact["timezone"], artifact["tzdata_version"])
        tzdata_version = PINNED_TZDATA_VERSION

    valid_from = _date(artifact["valid_from"], "valid_from")
    valid_through = _date(artifact["valid_through"], "valid_through")
    if valid_from > valid_through:
        raise ValueError("business-calendar coverage is reversed")
    weekend_days = artifact["weekend_days"]
    if (
        not isinstance(weekend_days, list)
        or any(type(value) is not int or not 0 <= value <= 6 for value in weekend_days)
        or weekend_days != sorted(set(weekend_days))
        or len(weekend_days) == 7
    ):
        raise ValueError("business-calendar weekend_days must be sorted unique weekdays")
    holidays = artifact["holidays"]
    if (
        not isinstance(holidays, list)
        or any(not isinstance(value, str) for value in holidays)
        or holidays != sorted(set(holidays))
    ):
        raise ValueError("business-calendar holidays must be sorted and unique")
    parsed_holidays = [_date(value, "holiday") for value in holidays]
    if any(not valid_from <= value <= valid_through for value in parsed_holidays):
        raise ValueError("business-calendar holiday falls outside coverage")

    return BusinessCalendar(
        calendar_id=calendar_id,
        revision_id=revision_id,
        timezone=timezone,
        tzdata_version=tzdata_version,
        valid_from=valid_from,
        valid_through=valid_through,
        weekend_days=frozenset(weekend_days),
        holidays=frozenset(parsed_holidays),
        sha256=hashlib.sha256(_canonical_bytes(artifact)).hexdigest(),
        _zone=zone,
    )
