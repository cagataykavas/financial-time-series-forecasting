from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from finforecast.business_calendar import SUPPORTED_ROLL_CONVENTIONS, BusinessCalendar

CALENDAR_BEHAVIOR_SCHEMA = "finforecast-calendar-behavior/v2"
CALENDAR_MIGRATION_SCHEMA = "finforecast-calendar-migration/v1"
PROBE_UTC_HOURS = (0, 6, 12, 18)
TRANSITION_SCAN_STEP_SECONDS = 900
TRANSITION_PROBE_OFFSETS_SECONDS = (-1, 0, 1)
_MAX_COVERAGE_DAYS = 3_660
_MAX_APPROVED_CHANGED_DAYS = 366
_MAX_TRANSITIONS = 512
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "calendar_id",
        "revision_id",
        "timezone",
        "tzdata_version",
        "valid_from",
        "valid_through",
        "probe_utc_hours",
        "transition_scan_step_seconds",
        "transition_probe_offsets_seconds",
        "calendar_rules_sha256",
        "day_digests",
        "transition_digests",
        "manifest_sha256",
    }
)


class CalendarMigrationError(ValueError):
    """A calendar behavior manifest or migration decision is invalid."""

    code = "CALENDAR_MIGRATION_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CalendarMigrationError("document is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalendarMigrationError("manifest JSON contains duplicate keys")
        result[key] = value
    return result


def _canonical_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise CalendarMigrationError(f"{field} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CalendarMigrationError(f"{field} must be a canonical ISO date") from exc
    if parsed.isoformat() != value:
        raise CalendarMigrationError(f"{field} must be a canonical ISO date")
    return parsed


def _coverage_days(valid_from: date, valid_through: date) -> int:
    days = (valid_through - valid_from).days + 1
    if not 1 <= days <= _MAX_COVERAGE_DAYS:
        raise CalendarMigrationError("manifest coverage must span from 1 through 3660 days")
    return days


def _rules_document(calendar: BusinessCalendar) -> dict[str, object]:
    return {
        "calendar_id": calendar.calendar_id,
        "timezone": calendar.timezone,
        "valid_from": calendar.valid_from.isoformat(),
        "valid_through": calendar.valid_through.isoformat(),
        "weekend_days": sorted(calendar.weekend_days),
        "holidays": sorted(value.isoformat() for value in calendar.holidays),
    }


def _probe_instant(calendar: BusinessCalendar, instant: pd.Timestamp) -> dict[str, object]:
    local = instant if calendar._zone is None else instant.tz_convert(calendar._zone)
    local_date = local.date()
    covered = calendar.valid_from <= local_date <= calendar.valid_through
    adjustments: dict[str, str] = {}
    for convention in sorted(SUPPORTED_ROLL_CONVENTIONS):
        try:
            adjustments[convention] = calendar.adjust(instant, convention).isoformat()
        except ValueError:
            adjustments[convention] = "REJECTED"
    offset = local.utcoffset()
    if offset is None:
        raise CalendarMigrationError("calendar timezone produced no UTC offset")
    return {
        "instant_utc": instant.isoformat(),
        "local_timestamp": local.isoformat(),
        "local_date": local_date.isoformat(),
        "utc_offset_seconds": int(offset.total_seconds()),
        "fold": int(local.fold),
        "business_day": calendar._is_business_day(local_date) if covered else None,
        "adjustments": adjustments,
    }


def _probe_day(calendar: BusinessCalendar, probe_date: date) -> str:
    return _digest(
        [
            _probe_instant(
                calendar,
                pd.Timestamp(datetime.combine(probe_date, time(hour), tzinfo=UTC)),
            )
            for hour in PROBE_UTC_HOURS
        ]
    )


def _offset_seconds(calendar: BusinessCalendar, instant: datetime) -> int:
    if calendar._zone is None:
        return 0
    offset = instant.astimezone(calendar._zone).utcoffset()
    if offset is None:
        raise CalendarMigrationError("calendar timezone produced no UTC offset")
    return int(offset.total_seconds())


def _locate_transition(
    calendar: BusinessCalendar,
    left: datetime,
    right: datetime,
    left_offset: int,
) -> datetime:
    while (right - left).total_seconds() > 1:
        middle = left + timedelta(seconds=int((right - left).total_seconds()) // 2)
        if _offset_seconds(calendar, middle) == left_offset:
            left = middle
        else:
            right = middle
    if _offset_seconds(calendar, right - timedelta(seconds=1)) == _offset_seconds(calendar, right):
        raise CalendarMigrationError("timezone transition boundary could not be isolated")
    return right


def _transition_scan_bounds(valid_from: date, valid_through: date) -> tuple[datetime, datetime]:
    try:
        start_date = valid_from - timedelta(days=2)
        end_date = valid_through + timedelta(days=3)
    except OverflowError as exc:
        raise CalendarMigrationError(
            "calendar coverage cannot support transition scanning"
        ) from exc
    return (
        datetime.combine(start_date, time(), tzinfo=UTC),
        datetime.combine(end_date, time(), tzinfo=UTC),
    )


def _calendar_transitions(calendar: BusinessCalendar) -> tuple[datetime, ...]:
    if calendar._zone is None:
        return ()
    start, end = _transition_scan_bounds(calendar.valid_from, calendar.valid_through)
    step = timedelta(seconds=TRANSITION_SCAN_STEP_SECONDS)
    previous = start
    previous_offset = _offset_seconds(calendar, previous)
    cursor = previous + step
    transitions: list[datetime] = []
    while cursor <= end:
        current_offset = _offset_seconds(calendar, cursor)
        if current_offset != previous_offset:
            transition = _locate_transition(calendar, previous, cursor, previous_offset)
            before_local = (transition - timedelta(seconds=1)).astimezone(calendar._zone).date()
            after_local = transition.astimezone(calendar._zone).date()
            if any(
                calendar.valid_from <= local_date <= calendar.valid_through
                for local_date in (before_local, after_local)
            ):
                transitions.append(transition)
                if len(transitions) > _MAX_TRANSITIONS:
                    raise CalendarMigrationError("calendar contains too many timezone transitions")
        previous = cursor
        previous_offset = current_offset
        cursor += step
    return tuple(transitions)


def _probe_transition(calendar: BusinessCalendar, transition: datetime) -> str:
    return _digest(
        [
            _probe_instant(
                calendar,
                pd.Timestamp(transition + timedelta(seconds=offset)),
            )
            for offset in TRANSITION_PROBE_OFFSETS_SECONDS
        ]
    )


def build_calendar_behavior_manifest(calendar: BusinessCalendar) -> dict[str, object]:
    """Commit a bounded calendar's observable behavior for a future migration audit."""

    if not isinstance(calendar, BusinessCalendar):
        raise TypeError("calendar must be a BusinessCalendar")
    coverage_days = _coverage_days(calendar.valid_from, calendar.valid_through)
    if calendar._zone is not None:
        _transition_scan_bounds(calendar.valid_from, calendar.valid_through)
    day_digests: list[dict[str, str]] = []
    for offset in range(coverage_days):
        probe_date = calendar.valid_from + timedelta(days=offset)
        day_digests.append(
            {
                "date": probe_date.isoformat(),
                "sha256": _probe_day(calendar, probe_date),
            }
        )
    transition_digests = [
        {
            "transition_utc": transition.isoformat(),
            "local_dates": sorted(
                {
                    local_date.isoformat()
                    for local_date in (
                        (transition - timedelta(seconds=1)).astimezone(calendar._zone).date(),
                        transition.astimezone(calendar._zone).date(),
                    )
                    if calendar.valid_from <= local_date <= calendar.valid_through
                }
            ),
            "sha256": _probe_transition(calendar, transition),
        }
        for transition in _calendar_transitions(calendar)
    ]
    document: dict[str, object] = {
        "schema_version": CALENDAR_BEHAVIOR_SCHEMA,
        "calendar_id": calendar.calendar_id,
        "revision_id": calendar.revision_id,
        "timezone": calendar.timezone,
        "tzdata_version": calendar.tzdata_version,
        "valid_from": calendar.valid_from.isoformat(),
        "valid_through": calendar.valid_through.isoformat(),
        "probe_utc_hours": list(PROBE_UTC_HOURS),
        "transition_scan_step_seconds": TRANSITION_SCAN_STEP_SECONDS,
        "transition_probe_offsets_seconds": list(TRANSITION_PROBE_OFFSETS_SECONDS),
        "calendar_rules_sha256": _digest(_rules_document(calendar)),
        "day_digests": day_digests,
        "transition_digests": transition_digests,
    }
    document["manifest_sha256"] = _digest(document)
    return document


def _validated_manifest(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise CalendarMigrationError("calendar behavior manifest does not match schema")
    if value.get("schema_version") != CALENDAR_BEHAVIOR_SCHEMA:
        raise CalendarMigrationError("unsupported calendar behavior manifest schema")
    for field in ("calendar_id", "revision_id"):
        item = value.get(field)
        if not isinstance(item, str) or _IDENTIFIER_RE.fullmatch(item) is None:
            raise CalendarMigrationError(f"manifest {field} is invalid")
    timezone_name = value.get("timezone")
    if (
        not isinstance(timezone_name, str)
        or not 1 <= len(timezone_name) <= 255
        or any(ord(character) < 32 for character in timezone_name)
    ):
        raise CalendarMigrationError("manifest timezone is invalid")
    tzdata_version = value.get("tzdata_version")
    if tzdata_version is not None and (
        not isinstance(tzdata_version, str) or _IDENTIFIER_RE.fullmatch(tzdata_version) is None
    ):
        raise CalendarMigrationError("manifest tzdata_version is invalid")
    valid_from = _canonical_date(value.get("valid_from"), "manifest valid_from")
    valid_through = _canonical_date(value.get("valid_through"), "manifest valid_through")
    coverage_days = _coverage_days(valid_from, valid_through)
    probe_hours = value.get("probe_utc_hours")
    if probe_hours != list(PROBE_UTC_HOURS):
        raise CalendarMigrationError("manifest probe schedule is unsupported")
    if value.get("transition_scan_step_seconds") != TRANSITION_SCAN_STEP_SECONDS:
        raise CalendarMigrationError("manifest transition scan step is unsupported")
    if value.get("transition_probe_offsets_seconds") != list(TRANSITION_PROBE_OFFSETS_SECONDS):
        raise CalendarMigrationError("manifest transition probe schedule is unsupported")
    for field in ("calendar_rules_sha256", "manifest_sha256"):
        item = value.get(field)
        if not isinstance(item, str) or _DIGEST_RE.fullmatch(item) is None:
            raise CalendarMigrationError(f"manifest {field} is invalid")
    day_digests = value.get("day_digests")
    if not isinstance(day_digests, list) or len(day_digests) != coverage_days:
        raise CalendarMigrationError("manifest day digests do not cover the exact date range")
    expected_date = valid_from
    for row in day_digests:
        if not isinstance(row, dict) or set(row) != {"date", "sha256"}:
            raise CalendarMigrationError("manifest day digest row is invalid")
        if row.get("date") != expected_date.isoformat():
            raise CalendarMigrationError("manifest day digests are not contiguous")
        digest = row.get("sha256")
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise CalendarMigrationError("manifest day digest is invalid")
        expected_date += timedelta(days=1)
    transition_digests = value.get("transition_digests")
    if not isinstance(transition_digests, list) or len(transition_digests) > _MAX_TRANSITIONS:
        raise CalendarMigrationError("manifest transition digests are not bounded")
    previous_transition: datetime | None = None
    earliest, latest = _transition_scan_bounds(valid_from, valid_through)
    for row in transition_digests:
        if not isinstance(row, dict) or set(row) != {"transition_utc", "local_dates", "sha256"}:
            raise CalendarMigrationError("manifest transition digest row is invalid")
        transition_value = row.get("transition_utc")
        if not isinstance(transition_value, str):
            raise CalendarMigrationError("manifest transition timestamp is invalid")
        try:
            transition = datetime.fromisoformat(transition_value)
        except ValueError as exc:
            raise CalendarMigrationError("manifest transition timestamp is invalid") from exc
        if (
            transition.tzinfo != UTC
            or transition.microsecond != 0
            or transition.isoformat() != transition_value
            or not earliest <= transition < latest
            or (previous_transition is not None and transition <= previous_transition)
        ):
            raise CalendarMigrationError("manifest transition timestamps are not canonical")
        digest = row.get("sha256")
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise CalendarMigrationError("manifest transition digest is invalid")
        local_dates = row.get("local_dates")
        if not isinstance(local_dates, list) or not 1 <= len(local_dates) <= 2:
            raise CalendarMigrationError("manifest transition local dates are invalid")
        parsed_local_dates = [
            _canonical_date(local_date, "manifest transition local date")
            for local_date in local_dates
        ]
        if local_dates != sorted(set(local_dates)) or any(
            not valid_from <= local_date <= valid_through for local_date in parsed_local_dates
        ):
            raise CalendarMigrationError("manifest transition local dates are invalid")
        previous_transition = transition
    document = dict(value)
    claimed = document.pop("manifest_sha256")
    if claimed != _digest(document):
        raise CalendarMigrationError("calendar behavior manifest digest mismatch")
    return value


def dump_calendar_behavior_manifest(manifest: Mapping[str, object]) -> bytes:
    validated = _validated_manifest(dict(manifest))
    return _canonical_bytes(validated) + b"\n"


def load_calendar_behavior_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise CalendarMigrationError("calendar behavior manifest does not exist or is not a file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalendarMigrationError("cannot read calendar behavior manifest") from exc
    return _validated_manifest(value)


def _approved_dates(
    values: Sequence[str], valid_from: date, valid_through: date
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("approved_changed_dates must be a sequence of ISO dates")
    if len(values) > _MAX_APPROVED_CHANGED_DAYS:
        raise CalendarMigrationError("too many approved behavior-change dates")
    approved: list[str] = []
    for value in values:
        parsed = _canonical_date(value, "approved behavior-change date")
        if not valid_from <= parsed <= valid_through:
            raise CalendarMigrationError("approved behavior-change date is outside coverage")
        approved.append(parsed.isoformat())
    if approved != sorted(set(approved)):
        raise CalendarMigrationError("approved behavior-change dates must be sorted and unique")
    return tuple(approved)


def audit_calendar_migration(
    previous: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    approved_changed_dates: Sequence[str] = (),
) -> dict[str, object]:
    """Compare manifests and fail closed on unreviewed timezone behavior drift."""

    old = _validated_manifest(dict(previous))
    new = _validated_manifest(dict(candidate))
    valid_from = _canonical_date(new["valid_from"], "candidate valid_from")
    valid_through = _canonical_date(new["valid_through"], "candidate valid_through")
    approved = _approved_dates(approved_changed_dates, valid_from, valid_through)

    findings: list[str] = []
    comparisons = (
        ("calendar_id", "CALENDAR_ID_MISMATCH"),
        ("timezone", "TIMEZONE_MISMATCH"),
        ("valid_from", "COVERAGE_MISMATCH"),
        ("valid_through", "COVERAGE_MISMATCH"),
        ("probe_utc_hours", "PROBE_SCHEDULE_MISMATCH"),
        ("transition_scan_step_seconds", "TRANSITION_PROBE_POLICY_MISMATCH"),
        ("transition_probe_offsets_seconds", "TRANSITION_PROBE_POLICY_MISMATCH"),
        ("calendar_rules_sha256", "CALENDAR_RULES_CHANGED"),
    )
    for field, code in comparisons:
        if old[field] != new[field] and code not in findings:
            findings.append(code)
    if old["revision_id"] == new["revision_id"]:
        findings.append("REVISION_NOT_CHANGED")
    if old["tzdata_version"] == new["tzdata_version"]:
        findings.append("TZDATA_VERSION_NOT_CHANGED")
    if not isinstance(old["tzdata_version"], str) or not isinstance(new["tzdata_version"], str):
        findings.append("TZDATA_VERSION_MISSING")

    old_days = {row["date"]: row["sha256"] for row in old["day_digests"]}  # type: ignore[index]
    new_days = {row["date"]: row["sha256"] for row in new["day_digests"]}  # type: ignore[index]
    observed = tuple(
        sorted(
            day for day in set(old_days) | set(new_days) if old_days.get(day) != new_days.get(day)
        )
    )
    old_transitions = {
        row["transition_utc"]: (tuple(row["local_dates"]), row["sha256"])
        for row in old["transition_digests"]  # type: ignore[index]
    }
    new_transitions = {
        row["transition_utc"]: (tuple(row["local_dates"]), row["sha256"])
        for row in new["transition_digests"]  # type: ignore[index]
    }
    changed_transitions = tuple(
        sorted(
            instant
            for instant in set(old_transitions) | set(new_transitions)
            if old_transitions.get(instant) != new_transitions.get(instant)
        )
    )
    observed = tuple(
        sorted(
            set(observed)
            | {
                local_date
                for instant in changed_transitions
                for transition_map in (old_transitions, new_transitions)
                if instant in transition_map
                for local_date in transition_map[instant][0]
            }
        )
    )
    unexpected = tuple(sorted(set(observed) - set(approved)))
    missing = tuple(sorted(set(approved) - set(observed)))
    if unexpected:
        findings.append("UNAPPROVED_BEHAVIOR_CHANGE")
    if missing:
        findings.append("APPROVED_BEHAVIOR_CHANGE_NOT_OBSERVED")

    report: dict[str, object] = {
        "schema_version": CALENDAR_MIGRATION_SCHEMA,
        "calendar_id": new["calendar_id"],
        "previous_revision_id": old["revision_id"],
        "candidate_revision_id": new["revision_id"],
        "previous_tzdata_version": old["tzdata_version"],
        "candidate_tzdata_version": new["tzdata_version"],
        "previous_manifest_sha256": old["manifest_sha256"],
        "candidate_manifest_sha256": new["manifest_sha256"],
        "approved_changed_dates": list(approved),
        "observed_changed_dates": list(observed),
        "observed_changed_transition_utc": list(changed_transitions),
        "unexpected_changed_dates": list(unexpected),
        "missing_approved_changed_dates": list(missing),
        "accepted": not findings,
        "finding_codes": sorted(findings),
    }
    report["evidence_sha256"] = _digest(report)
    return report


def require_calendar_migration(
    previous: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    approved_changed_dates: Sequence[str] = (),
) -> dict[str, object]:
    report = audit_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=approved_changed_dates,
    )
    if not report["accepted"]:
        findings = report["finding_codes"]
        assert isinstance(findings, list)
        raise CalendarMigrationError("policy failed: " + ",".join(str(item) for item in findings))
    return report
