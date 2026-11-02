"""Point-in-time volatility regimes with versioned definition evidence."""

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from statistics import stdev

VOLATILITY_REGIME_DEFINITION_SCHEMA = "finforecast-volatility-regime-definition/v1"
VOLATILITY_REGIME_ASSIGNMENT_SCHEMA = "finforecast-volatility-regime-assignment/v1"
LOW_VOLATILITY = "low-volatility"
NORMAL_VOLATILITY = "normal-volatility"
HIGH_VOLATILITY = "high-volatility"

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MAX_RETURNS = 1_000_000
_MAX_ISSUANCES = 100_000
_MAX_WINDOW_WORK = 10_000_000


class RegimeLabelingError(ValueError):
    """The volatility definition or point-in-time source history is invalid."""

    code = "REGIME_LABELING_INVALID"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise RegimeLabelingError("evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RegimeLabelingError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RegimeLabelingError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RegimeLabelingError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class VolatilityRegimeDefinition:
    """Immutable trailing-realized-volatility regime policy."""

    definition_id: str
    revision: str
    published_at: datetime
    source_schema_sha256: str
    window_observations: int
    annualization_periods: int
    low_volatility_upper: float
    normal_volatility_upper: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.definition_id, str)
            or _IDENTIFIER_RE.fullmatch(self.definition_id) is None
        ):
            raise RegimeLabelingError("definition_id must be a canonical identifier")
        if not isinstance(self.revision, str) or _REVISION_RE.fullmatch(self.revision) is None:
            raise RegimeLabelingError("revision must be a 40- to 64-character hexadecimal value")
        _utc(self.published_at, "published_at")
        if (
            not isinstance(self.source_schema_sha256, str)
            or _DIGEST_RE.fullmatch(self.source_schema_sha256) is None
        ):
            raise RegimeLabelingError("source_schema_sha256 must be a canonical SHA-256")
        if (
            isinstance(self.window_observations, bool)
            or not isinstance(self.window_observations, int)
            or not 2 <= self.window_observations <= 10_000
        ):
            raise RegimeLabelingError("window_observations must be an integer from 2 through 10000")
        if (
            isinstance(self.annualization_periods, bool)
            or not isinstance(self.annualization_periods, int)
            or not 1 <= self.annualization_periods <= 1_000_000
        ):
            raise RegimeLabelingError(
                "annualization_periods must be an integer from 1 through 1000000"
            )
        low = _finite(self.low_volatility_upper, "low_volatility_upper")
        normal = _finite(self.normal_volatility_upper, "normal_volatility_upper")
        if not 0 < low < normal:
            raise RegimeLabelingError(
                "volatility thresholds must be positive and strictly increasing"
            )

    def body(self) -> dict[str, object]:
        return {
            "schema_version": VOLATILITY_REGIME_DEFINITION_SCHEMA,
            "definition_id": self.definition_id,
            "revision": self.revision,
            "published_at": _timestamp(self.published_at),
            "source_schema_sha256": self.source_schema_sha256,
            "window_observations": self.window_observations,
            "annualization_periods": self.annualization_periods,
            "low_volatility_upper": float(self.low_volatility_upper),
            "normal_volatility_upper": float(self.normal_volatility_upper),
            "boundary_policy": "low_if_below_low_normal_if_below_normal_else_high",
            "regimes": [HIGH_VOLATILITY, LOW_VOLATILITY, NORMAL_VOLATILITY],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.body())

    def manifest(self) -> dict[str, object]:
        return {**self.body(), "definition_sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PointInTimeReturn:
    """One return revision and the time it became available."""

    observed_at: datetime
    available_at: datetime
    revision_id: str
    value: float


@dataclass(frozen=True, slots=True)
class VolatilityRegimeAssignment:
    schema_version: str
    issued_at: str
    regime: str
    regime_as_of: str
    realized_volatility: float
    window_observations: int
    window_start: str
    window_end: str
    definition_sha256: str
    source_window_sha256: str
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        document = asdict(self)
        document.pop("evidence_sha256")
        return document

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def regime_as_of_datetime(self) -> datetime:
        """Return the label availability timestamp for coverage-gate integration."""

        return datetime.fromisoformat(self.regime_as_of.replace("Z", "+00:00"))


def _validated_returns(
    rows: Sequence[PointInTimeReturn],
) -> list[tuple[datetime, datetime, str, float]]:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise RegimeLabelingError("returns must be a bounded sequence")
    if not 2 <= len(rows) <= _MAX_RETURNS:
        raise RegimeLabelingError("returns must contain from 2 through 1000000 revisions")
    normalized: list[tuple[datetime, datetime, str, float]] = []
    observed_availability: set[tuple[datetime, datetime]] = set()
    observed_revisions: set[tuple[datetime, str]] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, PointInTimeReturn):
            raise RegimeLabelingError(f"return {position} has an invalid type")
        observed_at = _utc(row.observed_at, f"return {position} observed_at")
        available_at = _utc(row.available_at, f"return {position} available_at")
        if available_at < observed_at:
            raise RegimeLabelingError(f"return {position} is available before observation")
        if (
            not isinstance(row.revision_id, str)
            or _IDENTIFIER_RE.fullmatch(row.revision_id) is None
        ):
            raise RegimeLabelingError(f"return {position} revision_id is invalid")
        availability_key = (observed_at, available_at)
        revision_key = (observed_at, row.revision_id)
        if availability_key in observed_availability or revision_key in observed_revisions:
            raise RegimeLabelingError("return revisions contain an ambiguous duplicate")
        observed_availability.add(availability_key)
        observed_revisions.add(revision_key)
        normalized.append(
            (
                observed_at,
                available_at,
                row.revision_id,
                _finite(row.value, f"return {position} value"),
            )
        )
    if normalized != sorted(normalized, key=lambda item: (item[0], item[1])):
        raise RegimeLabelingError(
            "return revisions must be ordered by observation and availability"
        )
    return normalized


def _validated_issuances(values: Sequence[datetime]) -> list[datetime]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise RegimeLabelingError("issuances must be a bounded sequence")
    if not 1 <= len(values) <= _MAX_ISSUANCES:
        raise RegimeLabelingError("issuances must contain from 1 through 100000 timestamps")
    normalized = [_utc(value, f"issuance {position}") for position, value in enumerate(values)]
    if any(
        current <= previous for previous, current in zip(normalized, normalized[1:], strict=False)
    ):
        raise RegimeLabelingError("issuances must be strictly increasing")
    return normalized


def _activate_return(
    row: tuple[datetime, datetime, str, float],
    latest: dict[datetime, tuple[datetime, datetime, str, float]],
    top_observations: list[datetime],
    window_observations: int,
) -> None:
    observed_at = row[0]
    first_revision = observed_at not in latest
    latest[observed_at] = row
    if not first_revision:
        return
    if len(top_observations) < window_observations:
        heapq.heappush(top_observations, observed_at)
    elif observed_at > top_observations[0]:
        heapq.heapreplace(top_observations, observed_at)


def _select_windows(
    rows: list[tuple[datetime, datetime, str, float]],
    issuances: list[datetime],
    window_observations: int,
) -> list[list[tuple[datetime, datetime, str, float]]]:
    events = sorted(rows, key=lambda row: (row[1], row[0], row[2]))
    event_position = 0
    latest: dict[datetime, tuple[datetime, datetime, str, float]] = {}
    top_observations: list[datetime] = []
    deferred: list[tuple[datetime, datetime, str, float]] = []
    windows: list[list[tuple[datetime, datetime, str, float]]] = []

    for issued_at in issuances:
        for row in deferred:
            _activate_return(row, latest, top_observations, window_observations)
        deferred = []
        while event_position < len(events) and events[event_position][1] <= issued_at:
            row = events[event_position]
            event_position += 1
            if row[0] == issued_at:
                deferred.append(row)
            else:
                _activate_return(row, latest, top_observations, window_observations)
        if len(top_observations) != window_observations:
            raise RegimeLabelingError("insufficient point-in-time return history for issuance")
        windows.append([latest[key] for key in sorted(top_observations)])
    return windows


def _window_digest(rows: list[tuple[datetime, datetime, str, float]]) -> str:
    return _digest(
        [
            {
                "observed_at": _timestamp(observed_at),
                "available_at": _timestamp(available_at),
                "revision_id": revision_id,
                "value_hex": value.hex(),
            }
            for observed_at, available_at, revision_id, value in rows
        ]
    )


def assign_point_in_time_volatility_regimes(
    returns: Sequence[PointInTimeReturn],
    issuances: Sequence[datetime],
    definition: VolatilityRegimeDefinition,
    *,
    expected_definition_sha256: str,
) -> tuple[VolatilityRegimeAssignment, ...]:
    """Assign trailing-volatility regimes using only revisions available at issuance."""

    if not isinstance(definition, VolatilityRegimeDefinition):
        raise TypeError("definition must be a VolatilityRegimeDefinition")
    if (
        not isinstance(expected_definition_sha256, str)
        or _DIGEST_RE.fullmatch(expected_definition_sha256) is None
        or not hmac.compare_digest(definition.sha256, expected_definition_sha256)
    ):
        raise RegimeLabelingError("definition does not match the published digest")
    normalized_returns = _validated_returns(returns)
    normalized_issuances = _validated_issuances(issuances)
    if len(normalized_issuances) * definition.window_observations > _MAX_WINDOW_WORK:
        raise RegimeLabelingError("issuance-window evidence work exceeds the bounded limit")
    published_at = _utc(definition.published_at, "published_at")
    if published_at > normalized_issuances[0]:
        raise RegimeLabelingError("regime definition was not published at issuance")
    windows = _select_windows(
        normalized_returns,
        normalized_issuances,
        definition.window_observations,
    )

    assignments: list[VolatilityRegimeAssignment] = []
    for issued_at, selected in zip(normalized_issuances, windows, strict=True):
        realized_volatility = stdev(row[3] for row in selected) * math.sqrt(
            definition.annualization_periods
        )
        if realized_volatility < definition.low_volatility_upper:
            regime = LOW_VOLATILITY
        elif realized_volatility < definition.normal_volatility_upper:
            regime = NORMAL_VOLATILITY
        else:
            regime = HIGH_VOLATILITY
        document = {
            "schema_version": VOLATILITY_REGIME_ASSIGNMENT_SCHEMA,
            "issued_at": _timestamp(issued_at),
            "regime": regime,
            "regime_as_of": _timestamp(max(row[1] for row in selected)),
            "realized_volatility": realized_volatility,
            "window_observations": len(selected),
            "window_start": _timestamp(selected[0][0]),
            "window_end": _timestamp(selected[-1][0]),
            "definition_sha256": definition.sha256,
            "source_window_sha256": _window_digest(selected),
        }
        assignments.append(
            VolatilityRegimeAssignment(
                **document,
                evidence_sha256=_digest(document),
            )
        )
    return tuple(assignments)
