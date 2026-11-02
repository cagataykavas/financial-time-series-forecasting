from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.regime_coverage import RegimeIntervalObservation
from finforecast.regime_labeling import (
    HIGH_VOLATILITY,
    LOW_VOLATILITY,
    NORMAL_VOLATILITY,
    PointInTimeReturn,
    RegimeLabelingError,
    VolatilityRegimeDefinition,
    assign_point_in_time_volatility_regimes,
)


def _definition(**overrides: object) -> VolatilityRegimeDefinition:
    values = {
        "definition_id": "daily-realized-volatility-v1",
        "revision": "a" * 40,
        "published_at": datetime(2025, 12, 1, tzinfo=UTC),
        "source_schema_sha256": "b" * 64,
        "window_observations": 3,
        "annualization_periods": 1,
        "low_volatility_upper": 0.05,
        "normal_volatility_upper": 0.15,
    }
    values.update(overrides)
    return VolatilityRegimeDefinition(**values)


def _returns(values: list[float]) -> list[PointInTimeReturn]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        PointInTimeReturn(
            observed_at=start + timedelta(days=position),
            available_at=start + timedelta(days=position, hours=1),
            revision_id=f"return-{position}",
            value=value,
        )
        for position, value in enumerate(values)
    ]


def _assign(
    rows: list[PointInTimeReturn],
    issuances: list[datetime],
    definition: VolatilityRegimeDefinition | None = None,
):
    definition = definition or _definition()
    return assign_point_in_time_volatility_regimes(
        rows,
        issuances,
        definition,
        expected_definition_sha256=definition.sha256,
    )


def test_assigns_low_normal_and_high_from_bounded_trailing_windows() -> None:
    rows = _returns([0.0, 0.0, 0.0, -0.1, 0.0, 0.1, -0.2, 0.0, 0.2])
    issuances = [rows[position].available_at + timedelta(minutes=1) for position in (2, 5, 8)]

    assignments = _assign(rows, issuances)

    assert [row.regime for row in assignments] == [
        LOW_VOLATILITY,
        NORMAL_VOLATILITY,
        HIGH_VOLATILITY,
    ]
    assert [row.realized_volatility for row in assignments] == pytest.approx([0.0, 0.1, 0.2])
    assert len({row.source_window_sha256 for row in assignments}) == 3
    assert all(
        row.regime_as_of_datetime <= issued
        for row, issued in zip(assignments, issuances, strict=True)
    )


def test_late_revision_is_ignored_until_it_is_available() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        PointInTimeReturn(start, start + timedelta(hours=1), "first-v1", 0.0),
        PointInTimeReturn(start, start + timedelta(days=3), "first-v2", 1.0),
        PointInTimeReturn(
            start + timedelta(days=1),
            start + timedelta(days=1, hours=1),
            "second-v1",
            0.0,
        ),
    ]
    definition = _definition(window_observations=2)

    first, second = _assign(
        rows,
        [start + timedelta(days=2), start + timedelta(days=4)],
        definition,
    )

    assert first.realized_volatility == 0.0
    assert first.regime == LOW_VOLATILITY
    assert second.realized_volatility == pytest.approx(2**-0.5)
    assert second.regime == HIGH_VOLATILITY
    assert first.source_window_sha256 != second.source_window_sha256


def test_observation_at_issuance_is_deferred_not_discarded() -> None:
    rows = _returns([0.0, 0.0, 1.0])
    definition = _definition(window_observations=2)
    first_issuance = rows[-1].observed_at

    first, second = _assign(
        rows,
        [first_issuance, rows[-1].available_at + timedelta(minutes=1)],
        definition,
    )

    assert first.window_end == rows[1].observed_at.isoformat().replace("+00:00", "Z")
    assert second.window_end == rows[2].observed_at.isoformat().replace("+00:00", "Z")
    assert first.source_window_sha256 != second.source_window_sha256


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.05, NORMAL_VOLATILITY), (0.15, HIGH_VOLATILITY)],
)
def test_threshold_boundaries_follow_manifest_policy(value: float, expected: str) -> None:
    rows = _returns([-value, 0.0, value])
    assignment = _assign(rows, [rows[-1].available_at + timedelta(minutes=1)])[0]

    assert assignment.realized_volatility == pytest.approx(value)
    assert assignment.regime == expected


def test_assignment_evidence_is_canonical_deterministic_and_privacy_reduced() -> None:
    rows = _returns([-0.1, 0.0, 0.1])
    issued_at = rows[-1].available_at + timedelta(minutes=1)

    first = _assign(rows, [issued_at])[0]
    second = _assign(rows, [issued_at])[0]
    body = first.body()
    expected = hashlib.sha256(
        json.dumps(body, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()

    assert first == second
    assert first.evidence_sha256 == expected
    assert json.loads(json.dumps(first.to_audit_dict(), allow_nan=False)) == first.to_audit_dict()
    serialized = json.dumps(first.to_audit_dict(), sort_keys=True)
    assert "return-0" not in serialized
    assert '"value"' not in serialized


def test_assignment_integrates_with_existing_coverage_observation() -> None:
    rows = _returns([-0.1, 0.0, 0.1])
    issued_at = rows[-1].available_at + timedelta(minutes=1)
    assignment = _assign(rows, [issued_at])[0]

    observation = RegimeIntervalObservation(
        issued_at=issued_at,
        observed_at=issued_at + timedelta(days=1),
        regime=assignment.regime,
        regime_as_of=assignment.regime_as_of_datetime,
        lower=-1.0,
        upper=1.0,
        actual=0.0,
    )

    assert observation.regime_as_of <= observation.issued_at


def test_rejects_definition_not_published_at_issuance_or_wrong_digest() -> None:
    rows = _returns([0.0, 0.0, 0.0])
    issued_at = rows[-1].available_at + timedelta(minutes=1)
    unpublished = _definition(published_at=issued_at + timedelta(seconds=1))

    with pytest.raises(RegimeLabelingError, match="not published"):
        _assign(rows, [issued_at], unpublished)
    with pytest.raises(RegimeLabelingError, match="published digest"):
        assign_point_in_time_volatility_regimes(
            rows,
            [issued_at],
            _definition(),
            expected_definition_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"definition_id": "bad id"}, "canonical identifier"),
        ({"revision": "abc"}, "hexadecimal"),
        ({"source_schema_sha256": "abc"}, "canonical SHA-256"),
        ({"window_observations": True}, "integer"),
        ({"window_observations": 1}, "from 2"),
        ({"annualization_periods": 0}, "from 1"),
        ({"low_volatility_upper": float("nan")}, "finite"),
        ({"normal_volatility_upper": 0.01}, "strictly increasing"),
    ],
)
def test_rejects_ambiguous_definition(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(RegimeLabelingError, match=message):
        _definition(**overrides)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda rows: [
                replace(rows[0], observed_at=rows[0].observed_at.replace(tzinfo=None)),
                *rows[1:],
            ],
            "timezone-aware",
        ),
        (
            lambda rows: [
                replace(
                    rows[0],
                    available_at=rows[0].observed_at - timedelta(seconds=1),
                ),
                *rows[1:],
            ],
            "available before",
        ),
        (lambda rows: [replace(rows[0], value=float("inf")), *rows[1:]], "finite"),
        (lambda rows: [rows[1], rows[0], *rows[2:]], "ordered"),
        (
            lambda rows: [
                rows[0],
                replace(rows[0], revision_id="duplicate"),
                *rows[2:],
            ],
            "ambiguous duplicate",
        ),
    ],
)
def test_rejects_malformed_or_ambiguous_return_history(mutator, message: str) -> None:
    rows = _returns([0.0, 0.0, 0.0])

    with pytest.raises(RegimeLabelingError, match=message):
        _assign(mutator(rows), [rows[-1].available_at + timedelta(minutes=1)])


def test_rejects_insufficient_history_future_observations_and_bad_issuances() -> None:
    rows = _returns([0.0, 0.0, 0.0])

    with pytest.raises(RegimeLabelingError, match="insufficient"):
        _assign(rows, [rows[-1].observed_at])
    with pytest.raises(RegimeLabelingError, match="strictly increasing"):
        _assign(
            rows,
            [
                rows[-1].available_at + timedelta(hours=2),
                rows[-1].available_at + timedelta(hours=1),
            ],
        )
    with pytest.raises(RegimeLabelingError, match="timezone-aware"):
        _assign(rows, [rows[-1].available_at.replace(tzinfo=None)])


def test_rejects_unbounded_issuance_window_work() -> None:
    rows = _returns([0.0, 0.0])
    definition = _definition(window_observations=10_000)
    start = rows[-1].available_at + timedelta(minutes=1)
    issuances = [start + timedelta(minutes=position) for position in range(1_001)]

    with pytest.raises(RegimeLabelingError, match="evidence work exceeds"):
        _assign(rows, issuances, definition)
