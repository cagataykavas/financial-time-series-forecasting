from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finforecast.prequential import (
    PrequentialConformalError,
    PrequentialConformalPolicy,
    prequential_conformal_intervals,
)


def policy(**overrides: object) -> PrequentialConformalPolicy:
    values: dict[str, object] = {
        "coverage": 0.8,
        "window": 4,
        "min_calibration": 2,
        "feedback_delay": 0,
        "max_points": 100,
        "max_calibration_operations": 1_000,
    }
    values.update(overrides)
    return PrequentialConformalPolicy(**values)  # type: ignore[arg-type]


def assert_error(code: str, actual: object, predicted: object, **kwargs: object) -> None:
    with pytest.raises(PrequentialConformalError) as error:
        prequential_conformal_intervals(actual, predicted, **kwargs)
    assert error.value.code == code


def test_uses_finite_sample_quantile_and_rolling_window() -> None:
    result = prequential_conformal_intervals(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        policy=policy(),
    )

    assert [interval.position for interval in result.intervals] == [2, 3, 4, 5]
    assert [interval.radius for interval in result.intervals] == [1.0, 2.0, 3.0, 4.0]
    assert result.intervals[-1].calibration_start == 1
    assert result.intervals[-1].calibration_end_exclusive == 5
    assert result.intervals[-1].calibration_size == 4


def test_current_and_future_outcomes_cannot_change_interval() -> None:
    baseline = prequential_conformal_intervals(
        [0.0, 1.0, 2.0, 3.0, 4.0], [0.0] * 5, policy=policy()
    )
    changed = prequential_conformal_intervals(
        [0.0, 1.0, 900.0, 800.0, 700.0], [0.0] * 5, policy=policy()
    )

    assert baseline.intervals[0] == changed.intervals[0]
    assert baseline.intervals[1].radius != changed.intervals[1].radius


def test_feedback_delay_excludes_unavailable_labels() -> None:
    delayed = prequential_conformal_intervals(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0] * 6,
        policy=policy(feedback_delay=2),
    )

    first = delayed.intervals[0]
    assert first.position == 4
    assert first.calibration_end_exclusive == 2
    assert first.radius == 1.0


def test_constant_residual_produces_exact_symmetric_bounds() -> None:
    result = prequential_conformal_intervals(
        [2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0], policy=policy()
    )

    assert result.intervals[0].lower == 2.0
    assert result.intervals[0].upper == 4.0
    assert all(item.lower <= item.point_forecast <= item.upper for item in result.intervals)


def test_deterministic_and_does_not_mutate_inputs() -> None:
    actual = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    predicted = np.array([0.8, 1.4, 2.1, 2.4, 3.2])
    original_actual = actual.copy()
    original_predicted = predicted.copy()

    first = prequential_conformal_intervals(actual, predicted, policy=policy())
    second = prequential_conformal_intervals(actual, predicted, policy=policy())

    assert first == second
    np.testing.assert_array_equal(actual, original_actual)
    np.testing.assert_array_equal(predicted, original_predicted)


def test_series_index_is_validated_and_preserved() -> None:
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    predicted = pd.Series([1.0, 1.0, 2.0, 3.0, 4.0], index=index)

    frame = prequential_conformal_intervals(actual, predicted, policy=policy()).as_frame()

    assert frame.index.tolist() == index[2:].tolist()
    assert frame.index.name == "forecast_time"
    assert frame.columns.tolist() == [
        "position",
        "point_forecast",
        "lower",
        "upper",
        "radius",
        "calibration_start",
        "calibration_end_exclusive",
        "calibration_size",
    ]


@pytest.mark.parametrize(
    ("actual", "predicted", "code"),
    [
        ([[1.0], [2.0]], [[1.0], [2.0]], "INVALID_SHAPE"),
        ([], [], "EMPTY_INPUT"),
        ([1.0, 2.0], [1.0], "LENGTH_MISMATCH"),
        ([1.0, float("nan"), 2.0], [1.0, 1.0, 1.0], "NON_FINITE_INPUT"),
        ([1.0, 2.0, 3.0], [1.0, float("inf"), 3.0], "NON_FINITE_INPUT"),
        (["bad", "data", "here"], [1.0, 2.0, 3.0], "NON_NUMERIC_INPUT"),
    ],
)
def test_rejects_malformed_vectors(actual: object, predicted: object, code: str) -> None:
    assert_error(code, actual, predicted, policy=policy())


def test_rejects_series_array_and_index_mismatch() -> None:
    series = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    assert_error("INDEX_TYPE_MISMATCH", series, [1.0, 2.0, 3.0], policy=policy())
    assert_error(
        "INDEX_MISMATCH",
        series,
        pd.Series([1.0, 2.0, 3.0], index=[1, 2, 4]),
        policy=policy(),
    )


def test_rejects_duplicate_and_non_monotonic_index() -> None:
    duplicate = pd.Series([1.0, 2.0, 3.0], index=[1, 1, 2])
    descending = pd.Series([1.0, 2.0, 3.0], index=[3, 2, 1])
    assert_error("DUPLICATE_INDEX", duplicate, duplicate.copy(), policy=policy())
    assert_error("NON_MONOTONIC_INDEX", descending, descending.copy(), policy=policy())


def test_enforces_point_history_and_operation_budgets() -> None:
    assert_error(
        "POINT_BUDGET_EXCEEDED",
        [0.0] * 6,
        [0.0] * 6,
        policy=policy(max_points=5),
    )
    assert_error(
        "INSUFFICIENT_HISTORY",
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        policy=policy(feedback_delay=1),
    )
    assert_error(
        "OPERATION_BUDGET_EXCEEDED",
        [0.0] * 10,
        [0.0] * 10,
        policy=policy(max_calibration_operations=5),
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"coverage": 0.0}, "INVALID_COVERAGE"),
        ({"coverage": float("nan")}, "INVALID_COVERAGE"),
        ({"coverage": True}, "INVALID_COVERAGE"),
        ({"window": 0}, "INVALID_WINDOW"),
        ({"window": True}, "INVALID_WINDOW"),
        ({"min_calibration": 0}, "INVALID_MIN_CALIBRATION"),
        ({"window": 2, "min_calibration": 3}, "MIN_CALIBRATION_EXCEEDS_WINDOW"),
        ({"feedback_delay": -1}, "INVALID_FEEDBACK_DELAY"),
        ({"max_points": 0}, "INVALID_MAX_POINTS"),
        ({"max_calibration_operations": 0}, "INVALID_OPERATION_BUDGET"),
    ],
)
def test_policy_rejects_invalid_values(overrides: dict[str, object], code: str) -> None:
    with pytest.raises(PrequentialConformalError) as error:
        policy(**overrides)
    assert error.value.code == code


def test_reports_exact_calibration_operation_count() -> None:
    result = prequential_conformal_intervals(
        [0.0] * 7,
        [0.0] * 7,
        policy=policy(window=3, min_calibration=2),
    )

    assert result.calibration_operations == 2 + 3 + 3 + 3 + 3
    assert len(result.intervals) == 5
