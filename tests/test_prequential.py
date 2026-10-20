from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from finforecast.prequential import (
    PrequentialConformalCalibrator,
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


def test_streaming_calibrator_matches_batch_replay() -> None:
    actual = [0.0, 1.0, 4.0, 3.0, 8.0, 5.0, 12.0]
    predicted = [0.0, 0.5, 2.0, 4.0, 5.0, 7.0, 9.0]
    selected_policy = policy(window=3, feedback_delay=1)
    batch = prequential_conformal_intervals(actual, predicted, policy=selected_policy)
    calibrator = PrequentialConformalCalibrator(selected_policy)

    streamed = []
    for position, (outcome, point) in enumerate(zip(actual, predicted, strict=True)):
        interval = calibrator.issue(point)
        if interval is not None:
            streamed.append(interval)
        calibrator.observe(position, outcome)

    assert tuple(streamed) == batch.intervals
    assert calibrator.calibration_operations == batch.calibration_operations


def test_feedback_contract_fails_closed_when_an_outcome_is_overdue() -> None:
    calibrator = PrequentialConformalCalibrator(policy(feedback_delay=2))

    assert calibrator.issue(10.0) is None
    assert calibrator.issue(11.0) is None
    assert calibrator.issue(12.0) is None
    with pytest.raises(PrequentialConformalError) as error:
        calibrator.issue(13.0)
    assert error.value.code == "MISSING_FEEDBACK"
    assert calibrator.next_position == 3

    calibrator.observe(0, 10.5)
    assert calibrator.issue(13.0) is None


def test_feedback_must_be_ordered_and_exactly_once() -> None:
    calibrator = PrequentialConformalCalibrator(policy(feedback_delay=1))
    calibrator.issue(1.0)
    calibrator.issue(2.0)

    with pytest.raises(PrequentialConformalError) as error:
        calibrator.observe(1, 2.5)
    assert error.value.code == "OUT_OF_ORDER_FEEDBACK"

    calibrator.observe(0, 1.5)
    with pytest.raises(PrequentialConformalError) as error:
        calibrator.observe(0, 1.5)
    assert error.value.code == "DUPLICATE_FEEDBACK"

    calibrator.observe(1, 2.5)
    with pytest.raises(PrequentialConformalError) as error:
        calibrator.observe(2, 3.0)
    assert error.value.code == "UNKNOWN_FORECAST"


def test_checkpoint_resume_matches_uninterrupted_stream() -> None:
    selected_policy = policy(window=3, feedback_delay=1)
    actual = [2.0, 2.5, 5.0, 3.0, 8.0, 6.0, 10.0, 8.0]
    predicted = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0]

    uninterrupted = PrequentialConformalCalibrator(selected_policy)
    expected = []
    for position, (outcome, point) in enumerate(zip(actual, predicted, strict=True)):
        interval = uninterrupted.issue(point)
        if interval is not None:
            expected.append(interval)
        uninterrupted.observe(position, outcome)

    resumed = PrequentialConformalCalibrator(selected_policy)
    actual_intervals = []
    for position in range(4):
        interval = resumed.issue(predicted[position])
        if interval is not None:
            actual_intervals.append(interval)
        resumed.observe(position, actual[position])
    checkpoint = resumed.checkpoint()
    resumed = PrequentialConformalCalibrator.restore(checkpoint, policy=selected_policy)
    for position in range(4, len(actual)):
        interval = resumed.issue(predicted[position])
        if interval is not None:
            actual_intervals.append(interval)
        resumed.observe(position, actual[position])

    assert actual_intervals == expected
    assert resumed.checkpoint() == uninterrupted.checkpoint()


def test_checkpoint_preserves_pending_forecasts() -> None:
    selected_policy = policy(feedback_delay=2)
    calibrator = PrequentialConformalCalibrator(selected_policy)
    calibrator.issue(10.0)
    calibrator.issue(11.0)
    calibrator.issue(12.0)

    restored = PrequentialConformalCalibrator.restore(
        calibrator.checkpoint(), policy=selected_policy
    )
    restored.observe(0, 10.5)
    assert restored.issue(13.0) is None


def test_checkpoint_detects_corruption_and_policy_drift() -> None:
    selected_policy = policy()
    calibrator = PrequentialConformalCalibrator(selected_policy)
    calibrator.issue(1.0)
    calibrator.observe(0, 2.0)
    checkpoint = calibrator.checkpoint()

    payload = json.loads(checkpoint)
    payload["state"]["residuals"][0][1] = 999.0
    corrupted = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    with pytest.raises(PrequentialConformalError) as error:
        PrequentialConformalCalibrator.restore(corrupted, policy=selected_policy)
    assert error.value.code == "CHECKPOINT_INTEGRITY_MISMATCH"

    with pytest.raises(PrequentialConformalError) as error:
        PrequentialConformalCalibrator.restore(
            checkpoint,
            policy=policy(coverage=0.9),
        )
    assert error.value.code == "CHECKPOINT_POLICY_MISMATCH"


def test_checkpoint_rejects_validly_checksummed_impossible_state() -> None:
    selected_policy = policy()
    checkpoint = json.loads(PrequentialConformalCalibrator(selected_policy).checkpoint())
    checkpoint["state"]["calibration_operations"] = 1
    content = {key: value for key, value in checkpoint.items() if key != "sha256"}
    canonical = json.dumps(content, allow_nan=False, separators=(",", ":"), sort_keys=True)
    checkpoint["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    with pytest.raises(PrequentialConformalError) as error:
        PrequentialConformalCalibrator.restore(
            json.dumps(checkpoint, separators=(",", ":"), sort_keys=True),
            policy=selected_policy,
        )
    assert error.value.code == "INVALID_CHECKPOINT_STATE"


@pytest.mark.parametrize(
    ("action", "code"),
    [
        (lambda item: item.issue(float("nan")), "INVALID_POINT_FORECAST"),
        (lambda item: item.observe(-1, 1.0), "INVALID_OBSERVATION_POSITION"),
        (lambda item: item.observe(0, float("inf")), "INVALID_OUTCOME"),
    ],
)
def test_streaming_calibrator_rejects_invalid_events(action: object, code: str) -> None:
    calibrator = PrequentialConformalCalibrator(policy())
    if code == "INVALID_OUTCOME":
        calibrator.issue(1.0)
    with pytest.raises(PrequentialConformalError) as error:
        action(calibrator)  # type: ignore[operator]
    assert error.value.code == code
