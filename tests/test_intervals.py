from __future__ import annotations

import numpy as np
import pytest

from finforecast.intervals import PurgedCalibrationBoundary, SplitConformalCalibrator


def test_purged_calibration_boundary_embargoes_the_complete_label_horizon():
    boundary = PurgedCalibrationBoundary.build(
        100,
        calibration_fraction=0.4,
        label_horizon=5,
    )

    assert boundary.calibration_slice == slice(0, 35)
    assert boundary.evaluation_slice == slice(40, 100)
    assert boundary.embargo_size == 5
    assert boundary.latest_calibration_label_end == 39
    assert boundary.latest_calibration_label_end < boundary.evaluation_start
    assert boundary.as_dict()["evaluation_size"] == 60
    assert len(boundary.sha256) == 64


def test_embargoed_residuals_cannot_influence_the_conformal_radius():
    boundary = PurgedCalibrationBoundary.build(
        10,
        calibration_fraction=0.5,
        label_horizon=2,
    )
    actual = np.zeros(10)
    predicted = np.zeros(10)
    predicted[boundary.calibration_end : boundary.evaluation_start] = 1_000.0

    calibrator = SplitConformalCalibrator(coverage=0.8).fit(
        actual[boundary.calibration_slice],
        predicted[boundary.calibration_slice],
    )
    lower, upper = calibrator.predict(predicted[boundary.evaluation_slice])

    assert calibrator.radius_ == 0.0
    assert lower.tolist() == [0.0] * boundary.evaluation_size
    assert upper.tolist() == [0.0] * boundary.evaluation_size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_count": 10, "calibration_fraction": 0.5, "label_horizon": 4},
        {
            "observation_count": 10,
            "calibration_fraction": 0.9,
            "label_horizon": 1,
            "minimum_evaluation_size": 2,
        },
        {"observation_count": 10, "calibration_fraction": 0.0, "label_horizon": 1},
        {"observation_count": 10, "calibration_fraction": float("nan"), "label_horizon": 1},
        {"observation_count": 10, "calibration_fraction": 0.5, "label_horizon": 0},
    ],
)
def test_purged_calibration_boundary_rejects_unsafe_segments(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PurgedCalibrationBoundary.build(**kwargs)


def test_purged_calibration_boundary_digest_commits_to_policy():
    first = PurgedCalibrationBoundary.build(100, calibration_fraction=0.4, label_horizon=1)
    replay = PurgedCalibrationBoundary.build(100, calibration_fraction=0.4, label_horizon=1)
    changed = PurgedCalibrationBoundary.build(100, calibration_fraction=0.5, label_horizon=1)

    assert first == replay
    assert first.sha256 == replay.sha256
    assert first.sha256 != changed.sha256


def test_conformal_interval_uses_finite_sample_quantile():
    calibrator = SplitConformalCalibrator(coverage=0.8).fit(
        [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]
    )
    assert calibrator.radius_ == 3.0
    lower, upper = calibrator.predict(np.array([10.0]))
    assert lower.tolist() == [7.0]
    assert upper.tolist() == [13.0]


def test_unfitted_calibrator_refuses_prediction():
    with pytest.raises(RuntimeError, match="fitted"):
        SplitConformalCalibrator().predict([0.0])


def test_calibrator_rejects_bad_shapes():
    with pytest.raises(ValueError, match="at least two"):
        SplitConformalCalibrator().fit([1.0], [1.0])


def test_calibrator_rejects_invalid_point_forecasts():
    calibrator = SplitConformalCalibrator().fit([0.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        calibrator.predict([float("nan")])
    with pytest.raises(ValueError, match="vector"):
        calibrator.predict([[1.0]])
