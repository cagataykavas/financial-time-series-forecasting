from __future__ import annotations

import numpy as np
import pytest

from finforecast.intervals import SplitConformalCalibrator


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
