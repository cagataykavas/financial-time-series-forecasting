from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConformalCalibrator:
    """Distribution-free symmetric intervals from held-out residuals."""

    coverage: float = 0.9
    radius_: float | None = None
    calibration_size_: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.coverage < 1.0:
            raise ValueError("coverage must be between zero and one")

    def fit(self, actual: object, predicted: object) -> SplitConformalCalibrator:
        truth = np.asarray(actual, dtype=float)
        forecast = np.asarray(predicted, dtype=float)
        if truth.ndim != 1 or truth.shape != forecast.shape or truth.size < 2:
            raise ValueError(
                "calibration inputs must be equally sized and contain at least two values"
            )
        residuals = np.abs(truth - forecast)
        if not np.isfinite(residuals).all():
            raise ValueError("calibration residuals must be finite")
        n = residuals.size
        rank = min(n, int(np.ceil((n + 1) * self.coverage)))
        self.radius_ = float(np.partition(residuals, rank - 1)[rank - 1])
        self.calibration_size_ = n
        return self

    def predict(self, point_forecast: object) -> tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None:
            raise RuntimeError("calibrator must be fitted before prediction")
        point = np.asarray(point_forecast, dtype=float)
        return point - self.radius_, point + self.radius_

    def metadata(self) -> dict[str, float | int]:
        if self.radius_ is None:
            raise RuntimeError("calibrator has not been fitted")
        return {
            "method": "split_conformal_absolute_residual",
            "target_coverage": self.coverage,
            "calibration_size": self.calibration_size_,
            "radius": self.radius_,
        }
