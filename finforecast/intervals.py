from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedCalibrationBoundary:
    """Auditable split between mature calibration labels and later evaluation."""

    observation_count: int
    calibration_fraction: float
    label_horizon: int
    minimum_calibration_size: int
    minimum_evaluation_size: int
    calibration_end: int
    evaluation_start: int

    def __post_init__(self) -> None:
        integer_fields = {
            "observation_count": self.observation_count,
            "label_horizon": self.label_horizon,
            "minimum_calibration_size": self.minimum_calibration_size,
            "minimum_evaluation_size": self.minimum_evaluation_size,
            "calibration_end": self.calibration_end,
            "evaluation_start": self.evaluation_start,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise TypeError("purged calibration boundary offsets must be integers")
        if self.observation_count < 1:
            raise ValueError("observation_count must be positive")
        if not 1 <= self.label_horizon <= 252:
            raise ValueError("label_horizon must be between 1 and 252")
        if self.minimum_calibration_size < 2:
            raise ValueError("minimum_calibration_size must be at least two")
        if self.minimum_evaluation_size < 1:
            raise ValueError("minimum_evaluation_size must be positive")
        if (
            isinstance(self.calibration_fraction, bool)
            or not isinstance(self.calibration_fraction, int | float)
            or not math.isfinite(self.calibration_fraction)
            or not 0.0 < self.calibration_fraction < 1.0
        ):
            raise ValueError("calibration_fraction must be finite and between zero and one")
        if self.calibration_end < self.minimum_calibration_size:
            raise ValueError("purged calibration segment is smaller than the configured minimum")
        if self.evaluation_start - self.calibration_end != self.label_horizon:
            raise ValueError("calibration embargo must equal the target label horizon")
        if self.observation_count - self.evaluation_start < self.minimum_evaluation_size:
            raise ValueError("evaluation segment is smaller than the configured minimum")
        if self.latest_calibration_label_end >= self.evaluation_start:
            raise ValueError("calibration labels overlap the evaluation period")

    @classmethod
    def build(
        cls,
        observation_count: int,
        *,
        calibration_fraction: float,
        label_horizon: int,
        minimum_calibration_size: int = 2,
        minimum_evaluation_size: int = 1,
    ) -> PurgedCalibrationBoundary:
        if isinstance(observation_count, bool) or not isinstance(observation_count, int):
            raise TypeError("observation_count must be an integer")
        if (
            isinstance(calibration_fraction, bool)
            or not isinstance(calibration_fraction, int | float)
            or not math.isfinite(calibration_fraction)
            or not 0.0 < calibration_fraction < 1.0
        ):
            raise ValueError("calibration_fraction must be finite and between zero and one")
        if isinstance(label_horizon, bool) or not isinstance(label_horizon, int):
            raise TypeError("label_horizon must be an integer")
        evaluation_start = int(observation_count * calibration_fraction)
        return cls(
            observation_count=observation_count,
            calibration_fraction=float(calibration_fraction),
            label_horizon=label_horizon,
            minimum_calibration_size=minimum_calibration_size,
            minimum_evaluation_size=minimum_evaluation_size,
            calibration_end=evaluation_start - label_horizon,
            evaluation_start=evaluation_start,
        )

    @property
    def calibration_slice(self) -> slice:
        return slice(0, self.calibration_end)

    @property
    def evaluation_slice(self) -> slice:
        return slice(self.evaluation_start, self.observation_count)

    @property
    def calibration_size(self) -> int:
        return self.calibration_end

    @property
    def embargo_size(self) -> int:
        return self.evaluation_start - self.calibration_end

    @property
    def evaluation_size(self) -> int:
        return self.observation_count - self.evaluation_start

    @property
    def latest_calibration_label_end(self) -> int:
        return self.calibration_end - 1 + self.label_horizon

    def as_dict(self) -> dict[str, int | float]:
        return {
            "schema_version": 1,
            "observation_count": self.observation_count,
            "calibration_fraction": self.calibration_fraction,
            "label_horizon": self.label_horizon,
            "minimum_calibration_size": self.minimum_calibration_size,
            "minimum_evaluation_size": self.minimum_evaluation_size,
            "calibration_start": 0,
            "calibration_end": self.calibration_end,
            "calibration_size": self.calibration_size,
            "embargo_start": self.calibration_end,
            "embargo_end": self.evaluation_start,
            "embargo_size": self.embargo_size,
            "latest_calibration_label_end": self.latest_calibration_label_end,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.observation_count,
            "evaluation_size": self.evaluation_size,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


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
        if point.ndim != 1 or point.size == 0 or not np.isfinite(point).all():
            raise ValueError("point forecasts must be a finite, non-empty vector")
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
