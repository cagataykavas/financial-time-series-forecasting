"""Leakage-safe prequential conformal intervals for time-ordered forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PrequentialConformalError(ValueError):
    """A stable, non-sensitive validation failure for prequential calibration."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PrequentialConformalPolicy:
    """Resource and statistical policy for rolling conformal calibration."""

    coverage: float = 0.9
    window: int = 250
    min_calibration: int = 50
    feedback_delay: int = 0
    max_points: int = 100_000
    max_calibration_operations: int = 5_000_000

    def __post_init__(self) -> None:
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, (int, float)):
            raise PrequentialConformalError("INVALID_COVERAGE")
        if not np.isfinite(self.coverage) or not 0.0 < float(self.coverage) < 1.0:
            raise PrequentialConformalError("INVALID_COVERAGE")
        _require_positive_int(self.window, "INVALID_WINDOW")
        _require_positive_int(self.min_calibration, "INVALID_MIN_CALIBRATION")
        if self.min_calibration > self.window:
            raise PrequentialConformalError("MIN_CALIBRATION_EXCEEDS_WINDOW")
        _require_nonnegative_int(self.feedback_delay, "INVALID_FEEDBACK_DELAY")
        _require_positive_int(self.max_points, "INVALID_MAX_POINTS")
        _require_positive_int(self.max_calibration_operations, "INVALID_OPERATION_BUDGET")


@dataclass(frozen=True)
class PrequentialInterval:
    """One interval and the exact historical slice used to calibrate it."""

    position: int
    point_forecast: float
    lower: float
    upper: float
    radius: float
    calibration_start: int
    calibration_end_exclusive: int
    calibration_size: int


@dataclass(frozen=True)
class PrequentialConformalResult:
    """Deterministic intervals plus replay metadata."""

    input_size: int
    coverage: float
    window: int
    min_calibration: int
    feedback_delay: int
    calibration_operations: int
    intervals: tuple[PrequentialInterval, ...]
    source_index: tuple[Any, ...] | None = None

    def as_frame(self) -> pd.DataFrame:
        """Return intervals without including outcomes unavailable at issuance time."""

        records = [
            {
                "position": item.position,
                "point_forecast": item.point_forecast,
                "lower": item.lower,
                "upper": item.upper,
                "radius": item.radius,
                "calibration_start": item.calibration_start,
                "calibration_end_exclusive": item.calibration_end_exclusive,
                "calibration_size": item.calibration_size,
            }
            for item in self.intervals
        ]
        frame = pd.DataFrame.from_records(records)
        if self.source_index is not None:
            labels = [self.source_index[item.position] for item in self.intervals]
            frame.index = pd.Index(labels, name="forecast_time")
        return frame


def prequential_conformal_intervals(
    actual: object,
    predicted: object,
    *,
    policy: PrequentialConformalPolicy | None = None,
) -> PrequentialConformalResult:
    """Replay symmetric conformal intervals without using current or future outcomes.

    The interval at position ``i`` uses absolute residuals ending strictly before
    ``i - feedback_delay``. This makes data availability explicit for delayed labels
    or multi-step forecasts while keeping the routine model-independent.
    """

    resolved = policy or PrequentialConformalPolicy()
    truth, forecast, source_index = _validated_inputs(actual, predicted, resolved)
    first_position = resolved.min_calibration + resolved.feedback_delay
    if first_position >= truth.size:
        raise PrequentialConformalError("INSUFFICIENT_HISTORY")

    calibration_operations = sum(
        min(resolved.window, position - resolved.feedback_delay)
        for position in range(first_position, truth.size)
    )
    if calibration_operations > resolved.max_calibration_operations:
        raise PrequentialConformalError("OPERATION_BUDGET_EXCEEDED")

    residuals = np.abs(truth - forecast)
    intervals: list[PrequentialInterval] = []
    for position in range(first_position, truth.size):
        calibration_end = position - resolved.feedback_delay
        calibration_start = max(0, calibration_end - resolved.window)
        history = residuals[calibration_start:calibration_end]
        calibration_size = int(history.size)
        if calibration_size < resolved.min_calibration:
            raise PrequentialConformalError("INSUFFICIENT_CALIBRATION")
        rank = min(
            calibration_size,
            int(np.ceil((calibration_size + 1) * float(resolved.coverage))),
        )
        radius = float(np.partition(history, rank - 1)[rank - 1])
        point = float(forecast[position])
        intervals.append(
            PrequentialInterval(
                position=position,
                point_forecast=point,
                lower=point - radius,
                upper=point + radius,
                radius=radius,
                calibration_start=calibration_start,
                calibration_end_exclusive=calibration_end,
                calibration_size=calibration_size,
            )
        )

    return PrequentialConformalResult(
        input_size=int(truth.size),
        coverage=float(resolved.coverage),
        window=resolved.window,
        min_calibration=resolved.min_calibration,
        feedback_delay=resolved.feedback_delay,
        calibration_operations=calibration_operations,
        intervals=tuple(intervals),
        source_index=source_index,
    )


def _validated_inputs(
    actual: object,
    predicted: object,
    policy: PrequentialConformalPolicy,
) -> tuple[np.ndarray, np.ndarray, tuple[Any, ...] | None]:
    actual_is_series = isinstance(actual, pd.Series)
    predicted_is_series = isinstance(predicted, pd.Series)
    if actual_is_series != predicted_is_series:
        raise PrequentialConformalError("INDEX_TYPE_MISMATCH")

    source_index: tuple[Any, ...] | None = None
    if actual_is_series and predicted_is_series:
        if not actual.index.equals(predicted.index):
            raise PrequentialConformalError("INDEX_MISMATCH")
        if not actual.index.is_unique:
            raise PrequentialConformalError("DUPLICATE_INDEX")
        if not actual.index.is_monotonic_increasing:
            raise PrequentialConformalError("NON_MONOTONIC_INDEX")
        source_index = tuple(actual.index.tolist())

    try:
        truth = np.asarray(actual, dtype=float)
        forecast = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PrequentialConformalError("NON_NUMERIC_INPUT") from exc
    if truth.ndim != 1 or forecast.ndim != 1:
        raise PrequentialConformalError("INVALID_SHAPE")
    if truth.size == 0:
        raise PrequentialConformalError("EMPTY_INPUT")
    if truth.shape != forecast.shape:
        raise PrequentialConformalError("LENGTH_MISMATCH")
    if truth.size > policy.max_points:
        raise PrequentialConformalError("POINT_BUDGET_EXCEEDED")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise PrequentialConformalError("NON_FINITE_INPUT")
    return truth.copy(), forecast.copy(), source_index


def _require_positive_int(value: object, code: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrequentialConformalError(code)


def _require_nonnegative_int(value: object, code: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrequentialConformalError(code)
