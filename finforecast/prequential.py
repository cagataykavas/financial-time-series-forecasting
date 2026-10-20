"""Leakage-safe prequential conformal intervals for time-ordered forecasts."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
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


_CHECKPOINT_SCHEMA_VERSION = 1
_MAX_CHECKPOINT_BYTES = 2_000_000


class PrequentialConformalCalibrator:
    """Issue intervals and ingest ordered outcomes with resumable state.

    ``issue`` never consumes the outcome for the forecast being issued. Outcomes are
    admitted separately through ``observe`` and must arrive exactly once, in order.
    This makes the information boundary enforceable in a long-running worker instead
    of relying on callers to slice arrays correctly.
    """

    def __init__(self, policy: PrequentialConformalPolicy | None = None) -> None:
        self.policy = policy or PrequentialConformalPolicy()
        self._next_position = 0
        self._next_observation_position = 0
        self._calibration_operations = 0
        self._residuals: dict[int, float] = {}
        self._pending_forecasts: dict[int, float] = {}

    @property
    def next_position(self) -> int:
        """Return the position assigned to the next forecast."""

        return self._next_position

    @property
    def calibration_operations(self) -> int:
        """Return the cumulative number of residual selections."""

        return self._calibration_operations

    def issue(self, point_forecast: object) -> PrequentialInterval | None:
        """Issue the next interval without reading its outcome.

        ``None`` is returned during calibration warm-up. A missing outcome that should
        already be available under ``feedback_delay`` fails closed.
        """

        point = _finite_scalar(point_forecast, "INVALID_POINT_FORECAST")
        position = self._next_position
        if position >= self.policy.max_points:
            raise PrequentialConformalError("POINT_BUDGET_EXCEEDED")

        calibration_end = max(0, position - self.policy.feedback_delay)
        if self._next_observation_position < calibration_end:
            raise PrequentialConformalError("MISSING_FEEDBACK")
        calibration_start = max(0, calibration_end - self.policy.window)

        interval: PrequentialInterval | None = None
        if calibration_end >= self.policy.min_calibration:
            positions = range(calibration_start, calibration_end)
            try:
                history = np.fromiter(
                    (self._residuals[item] for item in positions),
                    dtype=float,
                    count=calibration_end - calibration_start,
                )
            except KeyError as exc:
                raise PrequentialConformalError("INVALID_CHECKPOINT_STATE") from exc
            calibration_size = int(history.size)
            operations = self._calibration_operations + calibration_size
            if operations > self.policy.max_calibration_operations:
                raise PrequentialConformalError("OPERATION_BUDGET_EXCEEDED")
            rank = min(
                calibration_size,
                int(np.ceil((calibration_size + 1) * float(self.policy.coverage))),
            )
            radius = float(np.partition(history, rank - 1)[rank - 1])
            interval = PrequentialInterval(
                position=position,
                point_forecast=point,
                lower=point - radius,
                upper=point + radius,
                radius=radius,
                calibration_start=calibration_start,
                calibration_end_exclusive=calibration_end,
                calibration_size=calibration_size,
            )
            self._calibration_operations = operations

        self._pending_forecasts[position] = point
        self._next_position += 1
        self._discard_expired_residuals()
        return interval

    def observe(self, position: object, actual: object) -> None:
        """Admit one realized outcome for a previously issued forecast."""

        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise PrequentialConformalError("INVALID_OBSERVATION_POSITION")
        if position < self._next_observation_position:
            raise PrequentialConformalError("DUPLICATE_FEEDBACK")
        if position > self._next_observation_position:
            raise PrequentialConformalError("OUT_OF_ORDER_FEEDBACK")
        if position not in self._pending_forecasts:
            raise PrequentialConformalError("UNKNOWN_FORECAST")

        outcome = _finite_scalar(actual, "INVALID_OUTCOME")
        forecast = self._pending_forecasts.pop(position)
        self._residuals[position] = abs(outcome - forecast)
        self._next_observation_position += 1
        self._discard_expired_residuals()

    def checkpoint(self) -> str:
        """Return a bounded, checksummed JSON checkpoint without realized outcomes."""

        content = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "policy": asdict(self.policy),
            "state": {
                "next_position": self._next_position,
                "next_observation_position": self._next_observation_position,
                "calibration_operations": self._calibration_operations,
                "residuals": [[key, value] for key, value in sorted(self._residuals.items())],
                "pending_forecasts": [
                    [key, value] for key, value in sorted(self._pending_forecasts.items())
                ],
            },
        }
        digest = hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()
        payload = {**content, "sha256": digest}
        encoded = _canonical_json(payload)
        if len(encoded.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
            raise PrequentialConformalError("CHECKPOINT_TOO_LARGE")
        return encoded

    @classmethod
    def restore(
        cls,
        checkpoint: object,
        *,
        policy: PrequentialConformalPolicy,
    ) -> PrequentialConformalCalibrator:
        """Restore a checkpoint under an explicitly supplied expected policy."""

        if not isinstance(checkpoint, str):
            raise PrequentialConformalError("INVALID_CHECKPOINT")
        try:
            encoded = checkpoint.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PrequentialConformalError("INVALID_CHECKPOINT") from exc
        if len(encoded) > _MAX_CHECKPOINT_BYTES:
            raise PrequentialConformalError("CHECKPOINT_TOO_LARGE")
        try:
            payload = json.loads(checkpoint)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise PrequentialConformalError("INVALID_CHECKPOINT") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "policy",
            "state",
            "sha256",
        }:
            raise PrequentialConformalError("INVALID_CHECKPOINT")
        digest = payload.pop("sha256")
        if not isinstance(digest, str) or not hmac.compare_digest(
            digest,
            hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        ):
            raise PrequentialConformalError("CHECKPOINT_INTEGRITY_MISMATCH")
        schema_version = payload["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _CHECKPOINT_SCHEMA_VERSION
        ):
            raise PrequentialConformalError("UNSUPPORTED_CHECKPOINT_SCHEMA")
        if _canonical_json(payload["policy"]) != _canonical_json(asdict(policy)):
            raise PrequentialConformalError("CHECKPOINT_POLICY_MISMATCH")

        state = payload["state"]
        if not isinstance(state, dict) or set(state) != {
            "next_position",
            "next_observation_position",
            "calibration_operations",
            "residuals",
            "pending_forecasts",
        }:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        restored = cls(policy)
        restored._restore_state(state)
        return restored

    def _restore_state(self, state: dict[str, object]) -> None:
        next_position = _checkpoint_nonnegative_int(state["next_position"])
        next_observation = _checkpoint_nonnegative_int(state["next_observation_position"])
        operations = _checkpoint_nonnegative_int(state["calibration_operations"])
        if (
            next_observation > next_position
            or next_position > self.policy.max_points
            or next_position - next_observation > self.policy.feedback_delay + 1
        ):
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        if operations > self.policy.max_calibration_operations:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        first_position = self.policy.min_calibration + self.policy.feedback_delay
        expected_operations = sum(
            min(self.policy.window, position - self.policy.feedback_delay)
            for position in range(first_position, next_position)
        )
        if operations != expected_operations:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")

        residuals = _checkpoint_pairs(state["residuals"], residual=True)
        pending = _checkpoint_pairs(state["pending_forecasts"], residual=False)
        minimum_position = max(
            0,
            next_position - self.policy.feedback_delay - self.policy.window,
        )
        expected_residuals = list(range(minimum_position, next_observation))
        expected_pending = list(range(next_observation, next_position))
        if list(residuals) != expected_residuals or list(pending) != expected_pending:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")

        self._next_position = next_position
        self._next_observation_position = next_observation
        self._calibration_operations = operations
        self._residuals = residuals
        self._pending_forecasts = pending

    def _discard_expired_residuals(self) -> None:
        minimum_position = max(
            0,
            self._next_position - self.policy.feedback_delay - self.policy.window,
        )
        self._residuals = {
            position: residual
            for position, residual in self._residuals.items()
            if position >= minimum_position
        }


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

    calibrator = PrequentialConformalCalibrator(resolved)
    intervals: list[PrequentialInterval] = []
    for position, (outcome, point) in enumerate(zip(truth, forecast, strict=True)):
        interval = calibrator.issue(point)
        if interval is not None:
            intervals.append(interval)
        calibrator.observe(position, outcome)

    if calibrator.calibration_operations != calibration_operations:
        raise PrequentialConformalError("CALIBRATION_ACCOUNTING_MISMATCH")

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


def _finite_scalar(value: object, code: str) -> float:
    if isinstance(value, bool):
        raise PrequentialConformalError(code)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PrequentialConformalError(code) from exc
    if not np.isfinite(converted):
        raise PrequentialConformalError(code)
    return converted


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise PrequentialConformalError("INVALID_CHECKPOINT") from exc


def _checkpoint_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
    return value


def _checkpoint_pairs(value: object, *, residual: bool) -> dict[int, float]:
    if not isinstance(value, list):
        raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
    result: dict[int, float] = {}
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        position = _checkpoint_nonnegative_int(item[0])
        number = _finite_scalar(item[1], "INVALID_CHECKPOINT_STATE")
        if residual and number < 0.0:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        if position in result:
            raise PrequentialConformalError("INVALID_CHECKPOINT_STATE")
        result[position] = number
    return result
