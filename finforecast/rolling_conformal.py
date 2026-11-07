from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

from .intervals import SplitConformalCalibrator

ROLLING_CONFORMAL_SCHEMA = "finforecast-rolling-delayed-conformal/v1"
_MAX_OBSERVATIONS = 100_000


def _canonical_sha256(document: object) -> str:
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 1:
        raise ValueError(f"{field} must be positive")


@dataclass(frozen=True)
class RollingConformalPolicy:
    """Predeclared delayed-label recalibration policy."""

    target_coverage: float = 0.9
    calibration_window: int = 250
    minimum_calibration_size: int = 50
    refresh_interval: int = 25

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_coverage, bool)
            or not isinstance(self.target_coverage, int | float)
            or not math.isfinite(self.target_coverage)
            or not 0.0 < self.target_coverage < 1.0
        ):
            raise ValueError("target_coverage must be finite and strictly between zero and one")
        _positive_integer(self.calibration_window, "calibration_window")
        _positive_integer(self.minimum_calibration_size, "minimum_calibration_size")
        _positive_integer(self.refresh_interval, "refresh_interval")
        if self.minimum_calibration_size < 2:
            raise ValueError("minimum_calibration_size must be at least two")
        if self.minimum_calibration_size > self.calibration_window:
            raise ValueError("minimum_calibration_size cannot exceed calibration_window")
        if self.calibration_window > _MAX_OBSERVATIONS:
            raise ValueError("calibration_window exceeds the supported observation bound")
        if self.refresh_interval > _MAX_OBSERVATIONS:
            raise ValueError("refresh_interval exceeds the supported observation bound")

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "schema": ROLLING_CONFORMAL_SCHEMA,
            "target_coverage": float(self.target_coverage),
            "calibration_window": self.calibration_window,
            "minimum_calibration_size": self.minimum_calibration_size,
            "refresh_interval": self.refresh_interval,
            "label_maturity_rule": "label_available_at_strictly_before_refresh_issue_time",
            "window_selection": "most_recent_matured_observations",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class DelayedForecast:
    """One issued point forecast whose outcome may become available later."""

    issued_at: int
    label_available_at: int
    actual: float
    point_forecast: float

    def __post_init__(self) -> None:
        _positive_integer(self.issued_at, "issued_at")
        _positive_integer(self.label_available_at, "label_available_at")
        if self.label_available_at < self.issued_at:
            raise ValueError("label_available_at cannot precede forecast issuance")
        for field, value in (
            ("actual", self.actual),
            ("point_forecast", self.point_forecast),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field} must be finite")

    def _commitment_document(self) -> dict[str, float | int]:
        return {
            "issued_at": self.issued_at,
            "label_available_at": self.label_available_at,
            "actual": float(self.actual),
            "point_forecast": float(self.point_forecast),
        }


@dataclass(frozen=True)
class RollingConformalResult:
    """Generated intervals plus a privacy-reduced audit report."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    report: dict[str, object]


def _validated_observations(
    observations: Iterable[DelayedForecast], field: str
) -> tuple[DelayedForecast, ...]:
    try:
        result = tuple(observations)
    except TypeError as exc:
        raise TypeError(f"{field} must be iterable") from exc
    if any(not isinstance(item, DelayedForecast) for item in result):
        raise TypeError(f"{field} must contain only DelayedForecast values")
    return result


def rolling_delayed_conformal(
    history: Iterable[DelayedForecast],
    evaluation: Iterable[DelayedForecast],
    policy: RollingConformalPolicy,
) -> RollingConformalResult:
    """Generate rolling conformal intervals without admitting unavailable labels.

    ``history`` seeds the first refresh. Evaluation observations can enter later
    calibration windows only after their label availability time is strictly earlier
    than the issue time at a scheduled refresh. Between refreshes, the radius remains
    fixed even when more labels mature.
    """

    if not isinstance(policy, RollingConformalPolicy):
        raise TypeError("policy must be a RollingConformalPolicy")
    history_rows = _validated_observations(history, "history")
    evaluation_rows = _validated_observations(evaluation, "evaluation")
    if not evaluation_rows:
        raise ValueError("evaluation must contain at least one observation")
    if len(history_rows) + len(evaluation_rows) > _MAX_OBSERVATIONS:
        raise ValueError("rolling conformal input exceeds the supported observation bound")

    combined = history_rows + evaluation_rows
    issued_at = [row.issued_at for row in combined]
    if any(
        current <= previous for previous, current in zip(issued_at, issued_at[1:], strict=False)
    ):
        raise ValueError("forecast issue times must be strictly increasing")

    lower: list[float] = []
    upper: list[float] = []
    covered = 0
    width_sum = 0.0
    refreshes: list[dict[str, object]] = []
    radius: float | None = None

    for offset, current in enumerate(evaluation_rows):
        if offset % policy.refresh_interval == 0:
            prior = history_rows + evaluation_rows[:offset]
            matured = tuple(row for row in prior if row.label_available_at < current.issued_at)
            selected = matured[-policy.calibration_window :]
            if len(selected) < policy.minimum_calibration_size:
                raise ValueError(
                    "scheduled refresh has fewer matured labels than minimum_calibration_size"
                )
            calibrator = SplitConformalCalibrator(coverage=float(policy.target_coverage)).fit(
                [row.actual for row in selected],
                [row.point_forecast for row in selected],
            )
            radius = calibrator.radius_
            assert radius is not None
            refreshes.append(
                {
                    "evaluation_offset": offset,
                    "refresh_issued_at": current.issued_at,
                    "calibration_size": len(selected),
                    "calibration_first_issued_at": selected[0].issued_at,
                    "calibration_last_issued_at": selected[-1].issued_at,
                    "latest_label_available_at": max(row.label_available_at for row in selected),
                    "radius": radius,
                    "selection_sha256": _canonical_sha256(
                        [row._commitment_document() for row in selected]
                    ),
                }
            )

        assert radius is not None
        interval_lower = float(current.point_forecast) - radius
        interval_upper = float(current.point_forecast) + radius
        lower.append(interval_lower)
        upper.append(interval_upper)
        covered += int(interval_lower <= current.actual <= interval_upper)
        width_sum += interval_upper - interval_lower

    report: dict[str, object] = {
        "schema": ROLLING_CONFORMAL_SCHEMA,
        "policy": policy.as_dict(),
        "policy_sha256": policy.sha256,
        "history_size": len(history_rows),
        "evaluation_size": len(evaluation_rows),
        "refresh_count": len(refreshes),
        "observed_coverage": covered / len(evaluation_rows),
        "mean_interval_width": width_sum / len(evaluation_rows),
        "input_sha256": _canonical_sha256([row._commitment_document() for row in combined]),
        "refreshes": refreshes,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return RollingConformalResult(tuple(lower), tuple(upper), report)
