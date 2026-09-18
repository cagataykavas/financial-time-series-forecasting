"""Temporal diagnostics for probabilistic forecast interval coverage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageWindow:
    """Coverage evidence for one contiguous out-of-sample segment."""

    start: int
    end: int
    observations: int
    covered: int
    coverage: float
    coverage_gap: float
    mean_width: float
    passed: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalCoverageReport:
    """Aggregate and local coverage evidence for a release gate."""

    target_coverage: float
    tolerance: float
    minimum_coverage: float
    observations: int
    overall_coverage: float
    overall_mean_width: float
    worst_window_coverage: float
    failed_windows: int
    passed: bool
    reasons: tuple[str, ...]
    windows: tuple[CoverageWindow, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _interval_arrays(
    actual: object, lower: object, upper: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(actual, dtype=float)
    lows = np.asarray(lower, dtype=float)
    highs = np.asarray(upper, dtype=float)
    if truth.ndim != 1 or truth.shape != lows.shape or truth.shape != highs.shape:
        raise ValueError("actual, lower and upper must be equally sized one-dimensional arrays")
    if truth.size == 0 or not (
        np.isfinite(truth).all() and np.isfinite(lows).all() and np.isfinite(highs).all()
    ):
        raise ValueError("interval audit inputs must be finite and non-empty")
    if np.any(lows > highs):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    return truth, lows, highs


def _window_boundaries(
    observations: int, window_size: int, minimum_window_size: int
) -> tuple[tuple[int, int], ...]:
    boundaries = [
        (start, min(start + window_size, observations))
        for start in range(0, observations, window_size)
    ]
    if len(boundaries) > 1:
        tail_start, tail_end = boundaries[-1]
        if tail_end - tail_start < minimum_window_size:
            previous_start, _ = boundaries[-2]
            boundaries[-2:] = [(previous_start, tail_end)]
    return tuple(boundaries)


def audit_temporal_coverage(
    actual: object,
    lower: object,
    upper: object,
    *,
    target_coverage: float = 0.9,
    tolerance: float = 0.05,
    window_size: int = 50,
    minimum_window_size: int = 20,
) -> TemporalCoverageReport:
    """Detect local interval undercoverage hidden by an acceptable aggregate.

    Windows are contiguous and non-overlapping. An undersized final window is
    merged into its predecessor so every reported decision has the configured
    minimum amount of evidence.
    """

    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    if not 0.0 <= tolerance < target_coverage:
        raise ValueError("tolerance must be non-negative and smaller than target_coverage")
    if window_size < 2:
        raise ValueError("window_size must be at least two")
    if not 2 <= minimum_window_size <= window_size:
        raise ValueError("minimum_window_size must be between two and window_size")

    truth, lows, highs = _interval_arrays(actual, lower, upper)
    if truth.size < minimum_window_size:
        raise ValueError(
            f"coverage audit requires at least {minimum_window_size} observations"
        )

    covered = (truth >= lows) & (truth <= highs)
    widths = highs - lows
    minimum_coverage = target_coverage - tolerance
    windows: list[CoverageWindow] = []

    for start, end in _window_boundaries(truth.size, window_size, minimum_window_size):
        window_covered = covered[start:end]
        coverage = float(window_covered.mean())
        windows.append(
            CoverageWindow(
                start=start,
                end=end,
                observations=end - start,
                covered=int(window_covered.sum()),
                coverage=coverage,
                coverage_gap=coverage - target_coverage,
                mean_width=float(widths[start:end].mean()),
                passed=coverage >= minimum_coverage,
            )
        )

    overall_coverage = float(covered.mean())
    failed_windows = sum(not window.passed for window in windows)
    reasons: list[str] = []
    if overall_coverage < minimum_coverage:
        reasons.append("overall_undercoverage")
    if failed_windows:
        reasons.append("temporal_undercoverage")

    return TemporalCoverageReport(
        target_coverage=target_coverage,
        tolerance=tolerance,
        minimum_coverage=minimum_coverage,
        observations=int(truth.size),
        overall_coverage=overall_coverage,
        overall_mean_width=float(widths.mean()),
        worst_window_coverage=min(window.coverage for window in windows),
        failed_windows=failed_windows,
        passed=not reasons,
        reasons=tuple(reasons),
        windows=tuple(windows),
    )
