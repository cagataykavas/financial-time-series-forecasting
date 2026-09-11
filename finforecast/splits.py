from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .domain import FoldBoundary


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_size: int = 250
    test_size: int = 20
    gap: int = 0
    step_size: int | None = None
    max_train_size: int | None = None

    def __post_init__(self) -> None:
        if self.min_train_size < 2:
            raise ValueError("min_train_size must be at least 2")
        if self.test_size < 1:
            raise ValueError("test_size must be positive")
        if self.gap < 0:
            raise ValueError("gap cannot be negative")
        if self.step_size is not None and self.step_size < 1:
            raise ValueError("step_size must be positive")
        if self.max_train_size is not None and self.max_train_size < self.min_train_size:
            raise ValueError("max_train_size cannot be smaller than min_train_size")


class WalkForwardSplitter:
    """Deterministic expanding or rolling time-ordered folds.

    End indices follow Python's exclusive convention. ``gap`` purges observations
    immediately before the test window when labels or features overlap in time.
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def split(self, observation_count: int) -> Iterator[FoldBoundary]:
        if observation_count <= self.config.min_train_size + self.config.gap:
            raise ValueError("not enough observations for one out-of-sample fold")
        step = self.config.step_size or self.config.test_size
        test_start = self.config.min_train_size + self.config.gap
        fold = 0
        while test_start < observation_count:
            train_end = test_start - self.config.gap
            train_start = 0
            if self.config.max_train_size is not None:
                train_start = max(0, train_end - self.config.max_train_size)
            test_end = min(test_start + self.config.test_size, observation_count)
            boundary = FoldBoundary(fold, train_start, train_end, test_start, test_end)
            self._assert_temporal_isolation(boundary)
            yield boundary
            fold += 1
            test_start += step

    def materialize(self, observation_count: int) -> tuple[FoldBoundary, ...]:
        return tuple(self.split(observation_count))

    def _assert_temporal_isolation(self, boundary: FoldBoundary) -> None:
        if boundary.train_end + self.config.gap != boundary.test_start:
            raise AssertionError("fold violates configured purge gap")
        if boundary.train_size < self.config.min_train_size:
            raise AssertionError("fold has insufficient training history")
        if boundary.test_size < 1:
            raise AssertionError("fold has no test observations")
