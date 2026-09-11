from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    CHAMPION = "champion"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ForecastInterval:
    point: float
    lower: float
    upper: float
    coverage: float

    def __post_init__(self) -> None:
        if not 0.0 < self.coverage < 1.0:
            raise ValueError("coverage must be between zero and one")
        if self.lower > self.point or self.point > self.upper:
            raise ValueError("interval must contain the point forecast")


@dataclass(frozen=True)
class FoldBoundary:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    def as_dict(self) -> dict[str, int]:
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_size": self.train_size,
            "test_size": self.test_size,
        }


@dataclass(frozen=True)
class PromotionDecision:
    candidate: str
    baseline: str
    status: ModelStatus
    reasons: tuple[str, ...]
    metrics: dict[str, Any]

    @property
    def promoted(self) -> bool:
        return self.status is ModelStatus.CHAMPION

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "status": self.status.value,
            "promoted": self.promoted,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }
