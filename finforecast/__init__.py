"""Leakage-aware financial time-series forecasting reference project."""

from .backtest import BacktestRunner
from .engine import ForecastExperiment
from .intervals import SplitConformalCalibrator
from .splits import WalkForwardConfig, WalkForwardSplitter

__version__ = "2.0.1"

__all__ = [
    "__version__",
    "BacktestRunner",
    "ForecastExperiment",
    "SplitConformalCalibrator",
    "WalkForwardConfig",
    "WalkForwardSplitter",
]
