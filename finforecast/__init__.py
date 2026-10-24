"""Leakage-aware financial time-series forecasting reference project."""

from .backtest import BacktestRunner
from .data import (
    MarketData,
    MarketDataSnapshot,
    load_market_csv,
    load_point_in_time_market_csv,
)
from .engine import ForecastExperiment
from .features import ForecastTargetSpec
from .intervals import SplitConformalCalibrator
from .splits import WalkForwardConfig, WalkForwardSplitter

__version__ = "2.1.0"

__all__ = [
    "__version__",
    "BacktestRunner",
    "ForecastExperiment",
    "ForecastTargetSpec",
    "MarketData",
    "MarketDataSnapshot",
    "SplitConformalCalibrator",
    "WalkForwardConfig",
    "WalkForwardSplitter",
    "load_market_csv",
    "load_point_in_time_market_csv",
]
