"""Leakage-aware financial time-series forecasting reference project."""

from .backtest import BacktestRunner
from .corporate_actions import CorporateActionSnapshot, apply_point_in_time_splits
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
    "CorporateActionSnapshot",
    "ForecastExperiment",
    "ForecastTargetSpec",
    "MarketData",
    "MarketDataSnapshot",
    "SplitConformalCalibrator",
    "WalkForwardConfig",
    "WalkForwardSplitter",
    "apply_point_in_time_splits",
    "load_market_csv",
    "load_point_in_time_market_csv",
]
