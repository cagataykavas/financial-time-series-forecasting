"""Leakage-aware financial time-series forecasting reference project."""

from .backtest import BacktestRunner
from .data import MarketData, load_market_csv
from .engine import ForecastExperiment
from .intervals import SplitConformalCalibrator
from .splits import WalkForwardConfig, WalkForwardSplitter

__version__ = "2.1.0"

__all__ = [
    "__version__",
    "BacktestRunner",
    "ForecastExperiment",
    "MarketData",
    "SplitConformalCalibrator",
    "WalkForwardConfig",
    "WalkForwardSplitter",
    "load_market_csv",
]
