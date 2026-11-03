"""Leakage-aware financial time-series forecasting reference project."""

from .backtest import BacktestRunner
from .business_calendar import BusinessCalendar, load_business_calendar
from .corporate_actions import (
    CorporateActionSnapshot,
    apply_point_in_time_corporate_actions,
    apply_point_in_time_splits,
)
from .data import (
    MarketData,
    MarketDataSnapshot,
    load_market_csv,
    load_point_in_time_market_csv,
)
from .engine import ForecastExperiment
from .features import ForecastTargetSpec
from .intervals import SplitConformalCalibrator
from .regime_coverage import (
    RegimeCoveragePolicy,
    RegimeIntervalObservation,
    audit_regime_coverage,
    require_regime_coverage,
)
from .regime_labeling import (
    PointInTimeReturn,
    ReturnRevisionStreamCommitment,
    VolatilityRegimeAssignment,
    VolatilityRegimeDefinition,
    assign_point_in_time_volatility_regimes,
    commit_point_in_time_returns,
)
from .regime_source import (
    ReturnRevisionSnapshotSignature,
    assign_authenticated_point_in_time_volatility_regimes,
    create_return_revision_snapshot_signature,
    dump_return_revision_snapshot_signature,
    load_return_revision_snapshot_signature,
    verify_return_revision_snapshot_signature,
)
from .settlements import SettlementSnapshot, load_point_in_time_settlement
from .snapshot_observation import (
    SnapshotObservationReceipt,
    verify_and_record_return_revision_snapshot,
)
from .splits import WalkForwardConfig, WalkForwardSplitter

__version__ = "2.1.0"

__all__ = [
    "__version__",
    "BacktestRunner",
    "BusinessCalendar",
    "CorporateActionSnapshot",
    "ForecastExperiment",
    "ForecastTargetSpec",
    "MarketData",
    "MarketDataSnapshot",
    "PointInTimeReturn",
    "RegimeCoveragePolicy",
    "RegimeIntervalObservation",
    "ReturnRevisionSnapshotSignature",
    "ReturnRevisionStreamCommitment",
    "SplitConformalCalibrator",
    "SettlementSnapshot",
    "SnapshotObservationReceipt",
    "VolatilityRegimeAssignment",
    "VolatilityRegimeDefinition",
    "WalkForwardConfig",
    "WalkForwardSplitter",
    "apply_point_in_time_corporate_actions",
    "apply_point_in_time_splits",
    "audit_regime_coverage",
    "assign_authenticated_point_in_time_volatility_regimes",
    "assign_point_in_time_volatility_regimes",
    "commit_point_in_time_returns",
    "create_return_revision_snapshot_signature",
    "dump_return_revision_snapshot_signature",
    "load_market_csv",
    "load_business_calendar",
    "load_point_in_time_market_csv",
    "load_point_in_time_settlement",
    "load_return_revision_snapshot_signature",
    "require_regime_coverage",
    "verify_return_revision_snapshot_signature",
    "verify_and_record_return_revision_snapshot",
]
