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
from .snapshot_consistency import (
    SnapshotTransparencyConsistencyProof,
    create_snapshot_transparency_consistency_proof,
    dump_snapshot_transparency_consistency_proof,
    load_snapshot_transparency_consistency_proof,
    verify_snapshot_transparency_append_only_inclusion,
    verify_snapshot_transparency_consistency,
)
from .snapshot_gossip import (
    SnapshotCheckpointGossipObservation,
    create_snapshot_checkpoint_gossip_observation,
    dump_snapshot_checkpoint_gossip_observation,
    load_snapshot_checkpoint_gossip_observation,
    verify_gossiped_snapshot_transparency_promotion,
    verify_snapshot_checkpoint_gossip,
)
from .snapshot_observation import (
    SnapshotObservationReceipt,
    verify_and_record_return_revision_snapshot,
)
from .snapshot_transparency import (
    SnapshotTransparencyCheckpoint,
    SnapshotTransparencyProof,
    create_snapshot_transparency_checkpoint,
    create_snapshot_transparency_proof,
    dump_snapshot_transparency_checkpoint,
    dump_snapshot_transparency_proof,
    load_snapshot_transparency_checkpoint,
    load_snapshot_transparency_proof,
    verify_snapshot_transparency_inclusion,
)
from .snapshot_witness import (
    SnapshotConsistencyWitnessApproval,
    SnapshotConsistencyWitnessBundle,
    create_snapshot_consistency_witness_bundle,
    dump_snapshot_consistency_witness_bundle,
    load_snapshot_consistency_witness_bundle,
    verify_snapshot_consistency_witness_bundle,
    verify_witnessed_snapshot_transparency_promotion,
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
    "SnapshotConsistencyWitnessApproval",
    "SnapshotConsistencyWitnessBundle",
    "SnapshotCheckpointGossipObservation",
    "SnapshotTransparencyConsistencyProof",
    "SnapshotTransparencyCheckpoint",
    "SnapshotTransparencyProof",
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
    "create_snapshot_transparency_checkpoint",
    "create_snapshot_transparency_consistency_proof",
    "create_snapshot_transparency_proof",
    "create_snapshot_consistency_witness_bundle",
    "create_snapshot_checkpoint_gossip_observation",
    "dump_return_revision_snapshot_signature",
    "dump_snapshot_transparency_checkpoint",
    "dump_snapshot_transparency_consistency_proof",
    "dump_snapshot_transparency_proof",
    "dump_snapshot_consistency_witness_bundle",
    "dump_snapshot_checkpoint_gossip_observation",
    "load_market_csv",
    "load_business_calendar",
    "load_point_in_time_market_csv",
    "load_point_in_time_settlement",
    "load_return_revision_snapshot_signature",
    "load_snapshot_transparency_checkpoint",
    "load_snapshot_transparency_consistency_proof",
    "load_snapshot_transparency_proof",
    "load_snapshot_consistency_witness_bundle",
    "load_snapshot_checkpoint_gossip_observation",
    "require_regime_coverage",
    "verify_return_revision_snapshot_signature",
    "verify_and_record_return_revision_snapshot",
    "verify_snapshot_transparency_append_only_inclusion",
    "verify_snapshot_transparency_consistency",
    "verify_snapshot_transparency_inclusion",
    "verify_snapshot_consistency_witness_bundle",
    "verify_gossiped_snapshot_transparency_promotion",
    "verify_snapshot_checkpoint_gossip",
    "verify_witnessed_snapshot_transparency_promotion",
]
