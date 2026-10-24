from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .corporate_actions import apply_point_in_time_splits
from .data import MarketData, load_market_csv
from .engine import ForecastExperiment
from .evidence import write_evidence
from .features import make_supervised_features
from .report import render_report
from .synthetic import synthetic_market


def _rows(value: str) -> int:
    parsed = int(value)
    if parsed < 320:
        raise argparse.ArgumentTypeError("rows must be at least 320")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage-aware financial forecasting project")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--rows", type=_rows, default=1000)
    source.add_argument("--csv", type=Path, help="ordered market CSV to evaluate")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--close-column", default="close")
    parser.add_argument(
        "--as-of",
        help="timezone-aware cutoff for a revision-bearing point-in-time CSV",
    )
    parser.add_argument("--available-at-column", default="available_at")
    parser.add_argument("--revision-column", default="revision_id")
    parser.add_argument(
        "--corporate-actions-csv",
        type=Path,
        help="revision-bearing split feed applied at the --as-of cutoff",
    )
    parser.add_argument("--symbol", help="stable uppercase symbol for corporate actions")
    volume = parser.add_mutually_exclusive_group()
    volume.add_argument("--volume-column", default="volume")
    volume.add_argument("--no-volume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=_non_negative_float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--evidence", action="store_true", help="build governed evaluation evidence"
    )
    args = parser.parse_args(argv)

    if (args.corporate_actions_csv is None) != (args.symbol is None):
        parser.error("--corporate-actions-csv and --symbol must be supplied together")
    if args.corporate_actions_csv is not None and (args.csv is None or args.as_of is None):
        parser.error("corporate actions require --csv and --as-of")

    market: MarketData | None = None
    if args.csv is not None:
        try:
            market = load_market_csv(
                args.csv,
                timestamp_column=args.timestamp_column,
                close_column=args.close_column,
                volume_column=None if args.no_volume else args.volume_column,
                as_of=args.as_of,
                available_at_column=(args.available_at_column if args.as_of else None),
                revision_column=(args.revision_column if args.as_of else None),
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        if len(market.close) < 320:
            parser.error("market CSV must contain at least 320 observations")
        if args.corporate_actions_csv is not None:
            try:
                market = apply_point_in_time_splits(
                    market,
                    args.corporate_actions_csv,
                    as_of=args.as_of,
                    symbol=args.symbol,
                )
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))

    if args.evidence:
        result = write_evidence(
            args.output,
            rows=args.rows,
            seed=args.seed,
            market_data=market,
        )
        print(
            json.dumps(
                {
                    "evidence": str(args.output),
                    "decision": result["promotion_decision"],
                },
                indent=2,
            )
        )
        return 0

    if market is None:
        close, volume_values = synthetic_market(args.rows, args.seed)
        dataset_metadata = {
            "kind": "deterministic_synthetic_regime_shift",
            "seed": args.seed,
            "raw_observations": args.rows,
        }
    else:
        close, volume_values = market.close, market.volume
        dataset_metadata = market.metadata()
    data = make_supervised_features(close, volume_values)
    experiment = ForecastExperiment(seed=args.seed)
    predictions = experiment.walk_forward(data)
    result = experiment.evaluate(predictions, args.cost_bps)
    result["dataset"] = dataset_metadata
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "forecast_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    render_report(result, args.output / "forecast_report.html")
    print(
        json.dumps(
            {
                "observations": result["observations"],
                "models": result["models"],
                "dataset": result["dataset"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
