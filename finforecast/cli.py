from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    parser.add_argument("--rows", type=_rows, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=_non_negative_float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--evidence", action="store_true", help="build governed evaluation evidence"
    )
    args = parser.parse_args(argv)
    if args.evidence:
        result = write_evidence(args.output, rows=args.rows, seed=args.seed)
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
    close, volume = synthetic_market(args.rows, args.seed)
    data = make_supervised_features(close, volume)
    experiment = ForecastExperiment(seed=args.seed)
    predictions = experiment.walk_forward(data)
    result = experiment.evaluate(predictions, args.cost_bps)
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
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
