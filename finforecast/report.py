from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _model_row(name: str, metrics: dict[str, float]) -> str:
    cells = (
        escape(name),
        f"{metrics['mae']:.5f}",
        f"{metrics['rmse']:.5f}",
        f"{metrics['directional_accuracy']:.1%}",
        f"{metrics['strategy_sharpe']:.2f}",
        f"{metrics['net_cumulative_return']:.1%}",
    )
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    rows = "".join(_model_row(name, metrics) for name, metrics in result["models"].items())
    bootstrap = result["gradient_boosting_vs_naive_mae_bootstrap"]
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Forecasting Report</title>
  <style>
    body {{
      background:#09101d; color:#edf3ff; font-family:Inter,system-ui,sans-serif; padding:32px
    }}
    main {{ max-width:1100px; margin:auto }}
    .muted {{ color:#98a7c1 }}
    table {{ width:100%; border-collapse:collapse; background:#121c30; margin:20px 0 }}
    th,td {{ padding:11px; border-bottom:1px solid #293953; text-align:left }}
    th {{ color:#a9bfff }}
    .box {{ background:#121c30; border:1px solid #293953; border-radius:14px; padding:18px }}
  </style>
</head>
<body><main>
  <p class="muted">Synthetic walk-forward forecasting research · not investment advice</p>
  <h1>Financial Time-Series Forecasting</h1>
  <table>
    <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Direction</th>
    <th>Strategy Sharpe</th><th>Net return</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="box">
    <h2>Gradient boosting vs zero-return baseline</h2>
    <p>Observed MAE difference:
      <strong>{bootstrap["observed_mae_difference"]:.6f}</strong>
    </p>
    <p>Bootstrap 90% interval:
      [{bootstrap["bootstrap_p05"]:.6f}, {bootstrap["bootstrap_p95"]:.6f}]
    </p>
    <p class="muted">Negative values favor gradient boosting. This is a paired bootstrap
      diagnostic, not a Diebold–Mariano test.</p>
  </div>
</main></body>
</html>
"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
