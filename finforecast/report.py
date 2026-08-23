from __future__ import annotations

from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    rows = "".join(
        f"<tr><td>{name}</td><td>{m['mae']:.5f}</td><td>{m['rmse']:.5f}</td><td>{m['directional_accuracy']:.1%}</td><td>{m['strategy_sharpe']:.2f}</td><td>{m['net_cumulative_return']:.1%}</td></tr>"
        for name, m in result["models"].items()
    )
    boot = result["gradient_boosting_vs_naive_mae_bootstrap"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forecasting Report</title><style>body{{background:#09101d;color:#edf3ff;font-family:Inter,system-ui,sans-serif;padding:32px}}main{{max-width:1100px;margin:auto}}.muted{{color:#98a7c1}}table{{width:100%;border-collapse:collapse;background:#121c30;margin:20px 0}}th,td{{padding:11px;border-bottom:1px solid #293953;text-align:left}}th{{color:#a9bfff}}.box{{background:#121c30;border:1px solid #293953;border-radius:14px;padding:18px}}</style></head><body><main><p class="muted">Synthetic walk-forward forecasting research · not investment advice</p><h1>Financial Time-Series Forecasting</h1><table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Direction</th><th>Strategy Sharpe</th><th>Net return</th></tr></thead><tbody>{rows}</tbody></table><div class="box"><h2>Gradient boosting vs zero-return baseline</h2><p>Observed MAE difference: <strong>{boot['observed_mae_difference']:.6f}</strong></p><p>Bootstrap 90% interval: [{boot['bootstrap_p05']:.6f}, {boot['bootstrap_p95']:.6f}]</p><p class="muted">Negative values favor gradient boosting. This is a paired bootstrap diagnostic, not a Diebold–Mariano test.</p></div></main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
