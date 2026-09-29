# Prequential conformal intervals

`finforecast.prequential` turns an already-aligned sequence of point forecasts and
realized outcomes into rolling symmetric prediction intervals. It is intended for
offline replay, release validation, and a production implementation reference.

Unlike a single split-conformal fit, the interval issued at position `i` is calibrated
only from residuals whose outcomes would already be observable:

```text
available_end = i - feedback_delay
calibration = residuals[max(0, available_end - window):available_end]
```

The current outcome and every future outcome are excluded. `feedback_delay` makes the
availability boundary explicit for delayed labels or multi-step horizons. The radius is
the finite-sample conformal quantile at rank
`min(m, ceil((m + 1) * coverage))`, where `m` is the current calibration size.

## Example

```python
from finforecast.prequential import (
    PrequentialConformalPolicy,
    prequential_conformal_intervals,
)

policy = PrequentialConformalPolicy(
    coverage=0.9,
    window=250,
    min_calibration=50,
    feedback_delay=1,
)
result = prequential_conformal_intervals(actual, point_forecasts, policy=policy)
interval_frame = result.as_frame()
```

`interval_frame` deliberately omits realized outcomes. Each row exposes the positional
start and exclusive end of its calibration slice, so a reviewer can reconstruct the
information set used at issuance time. Pandas inputs must have identical, unique,
monotonically increasing indexes; the original labels are restored on the output.

## Failure and resource model

The runner fails closed with stable, non-sensitive error codes for malformed vectors,
index misalignment, non-finite values, insufficient history, and policy violations. It
also bounds:

- total input points;
- rolling calibration-window width; and
- the exact number of residual values selected across all emitted intervals.

These limits make accidental quadratic replays and unbounded untrusted artifacts
rejectable before interval construction. The implementation copies inputs and is fully
deterministic for identical ordered evidence.

## Production boundaries

This module does not claim that exchangeability holds. Volatility regimes, structural
breaks, dependent residuals, and a poorly chosen window can all invalidate nominal
coverage. Use the existing temporal coverage and breach-independence audits on a later,
untouched evaluation window before promotion.

`feedback_delay` must match the real label or horizon availability contract; setting it
too low reintroduces look-ahead bias. The routine is an offline replay, not a concurrent
state store: a live service must persist residual arrival order, make updates atomic,
and prevent duplicate labels. Symmetric intervals also do not represent asymmetric
financial loss, tradability, or expected profit.
