# Horizon-aware forecast comparison audit

Aggregate loss improvement does not establish that a challenger consistently beats a baseline. Financial forecast losses are paired in time and can remain serially dependent, especially when forecast horizons overlap. Treating those observations as independent can make a weak improvement look certain.

This audit implements a one-sided Diebold–Mariano comparison over the paired differential:

```text
d[t] = baseline_loss[t] - candidate_loss[t]
```

Positive values favor the candidate. The long-run variance uses a Newey–West estimator with Bartlett weights. Every case declares its forecast horizon and HAC lag; the lag must be at least `horizon - 1`. The reported statistic includes the Harvey–Leybourne–Newbold small-sample correction. Promotion requires both a prespecified one-sided significance level and a minimum relative mean-loss improvement, so statistical confidence cannot replace materiality.

## Evidence contract

Schema `finforecast-diebold-mariano/v1` binds the comparison to:

- content-addressed baseline and candidate models;
- a named loss and content-addressed loss configuration;
- a benchmark identity and timezone-aware generation time;
- one or more unique cases containing strictly ordered timestamps and paired, finite, non-negative losses;
- explicit forecast horizon and HAC lag per case.

Malformed, duplicate, stale, future-dated, non-finite, unordered, undersized, or over-budget evidence fails closed. The test is also undefined when the estimated HAC variance is zero or non-positive; the audit refuses to manufacture a p-value in that case.

## CLI

```bash
python -m finforecast.diebold_mariano paired-losses.json --output dm-report.json
```

Exit codes are `0` for accepted evidence, `2` for a valid policy rejection, and `3` for malformed or statistically undefined evidence. Report replacement is atomic. Case, benchmark, and loss identifiers are hashed, and raw timestamps or losses are not copied into the report.

## Interpretation and limitations

The asymptotic normal approximation can be poor in small or heavy-tailed samples. Newey–West lag choice must be fixed before evaluation; `horizon - 1` handles mechanical overlap but may not capture longer dependence. Multiple assets, horizons, losses, or candidate models require a prespecified multiplicity policy. Structural breaks can invalidate historical inference.

A pass establishes neither profitability nor live-capital suitability. Loss construction, point-in-time data, transaction costs, tuning isolation, corporate actions, market calendars, and model selection remain upstream responsibilities. The audit trusts the producer to compute the declared paired losses correctly; SHA-256 identities bind evidence but do not authenticate it.

## Next integration step

Emit the artifact directly from the existing walk-forward prediction table, prespecify horizon-specific HAC lags and economic thresholds, and apply family-wise or false-discovery control when a release evaluates multiple horizons or assets.
