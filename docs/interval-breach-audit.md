# Forecast interval breach audit

Aggregate interval coverage is necessary but not sufficient. A 90% interval can
hit its nominal rate while concentrating nearly every miss in one market
regime. That pattern matters operationally: consecutive breaches are evidence
that an interval can become unreliable exactly when uncertainty is elevated.

This module provides a bounded, model-free release audit for marginal coverage
and breach independence. It complements temporal coverage windows; it does not
replace them.

## Evidence contract

```json
{
  "schema_version": 1,
  "target_coverage": 0.9,
  "observations": [
    {
      "timestamp": "2026-01-02T16:00:00Z",
      "actual": 0.004,
      "lower": -0.006,
      "upper": 0.008
    }
  ]
}
```

Timestamps must be timezone-aware, unique and strictly increasing. Numeric
values must be finite and every lower bound must be no greater than its upper
bound. Values on an interval boundary count as covered. The loader also rejects
duplicate JSON fields and oversized artifacts.

Run the audit in CI or a promotion workflow:

```bash
python -m finforecast.breach_audit interval-evidence.json \
  --output artifacts/interval-breach-audit.json
```

Exit codes distinguish the outcomes:

- `0`: accepted by policy;
- `2`: valid evidence rejected by at least one gate;
- `3`: malformed evidence or invalid policy.

## Gates and evidence

The report contains:

- Kupiec's likelihood-ratio test for unconditional coverage;
- Christoffersen's first-order likelihood-ratio test for breach independence;
- their two-degree-of-freedom conditional-coverage statistic;
- covered→covered, covered→breach, breach→covered and breach→breach counts;
- the longest consecutive breach streak;
- separate lower- and upper-tail breach counts;
- bounded breach timestamps and a canonical SHA-256 evidence digest.

The release decision rejects marginal miscoverage, statistically detectable
breach dependence, or a breach streak above the configured operational budget.
The expected breach count must exceed a minimum evidence floor; a tiny sample is
invalid rather than an easy green result. Stable reason codes make the output
suitable for automation.

## Calibration and limitations

The significance level, evidence horizon and streak budget must be
prespecified per forecast horizon and operating decision. Testing many assets,
horizons or interval levels requires a multiple-testing policy. P-values should
not be tuned after observing the same evaluation sequence.

The first-order Markov alternative only detects one form of dependence and can
have low power. Overlapping forecast horizons create mechanical breach
dependence; use non-overlapping observations or a horizon-aware method in that
case. Adaptive interval recalibration, regime selection and rolling monitoring
also change the null assumptions. Exact small-sample or simulation-calibrated
tests may be preferable near policy boundaries.

Passing shows that the supplied sequence did not violate these release gates.
It does not establish exchangeability, future coverage, profitability, correct
market data, execution realism or suitability for live capital. The evidence
producer and storage channel remain outside this module's trust boundary.
