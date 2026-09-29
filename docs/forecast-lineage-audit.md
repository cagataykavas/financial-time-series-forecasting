# Backfill-resistant forecast lineage audit

Offline forecast metrics are only trustworthy when each prediction existed before its target
was observed. A statistically sound walk-forward evaluator can still accept a reconstructed file
whose predictions were generated after realization. This audit makes that producer/evaluator
boundary explicit.

## Contract

Each record binds a forecast identifier, series, immutable model revision, horizon, feature cutoff,
issuance time, target time, realization time and digests of the prediction and actual value. Records
form a canonical SHA-256 chain. The gate rejects:

- issuance after the target, features newer than issuance or realization before the target;
- a target inconsistent with `feature_cutoff + cadence * horizon`;
- issue, lead-time, realization or artifact-freshness budget breaches;
- repeated forecast identities or a second forecast for the same series/model/horizon/target;
- non-monotonic per-stream records, reordered chains and changed record payloads.

Input is bounded to 2 MiB and 50,000 records. JSON duplicate keys, non-finite numbers, extra schema
fields, non-UTC timestamps and unbounded identifiers fail closed. Reports contain counts, timing
extrema, policy and digests—not raw series names, forecast identifiers, predictions or actuals.

## Artifact example

Use `chain_records` at forecast-production time; do not reconstruct the ledger from an evaluation
table. `prediction_digest` and `actual_digest` should be computed from a versioned canonical value
encoding agreed by producer and verifier.

```python
from finforecast.forecast_lineage import chain_records

artifact = {
    "schema_version": 1,
    "created_at": "2026-09-29T03:00:00Z",
    "dataset_digest": "<lowercase sha256>",
    "config_digest": "<lowercase sha256>",
    "records": chain_records(records),
}
```

Run the independent release gate:

```bash
python -m finforecast.forecast_lineage evidence.json \
  --cadence-seconds 3600 \
  --max-issue-delay-seconds 300 \
  --min-lead-seconds 30 \
  --output lineage-report.json
```

Exit code `0` means accepted, `2` means malformed evidence or policy configuration, and `3` means
well-formed evidence violated the release policy.

## Trust boundary and limitations

The chain is tamper-evident, not an identity proof. A dishonest or compromised producer can create a
new internally consistent ledger with false timestamps. Production deployment should sign the
terminal chain hash in an external transparency or timestamp service and persist forecast events in
append-only storage before targets mature. The audit also does not validate exchange calendars,
corporate actions, prediction quality, calibration or market profitability; those remain separate
data and statistical controls.
