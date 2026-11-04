# Snapshot consistency witness quorum

An authenticated consistency proof detects rewritten history relative to a checkpoint already held
by one verifier. A malicious log can still present separate, internally consistent histories to
isolated operators. `finforecast.snapshot_witness` adds an independently administered quorum over
the exact verified old→new transition used for promotion.

## Create the witness bundle

Each witness adapter implements the existing `SnapshotTransparencySigner` protocol. A witness
should verify the log signatures and consistency proof against its independently retained history
before signing. Use separate KMS/HSM keys and control planes in production:

```python
from finforecast import create_snapshot_consistency_witness_bundle

witness_bundle = create_snapshot_consistency_witness_bundle(
    retained_checkpoint,
    candidate_checkpoint,
    consistency_proof,
    {
        "risk-governance": risk_kms_signer,
        "platform-operations": operations_hsm_signer,
        "external-auditor": auditor_signer,
    },
    trusted_log_public_keys=governed_log_key_revisions,
    expected_log_id="market-data-observation-log",
    expected_old_checkpoint_id="checkpoint-before-training",
    expected_new_checkpoint_id="checkpoint-before-promotion",
    required_witnesses=2,
)
```

The builder re-runs the complete log-signature and RFC 6962-style consistency verification before
requesting signatures. Every approval binds:

- the exact consistency-verification digest;
- the digest of the complete, canonically ordered witness set;
- the required quorum;
- the witness ID, key ID, and immutable public-key digest; and
- a versioned signature purpose and domain.

Distinct witness IDs, key IDs, and public-key revisions are mandatory. Threshold or witness-set
changes invalidate every signature. The complete bundle is self-digested and has a strict,
duplicate-free, canonical JSON loader with a 16 KiB limit.

## Require witnesses at promotion

The protected promotion workflow supplies the log and witness public-key registries separately:

```python
from finforecast import (
    load_snapshot_consistency_witness_bundle,
    verify_witnessed_snapshot_transparency_promotion,
)

witness_bundle = load_snapshot_consistency_witness_bundle(retained_witness_bytes)
promotion_evidence = verify_witnessed_snapshot_transparency_promotion(
    observation_receipt,
    retained_checkpoint,
    candidate_checkpoint,
    consistency_proof,
    inclusion_proof,
    witness_bundle,
    trusted_log_public_keys=governed_log_key_revisions,
    trusted_witness_public_keys=governed_witness_key_revisions,
    expected_log_id="market-data-observation-log",
    expected_old_checkpoint_id="checkpoint-before-training",
    expected_new_checkpoint_id="checkpoint-before-promotion",
)
```

The verifier independently repeats append-only consistency and observation inclusion, verifies the
witness quorum, and requires both paths to name the same consistency result. Missing, unavailable,
substituted, or invalid witness keys do not count toward the quorum. Provider failures do not copy
private diagnostics into boundary errors.

The final promotion evidence contains only transition, observation, witness-bundle, and verifier
digests plus tree sizes and witness counts. It excludes log/checkpoint/witness/key identifiers,
signatures, returns, revisions, vendor identifiers, and Merkle proof nodes.

## Trust boundary

Quorum signatures do not prove organizational independence or that a witness followed its stated
verification procedure. A deployment must separate key administration, distribute and revoke
trusted public keys out of band, and require each witness to maintain durable checkpoint state. The
module does not gossip checkpoints, authenticate time, provide key transparency, prove immutable
retention, or prevent a quorum of compromised witnesses from cosigning conflicting histories.
