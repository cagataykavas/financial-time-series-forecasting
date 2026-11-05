# Snapshot checkpoint gossip

Append-only proofs and witness signatures protect one checkpoint history, but a malicious log can
still present separate, internally consistent histories to isolated promotion workflows.
`finforecast.snapshot_gossip` adds a comparison boundary for checkpoints delivered through
independently operated relay channels.

## Sign an authenticated relay observation

Each relay re-authenticates the log checkpoint against the governed log key before signing its
observation. The relay signature binds the exact checkpoint authentication and content digests,
tree size, root, checkpoint issue time, observation time, relay identity, and immutable relay-key
revision:

```python
from finforecast import create_snapshot_checkpoint_gossip_observation

risk_observation = create_snapshot_checkpoint_gossip_observation(
    candidate_checkpoint,
    "risk-governance-relay",
    risk_relay_signer,
    trusted_log_public_keys=governed_log_key_revisions,
    expected_log_id="market-data-observation-log",
    expected_checkpoint_id="checkpoint-before-promotion",
    observed_at_epoch=trusted_now_epoch,
)
```

Relay observations have strict, duplicate-free canonical JSON serialization with a 4 KiB limit.
Use independently administered KMS/HSM keys and transport paths in production.

## Reject split views before promotion

The protected workflow supplies each checkpoint together with the relay observation that names it:

```python
from finforecast import verify_gossiped_snapshot_transparency_promotion

promotion_evidence = verify_gossiped_snapshot_transparency_promotion(
    observation_receipt,
    retained_checkpoint,
    candidate_checkpoint,
    consistency_proof,
    inclusion_proof,
    witness_bundle,
    [
        (risk_checkpoint, risk_observation),
        (operations_checkpoint, operations_observation),
        (auditor_checkpoint, auditor_observation),
    ],
    trusted_log_public_keys=governed_log_key_revisions,
    trusted_witness_public_keys=governed_witness_key_revisions,
    trusted_relay_public_keys=governed_relay_key_revisions,
    expected_log_id="market-data-observation-log",
    expected_old_checkpoint_id="checkpoint-before-training",
    expected_new_checkpoint_id="checkpoint-before-promotion",
    verification_epoch=trusted_now_epoch,
    required_relays=2,
)
```

Verification re-authenticates every supplied checkpoint, requires exact observation-to-checkpoint
cross-binding, enforces distinct relay identities, key IDs, and key revisions, and rejects relay
observations older than 15 minutes. Unavailable or invalid relay keys do not count toward quorum.
After signature verification, **any** trusted authenticated disagreement fails closed—even if the
majority view independently meets the configured quorum.

The final evidence contains governed identity hashes, checkpoint/root commitments, tree size,
freshness policy, counts, and aggregate observation/key-revision digests. It excludes relay IDs,
key IDs, signatures, return values, revision IDs, and Merkle proof nodes.

## Trust boundary

This boundary proves agreement among the configured signed relay observations; it does not prove
that network paths or organizations are independent, discover a split view never delivered to a
trusted relay, authenticate the supplied verification epoch, or prevent compromised relays from
colluding. A freshly relayed checkpoint may still have an old log issue time because gossip
freshness applies to relay observation time. Deployments must separate relay control planes,
authenticate runner time, revoke relay keys out of band, retain conflicting observations, and
combine gossip evidence with the existing inclusion, consistency, and witness checks.
