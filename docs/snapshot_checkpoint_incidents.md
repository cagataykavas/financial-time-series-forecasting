# Snapshot checkpoint incident retention

Checkpoint gossip fails promotion when independently authenticated relays report conflicting log
views. `finforecast.snapshot_incidents` adds durable, privacy-reduced forensic retention for that
failure without storing checkpoint documents, relay signatures, market returns, revision IDs, or
governed identifiers.

## Authenticate and record a conflict

The recorder runs the same log authentication, relay-signature, freshness, distinct-key, and
quorum checks as the promotion gate. Agreeing views, invalid signatures, and insufficient quorums
cannot create an incident.

```python
from finforecast import record_snapshot_checkpoint_gossip_conflict

incident = record_snapshot_checkpoint_gossip_conflict(
    [
        (risk_checkpoint, risk_observation),
        (operations_checkpoint, operations_observation),
        (auditor_checkpoint, auditor_observation),
    ],
    trusted_log_public_keys=governed_log_key_revisions,
    trusted_relay_public_keys=governed_relay_key_revisions,
    expected_log_id="market-data-observation-log",
    expected_checkpoint_id="checkpoint-before-promotion",
    incident_ledger_path="/var/lib/finforecast/checkpoint-incidents.sqlite3",
    detected_at_epoch=trusted_now_epoch,
    required_relays=2,
)
```

The receipt and ledger retain only:

- a digest of the governed log/checkpoint identity;
- the number and digest of distinct authenticated checkpoint views;
- valid-relay count and digests of the exact observations and key revisions;
- a digest of the freshness/quorum policy;
- first detection time, sequence number, previous-entry digest, and entry digest.

Exact evidence replay returns `already_recorded` and the original sequence. A different conflict is
appended with a monotonic sequence and a digest link to the preceding entry. Every write validates
the complete chain under `BEGIN IMMEDIATE`, uses full synchronous commits, and rejects corrupt or
extended schemas, unsafe permissions, symlinks, oversized files, and more than 10,000 incidents.

## Operating boundary

Create the ledger in an existing protected directory and retain it in immutable, independently
controlled storage. The application-level chain exposes accidental corruption and missing or
reordered entries, but a host administrator who can replace the entire database can rewrite the
chain. The ledger is not signed, transparency-logged, or independently timestamped; detection time
still comes from the protected runner. A conflict is evidence of authenticated inconsistent views,
not proof of which log operator or relay acted maliciously.
