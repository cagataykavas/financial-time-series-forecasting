from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from finforecast.snapshot_incidents import (
    SNAPSHOT_GOSSIP_INCIDENT_RECEIPT_SCHEMA,
    SnapshotGossipIncidentError,
    record_snapshot_checkpoint_gossip_conflict,
)
from finforecast.snapshot_transparency import (
    SnapshotTransparencyError,
    create_snapshot_transparency_checkpoint,
)
from tests.test_snapshot_gossip import (
    _VERIFIED_AT,
    _observation,
    _trusted_relay_keys,
    _views,
)
from tests.test_snapshot_transparency import _CHECKPOINT_EPOCH, _CHECKPOINT_ID, _LOG_ID, _entry


def _conflicting_views(tmp_path: Path, *, suffix: str = "alternate"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_signer, _checkpoint, relays, views = _views(tmp_path)
    alternate = create_snapshot_transparency_checkpoint(
        [_entry(f"{suffix}-entry-{position}") for position in range(5)],
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH,
        signer=log_signer,
    )
    views[2] = (
        alternate,
        _observation(alternate, "relay-3", relays["relay-3"], log_signer),
    )
    return log_signer, relays, views


def _record(
    ledger: Path,
    log_signer,
    relays,
    views,
):
    return record_snapshot_checkpoint_gossip_conflict(
        views,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_relay_public_keys=_trusted_relay_keys(relays),
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        incident_ledger_path=ledger,
        detected_at_epoch=_VERIFIED_AT,
    )


def test_conflict_is_recorded_privately_and_exact_replay_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "checkpoint-incidents.sqlite3"
    log_signer, relays, views = _conflicting_views(tmp_path)

    first = _record(ledger, log_signer, relays, views)
    replay = _record(ledger, log_signer, relays, views)

    assert first.schema_version == SNAPSHOT_GOSSIP_INCIDENT_RECEIPT_SCHEMA
    assert first.status == "recorded"
    assert replay.status == "already_recorded"
    assert replay.sequence == first.sequence == 1
    assert replay.entry_sha256 == first.entry_sha256
    assert first.view_count == 2
    assert first.valid_relays == 3
    assert ledger.stat().st_mode & 0o777 == 0o600
    retained = ledger.read_bytes()
    serialized = json.dumps(first.to_audit_dict(), sort_keys=True)
    assert _LOG_ID not in serialized
    assert _CHECKPOINT_ID not in serialized
    assert "relay-1" not in serialized
    assert relays["relay-1"].key_id not in serialized
    assert _LOG_ID.encode() not in retained
    assert b"alternate-entry" not in retained
    assert len(first.to_audit_dict()["receipt_sha256"]) == 64


def test_distinct_conflicts_append_a_hash_chained_sequence(tmp_path: Path) -> None:
    ledger = tmp_path / "checkpoint-incidents.sqlite3"
    first_log, first_relays, first_views = _conflicting_views(
        tmp_path / "first-fixture", suffix="first"
    )
    second_log, second_relays, second_views = _conflicting_views(
        tmp_path / "second-fixture", suffix="second"
    )

    first = _record(ledger, first_log, first_relays, first_views)
    second = _record(ledger, second_log, second_relays, second_views)

    assert second.status == "recorded"
    assert second.sequence == 2
    assert second.previous_entry_sha256 == first.entry_sha256
    assert second.incident_sha256 != first.incident_sha256
    with sqlite3.connect(ledger) as connection:
        assert connection.execute(
            """SELECT sequence, previous_entry_sha256
               FROM snapshot_gossip_incidents
               ORDER BY sequence"""
        ).fetchall() == [(1, "0" * 64), (2, first.entry_sha256)]


def test_agreement_and_insufficient_trust_cannot_create_incidents(tmp_path: Path) -> None:
    agreeing_ledger = tmp_path / "agreeing.sqlite3"
    log_signer, _checkpoint, relays, agreeing = _views(tmp_path)

    with pytest.raises(SnapshotGossipIncidentError, match="do not contain"):
        _record(agreeing_ledger, log_signer, relays, agreeing)
    assert not agreeing_ledger.exists()

    conflict_ledger = tmp_path / "untrusted.sqlite3"
    log_signer, relays, conflict = _conflicting_views(tmp_path / "conflict-fixture")
    only_one_key = {relays["relay-1"].key_id: relays["relay-1"].public_key_der}
    with pytest.raises(SnapshotTransparencyError, match="quorum is not satisfied"):
        record_snapshot_checkpoint_gossip_conflict(
            conflict,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=only_one_key,
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            incident_ledger_path=conflict_ledger,
            detected_at_epoch=_VERIFIED_AT,
        )
    assert not conflict_ledger.exists()


def test_concurrent_exact_conflicts_converge_on_one_entry(tmp_path: Path) -> None:
    ledger = tmp_path / "checkpoint-incidents.sqlite3"
    log_signer, relays, views = _conflicting_views(tmp_path)

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(
            pool.map(
                lambda _position: _record(ledger, log_signer, relays, views),
                range(6),
            )
        )

    assert [receipt.status for receipt in receipts].count("recorded") == 1
    assert len({receipt.entry_sha256 for receipt in receipts}) == 1
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM snapshot_gossip_incidents").fetchone() == (
            1,
        )


def test_corrupt_chain_is_detected_before_replay(tmp_path: Path) -> None:
    ledger = tmp_path / "checkpoint-incidents.sqlite3"
    log_signer, relays, views = _conflicting_views(tmp_path)
    _record(ledger, log_signer, relays, views)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "UPDATE snapshot_gossip_incidents SET detected_at_epoch = detected_at_epoch + 1"
        )

    with pytest.raises(SnapshotGossipIncidentError, match="entry integrity check failed"):
        _record(ledger, log_signer, relays, views)


def test_unrecognized_schema_broad_permissions_and_symlinks_are_rejected(tmp_path: Path) -> None:
    log_signer, relays, views = _conflicting_views(tmp_path)
    unrecognized = tmp_path / "unrecognized.sqlite3"
    with sqlite3.connect(unrecognized) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    os.chmod(unrecognized, 0o600)
    with pytest.raises(SnapshotGossipIncidentError, match="not empty or recognized"):
        _record(unrecognized, log_signer, relays, views)

    broad = tmp_path / "broad.sqlite3"
    broad.touch(mode=0o644)
    os.chmod(broad, 0o644)
    with pytest.raises(SnapshotGossipIncidentError, match="permissions are too broad"):
        _record(broad, log_signer, relays, views)

    target = tmp_path / "target.sqlite3"
    target.touch(mode=0o600)
    symlink = tmp_path / "linked.sqlite3"
    symlink.symlink_to(target)
    with pytest.raises(SnapshotGossipIncidentError, match="regular non-symlink"):
        _record(symlink, log_signer, relays, views)


def test_unexpected_schema_object_is_rejected(tmp_path: Path) -> None:
    ledger = tmp_path / "checkpoint-incidents.sqlite3"
    log_signer, relays, views = _conflicting_views(tmp_path)
    _record(ledger, log_signer, relays, views)
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE injected (value TEXT)")

    with pytest.raises(SnapshotGossipIncidentError, match="unexpected schema objects"):
        _record(ledger, log_signer, relays, views)
