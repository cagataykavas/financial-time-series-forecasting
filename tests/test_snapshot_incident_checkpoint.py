from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from finforecast.snapshot_incident_checkpoint import (
    SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_SCHEMA,
    SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_VERIFICATION_SCHEMA,
    SnapshotIncidentLedgerCheckpointError,
    create_snapshot_incident_ledger_checkpoint,
    dump_snapshot_incident_ledger_checkpoint,
    load_snapshot_incident_ledger_checkpoint,
    verify_snapshot_incident_ledger_checkpoint,
)
from tests.test_snapshot_incidents import _conflicting_views, _record
from tests.test_snapshot_transparency import _Signer

_LEDGER_ID = "production-snapshot-gossip-incidents"
_CHECKPOINT_ID = "incident-ledger-2026-11-06-001"
_ISSUED_AT = 1_799_001_000


def _record_conflict(ledger: Path, fixture: Path, suffix: str) -> None:
    log_signer, relays, views = _conflicting_views(fixture, suffix=suffix)
    _record(ledger, log_signer, relays, views)


def _checkpoint(ledger: Path, signer: _Signer):
    return create_snapshot_incident_ledger_checkpoint(
        ledger,
        ledger_id=_LEDGER_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_ISSUED_AT,
        signer=signer,
    )


def _verify(ledger: Path, checkpoint, signer: _Signer):
    return verify_snapshot_incident_ledger_checkpoint(
        checkpoint,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_ledger_id=_LEDGER_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        incident_ledger_path=ledger,
    )


def test_signed_checkpoint_is_canonical_private_and_verifies_extension(tmp_path: Path) -> None:
    ledger = tmp_path / "incidents.sqlite3"
    _record_conflict(ledger, tmp_path / "first", "private-first")
    signer = _Signer()
    checkpoint = _checkpoint(ledger, signer)
    payload = dump_snapshot_incident_ledger_checkpoint(checkpoint)

    assert checkpoint.schema_version == SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_SCHEMA
    assert load_snapshot_incident_ledger_checkpoint(payload) == checkpoint
    assert b"private-first" not in payload
    assert b"relay-1" not in payload

    _record_conflict(ledger, tmp_path / "second", "private-second")
    evidence = _verify(ledger, checkpoint, signer)

    assert evidence["schema_version"] == SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_VERIFICATION_SCHEMA
    assert evidence["verified"] is True
    assert evidence["checkpoint_entry_count"] == 1
    assert evidence["current_entry_count"] == 2
    assert len(str(evidence["verification_sha256"])) == 64


def test_checkpoint_detects_valid_replacement_and_truncation(tmp_path: Path) -> None:
    ledger = tmp_path / "incidents.sqlite3"
    retained_prefix = tmp_path / "retained-prefix.sqlite3"
    _record_conflict(ledger, tmp_path / "first", "first")
    shutil.copyfile(ledger, retained_prefix)
    os.chmod(retained_prefix, 0o600)
    _record_conflict(ledger, tmp_path / "second", "second")
    signer = _Signer()
    checkpoint = _checkpoint(ledger, signer)

    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="truncated"):
        _verify(retained_prefix, checkpoint, signer)

    replacement = tmp_path / "replacement.sqlite3"
    _record_conflict(replacement, tmp_path / "replacement-fixture", "replacement")
    _record_conflict(replacement, tmp_path / "replacement-second", "replacement-second")
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="does not extend"):
        _verify(replacement, checkpoint, signer)


def test_signature_key_and_identity_substitution_fail_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "incidents.sqlite3"
    _record_conflict(ledger, tmp_path / "fixture", "conflict")
    signer = _Signer()
    checkpoint = _checkpoint(ledger, signer)
    other = _Signer()

    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="key revision"):
        verify_snapshot_incident_ledger_checkpoint(
            checkpoint,
            trusted_public_keys={signer.key_id: other.public_key_der},
            expected_ledger_id=_LEDGER_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            incident_ledger_path=ledger,
        )
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="signature is invalid"):
        _verify(ledger, replace(checkpoint, head_entry_sha256="f" * 64), signer)
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="ledger identity"):
        verify_snapshot_incident_ledger_checkpoint(
            checkpoint,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_ledger_id="another-ledger",
            expected_checkpoint_id=_CHECKPOINT_ID,
            incident_ledger_path=ledger,
        )


def test_checkpoint_creation_rejects_corrupt_empty_and_unsafe_ledgers(tmp_path: Path) -> None:
    signer = _Signer()
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="unavailable or unsafe"):
        _checkpoint(missing, signer)
    assert not missing.exists()

    empty = tmp_path / "empty.sqlite3"
    with sqlite3.connect(empty) as connection:
        connection.execute("PRAGMA application_id = 0x46534749")
        connection.execute("PRAGMA user_version = 1")
    os.chmod(empty, 0o600)
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="validation failed"):
        _checkpoint(empty, signer)

    ledger = tmp_path / "incidents.sqlite3"
    _record_conflict(ledger, tmp_path / "fixture", "conflict")
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "UPDATE snapshot_gossip_incidents SET detected_at_epoch = detected_at_epoch + 1"
        )
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="validation failed"):
        _checkpoint(ledger, signer)


def test_checkpoint_loader_rejects_extensions_duplicates_and_reformatting(tmp_path: Path) -> None:
    ledger = tmp_path / "incidents.sqlite3"
    _record_conflict(ledger, tmp_path / "fixture", "conflict")
    checkpoint = _checkpoint(ledger, _Signer())
    document = checkpoint.to_audit_dict()

    extended = json.dumps({**document, "extra": True}, sort_keys=True).encode()
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="fields"):
        load_snapshot_incident_ledger_checkpoint(extended)

    duplicate = dump_snapshot_incident_ledger_checkpoint(checkpoint).replace(
        b'{"algorithm":', b'{"algorithm":"ed25519-sha512","algorithm":', 1
    )
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="duplicate"):
        load_snapshot_incident_ledger_checkpoint(duplicate)

    reformatted = json.dumps(document, sort_keys=True, indent=2).encode()
    with pytest.raises(SnapshotIncidentLedgerCheckpointError, match="not canonical"):
        load_snapshot_incident_ledger_checkpoint(reformatted)


def test_signer_failures_are_sanitized(tmp_path: Path) -> None:
    ledger = tmp_path / "incidents.sqlite3"
    _record_conflict(ledger, tmp_path / "fixture", "conflict")

    class FailingSigner(_Signer):
        def sign(self, message: bytes) -> bytes:
            raise RuntimeError("private provider diagnostic")

    with pytest.raises(
        SnapshotIncidentLedgerCheckpointError, match="checkpoint signer failed"
    ) as error:
        _checkpoint(ledger, FailingSigner())
    assert "private provider diagnostic" not in str(error.value)
