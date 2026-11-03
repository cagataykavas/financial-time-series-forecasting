from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from finforecast.regime_labeling import PointInTimeReturn
from finforecast.regime_source import create_return_revision_snapshot_signature
from finforecast.snapshot_observation import (
    SnapshotObservationError,
    verify_and_record_return_revision_snapshot,
)

_SCHEMA_DIGEST = "a" * 64
_OBSERVED_AT = 1_799_000_100


class _Signer:
    key_id = "vendor-key-2026"

    def __init__(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key_der = self.private_key.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)


def _rows() -> list[PointInTimeReturn]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        PointInTimeReturn(
            observed_at=start + timedelta(days=position),
            available_at=start + timedelta(days=position, hours=1),
            revision_id=f"private-revision-{position}",
            value=value,
        )
        for position, value in enumerate((-0.1, 0.0, 0.1))
    ]


def _signature(rows: list[PointInTimeReturn], signer: _Signer):
    return create_return_revision_snapshot_signature(
        rows,
        vendor_id="vendor.example",
        dataset_id="daily-total-returns",
        snapshot_id="2026-01-10-r1",
        issued_at_epoch=1_799_000_000,
        source_schema_sha256=_SCHEMA_DIGEST,
        signer=signer,
    )


def _record(path: Path, rows: list[PointInTimeReturn], signer: _Signer, *, observed_at: int):
    return verify_and_record_return_revision_snapshot(
        _signature(rows, signer),
        rows,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_vendor_id="vendor.example",
        expected_dataset_id="daily-total-returns",
        expected_snapshot_id="2026-01-10-r1",
        expected_source_schema_sha256=_SCHEMA_DIGEST,
        ledger_path=path,
        observed_at_epoch=observed_at,
    )


def test_first_observation_and_exact_replay_are_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    rows = _rows()
    signer = _Signer()

    first = _record(ledger, rows, signer, observed_at=_OBSERVED_AT)
    replay = _record(ledger, rows, signer, observed_at=_OBSERVED_AT + 10)

    assert first.status == "recorded"
    assert replay.status == "already_observed"
    assert replay.first_observed_at_epoch == first.first_observed_at_epoch == _OBSERVED_AT
    assert replay.entry_sha256 == first.entry_sha256
    assert replay.commitment_sha256 == first.commitment_sha256
    assert ledger.stat().st_mode & 0o777 == 0o600


def test_conflicting_signed_snapshot_is_rejected_without_overwrite(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    rows = _rows()
    signer = _Signer()
    first = _record(ledger, rows, signer, observed_at=_OBSERVED_AT)

    changed_rows = [replace(rows[0], value=0.8), *rows[1:]]
    with pytest.raises(SnapshotObservationError, match="conflicting commitment"):
        _record(ledger, changed_rows, signer, observed_at=_OBSERVED_AT + 10)

    replay = _record(ledger, rows, signer, observed_at=_OBSERVED_AT + 20)
    assert replay.status == "already_observed"
    assert replay.entry_sha256 == first.entry_sha256


def test_receipt_and_ledger_do_not_retain_returns_or_revision_ids(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    receipt = _record(ledger, _rows(), _Signer(), observed_at=_OBSERVED_AT)

    serialized = json.dumps(receipt.to_audit_dict(), sort_keys=True)
    database_bytes = ledger.read_bytes()
    assert "private-revision" not in serialized
    assert b"private-revision" not in database_bytes
    assert b"daily-total-returns" not in database_bytes
    assert b"vendor.example" not in database_bytes
    assert '"value"' not in serialized
    assert len(receipt.to_audit_dict()["receipt_sha256"]) == 64


def test_concurrent_first_observations_converge_on_one_entry(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    rows = _rows()
    signer = _Signer()

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(
            pool.map(
                lambda position: _record(
                    ledger,
                    rows,
                    signer,
                    observed_at=_OBSERVED_AT + position,
                ),
                range(6),
            )
        )

    assert [receipt.status for receipt in receipts].count("recorded") == 1
    assert len({receipt.entry_sha256 for receipt in receipts}) == 1
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM snapshot_observations").fetchone() == (1,)


def test_corrupt_entry_is_detected_before_replay(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    rows = _rows()
    signer = _Signer()
    _record(ledger, rows, signer, observed_at=_OBSERVED_AT)
    with sqlite3.connect(ledger) as connection:
        connection.execute(
            "UPDATE snapshot_observations SET first_observed_at_epoch = ?",
            (_OBSERVED_AT + 1,),
        )

    with pytest.raises(SnapshotObservationError, match="integrity check failed"):
        _record(ledger, rows, signer, observed_at=_OBSERVED_AT + 2)


def test_unrecognized_database_and_broad_permissions_are_rejected(tmp_path: Path) -> None:
    unrecognized = tmp_path / "unrecognized.sqlite3"
    with sqlite3.connect(unrecognized) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    os.chmod(unrecognized, 0o600)

    with pytest.raises(SnapshotObservationError, match="not empty or recognized"):
        _record(unrecognized, _rows(), _Signer(), observed_at=_OBSERVED_AT)

    broad = tmp_path / "broad.sqlite3"
    broad.touch(mode=0o644)
    os.chmod(broad, 0o644)
    with pytest.raises(SnapshotObservationError, match="permissions are too broad"):
        _record(broad, _rows(), _Signer(), observed_at=_OBSERVED_AT)


def test_unexpected_schema_object_is_rejected(tmp_path: Path) -> None:
    ledger = tmp_path / "snapshot-observations.sqlite3"
    rows = _rows()
    signer = _Signer()
    _record(ledger, rows, signer, observed_at=_OBSERVED_AT)
    with sqlite3.connect(ledger) as connection:
        connection.execute("CREATE TABLE injected (value TEXT)")

    with pytest.raises(SnapshotObservationError, match="unexpected schema objects"):
        _record(ledger, rows, signer, observed_at=_OBSERVED_AT + 1)


def test_symlink_and_invalid_observation_time_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite3"
    target.touch(mode=0o600)
    symlink = tmp_path / "ledger.sqlite3"
    symlink.symlink_to(target)

    with pytest.raises(SnapshotObservationError, match="regular non-symlink"):
        _record(symlink, _rows(), _Signer(), observed_at=_OBSERVED_AT)
    with pytest.raises(SnapshotObservationError, match="observation time is invalid"):
        _record(tmp_path / "other.sqlite3", _rows(), _Signer(), observed_at=True)
