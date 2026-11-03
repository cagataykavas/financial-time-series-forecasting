"""Detect reuse of a governed snapshot identity with conflicting signed contents."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finforecast.regime_labeling import PointInTimeReturn
from finforecast.regime_source import (
    RETURN_REVISION_SNAPSHOT_VERIFICATION_SCHEMA,
    ReturnRevisionSnapshotSignature,
    verify_return_revision_snapshot_signature,
)

SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA = "finforecast-snapshot-observation-receipt/v1"

_APPLICATION_ID = 0x4653464C
_LEDGER_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MAX_LEDGER_BYTES = 256 * 1024 * 1024
_TABLE_COLUMNS = (
    ("identity_sha256", "TEXT", 1, 1),
    ("commitment_sha256", "TEXT", 1, 0),
    ("first_observed_at_epoch", "INTEGER", 1, 0),
    ("entry_sha256", "TEXT", 1, 0),
)
_CREATE_TABLE_SQL = """CREATE TABLE snapshot_observations (
    identity_sha256 TEXT PRIMARY KEY
        CHECK(length(identity_sha256) = 64 AND identity_sha256 NOT GLOB '*[^0-9a-f]*'),
    commitment_sha256 TEXT NOT NULL
        CHECK(length(commitment_sha256) = 64 AND commitment_sha256 NOT GLOB '*[^0-9a-f]*'),
    first_observed_at_epoch INTEGER NOT NULL
        CHECK(first_observed_at_epoch BETWEEN 1 AND 9223372036854775807),
    entry_sha256 TEXT NOT NULL
        CHECK(length(entry_sha256) = 64 AND entry_sha256 NOT GLOB '*[^0-9a-f]*')
) STRICT, WITHOUT ROWID"""


class SnapshotObservationError(ValueError):
    """The observation ledger is unsafe, corrupt, or contains conflicting evidence."""

    code = "SNAPSHOT_OBSERVATION_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class SnapshotObservationReceipt:
    """Privacy-reduced evidence for a first or idempotent snapshot observation."""

    schema_version: str
    status: str
    identity_sha256: str
    commitment_sha256: str
    first_observed_at_epoch: int
    entry_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        receipt = asdict(self)
        receipt["receipt_sha256"] = _digest(receipt)
        return receipt


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotObservationError("observation evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _entry_sha256(
    identity_sha256: str,
    commitment_sha256: str,
    first_observed_at_epoch: int,
) -> str:
    return _digest(
        {
            "commitment_sha256": commitment_sha256,
            "first_observed_at_epoch": first_observed_at_epoch,
            "identity_sha256": identity_sha256,
            "schema_version": SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA,
        }
    )


def _prepare_ledger_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("ledger_path must be a filesystem path")
    ledger_path = Path(path)
    if not ledger_path.name or ledger_path.name in {".", ".."}:
        raise SnapshotObservationError("ledger path is invalid")
    parent = ledger_path.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise SnapshotObservationError("ledger parent must be an existing non-symlink directory")
    try:
        descriptor = os.open(
            ledger_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        pass
    except OSError as exc:
        raise SnapshotObservationError("ledger file cannot be created safely") from exc
    else:
        os.close(descriptor)
    try:
        ledger_stat = ledger_path.lstat()
    except OSError as exc:
        raise SnapshotObservationError("ledger file cannot be inspected") from exc
    if not stat.S_ISREG(ledger_stat.st_mode):
        raise SnapshotObservationError("ledger must be a regular non-symlink file")
    if stat.S_IMODE(ledger_stat.st_mode) & 0o077:
        raise SnapshotObservationError("ledger permissions are too broad")
    if ledger_stat.st_size > _MAX_LEDGER_BYTES:
        raise SnapshotObservationError("ledger exceeds the byte limit")
    return ledger_path


def _initialize_or_validate_schema(connection: sqlite3.Connection) -> None:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if application_id == 0 and user_version == 0:
        existing_objects = connection.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing_objects:
            raise SnapshotObservationError("ledger database is not empty or recognized")
        connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {_LEDGER_SCHEMA_VERSION}")
        connection.execute(_CREATE_TABLE_SQL)
    elif application_id != _APPLICATION_ID or user_version != _LEDGER_SCHEMA_VERSION:
        raise SnapshotObservationError("ledger database identity or schema is unsupported")
    columns = connection.execute("PRAGMA table_info(snapshot_observations)").fetchall()
    observed_columns = tuple((row[1], row[2], row[3], row[5]) for row in columns)
    if observed_columns != _TABLE_COLUMNS:
        raise SnapshotObservationError("ledger table schema is corrupt")
    objects = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    if objects != [("table", "snapshot_observations", _CREATE_TABLE_SQL)]:
        raise SnapshotObservationError("ledger database contains unexpected schema objects")


def _validate_stored_entry(row: tuple[Any, ...], identity_sha256: str) -> tuple[str, int, str]:
    commitment_sha256, first_observed_at_epoch, entry_sha256 = row
    if (
        not isinstance(commitment_sha256, str)
        or _DIGEST_RE.fullmatch(commitment_sha256) is None
        or type(first_observed_at_epoch) is not int
        or not 1 <= first_observed_at_epoch <= 2**63 - 1
        or not isinstance(entry_sha256, str)
        or _DIGEST_RE.fullmatch(entry_sha256) is None
        or not hmac.compare_digest(
            entry_sha256,
            _entry_sha256(identity_sha256, commitment_sha256, first_observed_at_epoch),
        )
    ):
        raise SnapshotObservationError("ledger entry integrity check failed")
    return commitment_sha256, first_observed_at_epoch, entry_sha256


def _record_verification(
    verification: Mapping[str, object],
    *,
    ledger_path: str | os.PathLike[str],
    observed_at_epoch: int,
) -> SnapshotObservationReceipt:
    if type(observed_at_epoch) is not int or not 1 <= observed_at_epoch <= 2**63 - 1:
        raise SnapshotObservationError("observation time is invalid")
    if verification.get("schema_version") != RETURN_REVISION_SNAPSHOT_VERIFICATION_SCHEMA:
        raise SnapshotObservationError("snapshot verification schema is unsupported")
    identity_sha256 = _digest(
        {
            "dataset_id": verification["dataset_id"],
            "snapshot_id": verification["snapshot_id"],
            "vendor_id": verification["vendor_id"],
        }
    )
    commitment_sha256 = _digest(dict(verification))
    path = _prepare_ledger_path(ledger_path)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN IMMEDIATE")
        _initialize_or_validate_schema(connection)
        row = connection.execute(
            """
            SELECT commitment_sha256, first_observed_at_epoch, entry_sha256
            FROM snapshot_observations
            WHERE identity_sha256 = ?
            """,
            (identity_sha256,),
        ).fetchone()
        if row is None:
            first_observed_at_epoch = observed_at_epoch
            entry_sha256 = _entry_sha256(
                identity_sha256,
                commitment_sha256,
                first_observed_at_epoch,
            )
            connection.execute(
                """
                INSERT INTO snapshot_observations (
                    identity_sha256,
                    commitment_sha256,
                    first_observed_at_epoch,
                    entry_sha256
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    identity_sha256,
                    commitment_sha256,
                    first_observed_at_epoch,
                    entry_sha256,
                ),
            )
            status = "recorded"
        else:
            stored_commitment, first_observed_at_epoch, entry_sha256 = _validate_stored_entry(
                row,
                identity_sha256,
            )
            if not hmac.compare_digest(stored_commitment, commitment_sha256):
                raise SnapshotObservationError(
                    "snapshot identity was previously observed with a conflicting commitment"
                )
            status = "already_observed"
        connection.execute("COMMIT")
    except SnapshotObservationError:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    except sqlite3.Error as exc:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SnapshotObservationError("ledger operation failed") from exc
    finally:
        if connection is not None:
            connection.close()
    return SnapshotObservationReceipt(
        schema_version=SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA,
        status=status,
        identity_sha256=identity_sha256,
        commitment_sha256=commitment_sha256,
        first_observed_at_epoch=first_observed_at_epoch,
        entry_sha256=entry_sha256,
    )


def verify_and_record_return_revision_snapshot(
    signature: ReturnRevisionSnapshotSignature,
    rows: Sequence[PointInTimeReturn],
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_vendor_id: str,
    expected_dataset_id: str,
    expected_snapshot_id: str,
    expected_source_schema_sha256: str,
    ledger_path: str | os.PathLike[str],
    observed_at_epoch: int,
) -> SnapshotObservationReceipt:
    """Verify one snapshot and atomically reject conflicting reuse of its identity."""

    verification = verify_return_revision_snapshot_signature(
        signature,
        rows,
        trusted_public_keys=trusted_public_keys,
        expected_vendor_id=expected_vendor_id,
        expected_dataset_id=expected_dataset_id,
        expected_snapshot_id=expected_snapshot_id,
        expected_source_schema_sha256=expected_source_schema_sha256,
    )
    return _record_verification(
        verification,
        ledger_path=ledger_path,
        observed_at_epoch=observed_at_epoch,
    )
