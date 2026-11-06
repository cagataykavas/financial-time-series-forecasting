"""Retain authenticated checkpoint split views in a privacy-reduced incident ledger."""

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

from finforecast.snapshot_gossip import (
    MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
    SnapshotCheckpointGossipObservation,
    _validated_snapshot_checkpoint_gossip_views,
)
from finforecast.snapshot_transparency import SnapshotTransparencyCheckpoint

SNAPSHOT_GOSSIP_INCIDENT_RECEIPT_SCHEMA = "finforecast-snapshot-gossip-incident-receipt/v1"

_APPLICATION_ID = 0x46534749
_LEDGER_SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MAX_LEDGER_BYTES = 64 * 1024 * 1024
_MAX_INCIDENTS = 10_000
_GENESIS_SHA256 = "0" * 64
_TABLE_COLUMNS = (
    ("sequence", "INTEGER", 1, 1),
    ("incident_sha256", "TEXT", 1, 0),
    ("detected_at_epoch", "INTEGER", 1, 0),
    ("identity_sha256", "TEXT", 1, 0),
    ("view_count", "INTEGER", 1, 0),
    ("view_set_sha256", "TEXT", 1, 0),
    ("valid_relays", "INTEGER", 1, 0),
    ("valid_observations_sha256", "TEXT", 1, 0),
    ("valid_relay_revisions_sha256", "TEXT", 1, 0),
    ("policy_sha256", "TEXT", 1, 0),
    ("previous_entry_sha256", "TEXT", 1, 0),
    ("entry_sha256", "TEXT", 1, 0),
)
_CREATE_TABLE_SQL = """CREATE TABLE snapshot_gossip_incidents (
    sequence INTEGER PRIMARY KEY
        CHECK(sequence BETWEEN 1 AND 10000),
    incident_sha256 TEXT NOT NULL UNIQUE
        CHECK(length(incident_sha256) = 64 AND incident_sha256 NOT GLOB '*[^0-9a-f]*'),
    detected_at_epoch INTEGER NOT NULL
        CHECK(detected_at_epoch BETWEEN 1 AND 9223372036854775807),
    identity_sha256 TEXT NOT NULL
        CHECK(length(identity_sha256) = 64 AND identity_sha256 NOT GLOB '*[^0-9a-f]*'),
    view_count INTEGER NOT NULL CHECK(view_count BETWEEN 2 AND 16),
    view_set_sha256 TEXT NOT NULL
        CHECK(length(view_set_sha256) = 64 AND view_set_sha256 NOT GLOB '*[^0-9a-f]*'),
    valid_relays INTEGER NOT NULL CHECK(valid_relays BETWEEN 2 AND 16),
    valid_observations_sha256 TEXT NOT NULL
        CHECK(length(valid_observations_sha256) = 64
            AND valid_observations_sha256 NOT GLOB '*[^0-9a-f]*'),
    valid_relay_revisions_sha256 TEXT NOT NULL
        CHECK(length(valid_relay_revisions_sha256) = 64
            AND valid_relay_revisions_sha256 NOT GLOB '*[^0-9a-f]*'),
    policy_sha256 TEXT NOT NULL
        CHECK(length(policy_sha256) = 64 AND policy_sha256 NOT GLOB '*[^0-9a-f]*'),
    previous_entry_sha256 TEXT NOT NULL
        CHECK(length(previous_entry_sha256) = 64
            AND previous_entry_sha256 NOT GLOB '*[^0-9a-f]*'),
    entry_sha256 TEXT NOT NULL
        CHECK(length(entry_sha256) = 64 AND entry_sha256 NOT GLOB '*[^0-9a-f]*')
) STRICT, WITHOUT ROWID"""


class SnapshotGossipIncidentError(ValueError):
    """Conflict evidence is absent or its incident ledger is unsafe."""

    code = "SNAPSHOT_GOSSIP_INCIDENT_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class SnapshotGossipIncidentReceipt:
    """Privacy-reduced receipt for one append-only split-view incident."""

    schema_version: str
    status: str
    sequence: int
    detected_at_epoch: int
    identity_sha256: str
    view_count: int
    view_set_sha256: str
    valid_relays: int
    valid_observations_sha256: str
    valid_relay_revisions_sha256: str
    policy_sha256: str
    incident_sha256: str
    previous_entry_sha256: str
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
        raise SnapshotGossipIncidentError("incident evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _entry_sha256(
    *,
    sequence: int,
    incident_sha256: str,
    detected_at_epoch: int,
    identity_sha256: str,
    view_count: int,
    view_set_sha256: str,
    valid_relays: int,
    valid_observations_sha256: str,
    valid_relay_revisions_sha256: str,
    policy_sha256: str,
    previous_entry_sha256: str,
) -> str:
    return _digest(
        {
            "detected_at_epoch": detected_at_epoch,
            "identity_sha256": identity_sha256,
            "incident_sha256": incident_sha256,
            "policy_sha256": policy_sha256,
            "previous_entry_sha256": previous_entry_sha256,
            "schema_version": SNAPSHOT_GOSSIP_INCIDENT_RECEIPT_SCHEMA,
            "sequence": sequence,
            "valid_observations_sha256": valid_observations_sha256,
            "valid_relay_revisions_sha256": valid_relay_revisions_sha256,
            "valid_relays": valid_relays,
            "view_count": view_count,
            "view_set_sha256": view_set_sha256,
        }
    )


def _incident_ledger_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("incident_ledger_path must be a filesystem path")
    ledger_path = Path(path)
    if not ledger_path.name or ledger_path.name in {".", ".."}:
        raise SnapshotGossipIncidentError("incident ledger path is invalid")
    parent = ledger_path.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise SnapshotGossipIncidentError(
            "incident ledger parent must be an existing non-symlink directory"
        )
    return ledger_path


def _inspect_existing_ledger_path(path: str | os.PathLike[str]) -> Path:
    ledger_path = _incident_ledger_path(path)
    try:
        ledger_stat = ledger_path.lstat()
    except OSError as exc:
        raise SnapshotGossipIncidentError("incident ledger cannot be inspected") from exc
    if not stat.S_ISREG(ledger_stat.st_mode):
        raise SnapshotGossipIncidentError("incident ledger must be a regular non-symlink file")
    if stat.S_IMODE(ledger_stat.st_mode) & 0o077:
        raise SnapshotGossipIncidentError("incident ledger permissions are too broad")
    if ledger_stat.st_size > _MAX_LEDGER_BYTES:
        raise SnapshotGossipIncidentError("incident ledger exceeds the byte limit")
    return ledger_path


def _prepare_ledger_path(path: str | os.PathLike[str]) -> Path:
    ledger_path = _incident_ledger_path(path)
    try:
        descriptor = os.open(
            ledger_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        pass
    except OSError as exc:
        raise SnapshotGossipIncidentError("incident ledger cannot be created safely") from exc
    else:
        os.close(descriptor)
    return _inspect_existing_ledger_path(ledger_path)


def _initialize_or_validate_schema(connection: sqlite3.Connection) -> None:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if application_id == 0 and user_version == 0:
        existing_objects = connection.execute(
            "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        if existing_objects:
            raise SnapshotGossipIncidentError("incident ledger is not empty or recognized")
        connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {_LEDGER_SCHEMA_VERSION}")
        connection.execute(_CREATE_TABLE_SQL)
    elif application_id != _APPLICATION_ID or user_version != _LEDGER_SCHEMA_VERSION:
        raise SnapshotGossipIncidentError("incident ledger identity or schema is unsupported")
    columns = connection.execute("PRAGMA table_info(snapshot_gossip_incidents)").fetchall()
    observed_columns = tuple((row[1], row[2], row[3], row[5]) for row in columns)
    if observed_columns != _TABLE_COLUMNS:
        raise SnapshotGossipIncidentError("incident ledger table schema is corrupt")
    objects = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    if objects != [("table", "snapshot_gossip_incidents", _CREATE_TABLE_SQL)]:
        raise SnapshotGossipIncidentError("incident ledger contains unexpected schema objects")


def _row_values(row: tuple[Any, ...]) -> dict[str, object]:
    names = tuple(item[0] for item in _TABLE_COLUMNS)
    return dict(zip(names, row, strict=True))


def _validate_chain(connection: sqlite3.Connection) -> list[dict[str, object]]:
    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
        raise SnapshotGossipIncidentError("incident ledger integrity check failed")
    rows = connection.execute(
        """
        SELECT sequence, incident_sha256, detected_at_epoch, identity_sha256,
               view_count, view_set_sha256, valid_relays, valid_observations_sha256,
               valid_relay_revisions_sha256, policy_sha256, previous_entry_sha256,
               entry_sha256
        FROM snapshot_gossip_incidents
        ORDER BY sequence
        """
    ).fetchall()
    if len(rows) > _MAX_INCIDENTS:
        raise SnapshotGossipIncidentError("incident ledger exceeds the entry limit")
    validated: list[dict[str, object]] = []
    previous = _GENESIS_SHA256
    for expected_sequence, row in enumerate(rows, start=1):
        item = _row_values(row)
        digest_fields = (
            "incident_sha256",
            "identity_sha256",
            "view_set_sha256",
            "valid_observations_sha256",
            "valid_relay_revisions_sha256",
            "policy_sha256",
            "previous_entry_sha256",
            "entry_sha256",
        )
        if (
            item["sequence"] != expected_sequence
            or any(
                not isinstance(item[name], str) or _DIGEST_RE.fullmatch(item[name]) is None
                for name in digest_fields
            )
            or type(item["detected_at_epoch"]) is not int
            or not 1 <= item["detected_at_epoch"] <= 2**63 - 1
            or type(item["view_count"]) is not int
            or not 2 <= item["view_count"] <= 16
            or type(item["valid_relays"]) is not int
            or not 2 <= item["valid_relays"] <= 16
            or not hmac.compare_digest(str(item["previous_entry_sha256"]), previous)
        ):
            raise SnapshotGossipIncidentError("incident ledger chain is corrupt")
        expected_entry = _entry_sha256(
            sequence=int(item["sequence"]),
            incident_sha256=str(item["incident_sha256"]),
            detected_at_epoch=int(item["detected_at_epoch"]),
            identity_sha256=str(item["identity_sha256"]),
            view_count=int(item["view_count"]),
            view_set_sha256=str(item["view_set_sha256"]),
            valid_relays=int(item["valid_relays"]),
            valid_observations_sha256=str(item["valid_observations_sha256"]),
            valid_relay_revisions_sha256=str(item["valid_relay_revisions_sha256"]),
            policy_sha256=str(item["policy_sha256"]),
            previous_entry_sha256=str(item["previous_entry_sha256"]),
        )
        if not hmac.compare_digest(str(item["entry_sha256"]), expected_entry):
            raise SnapshotGossipIncidentError("incident ledger entry integrity check failed")
        previous = expected_entry
        validated.append(item)
    return validated


def _receipt(item: Mapping[str, object], status: str) -> SnapshotGossipIncidentReceipt:
    return SnapshotGossipIncidentReceipt(
        schema_version=SNAPSHOT_GOSSIP_INCIDENT_RECEIPT_SCHEMA,
        status=status,
        sequence=int(item["sequence"]),
        detected_at_epoch=int(item["detected_at_epoch"]),
        identity_sha256=str(item["identity_sha256"]),
        view_count=int(item["view_count"]),
        view_set_sha256=str(item["view_set_sha256"]),
        valid_relays=int(item["valid_relays"]),
        valid_observations_sha256=str(item["valid_observations_sha256"]),
        valid_relay_revisions_sha256=str(item["valid_relay_revisions_sha256"]),
        policy_sha256=str(item["policy_sha256"]),
        incident_sha256=str(item["incident_sha256"]),
        previous_entry_sha256=str(item["previous_entry_sha256"]),
        entry_sha256=str(item["entry_sha256"]),
    )


def record_snapshot_checkpoint_gossip_conflict(
    views: Sequence[tuple[SnapshotTransparencyCheckpoint, SnapshotCheckpointGossipObservation]],
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_relay_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
    incident_ledger_path: str | os.PathLike[str],
    detected_at_epoch: int,
    required_relays: int = 2,
    max_observation_age_seconds: int = MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
) -> SnapshotGossipIncidentReceipt:
    """Authenticate a split view and append its privacy-reduced incident evidence."""

    valid, _verified_at = _validated_snapshot_checkpoint_gossip_views(
        views,
        trusted_log_public_keys=trusted_log_public_keys,
        trusted_relay_public_keys=trusted_relay_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_checkpoint_id,
        verification_epoch=detected_at_epoch,
        required_relays=required_relays,
        max_observation_age_seconds=max_observation_age_seconds,
    )
    checkpoint_views = sorted(
        {
            (
                str(authentication["checkpoint_sha256"]),
                int(authentication["tree_size"]),
                str(authentication["root_sha256"]),
            )
            for _observation, authentication in valid
        }
    )
    if len(checkpoint_views) < 2:
        raise SnapshotGossipIncidentError("gossip views do not contain an authenticated conflict")
    identity_sha256 = _digest(
        {
            "checkpoint_id_sha256": valid[0][1]["checkpoint_id_sha256"],
            "log_id_sha256": valid[0][1]["log_id_sha256"],
        }
    )
    view_set_sha256 = _digest(checkpoint_views)
    valid_observations_sha256 = _digest(
        sorted(observation.observation_sha256 for observation, _authentication in valid)
    )
    valid_relay_revisions_sha256 = _digest(
        sorted(observation.public_key_sha256 for observation, _authentication in valid)
    )
    policy_sha256 = _digest(
        {
            "max_observation_age_seconds": max_observation_age_seconds,
            "required_relays": required_relays,
        }
    )
    incident_sha256 = _digest(
        {
            "identity_sha256": identity_sha256,
            "policy_sha256": policy_sha256,
            "valid_observations_sha256": valid_observations_sha256,
            "valid_relay_revisions_sha256": valid_relay_revisions_sha256,
            "valid_relays": len(valid),
            "view_count": len(checkpoint_views),
            "view_set_sha256": view_set_sha256,
        }
    )
    path = _prepare_ledger_path(incident_ledger_path)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN IMMEDIATE")
        _initialize_or_validate_schema(connection)
        chain = _validate_chain(connection)
        existing = next(
            (item for item in chain if item["incident_sha256"] == incident_sha256),
            None,
        )
        if existing is not None:
            connection.execute("COMMIT")
            return _receipt(existing, "already_recorded")
        if len(chain) >= _MAX_INCIDENTS:
            raise SnapshotGossipIncidentError("incident ledger exceeds the entry limit")
        sequence = len(chain) + 1
        previous_entry_sha256 = str(chain[-1]["entry_sha256"]) if chain else _GENESIS_SHA256
        entry_sha256 = _entry_sha256(
            sequence=sequence,
            incident_sha256=incident_sha256,
            detected_at_epoch=detected_at_epoch,
            identity_sha256=identity_sha256,
            view_count=len(checkpoint_views),
            view_set_sha256=view_set_sha256,
            valid_relays=len(valid),
            valid_observations_sha256=valid_observations_sha256,
            valid_relay_revisions_sha256=valid_relay_revisions_sha256,
            policy_sha256=policy_sha256,
            previous_entry_sha256=previous_entry_sha256,
        )
        values: dict[str, object] = {
            "sequence": sequence,
            "incident_sha256": incident_sha256,
            "detected_at_epoch": detected_at_epoch,
            "identity_sha256": identity_sha256,
            "view_count": len(checkpoint_views),
            "view_set_sha256": view_set_sha256,
            "valid_relays": len(valid),
            "valid_observations_sha256": valid_observations_sha256,
            "valid_relay_revisions_sha256": valid_relay_revisions_sha256,
            "policy_sha256": policy_sha256,
            "previous_entry_sha256": previous_entry_sha256,
            "entry_sha256": entry_sha256,
        }
        connection.execute(
            """
            INSERT INTO snapshot_gossip_incidents (
                sequence, incident_sha256, detected_at_epoch, identity_sha256,
                view_count, view_set_sha256, valid_relays, valid_observations_sha256,
                valid_relay_revisions_sha256, policy_sha256, previous_entry_sha256,
                entry_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(values[name] for name, _type, _notnull, _pk in _TABLE_COLUMNS),
        )
        connection.execute("COMMIT")
        return _receipt(values, "recorded")
    except SnapshotGossipIncidentError:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    except sqlite3.Error as exc:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SnapshotGossipIncidentError("incident ledger operation failed") from exc
    finally:
        if connection is not None:
            connection.close()
