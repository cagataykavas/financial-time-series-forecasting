"""Sign and verify independently retained incident-ledger checkpoints."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.snapshot_incidents import (
    SnapshotGossipIncidentError,
    _initialize_or_validate_schema,
    _inspect_existing_ledger_path,
    _validate_chain,
)

SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_SCHEMA = (
    "finforecast-snapshot-gossip-incident-ledger-checkpoint/v1"
)
SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_VERIFICATION_SCHEMA = (
    "finforecast-snapshot-gossip-incident-ledger-checkpoint-verification/v1"
)
SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_PURPOSE = "snapshot-gossip-incident-ledger-checkpoint"
SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_ALGORITHM = "ed25519-sha512"

_DOMAIN = b"finforecast/snapshot-gossip-incident-ledger-checkpoint/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_PUBLIC_KEY_BYTES = 8_192
_MAX_CHECKPOINT_BYTES = 4_096
_CHECKPOINT_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "ledger_id",
    "checkpoint_id",
    "entry_count",
    "head_entry_sha256",
    "issued_at_epoch",
    "key_id",
    "public_key_sha256",
    "signature",
}


class SnapshotIncidentLedgerCheckpointError(ValueError):
    """A ledger checkpoint is malformed, unauthenticated, or inconsistent."""

    code = "SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class SnapshotIncidentLedgerCheckpointSigner(Protocol):
    """External Ed25519 signer controlled by the incident-retention workflow."""

    key_id: str
    public_key_der: bytes

    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class SnapshotIncidentLedgerCheckpoint:
    """Signed commitment to one validated incident-ledger prefix."""

    schema_version: str
    purpose: str
    algorithm: str
    ledger_id: str
    checkpoint_id: str
    entry_count: int
    head_entry_sha256: str
    issued_at_epoch: int
    key_id: str
    public_key_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def checkpoint_sha256(self) -> str:
        return hashlib.sha256(dump_snapshot_incident_ledger_checkpoint(self)).hexdigest()


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
        raise SnapshotIncidentLedgerCheckpointError(
            "incident ledger checkpoint is not canonical JSON"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SnapshotIncidentLedgerCheckpointError(f"{name} is invalid")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SnapshotIncidentLedgerCheckpointError(f"{name} is invalid")
    return value


def _load_public_key(encoded: object) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= _MAX_PUBLIC_KEY_BYTES:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint public key is not Ed25519")
    return key


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise SnapshotIncidentLedgerCheckpointError("signer returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or _SIGNATURE_RE.fullmatch(value) is None:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint signature encoding is malformed")
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotIncidentLedgerCheckpointError(
            "checkpoint signature encoding is malformed"
        ) from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint signature encoding is malformed")
    return decoded


def _unsigned(checkpoint: SnapshotIncidentLedgerCheckpoint) -> dict[str, object]:
    document = checkpoint.to_audit_dict()
    document.pop("signature")
    return document


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(document))


def _validate_checkpoint(checkpoint: SnapshotIncidentLedgerCheckpoint) -> None:
    if not isinstance(checkpoint, SnapshotIncidentLedgerCheckpoint):
        raise TypeError("checkpoint must be a SnapshotIncidentLedgerCheckpoint")
    document = checkpoint.to_audit_dict()
    if set(document) != _CHECKPOINT_FIELDS:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint fields do not match the schema")
    if (
        checkpoint.schema_version != SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_SCHEMA
        or checkpoint.purpose != SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_PURPOSE
        or checkpoint.algorithm != SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_ALGORITHM
    ):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint policy is unsupported")
    _identifier(checkpoint.ledger_id, "ledger_id")
    _identifier(checkpoint.checkpoint_id, "checkpoint_id")
    _identifier(checkpoint.key_id, "key_id")
    _require_digest(checkpoint.head_entry_sha256, "head_entry_sha256")
    _require_digest(checkpoint.public_key_sha256, "public_key_sha256")
    _decode_signature(checkpoint.signature)
    if (
        type(checkpoint.entry_count) is not int
        or not 1 <= checkpoint.entry_count <= 10_000
        or type(checkpoint.issued_at_epoch) is not int
        or not 1 <= checkpoint.issued_at_epoch <= 2**63 - 1
    ):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint metadata is malformed")


def _load_validated_chain(path: str | os.PathLike[str]) -> list[dict[str, object]]:
    try:
        ledger_path = _inspect_existing_ledger_path(path)
    except (SnapshotGossipIncidentError, TypeError) as exc:
        raise SnapshotIncidentLedgerCheckpointError(
            "incident ledger is unavailable or unsafe"
        ) from exc
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(ledger_path, isolation_level=None, timeout=5.0)
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        _initialize_or_validate_schema(connection)
        chain = _validate_chain(connection)
        connection.execute("COMMIT")
    except (sqlite3.Error, SnapshotGossipIncidentError) as exc:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise SnapshotIncidentLedgerCheckpointError("incident ledger validation failed") from exc
    finally:
        if connection is not None:
            connection.close()
    if not chain:
        raise SnapshotIncidentLedgerCheckpointError("incident ledger has no checkpointable entries")
    return chain


def create_snapshot_incident_ledger_checkpoint(
    incident_ledger_path: str | os.PathLike[str],
    *,
    ledger_id: str,
    checkpoint_id: str,
    issued_at_epoch: int,
    signer: SnapshotIncidentLedgerCheckpointSigner,
) -> SnapshotIncidentLedgerCheckpoint:
    """Validate the complete ledger and sign its current prefix head."""

    normalized_ledger_id = _identifier(ledger_id, "ledger_id")
    normalized_checkpoint_id = _identifier(checkpoint_id, "checkpoint_id")
    if type(issued_at_epoch) is not int or not 1 <= issued_at_epoch <= 2**63 - 1:
        raise SnapshotIncidentLedgerCheckpointError("issued_at_epoch is invalid")
    chain = _load_validated_chain(incident_ledger_path)
    try:
        key_id = signer.key_id
        public_key_der = signer.public_key_der
    except Exception as exc:
        raise SnapshotIncidentLedgerCheckpointError("signer metadata is unavailable") from exc
    _load_public_key(public_key_der)
    unsigned: dict[str, object] = {
        "schema_version": SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_SCHEMA,
        "purpose": SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_PURPOSE,
        "algorithm": SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_ALGORITHM,
        "ledger_id": normalized_ledger_id,
        "checkpoint_id": normalized_checkpoint_id,
        "entry_count": len(chain),
        "head_entry_sha256": str(chain[-1]["entry_sha256"]),
        "issued_at_epoch": issued_at_epoch,
        "key_id": _identifier(key_id, "key_id"),
        "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
    }
    try:
        signature = signer.sign(_message(unsigned))
    except Exception as exc:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint signer failed") from exc
    checkpoint = SnapshotIncidentLedgerCheckpoint(
        **unsigned,
        signature=_encode_signature(signature),
    )
    _validate_checkpoint(checkpoint)
    return checkpoint


def verify_snapshot_incident_ledger_checkpoint(
    checkpoint: SnapshotIncidentLedgerCheckpoint,
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_ledger_id: str,
    expected_checkpoint_id: str,
    incident_ledger_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Authenticate a retained checkpoint and prove the current ledger extends it."""

    _validate_checkpoint(checkpoint)
    if not isinstance(trusted_public_keys, Mapping) or not trusted_public_keys:
        raise SnapshotIncidentLedgerCheckpointError("trusted checkpoint keys are invalid")
    public_key_der = trusted_public_keys.get(checkpoint.key_id)
    public_key = _load_public_key(public_key_der)
    expected_revision = hashlib.sha256(public_key_der).hexdigest()
    if not hmac.compare_digest(expected_revision, checkpoint.public_key_sha256):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint key revision is not trusted")
    try:
        public_key.verify(_decode_signature(checkpoint.signature), _message(_unsigned(checkpoint)))
    except InvalidSignature as exc:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint signature is invalid") from exc
    if checkpoint.ledger_id != _identifier(expected_ledger_id, "expected_ledger_id"):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint ledger identity is not trusted")
    if checkpoint.checkpoint_id != _identifier(expected_checkpoint_id, "expected_checkpoint_id"):
        raise SnapshotIncidentLedgerCheckpointError("checkpoint identity is not trusted")

    chain = _load_validated_chain(incident_ledger_path)
    if len(chain) < checkpoint.entry_count:
        raise SnapshotIncidentLedgerCheckpointError(
            "incident ledger was truncated after checkpointing"
        )
    retained_head = str(chain[checkpoint.entry_count - 1]["entry_sha256"])
    if not hmac.compare_digest(retained_head, checkpoint.head_entry_sha256):
        raise SnapshotIncidentLedgerCheckpointError(
            "incident ledger does not extend the signed checkpoint"
        )
    evidence: dict[str, object] = {
        "schema_version": SNAPSHOT_INCIDENT_LEDGER_CHECKPOINT_VERIFICATION_SCHEMA,
        "verified": True,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "ledger_id_sha256": hashlib.sha256(checkpoint.ledger_id.encode("ascii")).hexdigest(),
        "checkpoint_id_sha256": hashlib.sha256(
            checkpoint.checkpoint_id.encode("ascii")
        ).hexdigest(),
        "checkpoint_entry_count": checkpoint.entry_count,
        "current_entry_count": len(chain),
        "checkpoint_head_entry_sha256": checkpoint.head_entry_sha256,
        "current_head_entry_sha256": str(chain[-1]["entry_sha256"]),
        "key_revision_sha256": checkpoint.public_key_sha256,
    }
    evidence["verification_sha256"] = _digest(evidence)
    return evidence


def dump_snapshot_incident_ledger_checkpoint(
    checkpoint: SnapshotIncidentLedgerCheckpoint,
) -> bytes:
    """Serialize one validated checkpoint as bounded canonical JSON."""

    _validate_checkpoint(checkpoint)
    payload = _canonical_bytes(checkpoint.to_audit_dict())
    if len(payload) > _MAX_CHECKPOINT_BYTES:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint exceeds the byte limit")
    return payload


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotIncidentLedgerCheckpointError("checkpoint contains duplicate fields")
        result[key] = value
    return result


def load_snapshot_incident_ledger_checkpoint(payload: bytes) -> SnapshotIncidentLedgerCheckpoint:
    """Load strict canonical checkpoint JSON without accepting extensions."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_CHECKPOINT_BYTES:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint payload size is invalid")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicate_fields)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint is not valid JSON") from exc
    if not isinstance(document, Mapping) or set(document) != _CHECKPOINT_FIELDS:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint fields do not match the schema")
    try:
        checkpoint = SnapshotIncidentLedgerCheckpoint(**document)
    except TypeError as exc:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint fields are invalid") from exc
    _validate_checkpoint(checkpoint)
    if dump_snapshot_incident_ledger_checkpoint(checkpoint) != payload:
        raise SnapshotIncidentLedgerCheckpointError("checkpoint JSON is not canonical")
    return checkpoint
