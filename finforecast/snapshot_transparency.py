"""Verify local snapshot observations against signed transparency checkpoints."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.snapshot_observation import (
    SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA,
    SnapshotObservationReceipt,
)

SNAPSHOT_TRANSPARENCY_CHECKPOINT_SCHEMA = "finforecast-snapshot-transparency-checkpoint/v1"
SNAPSHOT_TRANSPARENCY_CHECKPOINT_AUTH_SCHEMA = (
    "finforecast-snapshot-transparency-checkpoint-authentication/v1"
)
SNAPSHOT_TRANSPARENCY_PROOF_SCHEMA = "finforecast-snapshot-transparency-proof/v1"
SNAPSHOT_TRANSPARENCY_VERIFICATION_SCHEMA = "finforecast-snapshot-transparency-verification/v1"
SNAPSHOT_TRANSPARENCY_PURPOSE = "snapshot-observation-transparency-checkpoint"
SNAPSHOT_TRANSPARENCY_ALGORITHM = "ed25519-sha512"

_DOMAIN = b"finforecast/snapshot-transparency-checkpoint/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_TREE_SIZE = 1_000_000
_MAX_PUBLIC_KEY_BYTES = 8_192
_MAX_CHECKPOINT_BYTES = 4_096
_MAX_PROOF_BYTES = 4_096
_CHECKPOINT_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "log_id",
    "checkpoint_id",
    "tree_size",
    "root_sha256",
    "issued_at_epoch",
    "key_id",
    "public_key_sha256",
    "signature",
}
_PROOF_FIELDS = {"schema_version", "leaf_index", "tree_size", "audit_path"}


class SnapshotTransparencyError(ValueError):
    """The checkpoint or inclusion proof is malformed or unauthenticated."""

    code = "SNAPSHOT_TRANSPARENCY_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class SnapshotTransparencySigner(Protocol):
    """External Ed25519 signer controlled by the transparency-log operator."""

    key_id: str
    public_key_der: bytes

    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class SnapshotTransparencyCheckpoint:
    schema_version: str
    purpose: str
    algorithm: str
    log_id: str
    checkpoint_id: str
    tree_size: int
    root_sha256: str
    issued_at_epoch: int
    key_id: str
    public_key_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def checkpoint_sha256(self) -> str:
        return hashlib.sha256(dump_snapshot_transparency_checkpoint(self)).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotTransparencyProof:
    schema_version: str
    leaf_index: int
    tree_size: int
    audit_path: tuple[str, ...]


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
        raise SnapshotTransparencyError("transparency evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


def _load_public_key(encoded: object) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= _MAX_PUBLIC_KEY_BYTES:
        raise SnapshotTransparencyError("public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise SnapshotTransparencyError("public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SnapshotTransparencyError("public key algorithm is not Ed25519")
    return key


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise SnapshotTransparencyError("signer returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or _SIGNATURE_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError("signature encoding is malformed")
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotTransparencyError("signature encoding is malformed") from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise SnapshotTransparencyError("signature encoding is malformed")
    return decoded


def _leaf_hash(entry_sha256: str) -> bytes:
    return hashlib.sha256(b"\x00" + bytes.fromhex(entry_sha256)).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(value: int) -> int:
    return 1 << ((value - 1).bit_length() - 1)


def _tree_root(leaves: Sequence[bytes]) -> bytes:
    if len(leaves) == 1:
        return leaves[0]
    split = _largest_power_of_two_less_than(len(leaves))
    return _node_hash(_tree_root(leaves[:split]), _tree_root(leaves[split:]))


def _audit_path(leaves: Sequence[bytes], leaf_index: int) -> list[bytes]:
    if len(leaves) == 1:
        return []
    split = _largest_power_of_two_less_than(len(leaves))
    if leaf_index < split:
        return [*_audit_path(leaves[:split], leaf_index), _tree_root(leaves[split:])]
    return [*_audit_path(leaves[split:], leaf_index - split), _tree_root(leaves[:split])]


def _audit_path_length(tree_size: int, leaf_index: int) -> int:
    if tree_size == 1:
        return 0
    split = _largest_power_of_two_less_than(tree_size)
    if leaf_index < split:
        return 1 + _audit_path_length(split, leaf_index)
    return 1 + _audit_path_length(tree_size - split, leaf_index - split)


def _normalize_entries(entry_sha256s: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(entry_sha256s, (str, bytes))
        or not isinstance(entry_sha256s, Sequence)
        or not 1 <= len(entry_sha256s) <= _MAX_TREE_SIZE
    ):
        raise SnapshotTransparencyError("transparency entries are invalid")
    entries = tuple(_require_digest(value, "entry_sha256") for value in entry_sha256s)
    if len(set(entries)) != len(entries):
        raise SnapshotTransparencyError("transparency entries contain duplicates")
    return entries


def _unsigned(checkpoint: SnapshotTransparencyCheckpoint) -> dict[str, object]:
    document = checkpoint.to_audit_dict()
    document.pop("signature")
    return document


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(document))


def _validate_checkpoint(checkpoint: SnapshotTransparencyCheckpoint) -> None:
    if not isinstance(checkpoint, SnapshotTransparencyCheckpoint):
        raise TypeError("checkpoint must be a SnapshotTransparencyCheckpoint")
    document = checkpoint.to_audit_dict()
    if set(document) != _CHECKPOINT_FIELDS:
        raise SnapshotTransparencyError("checkpoint fields do not match the schema")
    if (
        checkpoint.schema_version != SNAPSHOT_TRANSPARENCY_CHECKPOINT_SCHEMA
        or checkpoint.purpose != SNAPSHOT_TRANSPARENCY_PURPOSE
        or checkpoint.algorithm != SNAPSHOT_TRANSPARENCY_ALGORITHM
    ):
        raise SnapshotTransparencyError("checkpoint policy is unsupported")
    _identifier(checkpoint.log_id, "log_id")
    _identifier(checkpoint.checkpoint_id, "checkpoint_id")
    _identifier(checkpoint.key_id, "key_id")
    _require_digest(checkpoint.root_sha256, "root_sha256")
    _require_digest(checkpoint.public_key_sha256, "public_key_sha256")
    _decode_signature(checkpoint.signature)
    if (
        type(checkpoint.tree_size) is not int
        or not 1 <= checkpoint.tree_size <= _MAX_TREE_SIZE
        or type(checkpoint.issued_at_epoch) is not int
        or not 1 <= checkpoint.issued_at_epoch <= 2**63 - 1
    ):
        raise SnapshotTransparencyError("checkpoint metadata is malformed")


def create_snapshot_transparency_checkpoint(
    entry_sha256s: Sequence[str],
    *,
    log_id: str,
    checkpoint_id: str,
    issued_at_epoch: int,
    signer: SnapshotTransparencySigner,
) -> SnapshotTransparencyCheckpoint:
    """Create a signed checkpoint over one ordered, duplicate-free log prefix."""

    entries = _normalize_entries(entry_sha256s)
    try:
        key_id = signer.key_id
        public_key_der = signer.public_key_der
    except Exception as exc:
        raise SnapshotTransparencyError("signer metadata is unavailable") from exc
    public_key = _load_public_key(public_key_der)
    unsigned: dict[str, object] = {
        "schema_version": SNAPSHOT_TRANSPARENCY_CHECKPOINT_SCHEMA,
        "purpose": SNAPSHOT_TRANSPARENCY_PURPOSE,
        "algorithm": SNAPSHOT_TRANSPARENCY_ALGORITHM,
        "log_id": log_id,
        "checkpoint_id": checkpoint_id,
        "tree_size": len(entries),
        "root_sha256": _tree_root(tuple(_leaf_hash(value) for value in entries)).hex(),
        "issued_at_epoch": issued_at_epoch,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
    }
    candidate = SnapshotTransparencyCheckpoint(**unsigned, signature="A" * 86)
    _validate_checkpoint(candidate)
    try:
        signature = signer.sign(_message(unsigned))
    except Exception as exc:
        raise SnapshotTransparencyError("checkpoint signing failed") from exc
    encoded = _encode_signature(signature)
    try:
        public_key.verify(signature, _message(unsigned))
    except InvalidSignature as exc:
        raise SnapshotTransparencyError(
            "signer output does not match the retained public key"
        ) from exc
    return SnapshotTransparencyCheckpoint(**unsigned, signature=encoded)


def create_snapshot_transparency_proof(
    entry_sha256s: Sequence[str],
    *,
    leaf_index: int,
) -> SnapshotTransparencyProof:
    """Build a CT-style inclusion path for a test or transparency-log implementation."""

    entries = _normalize_entries(entry_sha256s)
    if type(leaf_index) is not int or not 0 <= leaf_index < len(entries):
        raise SnapshotTransparencyError("leaf index is invalid")
    leaves = tuple(_leaf_hash(value) for value in entries)
    return SnapshotTransparencyProof(
        schema_version=SNAPSHOT_TRANSPARENCY_PROOF_SCHEMA,
        leaf_index=leaf_index,
        tree_size=len(entries),
        audit_path=tuple(value.hex() for value in _audit_path(leaves, leaf_index)),
    )


def _validate_proof(proof: SnapshotTransparencyProof) -> None:
    if not isinstance(proof, SnapshotTransparencyProof):
        raise TypeError("proof must be a SnapshotTransparencyProof")
    if (
        proof.schema_version != SNAPSHOT_TRANSPARENCY_PROOF_SCHEMA
        or type(proof.leaf_index) is not int
        or type(proof.tree_size) is not int
        or not 1 <= proof.tree_size <= _MAX_TREE_SIZE
        or not 0 <= proof.leaf_index < proof.tree_size
        or not isinstance(proof.audit_path, tuple)
        or len(proof.audit_path) != _audit_path_length(proof.tree_size, proof.leaf_index)
    ):
        raise SnapshotTransparencyError("inclusion proof metadata is invalid")
    for value in proof.audit_path:
        _require_digest(value, "audit path node")


def _validate_observation_receipt(receipt: SnapshotObservationReceipt) -> dict[str, object]:
    if not isinstance(receipt, SnapshotObservationReceipt):
        raise TypeError("receipt must be a SnapshotObservationReceipt")
    document = receipt.to_audit_dict()
    if (
        receipt.schema_version != SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA
        or receipt.status not in {"recorded", "already_observed"}
        or type(receipt.first_observed_at_epoch) is not int
        or not 1 <= receipt.first_observed_at_epoch <= 2**63 - 1
    ):
        raise SnapshotTransparencyError("observation receipt metadata is invalid")
    for name in ("identity_sha256", "commitment_sha256", "entry_sha256"):
        _require_digest(getattr(receipt, name), name)
    expected_entry = _digest(
        {
            "commitment_sha256": receipt.commitment_sha256,
            "first_observed_at_epoch": receipt.first_observed_at_epoch,
            "identity_sha256": receipt.identity_sha256,
            "schema_version": SNAPSHOT_OBSERVATION_RECEIPT_SCHEMA,
        }
    )
    claimed_receipt = document.pop("receipt_sha256")
    if not hmac.compare_digest(receipt.entry_sha256, expected_entry) or not hmac.compare_digest(
        str(claimed_receipt), _digest(document)
    ):
        raise SnapshotTransparencyError("observation receipt integrity check failed")
    document["receipt_sha256"] = claimed_receipt
    return document


def _verify_inclusion(
    entry_sha256: str,
    proof: SnapshotTransparencyProof,
    expected_root_sha256: str,
) -> None:
    _validate_proof(proof)
    path = tuple(bytes.fromhex(value) for value in proof.audit_path)
    node = _leaf_hash(entry_sha256)
    leaf = proof.leaf_index
    last = proof.tree_size - 1
    for sibling in path:
        if leaf & 1 or leaf == last:
            node = _node_hash(sibling, node)
            while leaf != 0 and leaf & 1 == 0:
                leaf >>= 1
                last >>= 1
        else:
            node = _node_hash(node, sibling)
        leaf >>= 1
        last >>= 1
    if last != 0 or not hmac.compare_digest(node.hex(), expected_root_sha256):
        raise SnapshotTransparencyError("inclusion proof does not match the checkpoint")


def authenticate_snapshot_transparency_checkpoint(
    checkpoint: SnapshotTransparencyCheckpoint,
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
) -> dict[str, object]:
    """Authenticate one governed checkpoint without making an inclusion claim."""

    _validate_checkpoint(checkpoint)
    if not hmac.compare_digest(
        checkpoint.log_id, _identifier(expected_log_id, "expected_log_id")
    ) or not hmac.compare_digest(
        checkpoint.checkpoint_id,
        _identifier(expected_checkpoint_id, "expected_checkpoint_id"),
    ):
        raise SnapshotTransparencyError("checkpoint does not match governance")
    if not isinstance(trusted_public_keys, Mapping) or checkpoint.key_id not in trusted_public_keys:
        raise SnapshotTransparencyError("trusted checkpoint key is unavailable")
    public_key_der = trusted_public_keys[checkpoint.key_id]
    public_key = _load_public_key(public_key_der)
    if not hmac.compare_digest(
        hashlib.sha256(public_key_der).hexdigest(), checkpoint.public_key_sha256
    ):
        raise SnapshotTransparencyError("trusted checkpoint key revision does not match")
    try:
        public_key.verify(_decode_signature(checkpoint.signature), _message(_unsigned(checkpoint)))
    except InvalidSignature as exc:
        raise SnapshotTransparencyError("checkpoint signature does not verify") from exc
    authentication: dict[str, object] = {
        "schema_version": SNAPSHOT_TRANSPARENCY_CHECKPOINT_AUTH_SCHEMA,
        "authenticated": True,
        "log_id_sha256": hashlib.sha256(checkpoint.log_id.encode("ascii")).hexdigest(),
        "checkpoint_id_sha256": hashlib.sha256(
            checkpoint.checkpoint_id.encode("ascii")
        ).hexdigest(),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "tree_size": checkpoint.tree_size,
        "root_sha256": checkpoint.root_sha256,
        "issued_at_epoch": checkpoint.issued_at_epoch,
        "public_key_sha256": checkpoint.public_key_sha256,
    }
    authentication["authentication_sha256"] = _digest(authentication)
    return authentication


def verify_snapshot_transparency_inclusion(
    receipt: SnapshotObservationReceipt,
    checkpoint: SnapshotTransparencyCheckpoint,
    proof: SnapshotTransparencyProof,
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
) -> dict[str, object]:
    """Authenticate a checkpoint and prove one local observation entry is included."""

    receipt_document = _validate_observation_receipt(receipt)
    authentication = authenticate_snapshot_transparency_checkpoint(
        checkpoint,
        trusted_public_keys=trusted_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_checkpoint_id,
    )
    if not isinstance(proof, SnapshotTransparencyProof):
        raise TypeError("proof must be a SnapshotTransparencyProof")
    if (
        checkpoint.tree_size != proof.tree_size
        or checkpoint.issued_at_epoch < receipt.first_observed_at_epoch
    ):
        raise SnapshotTransparencyError("checkpoint does not match governance or proof")
    _verify_inclusion(receipt.entry_sha256, proof, checkpoint.root_sha256)
    verification: dict[str, object] = {
        "schema_version": SNAPSHOT_TRANSPARENCY_VERIFICATION_SCHEMA,
        "verified": True,
        "identity_sha256": receipt.identity_sha256,
        "entry_sha256": receipt.entry_sha256,
        "observation_receipt_sha256": receipt_document["receipt_sha256"],
        "log_id_sha256": hashlib.sha256(checkpoint.log_id.encode("ascii")).hexdigest(),
        "checkpoint_id_sha256": hashlib.sha256(
            checkpoint.checkpoint_id.encode("ascii")
        ).hexdigest(),
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "tree_size": checkpoint.tree_size,
        "leaf_index": proof.leaf_index,
        "root_sha256": checkpoint.root_sha256,
        "issued_at_epoch": checkpoint.issued_at_epoch,
        "public_key_sha256": checkpoint.public_key_sha256,
        "checkpoint_authentication_sha256": authentication["authentication_sha256"],
    }
    verification["verification_sha256"] = _digest(verification)
    return verification


def dump_snapshot_transparency_checkpoint(
    checkpoint: SnapshotTransparencyCheckpoint,
) -> bytes:
    _validate_checkpoint(checkpoint)
    payload = _canonical_bytes(checkpoint.to_audit_dict())
    if len(payload) > _MAX_CHECKPOINT_BYTES:
        raise SnapshotTransparencyError("checkpoint exceeds the byte limit")
    return payload


def dump_snapshot_transparency_proof(proof: SnapshotTransparencyProof) -> bytes:
    """Serialize one validated inclusion proof as canonical JSON."""

    _validate_proof(proof)
    payload = _canonical_bytes(asdict(proof))
    if len(payload) > _MAX_PROOF_BYTES:
        raise SnapshotTransparencyError("inclusion proof exceeds the byte limit")
    return payload


def load_snapshot_transparency_checkpoint(payload: bytes) -> SnapshotTransparencyCheckpoint:
    """Load bounded, duplicate-free canonical checkpoint JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_CHECKPOINT_BYTES:
        raise SnapshotTransparencyError("checkpoint is empty or exceeds the byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotTransparencyError("checkpoint contains duplicate keys")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotTransparencyError(f"checkpoint contains unsupported {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotTransparencyError("checkpoint JSON is malformed") from exc
    if not isinstance(document, dict) or set(document) != _CHECKPOINT_FIELDS:
        raise SnapshotTransparencyError("checkpoint fields do not match the schema")
    try:
        checkpoint = SnapshotTransparencyCheckpoint(**document)
    except TypeError as exc:
        raise SnapshotTransparencyError("checkpoint fields do not match the schema") from exc
    _validate_checkpoint(checkpoint)
    if payload != _canonical_bytes(checkpoint.to_audit_dict()):
        raise SnapshotTransparencyError("checkpoint JSON is not canonical")
    return checkpoint


def load_snapshot_transparency_proof(payload: bytes) -> SnapshotTransparencyProof:
    """Load bounded, duplicate-free canonical inclusion-proof JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_PROOF_BYTES:
        raise SnapshotTransparencyError("inclusion proof is empty or exceeds the byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotTransparencyError("inclusion proof contains duplicate keys")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotTransparencyError(f"inclusion proof contains unsupported {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotTransparencyError("inclusion proof JSON is malformed") from exc
    if (
        not isinstance(document, dict)
        or set(document) != _PROOF_FIELDS
        or not isinstance(document.get("audit_path"), list)
    ):
        raise SnapshotTransparencyError("inclusion proof fields do not match the schema")
    document["audit_path"] = tuple(document["audit_path"])
    try:
        proof = SnapshotTransparencyProof(**document)
    except TypeError as exc:
        raise SnapshotTransparencyError("inclusion proof fields do not match the schema") from exc
    _validate_proof(proof)
    if payload != _canonical_bytes(asdict(proof)):
        raise SnapshotTransparencyError("inclusion proof JSON is not canonical")
    return proof
