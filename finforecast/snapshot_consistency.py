"""Verify append-only growth between authenticated snapshot transparency checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from finforecast.snapshot_observation import SnapshotObservationReceipt
from finforecast.snapshot_transparency import (
    SnapshotTransparencyCheckpoint,
    SnapshotTransparencyError,
    SnapshotTransparencyProof,
    authenticate_snapshot_transparency_checkpoint,
    verify_snapshot_transparency_inclusion,
)

SNAPSHOT_CONSISTENCY_PROOF_SCHEMA = "finforecast-snapshot-consistency-proof/v1"
SNAPSHOT_CONSISTENCY_VERIFICATION_SCHEMA = "finforecast-snapshot-consistency-verification/v1"
SNAPSHOT_APPEND_ONLY_INCLUSION_SCHEMA = "finforecast-snapshot-append-only-inclusion/v1"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MAX_TREE_SIZE = 1_000_000
_MAX_AUDIT_PATH = 64
_MAX_PROOF_BYTES = 8_192
_PROOF_FIELDS = {"schema_version", "old_tree_size", "new_tree_size", "audit_path"}


@dataclass(frozen=True, slots=True)
class SnapshotTransparencyConsistencyProof:
    schema_version: str
    old_tree_size: int
    new_tree_size: int
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
        raise SnapshotTransparencyError("consistency evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


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


def _subproof(leaves: Sequence[bytes], old_size: int, complete_subtree: bool) -> list[bytes]:
    if old_size == len(leaves):
        return [] if complete_subtree else [_tree_root(leaves)]
    split = _largest_power_of_two_less_than(len(leaves))
    if old_size <= split:
        return [
            *_subproof(leaves[:split], old_size, complete_subtree),
            _tree_root(leaves[split:]),
        ]
    return [
        *_subproof(leaves[split:], old_size - split, False),
        _tree_root(leaves[:split]),
    ]


def _normalize_entries(entry_sha256s: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(entry_sha256s, (str, bytes))
        or not isinstance(entry_sha256s, Sequence)
        or not 2 <= len(entry_sha256s) <= _MAX_TREE_SIZE
    ):
        raise SnapshotTransparencyError("consistency entries are invalid")
    entries = tuple(_require_digest(value, "entry_sha256") for value in entry_sha256s)
    if len(set(entries)) != len(entries):
        raise SnapshotTransparencyError("consistency entries contain duplicates")
    return entries


def _validate_proof(proof: SnapshotTransparencyConsistencyProof) -> None:
    if not isinstance(proof, SnapshotTransparencyConsistencyProof):
        raise TypeError("proof must be a SnapshotTransparencyConsistencyProof")
    if (
        set(asdict(proof)) != _PROOF_FIELDS
        or proof.schema_version != SNAPSHOT_CONSISTENCY_PROOF_SCHEMA
    ):
        raise SnapshotTransparencyError("consistency proof fields do not match the schema")
    if (
        type(proof.old_tree_size) is not int
        or type(proof.new_tree_size) is not int
        or not 1 <= proof.old_tree_size < proof.new_tree_size <= _MAX_TREE_SIZE
        or not isinstance(proof.audit_path, tuple)
        or not 1 <= len(proof.audit_path) <= _MAX_AUDIT_PATH
    ):
        raise SnapshotTransparencyError("consistency proof metadata is invalid")
    for node in proof.audit_path:
        _require_digest(node, "audit_path node")


def create_snapshot_transparency_consistency_proof(
    entry_sha256s: Sequence[str],
    *,
    old_tree_size: int,
) -> SnapshotTransparencyConsistencyProof:
    """Build an RFC 6962-style proof that one tree is a prefix of a larger tree."""

    entries = _normalize_entries(entry_sha256s)
    if (
        isinstance(old_tree_size, bool)
        or not isinstance(old_tree_size, int)
        or not 1 <= old_tree_size < len(entries)
    ):
        raise SnapshotTransparencyError("old tree size is invalid")
    leaves = tuple(_leaf_hash(value) for value in entries)
    proof = SnapshotTransparencyConsistencyProof(
        schema_version=SNAPSHOT_CONSISTENCY_PROOF_SCHEMA,
        old_tree_size=old_tree_size,
        new_tree_size=len(entries),
        audit_path=tuple(node.hex() for node in _subproof(leaves, old_tree_size, True)),
    )
    _validate_proof(proof)
    return proof


def _verify_consistency_roots(
    proof: SnapshotTransparencyConsistencyProof,
    old_root_sha256: str,
    new_root_sha256: str,
) -> None:
    _validate_proof(proof)
    old_root = bytes.fromhex(_require_digest(old_root_sha256, "old root"))
    new_root = bytes.fromhex(_require_digest(new_root_sha256, "new root"))
    path = tuple(bytes.fromhex(value) for value in proof.audit_path)
    old_position = proof.old_tree_size - 1
    new_position = proof.new_tree_size - 1
    while old_position & 1:
        old_position >>= 1
        new_position >>= 1

    if old_position == 0:
        old_hash = old_root
        new_hash = old_root
        remaining = path
    else:
        old_hash = path[0]
        new_hash = path[0]
        remaining = path[1:]

    for sibling in remaining:
        if new_position == 0:
            raise SnapshotTransparencyError("consistency proof has extra nodes")
        if old_position & 1 or old_position == new_position:
            old_hash = _node_hash(sibling, old_hash)
            new_hash = _node_hash(sibling, new_hash)
            while old_position != 0 and old_position & 1 == 0:
                old_position >>= 1
                new_position >>= 1
        else:
            new_hash = _node_hash(new_hash, sibling)
        old_position >>= 1
        new_position >>= 1

    if (
        new_position != 0
        or not hmac.compare_digest(old_hash, old_root)
        or not hmac.compare_digest(new_hash, new_root)
    ):
        raise SnapshotTransparencyError("consistency proof does not link the checkpoints")


def verify_snapshot_transparency_consistency(
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    proof: SnapshotTransparencyConsistencyProof,
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
) -> dict[str, object]:
    """Authenticate two checkpoints and prove append-only growth between them."""

    old_auth = authenticate_snapshot_transparency_checkpoint(
        old_checkpoint,
        trusted_public_keys=trusted_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_old_checkpoint_id,
    )
    new_auth = authenticate_snapshot_transparency_checkpoint(
        new_checkpoint,
        trusted_public_keys=trusted_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_new_checkpoint_id,
    )
    _validate_proof(proof)
    if (
        old_checkpoint.tree_size != proof.old_tree_size
        or new_checkpoint.tree_size != proof.new_tree_size
        or new_checkpoint.issued_at_epoch < old_checkpoint.issued_at_epoch
    ):
        raise SnapshotTransparencyError("checkpoints do not match the consistency proof")
    _verify_consistency_roots(proof, old_checkpoint.root_sha256, new_checkpoint.root_sha256)

    verification: dict[str, object] = {
        "schema_version": SNAPSHOT_CONSISTENCY_VERIFICATION_SCHEMA,
        "verified": True,
        "log_id_sha256": old_auth["log_id_sha256"],
        "old_checkpoint_sha256": old_auth["checkpoint_sha256"],
        "new_checkpoint_sha256": new_auth["checkpoint_sha256"],
        "old_checkpoint_authentication_sha256": old_auth["authentication_sha256"],
        "new_checkpoint_authentication_sha256": new_auth["authentication_sha256"],
        "old_tree_size": proof.old_tree_size,
        "new_tree_size": proof.new_tree_size,
        "old_root_sha256": old_checkpoint.root_sha256,
        "new_root_sha256": new_checkpoint.root_sha256,
        "proof_sha256": hashlib.sha256(
            dump_snapshot_transparency_consistency_proof(proof)
        ).hexdigest(),
    }
    verification["verification_sha256"] = _digest(verification)
    return verification


def verify_snapshot_transparency_append_only_inclusion(
    receipt: SnapshotObservationReceipt,
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    consistency_proof: SnapshotTransparencyConsistencyProof,
    inclusion_proof: SnapshotTransparencyProof,
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
) -> dict[str, object]:
    """Require both log-prefix consistency and inclusion in the authenticated new checkpoint."""

    consistency = verify_snapshot_transparency_consistency(
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        trusted_public_keys=trusted_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    inclusion = verify_snapshot_transparency_inclusion(
        receipt,
        new_checkpoint,
        inclusion_proof,
        trusted_public_keys=trusted_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_new_checkpoint_id,
    )
    combined: dict[str, object] = {
        "schema_version": SNAPSHOT_APPEND_ONLY_INCLUSION_SCHEMA,
        "verified": True,
        "identity_sha256": inclusion["identity_sha256"],
        "entry_sha256": inclusion["entry_sha256"],
        "observation_receipt_sha256": inclusion["observation_receipt_sha256"],
        "old_checkpoint_sha256": consistency["old_checkpoint_sha256"],
        "new_checkpoint_sha256": consistency["new_checkpoint_sha256"],
        "old_tree_size": consistency["old_tree_size"],
        "new_tree_size": consistency["new_tree_size"],
        "inclusion_verification_sha256": inclusion["verification_sha256"],
        "consistency_verification_sha256": consistency["verification_sha256"],
    }
    combined["verification_sha256"] = _digest(combined)
    return combined


def dump_snapshot_transparency_consistency_proof(
    proof: SnapshotTransparencyConsistencyProof,
) -> bytes:
    """Serialize one strict consistency proof as canonical JSON."""

    _validate_proof(proof)
    payload = _canonical_bytes(asdict(proof))
    if len(payload) > _MAX_PROOF_BYTES:
        raise SnapshotTransparencyError("consistency proof exceeds the byte limit")
    return payload


def load_snapshot_transparency_consistency_proof(
    payload: bytes,
) -> SnapshotTransparencyConsistencyProof:
    """Load bounded, duplicate-free canonical consistency-proof JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_PROOF_BYTES:
        raise SnapshotTransparencyError("consistency proof is empty or exceeds the byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotTransparencyError("consistency proof contains duplicate keys")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotTransparencyError(f"consistency proof contains unsupported {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotTransparencyError("consistency proof JSON is malformed") from exc
    if (
        not isinstance(document, dict)
        or set(document) != _PROOF_FIELDS
        or not isinstance(document.get("audit_path"), list)
    ):
        raise SnapshotTransparencyError("consistency proof fields do not match the schema")
    document["audit_path"] = tuple(document["audit_path"])
    try:
        proof = SnapshotTransparencyConsistencyProof(**document)
    except TypeError as exc:
        raise SnapshotTransparencyError("consistency proof fields do not match the schema") from exc
    _validate_proof(proof)
    if payload != _canonical_bytes(asdict(proof)):
        raise SnapshotTransparencyError("consistency proof JSON is not canonical")
    return proof
