"""Require independent witness quorum over a verified transparency-log transition."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.snapshot_consistency import (
    SnapshotTransparencyConsistencyProof,
    verify_snapshot_transparency_append_only_inclusion,
    verify_snapshot_transparency_consistency,
)
from finforecast.snapshot_observation import SnapshotObservationReceipt
from finforecast.snapshot_transparency import (
    SnapshotTransparencyCheckpoint,
    SnapshotTransparencyError,
    SnapshotTransparencyProof,
    SnapshotTransparencySigner,
)

SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA = "finforecast-snapshot-consistency-witness/v1"
SNAPSHOT_CONSISTENCY_WITNESS_PURPOSE = "snapshot-consistency-independent-witness"
SNAPSHOT_CONSISTENCY_WITNESS_ALGORITHM = "ed25519-sha512"
SNAPSHOT_CONSISTENCY_WITNESS_VERIFICATION_SCHEMA = (
    "finforecast-snapshot-consistency-witness-verification/v1"
)
SNAPSHOT_WITNESSED_PROMOTION_SCHEMA = "finforecast-snapshot-witnessed-promotion/v1"

_DOMAIN = b"finforecast/snapshot-consistency-witness/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_PUBLIC_KEY_BYTES = 8_192
_MAX_WITNESSES = 16
_MAX_BUNDLE_BYTES = 16_384
_APPROVAL_FIELDS = {"witness_id", "key_id", "public_key_sha256", "signature"}
_BUNDLE_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "consistency_verification_sha256",
    "witness_set_sha256",
    "required_witnesses",
    "approvals",
    "bundle_sha256",
}


@dataclass(frozen=True, slots=True)
class SnapshotConsistencyWitnessApproval:
    witness_id: str
    key_id: str
    public_key_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SnapshotConsistencyWitnessBundle:
    schema_version: str
    purpose: str
    algorithm: str
    consistency_verification_sha256: str
    witness_set_sha256: str
    required_witnesses: int
    approvals: tuple[SnapshotConsistencyWitnessApproval, ...]
    bundle_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        document = asdict(self)
        document["approvals"] = [approval.to_audit_dict() for approval in self.approvals]
        return document


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
        raise SnapshotTransparencyError("witness evidence is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


def _load_public_key(encoded: object) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= _MAX_PUBLIC_KEY_BYTES:
        raise SnapshotTransparencyError("witness public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise SnapshotTransparencyError("witness public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SnapshotTransparencyError("witness public key is not Ed25519")
    return key


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise SnapshotTransparencyError("witness signer returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or _SIGNATURE_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError("witness signature encoding is malformed")
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotTransparencyError("witness signature encoding is malformed") from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise SnapshotTransparencyError("witness signature encoding is malformed")
    return decoded


def _approval_metadata(approval: SnapshotConsistencyWitnessApproval) -> dict[str, str]:
    if not isinstance(approval, SnapshotConsistencyWitnessApproval):
        raise SnapshotTransparencyError("witness approval is malformed")
    document = approval.to_audit_dict()
    if set(document) != _APPROVAL_FIELDS:
        raise SnapshotTransparencyError("witness approval fields do not match the schema")
    _decode_signature(approval.signature)
    return {
        "witness_id": _identifier(approval.witness_id, "witness_id"),
        "key_id": _identifier(approval.key_id, "key_id"),
        "public_key_sha256": _require_digest(
            approval.public_key_sha256,
            "public_key_sha256",
        ),
    }


def _claims(
    metadata: Mapping[str, str],
    *,
    consistency_verification_sha256: str,
    witness_set_sha256: str,
    required_witnesses: int,
) -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA,
        "purpose": SNAPSHOT_CONSISTENCY_WITNESS_PURPOSE,
        "algorithm": SNAPSHOT_CONSISTENCY_WITNESS_ALGORITHM,
        "consistency_verification_sha256": consistency_verification_sha256,
        "witness_set_sha256": witness_set_sha256,
        "required_witnesses": required_witnesses,
        **dict(metadata),
    }


def _message(claims: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(claims))


def _bundle_without_digest(bundle: SnapshotConsistencyWitnessBundle) -> dict[str, object]:
    document = bundle.to_audit_dict()
    document.pop("bundle_sha256")
    return document


def _validate_bundle(bundle: SnapshotConsistencyWitnessBundle) -> list[dict[str, str]]:
    if not isinstance(bundle, SnapshotConsistencyWitnessBundle):
        raise TypeError("bundle must be a SnapshotConsistencyWitnessBundle")
    document = bundle.to_audit_dict()
    if set(document) != _BUNDLE_FIELDS:
        raise SnapshotTransparencyError("witness bundle fields do not match the schema")
    if (
        bundle.schema_version != SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA
        or bundle.purpose != SNAPSHOT_CONSISTENCY_WITNESS_PURPOSE
        or bundle.algorithm != SNAPSHOT_CONSISTENCY_WITNESS_ALGORITHM
    ):
        raise SnapshotTransparencyError("witness bundle policy is unsupported")
    _require_digest(bundle.consistency_verification_sha256, "consistency_verification_sha256")
    _require_digest(bundle.witness_set_sha256, "witness_set_sha256")
    _require_digest(bundle.bundle_sha256, "bundle_sha256")
    if (
        type(bundle.required_witnesses) is not int
        or not isinstance(bundle.approvals, tuple)
        or not 2 <= bundle.required_witnesses <= len(bundle.approvals) <= _MAX_WITNESSES
    ):
        raise SnapshotTransparencyError("witness quorum is invalid")
    metadata = [_approval_metadata(approval) for approval in bundle.approvals]
    if metadata != sorted(metadata, key=lambda item: item["witness_id"]):
        raise SnapshotTransparencyError("witness approvals are not canonically ordered")
    if len({item["witness_id"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness identifiers must be distinct")
    if len({item["key_id"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness key identifiers must be distinct")
    if len({item["public_key_sha256"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness key revisions must be distinct")
    if not hmac.compare_digest(bundle.witness_set_sha256, _digest(metadata)):
        raise SnapshotTransparencyError("witness set digest does not match its content")
    if not hmac.compare_digest(bundle.bundle_sha256, _digest(_bundle_without_digest(bundle))):
        raise SnapshotTransparencyError("witness bundle digest does not match its content")
    return metadata


def create_snapshot_consistency_witness_bundle(
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    consistency_proof: SnapshotTransparencyConsistencyProof,
    witnesses: Mapping[str, SnapshotTransparencySigner],
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
    required_witnesses: int = 2,
) -> SnapshotConsistencyWitnessBundle:
    """Verify one transition and collect a bounded quorum over its exact digest."""

    consistency = verify_snapshot_transparency_consistency(
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        trusted_public_keys=trusted_log_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    if not isinstance(witnesses, Mapping) or not 2 <= len(witnesses) <= _MAX_WITNESSES:
        raise SnapshotTransparencyError("witnesses must contain from 2 through 16 signers")
    if type(required_witnesses) is not int or not 2 <= required_witnesses <= len(witnesses):
        raise SnapshotTransparencyError("required_witnesses is invalid")

    signer_by_witness: dict[str, SnapshotTransparencySigner] = {}
    public_key_by_witness: dict[str, Ed25519PublicKey] = {}
    metadata: list[dict[str, str]] = []
    for raw_witness_id, signer in witnesses.items():
        witness_id = _identifier(raw_witness_id, "witness_id")
        try:
            key_id = _identifier(signer.key_id, "key_id")
            public_key_der = signer.public_key_der
        except Exception as exc:
            raise SnapshotTransparencyError("witness signer metadata is unavailable") from exc
        public_key = _load_public_key(public_key_der)
        signer_by_witness[witness_id] = signer
        public_key_by_witness[witness_id] = public_key
        metadata.append(
            {
                "witness_id": witness_id,
                "key_id": key_id,
                "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
            }
        )
    metadata.sort(key=lambda item: item["witness_id"])
    if len({item["witness_id"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness identifiers must be distinct")
    if len({item["key_id"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness key identifiers must be distinct")
    if len({item["public_key_sha256"] for item in metadata}) != len(metadata):
        raise SnapshotTransparencyError("witness key revisions must be distinct")

    consistency_sha256 = str(consistency["verification_sha256"])
    witness_set_sha256 = _digest(metadata)
    approvals: list[SnapshotConsistencyWitnessApproval] = []
    for item in metadata:
        claims = _claims(
            item,
            consistency_verification_sha256=consistency_sha256,
            witness_set_sha256=witness_set_sha256,
            required_witnesses=required_witnesses,
        )
        signer = signer_by_witness[item["witness_id"]]
        try:
            signature = signer.sign(_message(claims))
        except Exception as exc:
            raise SnapshotTransparencyError("witness signing failed") from exc
        encoded = _encode_signature(signature)
        try:
            public_key_by_witness[item["witness_id"]].verify(signature, _message(claims))
        except InvalidSignature as exc:
            raise SnapshotTransparencyError(
                "witness signer output does not match its retained public key"
            ) from exc
        approvals.append(SnapshotConsistencyWitnessApproval(**item, signature=encoded))

    unsigned: dict[str, object] = {
        "schema_version": SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA,
        "purpose": SNAPSHOT_CONSISTENCY_WITNESS_PURPOSE,
        "algorithm": SNAPSHOT_CONSISTENCY_WITNESS_ALGORITHM,
        "consistency_verification_sha256": consistency_sha256,
        "witness_set_sha256": witness_set_sha256,
        "required_witnesses": required_witnesses,
        "approvals": tuple(approvals),
    }
    bundle = SnapshotConsistencyWitnessBundle(**unsigned, bundle_sha256="0" * 64)
    bundle = SnapshotConsistencyWitnessBundle(
        **unsigned,
        bundle_sha256=_digest(_bundle_without_digest(bundle)),
    )
    _validate_bundle(bundle)
    return bundle


def verify_snapshot_consistency_witness_bundle(
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    consistency_proof: SnapshotTransparencyConsistencyProof,
    bundle: SnapshotConsistencyWitnessBundle,
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_witness_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
) -> dict[str, object]:
    """Require a trusted witness quorum over the locally reverified transition."""

    consistency = verify_snapshot_transparency_consistency(
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        trusted_public_keys=trusted_log_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    metadata = _validate_bundle(bundle)
    if not hmac.compare_digest(
        bundle.consistency_verification_sha256,
        str(consistency["verification_sha256"]),
    ):
        raise SnapshotTransparencyError("witness bundle names a different consistency transition")
    if not isinstance(trusted_witness_public_keys, Mapping):
        raise SnapshotTransparencyError("trusted witness keys must be a mapping")

    valid_revisions: list[str] = []
    for approval, item in zip(bundle.approvals, metadata, strict=True):
        public_key_der = trusted_witness_public_keys.get(item["key_id"])
        if public_key_der is None:
            continue
        try:
            public_key = _load_public_key(public_key_der)
        except SnapshotTransparencyError:
            continue
        revision = hashlib.sha256(public_key_der).hexdigest()
        if not hmac.compare_digest(revision, item["public_key_sha256"]):
            continue
        claims = _claims(
            item,
            consistency_verification_sha256=bundle.consistency_verification_sha256,
            witness_set_sha256=bundle.witness_set_sha256,
            required_witnesses=bundle.required_witnesses,
        )
        try:
            public_key.verify(_decode_signature(approval.signature), _message(claims))
        except (InvalidSignature, SnapshotTransparencyError):
            continue
        valid_revisions.append(revision)
    if len(valid_revisions) < bundle.required_witnesses:
        raise SnapshotTransparencyError("witness quorum is not satisfied")

    verification: dict[str, object] = {
        "schema_version": SNAPSHOT_CONSISTENCY_WITNESS_VERIFICATION_SCHEMA,
        "verified": True,
        "consistency_verification_sha256": consistency["verification_sha256"],
        "witness_bundle_sha256": bundle.bundle_sha256,
        "witness_set_sha256": bundle.witness_set_sha256,
        "required_witnesses": bundle.required_witnesses,
        "valid_witnesses": len(valid_revisions),
        "valid_witness_revisions_sha256": _digest(sorted(valid_revisions)),
    }
    verification["verification_sha256"] = _digest(verification)
    return verification


def verify_witnessed_snapshot_transparency_promotion(
    receipt: SnapshotObservationReceipt,
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    consistency_proof: SnapshotTransparencyConsistencyProof,
    inclusion_proof: SnapshotTransparencyProof,
    witness_bundle: SnapshotConsistencyWitnessBundle,
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_witness_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
) -> dict[str, object]:
    """Cross-bind local inclusion and consistency to an independent witness quorum."""

    promotion = verify_snapshot_transparency_append_only_inclusion(
        receipt,
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        inclusion_proof,
        trusted_public_keys=trusted_log_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    witness = verify_snapshot_consistency_witness_bundle(
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        witness_bundle,
        trusted_log_public_keys=trusted_log_public_keys,
        trusted_witness_public_keys=trusted_witness_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    if not hmac.compare_digest(
        str(promotion["consistency_verification_sha256"]),
        str(witness["consistency_verification_sha256"]),
    ):
        raise SnapshotTransparencyError("witness and promotion evidence do not cross-bind")
    combined: dict[str, object] = {
        "schema_version": SNAPSHOT_WITNESSED_PROMOTION_SCHEMA,
        "verified": True,
        "identity_sha256": promotion["identity_sha256"],
        "entry_sha256": promotion["entry_sha256"],
        "old_checkpoint_sha256": promotion["old_checkpoint_sha256"],
        "new_checkpoint_sha256": promotion["new_checkpoint_sha256"],
        "old_tree_size": promotion["old_tree_size"],
        "new_tree_size": promotion["new_tree_size"],
        "append_only_inclusion_verification_sha256": promotion["verification_sha256"],
        "witness_bundle_sha256": witness["witness_bundle_sha256"],
        "witness_verification_sha256": witness["verification_sha256"],
        "required_witnesses": witness["required_witnesses"],
        "valid_witnesses": witness["valid_witnesses"],
    }
    combined["verification_sha256"] = _digest(combined)
    return combined


def dump_snapshot_consistency_witness_bundle(
    bundle: SnapshotConsistencyWitnessBundle,
) -> bytes:
    """Serialize one strict witness bundle as bounded canonical JSON."""

    _validate_bundle(bundle)
    payload = _canonical_bytes(bundle.to_audit_dict())
    if len(payload) > _MAX_BUNDLE_BYTES:
        raise SnapshotTransparencyError("witness bundle exceeds the byte limit")
    return payload


def load_snapshot_consistency_witness_bundle(payload: bytes) -> SnapshotConsistencyWitnessBundle:
    """Load bounded, duplicate-free canonical witness-bundle JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_BUNDLE_BYTES:
        raise SnapshotTransparencyError("witness bundle is empty or exceeds the byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotTransparencyError("witness bundle contains duplicate keys")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotTransparencyError(f"witness bundle contains unsupported {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotTransparencyError("witness bundle JSON is malformed") from exc
    if (
        not isinstance(document, dict)
        or set(document) != _BUNDLE_FIELDS
        or not isinstance(document.get("approvals"), list)
    ):
        raise SnapshotTransparencyError("witness bundle fields do not match the schema")
    approvals: list[SnapshotConsistencyWitnessApproval] = []
    for approval in document["approvals"]:
        if not isinstance(approval, dict) or set(approval) != _APPROVAL_FIELDS:
            raise SnapshotTransparencyError("witness approval fields do not match the schema")
        try:
            approvals.append(SnapshotConsistencyWitnessApproval(**approval))
        except TypeError as exc:
            raise SnapshotTransparencyError(
                "witness approval fields do not match the schema"
            ) from exc
    document["approvals"] = tuple(approvals)
    try:
        bundle = SnapshotConsistencyWitnessBundle(**document)
    except TypeError as exc:
        raise SnapshotTransparencyError("witness bundle fields do not match the schema") from exc
    _validate_bundle(bundle)
    if payload != _canonical_bytes(bundle.to_audit_dict()):
        raise SnapshotTransparencyError("witness bundle JSON is not canonical")
    return bundle
