"""Authenticate point-in-time return revisions before volatility labeling."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.regime_labeling import (
    PointInTimeReturn,
    RegimeLabelingError,
    VolatilityRegimeAssignment,
    VolatilityRegimeDefinition,
    assign_point_in_time_volatility_regimes,
    commit_point_in_time_returns,
)

RETURN_REVISION_SNAPSHOT_SIGNATURE_SCHEMA = "finforecast-return-revision-snapshot-signature/v1"
RETURN_REVISION_SNAPSHOT_VERIFICATION_SCHEMA = (
    "finforecast-return-revision-snapshot-verification/v1"
)
AUTHENTICATED_REGIME_ASSIGNMENT_SCHEMA = "finforecast-authenticated-regime-assignments/v1"
RETURN_REVISION_SNAPSHOT_PURPOSE = "point-in-time-return-revision-snapshot"
RETURN_REVISION_SNAPSHOT_ALGORITHM = "ed25519-sha512"

_DOMAIN = b"finforecast/return-revision-snapshot/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_PUBLIC_KEY_BYTES = 8_192
_MAX_SIGNATURE_DOCUMENT_BYTES = 4_096
_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "key_id",
    "public_key_sha256",
    "vendor_id",
    "dataset_id",
    "snapshot_id",
    "issued_at_epoch",
    "source_schema_sha256",
    "record_count",
    "records_sha256",
    "signature",
}


class ReturnRevisionSnapshotError(ValueError):
    """The return-revision snapshot is malformed, mismatched, or unauthenticated."""

    code = "RETURN_REVISION_SNAPSHOT_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class ReturnRevisionSnapshotSigner(Protocol):
    """External Ed25519 signer with a retained immutable public key."""

    key_id: str
    public_key_der: bytes

    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReturnRevisionSnapshotSignature:
    schema_version: str
    purpose: str
    algorithm: str
    key_id: str
    public_key_sha256: str
    vendor_id: str
    dataset_id: str
    snapshot_id: str
    issued_at_epoch: int
    source_schema_sha256: str
    record_count: int
    records_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def signature_document_sha256(self) -> str:
        return hashlib.sha256(dump_return_revision_snapshot_signature(self)).hexdigest()


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
        raise ReturnRevisionSnapshotError("document is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ReturnRevisionSnapshotError(f"{name} is invalid")
    return value


def _load_public_key(encoded: object) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= _MAX_PUBLIC_KEY_BYTES:
        raise ReturnRevisionSnapshotError("public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ReturnRevisionSnapshotError("public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ReturnRevisionSnapshotError("public key algorithm is not Ed25519")
    return key


def _validate_document(document: Mapping[str, object]) -> None:
    if set(document) != _FIELDS:
        raise ReturnRevisionSnapshotError("signature fields do not match the schema")
    if (
        document["schema_version"] != RETURN_REVISION_SNAPSHOT_SIGNATURE_SCHEMA
        or document["purpose"] != RETURN_REVISION_SNAPSHOT_PURPOSE
        or document["algorithm"] != RETURN_REVISION_SNAPSHOT_ALGORITHM
    ):
        raise ReturnRevisionSnapshotError("signature policy is unsupported")
    for field in ("key_id", "vendor_id", "dataset_id", "snapshot_id"):
        _identifier(document[field], field)
    if (
        type(document["issued_at_epoch"]) is not int
        or not 1 <= document["issued_at_epoch"] <= 2**63 - 1
        or type(document["record_count"]) is not int
        or not 2 <= document["record_count"] <= 1_000_000
        or any(
            not isinstance(document[field], str) or _DIGEST_RE.fullmatch(document[field]) is None
            for field in ("public_key_sha256", "source_schema_sha256", "records_sha256")
        )
        or not isinstance(document["signature"], str)
        or _SIGNATURE_RE.fullmatch(document["signature"]) is None
    ):
        raise ReturnRevisionSnapshotError("signature fields are malformed")


def _unsigned(signature: ReturnRevisionSnapshotSignature) -> dict[str, object]:
    document = signature.to_audit_dict()
    document.pop("signature")
    return document


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(document))


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise ReturnRevisionSnapshotError("signer returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise ReturnRevisionSnapshotError("signature encoding is malformed") from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise ReturnRevisionSnapshotError("signature encoding is malformed")
    return decoded


def create_return_revision_snapshot_signature(
    rows: Sequence[PointInTimeReturn],
    *,
    vendor_id: str,
    dataset_id: str,
    snapshot_id: str,
    issued_at_epoch: int,
    source_schema_sha256: str,
    signer: ReturnRevisionSnapshotSigner,
) -> ReturnRevisionSnapshotSignature:
    """Sign an exact, ordered revision-stream commitment with an external key."""

    try:
        commitment = commit_point_in_time_returns(rows)
    except RegimeLabelingError as exc:
        raise ReturnRevisionSnapshotError("revision stream is invalid") from exc
    try:
        key_id = signer.key_id
        public_key_der = signer.public_key_der
    except Exception as exc:
        raise ReturnRevisionSnapshotError("signer metadata is unavailable") from exc
    public_key = _load_public_key(public_key_der)
    unsigned: dict[str, object] = {
        "schema_version": RETURN_REVISION_SNAPSHOT_SIGNATURE_SCHEMA,
        "purpose": RETURN_REVISION_SNAPSHOT_PURPOSE,
        "algorithm": RETURN_REVISION_SNAPSHOT_ALGORITHM,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
        "vendor_id": vendor_id,
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "issued_at_epoch": issued_at_epoch,
        "source_schema_sha256": source_schema_sha256,
        "record_count": commitment.record_count,
        "records_sha256": commitment.records_sha256,
    }
    _validate_document({**unsigned, "signature": "A" * 86})
    try:
        raw_signature = signer.sign(_message(unsigned))
    except Exception as exc:
        raise ReturnRevisionSnapshotError("return snapshot signing failed") from exc
    encoded_signature = _encode_signature(raw_signature)
    try:
        public_key.verify(raw_signature, _message(unsigned))
    except InvalidSignature as exc:
        raise ReturnRevisionSnapshotError(
            "signer output does not match the retained public key"
        ) from exc
    return ReturnRevisionSnapshotSignature(
        **unsigned,
        signature=encoded_signature,
    )


def verify_return_revision_snapshot_signature(
    signature: ReturnRevisionSnapshotSignature,
    rows: Sequence[PointInTimeReturn],
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_vendor_id: str,
    expected_dataset_id: str,
    expected_snapshot_id: str,
    expected_source_schema_sha256: str,
) -> dict[str, object]:
    """Verify producer identity, stream bytes, and governed source identity."""

    if not isinstance(signature, ReturnRevisionSnapshotSignature):
        raise TypeError("signature must be a ReturnRevisionSnapshotSignature")
    document = signature.to_audit_dict()
    _validate_document(document)
    expected = {
        "vendor_id": _identifier(expected_vendor_id, "expected_vendor_id"),
        "dataset_id": _identifier(expected_dataset_id, "expected_dataset_id"),
        "snapshot_id": _identifier(expected_snapshot_id, "expected_snapshot_id"),
    }
    if any(
        not hmac.compare_digest(str(getattr(signature, field)), value)
        for field, value in expected.items()
    ):
        raise ReturnRevisionSnapshotError("signature source identity does not match governance")
    if (
        not isinstance(expected_source_schema_sha256, str)
        or _DIGEST_RE.fullmatch(expected_source_schema_sha256) is None
        or not hmac.compare_digest(signature.source_schema_sha256, expected_source_schema_sha256)
    ):
        raise ReturnRevisionSnapshotError("source schema does not match governance")
    if not isinstance(trusted_public_keys, Mapping) or signature.key_id not in trusted_public_keys:
        raise ReturnRevisionSnapshotError("trusted signer key is unavailable")
    public_key_der = trusted_public_keys[signature.key_id]
    public_key = _load_public_key(public_key_der)
    if not hmac.compare_digest(
        hashlib.sha256(public_key_der).hexdigest(),
        signature.public_key_sha256,
    ):
        raise ReturnRevisionSnapshotError("trusted signer key revision does not match")
    try:
        public_key.verify(_decode_signature(signature.signature), _message(_unsigned(signature)))
    except InvalidSignature as exc:
        raise ReturnRevisionSnapshotError("return snapshot signature does not verify") from exc
    try:
        commitment = commit_point_in_time_returns(rows)
    except RegimeLabelingError as exc:
        raise ReturnRevisionSnapshotError("revision stream is invalid") from exc
    if signature.record_count != commitment.record_count or not hmac.compare_digest(
        signature.records_sha256, commitment.records_sha256
    ):
        raise ReturnRevisionSnapshotError("signature does not bind the supplied revision stream")
    return {
        "schema_version": RETURN_REVISION_SNAPSHOT_VERIFICATION_SCHEMA,
        "verified": True,
        "key_id": signature.key_id,
        "public_key_sha256": signature.public_key_sha256,
        "vendor_id": signature.vendor_id,
        "dataset_id": signature.dataset_id,
        "snapshot_id": signature.snapshot_id,
        "issued_at_epoch": signature.issued_at_epoch,
        "source_schema_sha256": signature.source_schema_sha256,
        "record_count": signature.record_count,
        "records_sha256": signature.records_sha256,
        "signature_document_sha256": signature.signature_document_sha256,
    }


def assign_authenticated_point_in_time_volatility_regimes(
    rows: Sequence[PointInTimeReturn],
    issuances: Sequence[datetime],
    definition: VolatilityRegimeDefinition,
    signature: ReturnRevisionSnapshotSignature,
    *,
    expected_definition_sha256: str,
    trusted_public_keys: Mapping[str, bytes],
    expected_vendor_id: str,
    expected_dataset_id: str,
    expected_snapshot_id: str,
) -> tuple[tuple[VolatilityRegimeAssignment, ...], dict[str, object]]:
    """Authenticate the complete source stream before deterministic regime assignment."""

    if not isinstance(definition, VolatilityRegimeDefinition):
        raise TypeError("definition must be a VolatilityRegimeDefinition")
    snapshot_verification = verify_return_revision_snapshot_signature(
        signature,
        rows,
        trusted_public_keys=trusted_public_keys,
        expected_vendor_id=expected_vendor_id,
        expected_dataset_id=expected_dataset_id,
        expected_snapshot_id=expected_snapshot_id,
        expected_source_schema_sha256=definition.source_schema_sha256,
    )
    assignments = assign_point_in_time_volatility_regimes(
        rows,
        issuances,
        definition,
        expected_definition_sha256=expected_definition_sha256,
    )
    evidence: dict[str, object] = {
        "schema_version": AUTHENTICATED_REGIME_ASSIGNMENT_SCHEMA,
        "accepted": True,
        "snapshot_verification": snapshot_verification,
        "definition_sha256": definition.sha256,
        "assignment_count": len(assignments),
        "assignment_evidence_sha256": _digest(
            [assignment.evidence_sha256 for assignment in assignments]
        ),
    }
    evidence["evidence_sha256"] = _digest(evidence)
    return assignments, evidence


def dump_return_revision_snapshot_signature(
    signature: ReturnRevisionSnapshotSignature,
) -> bytes:
    if not isinstance(signature, ReturnRevisionSnapshotSignature):
        raise TypeError("signature must be a ReturnRevisionSnapshotSignature")
    document = signature.to_audit_dict()
    _validate_document(document)
    payload = _canonical_bytes(document)
    if len(payload) > _MAX_SIGNATURE_DOCUMENT_BYTES:
        raise ReturnRevisionSnapshotError("signature document exceeds the byte limit")
    return payload


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReturnRevisionSnapshotError("signature contains a duplicate field")
        result[key] = value
    return result


def load_return_revision_snapshot_signature(payload: bytes) -> ReturnRevisionSnapshotSignature:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > _MAX_SIGNATURE_DOCUMENT_BYTES:
        raise ReturnRevisionSnapshotError("signature document size is invalid")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReturnRevisionSnapshotError("signature document is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ReturnRevisionSnapshotError("signature fields do not match the schema")
    _validate_document(document)
    if _canonical_bytes(document) != payload:
        raise ReturnRevisionSnapshotError("signature document is not canonical JSON")
    try:
        return ReturnRevisionSnapshotSignature(**document)
    except TypeError as exc:  # pragma: no cover - exact fields checked above
        raise ReturnRevisionSnapshotError("signature fields do not match the schema") from exc
