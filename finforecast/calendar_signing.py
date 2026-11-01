from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.calendar_migration import (
    CalendarMigrationError,
    dump_calendar_behavior_manifest,
    require_calendar_migration,
)

CALENDAR_BASELINE_SIGNATURE_SCHEMA = "finforecast-calendar-baseline-signature/v1"
CALENDAR_BASELINE_VERIFICATION_SCHEMA = "finforecast-calendar-baseline-verification/v1"
SIGNED_CALENDAR_MIGRATION_SCHEMA = "finforecast-signed-calendar-migration/v1"
CALENDAR_BASELINE_SIGNATURE_PURPOSE = "reviewed-calendar-behavior-baseline"
CALENDAR_BASELINE_SIGNATURE_ALGORITHM = "ed25519-sha512"

_DOMAIN = b"finforecast/calendar-baseline-signature/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_SIGNATURE_DOCUMENT_BYTES = 4_096
_MAX_KMS_MESSAGE_BYTES = 4_096
_MAX_PUBLIC_KEY_BYTES = 8_192
_SIGNATURE_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "key_id",
    "public_key_sha256",
    "source_commit_sha",
    "issued_at_epoch",
    "calendar_id",
    "revision_id",
    "tzdata_version",
    "manifest_sha256",
    "manifest_payload_sha256",
    "signature",
}


class CalendarBaselineSignatureError(ValueError):
    """A reviewed calendar baseline is malformed, mismatched, or unauthenticated."""

    code = "CALENDAR_BASELINE_SIGNATURE_REJECTED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class CalendarBaselineSigner(Protocol):
    """External signer whose immutable public key is retained with the release."""

    key_id: str
    public_key_der: bytes

    def sign(self, message: bytes) -> bytes: ...


class KmsEd25519Signer:
    """AWS KMS Ed25519 adapter that validates and pins the exported public key."""

    def __init__(self, client: Any, *, kms_key_ref: str, key_id: str) -> None:
        if not isinstance(kms_key_ref, str) or not 1 <= len(kms_key_ref) <= 2_048:
            raise CalendarBaselineSignatureError("KMS key reference is invalid")
        _validate_identifier(key_id, "key_id")
        try:
            response = client.get_public_key(KeyId=kms_key_ref)
        except Exception as exc:
            raise CalendarBaselineSignatureError("KMS public key lookup failed") from exc
        if not isinstance(response, Mapping):
            raise CalendarBaselineSignatureError("KMS public key response is malformed")
        provider_key_id = response.get("KeyId")
        public_key = response.get("PublicKey")
        algorithms = response.get("SigningAlgorithms")
        if (
            not isinstance(provider_key_id, str)
            or not 1 <= len(provider_key_id) <= 2_048
            or response.get("KeyUsage") != "SIGN_VERIFY"
            or not isinstance(algorithms, list)
            or any(not isinstance(item, str) for item in algorithms)
            or "ED25519_SHA_512" not in algorithms
            or not isinstance(public_key, bytes)
            or not 1 <= len(public_key) <= _MAX_PUBLIC_KEY_BYTES
        ):
            raise CalendarBaselineSignatureError("KMS public key response is malformed")
        _load_ed25519_public_key(public_key)
        self._client = client
        self._provider_key_id = provider_key_id
        self.key_id = key_id
        self.public_key_der = public_key

    def sign(self, message: bytes) -> bytes:
        if not isinstance(message, bytes) or not 1 <= len(message) <= _MAX_KMS_MESSAGE_BYTES:
            raise CalendarBaselineSignatureError("KMS signing message size is invalid")
        try:
            response = self._client.sign(
                KeyId=self._provider_key_id,
                Message=message,
                MessageType="RAW",
                SigningAlgorithm="ED25519_SHA_512",
            )
        except Exception as exc:
            raise CalendarBaselineSignatureError("KMS signing failed") from exc
        if not isinstance(response, Mapping):
            raise CalendarBaselineSignatureError("KMS signing response is malformed")
        signature = response.get("Signature")
        if (
            response.get("KeyId") != self._provider_key_id
            or response.get("SigningAlgorithm") != "ED25519_SHA_512"
            or not isinstance(signature, bytes)
            or len(signature) != 64
        ):
            raise CalendarBaselineSignatureError("KMS signing response is malformed")
        return signature


@dataclass(frozen=True, slots=True)
class CalendarBaselineSignature:
    schema_version: str
    purpose: str
    algorithm: str
    key_id: str
    public_key_sha256: str
    source_commit_sha: str
    issued_at_epoch: int
    calendar_id: str
    revision_id: str
    tzdata_version: str
    manifest_sha256: str
    manifest_payload_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def signature_document_sha256(self) -> str:
        return hashlib.sha256(dump_calendar_baseline_signature(self)).hexdigest()


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
        raise CalendarBaselineSignatureError("document is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CalendarBaselineSignatureError(f"{name} is invalid")
    return value


def _load_ed25519_public_key(encoded: bytes) -> Ed25519PublicKey:
    try:
        public_key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise CalendarBaselineSignatureError("public key is malformed") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise CalendarBaselineSignatureError("public key algorithm is not Ed25519")
    return public_key


def _validate_document(document: Mapping[str, object]) -> None:
    if set(document) != _SIGNATURE_FIELDS:
        raise CalendarBaselineSignatureError("signature fields do not match the schema")
    if (
        document["schema_version"] != CALENDAR_BASELINE_SIGNATURE_SCHEMA
        or document["purpose"] != CALENDAR_BASELINE_SIGNATURE_PURPOSE
        or document["algorithm"] != CALENDAR_BASELINE_SIGNATURE_ALGORITHM
    ):
        raise CalendarBaselineSignatureError("signature policy is unsupported")
    _validate_identifier(document["key_id"], "key_id")
    _validate_identifier(document["calendar_id"], "calendar_id")
    _validate_identifier(document["revision_id"], "revision_id")
    _validate_identifier(document["tzdata_version"], "tzdata_version")
    if (
        not isinstance(document["source_commit_sha"], str)
        or _GIT_SHA_RE.fullmatch(document["source_commit_sha"]) is None
        or type(document["issued_at_epoch"]) is not int
        or not 1 <= document["issued_at_epoch"] <= 2**63 - 1
        or any(
            not isinstance(document[field], str) or _DIGEST_RE.fullmatch(document[field]) is None
            for field in ("public_key_sha256", "manifest_sha256", "manifest_payload_sha256")
        )
        or not isinstance(document["signature"], str)
        or _SIGNATURE_RE.fullmatch(document["signature"]) is None
    ):
        raise CalendarBaselineSignatureError("signature fields are malformed")


def _unsigned_document(signature: CalendarBaselineSignature) -> dict[str, object]:
    return {key: value for key, value in signature.to_audit_dict().items() if key != "signature"}


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(document))


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise CalendarBaselineSignatureError("signer returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CalendarBaselineSignatureError("signature encoding is malformed") from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise CalendarBaselineSignatureError("signature encoding is malformed")
    return decoded


def _manifest_payload(manifest: Mapping[str, object]) -> tuple[dict[str, object], bytes]:
    try:
        normalized = dict(manifest)
        payload = dump_calendar_behavior_manifest(normalized)
    except (TypeError, CalendarMigrationError) as exc:
        raise CalendarBaselineSignatureError("calendar manifest is invalid") from exc
    return normalized, payload


def create_calendar_baseline_signature(
    manifest: Mapping[str, object],
    *,
    source_commit_sha: str,
    issued_at_epoch: int,
    signer: CalendarBaselineSigner,
) -> CalendarBaselineSignature:
    """Sign one captured, validated schema-v2 calendar behavior manifest."""

    normalized, payload = _manifest_payload(manifest)
    try:
        key_id = signer.key_id
        public_key_der = signer.public_key_der
    except Exception as exc:
        raise CalendarBaselineSignatureError("signer metadata is unavailable") from exc
    _validate_identifier(key_id, "key_id")
    if (
        not isinstance(public_key_der, bytes)
        or not 1 <= len(public_key_der) <= _MAX_PUBLIC_KEY_BYTES
    ):
        raise CalendarBaselineSignatureError("signer public key is malformed")
    _load_ed25519_public_key(public_key_der)
    unsigned: dict[str, object] = {
        "schema_version": CALENDAR_BASELINE_SIGNATURE_SCHEMA,
        "purpose": CALENDAR_BASELINE_SIGNATURE_PURPOSE,
        "algorithm": CALENDAR_BASELINE_SIGNATURE_ALGORITHM,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
        "source_commit_sha": source_commit_sha,
        "issued_at_epoch": issued_at_epoch,
        "calendar_id": normalized["calendar_id"],
        "revision_id": normalized["revision_id"],
        "tzdata_version": normalized["tzdata_version"],
        "manifest_sha256": normalized["manifest_sha256"],
        "manifest_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _validate_document({**unsigned, "signature": "A" * 86})
    try:
        raw_signature = signer.sign(_message(unsigned))
    except CalendarBaselineSignatureError:
        raise
    except Exception as exc:
        raise CalendarBaselineSignatureError("calendar baseline signing failed") from exc
    return CalendarBaselineSignature(**unsigned, signature=_encode_signature(raw_signature))


def verify_calendar_baseline_signature(
    signature: CalendarBaselineSignature,
    manifest: Mapping[str, object],
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_source_commit_sha: str,
) -> dict[str, object]:
    """Verify signer identity and exact baseline bytes before migration comparison."""

    if not isinstance(signature, CalendarBaselineSignature):
        raise TypeError("signature must be a CalendarBaselineSignature")
    document = signature.to_audit_dict()
    _validate_document(document)
    if not isinstance(expected_source_commit_sha, str) or (
        _GIT_SHA_RE.fullmatch(expected_source_commit_sha) is None
    ):
        raise CalendarBaselineSignatureError("expected source commit is malformed")
    if not hmac.compare_digest(signature.source_commit_sha, expected_source_commit_sha):
        raise CalendarBaselineSignatureError("source commit does not match the reviewed release")
    if not isinstance(trusted_public_keys, Mapping) or signature.key_id not in trusted_public_keys:
        raise CalendarBaselineSignatureError("trusted signer key is unavailable")
    public_key_der = trusted_public_keys[signature.key_id]
    if (
        not isinstance(public_key_der, bytes)
        or not 1 <= len(public_key_der) <= _MAX_PUBLIC_KEY_BYTES
    ):
        raise CalendarBaselineSignatureError("trusted signer key is malformed")
    if not hmac.compare_digest(
        hashlib.sha256(public_key_der).hexdigest(),
        signature.public_key_sha256,
    ):
        raise CalendarBaselineSignatureError("trusted signer key revision does not match")
    public_key = _load_ed25519_public_key(public_key_der)
    try:
        public_key.verify(
            _decode_signature(signature.signature),
            _message(_unsigned_document(signature)),
        )
    except InvalidSignature as exc:
        raise CalendarBaselineSignatureError("calendar baseline signature does not verify") from exc

    normalized, payload = _manifest_payload(manifest)
    bindings = {
        "calendar_id": normalized["calendar_id"],
        "revision_id": normalized["revision_id"],
        "tzdata_version": normalized["tzdata_version"],
        "manifest_sha256": normalized["manifest_sha256"],
        "manifest_payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if any(
        not hmac.compare_digest(str(getattr(signature, field)), str(value))
        for field, value in bindings.items()
    ):
        raise CalendarBaselineSignatureError("signature does not bind the supplied manifest")
    return {
        "schema_version": CALENDAR_BASELINE_VERIFICATION_SCHEMA,
        "verified": True,
        "key_id": signature.key_id,
        "public_key_sha256": signature.public_key_sha256,
        "source_commit_sha": signature.source_commit_sha,
        "calendar_id": signature.calendar_id,
        "revision_id": signature.revision_id,
        "tzdata_version": signature.tzdata_version,
        "manifest_sha256": signature.manifest_sha256,
        "signature_document_sha256": signature.signature_document_sha256,
    }


def require_signed_calendar_migration(
    baseline_signature: CalendarBaselineSignature,
    previous: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    trusted_public_keys: Mapping[str, bytes],
    expected_source_commit_sha: str,
    approved_changed_dates: Sequence[str] = (),
) -> dict[str, object]:
    """Authenticate the reviewed baseline, then require an accepted migration."""

    baseline_verification = verify_calendar_baseline_signature(
        baseline_signature,
        previous,
        trusted_public_keys=trusted_public_keys,
        expected_source_commit_sha=expected_source_commit_sha,
    )
    migration_report = require_calendar_migration(
        previous,
        candidate,
        approved_changed_dates=approved_changed_dates,
    )
    result: dict[str, object] = {
        "schema_version": SIGNED_CALENDAR_MIGRATION_SCHEMA,
        "accepted": True,
        "baseline_verification": baseline_verification,
        "migration_report": migration_report,
    }
    result["evidence_sha256"] = _digest(result)
    return result


def dump_calendar_baseline_signature(signature: CalendarBaselineSignature) -> bytes:
    if not isinstance(signature, CalendarBaselineSignature):
        raise TypeError("signature must be a CalendarBaselineSignature")
    document = signature.to_audit_dict()
    _validate_document(document)
    payload = _canonical_bytes(document)
    if len(payload) > _MAX_SIGNATURE_DOCUMENT_BYTES:
        raise CalendarBaselineSignatureError("signature document exceeds the byte limit")
    return payload


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalendarBaselineSignatureError("signature contains a duplicate field")
        result[key] = value
    return result


def load_calendar_baseline_signature(payload: bytes) -> CalendarBaselineSignature:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > _MAX_SIGNATURE_DOCUMENT_BYTES:
        raise CalendarBaselineSignatureError("signature document size is invalid")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CalendarBaselineSignatureError("signature document is not valid JSON") from exc
    if not isinstance(document, dict):
        raise CalendarBaselineSignatureError("signature fields do not match the schema")
    _validate_document(document)
    if _canonical_bytes(document) != payload:
        raise CalendarBaselineSignatureError("signature document is not canonical JSON")
    try:
        return CalendarBaselineSignature(**document)
    except TypeError as exc:  # pragma: no cover - exact fields checked above
        raise CalendarBaselineSignatureError("signature fields do not match the schema") from exc
