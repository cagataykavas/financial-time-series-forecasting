"""Compare independently relayed transparency checkpoints for split views."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key

from finforecast.snapshot_consistency import SnapshotTransparencyConsistencyProof
from finforecast.snapshot_observation import SnapshotObservationReceipt
from finforecast.snapshot_transparency import (
    SnapshotTransparencyCheckpoint,
    SnapshotTransparencyError,
    SnapshotTransparencyProof,
    SnapshotTransparencySigner,
    authenticate_snapshot_transparency_checkpoint,
)
from finforecast.snapshot_witness import (
    SnapshotConsistencyWitnessBundle,
    verify_witnessed_snapshot_transparency_promotion,
)

SNAPSHOT_GOSSIP_OBSERVATION_SCHEMA = "finforecast-snapshot-gossip-observation/v1"
SNAPSHOT_GOSSIP_VERIFICATION_SCHEMA = "finforecast-snapshot-gossip-verification/v1"
SNAPSHOT_GOSSIPED_PROMOTION_SCHEMA = "finforecast-snapshot-gossiped-promotion/v1"
SNAPSHOT_GOSSIP_PURPOSE = "independent-checkpoint-gossip"
SNAPSHOT_GOSSIP_ALGORITHM = "ed25519-sha512"
MAX_GOSSIP_OBSERVATION_AGE_SECONDS = 15 * 60

_DOMAIN = b"finforecast/snapshot-gossip-observation/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_PUBLIC_KEY_BYTES = 8_192
_MAX_RELAYS = 16
_MAX_OBSERVATION_BYTES = 4_096
_OBSERVATION_FIELDS = {
    "schema_version",
    "purpose",
    "algorithm",
    "relay_id",
    "checkpoint_authentication_sha256",
    "checkpoint_sha256",
    "tree_size",
    "root_sha256",
    "checkpoint_issued_at_epoch",
    "observed_at_epoch",
    "key_id",
    "public_key_sha256",
    "signature",
}


@dataclass(frozen=True, slots=True)
class SnapshotCheckpointGossipObservation:
    """One relay's signed observation of an authenticated log checkpoint."""

    schema_version: str
    purpose: str
    algorithm: str
    relay_id: str
    checkpoint_authentication_sha256: str
    checkpoint_sha256: str
    tree_size: int
    root_sha256: str
    checkpoint_issued_at_epoch: int
    observed_at_epoch: int
    key_id: str
    public_key_sha256: str
    signature: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def observation_sha256(self) -> str:
        return hashlib.sha256(dump_snapshot_checkpoint_gossip_observation(self)).hexdigest()


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
        raise SnapshotTransparencyError("gossip evidence is not canonical JSON") from exc


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


def _require_epoch(value: object, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 2**63 - 1:
        raise SnapshotTransparencyError(f"{name} is invalid")
    return value


def _load_public_key(encoded: object) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= _MAX_PUBLIC_KEY_BYTES:
        raise SnapshotTransparencyError("gossip relay public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise SnapshotTransparencyError("gossip relay public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SnapshotTransparencyError("gossip relay public key is not Ed25519")
    return key


def _encode_signature(value: object) -> str:
    if not isinstance(value, bytes) or len(value) != 64:
        raise SnapshotTransparencyError("gossip relay returned a malformed signature")
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str) or _SIGNATURE_RE.fullmatch(value) is None:
        raise SnapshotTransparencyError("gossip signature encoding is malformed")
    try:
        decoded = base64.b64decode(value + "==", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise SnapshotTransparencyError("gossip signature encoding is malformed") from exc
    if len(decoded) != 64 or _encode_signature(decoded) != value:
        raise SnapshotTransparencyError("gossip signature encoding is malformed")
    return decoded


def _unsigned(observation: SnapshotCheckpointGossipObservation) -> dict[str, object]:
    document = observation.to_audit_dict()
    document.pop("signature")
    return document


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_bytes(dict(document))


def _validate_observation(observation: SnapshotCheckpointGossipObservation) -> None:
    if not isinstance(observation, SnapshotCheckpointGossipObservation):
        raise TypeError("observation must be a SnapshotCheckpointGossipObservation")
    document = observation.to_audit_dict()
    if set(document) != _OBSERVATION_FIELDS:
        raise SnapshotTransparencyError("gossip observation fields do not match the schema")
    if (
        observation.schema_version != SNAPSHOT_GOSSIP_OBSERVATION_SCHEMA
        or observation.purpose != SNAPSHOT_GOSSIP_PURPOSE
        or observation.algorithm != SNAPSHOT_GOSSIP_ALGORITHM
    ):
        raise SnapshotTransparencyError("gossip observation policy is unsupported")
    _identifier(observation.relay_id, "relay_id")
    _identifier(observation.key_id, "key_id")
    _require_digest(
        observation.checkpoint_authentication_sha256,
        "checkpoint_authentication_sha256",
    )
    _require_digest(observation.checkpoint_sha256, "checkpoint_sha256")
    _require_digest(observation.root_sha256, "root_sha256")
    _require_digest(observation.public_key_sha256, "public_key_sha256")
    _require_epoch(observation.checkpoint_issued_at_epoch, "checkpoint_issued_at_epoch")
    _require_epoch(observation.observed_at_epoch, "observed_at_epoch")
    if type(observation.tree_size) is not int or not 1 <= observation.tree_size <= 1_000_000:
        raise SnapshotTransparencyError("gossip tree size is invalid")
    if observation.observed_at_epoch < observation.checkpoint_issued_at_epoch:
        raise SnapshotTransparencyError("gossip observation predates its checkpoint")
    _decode_signature(observation.signature)


def create_snapshot_checkpoint_gossip_observation(
    checkpoint: SnapshotTransparencyCheckpoint,
    relay_id: str,
    relay: SnapshotTransparencySigner,
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
    observed_at_epoch: int,
) -> SnapshotCheckpointGossipObservation:
    """Authenticate and sign one relay's exact checkpoint observation."""

    authentication = authenticate_snapshot_transparency_checkpoint(
        checkpoint,
        trusted_public_keys=trusted_log_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_checkpoint_id,
    )
    normalized_relay_id = _identifier(relay_id, "relay_id")
    observed_at = _require_epoch(observed_at_epoch, "observed_at_epoch")
    if observed_at < checkpoint.issued_at_epoch:
        raise SnapshotTransparencyError("gossip observation predates its checkpoint")
    try:
        key_id = _identifier(relay.key_id, "key_id")
        public_key_der = relay.public_key_der
    except Exception as exc:
        raise SnapshotTransparencyError("gossip relay metadata is unavailable") from exc
    public_key = _load_public_key(public_key_der)
    unsigned: dict[str, object] = {
        "schema_version": SNAPSHOT_GOSSIP_OBSERVATION_SCHEMA,
        "purpose": SNAPSHOT_GOSSIP_PURPOSE,
        "algorithm": SNAPSHOT_GOSSIP_ALGORITHM,
        "relay_id": normalized_relay_id,
        "checkpoint_authentication_sha256": authentication["authentication_sha256"],
        "checkpoint_sha256": authentication["checkpoint_sha256"],
        "tree_size": authentication["tree_size"],
        "root_sha256": authentication["root_sha256"],
        "checkpoint_issued_at_epoch": authentication["issued_at_epoch"],
        "observed_at_epoch": observed_at,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_key_der).hexdigest(),
    }
    try:
        signature = relay.sign(_message(unsigned))
    except Exception as exc:
        raise SnapshotTransparencyError("gossip relay signing failed") from exc
    encoded = _encode_signature(signature)
    try:
        public_key.verify(signature, _message(unsigned))
    except InvalidSignature as exc:
        raise SnapshotTransparencyError(
            "gossip relay output does not match its retained public key"
        ) from exc
    observation = SnapshotCheckpointGossipObservation(**unsigned, signature=encoded)
    _validate_observation(observation)
    return observation


def _validated_snapshot_checkpoint_gossip_views(
    views: Sequence[tuple[SnapshotTransparencyCheckpoint, SnapshotCheckpointGossipObservation]],
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_relay_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
    verification_epoch: int,
    required_relays: int = 2,
    max_observation_age_seconds: int = MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
) -> tuple[
    list[tuple[SnapshotCheckpointGossipObservation, dict[str, object]]],
    int,
]:
    """Authenticate fresh, distinct gossip views and enforce the relay quorum."""

    if (
        isinstance(views, (str, bytes))
        or not isinstance(views, Sequence)
        or not 2 <= len(views) <= _MAX_RELAYS
    ):
        raise SnapshotTransparencyError("gossip views must contain from 2 through 16 observations")
    if type(required_relays) is not int or not 2 <= required_relays <= len(views):
        raise SnapshotTransparencyError("required_relays is invalid")
    verified_at = _require_epoch(verification_epoch, "verification_epoch")
    if (
        type(max_observation_age_seconds) is not int
        or not 1 <= max_observation_age_seconds <= 60 * 60
    ):
        raise SnapshotTransparencyError("max_observation_age_seconds is invalid")
    if not isinstance(trusted_relay_public_keys, Mapping):
        raise SnapshotTransparencyError("trusted relay keys must be a mapping")

    observations: list[SnapshotCheckpointGossipObservation] = []
    authentications: list[dict[str, object]] = []
    for view in views:
        if not isinstance(view, tuple) or len(view) != 2:
            raise SnapshotTransparencyError("gossip view is malformed")
        checkpoint, observation = view
        _validate_observation(observation)
        authentication = authenticate_snapshot_transparency_checkpoint(
            checkpoint,
            trusted_public_keys=trusted_log_public_keys,
            expected_log_id=expected_log_id,
            expected_checkpoint_id=expected_checkpoint_id,
        )
        expected = {
            "checkpoint_authentication_sha256": authentication["authentication_sha256"],
            "checkpoint_sha256": authentication["checkpoint_sha256"],
            "tree_size": authentication["tree_size"],
            "root_sha256": authentication["root_sha256"],
            "checkpoint_issued_at_epoch": authentication["issued_at_epoch"],
        }
        actual = {key: getattr(observation, key) for key in expected}
        if actual != expected:
            raise SnapshotTransparencyError("gossip observation does not match its checkpoint")
        observations.append(observation)
        authentications.append(authentication)

    if len({item.relay_id for item in observations}) != len(observations):
        raise SnapshotTransparencyError("gossip relay identifiers must be distinct")
    if len({item.key_id for item in observations}) != len(observations):
        raise SnapshotTransparencyError("gossip relay key identifiers must be distinct")
    if len({item.public_key_sha256 for item in observations}) != len(observations):
        raise SnapshotTransparencyError("gossip relay key revisions must be distinct")

    valid: list[tuple[SnapshotCheckpointGossipObservation, dict[str, object]]] = []
    for observation, authentication in zip(observations, authentications, strict=True):
        public_key_der = trusted_relay_public_keys.get(observation.key_id)
        if public_key_der is None:
            continue
        try:
            public_key = _load_public_key(public_key_der)
        except SnapshotTransparencyError:
            continue
        revision = hashlib.sha256(public_key_der).hexdigest()
        if not hmac.compare_digest(revision, observation.public_key_sha256):
            continue
        try:
            public_key.verify(
                _decode_signature(observation.signature),
                _message(_unsigned(observation)),
            )
        except (InvalidSignature, SnapshotTransparencyError):
            continue
        if observation.observed_at_epoch > verified_at:
            raise SnapshotTransparencyError("gossip observation is from the future")
        if verified_at - observation.observed_at_epoch > max_observation_age_seconds:
            raise SnapshotTransparencyError("gossip observation is stale")
        valid.append((observation, authentication))

    if len(valid) < required_relays:
        raise SnapshotTransparencyError("gossip relay quorum is not satisfied")
    return valid, verified_at


def verify_snapshot_checkpoint_gossip(
    views: Sequence[tuple[SnapshotTransparencyCheckpoint, SnapshotCheckpointGossipObservation]],
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_relay_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_checkpoint_id: str,
    verification_epoch: int,
    required_relays: int = 2,
    max_observation_age_seconds: int = MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
) -> dict[str, object]:
    """Require fresh relay quorum and reject any trusted authenticated split view."""

    valid, verified_at = _validated_snapshot_checkpoint_gossip_views(
        views,
        trusted_log_public_keys=trusted_log_public_keys,
        trusted_relay_public_keys=trusted_relay_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_checkpoint_id,
        verification_epoch=verification_epoch,
        required_relays=required_relays,
        max_observation_age_seconds=max_observation_age_seconds,
    )
    checkpoint_views = {
        (
            str(authentication["checkpoint_sha256"]),
            int(authentication["tree_size"]),
            str(authentication["root_sha256"]),
        )
        for _observation, authentication in valid
    }
    if len(checkpoint_views) != 1:
        raise SnapshotTransparencyError("gossip detected conflicting authenticated checkpoints")

    checkpoint_sha256, tree_size, root_sha256 = checkpoint_views.pop()
    valid_observations = sorted(item.observation_sha256 for item, _authentication in valid)
    valid_revisions = sorted(item.public_key_sha256 for item, _authentication in valid)
    verification: dict[str, object] = {
        "schema_version": SNAPSHOT_GOSSIP_VERIFICATION_SCHEMA,
        "verified": True,
        "log_id_sha256": valid[0][1]["log_id_sha256"],
        "checkpoint_id_sha256": valid[0][1]["checkpoint_id_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "tree_size": tree_size,
        "root_sha256": root_sha256,
        "verified_at_epoch": verified_at,
        "max_observation_age_seconds": max_observation_age_seconds,
        "required_relays": required_relays,
        "valid_relays": len(valid),
        "valid_observations_sha256": _digest(valid_observations),
        "valid_relay_revisions_sha256": _digest(valid_revisions),
    }
    verification["verification_sha256"] = _digest(verification)
    return verification


def verify_gossiped_snapshot_transparency_promotion(
    receipt: SnapshotObservationReceipt,
    old_checkpoint: SnapshotTransparencyCheckpoint,
    new_checkpoint: SnapshotTransparencyCheckpoint,
    consistency_proof: SnapshotTransparencyConsistencyProof,
    inclusion_proof: SnapshotTransparencyProof,
    witness_bundle: SnapshotConsistencyWitnessBundle,
    gossip_views: Sequence[
        tuple[SnapshotTransparencyCheckpoint, SnapshotCheckpointGossipObservation]
    ],
    *,
    trusted_log_public_keys: Mapping[str, bytes],
    trusted_witness_public_keys: Mapping[str, bytes],
    trusted_relay_public_keys: Mapping[str, bytes],
    expected_log_id: str,
    expected_old_checkpoint_id: str,
    expected_new_checkpoint_id: str,
    verification_epoch: int,
    required_relays: int = 2,
    max_observation_age_seconds: int = MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
) -> dict[str, object]:
    """Cross-bind observation inclusion, append-only witnesses, and checkpoint gossip."""

    promotion = verify_witnessed_snapshot_transparency_promotion(
        receipt,
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        inclusion_proof,
        witness_bundle,
        trusted_log_public_keys=trusted_log_public_keys,
        trusted_witness_public_keys=trusted_witness_public_keys,
        expected_log_id=expected_log_id,
        expected_old_checkpoint_id=expected_old_checkpoint_id,
        expected_new_checkpoint_id=expected_new_checkpoint_id,
    )
    gossip = verify_snapshot_checkpoint_gossip(
        gossip_views,
        trusted_log_public_keys=trusted_log_public_keys,
        trusted_relay_public_keys=trusted_relay_public_keys,
        expected_log_id=expected_log_id,
        expected_checkpoint_id=expected_new_checkpoint_id,
        verification_epoch=verification_epoch,
        required_relays=required_relays,
        max_observation_age_seconds=max_observation_age_seconds,
    )
    if not hmac.compare_digest(
        str(promotion["new_checkpoint_sha256"]),
        str(gossip["checkpoint_sha256"]),
    ):
        raise SnapshotTransparencyError("gossip and promotion evidence do not cross-bind")
    combined: dict[str, object] = {
        "schema_version": SNAPSHOT_GOSSIPED_PROMOTION_SCHEMA,
        "verified": True,
        "identity_sha256": promotion["identity_sha256"],
        "entry_sha256": promotion["entry_sha256"],
        "old_checkpoint_sha256": promotion["old_checkpoint_sha256"],
        "new_checkpoint_sha256": promotion["new_checkpoint_sha256"],
        "old_tree_size": promotion["old_tree_size"],
        "new_tree_size": promotion["new_tree_size"],
        "witnessed_promotion_verification_sha256": promotion["verification_sha256"],
        "gossip_verification_sha256": gossip["verification_sha256"],
        "required_witnesses": promotion["required_witnesses"],
        "valid_witnesses": promotion["valid_witnesses"],
        "required_relays": gossip["required_relays"],
        "valid_relays": gossip["valid_relays"],
    }
    combined["verification_sha256"] = _digest(combined)
    return combined


def dump_snapshot_checkpoint_gossip_observation(
    observation: SnapshotCheckpointGossipObservation,
) -> bytes:
    """Serialize one strict gossip observation as bounded canonical JSON."""

    _validate_observation(observation)
    payload = _canonical_bytes(observation.to_audit_dict())
    if len(payload) > _MAX_OBSERVATION_BYTES:
        raise SnapshotTransparencyError("gossip observation exceeds the byte limit")
    return payload


def load_snapshot_checkpoint_gossip_observation(
    payload: bytes,
) -> SnapshotCheckpointGossipObservation:
    """Load bounded, duplicate-free canonical gossip observation JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_OBSERVATION_BYTES:
        raise SnapshotTransparencyError("gossip observation is empty or exceeds the byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SnapshotTransparencyError("gossip observation contains duplicate keys")
            result[key] = value
        return result

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SnapshotTransparencyError(f"gossip observation contains unsupported {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotTransparencyError("gossip observation JSON is malformed") from exc
    if not isinstance(document, dict) or set(document) != _OBSERVATION_FIELDS:
        raise SnapshotTransparencyError("gossip observation fields do not match the schema")
    try:
        observation = SnapshotCheckpointGossipObservation(**document)
    except TypeError as exc:
        raise SnapshotTransparencyError(
            "gossip observation fields do not match the schema"
        ) from exc
    _validate_observation(observation)
    if payload != _canonical_bytes(observation.to_audit_dict()):
        raise SnapshotTransparencyError("gossip observation JSON is not canonical")
    return observation
