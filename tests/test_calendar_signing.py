from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from finforecast.business_calendar import PINNED_TZDATA_VERSION, load_business_calendar
from finforecast.calendar_migration import (
    CalendarMigrationError,
    build_calendar_behavior_manifest,
)
from finforecast.calendar_signing import (
    CalendarBaselineSignature,
    CalendarBaselineSignatureError,
    KmsEd25519Signer,
    create_calendar_baseline_signature,
    dump_calendar_baseline_signature,
    load_calendar_baseline_signature,
    require_signed_calendar_migration,
    verify_calendar_baseline_signature,
)

_SOURCE_COMMIT = "a" * 40


class _LocalSigner:
    key_id = "calendar-release-key-2026"

    def __init__(self, private_key: Ed25519PrivateKey | None = None) -> None:
        self.private_key = private_key or Ed25519PrivateKey.generate()
        self.public_key_der = self.private_key.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        self.messages: list[bytes] = []

    def sign(self, message: bytes) -> bytes:
        self.messages.append(message)
        return self.private_key.sign(message)


class _FakeKmsClient:
    provider_key_id = "arn:aws:kms:eu-central-1:111122223333:key/key-id"

    def __init__(self, signer: _LocalSigner) -> None:
        self.signer = signer
        self.sign_requests: list[dict[str, object]] = []

    def get_public_key(self, **kwargs: object) -> dict[str, object]:
        assert kwargs == {"KeyId": "alias/calendar-release"}
        return {
            "KeyId": self.provider_key_id,
            "KeyUsage": "SIGN_VERIFY",
            "SigningAlgorithms": ["ED25519_SHA_512"],
            "PublicKey": self.signer.public_key_der,
        }

    def sign(self, **kwargs: object) -> dict[str, object]:
        self.sign_requests.append(dict(kwargs))
        message = kwargs["Message"]
        assert isinstance(message, bytes)
        return {
            "KeyId": self.provider_key_id,
            "SigningAlgorithm": "ED25519_SHA_512",
            "Signature": self.signer.private_key.sign(message),
        }


def _calendar_artifact() -> dict[str, object]:
    return {
        "schema_version": 2,
        "calendar_id": "USD-LOCAL-SETTLEMENT",
        "revision_id": "2026f-r1",
        "timezone": "America/New_York",
        "tzdata_version": PINNED_TZDATA_VERSION,
        "valid_from": "2024-03-07",
        "valid_through": "2024-03-13",
        "weekend_days": [5, 6],
        "holidays": ["2024-03-08"],
    }


def _manifest(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "calendar.json"
    path.write_text(json.dumps(_calendar_artifact()), encoding="utf-8")
    return build_calendar_behavior_manifest(load_business_calendar(path))


def _resign_manifest(manifest: dict[str, object]) -> None:
    document = dict(manifest)
    document.pop("manifest_sha256", None)
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()


def _previous_manifest(candidate: dict[str, object]) -> dict[str, object]:
    previous = copy.deepcopy(candidate)
    previous["revision_id"] = "2026e-r1"
    previous["tzdata_version"] = "2026.5"
    _resign_manifest(previous)
    return previous


def _signature(manifest: dict[str, object], signer: _LocalSigner) -> CalendarBaselineSignature:
    return create_calendar_baseline_signature(
        manifest,
        source_commit_sha=_SOURCE_COMMIT,
        issued_at_epoch=1_793_500_000,
        signer=signer,
    )


def test_baseline_signature_is_deterministic_verifiable_and_canonical(tmp_path: Path) -> None:
    manifest = _previous_manifest(_manifest(tmp_path))
    signer = _LocalSigner()

    first = _signature(manifest, signer)
    second = _signature(dict(reversed(tuple(manifest.items()))), signer)
    payload = dump_calendar_baseline_signature(first)
    loaded = load_calendar_baseline_signature(payload)
    verification = verify_calendar_baseline_signature(
        loaded,
        manifest,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_source_commit_sha=_SOURCE_COMMIT,
    )

    assert first == second == loaded
    assert verification["verified"] is True
    assert verification["key_id"] == signer.key_id
    assert verification["manifest_sha256"] == manifest["manifest_sha256"]
    assert verification["signature_document_sha256"] == first.signature_document_sha256
    assert payload == dump_calendar_baseline_signature(loaded)
    assert len(signer.messages[-1]) < 4_096


def test_signed_migration_authenticates_baseline_before_comparison(tmp_path: Path) -> None:
    candidate = _manifest(tmp_path)
    previous = _previous_manifest(candidate)
    signer = _LocalSigner()
    baseline_signature = _signature(previous, signer)

    result = require_signed_calendar_migration(
        baseline_signature,
        previous,
        candidate,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_source_commit_sha=_SOURCE_COMMIT,
    )

    assert result["accepted"] is True
    assert result["baseline_verification"]["verified"] is True
    assert result["migration_report"]["accepted"] is True
    assert result["migration_report"]["previous_manifest_sha256"] == previous["manifest_sha256"]
    assert len(result["evidence_sha256"]) == 64


def test_signature_rejects_manifest_and_signature_substitution(tmp_path: Path) -> None:
    previous = _previous_manifest(_manifest(tmp_path))
    signer = _LocalSigner()
    signature = _signature(previous, signer)

    substituted = copy.deepcopy(previous)
    substituted["revision_id"] = "substituted-r1"
    _resign_manifest(substituted)
    with pytest.raises(CalendarBaselineSignatureError, match="does not bind"):
        verify_calendar_baseline_signature(
            signature,
            substituted,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_source_commit_sha=_SOURCE_COMMIT,
        )

    tampered = CalendarBaselineSignature(
        **{**signature.to_audit_dict(), "manifest_sha256": "f" * 64}
    )
    with pytest.raises(CalendarBaselineSignatureError, match="does not verify"):
        verify_calendar_baseline_signature(
            tampered,
            previous,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_source_commit_sha=_SOURCE_COMMIT,
        )


def test_signature_requires_expected_commit_and_trusted_key_revision(tmp_path: Path) -> None:
    previous = _previous_manifest(_manifest(tmp_path))
    signer = _LocalSigner()
    signature = _signature(previous, signer)

    with pytest.raises(CalendarBaselineSignatureError, match="source commit"):
        verify_calendar_baseline_signature(
            signature,
            previous,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_source_commit_sha="b" * 40,
        )
    with pytest.raises(CalendarBaselineSignatureError, match="unavailable"):
        verify_calendar_baseline_signature(
            signature,
            previous,
            trusted_public_keys={},
            expected_source_commit_sha=_SOURCE_COMMIT,
        )
    replacement = _LocalSigner()
    with pytest.raises(CalendarBaselineSignatureError, match="revision"):
        verify_calendar_baseline_signature(
            signature,
            previous,
            trusted_public_keys={signer.key_id: replacement.public_key_der},
            expected_source_commit_sha=_SOURCE_COMMIT,
        )


def test_signed_migration_still_fails_closed_on_unreviewed_drift(tmp_path: Path) -> None:
    candidate = _manifest(tmp_path)
    previous = _previous_manifest(candidate)
    signer = _LocalSigner()
    signature = _signature(previous, signer)
    candidate["day_digests"][0]["sha256"] = "f" * 64
    _resign_manifest(candidate)

    with pytest.raises(CalendarMigrationError, match="UNAPPROVED_BEHAVIOR_CHANGE"):
        require_signed_calendar_migration(
            signature,
            previous,
            candidate,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_source_commit_sha=_SOURCE_COMMIT,
        )


def test_signature_loader_rejects_duplicate_and_noncanonical_json(tmp_path: Path) -> None:
    signer = _LocalSigner()
    signature = _signature(_previous_manifest(_manifest(tmp_path)), signer)
    payload = dump_calendar_baseline_signature(signature)

    with pytest.raises(CalendarBaselineSignatureError, match="duplicate"):
        load_calendar_baseline_signature(b'{"schema_version":"x","schema_version":"x"}')
    with pytest.raises(CalendarBaselineSignatureError, match="not canonical"):
        load_calendar_baseline_signature(payload + b"\n")


def test_kms_adapter_pins_public_key_and_uses_raw_ed25519(tmp_path: Path) -> None:
    local_signer = _LocalSigner()
    client = _FakeKmsClient(local_signer)
    signer = KmsEd25519Signer(
        client,
        kms_key_ref="alias/calendar-release",
        key_id=local_signer.key_id,
    )
    manifest = _previous_manifest(_manifest(tmp_path))

    signature = _signature(manifest, signer)
    verify_calendar_baseline_signature(
        signature,
        manifest,
        trusted_public_keys={local_signer.key_id: local_signer.public_key_der},
        expected_source_commit_sha=_SOURCE_COMMIT,
    )

    assert len(client.sign_requests) == 1
    request = client.sign_requests[0]
    assert request["KeyId"] == client.provider_key_id
    assert request["MessageType"] == "RAW"
    assert request["SigningAlgorithm"] == "ED25519_SHA_512"
    assert len(request["Message"]) < 4_096


@pytest.mark.parametrize(
    "response,match",
    [
        ({}, "malformed"),
        (
            {
                "KeyId": _FakeKmsClient.provider_key_id,
                "KeyUsage": "ENCRYPT_DECRYPT",
                "SigningAlgorithms": ["ED25519_SHA_512"],
                "PublicKey": b"not-a-key",
            },
            "malformed",
        ),
    ],
)
def test_kms_adapter_rejects_malformed_public_key_contract(response, match) -> None:
    class Client:
        def get_public_key(self, **kwargs: object) -> dict[str, object]:
            return response

    with pytest.raises(CalendarBaselineSignatureError, match=match):
        KmsEd25519Signer(Client(), kms_key_ref="alias/key", key_id="release-key")


def test_kms_adapter_sanitizes_provider_failure() -> None:
    class Client:
        def get_public_key(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("secret provider diagnostic")

    with pytest.raises(CalendarBaselineSignatureError, match="lookup failed") as error:
        KmsEd25519Signer(Client(), kms_key_ref="alias/key", key_id="release-key")
    assert "secret provider diagnostic" not in str(error.value)
