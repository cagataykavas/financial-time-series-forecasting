from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from finforecast.regime_labeling import PointInTimeReturn, VolatilityRegimeDefinition
from finforecast.regime_source import (
    ReturnRevisionSnapshotError,
    ReturnRevisionSnapshotSignature,
    assign_authenticated_point_in_time_volatility_regimes,
    create_return_revision_snapshot_signature,
    dump_return_revision_snapshot_signature,
    load_return_revision_snapshot_signature,
    verify_return_revision_snapshot_signature,
)

_SCHEMA_DIGEST = "a" * 64
_VENDOR = "vendor.example"
_DATASET = "daily-total-returns"
_SNAPSHOT = "2026-01-10T000000Z-r1"


class _LocalSigner:
    key_id = "vendor-return-key-2026"

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


def _rows() -> list[PointInTimeReturn]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        PointInTimeReturn(
            observed_at=start + timedelta(days=position),
            available_at=start + timedelta(days=position, hours=1),
            revision_id=f"revision-{position}",
            value=value,
        )
        for position, value in enumerate((-0.1, 0.0, 0.1, -0.2, 0.0, 0.2))
    ]


def _definition() -> VolatilityRegimeDefinition:
    return VolatilityRegimeDefinition(
        definition_id="daily-realized-volatility-v1",
        revision="b" * 40,
        published_at=datetime(2025, 12, 1, tzinfo=UTC),
        source_schema_sha256=_SCHEMA_DIGEST,
        window_observations=3,
        annualization_periods=1,
        low_volatility_upper=0.05,
        normal_volatility_upper=0.15,
    )


def _signature(
    rows: list[PointInTimeReturn],
    signer: _LocalSigner,
) -> ReturnRevisionSnapshotSignature:
    return create_return_revision_snapshot_signature(
        rows,
        vendor_id=_VENDOR,
        dataset_id=_DATASET,
        snapshot_id=_SNAPSHOT,
        issued_at_epoch=1_799_000_000,
        source_schema_sha256=_SCHEMA_DIGEST,
        signer=signer,
    )


def _verify(
    signature: ReturnRevisionSnapshotSignature,
    rows: list[PointInTimeReturn],
    signer: _LocalSigner,
) -> dict[str, object]:
    return verify_return_revision_snapshot_signature(
        signature,
        rows,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_vendor_id=_VENDOR,
        expected_dataset_id=_DATASET,
        expected_snapshot_id=_SNAPSHOT,
        expected_source_schema_sha256=_SCHEMA_DIGEST,
    )


def test_signature_is_deterministic_canonical_and_privacy_reduced() -> None:
    rows = _rows()
    signer = _LocalSigner()

    first = _signature(rows, signer)
    second = _signature(rows, signer)
    payload = dump_return_revision_snapshot_signature(first)
    loaded = load_return_revision_snapshot_signature(payload)
    verification = _verify(loaded, rows, signer)

    assert first == second == loaded
    assert verification["verified"] is True
    assert verification["record_count"] == len(rows)
    assert verification["signature_document_sha256"] == first.signature_document_sha256
    assert len(signer.messages[-1]) < 4_096
    serialized = json.dumps(verification, sort_keys=True)
    assert "revision-0" not in serialized
    assert "-0.1" not in serialized


def test_signature_rejects_value_revision_and_order_substitution() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)

    changed_value = [replace(rows[0], value=0.9), *rows[1:]]
    with pytest.raises(ReturnRevisionSnapshotError, match="does not bind"):
        _verify(signature, changed_value, signer)

    changed_revision = [replace(rows[0], revision_id="replacement"), *rows[1:]]
    with pytest.raises(ReturnRevisionSnapshotError, match="does not bind"):
        _verify(signature, changed_revision, signer)

    with pytest.raises(ReturnRevisionSnapshotError, match="stream is invalid"):
        _verify(signature, [rows[1], rows[0], *rows[2:]], signer)


def test_signature_requires_governed_source_identity_and_schema() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)

    with pytest.raises(ReturnRevisionSnapshotError, match="source identity"):
        verify_return_revision_snapshot_signature(
            signature,
            rows,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_vendor_id="other-vendor",
            expected_dataset_id=_DATASET,
            expected_snapshot_id=_SNAPSHOT,
            expected_source_schema_sha256=_SCHEMA_DIGEST,
        )
    with pytest.raises(ReturnRevisionSnapshotError, match="source schema"):
        verify_return_revision_snapshot_signature(
            signature,
            rows,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_vendor_id=_VENDOR,
            expected_dataset_id=_DATASET,
            expected_snapshot_id=_SNAPSHOT,
            expected_source_schema_sha256="f" * 64,
        )


def test_signature_requires_trusted_key_and_exact_key_revision() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)

    with pytest.raises(ReturnRevisionSnapshotError, match="unavailable"):
        verify_return_revision_snapshot_signature(
            signature,
            rows,
            trusted_public_keys={},
            expected_vendor_id=_VENDOR,
            expected_dataset_id=_DATASET,
            expected_snapshot_id=_SNAPSHOT,
            expected_source_schema_sha256=_SCHEMA_DIGEST,
        )
    replacement = _LocalSigner()
    with pytest.raises(ReturnRevisionSnapshotError, match="key revision"):
        verify_return_revision_snapshot_signature(
            signature,
            rows,
            trusted_public_keys={signer.key_id: replacement.public_key_der},
            expected_vendor_id=_VENDOR,
            expected_dataset_id=_DATASET,
            expected_snapshot_id=_SNAPSHOT,
            expected_source_schema_sha256=_SCHEMA_DIGEST,
        )


def test_authenticated_assignment_binds_snapshot_definition_and_results() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)
    definition = _definition()
    issuances = [rows[position].available_at + timedelta(minutes=1) for position in (2, 5)]

    first_assignments, first_evidence = assign_authenticated_point_in_time_volatility_regimes(
        rows,
        issuances,
        definition,
        signature,
        expected_definition_sha256=definition.sha256,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_vendor_id=_VENDOR,
        expected_dataset_id=_DATASET,
        expected_snapshot_id=_SNAPSHOT,
    )
    second_assignments, second_evidence = assign_authenticated_point_in_time_volatility_regimes(
        rows,
        issuances,
        definition,
        signature,
        expected_definition_sha256=definition.sha256,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_vendor_id=_VENDOR,
        expected_dataset_id=_DATASET,
        expected_snapshot_id=_SNAPSHOT,
    )

    assert first_assignments == second_assignments
    assert first_evidence == second_evidence
    assert first_evidence["accepted"] is True
    assert first_evidence["assignment_count"] == 2
    assert first_evidence["snapshot_verification"]["records_sha256"] == signature.records_sha256
    assert len(first_evidence["evidence_sha256"]) == 64
    serialized = json.dumps(first_evidence, sort_keys=True)
    assert "revision-0" not in serialized
    assert '"value"' not in serialized


def test_authenticated_assignment_rejects_definition_source_schema_mismatch() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)
    definition = replace(_definition(), source_schema_sha256="f" * 64)

    with pytest.raises(ReturnRevisionSnapshotError, match="source schema"):
        assign_authenticated_point_in_time_volatility_regimes(
            rows,
            [rows[2].available_at + timedelta(minutes=1)],
            definition,
            signature,
            expected_definition_sha256=definition.sha256,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_vendor_id=_VENDOR,
            expected_dataset_id=_DATASET,
            expected_snapshot_id=_SNAPSHOT,
        )


def test_loader_and_signature_verification_reject_tampering() -> None:
    rows = _rows()
    signer = _LocalSigner()
    signature = _signature(rows, signer)
    payload = dump_return_revision_snapshot_signature(signature)

    with pytest.raises(ReturnRevisionSnapshotError, match="duplicate"):
        load_return_revision_snapshot_signature(b'{"schema_version":"x","schema_version":"x"}')
    with pytest.raises(ReturnRevisionSnapshotError, match="not canonical"):
        load_return_revision_snapshot_signature(payload + b"\n")

    tampered = ReturnRevisionSnapshotSignature(
        **{**signature.to_audit_dict(), "records_sha256": "f" * 64}
    )
    with pytest.raises(ReturnRevisionSnapshotError, match="does not verify"):
        _verify(tampered, rows, signer)


def test_signing_failure_is_sanitized() -> None:
    class FailingSigner(_LocalSigner):
        def sign(self, message: bytes) -> bytes:
            raise RuntimeError("private vendor diagnostic")

    with pytest.raises(ReturnRevisionSnapshotError, match="signing failed") as error:
        _signature(_rows(), FailingSigner())

    assert "private vendor diagnostic" not in str(error.value)
    assert hashlib.sha256(b"private vendor diagnostic").hexdigest() not in str(error.value)


def test_signer_cannot_substitute_a_key_after_metadata_capture() -> None:
    signer = _LocalSigner()
    replacement_key = Ed25519PrivateKey.generate()

    def substituted_sign(message: bytes) -> bytes:
        return replacement_key.sign(message)

    signer.sign = substituted_sign  # type: ignore[method-assign]

    with pytest.raises(ReturnRevisionSnapshotError, match="retained public key"):
        _signature(_rows(), signer)
