from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from finforecast.snapshot_consistency import create_snapshot_transparency_consistency_proof
from finforecast.snapshot_transparency import (
    SnapshotTransparencyError,
    create_snapshot_transparency_checkpoint,
)
from finforecast.snapshot_witness import (
    SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA,
    SNAPSHOT_CONSISTENCY_WITNESS_VERIFICATION_SCHEMA,
    SNAPSHOT_WITNESSED_PROMOTION_SCHEMA,
    create_snapshot_consistency_witness_bundle,
    dump_snapshot_consistency_witness_bundle,
    load_snapshot_consistency_witness_bundle,
    verify_snapshot_consistency_witness_bundle,
    verify_witnessed_snapshot_transparency_promotion,
)
from tests.test_snapshot_transparency import (
    _CHECKPOINT_EPOCH,
    _CHECKPOINT_ID,
    _LOG_ID,
    _entry,
    _fixture,
    _Signer,
)

_OLD_CHECKPOINT_ID = "checkpoint-before-training"


class _WitnessSigner(_Signer):
    def __init__(self, number: int) -> None:
        super().__init__()
        self.key_id = f"independent-witness-key-{number}"


class _FailingWitness(_WitnessSigner):
    def sign(self, message: bytes) -> bytes:
        raise RuntimeError("private witness provider diagnostic")


def _witnesses(count: int = 3) -> dict[str, _WitnessSigner]:
    return {f"witness-{number}": _WitnessSigner(number) for number in range(1, count + 1)}


def _trusted_witness_keys(witnesses: dict[str, _WitnessSigner]) -> dict[str, bytes]:
    return {signer.key_id: signer.public_key_der for signer in witnesses.values()}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _replace_bundle(bundle, **changes):
    candidate = replace(bundle, **changes, bundle_sha256="0" * 64)
    document = candidate.to_audit_dict()
    document.pop("bundle_sha256")
    return replace(candidate, bundle_sha256=_digest(document))


def _transition(tmp_path):
    log_signer, receipt, entries, new_checkpoint, inclusion_proof = _fixture(tmp_path)
    old_checkpoint = create_snapshot_transparency_checkpoint(
        entries[:3],
        log_id=_LOG_ID,
        checkpoint_id=_OLD_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH - 30,
        signer=log_signer,
    )
    consistency_proof = create_snapshot_transparency_consistency_proof(
        entries,
        old_tree_size=3,
    )
    return (
        log_signer,
        receipt,
        entries,
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        inclusion_proof,
    )


def _bundle(tmp_path, witnesses=None, *, required_witnesses: int = 2):
    transition = _transition(tmp_path)
    log_signer, _receipt, _entries, old, new, proof, _inclusion = transition
    active_witnesses = _witnesses() if witnesses is None else witnesses
    bundle = create_snapshot_consistency_witness_bundle(
        old,
        new,
        proof,
        active_witnesses,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
        expected_new_checkpoint_id=_CHECKPOINT_ID,
        required_witnesses=required_witnesses,
    )
    return transition, active_witnesses, bundle


def test_witnessed_promotion_is_deterministic_cross_bound_and_privacy_reduced(tmp_path) -> None:
    transition, witnesses, bundle = _bundle(tmp_path)
    log_signer, receipt, _entries, old, new, proof, inclusion = transition
    recreated = create_snapshot_consistency_witness_bundle(
        old,
        new,
        proof,
        witnesses,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
        expected_new_checkpoint_id=_CHECKPOINT_ID,
        required_witnesses=2,
    )
    evidence = verify_witnessed_snapshot_transparency_promotion(
        receipt,
        old,
        new,
        proof,
        inclusion,
        bundle,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_witness_public_keys=_trusted_witness_keys(witnesses),
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
        expected_new_checkpoint_id=_CHECKPOINT_ID,
    )

    assert bundle == recreated
    assert bundle.schema_version == SNAPSHOT_CONSISTENCY_WITNESS_SCHEMA
    assert evidence["schema_version"] == SNAPSHOT_WITNESSED_PROMOTION_SCHEMA
    assert evidence["verified"] is True
    assert evidence["required_witnesses"] == 2
    assert evidence["valid_witnesses"] == 3
    assert evidence["entry_sha256"] == receipt.entry_sha256
    assert _LOG_ID not in str(evidence)
    assert _CHECKPOINT_ID not in str(evidence)
    assert "witness-1" not in str(evidence)
    assert witnesses["witness-1"].key_id not in str(evidence)


def test_quorum_tolerates_one_unavailable_or_invalid_witness(tmp_path) -> None:
    transition, witnesses, bundle = _bundle(tmp_path)
    log_signer, _receipt, _entries, old, new, proof, _inclusion = transition
    trusted = _trusted_witness_keys(witnesses)
    unavailable = verify_snapshot_consistency_witness_bundle(
        old,
        new,
        proof,
        bundle,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_witness_public_keys=dict(list(trusted.items())[:2]),
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
        expected_new_checkpoint_id=_CHECKPOINT_ID,
    )
    approval = bundle.approvals[2]
    changed_signature = ("A" if approval.signature[0] != "A" else "B") + approval.signature[1:]
    tampered = _replace_bundle(
        bundle,
        approvals=(*bundle.approvals[:2], replace(approval, signature=changed_signature)),
    )
    invalid = verify_snapshot_consistency_witness_bundle(
        old,
        new,
        proof,
        tampered,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_witness_public_keys=trusted,
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
        expected_new_checkpoint_id=_CHECKPOINT_ID,
    )

    assert unavailable["schema_version"] == SNAPSHOT_CONSISTENCY_WITNESS_VERIFICATION_SCHEMA
    assert unavailable["valid_witnesses"] == 2
    assert invalid["valid_witnesses"] == 2


def test_threshold_and_complete_witness_set_are_bound_into_each_signature(tmp_path) -> None:
    transition, witnesses, bundle = _bundle(tmp_path)
    log_signer, _receipt, _entries, old, new, proof, _inclusion = transition
    trusted = _trusted_witness_keys(witnesses)
    lowered_or_raised = _replace_bundle(bundle, required_witnesses=3)

    with pytest.raises(SnapshotTransparencyError, match="quorum is not satisfied"):
        verify_snapshot_consistency_witness_bundle(
            old,
            new,
            proof,
            lowered_or_raised,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_witness_public_keys=trusted,
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )
    reordered = _replace_bundle(bundle, approvals=tuple(reversed(bundle.approvals)))
    with pytest.raises(SnapshotTransparencyError, match="canonically ordered"):
        verify_snapshot_consistency_witness_bundle(
            old,
            new,
            proof,
            reordered,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_witness_public_keys=trusted,
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )


def test_bundle_cannot_be_reused_for_another_valid_transition(tmp_path) -> None:
    transition, witnesses, bundle = _bundle(tmp_path)
    log_signer, _receipt, entries, old, _new, _proof, _inclusion = transition
    alternate_entries = [*entries]
    alternate_entries[-1] = _entry("alternate-valid-tail")
    alternate_new = create_snapshot_transparency_checkpoint(
        alternate_entries,
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH,
        signer=log_signer,
    )
    alternate_proof = create_snapshot_transparency_consistency_proof(
        alternate_entries,
        old_tree_size=3,
    )

    with pytest.raises(SnapshotTransparencyError, match="different consistency transition"):
        verify_snapshot_consistency_witness_bundle(
            old,
            alternate_new,
            alternate_proof,
            bundle,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_witness_public_keys=_trusted_witness_keys(witnesses),
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )


def test_creation_rejects_reused_key_revisions_and_sanitizes_provider_errors(tmp_path) -> None:
    transition = _transition(tmp_path)
    log_signer, _receipt, _entries, old, new, proof, _inclusion = transition
    first = _WitnessSigner(1)
    alias = _WitnessSigner(2)
    alias.public_key_der = first.public_key_der

    with pytest.raises(SnapshotTransparencyError, match="key revisions must be distinct"):
        create_snapshot_consistency_witness_bundle(
            old,
            new,
            proof,
            {"witness-1": first, "witness-2": alias},
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )
    with pytest.raises(SnapshotTransparencyError, match="witness signing failed") as error:
        create_snapshot_consistency_witness_bundle(
            old,
            new,
            proof,
            {"witness-1": first, "witness-2": _FailingWitness(2)},
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )

    assert "private witness provider diagnostic" not in str(error.value)


def test_unknown_or_substituted_witness_keys_cannot_satisfy_quorum(tmp_path) -> None:
    transition, witnesses, bundle = _bundle(tmp_path)
    log_signer, _receipt, _entries, old, new, proof, _inclusion = transition
    wrong = _WitnessSigner(99)
    trusted = _trusted_witness_keys(witnesses)
    trusted[witnesses["witness-1"].key_id] = wrong.public_key_der
    trusted.pop(witnesses["witness-2"].key_id)

    with pytest.raises(SnapshotTransparencyError, match="quorum is not satisfied"):
        verify_snapshot_consistency_witness_bundle(
            old,
            new,
            proof,
            bundle,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_witness_public_keys=trusted,
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=_OLD_CHECKPOINT_ID,
            expected_new_checkpoint_id=_CHECKPOINT_ID,
        )


def test_witness_bundle_loader_is_canonical_bounded_and_strict(tmp_path) -> None:
    _transition_data, _witnesses_data, bundle = _bundle(tmp_path)
    payload = dump_snapshot_consistency_witness_bundle(bundle)

    assert load_snapshot_consistency_witness_bundle(payload) == bundle
    with pytest.raises(SnapshotTransparencyError, match="not canonical"):
        load_snapshot_consistency_witness_bundle(payload + b"\n")
    document = json.loads(payload)
    document["unreviewed"] = True
    with pytest.raises(SnapshotTransparencyError, match="fields"):
        load_snapshot_consistency_witness_bundle(
            json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
        )
    with pytest.raises(SnapshotTransparencyError, match="duplicate keys"):
        load_snapshot_consistency_witness_bundle(
            b'{"schema_version":"first","schema_version":"second"}'
        )
    with pytest.raises(SnapshotTransparencyError, match="exceeds"):
        load_snapshot_consistency_witness_bundle(b"x" * 16_385)
