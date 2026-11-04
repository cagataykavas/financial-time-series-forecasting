from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from finforecast.snapshot_consistency import (
    SNAPSHOT_APPEND_ONLY_INCLUSION_SCHEMA,
    SNAPSHOT_CONSISTENCY_PROOF_SCHEMA,
    SNAPSHOT_CONSISTENCY_VERIFICATION_SCHEMA,
    SnapshotTransparencyConsistencyProof,
    create_snapshot_transparency_consistency_proof,
    dump_snapshot_transparency_consistency_proof,
    load_snapshot_transparency_consistency_proof,
    verify_snapshot_transparency_append_only_inclusion,
    verify_snapshot_transparency_consistency,
)
from finforecast.snapshot_transparency import (
    SnapshotTransparencyError,
    create_snapshot_transparency_checkpoint,
)
from tests.test_snapshot_transparency import (
    _CHECKPOINT_EPOCH,
    _CHECKPOINT_ID,
    _LOG_ID,
    _entry,
    _fixture,
    _Signer,
)


def _checkpoint(entries, signer, *, checkpoint_id: str, epoch: int):
    return create_snapshot_transparency_checkpoint(
        entries,
        log_id=_LOG_ID,
        checkpoint_id=checkpoint_id,
        issued_at_epoch=epoch,
        signer=signer,
    )


@pytest.mark.parametrize("new_size", range(2, 11))
def test_consistency_verifies_every_strict_prefix_size(new_size: int) -> None:
    signer = _Signer()
    entries = [_entry(f"entry-{index}") for index in range(new_size)]
    for old_size in range(1, new_size):
        old_id = f"checkpoint-old-{old_size}"
        new_id = f"checkpoint-new-{new_size}"
        old = _checkpoint(entries[:old_size], signer, checkpoint_id=old_id, epoch=100)
        new = _checkpoint(entries, signer, checkpoint_id=new_id, epoch=200)
        proof = create_snapshot_transparency_consistency_proof(
            entries,
            old_tree_size=old_size,
        )

        verification = verify_snapshot_transparency_consistency(
            old,
            new,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id=old_id,
            expected_new_checkpoint_id=new_id,
        )

        assert verification["schema_version"] == SNAPSHOT_CONSISTENCY_VERIFICATION_SCHEMA
        assert verification["verified"] is True
        assert verification["old_tree_size"] == old_size
        assert verification["new_tree_size"] == new_size
        assert len(str(verification["verification_sha256"])) == 64


def test_split_view_with_rewritten_prefix_is_rejected() -> None:
    signer = _Signer()
    original = [_entry(f"original-{index}") for index in range(5)]
    split_view = [*original]
    split_view[1] = _entry("rewritten-history")
    split_view.extend((_entry("new-5"), _entry("new-6")))
    old = _checkpoint(original, signer, checkpoint_id="old", epoch=100)
    new = _checkpoint(split_view, signer, checkpoint_id="new", epoch=200)
    proof = create_snapshot_transparency_consistency_proof(split_view, old_tree_size=5)

    with pytest.raises(SnapshotTransparencyError, match="does not link"):
        verify_snapshot_transparency_consistency(
            old,
            new,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id="old",
            expected_new_checkpoint_id="new",
        )


def test_consistency_rejects_extra_missing_and_mutated_nodes() -> None:
    signer = _Signer()
    entries = [_entry(f"entry-{index}") for index in range(7)]
    old = _checkpoint(entries[:3], signer, checkpoint_id="old", epoch=100)
    new = _checkpoint(entries, signer, checkpoint_id="new", epoch=200)
    proof = create_snapshot_transparency_consistency_proof(entries, old_tree_size=3)
    changed = ("a" * 64, *proof.audit_path[1:])

    for path in ((*proof.audit_path, "b" * 64), proof.audit_path[:-1], changed):
        candidate = replace(proof, audit_path=path)
        with pytest.raises(SnapshotTransparencyError, match="extra nodes|does not link"):
            verify_snapshot_transparency_consistency(
                old,
                new,
                candidate,
                trusted_public_keys={signer.key_id: signer.public_key_der},
                expected_log_id=_LOG_ID,
                expected_old_checkpoint_id="old",
                expected_new_checkpoint_id="new",
            )


def test_consistency_supports_authenticated_key_rotation() -> None:
    old_signer = _Signer()
    new_signer = _Signer()
    new_signer.key_id = "rotated-log-key"
    entries = [_entry(f"entry-{index}") for index in range(6)]
    old = _checkpoint(entries[:4], old_signer, checkpoint_id="old", epoch=100)
    new = _checkpoint(entries, new_signer, checkpoint_id="new", epoch=200)
    proof = create_snapshot_transparency_consistency_proof(entries, old_tree_size=4)

    verification = verify_snapshot_transparency_consistency(
        old,
        new,
        proof,
        trusted_public_keys={
            old_signer.key_id: old_signer.public_key_der,
            new_signer.key_id: new_signer.public_key_der,
        },
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id="old",
        expected_new_checkpoint_id="new",
    )

    assert verification["verified"] is True
    assert old.public_key_sha256 != new.public_key_sha256


def test_consistency_rejects_size_epoch_and_governance_mismatch() -> None:
    signer = _Signer()
    entries = [_entry(f"entry-{index}") for index in range(5)]
    old = _checkpoint(entries[:3], signer, checkpoint_id="old", epoch=200)
    new = _checkpoint(entries, signer, checkpoint_id="new", epoch=100)
    proof = create_snapshot_transparency_consistency_proof(entries, old_tree_size=3)

    with pytest.raises(SnapshotTransparencyError, match="do not match"):
        verify_snapshot_transparency_consistency(
            old,
            new,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_old_checkpoint_id="old",
            expected_new_checkpoint_id="new",
        )
    with pytest.raises(SnapshotTransparencyError, match="governance"):
        verify_snapshot_transparency_consistency(
            old,
            replace(new, issued_at_epoch=300),
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id="other-log",
            expected_old_checkpoint_id="old",
            expected_new_checkpoint_id="new",
        )


def test_combined_promotion_receipt_binds_inclusion_and_append_only_growth(tmp_path) -> None:
    signer, receipt, entries, new_checkpoint, inclusion_proof = _fixture(tmp_path)
    old_checkpoint = _checkpoint(
        entries[:3],
        signer,
        checkpoint_id="checkpoint-before-promotion",
        epoch=_CHECKPOINT_EPOCH - 30,
    )
    consistency_proof = create_snapshot_transparency_consistency_proof(
        entries,
        old_tree_size=3,
    )

    evidence = verify_snapshot_transparency_append_only_inclusion(
        receipt,
        old_checkpoint,
        new_checkpoint,
        consistency_proof,
        inclusion_proof,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id="checkpoint-before-promotion",
        expected_new_checkpoint_id=_CHECKPOINT_ID,
    )

    assert evidence["schema_version"] == SNAPSHOT_APPEND_ONLY_INCLUSION_SCHEMA
    assert evidence["verified"] is True
    assert evidence["entry_sha256"] == receipt.entry_sha256
    assert evidence["old_tree_size"] == 3
    assert evidence["new_tree_size"] == 5
    assert _LOG_ID not in str(evidence)
    assert _CHECKPOINT_ID not in str(evidence)
    assert "private-revision" not in str(evidence)


def test_consistency_proof_is_canonical_bounded_and_strict() -> None:
    entries = [_entry(f"entry-{index}") for index in range(7)]
    proof = create_snapshot_transparency_consistency_proof(entries, old_tree_size=3)
    payload = dump_snapshot_transparency_consistency_proof(proof)

    assert proof.schema_version == SNAPSHOT_CONSISTENCY_PROOF_SCHEMA
    assert load_snapshot_transparency_consistency_proof(payload) == proof
    assert hashlib.sha256(payload).hexdigest()

    with pytest.raises(SnapshotTransparencyError, match="not canonical"):
        load_snapshot_transparency_consistency_proof(payload + b"\n")
    document = json.loads(payload)
    document["unreviewed"] = True
    extended = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
    with pytest.raises(SnapshotTransparencyError, match="fields"):
        load_snapshot_transparency_consistency_proof(extended)
    with pytest.raises(SnapshotTransparencyError, match="duplicate keys"):
        load_snapshot_transparency_consistency_proof(
            b'{"schema_version":"first","schema_version":"second"}'
        )


def test_consistency_builder_rejects_non_strict_or_duplicate_inputs() -> None:
    entries = [_entry("one"), _entry("two")]
    for old_size in (False, 0, 2, 3):
        with pytest.raises(SnapshotTransparencyError, match="old tree size"):
            create_snapshot_transparency_consistency_proof(
                entries,
                old_tree_size=old_size,
            )
    with pytest.raises(SnapshotTransparencyError, match="duplicates"):
        create_snapshot_transparency_consistency_proof(
            [entries[0], entries[0]],
            old_tree_size=1,
        )
    with pytest.raises(SnapshotTransparencyError, match="metadata"):
        dump_snapshot_transparency_consistency_proof(
            SnapshotTransparencyConsistencyProof(
                schema_version=SNAPSHOT_CONSISTENCY_PROOF_SCHEMA,
                old_tree_size=1,
                new_tree_size=2,
                audit_path=(),
            )
        )
