from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from finforecast.snapshot_transparency import (
    SNAPSHOT_TRANSPARENCY_CHECKPOINT_SCHEMA,
    SNAPSHOT_TRANSPARENCY_VERIFICATION_SCHEMA,
    SnapshotTransparencyError,
    create_snapshot_transparency_checkpoint,
    create_snapshot_transparency_proof,
    dump_snapshot_transparency_checkpoint,
    dump_snapshot_transparency_proof,
    load_snapshot_transparency_checkpoint,
    load_snapshot_transparency_proof,
    verify_snapshot_transparency_inclusion,
)
from tests.test_snapshot_observation import _OBSERVED_AT, _record, _rows, _Signer

_LOG_ID = "market-data-observation-log"
_CHECKPOINT_ID = "checkpoint-2026-11-03-001"
_CHECKPOINT_EPOCH = _OBSERVED_AT + 60


def _entry(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture(tmp_path: Path, *, leaf_index: int = 2, tree_size: int = 5):
    signer = _Signer()
    receipt = _record(
        tmp_path / "observations.sqlite3",
        _rows(),
        signer,
        observed_at=_OBSERVED_AT,
    )
    entries = [_entry(f"independent-entry-{index}") for index in range(tree_size)]
    entries[leaf_index] = receipt.entry_sha256
    checkpoint = create_snapshot_transparency_checkpoint(
        entries,
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH,
        signer=signer,
    )
    proof = create_snapshot_transparency_proof(entries, leaf_index=leaf_index)
    return signer, receipt, entries, checkpoint, proof


@pytest.mark.parametrize("leaf_index", range(5))
def test_signed_checkpoint_verifies_every_non_power_of_two_leaf(
    tmp_path: Path,
    leaf_index: int,
) -> None:
    signer, receipt, _entries, checkpoint, proof = _fixture(
        tmp_path,
        leaf_index=leaf_index,
    )

    verification = verify_snapshot_transparency_inclusion(
        receipt,
        checkpoint,
        proof,
        trusted_public_keys={signer.key_id: signer.public_key_der},
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
    )

    assert verification["schema_version"] == SNAPSHOT_TRANSPARENCY_VERIFICATION_SCHEMA
    assert verification["verified"] is True
    assert verification["entry_sha256"] == receipt.entry_sha256
    assert verification["leaf_index"] == leaf_index
    assert verification["tree_size"] == 5
    assert len(str(verification["verification_sha256"])) == 64


def test_checkpoint_is_deterministic_canonical_and_privacy_reduced(tmp_path: Path) -> None:
    signer, receipt, entries, checkpoint, proof = _fixture(tmp_path)
    recreated = create_snapshot_transparency_checkpoint(
        entries,
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH,
        signer=signer,
    )
    payload = dump_snapshot_transparency_checkpoint(checkpoint)

    assert checkpoint == recreated
    assert checkpoint.schema_version == SNAPSHOT_TRANSPARENCY_CHECKPOINT_SCHEMA
    assert load_snapshot_transparency_checkpoint(payload) == checkpoint
    assert load_snapshot_transparency_proof(dump_snapshot_transparency_proof(proof)) == proof
    assert receipt.identity_sha256.encode("ascii") not in payload
    assert b"vendor.example" not in payload
    assert b"private-revision" not in payload


def test_inclusion_rejects_wrong_leaf_and_mutated_path(tmp_path: Path) -> None:
    signer, receipt, entries, checkpoint, proof = _fixture(tmp_path)
    other_receipt = _record(
        tmp_path / "other.sqlite3",
        _rows(),
        signer,
        observed_at=_OBSERVED_AT + 1,
    )

    with pytest.raises(SnapshotTransparencyError, match="does not match"):
        verify_snapshot_transparency_inclusion(
            other_receipt,
            checkpoint,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
        )

    changed_path = replace(proof, audit_path=(_entry("replacement"), *proof.audit_path[1:]))
    with pytest.raises(SnapshotTransparencyError, match="does not match"):
        verify_snapshot_transparency_inclusion(
            receipt,
            checkpoint,
            changed_path,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
        )

    assert receipt.entry_sha256 == entries[proof.leaf_index]


def test_inclusion_rejects_extra_or_missing_path_nodes(tmp_path: Path) -> None:
    signer, receipt, _entries, checkpoint, proof = _fixture(tmp_path)

    with pytest.raises(SnapshotTransparencyError, match="leaf index is invalid"):
        create_snapshot_transparency_proof([_entry("only")], leaf_index=True)

    for changed_path in (
        (*proof.audit_path, _entry("extra")),
        proof.audit_path[:-1],
    ):
        with pytest.raises(SnapshotTransparencyError, match="metadata is invalid"):
            verify_snapshot_transparency_inclusion(
                receipt,
                checkpoint,
                replace(proof, audit_path=changed_path),
                trusted_public_keys={signer.key_id: signer.public_key_der},
                expected_log_id=_LOG_ID,
                expected_checkpoint_id=_CHECKPOINT_ID,
            )


def test_signature_and_governed_checkpoint_identity_are_required(tmp_path: Path) -> None:
    signer, receipt, _entries, checkpoint, proof = _fixture(tmp_path)
    tampered = replace(
        checkpoint,
        signature=("A" if checkpoint.signature[0] != "A" else "B") + checkpoint.signature[1:],
    )

    with pytest.raises(SnapshotTransparencyError, match="signature does not verify"):
        verify_snapshot_transparency_inclusion(
            receipt,
            tampered,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
        )
    with pytest.raises(SnapshotTransparencyError, match="governance"):
        verify_snapshot_transparency_inclusion(
            receipt,
            checkpoint,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id="other-log",
            expected_checkpoint_id=_CHECKPOINT_ID,
        )
    with pytest.raises(SnapshotTransparencyError, match="trusted checkpoint key"):
        verify_snapshot_transparency_inclusion(
            receipt,
            checkpoint,
            proof,
            trusted_public_keys={},
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
        )


def test_checkpoint_cannot_predate_local_observation(tmp_path: Path) -> None:
    signer, receipt, entries, _checkpoint, proof = _fixture(tmp_path)
    early = create_snapshot_transparency_checkpoint(
        entries,
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_OBSERVED_AT - 1,
        signer=signer,
    )

    with pytest.raises(SnapshotTransparencyError, match="governance or proof"):
        verify_snapshot_transparency_inclusion(
            receipt,
            early,
            proof,
            trusted_public_keys={signer.key_id: signer.public_key_der},
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
        )


def test_checkpoint_loader_rejects_reformatting_duplicate_keys_and_extension(
    tmp_path: Path,
) -> None:
    _signer, _receipt, _entries, checkpoint, _proof = _fixture(tmp_path)
    payload = dump_snapshot_transparency_checkpoint(checkpoint)

    with pytest.raises(SnapshotTransparencyError, match="not canonical"):
        load_snapshot_transparency_checkpoint(payload + b"\n")
    with pytest.raises(SnapshotTransparencyError, match="duplicate keys"):
        load_snapshot_transparency_checkpoint(
            b'{"schema_version":"first","schema_version":"second"}'
        )
    document = json.loads(payload)
    document["unreviewed"] = True
    extended = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
    with pytest.raises(SnapshotTransparencyError, match="fields"):
        load_snapshot_transparency_checkpoint(extended)


def test_proof_loader_rejects_noncanonical_and_malformed_paths(tmp_path: Path) -> None:
    _signer, _receipt, _entries, _checkpoint, proof = _fixture(tmp_path)
    payload = dump_snapshot_transparency_proof(proof)

    with pytest.raises(SnapshotTransparencyError, match="not canonical"):
        load_snapshot_transparency_proof(payload + b"\n")
    document = json.loads(payload)
    document["audit_path"].append(_entry("extra"))
    malformed = json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
    with pytest.raises(SnapshotTransparencyError, match="metadata is invalid"):
        load_snapshot_transparency_proof(malformed)


def test_tree_builder_rejects_duplicate_entries_and_bad_signer_output(tmp_path: Path) -> None:
    signer, _receipt, entries, _checkpoint, _proof = _fixture(tmp_path)
    with pytest.raises(SnapshotTransparencyError, match="duplicates"):
        create_snapshot_transparency_checkpoint(
            [entries[0], entries[0]],
            log_id=_LOG_ID,
            checkpoint_id=_CHECKPOINT_ID,
            issued_at_epoch=_CHECKPOINT_EPOCH,
            signer=signer,
        )

    bad_signer = copy.copy(signer)
    bad_signer.sign = lambda _message: b"short"  # type: ignore[method-assign]
    with pytest.raises(SnapshotTransparencyError, match="malformed signature"):
        create_snapshot_transparency_checkpoint(
            entries,
            log_id=_LOG_ID,
            checkpoint_id=_CHECKPOINT_ID,
            issued_at_epoch=_CHECKPOINT_EPOCH,
            signer=bad_signer,
        )
