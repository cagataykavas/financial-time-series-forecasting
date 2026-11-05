from __future__ import annotations

import json
from dataclasses import replace

import pytest

from finforecast.snapshot_gossip import (
    MAX_GOSSIP_OBSERVATION_AGE_SECONDS,
    SNAPSHOT_GOSSIP_OBSERVATION_SCHEMA,
    SNAPSHOT_GOSSIP_VERIFICATION_SCHEMA,
    SNAPSHOT_GOSSIPED_PROMOTION_SCHEMA,
    create_snapshot_checkpoint_gossip_observation,
    dump_snapshot_checkpoint_gossip_observation,
    load_snapshot_checkpoint_gossip_observation,
    verify_gossiped_snapshot_transparency_promotion,
    verify_snapshot_checkpoint_gossip,
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
from tests.test_snapshot_witness import _bundle, _trusted_witness_keys

_OBSERVED_AT = _CHECKPOINT_EPOCH + 10
_VERIFIED_AT = _OBSERVED_AT + 20


class _RelaySigner(_Signer):
    def __init__(self, number: int) -> None:
        super().__init__()
        self.key_id = f"gossip-relay-key-{number}"


class _FailingRelay(_RelaySigner):
    def sign(self, message: bytes) -> bytes:
        raise RuntimeError("private relay provider diagnostic")


def _relays(count: int = 3) -> dict[str, _RelaySigner]:
    return {f"relay-{number}": _RelaySigner(number) for number in range(1, count + 1)}


def _trusted_relay_keys(relays: dict[str, _RelaySigner]) -> dict[str, bytes]:
    return {relay.key_id: relay.public_key_der for relay in relays.values()}


def _observation(checkpoint, relay_id, relay, log_signer, *, observed_at=_OBSERVED_AT):
    return create_snapshot_checkpoint_gossip_observation(
        checkpoint,
        relay_id,
        relay,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        observed_at_epoch=observed_at,
    )


def _views(tmp_path, *, count: int = 3):
    log_signer, _receipt, _entries, checkpoint, _proof = _fixture(tmp_path)
    relays = _relays(count)
    views = [
        (checkpoint, _observation(checkpoint, relay_id, relay, log_signer))
        for relay_id, relay in relays.items()
    ]
    return log_signer, checkpoint, relays, views


def test_gossip_quorum_is_deterministic_cross_bound_and_privacy_reduced(tmp_path) -> None:
    log_signer, checkpoint, relays, views = _views(tmp_path)

    verification = verify_snapshot_checkpoint_gossip(
        views,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_relay_public_keys=_trusted_relay_keys(relays),
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        verification_epoch=_VERIFIED_AT,
    )
    repeated = verify_snapshot_checkpoint_gossip(
        views,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_relay_public_keys=_trusted_relay_keys(relays),
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        verification_epoch=_VERIFIED_AT,
    )

    assert views[0][1].schema_version == SNAPSHOT_GOSSIP_OBSERVATION_SCHEMA
    assert verification == repeated
    assert verification["schema_version"] == SNAPSHOT_GOSSIP_VERIFICATION_SCHEMA
    assert verification["verified"] is True
    assert verification["valid_relays"] == 3
    assert verification["checkpoint_sha256"] == checkpoint.checkpoint_sha256
    assert _LOG_ID not in str(verification)
    assert _CHECKPOINT_ID not in str(verification)
    assert "relay-1" not in str(verification)
    assert relays["relay-1"].key_id not in str(verification)


def test_gossip_is_cross_bound_into_the_witnessed_promotion_receipt(tmp_path) -> None:
    transition, witnesses, witness_bundle = _bundle(tmp_path)
    log_signer, receipt, _entries, old, new, proof, inclusion = transition
    relays = _relays(2)
    views = [
        (new, _observation(new, relay_id, relay, log_signer)) for relay_id, relay in relays.items()
    ]

    evidence = verify_gossiped_snapshot_transparency_promotion(
        receipt,
        old,
        new,
        proof,
        inclusion,
        witness_bundle,
        views,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_witness_public_keys=_trusted_witness_keys(witnesses),
        trusted_relay_public_keys=_trusted_relay_keys(relays),
        expected_log_id=_LOG_ID,
        expected_old_checkpoint_id="checkpoint-before-training",
        expected_new_checkpoint_id=_CHECKPOINT_ID,
        verification_epoch=_VERIFIED_AT,
    )

    assert evidence["schema_version"] == SNAPSHOT_GOSSIPED_PROMOTION_SCHEMA
    assert evidence["verified"] is True
    assert evidence["new_checkpoint_sha256"] == new.checkpoint_sha256
    assert evidence["valid_witnesses"] == 3
    assert evidence["valid_relays"] == 2
    assert _LOG_ID not in str(evidence)


def test_any_trusted_authenticated_split_view_fails_even_when_one_view_has_quorum(tmp_path) -> None:
    log_signer, _checkpoint, relays, views = _views(tmp_path)
    alternate_entries = [_entry(f"alternate-entry-{position}") for position in range(5)]
    alternate = create_snapshot_transparency_checkpoint(
        alternate_entries,
        log_id=_LOG_ID,
        checkpoint_id=_CHECKPOINT_ID,
        issued_at_epoch=_CHECKPOINT_EPOCH,
        signer=log_signer,
    )
    views[2] = (
        alternate,
        _observation(alternate, "relay-3", relays["relay-3"], log_signer),
    )

    with pytest.raises(SnapshotTransparencyError, match="conflicting authenticated checkpoints"):
        verify_snapshot_checkpoint_gossip(
            views,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=_trusted_relay_keys(relays),
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            verification_epoch=_VERIFIED_AT,
            required_relays=2,
        )


def test_invalid_or_unavailable_relays_only_pass_while_quorum_remains(tmp_path) -> None:
    log_signer, _checkpoint, relays, views = _views(tmp_path)
    trusted = _trusted_relay_keys(relays)
    signature = views[2][1].signature
    tampered = replace(
        views[2][1],
        signature=("A" if signature[0] != "A" else "B") + signature[1:],
    )
    invalid = verify_snapshot_checkpoint_gossip(
        [*views[:2], (views[2][0], tampered)],
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_relay_public_keys=trusted,
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        verification_epoch=_VERIFIED_AT,
    )
    assert invalid["valid_relays"] == 2

    trusted.pop(relays["relay-3"].key_id)

    accepted = verify_snapshot_checkpoint_gossip(
        views,
        trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
        trusted_relay_public_keys=trusted,
        expected_log_id=_LOG_ID,
        expected_checkpoint_id=_CHECKPOINT_ID,
        verification_epoch=_VERIFIED_AT,
    )
    assert accepted["valid_relays"] == 2

    trusted.pop(relays["relay-2"].key_id)
    with pytest.raises(SnapshotTransparencyError, match="quorum is not satisfied"):
        verify_snapshot_checkpoint_gossip(
            views,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=trusted,
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            verification_epoch=_VERIFIED_AT,
        )


@pytest.mark.parametrize(
    ("observed_at", "verified_at", "message"),
    [
        (_OBSERVED_AT, _OBSERVED_AT + MAX_GOSSIP_OBSERVATION_AGE_SECONDS + 1, "is stale"),
        (_OBSERVED_AT + 1, _OBSERVED_AT, "from the future"),
    ],
)
def test_gossip_observations_must_be_fresh(
    tmp_path,
    observed_at: int,
    verified_at: int,
    message: str,
) -> None:
    log_signer, _receipt, _entries, checkpoint, _proof = _fixture(tmp_path)
    relays = _relays(2)
    views = [
        (
            checkpoint,
            _observation(
                checkpoint,
                relay_id,
                relay,
                log_signer,
                observed_at=observed_at,
            ),
        )
        for relay_id, relay in relays.items()
    ]

    with pytest.raises(SnapshotTransparencyError, match=message):
        verify_snapshot_checkpoint_gossip(
            views,
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=_trusted_relay_keys(relays),
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            verification_epoch=verified_at,
        )


def test_gossip_rejects_checkpoint_substitution_and_reused_relay_keys(tmp_path) -> None:
    log_signer, checkpoint, relays, views = _views(tmp_path, count=2)
    substituted = replace(views[0][1], root_sha256="f" * 64)

    with pytest.raises(SnapshotTransparencyError, match="does not match its checkpoint"):
        verify_snapshot_checkpoint_gossip(
            [(checkpoint, substituted), views[1]],
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=_trusted_relay_keys(relays),
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            verification_epoch=_VERIFIED_AT,
        )

    reused = _observation(checkpoint, "relay-alias", relays["relay-1"], log_signer)
    with pytest.raises(SnapshotTransparencyError, match="key identifiers must be distinct"):
        verify_snapshot_checkpoint_gossip(
            [views[0], (checkpoint, reused)],
            trusted_log_public_keys={log_signer.key_id: log_signer.public_key_der},
            trusted_relay_public_keys=_trusted_relay_keys(relays),
            expected_log_id=_LOG_ID,
            expected_checkpoint_id=_CHECKPOINT_ID,
            verification_epoch=_VERIFIED_AT,
        )


def test_gossip_creation_sanitizes_provider_errors_and_rejects_predating(tmp_path) -> None:
    log_signer, _receipt, _entries, checkpoint, _proof = _fixture(tmp_path)

    with pytest.raises(SnapshotTransparencyError, match="predates"):
        _observation(
            checkpoint,
            "relay-1",
            _RelaySigner(1),
            log_signer,
            observed_at=_CHECKPOINT_EPOCH - 1,
        )
    with pytest.raises(SnapshotTransparencyError, match="relay signing failed") as error:
        _observation(checkpoint, "relay-2", _FailingRelay(2), log_signer)
    assert "private relay provider diagnostic" not in str(error.value)


def test_gossip_observation_loader_is_canonical_bounded_and_strict(tmp_path) -> None:
    _log_signer, _checkpoint, _relays_data, views = _views(tmp_path, count=2)
    observation = views[0][1]
    payload = dump_snapshot_checkpoint_gossip_observation(observation)

    assert load_snapshot_checkpoint_gossip_observation(payload) == observation
    with pytest.raises(SnapshotTransparencyError, match="not canonical"):
        load_snapshot_checkpoint_gossip_observation(payload + b"\n")
    document = json.loads(payload)
    document["unreviewed"] = True
    with pytest.raises(SnapshotTransparencyError, match="fields"):
        load_snapshot_checkpoint_gossip_observation(
            json.dumps(document, separators=(",", ":"), sort_keys=True).encode("ascii")
        )
    with pytest.raises(SnapshotTransparencyError, match="duplicate keys"):
        load_snapshot_checkpoint_gossip_observation(
            b'{"schema_version":"first","schema_version":"second"}'
        )
    with pytest.raises(SnapshotTransparencyError, match="exceeds"):
        load_snapshot_checkpoint_gossip_observation(b"x" * 4_097)
