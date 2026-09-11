from __future__ import annotations

import pytest

from finforecast.splits import WalkForwardConfig, WalkForwardSplitter


def test_expanding_split_respects_gap_and_boundaries():
    folds = WalkForwardSplitter(
        WalkForwardConfig(min_train_size=10, test_size=4, gap=2)
    ).materialize(25)
    assert folds[0].as_dict() == {
        "fold": 0,
        "train_start": 0,
        "train_end": 10,
        "test_start": 12,
        "test_end": 16,
        "train_size": 10,
        "test_size": 4,
    }
    assert all(fold.train_end + 2 == fold.test_start for fold in folds)


def test_rolling_split_caps_training_history():
    folds = WalkForwardSplitter(
        WalkForwardConfig(min_train_size=10, max_train_size=12, test_size=5)
    ).materialize(30)
    assert folds[-1].train_size == 12
    assert folds[-1].train_start > 0


@pytest.mark.parametrize(
    "config",
    [
        WalkForwardConfig(min_train_size=2, test_size=1),
        WalkForwardConfig(min_train_size=10, test_size=2, gap=3),
    ],
)
def test_insufficient_history_is_rejected(config: WalkForwardConfig):
    with pytest.raises(ValueError, match="not enough"):
        tuple(WalkForwardSplitter(config).split(config.min_train_size + config.gap))


def test_invalid_split_configuration_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardConfig(min_train_size=1)
    with pytest.raises(ValueError):
        WalkForwardConfig(max_train_size=20, min_train_size=30)
