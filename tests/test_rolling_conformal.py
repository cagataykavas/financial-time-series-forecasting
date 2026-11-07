from __future__ import annotations

import copy

import pytest

from finforecast.rolling_conformal import (
    DelayedForecast,
    RollingConformalPolicy,
    rolling_delayed_conformal,
)


def _row(
    issued_at: int,
    *,
    residual: float = 1.0,
    label_available_at: int | None = None,
) -> DelayedForecast:
    return DelayedForecast(
        issued_at=issued_at,
        label_available_at=label_available_at or issued_at + 1,
        actual=residual,
        point_forecast=0.0,
    )


def _policy(**overrides: object) -> RollingConformalPolicy:
    values: dict[str, object] = {
        "target_coverage": 0.8,
        "calibration_window": 4,
        "minimum_calibration_size": 2,
        "refresh_interval": 2,
    }
    values.update(overrides)
    return RollingConformalPolicy(**values)  # type: ignore[arg-type]


def test_delayed_label_is_excluded_until_a_later_scheduled_refresh() -> None:
    history = [_row(1), _row(2)]
    evaluation = [
        _row(10, residual=100.0, label_available_at=13),
        _row(11),
        _row(12),
        _row(14),
    ]

    result = rolling_delayed_conformal(history, evaluation, _policy())

    assert result.lower == (-1.0, -1.0, -1.0, -1.0)
    assert result.upper == (1.0, 1.0, 1.0, 1.0)
    assert [item["calibration_size"] for item in result.report["refreshes"]] == [2, 2]
    assert all(
        item["latest_label_available_at"] < item["refresh_issued_at"]
        for item in result.report["refreshes"]
    )


def test_matured_extreme_residual_enters_the_next_refresh() -> None:
    history = [_row(1), _row(2)]
    evaluation = [
        _row(10, residual=100.0, label_available_at=11),
        _row(11),
        _row(12),
    ]

    result = rolling_delayed_conformal(history, evaluation, _policy())

    assert result.upper == (1.0, 1.0, 100.0)
    assert result.report["refreshes"][1]["radius"] == 100.0


def test_label_available_exactly_at_refresh_is_not_mature() -> None:
    history = [_row(1), _row(2)]
    evaluation = [
        _row(10, residual=100.0, label_available_at=12),
        _row(11),
        _row(12),
    ]

    result = rolling_delayed_conformal(history, evaluation, _policy())

    assert result.upper[-1] == 1.0
    assert result.report["refreshes"][1]["calibration_size"] == 2


def test_radius_remains_fixed_between_predeclared_refreshes() -> None:
    history = [_row(1), _row(2)]
    evaluation = [_row(10, residual=100.0), _row(12), _row(13)]

    result = rolling_delayed_conformal(
        history,
        evaluation,
        _policy(refresh_interval=3),
    )

    assert result.upper == (1.0, 1.0, 1.0)
    assert result.report["refresh_count"] == 1


def test_recent_window_evicts_old_residuals() -> None:
    history = [_row(1, residual=100.0), _row(2), _row(3), _row(4), _row(5)]

    result = rolling_delayed_conformal(
        history,
        [_row(10)],
        _policy(calibration_window=4),
    )

    assert result.upper == (1.0,)
    assert result.report["refreshes"][0]["calibration_first_issued_at"] == 2


def test_report_is_deterministic_self_digested_and_privacy_reduced() -> None:
    history = [_row(1), _row(2)]
    evaluation = [_row(10), _row(11)]

    first = rolling_delayed_conformal(history, evaluation, _policy())
    replay = rolling_delayed_conformal(history, evaluation, _policy())

    assert first.report == replay.report
    assert len(first.report["policy_sha256"]) == 64
    assert len(first.report["input_sha256"]) == 64
    assert len(first.report["report_sha256"]) == 64
    assert "actual" not in first.report
    assert "point_forecast" not in first.report
    changed = copy.deepcopy(first.report)
    changed["observed_coverage"] = 0.0
    assert changed["report_sha256"] == first.report["report_sha256"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_coverage": 1.0}, "strictly between"),
        ({"calibration_window": 0}, "positive"),
        ({"minimum_calibration_size": 1}, "at least two"),
        ({"minimum_calibration_size": 5}, "cannot exceed"),
        ({"refresh_interval": True}, "must be an integer"),
    ],
)
def test_policy_rejects_unsafe_values(overrides: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _policy(**overrides)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("issued_at", True, "must be an integer"),
        ("label_available_at", 0, "must be positive"),
        ("actual", float("nan"), "must be finite"),
        ("point_forecast", float("inf"), "must be finite"),
    ],
)
def test_observation_rejects_malformed_values(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "issued_at": 1,
        "label_available_at": 2,
        "actual": 1.0,
        "point_forecast": 0.0,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        DelayedForecast(**values)  # type: ignore[arg-type]


def test_observation_rejects_label_availability_before_issuance() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        DelayedForecast(issued_at=2, label_available_at=1, actual=1.0, point_forecast=0.0)


def test_run_rejects_non_monotonic_issue_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        rolling_delayed_conformal([_row(2), _row(1)], [_row(10)], _policy())


def test_run_fails_closed_when_scheduled_refresh_has_too_few_matured_labels() -> None:
    history = [_row(1, label_available_at=20), _row(2, label_available_at=20)]

    with pytest.raises(ValueError, match="fewer matured labels"):
        rolling_delayed_conformal(history, [_row(10)], _policy())


def test_run_rejects_empty_evaluation_and_wrong_row_types() -> None:
    with pytest.raises(ValueError, match="at least one"):
        rolling_delayed_conformal([_row(1), _row(2)], [], _policy())
    with pytest.raises(TypeError, match="DelayedForecast"):
        rolling_delayed_conformal([_row(1), object()], [_row(10)], _policy())
