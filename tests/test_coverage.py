import json

import numpy as np
import pytest

from finforecast.coverage import audit_temporal_coverage


def test_temporal_gate_detects_local_failure_hidden_by_aggregate() -> None:
    actual = np.zeros(100)
    lower = np.full(100, -1.0)
    upper = np.full(100, 1.0)
    lower[90:] = 1.0
    upper[90:] = 2.0

    report = audit_temporal_coverage(actual, lower, upper)

    assert report.overall_coverage == pytest.approx(0.9)
    assert report.windows[0].coverage == pytest.approx(1.0)
    assert report.windows[1].coverage == pytest.approx(0.8)
    assert report.failed_windows == 1
    assert report.passed is False
    assert report.reasons == ("temporal_undercoverage",)


def test_aggregate_and_temporal_failures_are_reported_separately() -> None:
    actual = np.zeros(100)
    lower = np.ones(100)
    upper = np.full(100, 2.0)

    report = audit_temporal_coverage(actual, lower, upper)

    assert report.passed is False
    assert report.reasons == ("overall_undercoverage", "temporal_undercoverage")
    assert report.worst_window_coverage == 0.0


def test_short_tail_is_merged_instead_of_becoming_noisy_window() -> None:
    actual = np.zeros(115)
    lower = np.full(115, -1.0)
    upper = np.full(115, 1.0)

    report = audit_temporal_coverage(actual, lower, upper)

    assert [(window.start, window.end) for window in report.windows] == [(0, 50), (50, 115)]
    assert all(window.passed for window in report.windows)
    assert report.passed is True


def test_report_is_json_ready_and_contains_width_evidence() -> None:
    report = audit_temporal_coverage(
        actual=[0.0] * 20,
        lower=[-2.0] * 20,
        upper=[2.0] * 20,
        window_size=20,
    )

    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["passed"] is True
    assert payload["overall_mean_width"] == pytest.approx(4.0)
    assert payload["windows"][0]["observations"] == 20


@pytest.mark.parametrize(
    ("actual", "lower", "upper", "message"),
    [
        ([0.0], [-1.0], [1.0], "at least"),
        ([0.0] * 20, [-1.0] * 19, [1.0] * 20, "equally sized"),
        ([0.0] * 20, [1.0] * 20, [-1.0] * 20, "cannot exceed"),
        ([0.0] * 20, [-1.0] * 20, [float("nan")] * 20, "finite"),
    ],
)
def test_invalid_interval_evidence_fails_closed(
    actual: list[float], lower: list[float], upper: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_temporal_coverage(actual, lower, upper)


def test_policy_configuration_is_validated() -> None:
    values = [0.0] * 20

    with pytest.raises(ValueError, match="target_coverage"):
        audit_temporal_coverage(values, values, values, target_coverage=1.0)
    with pytest.raises(ValueError, match="tolerance"):
        audit_temporal_coverage(values, values, values, tolerance=0.9)
    with pytest.raises(ValueError, match="minimum_window_size"):
        audit_temporal_coverage(values, values, values, minimum_window_size=51)
