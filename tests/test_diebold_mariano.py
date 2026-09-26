from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from finforecast.diebold_mariano import (
    AuditInputError,
    DieboldMarianoPolicy,
    audit_artifact,
    load_artifact,
    main,
)

NOW = datetime(2026, 9, 26, 3, 30, tzinfo=UTC)


def _observations(*, candidate_shift: float = 0.20, count: int = 80) -> list[dict]:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        baseline = 1.0 + ((index % 9) - 4) * 0.025
        variable_effect = ((index * 7) % 13 - 6) * 0.006
        candidate = baseline - candidate_shift - variable_effect
        rows.append(
            {
                "timestamp": (start + timedelta(days=index)).isoformat(),
                "baseline_loss": baseline,
                "candidate_loss": candidate,
            }
        )
    return rows


def _artifact(*, candidate_shift: float = 0.20) -> dict:
    return {
        "schema": "finforecast-diebold-mariano/v1",
        "generated_at": "2026-09-26T03:29:00Z",
        "benchmark_id": "walk-forward-v3",
        "baseline_model_sha256": "a" * 64,
        "candidate_model_sha256": "b" * 64,
        "loss_name": "absolute_error",
        "loss_config_sha256": "c" * 64,
        "cases": [
            {
                "case_id": "asset-private-1d",
                "forecast_horizon": 1,
                "hac_lag": 0,
                "observations": _observations(candidate_shift=candidate_shift),
            }
        ],
    }


def _audit(payload: dict, policy: DieboldMarianoPolicy | None = None) -> dict:
    return audit_artifact(payload, policy=policy, now=NOW)


def test_material_statistically_significant_improvement_is_accepted():
    report = _audit(_artifact())
    case = report["cases"][0]
    assert report["accepted"] is True
    assert case["metrics"]["relative_loss_improvement"] > 0.19
    assert case["metrics"]["one_sided_p_value"] < 0.05
    assert case["metrics"]["harvey_adjusted_statistic"] > 0
    assert report["summary"]["total_observations"] == 80


def test_worse_candidate_is_rejected_with_stable_reasons():
    report = _audit(_artifact(candidate_shift=-0.05))
    assert report["accepted"] is False
    assert report["cases"][0]["reasons"] == [
        "candidate_mean_loss_not_better",
        "relative_loss_improvement_below_threshold",
        "dm_significance_not_reached",
    ]


def test_economic_threshold_can_reject_statistically_clear_small_gain():
    report = _audit(
        _artifact(candidate_shift=0.02),
        DieboldMarianoPolicy(min_relative_loss_improvement=0.03),
    )
    assert report["accepted"] is False
    assert report["cases"][0]["reasons"] == ["relative_loss_improvement_below_threshold"]


def test_horizon_aware_lag_and_harvey_correction_are_reported():
    payload = _artifact()
    payload["cases"][0]["forecast_horizon"] = 5
    payload["cases"][0]["hac_lag"] = 4
    report = _audit(payload)
    case = report["cases"][0]
    assert case["hac_lag"] == 4
    assert case["metrics"]["harvey_correction_factor"] < 1.0


def test_positive_autocorrelation_increases_hac_uncertainty():
    payload = _artifact()
    rows = payload["cases"][0]["observations"]
    for index, row in enumerate(rows):
        block_effect = 0.28 if (index // 8) % 2 == 0 else 0.12
        row["candidate_loss"] = row["baseline_loss"] - block_effect
    lag_zero = _audit(payload)["cases"][0]["metrics"]
    payload["cases"][0]["forecast_horizon"] = 5
    payload["cases"][0]["hac_lag"] = 4
    lag_four = _audit(payload)["cases"][0]["metrics"]
    assert lag_four["standard_error"] > lag_zero["standard_error"]
    assert lag_four["dm_statistic"] < lag_zero["dm_statistic"]


def test_governed_failed_case_fraction_can_admit_one_failure():
    payload = _artifact()
    failed = deepcopy(payload["cases"][0])
    failed["case_id"] = "asset-private-5d"
    failed["observations"] = _observations(candidate_shift=-0.05)
    payload["cases"].append(failed)
    report = _audit(payload, DieboldMarianoPolicy(max_failed_case_fraction=0.5))
    assert report["accepted"] is True
    assert report["summary"]["failed_case_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(baseline_model_sha256="bad"), "baseline_model_sha256"),
        (
            lambda item: item.update(candidate_model_sha256=item["baseline_model_sha256"]),
            "must differ",
        ),
        (lambda item: item.update(generated_at="2026-09-26T03:29:00"), "timezone"),
        (lambda item: item["cases"].clear(), "cases"),
        (
            lambda item: item["cases"][0].update(hac_lag=-1),
            "hac_lag",
        ),
        (
            lambda item: item["cases"][0]["observations"][1].update(
                timestamp=item["cases"][0]["observations"][0]["timestamp"]
            ),
            "strictly increasing",
        ),
        (
            lambda item: item["cases"][0]["observations"][0].update(candidate_loss=-1.0),
            "candidate_loss",
        ),
        (
            lambda item: item["cases"][0]["observations"][0].update(baseline_loss=float("nan")),
            "canonical JSON",
        ),
    ],
)
def test_malformed_artifacts_fail_closed(mutation, match):
    payload = _artifact()
    mutation(payload)
    with pytest.raises(AuditInputError, match=match):
        _audit(payload)


def test_hac_lag_must_cover_overlapping_forecast_horizon():
    payload = _artifact()
    payload["cases"][0]["forecast_horizon"] = 5
    payload["cases"][0]["hac_lag"] = 3
    with pytest.raises(AuditInputError, match="forecast_horizon - 1"):
        _audit(payload)


def test_horizon_and_lag_must_be_smaller_than_sample():
    payload = _artifact()
    payload["cases"][0]["forecast_horizon"] = 80
    payload["cases"][0]["hac_lag"] = 79
    with pytest.raises(AuditInputError, match="forecast_horizon must be smaller"):
        _audit(payload, DieboldMarianoPolicy(max_hac_lag=100))


def test_constant_loss_differential_rejects_undefined_test():
    payload = _artifact()
    for row in payload["cases"][0]["observations"]:
        row["candidate_loss"] = row["baseline_loss"] - 0.2
    with pytest.raises(AuditInputError, match="DM test is undefined"):
        _audit(payload)


def test_duplicate_case_identity_is_rejected():
    payload = _artifact()
    payload["cases"].append(deepcopy(payload["cases"][0]))
    with pytest.raises(AuditInputError, match="case_id values must be unique"):
        _audit(payload)


def test_stale_and_future_artifacts_are_rejected():
    stale = _artifact()
    stale["generated_at"] = "2026-09-18T03:29:00Z"
    with pytest.raises(AuditInputError, match="stale"):
        _audit(stale)
    future = _artifact()
    future["generated_at"] = "2026-09-26T03:36:00Z"
    with pytest.raises(AuditInputError, match="future"):
        _audit(future)


def test_report_is_deterministic_and_omits_raw_case_identity():
    left = _audit(_artifact())
    right = _audit(deepcopy(_artifact()))
    assert left == right
    encoded = json.dumps(left)
    assert "asset-private-1d" not in encoded
    assert "walk-forward-v3" not in encoded
    assert len(left["evidence_sha256"]) == 64


def test_load_rejects_duplicate_keys_nonfinite_and_oversize(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(AuditInputError, match="duplicate JSON key"):
        load_artifact(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":Infinity}', encoding="utf-8")
    with pytest.raises(AuditInputError, match="non-finite JSON"):
        load_artifact(nonfinite)
    oversize = tmp_path / "oversize.json"
    oversize.write_text("{} ", encoding="utf-8")
    with pytest.raises(AuditInputError, match="byte budget"):
        load_artifact(oversize, max_bytes=2)


def test_cli_accepts_and_atomically_writes_report(tmp_path, monkeypatch, capsys):
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "nested" / "report.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    monkeypatch.setattr(
        "finforecast.diebold_mariano.datetime",
        type("FixedDatetime", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}),
    )
    assert main([str(artifact_path), "--output", str(output_path)]) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True
    assert json.loads(capsys.readouterr().out)["accepted"] is True
    assert list(output_path.parent.glob(f".{output_path.name}.*")) == []


def test_cli_distinguishes_rejection_from_malformed_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "finforecast.diebold_mariano.datetime",
        type("FixedDatetime", (datetime,), {"now": classmethod(lambda cls, tz=None: NOW)}),
    )
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(_artifact(candidate_shift=-0.05)), encoding="utf-8")
    assert main([str(rejected)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "forecast_comparison_policy_rejected"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert main([str(malformed)]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "malformed_artifact"


@pytest.mark.parametrize(
    "policy",
    [
        DieboldMarianoPolicy(min_observations=True),
        DieboldMarianoPolicy(max_observations_per_case="many"),  # type: ignore[arg-type]
        DieboldMarianoPolicy(significance_level=float("nan")),
        DieboldMarianoPolicy(max_failed_case_fraction=1.1),
        DieboldMarianoPolicy(variance_epsilon=0.0),
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(AuditInputError):
        _audit(_artifact(), policy)
