from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.sceneconfig_top6_exact_replay import (
    choose_calibrated_candidate,
    evaluate_gate,
    select_top_k,
    temporal_calibration_frames,
)


def test_temporal_calibration_is_10_of_32_without_overlap() -> None:
    selected = temporal_calibration_frames(range(1, 33), 10)
    assert selected == (2, 5, 9, 12, 15, 18, 21, 25, 28, 31)
    assert len(set(selected)) == 10


def test_top6_is_proxy_only_and_deterministic() -> None:
    panel = [{"config_id": f"C{i:02d}", "tiles": []} for i in range(8)]
    proxy = {
        "holdout_opened": False,
        "evaluation_gt_used_for_proxy": False,
        "panel": panel,
        "per_camera": {camera: [{"config_id": f"C{i:02d}", "proxy_utility": float(i)} for i in range(8)] for camera in EXPECTED_CAMERAS},
    }
    assert [row["config_id"] for row in select_top_k(proxy)[EXPECTED_CAMERAS[0]]] == ["C07", "C06", "C05", "C04", "C03", "C02"]


def test_calibrated_selection_falls_back_when_precision_is_unsafe() -> None:
    candidates = [
        {"config_id": "C01", "oom": False, "latency_ratio_p95": 2.0, "exact_proxy_utility": 1.0, "calibration_metrics": {"precision": .70, "count_mae": 1.0}},
        {"config_id": "C02", "oom": False, "latency_ratio_p95": 2.0, "exact_proxy_utility": 2.0, "calibration_metrics": {"precision": .74, "count_mae": .5}},
    ]
    chosen, diagnostics = choose_calibrated_candidate(candidates, {"precision": .80})
    assert chosen == "B0"
    assert not any(row["admissible"] for row in diagnostics)


def test_calibrated_selection_minimizes_mae_inside_safety_gate() -> None:
    candidates = [
        {"config_id": "C01", "oom": False, "latency_ratio_p95": 2.0, "exact_proxy_utility": 1.0, "calibration_metrics": {"precision": .78, "count_mae": 1.0}},
        {"config_id": "C02", "oom": False, "latency_ratio_p95": 2.5, "exact_proxy_utility": .5, "calibration_metrics": {"precision": .76, "count_mae": .5}},
    ]
    chosen, _ = choose_calibrated_candidate(candidates, {"precision": .80})
    assert chosen == "C02"


def test_gate_cannot_drop_a_failed_camera() -> None:
    rows = {
        camera: {"c10": {"latency_ratio_p95": 2.0, "relative_count_mae_improvement": .2, "precision_change": -.01}}
        for camera in EXPECTED_CAMERAS[:-1]
    }
    gate = evaluate_gate(rows, deployment_oom=False)
    assert gate["all_four_cameras_retained"] is False
    assert gate["passed"] is False
