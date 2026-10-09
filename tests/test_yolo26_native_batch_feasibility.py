from pathlib import Path

from research_tools.execution_policy.run_yolo26_native_batch_feasibility import (
    candidate_name,
    pick_history_candidate,
    tensor_proxy,
)


def test_candidate_identity_and_proxy() -> None:
    assert candidate_name(320, 192, 0.25) == "b320_c192_p25"
    assert tensor_proxy(base_imgsz=320, crop_imgsz=320, foci_count=2) == 0.5625


def test_history_selection_requires_all_history_pass() -> None:
    rows = [
        {
            "name": "bad",
            "all_history_pass": False,
            "history_mae_mean": 1.0,
            "history_precision_mean": 1.0,
            "history_time_ms_mean": 1.0,
        },
        {
            "name": "slow",
            "all_history_pass": True,
            "history_mae_mean": 2.0,
            "history_precision_mean": 0.8,
            "history_time_ms_mean": 20.0,
        },
        {
            "name": "fast",
            "all_history_pass": True,
            "history_mae_mean": 2.0,
            "history_precision_mean": 0.8,
            "history_time_ms_mean": 10.0,
        },
    ]
    assert pick_history_candidate(rows)["name"] == "fast"
    assert pick_history_candidate([{**rows[0]}]) is None


def test_launcher_is_development_only_and_pins_gpu_one() -> None:
    text = Path("research_tools/remote/launch_yolo26_native_batch_feasibility.sh").read_text(encoding="utf-8")
    assert "run_yolo26_native_batch_feasibility" in text
    assert "DEVELOPMENT_ONLY" in text
    assert "official_h1" in text
    assert "GPU=${GPU:-1}" in text
    assert "candidate_count\"] == 36" in text
    assert "input-hashes.txt" in text
    assert "SHA256SUMS" in text
    assert ".env" not in text
