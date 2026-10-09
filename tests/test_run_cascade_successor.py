import numpy as np

from evidencegate.cascade import BenefitRouter
from research_tools.cascade_experiment import Box, CallableDetectorAdapter, GroundTruth, Prediction
from research_tools.run_cascade_successor import _frame_rois, _select_threshold, calibration_phase, evaluate_phase, fit_phase, screen_phase


def _detector(boxes, name):
    return CallableDetectorAdapter(lambda frame, frame_index: tuple(Prediction(frame_index, Box(*box), score, name) for box, score in boxes), name)


def test_frame_rois_use_numpy_dimensions():
    rois = _frame_rois(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(rois) > 1
    assert max(r.x2 for r in rois) == 200


def test_fit_uses_roi_level_ground_truth_benefit(tmp_path):
    frames = {1: np.zeros((100, 100, 3), dtype=np.uint8)}
    truth = (GroundTruth(1, 1, Box(10, 10, 30, 50)),)
    small = _detector([], "small")
    local = _detector([((10, 10, 30, 50), 0.9)], "local")
    result = fit_phase(frames, truth, small, local, router_path=tmp_path / "router.json")
    assert result["roi_samples"] == len(_frame_rois(frames[1]))
    assert any(row["benefit"] > 0 for row in result["roi_receipts"])


def test_calibration_measures_before_and_after_and_writes_distinct_router(tmp_path):
    frames = {1: np.zeros((100, 100, 3), dtype=np.uint8)}
    truth = (GroundTruth(1, 1, Box(10, 10, 30, 50)),)
    small = _detector([], "small"); local = _detector([((10, 10, 30, 50), .9)], "local")
    fit_phase(frames, truth, small, local, router_path=tmp_path / "router.json")
    result = calibration_phase(frames, truth, small, local, router_path=tmp_path / "router.json", calibrated_router_path=tmp_path / "calibrated.json")
    assert (tmp_path / "calibrated.json").exists()
    assert set(result["before"]) == {"brier", "ece"}
    assert set(result["after"]) == {"brier", "ece"}


def test_evaluation_separates_heuristic_and_benefit_routes_and_keeps_boxes():
    frames = {1: np.zeros((100, 100, 3), dtype=np.uint8)}
    truth = (GroundTruth(1, 1, Box(10, 10, 30, 50)),)
    small = _detector([], "small"); local = _detector([((10, 10, 30, 50), .9)], "local")
    router = BenefitRouter(); router.weights = np.array([-40.0, 0, 0, 0, 0])
    result = evaluate_phase(frames, truth, small, local, router=router, max_rois=1)
    assert result["receipts"]["heuristic_nms"][0]["selected_roi_ids"]
    assert result["receipts"]["benefit_nms"][0]["selected_roi_ids"] == []
    assert "predictions" in result["receipts"]["yolo11s_global"][0]


def test_evaluation_accepts_empty_frames():
    detector = _detector([], "empty")
    router = BenefitRouter(); router.weights = np.zeros(5)
    result = evaluate_phase({}, (), detector, detector, router=router)
    assert result["frames"] == 0
    assert result["conditions"]["benefit_wbf"]["latency_ms"] == 0.0


def test_threshold_is_selected_on_calibration_probabilities():
    assert _select_threshold([.05, .1, .2, .8], [0, 1, 1, 0]) == .1


def test_screen_phase_compares_fixed_budget_random_and_gt_oracle_routes():
    frames = {1: np.zeros((100, 100, 3), dtype=np.uint8)}
    truth = (GroundTruth(1, 1, Box(10, 10, 30, 50)),)
    small = _detector([], "small")
    large = _detector([((10, 10, 30, 50), .9)], "large")
    result = screen_phase(frames, truth, small, large, large, max_rois=1, seed=7)
    assert set(result["conditions"]) == {
        "yolo11s_global", "yolo11m_global", "heuristic_nms", "random_nms", "oracle_nms"
    }
    assert result["receipts"]["random_nms"][0]["selected_roi_ids"]
    assert result["receipts"]["oracle_nms"][0]["selected_roi_ids"]
    assert result["conditions"]["oracle_nms"]["model_input_pixels"] == 640**2 + 512**2
    assert result["conditions"]["oracle_nms"]["model_input_tensor_ratio_to_small_global"] == (640**2 + 512**2) / 640**2
