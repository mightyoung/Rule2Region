import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.run_yolo26_depth_spatial_ablation import (
    depth_edge_support,
    depth_gate,
    depth_rerank_semantic,
    detection_gate,
    greedy_counts,
    ultralytics_bgr,
)


def test_depth_support_is_bounded_and_geometry_preserved() -> None:
    depth = np.tile(np.linspace(1, 10, 20, dtype=np.float32), (10, 1))
    result = depth_edge_support(depth, 5, 8)
    assert result.shape == (5, 8)
    assert 0.0 <= result.min() <= result.max() <= 1.0


def test_ultralytics_numpy_input_is_converted_from_rgb_to_bgr() -> None:
    rgb = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    bgr = ultralytics_bgr(rgb)
    np.testing.assert_array_equal(bgr, np.asarray([[[3, 2, 1]]], dtype=np.uint8))
    assert bgr.flags.c_contiguous


def test_depth_cannot_create_candidate_without_semantics() -> None:
    semantic = np.zeros((5, 8), dtype=np.float32)
    depth = np.random.default_rng(1).random((5, 8), dtype=np.float32)
    np.testing.assert_allclose(depth_rerank_semantic(semantic, depth), 0.0)


def test_depth_only_reranks_semantic_support() -> None:
    semantic = np.zeros((3, 3), dtype=np.float32)
    semantic[1, 1] = 1.0
    depth = np.ones((3, 3), dtype=np.float32)
    assert np.unravel_index(depth_rerank_semantic(semantic, depth).argmax(), semantic.shape) == (1, 1)


def test_greedy_counts_matches_one_prediction_once() -> None:
    truth = [Box(0, 0, 10, 10)]
    predictions = [(Box(0, 0, 10, 10), .9), (Box(0, 0, 10, 10), .8)]
    assert greedy_counts(truth, predictions) == (1, 1, 0)


def test_detection_gate_allows_passing_head_when_higher_recall_head_fails_latency() -> None:
    detection = {
        "yolo11s": {"equal_camera_recall": .50, "worst_camera_recall": .40, "equal_camera_precision": .80, "latency_ms_per_frame": 10.0},
        "yolo26s_one_to_one": {"equal_camera_recall": .56, "worst_camera_recall": .40, "equal_camera_precision": .78, "latency_ms_per_frame": 12.0},
        "yolo26s_one_to_many": {"equal_camera_recall": .60, "worst_camera_recall": .42, "equal_camera_precision": .79, "latency_ms_per_frame": 16.0},
    }
    result = detection_gate(detection)
    assert result["recommended"] is True
    assert result["selected_variant"] == "yolo26s_one_to_one"
    assert result["variants"]["yolo26s_one_to_many"]["checks"]["latency"] is False


def test_depth_gate_fails_closed_when_static_recall_is_zero() -> None:
    cameras = {
        camera: {"joint_proposal_recall": 0.0}
        for camera in ("VIRAT_S_010004", "VIRAT_S_010111", "VIRAT_S_010112", "VIRAT_S_010204")
    }
    proposals = {
        "static_semantic": {"equal_camera_joint_proposal_recall": 0.0, "per_camera": cameras},
        "semantic_75_depth_25": {
            "equal_camera_joint_proposal_recall": 0.0,
            "per_camera": cameras,
            "mean_roi_area": .095,
            "max_roi_area": .095,
        },
    }
    result = depth_gate(proposals, 12.0)
    assert result["recommended"] is False
    assert result["relative_recall_gain"] is None
