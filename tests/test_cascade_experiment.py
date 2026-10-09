import json

import pytest

from research_tools.cascade_experiment import (
    Box,
    Prediction,
    compute_metrics,
    immutable_json_write,
    parse_mot17_gt,
    select_frames,
    CallableDetectorAdapter,
    candidate_rois,
    run_cascade_frame,
)


def test_mot17_parser_filters_non_pedestrian_and_ignored_rows():
    rows = "\n".join(("1,5,10,20,4,8,1,1,0.8", "1,6,0,0,5,5,0,1,1", "2,7,0,0,4,4,1,2,1"))
    result = parse_mot17_gt(rows)
    assert len(result) == 1
    assert result[0].frame == 1 and result[0].box == Box(10, 20, 14, 28)


def test_frame_selection_is_reproducible_and_sorted():
    assert select_frames([4, 1, 3, 2], 2, seed=11) == select_frames([2, 4, 1, 3], 2, seed=11)
    assert select_frames([4, 1, 3, 2], 4) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        select_frames([1], -1)


def test_metrics_are_deterministic_and_include_small_recall():
    truth = parse_mot17_gt("1,1,0,0,10,10,1,1,1\n2,2,0,0,64,64,1,1,1\n")
    predictions = (Prediction(1, Box(0, 0, 10, 10), 0.9), Prediction(2, Box(50, 50, 60, 60), 0.8))
    metrics = compute_metrics(predictions, truth)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["small_object_recall"] == pytest.approx(1.0)
    assert metrics["count_mae"] == pytest.approx(0.0)


def test_immutable_json_write_refuses_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    digest = immutable_json_write(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        immutable_json_write(path, {"changed": True})


def test_edge_roi_crop_maps_from_clipped_origin():
    import numpy as np

    global_detector = CallableDetectorAdapter(lambda frame, frame_index: (), "small")
    local_detector = CallableDetectorAdapter(
        lambda frame, frame_index: (Prediction(frame_index, Box(1, 2, 11, 12), .9, "large"),),
        "large",
    )
    roi = candidate_rois(100, 100, cell_width=50, overlap=0.0)[0]
    predictions, _ = run_cascade_frame(
        np.zeros((100, 100, 3), dtype=np.uint8), 1, global_detector, local_detector,
        selected_rois=(roi,), condition="benefit_wbf",
    )
    assert predictions[0].box == Box(1, 2, 11, 12)


def test_nonempty_global_predictions_enter_cascade_fusion():
    import numpy as np

    global_detector = CallableDetectorAdapter(
        lambda frame, frame_index: (Prediction(frame_index, Box(10, 10, 30, 50), .8, "small"),),
        "small",
    )
    local_detector = CallableDetectorAdapter(lambda frame, frame_index: (), "large")
    predictions, _ = run_cascade_frame(
        np.zeros((100, 100, 3), dtype=np.uint8), 1, global_detector, local_detector,
        selected_rois=(), condition="benefit_wbf",
    )
    assert predictions == (Prediction(1, Box(10, 10, 30, 50), .8, "small"),)


def test_nms_keeps_crowded_same_roi_and_reports_clipped_pixels():
    import numpy as np
    from evidencegate.cascade import ROI
    global_detector = CallableDetectorAdapter(lambda frame, frame_index: (), "small")
    local_detector = CallableDetectorAdapter(
        lambda frame, frame_index: (
            Prediction(frame_index, Box(1, 1, 21, 41), .9, "large"),
            Prediction(frame_index, Box(15, 1, 35, 41), .8, "large"),
        ), "large",
    )
    roi = ROI(-10, -10, 40, 40, "edge")
    predictions, stats = run_cascade_frame(
        np.zeros((100, 100, 3), dtype=np.uint8), 1, global_detector, local_detector,
        selected_rois=(roi,), condition="heuristic", nms_threshold=.1,
    )
    assert len(predictions) == 2
    assert stats["extra_roi_pixel_ratio"] == pytest.approx(1849 / 10000)
    assert stats["processed_pixel_ratio"] == pytest.approx(1 + 1849 / 10000)
