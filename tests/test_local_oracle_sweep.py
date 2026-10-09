import numpy as np
import pytest

from research_tools.cascade_experiment import Box, CallableDetectorAdapter, GroundTruth, Prediction
from research_tools.local_oracle_sweep import merge_sweep_predictions, motion_residual_boxes, proposal_coverage, run_local_oracle_sweep, select_disjoint_sample


class BatchDetector(CallableDetectorAdapter):
    def __init__(self, fn, name):
        super().__init__(fn, name)
        self.batch_calls = 0

    def predict_batch(self, frames, *, frame_indices):
        self.batch_calls += 1
        return tuple(tuple(self.predict(frame, frame_index=index)) for frame, index in zip(frames, frame_indices))


def detector(boxes, name):
    return BatchDetector(lambda frame, frame_index: tuple(Prediction(frame_index, Box(*box), score, name) for box, score in boxes), name)


def test_merge_sweep_applies_local_confidence_and_fusion():
    global_rows = (Prediction(1, Box(10, 10, 30, 50), .7, "small"),)
    local_rows = (
        Prediction(1, Box(10, 10, 30, 50), .2, "local"),
        Prediction(1, Box(40, 10, 60, 50), .8, "local"),
    )
    from evidencegate.cascade import ROI
    roi = ROI(0, 0, 100, 100, "r", 0)
    nms = merge_sweep_predictions(1, global_rows, local_rows, roi, (0, 0, 100, 100), local_confidence=.5, fusion="nms")
    wbf = merge_sweep_predictions(1, global_rows, local_rows, roi, (0, 0, 100, 100), local_confidence=.1, fusion="wbf")
    assert len(nms) == 2
    assert len(wbf) == 2
    with pytest.raises(ValueError, match="unsupported_fusion"):
        merge_sweep_predictions(1, global_rows, local_rows, roi, (0, 0, 100, 100), local_confidence=.5, fusion="bad")


def test_local_oracle_factorial_reuses_predictions_across_threshold_and_fusion():
    frames = {index: np.zeros((100, 100, 3), dtype=np.uint8) for index in range(1, 11)}
    truth = tuple(GroundTruth(index, index, Box(10, 10, 30, 50)) for index in frames)
    small = detector([], "small")
    local_384 = detector([((10, 10, 30, 50), .2)], "local384")
    local_512 = detector([((10, 10, 30, 50), .8)], "local512")
    result = run_local_oracle_sweep(
        frames, truth, small, {384: local_384, 512: local_512},
        roi_sizes=(384, 512), confidence_thresholds=(.1, .5), fusions=("nms", "wbf"),
        budget=.2, batch_size=8, seed=3, sequence_id="fixture",
    )
    assert len(result["conditions"]) == 1 + 2 * 2 * 2
    assert local_384.batch_calls == 1
    assert local_512.batch_calls == 1
    assert result["conditions"]["oracle_i384_c50_nms"]["count_mae"] == 1
    assert result["conditions"]["oracle_i512_c50_nms"]["count_mae"] < 1
    assert result["route_scope"].endswith("NOT_DEPLOYABLE")
    assert result["context_expansion"] == 0


def test_local_oracle_sweep_rejects_incomplete_detector_matrix():
    empty = detector([], "empty")
    with pytest.raises(ValueError, match="roi_sizes_and_detectors_mismatch"):
        run_local_oracle_sweep(
            {1: np.zeros((10, 10, 3), dtype=np.uint8)}, (), empty, {384: empty},
            roi_sizes=(384, 512), confidence_thresholds=(.25,), fusions=("nms",),
            budget=.2, batch_size=1, seed=1, sequence_id="bad",
        )


def test_disjoint_sample_excludes_exploratory_frames():
    available = tuple(range(1, 101))
    exploratory = set(select_disjoint_sample(available, 20, seed=20260812))
    confirmation = set(select_disjoint_sample(available, 20, seed=20260813, exclude_seed=20260812, exclude_sample_size=20))
    assert len(confirmation) == 20
    assert exploratory.isdisjoint(confirmation)


def test_disjoint_sample_fails_when_remaining_pool_is_too_small():
    with pytest.raises(ValueError, match="insufficient_disjoint_frames"):
        select_disjoint_sample(tuple(range(5)), 4, seed=2, exclude_seed=1, exclude_sample_size=4)


def test_heuristic_route_is_label_blind_and_uses_distinct_condition_names():
    frames = {index: np.zeros((100, 100, 3), dtype=np.uint8) for index in range(1, 6)}
    truth = tuple(GroundTruth(index, index, Box(70, 70, 90, 95)) for index in frames)
    small = detector([((10, 10, 30, 50), .4)], "small")
    local = detector([((10, 10, 30, 50), .8)], "local")
    result = run_local_oracle_sweep(
        frames, truth, small, {512: local}, roi_sizes=(512,), confidence_thresholds=(.1,),
        fusions=("wbf",), budget=.2, batch_size=8, seed=3, sequence_id="heuristic", route="heuristic",
    )
    assert "heuristic_i512_c10_wbf" in result["conditions"]
    assert result["route_scope"] == "LABEL_BLIND_DEPLOYABLE_CANDIDATE"


def test_motion_residual_finds_changed_region_and_supports_route():
    reference = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = reference.copy(); frame[60:90, 50:80] = 255
    boxes, energy = motion_residual_boxes(frame, reference, block_size=10, percentile=80, minimum_difference=5)
    assert boxes and energy > 0
    assert any(box.x1 <= 50 and box.x2 >= 80 and box.y1 <= 60 and box.y2 >= 90 for box in boxes)
    frames = {index: frame.copy() for index in range(1, 6)}
    references = {index: reference.copy() for index in frames}
    truth = tuple(GroundTruth(index, index, Box(50, 60, 80, 90)) for index in frames)
    small = detector([], "small"); local = detector([((0, 0, 30, 30), .8)], "local")
    result = run_local_oracle_sweep(
        frames, truth, small, {512: local}, roi_sizes=(512,), confidence_thresholds=(.1,),
        fusions=("wbf",), budget=.2, batch_size=8, seed=3, sequence_id="motion", route="motion", reference_frames=references,
    )
    assert "motion_i512_c10_wbf" in result["conditions"]


def test_proposal_coverage_separates_location_from_local_detection():
    from evidencegate.cascade import ROI
    coverage = proposal_coverage(
        (1,), {1: ROI(0, 0, 50, 50, "r", 0)},
        {1: (Box(10, 10, 20, 30), Box(70, 70, 80, 90))},
    )
    assert coverage["proposal_recall_on_selected_misses"] == .5
    assert coverage["selected_frames_with_miss_fraction"] == 1
