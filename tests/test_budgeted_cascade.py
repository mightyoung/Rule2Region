import numpy as np
import pytest

from research_tools.budgeted_cascade import deterministic_budget, rotated_condition_order, run_budgeted_phase, tight_candidate_rois
from research_tools.cascade_experiment import Box, CallableDetectorAdapter, GroundTruth, Prediction


class BatchDetector(CallableDetectorAdapter):
    def predict_batch(self, frames, *, frame_indices):
        return tuple(tuple(self.predict(frame, frame_index=index)) for frame, index in zip(frames, frame_indices))


def detector(boxes, name):
    return BatchDetector(lambda frame, frame_index: tuple(Prediction(frame_index, Box(*box), score, name) for box, score in boxes), name)


def test_deterministic_budget_uses_ceiling_and_stable_scores():
    frames = tuple(range(10))
    assert deterministic_budget(frames, .05, {}, seed=1, namespace="x") == deterministic_budget(frames, .05, {}, seed=1, namespace="x")
    assert len(deterministic_budget(frames, .05, {}, seed=1, namespace="x")) == 1
    assert 4 in deterministic_budget(frames, .2, {4: 9}, seed=1, namespace="x")


def test_rotated_order_is_deterministic_and_preserves_conditions():
    first = rotated_condition_order(("a", "b", "c"), seed=7, sequence_id="s", batch_index=2)
    assert first == rotated_condition_order(("a", "b", "c"), seed=7, sequence_id="s", batch_index=2)
    assert set(first) == {"a", "b", "c"}


def test_budgeted_phase_reuses_small_predictions_for_unselected_frames():
    frames = {index: np.zeros((100, 100, 3), dtype=np.uint8) for index in range(1, 11)}
    truth = tuple(GroundTruth(index, index, Box(10, 10, 30, 50)) for index in frames)
    small = detector([], "small")
    large = detector([((10, 10, 30, 50), .9)], "large")
    result = run_budgeted_phase(frames, truth, small, large, large, budgets=(.1,), batch_size=4, seed=3, sequence_id="fixture")
    condition = result["conditions"]["oracle_budget_10"]
    assert condition["selected_frames"] == 1
    rows = result["receipts"]["oracle_budget_10"]
    assert sum(row["upgraded"] for row in rows) == 1
    assert all(row["predictions"] == [] for row in rows if not row["upgraded"])
    assert result["condition_order_policy"] == "deterministic_batch_rotation"


def test_tight_rois_merge_adjacent_boxes_and_bound_expanded_area():
    boxes = (Box(500, 100, 520, 160), Box(535, 105, 555, 165), Box(50, 500, 70, 560))
    rois = tight_candidate_rois(1280, 720, boxes, prefix="t")
    assert len(rois) == 2
    for roi in rois:
        ex = roi.expanded
        expanded_area = (min(1280, ex[2]) - max(0, ex[0])) * (min(720, ex[3]) - max(0, ex[1]))
        assert .05 <= roi.pixel_area / (1280 * 720) <= .151
        assert expanded_area / (1280 * 720) <= .151


@pytest.mark.parametrize("expansion", (0.0, 0.15, 0.30))
def test_tight_rois_support_bounded_context_scan(expansion):
    roi = tight_candidate_rois(
        1280, 720, (Box(600, 100, 620, 160),), prefix="c", expansion=expansion
    )[0]
    ex = roi.expanded
    expanded_area = (min(1280, ex[2]) - max(0, ex[0])) * (min(720, ex[3]) - max(0, ex[1]))
    assert roi.expansion == expansion
    assert roi.pixel_area / (1280 * 720) >= .05
    assert expanded_area / (1280 * 720) <= .151


def test_tight_rois_reject_context_that_cannot_meet_area_contract():
    with pytest.raises(ValueError, match="context_expansion_incompatible"):
        tight_candidate_rois(1280, 720, (Box(600, 100, 620, 160),), prefix="c", expansion=1.0)


def test_budgeted_receipt_records_core_and_actual_crop_boundaries():
    frames = {index: np.zeros((720, 1280, 3), dtype=np.uint8) for index in range(1, 6)}
    truth = tuple(GroundTruth(index, index, Box(600, 100, 620, 160)) for index in frames)
    small = detector([], "small"); large = detector([((40, 40, 60, 100), .9)], "large")
    result = run_budgeted_phase(frames, truth, small, large, large, budgets=(.2,), batch_size=4, seed=3, sequence_id="fixture")
    upgraded = next(row for row in result["receipts"]["oracle_budget_20"] if row["upgraded"])
    roi = upgraded["selected_roi"]
    assert roi["core_area_ratio"] >= .05
    assert roi["crop_area_ratio"] <= .151
    assert roi["core_xyxy"] != roi["crop_xyxy"]
    assert roi["context_expansion"] == .15


def test_budgeted_phase_records_context_scan_lineage():
    frames = {index: np.zeros((720, 1280, 3), dtype=np.uint8) for index in range(1, 6)}
    truth = tuple(GroundTruth(index, index, Box(600, 100, 620, 160)) for index in frames)
    small = detector([], "small"); large = detector([((40, 40, 60, 100), .9)], "large")
    result = run_budgeted_phase(
        frames, truth, small, large, large, budgets=(.2,), batch_size=4,
        seed=3, sequence_id="fixture", context_expansion=.30,
    )
    assert result["context_expansion"] == .30
    assert result["max_expanded_area_ratio"] == .15
    upgraded = next(row for row in result["receipts"]["oracle_budget_20"] if row["upgraded"])
    assert upgraded["selected_roi"]["context_expansion"] == .30
    crop = upgraded["selected_roi"]["crop_xyxy"]
    assert all(isinstance(value, int) for value in crop)
    assert result["conditions"]["oracle_budget_20"]["timing_scope"] == "MODEL_CALLS_ONLY_NOT_END_TO_END"
    assert result["route_comparability"]["oracle"].startswith("GT_SELECTED")


def test_budgeted_phase_rejects_empty_frames():
    empty = detector([], "empty")
    with pytest.raises(ValueError, match="frames_must_not_be_empty"):
        run_budgeted_phase({}, (), empty, empty, empty, budgets=(.2,), batch_size=4, seed=3, sequence_id="empty")
