import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.run_missed_person_proposal_baseline import (
    FrameRecord,
    _camera,
    _contains_center,
    _roi_boxes,
    _split_records,
    _union_area_ratio,
    evaluate,
)


def _record(sequence):
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    return FrameRecord(sequence, _camera(sequence), 1, 40, 20, (), (Box(1, 1, 3, 5),), rgb, rgb)


def test_camera_parser_and_split_are_camera_disjoint():
    rows = [_record("VIRAT_S_010004_01_x"), _record("VIRAT_S_010112_00_x"), _record("VIRAT_S_010204_03_x")]
    splits = _split_records(rows)
    cameras = [{row.camera_id for row in splits[name]} for name in ("fit", "calibration", "evaluation")]
    assert all(cameras)
    assert not cameras[0] & cameras[1]
    assert not cameras[0] & cameras[2]
    assert not cameras[1] & cameras[2]


def test_roi_compiler_respects_total_area_budget():
    score = np.zeros((23, 40), dtype=np.float32)
    score[2, 2] = 1
    score[20, 37] = .9
    boxes = _roi_boxes(score, 2, .10, 1280, 720)
    assert _union_area_ratio(boxes, 1280, 720) <= .11
    assert _contains_center(Box(0, 0, 10, 10), Box(4, 4, 6, 6))
    assert not _contains_center(Box(0, 0, 2, 2), Box(4, 4, 6, 6))


def test_evaluation_applies_budget_per_sequence_and_equal_weights_recall():
    rows = []
    for sequence in ("VIRAT_S_010004_01_x", "VIRAT_S_010004_07_x"):
        for frame in range(5):
            base = _record(sequence)
            rows.append(FrameRecord(base.sequence_id, base.camera_id, frame, 40, 20, (), base.truth, base.rgb, base.reference_rgb))
    scores = np.zeros((10, 23, 40), dtype=np.float32)
    scores[:, 3, 2] = np.arange(10)
    result = evaluate(rows, scores, peak_count=1, total_area=.095, budget=.20)
    assert result["selected_by_sequence"] == {
        "VIRAT_S_010004_01_x": 1,
        "VIRAT_S_010004_07_x": 1,
    }
    assert result["selected_frames"] == 2
