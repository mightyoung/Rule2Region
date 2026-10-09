from research_tools.cascade_experiment import Box, Prediction
from research_tools.residual_compiler import CONF_MAX, HEIGHT_MAX
from research_tools.residual_query import (
    EXPAND,
    MAX_QUERIES_PER_FRAME,
    MIN_SIDE,
    SUPPORT_IOU,
    apply_queries,
    expand_roi,
    propose_queries,
    unmatched_previous,
)


def test_r1_thresholds_untouched() -> None:
    assert CONF_MAX == 0.35
    assert HEIGHT_MAX == 28.0
    assert SUPPORT_IOU == 0.30
    assert MAX_QUERIES_PER_FRAME == 2
    assert EXPAND == 0.50
    assert MIN_SIDE == 64.0


def test_unmatched_previous_requires_dropout() -> None:
    past = Prediction(1, Box(10, 10, 40, 80), 0.90, "yolo11s_global")
    matched = Prediction(2, Box(12, 12, 42, 82), 0.80, "yolo11s_global")
    assert unmatched_previous((past,), (matched,)) == ()
    assert unmatched_previous((past,), ()) == (past,)


def test_propose_queries_ranks_and_caps() -> None:
    previous = (
        Prediction(1, Box(0, 0, 20, 40), 0.40, "yolo11s_global"),
        Prediction(1, Box(100, 100, 140, 180), 0.90, "yolo11s_global"),
        Prediction(1, Box(400, 200, 430, 250), 0.70, "yolo11s_global"),
    )
    rois = propose_queries(previous, ())
    assert len(rois) == 2
    assert rois[0].x1 >= 100 - 20 - 1


def test_expand_roi_enforces_min_side_and_clip() -> None:
    roi = expand_roi(Box(0, 0, 10, 10), width=1280, height=720)
    assert roi.x2 - roi.x1 >= MIN_SIDE
    assert roi.x1 == 0.0
    assert roi.y1 == 0.0


def test_apply_queries_is_causal_and_val_only() -> None:
    by_frame = {
        1: (Prediction(1, Box(10, 10, 40, 80), 0.90, "yolo11s_global"),),
        2: (),
        3: (),
    }
    calls: list[int] = []

    def detect(frame_id: int, roi: Box, _image: object) -> tuple[Prediction, ...]:
        calls.append(frame_id)
        return (Prediction(frame_id, Box(roi.x1 + 2, roi.y1 + 2, roi.x1 + 20, roi.y1 + 40), 0.80, "yolo26_query"),)

    compiled, stats = apply_queries(by_frame, [1, 2, 3], [2, 3], detect, {2: None, 3: None})
    assert 1 not in calls
    assert calls == [2]
    assert stats["val_queries"] == 1
    assert compiled[1][0].source == "yolo11s_global"
    assert any(row.source == "yolo26_query" for row in compiled[2])
    assert compiled[3] == ()


def test_apply_queries_does_not_rewrite_b0_when_idle() -> None:
    box = Prediction(2, Box(10, 10, 40, 80), 0.90, "yolo11s_global")
    overlap = Prediction(2, Box(12, 12, 42, 82), 0.80, "yolo11s_global")
    by_frame = {1: (), 2: (box, overlap)}

    def detect(_frame_id: int, _roi: Box, _image: object) -> tuple[Prediction, ...]:
        raise AssertionError("detect_must_not_run_without_seeds")

    compiled, stats = apply_queries(by_frame, [1, 2], [2], detect, {2: None})
    assert stats["val_queries"] == 0
    assert compiled[2] == (box, overlap)
