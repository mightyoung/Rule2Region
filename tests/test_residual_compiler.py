from research_tools.cascade_experiment import Box, Prediction
from research_tools.residual_compiler import apply_reject, reject_box


def test_reject_drops_small_low_conf_box_without_past_support() -> None:
    lonely = Prediction(2, Box(10, 10, 18, 20), 0.20, "yolo11s_global")
    assert reject_box(lonely, ()) is True
    supported = Prediction(2, Box(10, 10, 18, 20), 0.20, "yolo11s_global")
    past = (Prediction(1, Box(11, 11, 19, 21), 0.80, "yolo11s_global"),)
    assert reject_box(supported, past) is False


def test_apply_reject_is_causal_and_keeps_tall_boxes() -> None:
    tall = Prediction(5, Box(0, 0, 40, 80), 0.20, "yolo11s_global")
    by_frame = {5: (tall,)}
    kept = apply_reject(by_frame, [5])
    assert kept[5] == (tall,)
