from research_tools.cascade_experiment import Box, Prediction
from research_tools.route_a_clip_commissioning import compile_m0, score_condition


def test_saturated_baseline_cannot_pass_relative_mae_gate() -> None:
    clip = {
        "validation_frames": [1, 2],
        "global_by_frame": {1: (), 2: ()},
        "truth": [],
    }
    scored = score_condition(clip, {1: (), 2: ()}, tile_cost_ratio=0.0)
    assert scored["baseline_mae"] == 0.0
    assert scored["verdict"] == "fail"
    assert scored["intention"] == "B0"


def test_compile_m0_keeps_global_when_no_tiles() -> None:
    box = Box(10, 10, 20, 20)
    clip = {
        "validation_frames": [3],
        "global_by_frame": {3: (Prediction(3, box, 0.9, "yolo11s_global"),)},
    }
    compiled = compile_m0(clip, {"tiles": ()}, None)
    assert len(compiled[3]) == 1
