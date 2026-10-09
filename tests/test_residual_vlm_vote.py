from research_tools.cascade_experiment import Box, Prediction
from research_tools.residual_compiler import CONF_MAX, HEIGHT_MAX
from research_tools.residual_vlm_vote import (
    HOLD_RATIO,
    apply_tau,
    box_id_for,
    clip_visual_hold,
    fit_tau,
    parse_vote,
    select_history_targets,
    vote_prompt,
)


def test_r1_thresholds_untouched() -> None:
    assert CONF_MAX == 0.35
    assert HEIGHT_MAX == 28.0
    assert HOLD_RATIO == 0.30


def test_select_history_targets_skips_validation_and_picks_lowest_conf() -> None:
    high = Prediction(1, Box(10, 10, 40, 80), 0.90, "yolo11s_global")
    low = Prediction(1, Box(100, 100, 120, 130), 0.20, "yolo11s_global")
    val_only = Prediction(20, Box(10, 10, 40, 80), 0.05, "yolo11s_global")
    clip = {
        "camera_id": "VIRAT_S_010004",
        "sequence_id": "VIRAT_S_010004_01_000163_000188",
        "history_frames": [1],
        "validation_frames": [20],
        "global_by_frame": {1: (high, low), 20: (val_only,)},
    }
    jobs = select_history_targets(clip)
    assert len(jobs) == 1
    assert jobs[0]["frame"] == 1
    assert jobs[0]["confidence"] == 0.20
    assert jobs[0]["split"] == "history"
    assert jobs[0]["box_id"].startswith("b010004_01_")


def test_parse_vote_requires_exact_box_id() -> None:
    parsed = parse_vote('{"box_id":"b010004_01_000001","judgment":"FP"}', "b010004_01_000001")
    assert parsed["judgment"] == "FP"
    try:
        parse_vote('{"box_id":"other","judgment":"FP"}', "b010004_01_000001")
    except ValueError as exc:
        assert "box_id" in str(exc)
    else:
        raise AssertionError("mismatch_must_fail")


def test_fit_tau_is_fail_closed_on_overlap_or_missing_class() -> None:
    empty = fit_tau([{"judgment": "UNSURE", "parse_ok": True, "confidence": 0.2}])
    assert empty["tau"] is None
    overlap = fit_tau(
        [
            {"judgment": "FP", "parse_ok": True, "confidence": 0.80},
            {"judgment": "PERSON", "parse_ok": True, "confidence": 0.20},
        ]
    )
    assert overlap["tau"] is None
    separated = fit_tau(
        [
            {"judgment": "FP", "parse_ok": True, "confidence": 0.20},
            {"judgment": "PERSON", "parse_ok": True, "confidence": 0.80},
        ]
    )
    assert separated["tau"] == 0.50


def test_apply_tau_keeps_supported_or_confident_boxes() -> None:
    past = Prediction(1, Box(10, 10, 40, 80), 0.90, "yolo11s_global")
    lonely = Prediction(2, Box(200, 200, 220, 220), 0.20, "yolo11s_global")
    supported = Prediction(2, Box(12, 12, 42, 82), 0.20, "yolo11s_global")
    compiled = apply_tau({1: (past,), 2: (lonely, supported)}, [1, 2], 0.50)
    sources = {(row.box.x1, row.confidence) for row in compiled[2]}
    assert (200.0, 0.20) not in sources
    assert (12.0, 0.20) in sources


def test_visual_hold_and_prompt_forbid_config_search() -> None:
    votes = [
        {"parse_ok": False, "token_position_warning": False},
        {"parse_ok": True, "token_position_warning": True},
        {"parse_ok": True, "token_position_warning": False},
    ]
    assert clip_visual_hold(votes) is True
    text = vote_prompt(box_id_for("VIRAT_S_010004_01_000163_000188", 1), "VIRAT_S_010004", 0.2)
    assert "config_id" in text
    assert "PERSON, FP, or UNSURE" in text
    assert "collage" in text
