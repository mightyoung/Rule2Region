from pathlib import Path

import pytest

from research_tools.execution_policy.compile import h1_pass
from research_tools.execution_policy.stack import MAE_TOLERANCE, decide_hard_path, should_open_hard
from research_tools.execution_policy.v6 import (
    HISTORY_BLOCKS,
    admission_cell,
    conservative_bounds,
    conservative_ok,
    count_cells,
    decide_v6_hard,
    policy_yolo26_calls,
    split_contiguous_blocks,
    summarize_method,
)


def test_v6_keeps_h1_floor_and_four_time_blocks() -> None:
    assert MAE_TOLERANCE == -0.05
    assert HISTORY_BLOCKS == 4
    gate = h1_pass(
        compiled_mae=5.0,
        b11_mae=5.0,
        precision_change_vs_b11=0.0,
        compiled_proxy=0.5,
        compiled_time_ms=20.0,
        b26_time_ms=25.0,
    )
    assert gate["passed"] is True


def test_split_contiguous_blocks_are_time_ordered_not_shuffled() -> None:
    frames = tuple(range(32))
    blocks = split_contiguous_blocks(frames, n=4)
    assert len(blocks) == 4
    assert blocks[0] == tuple(range(0, 8))
    assert blocks[-1] == tuple(range(24, 32))
    assert [item for block in blocks for item in block] == list(frames)
    with pytest.raises(ValueError, match="insufficient_history_for_v6_blocks"):
        split_contiguous_blocks(range(3), n=4)
    with pytest.raises(ValueError, match="history_frames_not_strictly_increasing"):
        split_contiguous_blocks([0, 2, 1, 3], n=4)
    with pytest.raises(ValueError, match="history_frames_not_strictly_increasing"):
        split_contiguous_blocks([0, 1, 1, 2], n=4)


def test_conservative_ok_uses_worst_block_not_mean() -> None:
    good = {"mae_change_vs_b11": 0.0, "precision_change_vs_b11": 0.0, "time_ms": 20.0, "b26_time_ms": 30.0}
    bad_mae = {**good, "mae_change_vs_b11": -0.08}
    assert conservative_ok([good, good, good, good]) is True
    assert conservative_ok([good, good, good, bad_mae]) is False
    slow = {**good, "time_ms": 40.0, "b26_time_ms": 30.0}
    assert conservative_ok([good, slow, good, good]) is False
    bounds = conservative_bounds([good, good, good, bad_mae])
    assert bounds == {
        "passed": False,
        "reason": "mae_bound_below_floor",
        "block_count": 4,
        "mae_change_lower_bound": -0.08,
        "precision_change_lower_bound": 0.0,
        "time_ms_upper_bound": 20.0,
        "b26_time_ms_lower_bound": 30.0,
    }


def test_v6_opens_only_if_point_h1_and_conservative() -> None:
    opened = {"mae": 5.0, "h1": {"passed": True}}
    point = decide_hard_path(foci_count=2, knob_model="yolo11s", closed_mae=7.0, opened=opened)
    assert point is True
    assert decide_v6_hard(point_open=True, conservative=True, foci_count=2, knob_model="yolo11s") is True
    assert decide_v6_hard(point_open=True, conservative=False, foci_count=2, knob_model="yolo11s") is False
    assert decide_v6_hard(point_open=False, conservative=True, foci_count=2, knob_model="yolo11s") is False
    assert should_open_hard(b11_mae=8.0, b26_mae=5.0, foci_count=2) is True


def test_admission_cells_match_prereg_names() -> None:
    assert admission_cell(opened=True, future_opened_pass=False) == "false_admission"
    assert admission_cell(opened=True, future_opened_pass=True) == "true_admission"
    assert admission_cell(opened=False, future_opened_pass=False) == "true_rejection"
    assert admission_cell(opened=False, future_opened_pass=True) == "false_rejection"
    with pytest.raises(ValueError, match="unknown_admission_cells"):
        count_cells(["not_a_cell"])


def test_v6_summary_is_view_level_and_reports_calls_time_and_future_rate() -> None:
    assert policy_yolo26_calls(frame_count=32, knob_model="yolo11s", opened=False, foci_count=2) == 0
    assert policy_yolo26_calls(frame_count=32, knob_model="yolo11s", opened=True, foci_count=2) == 64
    assert policy_yolo26_calls(frame_count=32, knob_model="yolo26", opened=True, foci_count=2) == 32
    rows = [
        {
            "cell": "false_admission",
            "future_h1_passed": False,
            "policy_model_time_ms_mean": 30.0,
            "yolo26_call_count": 64,
        },
        {
            "cell": "true_rejection",
            "future_h1_passed": False,
            "policy_model_time_ms_mean": 20.0,
            "yolo26_call_count": 0,
        },
        {
            "cell": "false_rejection",
            "future_h1_passed": True,
            "policy_model_time_ms_mean": 25.0,
            "yolo26_call_count": 0,
        },
    ]
    summary = summarize_method(rows)
    assert summary["view_count"] == 3
    assert summary["false_admission"] == 1
    assert summary["future_h1_pass_count"] == 1
    assert summary["future_h1_pass_rate"] == pytest.approx(1 / 3)
    assert summary["policy_model_time_ms_mean"] == 25.0
    assert summary["yolo26_call_count"] == 64


def test_v6_prereg_and_runner_forbid_official_on_seen_pack() -> None:
    prereg = Path("docs/plans/2026-08-20-future-qualification-v6-preregistration.md").read_text(encoding="utf-8")
    assert "错误放行" in prereg
    assert "不得再当 v6 的正式未来集" in prereg
    runner = Path("research_tools/execution_policy/run_v6.py").read_text(encoding="utf-8")
    assert "v6_official" in runner
    assert "seen_confirmatory_validation" in runner
    assert "HISTORY_BLOCKS" in runner
