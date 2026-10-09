from pathlib import Path

import pytest

from research_tools.execution_policy.card import parse_policy_card
from research_tools.execution_policy.stack import (
    H1_PLANES,
    NOT_IN_H1,
    apply_idle_card,
    force_close_hard,
    assert_not_clock_schedule,
    card_beats_incumbent,
    card_not_worse_than_incumbent,
    compile_view_plan,
    hard_crop_frames,
    idle_source_reason,
    pick_time_bin_knob,
    promote_card,
    decide_hard_path,
    should_open_hard,
    stack_proxy,
    step_online_conf,
)


def _cand(**overrides):
    row = {
        "model": "yolo11s",
        "imgsz": 640,
        "conf": 0.15,
        "mae": 4.0,
        "precision": 0.80,
        "time_ms": 27.0,
        "b11_mae": 5.0,
        "b11_precision": 0.75,
        "b26_time_ms": 30.0,
        "proxy": 0.25,
    }
    row.update(overrides)
    return row


def test_hard_path_opens_only_when_26_beats_11s_and_foci_exist() -> None:
    assert should_open_hard(b11_mae=8.0, b26_mae=5.0, foci_count=2) is True
    assert should_open_hard(b11_mae=8.0, b26_mae=5.0, foci_count=0) is False
    assert should_open_hard(b11_mae=5.0, b26_mae=6.0, foci_count=2) is False
    assert should_open_hard(b11_mae=0.0, b26_mae=0.0, foci_count=2) is False
    opened_ok = {"mae": 5.0, "h1": {"passed": True}}
    assert decide_hard_path(foci_count=2, knob_model="yolo11s", closed_mae=6.0, opened=opened_ok) is True
    assert decide_hard_path(foci_count=2, knob_model="yolo11s", closed_mae=4.0, opened=opened_ok) is False
    assert decide_hard_path(foci_count=2, knob_model="yolo11s", closed_mae=6.0, opened={"mae": 5.0, "h1": {"passed": False}}) is False
    assert decide_hard_path(foci_count=0, knob_model="yolo11s", closed_mae=6.0, opened=opened_ok) is False
    assert decide_hard_path(foci_count=2, knob_model="yolo26", closed_mae=6.0, opened=opened_ok) is False


def test_hard_crops_on_validation_only_after_gate_opens() -> None:
    history = (1, 2)
    val = (3, 4)
    assert hard_crop_frames(
        history_frames=history, validation_frames=val, foci_count=2, open_hard=None, force_closed=False, phase="history"
    ) == history
    assert hard_crop_frames(
        history_frames=history, validation_frames=val, foci_count=2, open_hard=False, force_closed=False, phase="validation"
    ) == ()
    assert hard_crop_frames(
        history_frames=history, validation_frames=val, foci_count=2, open_hard=True, force_closed=False, phase="validation"
    ) == val
    assert hard_crop_frames(
        history_frames=history, validation_frames=val, foci_count=0, open_hard=True, force_closed=False, phase="history"
    ) == ()
    assert hard_crop_frames(
        history_frames=history, validation_frames=val, foci_count=2, open_hard=True, force_closed=True, phase="validation"
    ) == ()
    runner = Path("research_tools/execution_policy/run_execution_stack.py").read_text(encoding="utf-8")
    assert "hard_crop_frames" in runner
    history_at = runner.index('phase="history"')
    decide_at = runner.index('plan["open_hard"] = decide_hard_path')
    val_at = runner.index('phase="validation"')
    assert history_at < decide_at < val_at


def test_online_conf_still_steps() -> None:
    assert step_online_conf(0.25, 10, 4) == 0.35
    assert step_online_conf(0.25, 0, 3) == 0.15


def test_compile_view_plan_defaults_to_a_and_gates_hard_path() -> None:
    closed = compile_view_plan(
        knob_candidates=[_cand(), _cand(model="yolo26", mae=6.0, proxy=1.0, time_ms=28.0)],
        b11_mae=5.0,
        b26_mae=5.1,
        foci_count=3,
    )
    assert closed["plane"] == "compile"
    assert closed["open_hard"] is False
    assert closed["knob"]["model"] == "yolo11s"
    opened = compile_view_plan(
        knob_candidates=[_cand(mae=7.5, b11_mae=8.0)],
        b11_mae=8.0,
        b26_mae=5.0,
        foci_count=2,
        time_bin="evening_peak",
    )
    assert opened["open_hard"] is True
    assert opened["time_bin"] == "evening_peak"
    assert stack_proxy(open_hard=False) == 0.25
    assert stack_proxy(open_hard=True) == 0.5


def test_clock_schedule_is_not_a_time_bin() -> None:
    with pytest.raises(ValueError, match="clock_schedule_forbidden"):
        assert_not_clock_schedule("19:00-21:00")
    with pytest.raises(ValueError, match="clock_schedule_forbidden"):
        compile_view_plan(knob_candidates=[_cand()], b11_mae=4.0, b26_mae=4.0, foci_count=0, time_bin="9:00")
    with pytest.raises(ValueError, match="clock_schedule_forbidden"):
        pick_time_bin_knob("19:00", {"19:00": [_cand()]})
    picked = pick_time_bin_knob("evening_peak", {"evening_peak": [_cand()], "offpeak": [_cand(conf=0.35, mae=5.0)]})
    assert picked["conf"] == 0.15


def test_f_card_promotion_is_fail_closed_on_history_only() -> None:
    old, new = {"id": "old"}, {"id": "new"}
    ok = promote_card(
        old_card=old,
        new_card=new,
        old_mae=6.0,
        new_mae=5.5,
        old_precision=0.70,
        new_precision=0.72,
        old_time_ms=30.0,
        new_time_ms=28.0,
        compile_frames=(1, 2, 3),
        promote_frames=(4, 5),
        forbidden_frames=(100, 101),
    )
    assert ok["promoted"] is True
    assert ok["card"] is new
    assert ok["plane"] == "idle"
    overlap = promote_card(
        old_card=old,
        new_card=new,
        old_mae=6.0,
        new_mae=5.5,
        old_precision=0.70,
        new_precision=0.72,
        old_time_ms=30.0,
        new_time_ms=28.0,
        compile_frames=(1, 2, 3),
        promote_frames=(3, 4),
    )
    assert overlap == {"promoted": False, "card": old, "reason": "promote_overlaps_compile", "plane": "idle"}
    sealed = promote_card(
        old_card=old,
        new_card=new,
        old_mae=6.0,
        new_mae=5.5,
        old_precision=0.70,
        new_precision=0.72,
        old_time_ms=30.0,
        new_time_ms=28.0,
        compile_frames=(1, 2),
        promote_frames=(8,),
        forbidden_frames=(8, 9),
    )
    assert sealed["reason"] == "promote_uses_forbidden"
    assert sealed["card"] is old
    empty = idle_source_reason(compile_frames=(1,), promote_frames=())
    assert empty == "empty_promote_frames"


def test_f_card_keeps_old_when_mae_precision_or_time_regress() -> None:
    old, new = {"id": "old"}, {"id": "new"}
    worse_mae = promote_card(
        old_card=old,
        new_card=new,
        old_mae=5.0,
        new_mae=6.0,
        old_precision=0.80,
        new_precision=0.80,
        old_time_ms=30.0,
        new_time_ms=20.0,
        compile_frames=(1,),
        promote_frames=(2,),
    )
    assert worse_mae["promoted"] is False
    assert worse_mae["reason"] == "new_card_not_better"
    worse_precision = card_beats_incumbent(
        old_mae=5.0,
        new_mae=4.8,
        old_precision=0.80,
        new_precision=0.70,
        old_time_ms=30.0,
        new_time_ms=29.0,
    )
    assert worse_precision is False
    slower = card_beats_incumbent(
        old_mae=5.0,
        new_mae=4.8,
        old_precision=0.80,
        new_precision=0.80,
        old_time_ms=30.0,
        new_time_ms=31.0,
    )
    assert slower is False
    perfect_old = card_beats_incumbent(
        old_mae=0.0,
        new_mae=0.1,
        old_precision=1.0,
        new_precision=1.0,
        old_time_ms=27.0,
        new_time_ms=27.0,
    )
    assert perfect_old is False
    slight_mae_worse = card_beats_incumbent(
        old_mae=6.0,
        new_mae=6.125,
        old_precision=1.0,
        new_precision=1.0,
        old_time_ms=0.0,
        new_time_ms=0.0,
    )
    assert slight_mae_worse is False
    better_mae_small_precision_drop = card_beats_incumbent(
        old_mae=7.6875,
        new_mae=5.6875,
        old_precision=1.0,
        new_precision=0.973,
        old_time_ms=0.0,
        new_time_ms=0.0,
    )
    assert better_mae_small_precision_drop is True
    tied = card_beats_incumbent(
        old_mae=6.0,
        new_mae=6.0,
        old_precision=1.0,
        new_precision=1.0,
        old_time_ms=0.0,
        new_time_ms=0.0,
    )
    assert tied is False
    assert (
        card_not_worse_than_incumbent(old_mae=6.0, new_mae=6.0, old_precision=1.0, new_precision=1.0) is True
    )
    assert (
        card_not_worse_than_incumbent(old_mae=6.0, new_mae=6.2, old_precision=1.0, new_precision=1.0) is False
    )


def test_h1_planes_exclude_idle_and_training() -> None:
    assert "execute_F_conf" in H1_PLANES
    assert "F_card" in NOT_IN_H1
    assert "B_v1_train" in NOT_IN_H1
    assert "time_bin" in NOT_IN_H1
    runner = Path("research_tools/execution_policy/run_execution_stack.py").read_text(encoding="utf-8")
    assert "MUST_PASS_OFFICIAL" in runner
    assert "confirmatory_run" in runner
    assert "compile_view_plan" in runner
    assert "stack_proxy" in runner
    assert "apply_idle_card" in runner
    assert "force_close_hard" in runner
    assert "decide_hard_path" in runner
    assert "rehearse_view" in runner
    assert "overlay_promoted_cards" in runner
    assert "promote_card" not in runner


def test_zero_26_foci_closes_hard_path_without_changing_a_knob() -> None:
    plan = {
        "plane": "compile",
        "knob": {"model": "yolo11s", "imgsz": 640, "conf": 0.15, "mae": 4.0, "time_ms": 27.0},
        "open_hard": True,
    }
    card = parse_policy_card(
        {"regions": [{"region_id": "near", "xyxy": [0, 0, 80, 80], "model": "yolo11s", "conf": 0.25}]}
    )
    out = apply_idle_card(plan, card)
    assert out["open_hard"] is False
    assert out["knob"]["conf"] == 0.15
    assert out["hard_path_source"] == "f_card_closed"
    assert "knob_source" not in out
    closed = force_close_hard(
        {
            "plane": "compile",
            "knob": {"model": "yolo11s", "imgsz": 640, "conf": 0.15, "mae": 7.3, "time_ms": 27.0},
            "open_hard": True,
        },
        view="VIRAT_S_010204",
        views=("VIRAT_S_010204",),
    )
    assert closed["open_hard"] is False
    assert closed["knob"]["conf"] == 0.15
    assert closed["hard_path_source"] == "ablation_force_close"
    kept = force_close_hard(
        {
            "plane": "compile",
            "knob": {"model": "yolo11s", "imgsz": 640, "conf": 0.15, "mae": 4.0, "time_ms": 27.0},
            "open_hard": False,
        },
        view="VIRAT_S_010004",
        views=("VIRAT_S_010204",),
    )
    assert kept["open_hard"] is False
    assert "ablation_force_close" not in str(kept.get("hard_path_source"))


def test_stack_plan_mentions_coco_pretrain_only() -> None:
    text = Path("docs/plans/2026-08-19-camera-execution-stack-preregistration.md").read_text(encoding="utf-8")
    assert "COCO" in text
    assert "预训练" in text
    assert "不得只用 COCO 权重报 H1" in text
    assert "open_hard" not in text or "硬路径门" in text
    assert "F_card" in text
    assert "空闲重编译" in text
    assert "MAE 不得高于旧卡" in text
    assert "全部编译历史" in text
    assert "交错" in text
    assert "不得改 A 档" in text
    assert "decide_hard_path" in text
    assert "play_stack_frames" in text
    assert "clock_schedule" not in text
