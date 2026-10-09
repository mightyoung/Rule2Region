"""v0.2 camera stack: compile A+gate, execute A±C/D+F_conf, idle F_card."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from research_tools.execution_policy.card import PolicyCard
from research_tools.execution_policy.route_select import next_conf, pick_knob

MAE_TOLERANCE = -0.05
CLOSED_HARD_PROXY = 0.25
OPEN_HARD_EXTRA_PROXY = 0.25
H1_PLANES = ("compile_A", "execute_gated_CD", "execute_F_conf")
NOT_IN_H1 = ("F_card", "time_bin", "B_v1_train")
_CLOCK = re.compile(r"(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d")


def should_open_hard(*, b11_mae: float, b26_mae: float, foci_count: int, min_mae_gain: float = 0.05) -> bool:
    if foci_count <= 0:
        return False
    if b11_mae <= 0:
        return False
    return (b11_mae - b26_mae) / b11_mae > min_mae_gain


def hard_crop_frames(
    *,
    history_frames: Sequence[int],
    validation_frames: Sequence[int],
    foci_count: int,
    open_hard: bool | None,
    force_closed: bool,
    phase: str,
) -> tuple[int, ...]:
    if foci_count <= 0 or force_closed:
        return ()
    if phase == "history":
        return tuple(int(frame) for frame in history_frames)
    if phase == "validation":
        if open_hard is not True:
            return ()
        return tuple(int(frame) for frame in validation_frames)
    raise ValueError(f"unknown_hard_crop_phase:{phase}")


def decide_hard_path(
    *,
    foci_count: int,
    knob_model: str,
    closed_mae: float,
    opened: Mapping[str, Any],
) -> bool:
    """Open C/D only if history rehearsal of the same H1 predicate passes and MAE improves."""
    if foci_count <= 0 or knob_model != "yolo11s":
        return False
    gate = opened.get("h1") or {}
    if not bool(gate.get("passed")):
        return False
    return float(opened["mae"]) < float(closed_mae)


def pick_default_knob(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return pick_knob(candidates)


def step_online_conf(current: float, prev_count: float, history_mean: float) -> float:
    return next_conf(current, prev_count, history_mean)


def stack_proxy(*, open_hard: bool) -> float:
    return CLOSED_HARD_PROXY + (OPEN_HARD_EXTRA_PROXY if open_hard else 0.0)


def assert_not_clock_schedule(bin_id: str) -> None:
    if _CLOCK.search(bin_id):
        raise ValueError("clock_schedule_forbidden")


def compile_view_plan(
    *,
    knob_candidates: Sequence[Mapping[str, Any]],
    b11_mae: float,
    b26_mae: float,
    foci_count: int,
    time_bin: str | None = None,
) -> dict[str, Any]:
    if time_bin is not None:
        assert_not_clock_schedule(time_bin)
    knob = pick_default_knob(knob_candidates)
    plan = {
        "plane": "compile",
        "knob": {key: knob[key] for key in ("model", "imgsz", "conf", "mae", "time_ms")},
        "open_hard": should_open_hard(b11_mae=b11_mae, b26_mae=b26_mae, foci_count=foci_count),
    }
    if time_bin is not None:
        plan["time_bin"] = time_bin
    return plan


def pick_time_bin_knob(bin_id: str, candidates_by_bin: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    assert_not_clock_schedule(bin_id)
    if bin_id not in candidates_by_bin:
        raise ValueError(f"unknown_time_bin:{bin_id}")
    return pick_default_knob(candidates_by_bin[bin_id])


def idle_source_reason(
    *,
    compile_frames: Sequence[Any],
    promote_frames: Sequence[Any],
    forbidden_frames: Sequence[Any] = (),
) -> str | None:
    compile_set = set(compile_frames)
    promote_set = set(promote_frames)
    forbidden_set = set(forbidden_frames)
    if not promote_set:
        return "empty_promote_frames"
    if promote_set & compile_set:
        return "promote_overlaps_compile"
    if promote_set & forbidden_set:
        return "promote_uses_forbidden"
    return None


def card_beats_incumbent(
    *,
    old_mae: float,
    new_mae: float,
    old_precision: float,
    new_precision: float,
    old_time_ms: float,
    new_time_ms: float,
) -> bool:
    mae_ok = new_mae <= old_mae
    precision_ok = (new_precision - old_precision) >= MAE_TOLERANCE
    time_ok = new_time_ms <= old_time_ms
    improved = new_mae < old_mae or new_precision > old_precision or new_time_ms < old_time_ms
    return mae_ok and precision_ok and time_ok and improved


def card_not_worse_than_incumbent(
    *,
    old_mae: float,
    new_mae: float,
    old_precision: float,
    new_precision: float,
) -> bool:
    return new_mae <= old_mae and (new_precision - old_precision) >= MAE_TOLERANCE


def count_26_foci(card: PolicyCard) -> int:
    return sum(1 for row in card.regions if row.role == "count" and row.model == "yolo26")


def apply_idle_card(plan: Mapping[str, Any], card: PolicyCard) -> dict[str, Any]:
    out = dict(plan)
    out["knob"] = dict(plan["knob"])
    foci = count_26_foci(card)
    out["foci_count"] = foci
    if foci == 0:
        out["open_hard"] = False
        out["hard_path_source"] = "f_card_closed"
    return out


def force_close_hard(plan: Mapping[str, Any], *, view: str, views: Sequence[str]) -> dict[str, Any]:
    out = dict(plan)
    out["knob"] = dict(plan.get("knob") or {})
    if view in set(views):
        out["open_hard"] = False
        out["hard_path_source"] = "ablation_force_close"
    return out


def promote_card(
    *,
    old_card: Any,
    new_card: Any,
    old_mae: float,
    new_mae: float,
    old_precision: float,
    new_precision: float,
    old_time_ms: float,
    new_time_ms: float,
    compile_frames: Sequence[Any],
    promote_frames: Sequence[Any],
    forbidden_frames: Sequence[Any] = (),
) -> dict[str, Any]:
    reason = idle_source_reason(
        compile_frames=compile_frames,
        promote_frames=promote_frames,
        forbidden_frames=forbidden_frames,
    )
    if reason is not None:
        return {"promoted": False, "card": old_card, "reason": reason, "plane": "idle"}
    if not card_beats_incumbent(
        old_mae=old_mae,
        new_mae=new_mae,
        old_precision=old_precision,
        new_precision=new_precision,
        old_time_ms=old_time_ms,
        new_time_ms=new_time_ms,
    ):
        return {"promoted": False, "card": old_card, "reason": "new_card_not_better", "plane": "idle"}
    return {"promoted": True, "card": new_card, "reason": "promoted", "plane": "idle"}
