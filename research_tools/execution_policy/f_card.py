"""Idle F_card replay: rewrite cards on history holdout. No GPU, no val, no H1."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from research_tools.execution_policy.card import (
    CONFS,
    PolicyCard,
    card_to_dict,
    merge_overlapping_regions,
    parse_policy_card,
)
from research_tools.execution_policy.context import _history_focus
from research_tools.execution_policy.score import compare_camera
from research_tools.execution_policy.stack import card_not_worse_than_incumbent, idle_source_reason, promote_card
from research_tools.run_dcrq_development_gate import SEALED_CAMERA

CONF_STEPS = tuple(sorted(CONFS))
TIME_BASIS = "unmeasured_equal_not_wallclock"


def frame_key(sequence: str, frame: int) -> str:
    return f"{sequence}:{int(frame)}"


def collect_keys(sources: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    return [frame_key(str(row["sequence"]), int(frame)) for row in sources for frame in row.get(field) or ()]


def assert_history_only(sources: Sequence[Mapping[str, Any]]) -> None:
    for row in sources:
        sequence = str(row["sequence"])
        camera = str(row.get("camera_id") or "")
        if SEALED_CAMERA in sequence or camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        history = {int(frame) for frame in row["history"]}
        validation = {int(frame) for frame in row.get("validation") or ()}
        if history & validation:
            raise ValueError(f"history_contains_validation:{sequence}")


def split_history_sources(sources: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sources:
        raise ValueError("no_history_sources")
    rows = [dict(row) for row in sources]
    for row in rows:
        validation = {int(frame) for frame in row.get("validation") or ()}
        row["history"] = [int(frame) for frame in row["history"] if int(frame) not in validation]
    keys = [(str(row["sequence"]), int(frame)) for row in rows for frame in row["history"]]
    keys.sort()
    if len(keys) < 2:
        raise ValueError("insufficient_history")
    compile_keys = set(keys[0::2])
    promote_keys = set(keys[1::2])

    def _subset(wanted: set[tuple[str, int]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            history = [int(frame) for frame in row["history"] if (str(row["sequence"]), int(frame)) in wanted]
            if history:
                out.append({**row, "history": history})
        return out

    compile_rows = _subset(compile_keys)
    promote_rows = _subset(promote_keys)
    if not compile_rows or not promote_rows:
        raise ValueError("insufficient_history")
    assert_history_only(compile_rows + promote_rows)
    return compile_rows, promote_rows


def _same_card(left: PolicyCard, right: PolicyCard) -> bool:
    return json.dumps(card_to_dict(left), sort_keys=True) == json.dumps(card_to_dict(right), sort_keys=True)


def rebuild_foci_card(compile_sources: Sequence[Mapping[str, Any]]) -> PolicyCard | None:
    if not compile_sources:
        return None
    focus = _history_focus(compile_sources, observer_sequence="", observer=-1)
    raw = [
        {key: row[key] for key in ("region_id", "model", "conf", "role", "xyxy")}
        for row in focus.get("proposed_regions") or ()
    ]
    if not raw:
        return None
    return parse_policy_card(merge_overlapping_regions({"regions": raw}))


def propose_idle_cards(old: PolicyCard, compile_sources: Sequence[Mapping[str, Any]]) -> list[tuple[str, PolicyCard]]:
    raw = card_to_dict(old)["regions"]
    proposed: list[tuple[str, PolicyCard]] = []

    raised = {"regions": [dict(row) for row in raw]}
    changed = False
    for row in raised["regions"]:
        if row["model"] != "yolo26" or row["role"] != "count":
            continue
        index = CONF_STEPS.index(float(row["conf"])) if float(row["conf"]) in CONF_STEPS else 1
        if index < len(CONF_STEPS) - 1:
            row["conf"] = CONF_STEPS[index + 1]
            changed = True
    if changed:
        proposed.append(("raise_26_conf", parse_policy_card(raised)))

    if any(row["model"] == "yolo26" and row["role"] == "count" for row in raw):
        dropped = {"regions": [dict(row) for row in raw]}
        for row in dropped["regions"]:
            if row["model"] == "yolo26":
                row["model"] = "yolo11s"
        proposed.append(("drop_26", parse_policy_card(dropped)))

    rebuilt = rebuild_foci_card(compile_sources)
    if rebuilt is not None:
        proposed.append(("rebuild_foci", rebuilt))

    unique: list[tuple[str, PolicyCard]] = []
    seen: set[str] = {json.dumps(card_to_dict(old), sort_keys=True)}
    for name, card in proposed:
        key = json.dumps(card_to_dict(card), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, card))
    return unique


def score_card(card: PolicyCard, promote_sources: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scored = compare_camera(card, promote_sources, frames_key="history", compiled_time_ms=0.0, b26_time_ms=1.0)
    compiled = scored["compiled"]
    return {
        "mae": float(compiled["mae"]),
        "precision": float(compiled["precision"]),
        "pred_box_count": float(compiled["pred_box_count"]),
        "truth_box_count": float(compiled["truth_box_count"]),
        "latency_proxy": float(compiled["latency_proxy"]),
    }


def replay_view(old: PolicyCard, sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assert_history_only(sources)
    try:
        compile_rows, promote_rows = split_history_sources(sources)
    except ValueError as exc:
        return {
            "promoted": False,
            "reason": str(exc),
            "candidate": None,
            "kept_card": card_to_dict(old),
            "tried": [],
        }
    compile_keys = collect_keys(compile_rows, "history")
    promote_keys = collect_keys(promote_rows, "history")
    forbidden_keys = collect_keys(sources, "validation")
    illegal = idle_source_reason(
        compile_frames=compile_keys,
        promote_frames=promote_keys,
        forbidden_frames=forbidden_keys,
    )
    old_holdout = score_card(old, promote_rows)
    old_compile = score_card(old, compile_rows)
    if illegal is not None:
        return {
            "promoted": False,
            "reason": illegal,
            "candidate": None,
            "kept_card": card_to_dict(old),
            "compile_frames": compile_keys,
            "promote_frames": promote_keys,
            "forbidden_frames": forbidden_keys,
            "old_metrics": old_holdout,
            "old_compile_metrics": old_compile,
            "tried": [],
        }
    tried: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    for name, card in propose_idle_cards(old, compile_rows):
        new_holdout = score_card(card, promote_rows)
        new_compile = score_card(card, compile_rows)
        decision = promote_card(
            old_card=old,
            new_card=card,
            old_mae=old_holdout["mae"],
            new_mae=new_holdout["mae"],
            old_precision=old_holdout["precision"],
            new_precision=new_holdout["precision"],
            old_time_ms=0.0,
            new_time_ms=0.0,
            compile_frames=compile_keys,
            promote_frames=promote_keys,
            forbidden_frames=forbidden_keys,
        )
        reason = str(decision["reason"])
        promoted = bool(decision["promoted"])
        if promoted and not card_not_worse_than_incumbent(
            old_mae=old_compile["mae"],
            new_mae=new_compile["mae"],
            old_precision=old_compile["precision"],
            new_precision=new_compile["precision"],
        ):
            promoted = False
            reason = "worse_on_compile_history"
        row = {
            "name": name,
            "promoted": promoted,
            "reason": reason,
            "card": card_to_dict(card),
            "metrics": new_holdout,
            "compile_metrics": new_compile,
        }
        tried.append(row)
        if promoted:
            winners.append(row)
    winners.sort(key=lambda row: (row["metrics"]["mae"], -row["metrics"]["precision"], row["metrics"]["latency_proxy"]))
    if not winners:
        return {
            "promoted": False,
            "reason": "no_candidate_beats_incumbent",
            "candidate": None,
            "kept_card": card_to_dict(old),
            "compile_frames": compile_keys,
            "promote_frames": promote_keys,
            "forbidden_frames": forbidden_keys,
            "old_metrics": old_holdout,
            "old_compile_metrics": old_compile,
            "tried": tried,
        }
    best = winners[0]
    return {
        "promoted": True,
        "reason": "promoted",
        "candidate": best["name"],
        "kept_card": best["card"],
        "compile_frames": compile_keys,
        "promote_frames": promote_keys,
        "forbidden_frames": forbidden_keys,
        "old_metrics": old_holdout,
        "old_compile_metrics": old_compile,
        "new_metrics": best["metrics"],
        "new_compile_metrics": best["compile_metrics"],
        "tried": tried,
    }


def overlay_promoted_cards(cards: Mapping[str, PolicyCard], payload: Mapping[str, Any]) -> dict[str, PolicyCard]:
    out = dict(cards)
    for view, row in (payload.get("per_view") or {}).items():
        if row.get("promoted") and row.get("kept_card"):
            out[str(view)] = parse_policy_card(row["kept_card"])
    return out


def replay(*, view_sources: Mapping[str, Sequence[Mapping[str, Any]]], cards: Mapping[str, PolicyCard]) -> dict[str, Any]:
    per_view: dict[str, Any] = {}
    for view, sources in sorted(view_sources.items()):
        if SEALED_CAMERA in view or any(SEALED_CAMERA in str(row["sequence"]) for row in sources):
            raise ValueError("sealed_camera_present")
        if view not in cards:
            raise ValueError(f"missing_card:{view}")
        per_view[view] = replay_view(cards[view], sources)
    return {
        "schema_version": "sceneconfig.execution-policy.f-card.v1",
        "control_plane": "v0.2.4",
        "plane": "idle",
        "history_split": "interleave_sequence_frame",
        "promotion_rule": "holdout_improve_and_compile_not_worse",
        "a_knob_locked": True,
        "allowed_rewrites": ["raise_26_conf", "drop_26", "rebuild_foci"],
        "official_h1": False,
        "evidence_status": "DEVELOPMENT_ONLY",
        "scoring": "ledger_filter_s1",
        "time_basis": TIME_BASIS,
        "val_frames_used": False,
        "sealed_camera_used": False,
        "view_count": len(per_view),
        "promoted_view_count": sum(1 for row in per_view.values() if row["promoted"]),
        "per_view": per_view,
        "h1": {"reason": "idle_f_card_not_official_h1", "passed": False},
    }
