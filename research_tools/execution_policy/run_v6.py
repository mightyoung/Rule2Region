"""v6 development demo: conservative gate vs five policies. Not official."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.execution_policy.crop_infer import infer_card_frame, infer_full_frame, load_bgr
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.run_execution_stack import (
    FULL_CONF,
    KNOB_CONFS,
    _filter,
    _score,
    play_stack_frames,
    rehearse_view,
)
from research_tools.execution_policy.score import _arm_metrics
from research_tools.execution_policy.stack import (
    compile_view_plan,
    count_26_foci,
    decide_hard_path,
    hard_crop_frames,
    should_open_hard,
    stack_proxy,
)
from research_tools.execution_policy.v6 import (
    HISTORY_BLOCKS,
    METHODS,
    admission_cell,
    conservative_bounds,
    decide_v6_hard,
    policy_yolo26_calls,
    split_contiguous_blocks,
    summarize_method,
)
from research_tools.route_a_clip_commissioning import load_clips
from research_tools.run_dcrq_development_gate import SEALED_CAMERA


def _block_row(
    store_seq: Mapping[str, Any],
    clip: Mapping[str, Any],
    frames: Sequence[int],
    knob: Mapping[str, Any],
    card,
    *,
    dual_seq: Mapping[int, Any],
    hist_mean: float,
) -> dict[str, Any]:
    preds, time_ms = play_stack_frames(
        store_seq, list(frames), knob, card, open_hard=True, hist_mean=hist_mean
    )
    b11 = _arm_metrics(frames, {frame: dual_seq[frame].yolo11s for frame in frames}, clip["truth"])
    b26_time = sum(store_seq["full26"][frame][1] for frame in frames) / len(frames)
    scored = _score(
        preds,
        list(frames),
        clip["truth"],
        time_ms=time_ms,
        b11_mae=b11["mae"],
        b11_p=b11["precision"],
        b26_time=b26_time,
        proxy=stack_proxy(open_hard=True),
    )
    return {
        "frames": list(frames),
        "mae_change_vs_b11": float(scored["mae_change_vs_b11"]),
        "precision_change_vs_b11": float(scored["precision_change_vs_b11"]),
        "time_ms": float(scored["time_ms_mean"]),
        "b26_time_ms": float(b26_time),
        "h1": scored["h1"],
    }


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    dual_ledger_path: Path,
    e4_result: Path,
    output_dir: Path,
    yolo11s: Path,
    yolo26: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    from ultralytics import YOLO  # type: ignore

    split_path = pilot_root / "split-manifest.json"
    if not split_path.is_file():
        raise ValueError("v6_requires_sealed_split")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    seen_confirmatory = split.get("schema_version") == "fieldpack.virat-h1-confirmatory-split.v1"
    clips = load_clips(cache_root, pilot_root)
    if any(SEALED_CAMERA in sequence or str(clip["camera_id"]) == SEALED_CAMERA for sequence, clip in clips.items()):
        raise ValueError("sealed_camera_present")
    dual = load_dual_ledger(dual_ledger_path)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    models = {"yolo11s": YOLO(str(yolo11s)), "yolo26": YOLO(str(yolo26))}
    first = next(iter(clips.values()))
    warm = int(first["history_frames"][0])
    warm_bgr = load_bgr(pilot_root / "sequences" / first["sequence_id"] / "img1" / f"{warm:06d}.jpg")
    infer_full_frame(models["yolo11s"], warm_bgr, warm, conf=FULL_CONF, source="yolo11s", device=device, imgsz=640)
    infer_full_frame(models["yolo26"], warm_bgr, warm, conf=FULL_CONF, source="yolo26", device=device, imgsz=640)

    by_view: dict[str, list[str]] = defaultdict(list)
    for sequence, clip in clips.items():
        by_view[view_id(str(clip["camera_id"]))].append(sequence)

    store: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        hist = list(clip["history_frames"])
        val = list(clip["validation_frames"])
        frames = hist + val
        foci_count = count_26_foci(card)
        history_hard = hard_crop_frames(
            history_frames=hist,
            validation_frames=val,
            foci_count=foci_count,
            open_hard=None,
            force_closed=False,
            phase="history",
        )
        # Counterfactual: opened-arm future scores need val crops even if a policy stays closed.
        val_hard = hard_crop_frames(
            history_frames=hist,
            validation_frames=val,
            foci_count=foci_count,
            open_hard=True,
            force_closed=False,
            phase="validation",
        )
        full11: dict[int, tuple] = {}
        full26: dict[int, tuple] = {}
        hard: dict[int, tuple] = {}
        for frame in frames:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            p11, t11 = infer_full_frame(models["yolo11s"], bgr, frame, conf=FULL_CONF, source="yolo11s", device=device, imgsz=640)
            p26, t26 = infer_full_frame(models["yolo26"], bgr, frame, conf=FULL_CONF, source="yolo26", device=device, imgsz=640)
            full11[frame] = (p11, t11)
            full26[frame] = (p26, t26)
            hard[frame] = ((), 0.0)
            if frame in history_hard or frame in val_hard:
                part, th = infer_card_frame(models, card, bgr, frame, device=device, imgsz_mode="adaptive")
                hard[frame] = (part.yolo26, th)
        b11_h = _arm_metrics(hist, {frame: dual[sequence][frame].yolo11s for frame in hist}, clip["truth"])
        b26_h = _arm_metrics(hist, {frame: dual[sequence][frame].yolo26 for frame in hist}, clip["truth"])
        knobs = []
        for conf in KNOB_CONFS:
            scored = _arm_metrics(hist, {frame: _filter(full11[frame][0], conf) for frame in hist}, clip["truth"])
            knobs.append(
                {
                    "model": "yolo11s",
                    "imgsz": 640,
                    "conf": conf,
                    "mae": scored["mae"],
                    "precision": scored["precision"],
                    "time_ms": sum(full11[frame][1] for frame in hist) / len(hist),
                    "b11_mae": b11_h["mae"],
                    "b11_precision": b11_h["precision"],
                    "b26_time_ms": sum(full26[frame][1] for frame in hist) / len(hist),
                    "proxy": 0.25,
                }
            )
        for conf in KNOB_CONFS:
            scored = _arm_metrics(hist, {frame: _filter(full26[frame][0], conf) for frame in hist}, clip["truth"])
            knobs.append(
                {
                    "model": "yolo26",
                    "imgsz": 640,
                    "conf": conf,
                    "mae": scored["mae"],
                    "precision": scored["precision"],
                    "time_ms": sum(full26[frame][1] for frame in hist) / len(hist),
                    "b11_mae": b11_h["mae"],
                    "b11_precision": b11_h["precision"],
                    "b26_time_ms": sum(full26[frame][1] for frame in hist) / len(hist),
                    "proxy": 1.0,
                }
            )
        store[sequence] = {
            "full11": full11,
            "full26": full26,
            "hard": hard,
            "b11_h": b11_h,
            "b26_h": b26_h,
            "knobs": knobs,
            "foci": foci_count,
        }

    view_plan: dict[str, Any] = {}
    for view, sequences in by_view.items():
        knobs_pool: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        b11s, b26s, foci = [], [], []
        for sequence in sequences:
            for row in store[sequence]["knobs"]:
                knobs_pool[(row["model"], row["conf"])].append(row)
            b11s.append(store[sequence]["b11_h"]["mae"])
            b26s.append(store[sequence]["b26_h"]["mae"])
            foci.append(store[sequence]["foci"])
        candidates = []
        for key, rows in knobs_pool.items():
            candidates.append(
                {
                    "model": key[0],
                    "imgsz": 640,
                    "conf": key[1],
                    "mae": sum(row["mae"] for row in rows) / len(rows),
                    "precision": sum(row["precision"] for row in rows) / len(rows),
                    "time_ms": sum(row["time_ms"] for row in rows) / len(rows),
                    "b11_mae": sum(row["b11_mae"] for row in rows) / len(rows),
                    "b11_precision": sum(row["b11_precision"] for row in rows) / len(rows),
                    "b26_time_ms": sum(row["b26_time_ms"] for row in rows) / len(rows),
                    "proxy": rows[0]["proxy"],
                }
            )
        plan = compile_view_plan(
            knob_candidates=candidates,
            b11_mae=sum(b11s) / len(b11s),
            b26_mae=sum(b26s) / len(b26s),
            foci_count=max(foci),
        )
        closed_h = rehearse_view(sequences, store, clips, plan["knob"], cards[view], open_hard=False)
        opened_h = rehearse_view(sequences, store, clips, plan["knob"], cards[view], open_hard=True)
        foci_count = int(max(foci))
        knob_model = str(plan["knob"]["model"])
        v5_open = decide_hard_path(
            foci_count=foci_count,
            knob_model=knob_model,
            closed_mae=float(closed_h["mae"]),
            opened=opened_h,
        )
        mae_gate = should_open_hard(
            b11_mae=sum(b11s) / len(b11s),
            b26_mae=sum(b26s) / len(b26s),
            foci_count=foci_count,
        )
        block_rows: list[dict[str, Any]] = []
        for sequence in sequences:
            hist = list(clips[sequence]["history_frames"])
            hist_mean = store[sequence]["b11_h"]["pred_box_count"] / max(1, len(hist))
            for chunk in split_contiguous_blocks(hist, n=HISTORY_BLOCKS):
                block_rows.append(
                    _block_row(
                        store[sequence],
                        clips[sequence],
                        chunk,
                        plan["knob"],
                        cards[view],
                        dual_seq=dual[sequence],
                        hist_mean=hist_mean,
                    )
                )
        bounds = conservative_bounds(block_rows)
        cons = bool(bounds["passed"])
        v6_open = decide_v6_hard(
            point_open=v5_open,
            conservative=cons,
            foci_count=foci_count,
            knob_model=knob_model,
        )
        can_open = foci_count > 0 and knob_model == "yolo11s"
        view_plan[view] = {
            "knob": plan["knob"],
            "foci_count": foci_count,
            "mae_gate": mae_gate,
            "methods": {
                "always_closed": False,
                "always_open": can_open,
                "mae_only": mae_gate,
                "v5_point": v5_open,
                "v6_conservative": v6_open,
            },
            "conservative": cons,
            "conservative_bounds": bounds,
            "history_rehearsal": {"closed": closed_h, "opened": opened_h},
            "blocks": block_rows,
            "history_blocks": HISTORY_BLOCKS,
        }

    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        plan = view_plan[view]
        knob = plan["knob"]
        hist, val = list(clip["history_frames"]), list(clip["validation_frames"])
        truth = clip["truth"]
        b11_v = _arm_metrics(val, {frame: dual[sequence][frame].yolo11s for frame in val}, truth)
        b26_time = sum(store[sequence]["full26"][frame][1] for frame in val) / len(val)
        hist_mean = store[sequence]["b11_h"]["pred_box_count"] / max(1, len(hist))
        closed_preds, closed_ms = play_stack_frames(
            store[sequence], val, knob, card, open_hard=False, hist_mean=hist_mean
        )
        opened_preds, opened_ms = play_stack_frames(
            store[sequence], val, knob, card, open_hard=True, hist_mean=hist_mean
        )
        closed_s = _score(
            closed_preds, val, truth, time_ms=closed_ms, b11_mae=b11_v["mae"], b11_p=b11_v["precision"],
            b26_time=b26_time, proxy=stack_proxy(open_hard=False),
        )
        opened_s = _score(
            opened_preds, val, truth, time_ms=opened_ms, b11_mae=b11_v["mae"], b11_p=b11_v["precision"],
            b26_time=b26_time, proxy=stack_proxy(open_hard=True),
        )
        future_open_pass = bool(opened_s["h1"]["passed"])
        oracle_open = future_open_pass and plan["foci_count"] > 0 and plan["knob"]["model"] == "yolo11s"
        cells = {}
        executed = {}
        for name in METHODS:
            opened = bool(plan["methods"][name])
            cell = admission_cell(opened=opened, future_opened_pass=future_open_pass)
            cells[name] = cell
            arm = opened_s if opened else closed_s
            executed[name] = {
                "open_hard": opened,
                "cell": cell,
                "verdict": arm["h1"]["verdict"],
                "mae": arm["mae"],
                "mae_change_vs_b11": arm["mae_change_vs_b11"],
                "precision_change_vs_b11": arm["precision_change_vs_b11"],
                "time_ms_mean": arm["time_ms_mean"],
                "yolo26_call_count": policy_yolo26_calls(
                    frame_count=len(val),
                    knob_model=str(plan["knob"]["model"]),
                    opened=opened,
                    foci_count=int(plan["foci_count"]),
                ),
            }
        per_clip[sequence] = {
            "view_id": view,
            "validation_frame_count": len(val),
            "closed": {"h1": closed_s["h1"], "mae": closed_s["mae"], "time_ms_mean": closed_s["time_ms_mean"]},
            "opened": {"h1": opened_s["h1"], "mae": opened_s["mae"], "time_ms_mean": opened_s["time_ms_mean"]},
            "oracle_open": oracle_open,
            "diagnostic_clip_cells": cells,
            "methods": executed,
        }

    view_outcomes: dict[str, Any] = {}
    method_view_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in METHODS}
    for view, sequences in sorted(by_view.items()):
        plan = view_plan[view]
        future_opened_pass = all(bool(per_clip[sequence]["opened"]["h1"]["passed"]) for sequence in sequences)
        future_closed_pass = all(bool(per_clip[sequence]["closed"]["h1"]["passed"]) for sequence in sequences)
        total_frames = sum(int(per_clip[sequence]["validation_frame_count"]) for sequence in sequences)
        methods: dict[str, Any] = {}
        for name in METHODS:
            opened = bool(plan["methods"][name])
            future_h1_passed = future_opened_pass if opened else future_closed_pass
            weighted_time = sum(
                float(per_clip[sequence]["methods"][name]["time_ms_mean"])
                * int(per_clip[sequence]["validation_frame_count"])
                for sequence in sequences
            ) / max(1, total_frames)
            row = {
                "open_hard": opened,
                "cell": admission_cell(opened=opened, future_opened_pass=future_opened_pass),
                "future_h1_passed": future_h1_passed,
                "policy_model_time_ms_mean": weighted_time,
                "yolo26_call_count": sum(
                    int(per_clip[sequence]["methods"][name]["yolo26_call_count"]) for sequence in sequences
                ),
            }
            methods[name] = row
            method_view_rows[name].append(row)
        view_outcomes[view] = {
            "sequences": sorted(sequences),
            "validation_frame_count": total_frames,
            "future_opened_all_clips_pass": future_opened_pass,
            "future_closed_all_clips_pass": future_closed_pass,
            "methods": methods,
        }

    summary = {name: summarize_method(method_view_rows[name]) for name in METHODS}
    formal_eligibility = {
        "eligible": False,
        "unseen_future_pack": not seen_confirmatory,
        "minimum_independent_views": 8,
        "minimum_independent_views_met": len(view_plan) >= 8,
        "runner_is_development_only": True,
        "reason": "seen_confirmatory_validation" if seen_confirmatory else "runner_development_only",
    }
    result = {
        "schema_version": "sceneconfig.execution-policy.v6.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "official_h1": False,
        "v6_official": False,
        "seen_confirmatory_validation": seen_confirmatory,
        "reason": "seen_confirmatory_validation" if seen_confirmatory else "v6_development_demo",
        "decision_unit": "view_id",
        "future_view_rule": "all_clips_pass",
        "metric_time_scope": "MODEL_CALLS_ONLY",
        "input_split_schema": split.get("schema_version"),
        "history_blocks": HISTORY_BLOCKS,
        "methods": list(METHODS),
        "independent_views": sorted(view_plan),
        "view_count": len(view_plan),
        "clip_count": len(per_clip),
        "sealed_camera_used": False,
        "theoretical_guarantee": False,
        "formal_eligibility": formal_eligibility,
        "view_plan": view_plan,
        "view_outcomes": view_outcomes,
        "per_clip": per_clip,
        "cells": summary,
        "h1": {"passed": False, "reason": "development_only", "official": False},
    }
    result["h1"]["reason"] = result["reason"]
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--dual-ledger", type=Path, required=True)
    parser.add_argument("--e4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        dual_ledger_path=args.dual_ledger,
        e4_result=args.e4_result,
        output_dir=args.output_dir,
        yolo11s=args.yolo11s,
        yolo26=args.yolo26,
        device=args.device,
    )
    print(json.dumps({
        "v6_official": False,
        "seen_confirmatory_validation": result["seen_confirmatory_validation"],
        "cells": result["cells"],
        "view_methods": {view: row["methods"] for view, row in result["view_plan"].items()},
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
