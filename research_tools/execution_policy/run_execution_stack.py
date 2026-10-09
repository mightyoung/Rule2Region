"""Run the v0.2 hot path: compile A+gate, execute A±C/D+F_conf. No F_card, no B train."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.compile import h1_pass
from research_tools.execution_policy.crop_infer import exclusive_mix, infer_card_frame, infer_full_frame, load_bgr
from research_tools.execution_policy.f_card import overlay_promoted_cards
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.score import _arm_metrics
from research_tools.execution_policy.stack import (
    H1_PLANES,
    NOT_IN_H1,
    apply_idle_card,
    compile_view_plan,
    count_26_foci,
    decide_hard_path,
    force_close_hard,
    hard_crop_frames,
    should_open_hard,
    stack_proxy,
    step_online_conf,
)
from research_tools.residual_compiler import PRIMARY_EVAL

MUST_PASS_OFFICIAL = (
    "VIRAT_S_010004_01_000163_000188",
    "VIRAT_S_010004_07_000729_000809",
    "VIRAT_S_010204_03_000606_000632",
    "VIRAT_S_010204_09_001285_001336",
)
from research_tools.route_a_clip_commissioning import load_clips


FULL_CONF = 0.15
KNOB_CONFS = (0.15, 0.25, 0.35)


def _filter(preds: tuple[Prediction, ...], conf: float) -> tuple[Prediction, ...]:
    return tuple(row for row in preds if row.confidence >= conf)


def play_stack_frames(
    store_seq: Mapping[str, Any],
    frames: Sequence[int],
    knob: Mapping[str, Any],
    card,
    *,
    open_hard: bool,
    hist_mean: float,
) -> tuple[dict[int, tuple[Prediction, ...]], float]:
    src = store_seq["full11" if knob["model"] == "yolo11s" else "full26"]
    conf = float(knob["conf"])
    prev = hist_mean
    out: dict[int, tuple[Prediction, ...]] = {}
    times: list[float] = []
    for frame in frames:
        conf = step_online_conf(conf, prev, hist_mean)
        glance = _filter(src[frame][0], conf)
        ms = src[frame][1]
        if open_hard and knob["model"] == "yolo11s":
            mixed = exclusive_mix(glance, _filter(store_seq["hard"][frame][0], conf), card.regions)
            out[frame] = mixed.for_metrics()
            ms += store_seq["hard"][frame][1]
        else:
            out[frame] = glance
        times.append(ms)
        prev = float(len(out[frame]))
    return out, (sum(times) / len(times) if times else 0.0)


def rehearse_view(
    sequences: Sequence[str],
    store: Mapping[str, Any],
    clips: Mapping[str, Any],
    knob: Mapping[str, Any],
    card,
    *,
    open_hard: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        hist = list(clips[sequence]["history_frames"])
        hist_mean = store[sequence]["b11_h"]["pred_box_count"] / max(1, len(hist))
        preds, time_ms = play_stack_frames(
            store[sequence], hist, knob, card, open_hard=open_hard, hist_mean=hist_mean
        )
        b26_time = sum(store[sequence]["full26"][f][1] for f in hist) / len(hist)
        rows.append(
            _score(
                preds,
                hist,
                clips[sequence]["truth"],
                time_ms=time_ms,
                b11_mae=store[sequence]["b11_h"]["mae"],
                b11_p=store[sequence]["b11_h"]["precision"],
                b26_time=b26_time,
                proxy=stack_proxy(open_hard=open_hard),
            )
        )
    n = max(1, len(rows))
    mae = sum(row["mae"] for row in rows) / n
    precision = sum(row["precision"] for row in rows) / n
    time_ms = sum(row["time_ms_mean"] for row in rows) / n
    b11_mae = sum(store[sequence]["b11_h"]["mae"] for sequence in sequences) / n
    b11_p = sum(store[sequence]["b11_h"]["precision"] for sequence in sequences) / n
    b26_time = sum(sum(store[sequence]["full26"][f][1] for f in clips[sequence]["history_frames"]) / len(clips[sequence]["history_frames"]) for sequence in sequences) / n
    gate = h1_pass(
        compiled_mae=mae,
        b11_mae=b11_mae,
        precision_change_vs_b11=precision - b11_p,
        compiled_proxy=stack_proxy(open_hard=open_hard),
        compiled_time_ms=time_ms,
        b26_time_ms=b26_time,
    )
    return {"mae": mae, "precision": precision, "time_ms_mean": time_ms, "h1": gate}


def _score(preds, frames, truth, *, time_ms: float, b11_mae: float, b11_p: float, b26_time: float, proxy: float) -> dict[str, Any]:
    metrics = _arm_metrics(frames, preds, truth)
    gate = h1_pass(
        compiled_mae=metrics["mae"],
        b11_mae=b11_mae,
        precision_change_vs_b11=metrics["precision"] - b11_p,
        compiled_proxy=proxy,
        compiled_time_ms=time_ms,
        b26_time_ms=b26_time,
    )
    return {**metrics, "time_ms_mean": time_ms, "h1": gate, "mae_change_vs_b11": gate["mae_change_vs_b11"], "precision_change_vs_b11": metrics["precision"] - b11_p}


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
    f_card_result: Path | None = None,
    close_hard_views: Sequence[str] = (),
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    from ultralytics import YOLO  # type: ignore

    clips = load_clips(cache_root, pilot_root)
    dual = load_dual_ledger(dual_ledger_path)
    closed = tuple(str(view) for view in close_hard_views)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    f_card_payload = None
    if f_card_result is not None:
        f_card_payload = json.loads(f_card_result.read_text(encoding="utf-8"))
        if f_card_payload.get("val_frames_used") or f_card_payload.get("sealed_camera_used"):
            raise ValueError("f_card_used_forbidden_split")
        cards = overlay_promoted_cards(cards, f_card_payload)
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
        full11: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        full26: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        hard: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        foci_count = count_26_foci(card)
        history_hard = hard_crop_frames(
            history_frames=hist,
            validation_frames=val,
            foci_count=foci_count,
            open_hard=None,
            force_closed=view in closed,
            phase="history",
        )
        for frame in frames:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            p11, t11 = infer_full_frame(models["yolo11s"], bgr, frame, conf=FULL_CONF, source="yolo11s", device=device, imgsz=640)
            p26, t26 = infer_full_frame(models["yolo26"], bgr, frame, conf=FULL_CONF, source="yolo26", device=device, imgsz=640)
            full11[frame] = (p11, t11)
            full26[frame] = (p26, t26)
            hard[frame] = ((), 0.0)
            if frame in history_hard:
                part, th = infer_card_frame(models, card, bgr, frame, device=device, imgsz_mode="adaptive")
                hard[frame] = (part.yolo26, th)
        b11_h = _arm_metrics(hist, {f: dual[sequence][f].yolo11s for f in hist}, clip["truth"])
        b26_h = _arm_metrics(hist, {f: dual[sequence][f].yolo26 for f in hist}, clip["truth"])
        knobs = []
        for conf in KNOB_CONFS:
            scored = _arm_metrics(hist, {f: _filter(full11[f][0], conf) for f in hist}, clip["truth"])
            knobs.append(
                {
                    "model": "yolo11s",
                    "imgsz": 640,
                    "conf": conf,
                    "mae": scored["mae"],
                    "precision": scored["precision"],
                    "time_ms": sum(full11[f][1] for f in hist) / len(hist),
                    "b11_mae": b11_h["mae"],
                    "b11_precision": b11_h["precision"],
                    "b26_time_ms": sum(full26[f][1] for f in hist) / len(hist),
                    "proxy": 0.25,
                }
            )
        for conf in KNOB_CONFS:
            scored = _arm_metrics(hist, {f: _filter(full26[f][0], conf) for f in hist}, clip["truth"])
            knobs.append(
                {
                    "model": "yolo26",
                    "imgsz": 640,
                    "conf": conf,
                    "mae": scored["mae"],
                    "precision": scored["precision"],
                    "time_ms": sum(full26[f][1] for f in hist) / len(hist),
                    "b11_mae": b11_h["mae"],
                    "b11_precision": b11_h["precision"],
                    "b26_time_ms": sum(full26[f][1] for f in hist) / len(hist),
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
                    "mae": sum(r["mae"] for r in rows) / len(rows),
                    "precision": sum(r["precision"] for r in rows) / len(rows),
                    "time_ms": sum(r["time_ms"] for r in rows) / len(rows),
                    "b11_mae": sum(r["b11_mae"] for r in rows) / len(rows),
                    "b11_precision": sum(r["b11_precision"] for r in rows) / len(rows),
                    "b26_time_ms": sum(r["b26_time_ms"] for r in rows) / len(rows),
                    "proxy": rows[0]["proxy"],
                }
            )
        plan = apply_idle_card(
            compile_view_plan(
                knob_candidates=candidates,
                b11_mae=sum(b11s) / len(b11s),
                b26_mae=sum(b26s) / len(b26s),
                foci_count=max(foci),
            ),
            cards[view],
        )
        closed_h = rehearse_view(sequences, store, clips, plan["knob"], cards[view], open_hard=False)
        opened_h = rehearse_view(sequences, store, clips, plan["knob"], cards[view], open_hard=True)
        plan["open_hard"] = decide_hard_path(
            foci_count=int(plan.get("foci_count") or max(foci)),
            knob_model=str(plan["knob"]["model"]),
            closed_mae=float(closed_h["mae"]),
            opened=opened_h,
        )
        plan["mae_gate"] = should_open_hard(
            b11_mae=sum(b11s) / len(b11s),
            b26_mae=sum(b26s) / len(b26s),
            foci_count=max(foci),
        )
        plan["history_rehearsal"] = {"closed": closed_h, "opened": opened_h}
        view_plan[view] = force_close_hard(plan, view=view, views=closed)

    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        plan = view_plan[view]
        val_hard = hard_crop_frames(
            history_frames=list(clip["history_frames"]),
            validation_frames=list(clip["validation_frames"]),
            foci_count=int(store[sequence]["foci"]),
            open_hard=bool(plan["open_hard"]),
            force_closed=view in closed,
            phase="validation",
        )
        for frame in val_hard:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            part, th = infer_card_frame(models, card, bgr, frame, device=device, imgsz_mode="adaptive")
            store[sequence]["hard"][frame] = (part.yolo26, th)

    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        plan = view_plan[view]
        knob = plan["knob"]
        hist, val = list(clip["history_frames"]), list(clip["validation_frames"])
        truth = clip["truth"]
        b11_v = _arm_metrics(val, {f: dual[sequence][f].yolo11s for f in val}, truth)
        b26_time = sum(store[sequence]["full26"][f][1] for f in val) / len(val)
        hist_mean = store[sequence]["b11_h"]["pred_box_count"] / max(1, len(hist))
        stack, time_ms = play_stack_frames(
            store[sequence], val, knob, card, open_hard=bool(plan["open_hard"]), hist_mean=hist_mean
        )
        scored = _score(
            stack,
            val,
            truth,
            time_ms=time_ms,
            b11_mae=b11_v["mae"],
            b11_p=b11_v["precision"],
            b26_time=b26_time,
            proxy=stack_proxy(open_hard=bool(plan["open_hard"])),
        )
        per_clip[sequence] = {
            "view_id": view,
            "open_hard": plan["open_hard"],
            "knob": knob,
            "validation": scored,
            "verdict": scored["h1"]["verdict"],
        }

    primary = [PRIMARY_EVAL["VIRAT_S_010004"], PRIMARY_EVAL["VIRAT_S_010204"]]
    split_path = pilot_root / "split-manifest.json"
    confirmatory_pack = split_path.is_file()
    confirmatory_run = confirmatory_pack and f_card_result is None and not closed
    must_pass_ok = confirmatory_run and all(
        seq in per_clip and per_clip[seq]["verdict"] == "pass" for seq in MUST_PASS_OFFICIAL
    )
    official_h1 = bool(must_pass_ok)
    if not confirmatory_run:
        h1_reason = "development_stack_not_official_h1"
    elif official_h1:
        h1_reason = "confirmatory_must_pass_all_passed"
    else:
        h1_reason = "confirmatory_must_pass_failed_or_incomplete"
    result = {
        "schema_version": "sceneconfig.execution-policy.stack.v0",
        "control_plane": "v0.2",
        "h1_planes": list(H1_PLANES),
        "not_in_h1": list(NOT_IN_H1),
        "evidence_status": "OFFICIAL_H1" if official_h1 else "DEVELOPMENT_ONLY",
        "official_h1": official_h1,
        "confirmatory_pack": confirmatory_pack,
        "confirmatory_run": confirmatory_run,
        "must_pass": list(MUST_PASS_OFFICIAL),
        "must_pass_all_passed": must_pass_ok,
        "hard_path": "yolo26_adaptive_crop_v0_not_dynamicdet",
        "hard_gate": "history_h1_rehearsal",
        "f_card_result": None if f_card_result is None else str(f_card_result),
        "f_card_promoted_views": sorted(
            str(view) for view, row in (f_card_payload or {}).get("per_view", {}).items() if row.get("promoted")
        ),
        "ablation": "force_close_hard" if closed else None,
        "close_hard_views": list(closed),
        "view_plan": view_plan,
        "clip_count": len(per_clip),
        "pass_count": sum(row["verdict"] == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": primary,
            "primary_both_pass": all(per_clip[seq]["verdict"] == "pass" for seq in primary),
            "reason": h1_reason,
        },
        "per_clip": per_clip,
    }
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
    parser.add_argument("--f-card-result", type=Path, default=None)
    parser.add_argument("--close-hard-views", default="", help="comma-separated view ids to force-close")
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
        f_card_result=args.f_card_result,
        close_hard_views=tuple(item.strip() for item in args.close_hard_views.split(",") if item.strip()),
    )
    print(json.dumps({
        "official_h1": result["official_h1"],
        "confirmatory_run": result.get("confirmatory_run"),
        "pass_count": result["pass_count"],
        "must_pass_all_passed": result.get("must_pass_all_passed"),
        "h1": result["h1"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
