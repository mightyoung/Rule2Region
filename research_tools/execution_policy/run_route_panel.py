"""Run A/B/C/D/F arms on frozen E4 cards with measured wall-clock time."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.card import PolicyCard, latency_proxy
from research_tools.execution_policy.compile import h1_pass
from research_tools.execution_policy.crop_infer import (
    IMGSZ,
    exclusive_mix,
    infer_card_frame,
    infer_full_frame,
    infer_mosaic_frame,
    load_bgr,
)
from research_tools.execution_policy.route_select import next_conf, pick_knob
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.score import _arm_metrics
from research_tools.residual_compiler import PRIMARY_EVAL
from research_tools.route_a_clip_commissioning import load_clips


FULL_CONF = 0.15
KNOB_CONFS = (0.15, 0.25, 0.35)
KNOB_IMGSZ = (320, 640)


def _filter_conf(preds: tuple[Prediction, ...], conf: float) -> tuple[Prediction, ...]:
    return tuple(row for row in preds if row.confidence >= conf)


def _score(
    preds: dict[int, tuple[Prediction, ...]],
    frames: list[int],
    truth,
    *,
    time_ms: float,
    b11_mae: float,
    b11_precision: float,
    b26_time_ms: float,
    proxy: float,
) -> dict[str, Any]:
    metrics = _arm_metrics(frames, preds, truth)
    gate = h1_pass(
        compiled_mae=metrics["mae"],
        b11_mae=b11_mae,
        precision_change_vs_b11=metrics["precision"] - b11_precision,
        compiled_proxy=proxy,
        compiled_time_ms=time_ms,
        b26_time_ms=b26_time_ms,
    )
    return {
        **metrics,
        "time_ms_mean": time_ms,
        "latency_proxy": proxy,
        "h1": gate,
        "mae_change_vs_b11": gate["mae_change_vs_b11"],
        "precision_change_vs_b11": metrics["precision"] - b11_precision,
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

    clips = load_clips(cache_root, pilot_root)
    dual = load_dual_ledger(dual_ledger_path)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    models = {"yolo11s": YOLO(str(yolo11s)), "yolo26": YOLO(str(yolo26))}
    first = next(iter(clips.values()))
    warm = int(first["history_frames"][0])
    warm_bgr = load_bgr(pilot_root / "sequences" / first["sequence_id"] / "img1" / f"{warm:06d}.jpg")
    infer_full_frame(models["yolo11s"], warm_bgr, warm, conf=FULL_CONF, source="yolo11s", device=device, imgsz=320)
    infer_full_frame(models["yolo26"], warm_bgr, warm, conf=FULL_CONF, source="yolo26", device=device, imgsz=640)

    by_view: dict[str, list[str]] = defaultdict(list)
    for sequence, clip in sorted(clips.items()):
        by_view[view_id(str(clip["camera_id"]))].append(sequence)

    full_store: dict[tuple[str, int, str, int], tuple[tuple[Prediction, ...], float]] = {}
    per_clip: dict[str, Any] = {}

    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards.get(view)
        if card is None:
            per_clip[sequence] = {"view_id": view, "skipped": "no_card"}
            continue
        ledgers = dual[sequence]
        frames_h = list(clip["history_frames"])
        frames_v = list(clip["validation_frames"])
        all_frames = frames_h + frames_v
        full: dict[tuple[str, int], list[tuple[int, tuple[Prediction, ...], float]]] = {
            ("yolo11s", 320): [],
            ("yolo11s", 640): [],
            ("yolo26", 320): [],
            ("yolo26", 640): [],
        }
        mosaic: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        crop_fix: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        crop_adp: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        crop_26_only: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        for frame in all_frames:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            for name, imgsz in (("yolo11s", 320), ("yolo11s", 640), ("yolo26", 320), ("yolo26", 640)):
                preds, ms = infer_full_frame(
                    models[name], bgr, frame, conf=FULL_CONF, source=name, device=device, imgsz=imgsz
                )
                full[(name, imgsz)].append((frame, preds, ms))
                full_store[(sequence, frame, name, imgsz)] = (preds, ms)
            mosaic_part, mosaic_ms = infer_mosaic_frame(
                models["yolo26"], card, bgr, frame, device=device, conf=0.25
            )
            mosaic[frame] = (mosaic_part.for_metrics(), mosaic_ms)
            fix_part, fix_ms = infer_card_frame(models, card, bgr, frame, device=device, imgsz_mode="fixed")
            crop_fix[frame] = (fix_part.for_metrics(), fix_ms)
            adp_part, adp_ms = infer_card_frame(models, card, bgr, frame, device=device, imgsz_mode="adaptive")
            crop_adp[frame] = (adp_part.for_metrics(), adp_ms)
            crop_26_only[frame] = (adp_part.yolo26, adp_ms)

        def mean_time(items: list[tuple[int, tuple[Prediction, ...], float]], wanted: list[int]) -> float:
            rows = [ms for frame, _preds, ms in items if frame in set(wanted)]
            return sum(rows) / len(rows)

        def preds_at(items: list[tuple[int, tuple[Prediction, ...], float]], wanted: list[int], conf: float) -> dict[int, tuple[Prediction, ...]]:
            return {frame: _filter_conf(preds, conf) for frame, preds, _ms in items if frame in set(wanted)}

        def dict_time(store: dict[int, tuple[tuple[Prediction, ...], float]], wanted: list[int]) -> float:
            rows = [store[frame][1] for frame in wanted]
            return sum(rows) / len(rows)

        def dict_preds(store: dict[int, tuple[tuple[Prediction, ...], float]], wanted: list[int]) -> dict[int, tuple[Prediction, ...]]:
            return {frame: store[frame][0] for frame in wanted}

        b11_h = _arm_metrics(frames_h, {f: ledgers[f].yolo11s for f in frames_h}, clip["truth"])
        b11_v = _arm_metrics(frames_v, {f: ledgers[f].yolo11s for f in frames_v}, clip["truth"])
        b26_time_h = mean_time(full[("yolo26", 640)], frames_h)
        b26_time_v = mean_time(full[("yolo26", 640)], frames_v)
        proxy = latency_proxy(card)

        knobs = []
        for model, imgsz in (("yolo11s", 320), ("yolo11s", 640), ("yolo26", 320), ("yolo26", 640)):
            for conf in KNOB_CONFS:
                scored = _arm_metrics(frames_h, preds_at(full[(model, imgsz)], frames_h, conf), clip["truth"])
                knobs.append(
                    {
                        "model": model,
                        "imgsz": imgsz,
                        "conf": conf,
                        "mae": scored["mae"],
                        "precision": scored["precision"],
                        "time_ms": mean_time(full[(model, imgsz)], frames_h),
                        "b11_mae": b11_h["mae"],
                        "b11_precision": b11_h["precision"],
                        "b26_time_ms": b26_time_h,
                        "proxy": 1.0 if model == "yolo26" else 0.25,
                    }
                )
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "view_id": view,
            "knobs_history": knobs,
            "b11_history": b11_h,
            "b11_val": b11_v,
            "b26_time_history": b26_time_h,
            "b26_time_val": b26_time_v,
            "proxy": proxy,
            "card_conf": 0.25,
            "stores": {
                "full": {f"{model}_{imgsz}": [(frame, len(preds), ms) for frame, preds, ms in rows] for (model, imgsz), rows in full.items()},
            },
            "mosaic": mosaic,
            "crop_fix": crop_fix,
            "crop_adp": crop_adp,
            "crop_26_only": crop_26_only,
            "full_11s_640": {frame: preds for frame, preds, _ms in full[("yolo11s", 640)]},
            "full_26_640": {frame: preds for frame, preds, _ms in full[("yolo26", 640)]},
            "full_11s_640_ms": {frame: ms for frame, _preds, ms in full[("yolo11s", 640)]},
        }

    view_knob: dict[str, dict[str, Any]] = {}
    for view, sequences in by_view.items():
        pooled: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
        for sequence in sequences:
            for row in per_clip[sequence]["knobs_history"]:
                pooled[(row["model"], row["imgsz"], row["conf"])].append(row)
        candidates = []
        for key, rows in pooled.items():
            candidates.append(
                {
                    "model": key[0],
                    "imgsz": key[1],
                    "conf": key[2],
                    "mae": sum(row["mae"] for row in rows) / len(rows),
                    "precision": sum(row["precision"] for row in rows) / len(rows),
                    "time_ms": sum(row["time_ms"] for row in rows) / len(rows),
                    "b11_mae": sum(row["b11_mae"] for row in rows) / len(rows),
                    "b11_precision": sum(row["b11_precision"] for row in rows) / len(rows),
                    "b26_time_ms": sum(row["b26_time_ms"] for row in rows) / len(rows),
                    "proxy": rows[0]["proxy"],
                }
            )
        view_knob[view] = pick_knob(candidates)

    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card: PolicyCard = cards[view]
        row = per_clip[sequence]
        frames_h = list(clip["history_frames"])
        frames_v = list(clip["validation_frames"])
        truth = clip["truth"]
        b11_h, b11_v = row["b11_history"], row["b11_val"]
        knob = view_knob[view]

        def full_preds(model: str, imgsz: int, conf: float, frames: list[int]) -> dict[int, tuple[Prediction, ...]]:
            out = {}
            for frame in frames:
                preds, _ms = full_store[(sequence, frame, model, imgsz)]
                out[frame] = _filter_conf(preds, conf)
            return out

        def full_time(model: str, imgsz: int, frames: list[int]) -> float:
            return sum(full_store[(sequence, frame, model, imgsz)][1] for frame in frames) / len(frames)

        glance_h: dict[int, tuple[Prediction, ...]] = {}
        glance_v: dict[int, tuple[Prediction, ...]] = {}
        glance_time_h: list[float] = []
        glance_time_v: list[float] = []
        for frame in frames_h + frames_v:
            mixed = exclusive_mix(row["full_11s_640"][frame], row["crop_26_only"][frame][0], card.regions)
            packed = mixed.for_metrics()
            ms = row["full_11s_640_ms"][frame] + row["crop_26_only"][frame][1]
            if frame in set(frames_h):
                glance_h[frame] = packed
                glance_time_h.append(ms)
            else:
                glance_v[frame] = packed
                glance_time_v.append(ms)

        online_v: dict[int, tuple[Prediction, ...]] = {}
        online_times: list[float] = []
        conf = 0.25
        hist_mean = b11_h["pred_box_count"] / max(1, len(frames_h))
        prev = hist_mean
        for frame in frames_v:
            conf = next_conf(conf, prev, hist_mean)
            preds = _filter_conf(row["crop_adp"][frame][0], conf)
            online_v[frame] = preds
            online_times.append(row["crop_adp"][frame][1])
            prev = float(len(preds))

        arms_h = {
            "b11": _score(
                {f: dual[sequence][f].yolo11s for f in frames_h},
                frames_h,
                truth,
                time_ms=full_time("yolo11s", 640, frames_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=0.25,
            ),
            "b26": _score(
                {f: dual[sequence][f].yolo26 for f in frames_h},
                frames_h,
                truth,
                time_ms=row["b26_time_history"],
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=1.0,
            ),
            "A_knob": _score(
                full_preds(knob["model"], int(knob["imgsz"]), float(knob["conf"]), frames_h),
                frames_h,
                truth,
                time_ms=full_time(knob["model"], int(knob["imgsz"]), frames_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=float(knob["proxy"]),
            ),
            "B_mosaic": _score(
                dict_preds(row["mosaic"], frames_h),
                frames_h,
                truth,
                time_ms=dict_time(row["mosaic"], frames_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=proxy,
            ),
            "C_glance": _score(
                glance_h,
                frames_h,
                truth,
                time_ms=sum(glance_time_h) / len(glance_time_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=0.25 + proxy,
            ),
            "D_scale": _score(
                dict_preds(row["crop_adp"], frames_h),
                frames_h,
                truth,
                time_ms=dict_time(row["crop_adp"], frames_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=proxy,
            ),
            "crop640": _score(
                dict_preds(row["crop_fix"], frames_h),
                frames_h,
                truth,
                time_ms=dict_time(row["crop_fix"], frames_h),
                b11_mae=b11_h["mae"],
                b11_precision=b11_h["precision"],
                b26_time_ms=row["b26_time_history"],
                proxy=proxy,
            ),
        }
        arms_v = {
            "b11": _score(
                {f: dual[sequence][f].yolo11s for f in frames_v},
                frames_v,
                truth,
                time_ms=full_time("yolo11s", 640, frames_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=0.25,
            ),
            "b26": _score(
                {f: dual[sequence][f].yolo26 for f in frames_v},
                frames_v,
                truth,
                time_ms=row["b26_time_val"],
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=1.0,
            ),
            "A_knob": _score(
                full_preds(knob["model"], int(knob["imgsz"]), float(knob["conf"]), frames_v),
                frames_v,
                truth,
                time_ms=full_time(knob["model"], int(knob["imgsz"]), frames_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=float(knob["proxy"]),
            ),
            "B_mosaic": _score(
                dict_preds(row["mosaic"], frames_v),
                frames_v,
                truth,
                time_ms=dict_time(row["mosaic"], frames_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=proxy,
            ),
            "C_glance": _score(
                glance_v,
                frames_v,
                truth,
                time_ms=sum(glance_time_v) / len(glance_time_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=0.25 + proxy,
            ),
            "D_scale": _score(
                dict_preds(row["crop_adp"], frames_v),
                frames_v,
                truth,
                time_ms=dict_time(row["crop_adp"], frames_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=proxy,
            ),
            "F_online": _score(
                online_v,
                frames_v,
                truth,
                time_ms=sum(online_times) / len(online_times),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=proxy,
            ),
            "crop640": _score(
                dict_preds(row["crop_fix"], frames_v),
                frames_v,
                truth,
                time_ms=dict_time(row["crop_fix"], frames_v),
                b11_mae=b11_v["mae"],
                b11_precision=b11_v["precision"],
                b26_time_ms=row["b26_time_val"],
                proxy=proxy,
            ),
        }
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "view_id": view,
            "picked_knob": {k: knob[k] for k in ("model", "imgsz", "conf", "mae", "precision", "time_ms")},
            "history": {name: {k: arm[k] for k in ("mae", "precision", "pred_box_count", "time_ms_mean", "h1", "mae_change_vs_b11", "precision_change_vs_b11")} for name, arm in arms_h.items()},
            "validation": {name: {k: arm[k] for k in ("mae", "precision", "pred_box_count", "time_ms_mean", "h1", "mae_change_vs_b11", "precision_change_vs_b11")} for name, arm in arms_v.items()},
            "val_pass": {name: bool(arm["h1"]["passed"]) for name, arm in arms_v.items()},
        }

    primary = [PRIMARY_EVAL["VIRAT_S_010004"], PRIMARY_EVAL["VIRAT_S_010204"]]
    summary = {}
    for arm in ("A_knob", "B_mosaic", "C_glance", "D_scale", "F_online", "crop640"):
        val_pass = sum(row["val_pass"].get(arm, False) for row in per_clip.values())
        primary_pass = all(per_clip[seq]["val_pass"].get(arm, False) for seq in primary)
        summary[arm] = {"val_pass_count": val_pass, "primary_both_pass": primary_pass}
    result = {
        "schema_version": "sceneconfig.execution-policy.route-panel.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "official_h1": False,
        "e4_source": str(e4_result),
        "arms": ["A_knob", "B_mosaic", "C_glance", "D_scale", "F_online"],
        "b_is_mosaic_not_dynamicdet": True,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "view_knob": {view: {k: row[k] for k in ("model", "imgsz", "conf", "mae", "time_ms")} for view, row in view_knob.items()},
        "summary": summary,
        "h1": {
            "reason": "development_route_panel_not_official_h1",
            "primary_clips": primary,
        },
        "per_clip": per_clip,
        "next_gate": "PICK_MAINLINE_FROM_PANEL",
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
    print(json.dumps({"official_h1": False, "summary": result["summary"], "view_knob": result["view_knob"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
