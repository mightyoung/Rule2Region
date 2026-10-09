"""Development-only feasibility panel for native batched YOLO26 crops."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.crop_infer import (
    exclusive_mix,
    infer_batched_yolo26_frame,
    infer_full_frame,
    load_bgr,
)
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.run_execution_stack import FULL_CONF, _filter, _score
from research_tools.execution_policy.score import _arm_metrics
from research_tools.route_a_clip_commissioning import load_clips
from research_tools.run_dcrq_development_gate import SEALED_CAMERA


BASE_IMGSZ = (320, 480, 640)
CROP_IMGSZ = (160, 192, 256, 320)
CROP_CONFS = (0.15, 0.25, 0.35)


def candidate_name(base_imgsz: int, crop_imgsz: int, crop_conf: float) -> str:
    return f"b{base_imgsz}_c{crop_imgsz}_p{int(round(crop_conf * 100)):02d}"


def tensor_proxy(*, base_imgsz: int, crop_imgsz: int, foci_count: int) -> float:
    return 0.25 * (base_imgsz / 640.0) ** 2 + foci_count * (crop_imgsz / 640.0) ** 2


def pick_history_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if bool(row["all_history_pass"])]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            float(row["history_mae_mean"]),
            -float(row["history_precision_mean"]),
            float(row["history_time_ms_mean"]),
            str(row["name"]),
        ),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pack_score(score: Mapping[str, Any]) -> dict[str, Any]:
    packed = {
        key: score[key]
        for key in (
            "mae",
            "precision",
            "pred_box_count",
            "time_ms_mean",
            "mae_change_vs_b11",
            "precision_change_vs_b11",
            "h1",
        )
    }
    packed["latency_proxy"] = score["h1"]["compiled_proxy"]
    return packed


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
        raise ValueError("sealed_split_required")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    clips = load_clips(cache_root, pilot_root)
    if any(SEALED_CAMERA in sequence or str(clip["camera_id"]) == SEALED_CAMERA for sequence, clip in clips.items()):
        raise ValueError("sealed_camera_present")
    dual = load_dual_ledger(dual_ledger_path)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    models = {"yolo11s": YOLO(str(yolo11s)), "yolo26": YOLO(str(yolo26))}

    first = next(iter(clips.values()))
    warm_frame = int(first["history_frames"][0])
    warm_bgr = load_bgr(pilot_root / "sequences" / first["sequence_id"] / "img1" / f"{warm_frame:06d}.jpg")
    infer_full_frame(models["yolo11s"], warm_bgr, warm_frame, conf=FULL_CONF, source="yolo11s", device=device, imgsz=640)
    infer_full_frame(models["yolo26"], warm_bgr, warm_frame, conf=FULL_CONF, source="yolo26", device=device, imgsz=640)

    by_view: dict[str, list[str]] = defaultdict(list)
    stores: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        by_view[view].append(sequence)
        card = cards[view]
        frames = list(clip["history_frames"]) + list(clip["validation_frames"])
        base: dict[int, dict[int, tuple[tuple[Prediction, ...], float]]] = {}
        crops: dict[int, dict[int, tuple[tuple[Prediction, ...], float]]] = {}
        full26: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        for frame in frames:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            base[frame] = {}
            for imgsz in BASE_IMGSZ:
                base[frame][imgsz] = infer_full_frame(
                    models["yolo11s"], bgr, frame, conf=FULL_CONF, source="yolo11s", device=device, imgsz=imgsz
                )
            full26[frame] = infer_full_frame(
                models["yolo26"], bgr, frame, conf=FULL_CONF, source="yolo26", device=device, imgsz=640
            )
            crops[frame] = {}
            for imgsz in CROP_IMGSZ:
                part, elapsed_ms = infer_batched_yolo26_frame(
                    models["yolo26"], card, bgr, frame, device=device, imgsz=imgsz, conf=min(CROP_CONFS)
                )
                crops[frame][imgsz] = (part.yolo26, elapsed_ms)
        stores[sequence] = {"base": base, "crops": crops, "full26": full26}

    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        store = stores[sequence]
        truth = clip["truth"]
        phases = {"history": list(clip["history_frames"]), "validation": list(clip["validation_frames"])}
        phase_out: dict[str, Any] = {}
        for phase, frames in phases.items():
            b11 = _arm_metrics(frames, {frame: dual[sequence][frame].yolo11s for frame in frames}, truth)
            b26_time = _mean([float(store["full26"][frame][1]) for frame in frames])
            candidates: dict[str, Any] = {}
            for base_imgsz in BASE_IMGSZ:
                for crop_imgsz in CROP_IMGSZ:
                    for crop_conf in CROP_CONFS:
                        preds: dict[int, tuple[Prediction, ...]] = {}
                        times: list[float] = []
                        for frame in frames:
                            base_preds, base_ms = store["base"][frame][base_imgsz]
                            crop_preds, crop_ms = store["crops"][frame][crop_imgsz]
                            filtered_crop = _filter(crop_preds, crop_conf)
                            preds[frame] = exclusive_mix(base_preds, filtered_crop, card.regions).for_metrics()
                            times.append(float(base_ms) + float(crop_ms))
                        proxy = tensor_proxy(
                            base_imgsz=base_imgsz,
                            crop_imgsz=crop_imgsz,
                            foci_count=sum(1 for row in card.regions if row.role == "count" and row.model == "yolo26"),
                        )
                        scored = _score(
                            preds,
                            frames,
                            truth,
                            time_ms=_mean(times),
                            b11_mae=float(b11["mae"]),
                            b11_p=float(b11["precision"]),
                            b26_time=b26_time,
                            proxy=proxy,
                        )
                        name = candidate_name(base_imgsz, crop_imgsz, crop_conf)
                        candidates[name] = {
                            "base_imgsz": base_imgsz,
                            "crop_imgsz": crop_imgsz,
                            "crop_conf": crop_conf,
                            **_pack_score(scored),
                        }
            phase_out[phase] = {
                "b11_reference": b11,
                "b26_time_ms_mean": b26_time,
                "candidates": candidates,
            }
        per_clip[sequence] = {"view_id": view, **phase_out}

    view_results: dict[str, Any] = {}
    for view, sequences in sorted(by_view.items()):
        candidate_rows: list[dict[str, Any]] = []
        for name in sorted(per_clip[sequences[0]]["history"]["candidates"]):
            hist = [per_clip[sequence]["history"]["candidates"][name] for sequence in sequences]
            val = [per_clip[sequence]["validation"]["candidates"][name] for sequence in sequences]
            first_row = hist[0]
            candidate_rows.append(
                {
                    "name": name,
                    "base_imgsz": first_row["base_imgsz"],
                    "crop_imgsz": first_row["crop_imgsz"],
                    "crop_conf": first_row["crop_conf"],
                    "all_history_pass": all(bool(row["h1"]["passed"]) for row in hist),
                    "all_validation_pass": all(bool(row["h1"]["passed"]) for row in val),
                    "history_mae_mean": _mean([float(row["mae"]) for row in hist]),
                    "history_precision_mean": _mean([float(row["precision"]) for row in hist]),
                    "history_time_ms_mean": _mean([float(row["time_ms_mean"]) for row in hist]),
                    "validation_mae_mean": _mean([float(row["mae"]) for row in val]),
                    "validation_precision_mean": _mean([float(row["precision"]) for row in val]),
                    "validation_time_ms_mean": _mean([float(row["time_ms_mean"]) for row in val]),
                }
            )
        selected = pick_history_candidate(candidate_rows)
        history_eligible = [row["name"] for row in candidate_rows if row["all_history_pass"]]
        validation_oracle = [row["name"] for row in candidate_rows if row["all_validation_pass"]]
        view_results[view] = {
            "sequences": sorted(sequences),
            "history_eligible_count": len(history_eligible),
            "history_eligible": history_eligible,
            "history_selected": dict(selected) if selected is not None else None,
            "selected_validation_pass": bool(selected and selected["all_validation_pass"]),
            "validation_oracle_count": len(validation_oracle),
            "validation_oracle": validation_oracle,
            "candidates": candidate_rows,
        }

    selected_views = [row for row in view_results.values() if row["history_selected"] is not None]
    result = {
        "schema_version": "sceneconfig.execution-policy.yolo26-native-batch-feasibility.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "official_h1": False,
        "v6_official": False,
        "reason": "seen_confirmatory_feasibility_development",
        "input_split_schema": split.get("schema_version"),
        "decision_unit": "view_id",
        "sealed_camera_used": False,
        "metric_time_scope": "MODEL_CALLS_ONLY",
        "base_imgsz": list(BASE_IMGSZ),
        "crop_imgsz": list(CROP_IMGSZ),
        "crop_confs": list(CROP_CONFS),
        "candidate_count": len(BASE_IMGSZ) * len(CROP_IMGSZ) * len(CROP_CONFS),
        "yolo26_calls_per_frame_per_candidate": 1,
        "view_count": len(view_results),
        "clip_count": len(per_clip),
        "summary": {
            "history_selected_view_count": len(selected_views),
            "selected_validation_pass_view_count": sum(bool(row["selected_validation_pass"]) for row in view_results.values()),
            "validation_oracle_possible_view_count": sum(int(row["validation_oracle_count"]) > 0 for row in view_results.values()),
        },
        "view_results": view_results,
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
    print(json.dumps({"summary": result["summary"], "views": result["view_results"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
