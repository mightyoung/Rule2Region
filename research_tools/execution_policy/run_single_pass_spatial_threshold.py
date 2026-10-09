"""Development-only single-pass YOLO11s spatial-threshold panel."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.card import PolicyCard
from research_tools.execution_policy.crop_infer import infer_full_frame, load_bgr
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.execution_policy.run_execution_stack import _score
from research_tools.execution_policy.score import _arm_metrics
from research_tools.residual_query import box_center
from research_tools.route_a_clip_commissioning import load_clips
from research_tools.run_dcrq_development_gate import SEALED_CAMERA


INFERENCE_CONF = 0.05
OUTSIDE_CONFS = (0.15, 0.25, 0.35)
FOCUS_CONFS = (0.05, 0.10, 0.15, 0.25, 0.35)
COMPILED_PROXY = 0.25


def candidate_name(outside_conf: float, focus_conf: float) -> str:
    return f"o{int(round(outside_conf * 100)):02d}_f{int(round(focus_conf * 100)):02d}"


def focus_boxes(card: PolicyCard):
    return tuple(row.box for row in card.regions if row.role == "count" and row.model == "yolo26")


def spatial_filter(
    predictions: Sequence[Prediction],
    card: PolicyCard,
    *,
    outside_conf: float,
    focus_conf: float,
) -> tuple[Prediction, ...]:
    boxes = focus_boxes(card)
    kept: list[Prediction] = []
    for prediction in predictions:
        x, y = box_center(prediction.box)
        inside = any(box.x1 <= x < box.x2 and box.y1 <= y < box.y2 for box in boxes)
        threshold = focus_conf if inside else outside_conf
        if prediction.confidence >= threshold:
            kept.append(prediction)
    return tuple(kept)


def pick_history_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if bool(row["all_history_pass"])]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            float(row["history_mae_mean"]),
            -float(row["history_precision_mean"]),
            float(row["threshold_change"]),
            str(row["name"]),
        ),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pack_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: score[key]
        for key in (
            "mae", "precision", "pred_box_count", "time_ms_mean",
            "mae_change_vs_b11", "precision_change_vs_b11", "h1",
        )
    }


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    dual_ledger_path: Path,
    e4_result: Path,
    b26_reference: Path,
    output_dir: Path,
    yolo11s: Path,
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
    b26 = json.loads(b26_reference.read_text(encoding="utf-8"))
    if b26.get("schema_version") != "sceneconfig.execution-policy.yolo26-native-batch-feasibility.v1":
        raise ValueError("unexpected_b26_reference_schema")
    model = YOLO(str(yolo11s))

    first = next(iter(clips.values()))
    warm_frame = int(first["history_frames"][0])
    warm_bgr = load_bgr(pilot_root / "sequences" / first["sequence_id"] / "img1" / f"{warm_frame:06d}.jpg")
    infer_full_frame(model, warm_bgr, warm_frame, conf=INFERENCE_CONF, source="yolo11s", device=device, imgsz=640)

    by_view: dict[str, list[str]] = defaultdict(list)
    stores: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        by_view[view].append(sequence)
        frames = list(clip["history_frames"]) + list(clip["validation_frames"])
        full11: dict[int, tuple[tuple[Prediction, ...], float]] = {}
        for frame in frames:
            bgr = load_bgr(pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg")
            full11[frame] = infer_full_frame(
                model, bgr, frame, conf=INFERENCE_CONF, source="yolo11s", device=device, imgsz=640
            )
        stores[sequence] = full11

    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        card = cards[view]
        truth = clip["truth"]
        phase_out: dict[str, Any] = {}
        for phase, frames in {
            "history": list(clip["history_frames"]),
            "validation": list(clip["validation_frames"]),
        }.items():
            b11 = _arm_metrics(frames, {frame: dual[sequence][frame].yolo11s for frame in frames}, truth)
            b26_time = float(b26["per_clip"][sequence][phase]["b26_time_ms_mean"])
            measured_time = _mean([float(stores[sequence][frame][1]) for frame in frames])
            candidates: dict[str, Any] = {}
            for outside_conf in OUTSIDE_CONFS:
                for focus_conf in FOCUS_CONFS:
                    preds = {
                        frame: spatial_filter(
                            stores[sequence][frame][0], card,
                            outside_conf=outside_conf, focus_conf=focus_conf,
                        )
                        for frame in frames
                    }
                    score = _score(
                        preds, frames, truth, time_ms=measured_time,
                        b11_mae=float(b11["mae"]), b11_p=float(b11["precision"]),
                        b26_time=b26_time, proxy=COMPILED_PROXY,
                    )
                    name = candidate_name(outside_conf, focus_conf)
                    candidates[name] = {
                        "outside_conf": outside_conf,
                        "focus_conf": focus_conf,
                        **_pack_score(score),
                    }
            phase_out[phase] = {
                "b11_reference": b11,
                "b26_time_ms_mean": b26_time,
                "single_pass_time_ms_mean": measured_time,
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
            outside_conf, focus_conf = float(first_row["outside_conf"]), float(first_row["focus_conf"])
            candidate_rows.append({
                "name": name,
                "outside_conf": outside_conf,
                "focus_conf": focus_conf,
                "threshold_change": abs(outside_conf - 0.15) + abs(focus_conf - 0.15),
                "all_history_pass": all(bool(row["h1"]["passed"]) for row in hist),
                "all_validation_pass": all(bool(row["h1"]["passed"]) for row in val),
                "history_mae_mean": _mean([float(row["mae"]) for row in hist]),
                "history_precision_mean": _mean([float(row["precision"]) for row in hist]),
                "history_time_ms_mean": _mean([float(row["time_ms_mean"]) for row in hist]),
                "validation_mae_mean": _mean([float(row["mae"]) for row in val]),
                "validation_precision_mean": _mean([float(row["precision"]) for row in val]),
                "validation_time_ms_mean": _mean([float(row["time_ms_mean"]) for row in val]),
            })
        selected = pick_history_candidate(candidate_rows)
        history_eligible = [row["name"] for row in candidate_rows if row["all_history_pass"]]
        validation_oracle = [row["name"] for row in candidate_rows if row["all_validation_pass"]]
        view_results[view] = {
            "sequences": sorted(sequences),
            "focus_region_count": len(focus_boxes(cards[view])),
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
        "schema_version": "sceneconfig.execution-policy.single-pass-spatial-threshold.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "official_h1": False,
        "v6_official": False,
        "reason": "seen_confirmatory_single_pass_development",
        "input_split_schema": split.get("schema_version"),
        "decision_unit": "view_id",
        "sealed_camera_used": False,
        "metric_time_scope": "MODEL_CALLS_ONLY",
        "model": "yolo11s",
        "imgsz": 640,
        "inference_conf": INFERENCE_CONF,
        "outside_confs": list(OUTSIDE_CONFS),
        "focus_confs": list(FOCUS_CONFS),
        "candidate_count": len(OUTSIDE_CONFS) * len(FOCUS_CONFS),
        "model_calls_per_frame": 1,
        "second_model_called": False,
        "b26_time_source": "prior_same_day_verified_reference",
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
    parser.add_argument("--b26-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root, pilot_root=args.pilot_root,
        dual_ledger_path=args.dual_ledger, e4_result=args.e4_result,
        b26_reference=args.b26_reference, output_dir=args.output_dir,
        yolo11s=args.yolo11s, device=args.device,
    )
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
