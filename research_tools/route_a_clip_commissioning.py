"""Route A: compiler commissioning on eight frozen camera-sequence clips."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, compute_metrics, parse_mot17_gt
from research_tools.roi_structure_protocol import count_mae, gate_validation
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA
from research_tools.run_missed_person_proposal_baseline import _camera
from research_tools.run_scene_adaptive_small_track_ablation import _owned, fixed_tiles, nms
from research_tools.sceneconfig_e4_m0_deployability import select_m0_configs


VAL_COUNT = 16


def _owned_cell(box: Box, width: int, height: int) -> str:
    for tile in fixed_tiles(width, height):
        if _owned(box, tile.core):
            return tile.tile_id
    return "t3"


def _load_clips_from_split(pilot_root: Path) -> dict[str, dict[str, Any]]:
    split = json.loads((pilot_root / "split-manifest.json").read_text(encoding="utf-8"))
    if split.get("schema_version") != "fieldpack.virat-h1-confirmatory-split.v1":
        raise ValueError("unknown_confirmatory_split_schema")
    val_count = int(split["validation_frame_count"])
    if val_count <= VAL_COUNT:
        raise ValueError("confirmatory_val_count_must_exceed_development_16")
    clips: dict[str, dict[str, Any]] = {}
    for sequence, row in split["sequences"].items():
        camera = str(row["camera_id"])
        if camera == SEALED_CAMERA or SEALED_CAMERA in sequence:
            raise ValueError("sealed_camera_present")
        history = [int(frame) for frame in row["history_frames"]]
        validation = [int(frame) for frame in row["validation_frames"]]
        if len(validation) != val_count:
            raise ValueError(f"split_val_count_mismatch:{sequence}")
        frames = history + validation
        gt_path = pilot_root / "sequences" / sequence / "gt.txt"
        clips[sequence] = {
            "camera_id": camera,
            "sequence_id": sequence,
            "frames": frames,
            "history_frames": history,
            "validation_frames": validation,
            "global_by_frame": {},
            "truth": list(parse_mot17_gt(gt_path)),
        }
    cameras = {clip["camera_id"] for clip in clips.values()}
    if cameras != set(EXPECTED_CAMERAS) or len(clips) != 8:
        raise ValueError("expected_eight_development_clips")
    return clips


def load_clips(cache_root: Path, pilot_root: Path) -> dict[str, dict[str, Any]]:
    if (pilot_root / "split-manifest.json").is_file():
        return _load_clips_from_split(pilot_root)
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("numpy_input_color") != "BGR_CONVERTED_FROM_PIL_RGB":
        raise ValueError("bgr_corrected_cache_required")
    clips: dict[str, dict[str, Any]] = {}
    for row in manifest["sequences"]:
        sequence = row["sequence_id"]
        camera = _camera(sequence)
        if camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        receipt = json.loads((cache_root / row["receipt"]).read_text(encoding="utf-8"))
        by_frame = {int(item["frame"]): item for item in receipt["receipts"]["yolo11s_global"]}
        gt_path = pilot_root / "sequences" / sequence / "gt.txt"
        truth = list(parse_mot17_gt(gt_path))
        frames = sorted(by_frame)
        if len(frames) < VAL_COUNT + 1:
            raise ValueError(f"clip_too_short:{sequence}")
        clips[sequence] = {
            "camera_id": camera,
            "sequence_id": sequence,
            "frames": frames,
            "history_frames": frames[: len(frames) - VAL_COUNT],
            "validation_frames": frames[-VAL_COUNT:],
            "global_by_frame": {
                frame: tuple(
                    Prediction(frame, Box(**item["box"]), float(item["confidence"]), "yolo11s_global")
                    for item in by_frame[frame]["predictions"]
                )
                for frame in frames
            },
            "truth": truth,
        }
    cameras = {clip["camera_id"] for clip in clips.values()}
    if cameras != set(EXPECTED_CAMERAS) or len(clips) != 8:
        raise ValueError("expected_eight_development_clips")
    return clips


def _precision(preds: list[Prediction], truth: list[GroundTruth]) -> float:
    if not truth and not preds:
        return 1.0
    if not truth:
        return 0.0
    return float(compute_metrics(preds, truth)["precision"])


def score_condition(
    clip: Mapping[str, Any],
    pred_by_frame: dict[int, tuple[Prediction, ...]],
    *,
    tile_cost_ratio: float,
) -> dict[str, Any]:
    val = tuple(clip["validation_frames"])
    val_set = set(val)
    truth_boxes = [row for row in clip["truth"] if row.frame in val_set]
    truth_counts = {frame: sum(1 for row in truth_boxes if row.frame == frame) for frame in val}
    pred_counts = {frame: len(pred_by_frame.get(frame, ())) for frame in val}
    baseline = {frame: len(clip["global_by_frame"].get(frame, ())) for frame in val}
    baseline_preds = [row for frame in val for row in clip["global_by_frame"].get(frame, ())]
    compiled_preds = [row for frame in val for row in pred_by_frame.get(frame, ())]
    baseline_mae = count_mae(baseline, truth_counts, val)
    mae = count_mae(pred_counts, truth_counts, val)
    baseline_precision = _precision(baseline_preds, truth_boxes)
    precision = _precision(compiled_preds, truth_boxes)
    gate = gate_validation(
        mae=mae,
        baseline_mae=baseline_mae,
        precision_change=precision - baseline_precision,
        tile_cost_ratio=tile_cost_ratio,
        oom=False,
    )
    return {
        "baseline_mae": baseline_mae,
        "mae": mae if gate["passed"] else baseline_mae,
        "compiled_mae": mae,
        "mae_relative_gain": gate["mae_relative_gain"] if gate["passed"] else 0.0,
        "compiled_mae_relative_gain": gate["mae_relative_gain"],
        "precision_change": gate["precision_change"] if gate["passed"] else 0.0,
        "compiled_precision_change": gate["precision_change"],
        "truth_box_count": len(truth_boxes),
        "validation_gate": gate,
        "intention": "compiled" if gate["passed"] else "B0",
        "verdict": gate["verdict"],
    }


def compile_m0(
    clip: Mapping[str, Any],
    card: Mapping[str, Any],
    tile_by_frame: dict[int, tuple[Prediction, ...]] | None,
) -> dict[int, tuple[Prediction, ...]]:
    tiles = set(card["tiles"])
    width, height = 1280, 720
    output: dict[int, tuple[Prediction, ...]] = {}
    for frame in clip["validation_frames"]:
        kept = list(clip["global_by_frame"].get(frame, ()))
        if tiles and tile_by_frame:
            kept.extend(row for row in tile_by_frame.get(frame, ()) if _owned_cell(row.box, width, height) in tiles)
        merged = nms(kept)
        if tiles:
            merged = tuple(row for row in merged if _owned_cell(row.box, width, height) in tiles or row.source == "yolo11s_global")
        output[frame] = tuple(merged)
    return output


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    proxy_path: Path,
    output_path: Path,
    tile_predictions_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    clips = load_clips(cache_root, pilot_root)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    selected = select_m0_configs(proxy)
    tile_store: dict[str, dict[int, list[Prediction]]] = defaultdict(lambda: defaultdict(list))
    if tile_predictions_path and tile_predictions_path.is_file():
        if tile_predictions_path.suffix == ".jsonl":
            for line in tile_predictions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                camera_id = row.get("camera_id")
                if camera_id in selected and row.get("config_id") != selected[camera_id]["config_id"]:
                    continue
                if "frames" in row:
                    for frame in row["frames"]:
                        sequence = str(frame.get("sequence_id") or "")
                        frame_id = int(frame["frame"])
                        for item in frame.get("tile_raw") or ():
                            box = item["box"]
                            tile_store[sequence][frame_id].append(
                                Prediction(
                                    frame_id,
                                    Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                                    float(item.get("confidence") or 0.0),
                                    str(item.get("source") or "yolo26_tile"),
                                )
                            )
                else:
                    box = row["box"]
                    tile_store[row["sequence_id"]][int(row["frame"])].append(
                        Prediction(
                            int(row["frame"]),
                            Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                            float(row["confidence"]),
                            str(row.get("source") or "yolo26_tile"),
                        )
                    )
        else:
            raise ValueError("tile_predictions_must_be_jsonl")
    per_clip = {}
    for sequence, clip in sorted(clips.items()):
        camera = clip["camera_id"]
        card = selected[camera]
        b0 = {frame: clip["global_by_frame"][frame] for frame in clip["validation_frames"]}
        b0_score = score_condition(clip, b0, tile_cost_ratio=0.0)
        b0_score["config_id"] = "B0"
        tiles = tuple(card["tiles"])
        if tiles and sequence not in tile_store:
            m0_score = {
                "config_id": card["config_id"],
                "tiles": list(tiles),
                "verdict": "hold_or_escalate",
                "intention": "B0",
                "baseline_mae": b0_score["baseline_mae"],
                "mae": b0_score["baseline_mae"],
                "mae_relative_gain": 0.0,
                "precision_change": 0.0,
                "reason": "tile_predictions_missing_for_transfer_clip",
            }
        else:
            compiled = compile_m0(
                clip,
                {"tiles": tiles},
                {frame: tuple(rows) for frame, rows in tile_store.get(sequence, {}).items()} if tiles else None,
            )
            m0_score = score_condition(clip, compiled, tile_cost_ratio=len(tiles) / 4.0)
            m0_score["config_id"] = card["config_id"]
            m0_score["tiles"] = list(tiles)
        per_clip[sequence] = {
            "camera_id": camera,
            "sequence_id": sequence,
            "role": "transfer" if sequence != sorted(item["sequence_id"] for item in clips.values() if item["camera_id"] == camera)[0] else "development_eval",
            "history_frames": clip["history_frames"],
            "validation_frames": clip["validation_frames"],
            "b0": b0_score,
            "m0": m0_score,
            "vlm_intention": "B0",
            "vlm_reason": "h_b1_already_legal_reject_not_rerun",
        }
    passed = sum(row["m0"].get("verdict") == "pass" for row in per_clip.values())
    result = {
        "schema_version": "sceneconfig.route-a-clip-commissioning.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "route": "A_compiler_legal_reject",
        "qwen_called": False,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "m0_pass_count": passed,
        "per_clip": per_clip,
        "next_gate": "WRITE_ROUTE_A_COMMISSIONING_TABLE",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-predictions", type=Path)
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        proxy_path=args.proxy,
        output_path=args.output,
        tile_predictions_path=args.tile_predictions,
    )
    print(json.dumps({"clip_count": result["clip_count"], "m0_pass_count": result["m0_pass_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
