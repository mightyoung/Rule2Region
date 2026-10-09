"""Diagnose whether SceneConfig ROI predictions change with inference batching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.cascade_experiment import Prediction, compute_metrics
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.run_scene_adaptive_small_track_ablation import _load_records, _tile_predictions, fixed_tiles, nms
from research_tools.sceneconfig_e4_m0_deployability import select_m0_configs


def _compile(records: list[Any], rows: dict[tuple[str, int], tuple[Prediction, ...]], card: dict[str, Any]) -> dict[tuple[str, int], tuple[Prediction, ...]]:
    selected = set(card["tiles"])
    output = {}
    for record in records:
        key = (record.sequence_id, record.frame)
        local = tuple(
            row for row in rows.get(key, ())
            if row.source.rsplit(":", 1)[-1] in selected and row.confidence >= float(card["tile_confidence"])
        )
        output[key] = nms((*record.global_predictions, *local))
    return output


def run(
    proxy_path: Path,
    cache_root: Path,
    pilot_root: Path,
    yolo26: Path,
    output_path: Path,
    *,
    device: str = "0",
    batch_a: int = 1,
    batch_b: int = 8,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    selected = select_m0_configs(proxy)
    _history, evaluation, _split = _load_records(cache_root, pilot_root)
    tiles = {
        camera: tuple(tile for tile in fixed_tiles(1280, 720) if tile.tile_id in set(selected[camera]["tiles"]))
        for camera in EXPECTED_CAMERAS
    }
    from ultralytics import YOLO

    rows1, ms1, calls1 = _tile_predictions(YOLO(yolo26), evaluation, tiles, device=device, batch_size=batch_a, confidence=.05, rect=False)
    rows8, ms8, calls8 = _tile_predictions(YOLO(yolo26), evaluation, tiles, device=device, batch_size=batch_b, confidence=.05, rect=False)
    per_camera = {}
    all_disagreements = 0
    all_frames = 0
    for camera in EXPECTED_CAMERAS:
        records = [row for row in evaluation if row.camera_id == camera]
        compiled1 = _compile(records, rows1, selected[camera])
        compiled8 = _compile(records, rows8, selected[camera])
        count_differences = [abs(len(compiled1[key]) - len(compiled8[key])) for key in compiled1]
        truth = [gt for record in records for gt in record.truth]
        metrics1 = compute_metrics([row for values in compiled1.values() for row in values], truth)
        metrics8 = compute_metrics([row for values in compiled8.values() for row in values], truth)
        disagreements = sum(value != 0 for value in count_differences)
        all_disagreements += disagreements
        all_frames += len(count_differences)
        per_camera[camera] = {
            "config_id": selected[camera]["config_id"],
            "frames": len(count_differences),
            "count_disagreement_frames": disagreements,
            "mean_absolute_count_difference": float(np.mean(count_differences)),
            "run_a_count_mae": metrics1["count_mae"],
            "run_b_count_mae": metrics8["count_mae"],
            "run_a_precision": metrics1["precision"],
            "run_b_precision": metrics8["precision"],
        }
    result = {
        "schema_version": "sceneconfig.batch-invariance-diagnostic.v1",
        "evidence_status": "DEVELOPMENT_ONLY_DIAGNOSTIC",
        "holdout_opened": False,
        "selected_from_proxy_only": True,
        "roi_letterbox_rect": False,
        "batch_sizes": {"run_a": batch_a, "run_b": batch_b},
        "tile_calls": {"run_a": calls1, "run_b": calls8},
        "elapsed_ms": {"run_a": ms1, "run_b": ms8},
        "count_disagreement_frame_rate": all_disagreements / all_frames if all_frames else 0.0,
        "batch_invariant": all_disagreements == 0,
        "per_camera": per_camera,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-a", type=int, default=1)
    parser.add_argument("--batch-b", type=int, default=8)
    args = parser.parse_args()
    result = run(args.proxy, args.cache_root, args.pilot_root, args.yolo26, args.output, device=args.device, batch_a=args.batch_a, batch_b=args.batch_b)
    print(json.dumps({"batch_invariant": result["batch_invariant"], "count_disagreement_frame_rate": result["count_disagreement_frame_rate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
