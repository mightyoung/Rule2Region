"""Validation-gated ROI structure protocol (B + D-prime dispositions)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, compute_metrics, parse_mot17_gt
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA
from research_tools.run_scene_adaptive_small_track_ablation import _owned, fixed_tiles, nms
from research_tools.select_dense_detection_frames import (
    choose_observer_config,
    materialize_native_scene_pack,
    select_dense_frames,
)


CELLS = ("t0", "t1", "t2", "t3")
TYPES = ("INCLUDE", "EXCLUDE", "EXPERT")
LEVELS = ("L0", "L1", "L2", "L3", "L4")
HOLD_RATIO = 0.30
HISTORY_COUNT = 16
VAL_COUNT = 10
PROPOSAL_FRAMES = 8
REVIEW_FRAMES = 8
PRIMARY_CAMERAS = ("VIRAT_S_010004", "VIRAT_S_010204")
TOKEN_POSITION_WARNING = re.compile(r"non-consecutive token position", re.I)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_from_row(row: Mapping[str, Any]) -> Box:
    box = row["box"] if "box" in row else row
    return Box(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))


def _prediction(row: Mapping[str, Any]) -> Prediction:
    return Prediction(int(row.get("frame") or 0), _box_from_row(row), float(row.get("confidence") or 0.0), str(row.get("source") or "detector"))


def temporal_protocol_split(frames: Sequence[int]) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted(set(int(frame) for frame in frames)))
    if len(ordered) < HISTORY_COUNT + VAL_COUNT + 1:
        raise ValueError("insufficient_frames_for_temporal_split")
    history = ordered[:HISTORY_COUNT]
    validation = ordered[HISTORY_COUNT : HISTORY_COUNT + VAL_COUNT]
    confirm = ordered[HISTORY_COUNT + VAL_COUNT :]
    if len(validation) != VAL_COUNT:
        raise ValueError("validation_must_be_ten_contiguous_frames")
    if set(history) & set(validation) or set(validation) & set(confirm):
        raise ValueError("temporal_partitions_must_be_disjoint")
    return {"history": history, "validation": validation, "confirm": confirm}


def parse_structure(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("structure_not_exact_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("structure_must_be_object")
    if payload.get("abstain") is True:
        if payload.get("cells"):
            raise ValueError("abstain_cannot_include_cells")
        return {"abstain": True, "cells": ()}
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) > 4:
        raise ValueError("cells_must_be_list_of_at_most_four")
    seen: set[str] = set()
    parsed: list[dict[str, str]] = []
    expert = 0
    for item in cells:
        if not isinstance(item, dict) or set(item) != {"cell", "type"}:
            raise ValueError("cell_keys_invalid")
        cell, kind = item["cell"], item["type"]
        if cell not in CELLS or kind not in TYPES:
            raise ValueError("cell_or_type_out_of_contract")
        if cell in seen:
            raise ValueError("duplicate_cell")
        if kind == "EXPERT":
            expert += 1
        seen.add(cell)
        parsed.append({"cell": cell, "type": kind})
    types_by_cell = {row["cell"]: row["type"] for row in parsed}
    if any(types_by_cell.get(cell) == "INCLUDE" and types_by_cell.get(cell) == "EXCLUDE" for cell in CELLS):
        raise ValueError("include_exclude_conflict")
    if expert > 2:
        raise ValueError("expert_cells_exceed_two")
    return {"abstain": False, "cells": tuple(parsed)}


def parse_review(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("review_not_exact_json") from exc
    required = {"level", "suspect_frame_ids", "patch", "rationale"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("review_keys_invalid")
    if payload["level"] not in LEVELS:
        raise ValueError("review_level_invalid")
    suspects = payload["suspect_frame_ids"]
    if not isinstance(suspects, list) or not all(isinstance(item, int) for item in suspects):
        raise ValueError("suspect_frame_ids_invalid")
    if len(suspects) != len(set(suspects)):
        raise ValueError("suspect_frame_ids_duplicate")
    patch = payload["patch"]
    if patch is not None:
        if not isinstance(patch, dict) or set(patch) != {"cell", "type"}:
            raise ValueError("patch_keys_invalid")
        if patch["cell"] not in CELLS or patch["type"] not in {"EXCLUDE", "EXPERT"}:
            raise ValueError("patch_out_of_contract")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("rationale_required")
    return {
        "level": payload["level"],
        "suspect_frame_ids": tuple(suspects),
        "patch": None if patch is None else {"cell": patch["cell"], "type": patch["type"]},
        "rationale": payload["rationale"].strip(),
    }


def assign_disposition(
    review: Mapping[str, Any] | None,
    *,
    history_frames: Sequence[int],
    parse_error: str | None = None,
    proposal_abstain: bool = False,
) -> dict[str, Any]:
    if parse_error or proposal_abstain or review is None:
        return {"level": "L4", "patch": None, "reason": parse_error or "proposal_abstain_or_missing_review"}
    allowed = set(history_frames)
    suspects = [frame for frame in review["suspect_frame_ids"] if frame in allowed]
    leaked = [frame for frame in review["suspect_frame_ids"] if frame not in allowed]
    if leaked:
        return {"level": "L4", "patch": None, "reason": "review_cited_non_history_frame"}
    ratio = len(suspects) / max(1, len(history_frames))
    stated = review["level"]
    if stated == "L4":
        return {"level": "L4", "patch": None, "suspect_ratio": ratio, "reason": "reviewer_escalated"}
    patch = review["patch"]
    if patch is not None:
        return {"level": "L3", "patch": patch, "suspect_ratio": ratio, "reason": "bounded_structure_patch"}
    if stated == "L3":
        return {"level": "L4", "patch": None, "suspect_ratio": ratio, "reason": "l3_requires_one_structure_patch"}
    if stated == "L2" or ratio >= HOLD_RATIO:
        return {"level": "L2", "patch": None, "suspect_ratio": ratio, "reason": "suspect_density_hold" if ratio >= HOLD_RATIO else "reviewer_hold"}
    if suspects:
        return {"level": "L1", "patch": None, "suspect_ratio": ratio, "reason": "sparse_note"}
    return {"level": "L0", "patch": None, "suspect_ratio": 0.0, "reason": "agree"}


def apply_patch(cells: Sequence[Mapping[str, str]], patch: Mapping[str, str] | None) -> tuple[dict[str, str], ...]:
    by_cell = {row["cell"]: row["type"] for row in cells}
    if patch is None:
        return tuple({"cell": cell, "type": kind} for cell, kind in by_cell.items())
    existing = by_cell.get(patch["cell"])
    if existing == "INCLUDE" and patch["type"] == "EXCLUDE":
        raise ValueError("patch_include_exclude_conflict")
    by_cell[patch["cell"]] = patch["type"]
    expert = sum(kind == "EXPERT" for kind in by_cell.values())
    if expert > 2:
        raise ValueError("patch_expert_cells_exceed_two")
    return tuple({"cell": cell, "type": by_cell[cell]} for cell in CELLS if cell in by_cell)


def _owned_cell(box: Box, width: int, height: int) -> str:
    for tile in fixed_tiles(width, height):
        if _owned(box, tile.core):
            return tile.tile_id
    return "t3"


def materialize_counts(
    frames: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, str]],
    *,
    width: int,
    height: int,
) -> dict[int, int]:
    by_cell = {row["cell"]: row["type"] for row in cells}
    include = {cell for cell, kind in by_cell.items() if kind == "INCLUDE"}
    exclude = {cell for cell, kind in by_cell.items() if kind == "EXCLUDE"}
    expert = {cell for cell, kind in by_cell.items() if kind == "EXPERT"}
    counts: dict[int, int] = {}
    for frame in frames:
        frame_id = int(frame["frame"])
        global_rows = [_prediction(row) for row in frame.get("global") or ()]
        tile_rows = [_prediction(row) for row in frame.get("tile_raw") or ()]
        kept = list(global_rows)
        if expert:
            kept.extend(row for row in tile_rows if _owned_cell(row.box, width, height) in expert)
        merged = nms(kept)
        filtered = []
        for row in merged:
            cell = _owned_cell(row.box, width, height)
            if cell in exclude:
                continue
            if include and cell not in include and cell not in expert:
                continue
            filtered.append(row)
        counts[frame_id] = len(filtered)
    return counts


def scan_visual_logs(response_dir: Path, cameras: Sequence[str]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for camera in cameras:
        for stage in ("structure", "review"):
            log_path = response_dir / f"{camera}.{stage}.log"
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            calls.append(
                {
                    "camera_id": camera,
                    "stage": stage,
                    "log_present": log_path.is_file(),
                    "token_position_warning": bool(TOKEN_POSITION_WARNING.search(text)),
                }
            )
    warned = sum(1 for row in calls if row["token_position_warning"])
    return {
        "visual_calls": len(calls),
        "token_position_warning_calls": warned,
        "token_position_warning_ignored": False,
        "calls": calls,
    }


def count_mae(pred_counts: Mapping[int, int], truth_counts: Mapping[int, int], frames: Sequence[int]) -> float:
    if not frames:
        raise ValueError("frames_required_for_mae")
    return float(sum(abs(pred_counts.get(frame, 0) - truth_counts.get(frame, 0)) for frame in frames) / len(frames))


def gate_validation(
    *,
    mae: float,
    baseline_mae: float,
    precision_change: float,
    tile_cost_ratio: float,
    oom: bool,
) -> dict[str, Any]:
    mae_gain = 0.0 if baseline_mae == 0 else (baseline_mae - mae) / baseline_mae
    latency_proxy = tile_cost_ratio * 2.0
    passed = (
        mae_gain >= 0.10
        and precision_change >= -0.05
        and latency_proxy <= 3.0
        and not oom
    )
    return {
        "passed": passed,
        "mae_relative_gain": mae_gain,
        "precision_change": precision_change,
        "latency_proxy": latency_proxy,
        "latency_scope": "TILE_COST_PROXY_NOT_END_TO_END",
        "oom": oom,
        "verdict": "pass" if passed else "fail",
    }


def _load_prediction_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index[(row["camera_id"], row["config_id"])] = row
    return index


def _truth_counts(gt_path: Path, frames: Sequence[int]) -> dict[int, int]:
    allowed = set(frames)
    counts: dict[int, int] = defaultdict(int)
    for row in parse_mot17_gt(gt_path):
        if row.frame in allowed:
            counts[row.frame] += 1
    return {frame: counts.get(frame, 0) for frame in frames}


def _precision(pred_rows: Sequence[Prediction], truth: Sequence[GroundTruth]) -> float:
    return float(compute_metrics(pred_rows, truth)["precision"])


def _frame_predictions(frame: Mapping[str, Any], cells: Sequence[Mapping[str, str]], width: int, height: int) -> tuple[Prediction, ...]:
    counts_only = materialize_counts([frame], cells, width=width, height=height)
    # Rebuild kept boxes for precision: reuse same filter.
    by_cell = {row["cell"]: row["type"] for row in cells}
    include = {cell for cell, kind in by_cell.items() if kind == "INCLUDE"}
    exclude = {cell for cell, kind in by_cell.items() if kind == "EXCLUDE"}
    expert = {cell for cell, kind in by_cell.items() if kind == "EXPERT"}
    global_rows = [_prediction(row) for row in frame.get("global") or ()]
    tile_rows = [_prediction(row) for row in frame.get("tile_raw") or ()]
    kept = list(global_rows)
    if expert:
        kept.extend(row for row in tile_rows if _owned_cell(row.box, width, height) in expert)
    merged = nms(kept)
    output = []
    for row in merged:
        cell = _owned_cell(row.box, width, height)
        if cell in exclude:
            continue
        if include and cell not in include and cell not in expert:
            continue
        output.append(row)
    assert counts_only[int(frame["frame"])] == len(output)
    return tuple(output)


def structure_prompt(camera_id: str, frame_ids: Sequence[int]) -> str:
    listed = ", ".join(f"Image {index}: native unstitched history frame {frame}" for index, frame in enumerate(frame_ids, start=1))
    return f"""You are the offline ROI-structure proposer for fixed camera {camera_id}.
The attached images are separate native 1280x720 scene frames, not a collage. Do not output pixel coordinates or config_id values.
You may assign at most four cells from t0,t1,t2,t3 to INCLUDE, EXCLUDE, or EXPERT. At most two EXPERT cells. A cell cannot be INCLUDE and EXCLUDE.
If you cannot decide, return abstain true and no cells.
{listed}
Return only JSON: {{"abstain":false,"cells":[{{"cell":"t3","type":"EXPERT"}}]}}"""


def review_prompt(camera_id: str, frame_ids: Sequence[int], cells: Sequence[Mapping[str, str]]) -> str:
    listed = ", ".join(f"Image {index}: native frame {frame} with same-frame boxes" for index, frame in enumerate(frame_ids, start=1))
    return f"""You review compiled detections for camera {camera_id}. Structure so far: {json.dumps(list(cells))}.
Images are unstitched native frames. Boxes use that frame's coordinates. You cannot see the validation set.
level must be L0, L1, L2, L3, or L4. suspect_frame_ids must be a subset of {list(frame_ids)}.
If you request a patch, it must be exactly one cell with type EXCLUDE or EXPERT. Otherwise patch is null.
Do not change thresholds or emit config_id.
{listed}
Return only JSON: {{"level":"L1","suspect_frame_ids":[],"patch":null,"rationale":"..."}}"""


def prepare_camera_assets(
    *,
    camera_id: str,
    observer_row: Mapping[str, Any],
    split: Mapping[str, Sequence[int]],
    sequence_id: str,
    pilot_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    history = set(split["history"])
    proposal = select_dense_frames(observer_row["frames"], allowed_frames=history, k=min(PROPOSAL_FRAMES, len(history)))
    if any(row.frame in set(split["validation"]) | set(split["confirm"]) for row in proposal):
        raise ValueError("proposal_frames_leaked_into_val_or_confirm")
    camera_dir = output_dir / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)
    clean_paths = []
    overlay_paths = []
    for row in proposal:
        source = pilot_root / "sequences" / sequence_id / "img1" / f"{row.frame:06d}.jpg"
        clean = camera_dir / f"{row.frame:06d}.clean.jpg"
        overlay = camera_dir / f"{row.frame:06d}.overlay.jpg"
        materialize_native_scene_pack(
            source_path=source,
            output_clean=clean,
            output_overlay=overlay,
            boxes=row.merged,
            box_id_prefix=f"f{row.frame}b",
        )
        clean_paths.append(clean)
        overlay_paths.append(overlay)
    frame_ids = [row.frame for row in proposal]
    (camera_dir / "proposal-prompt.txt").write_text(structure_prompt(camera_id, frame_ids), encoding="utf-8")
    return {
        "camera_id": camera_id,
        "sequence_id": sequence_id,
        "proposal_frames": frame_ids,
        "clean_images": [path.as_posix() for path in clean_paths],
        "overlay_images": [path.as_posix() for path in overlay_paths],
        "split": {key: list(value) for key, value in split.items()},
        "confirm_opened": False,
    }


def evaluate_camera(
    *,
    camera_id: str,
    observer_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    split: Mapping[str, Sequence[int]],
    structure: Mapping[str, Any],
    disposition: Mapping[str, Any],
    gt_path: Path,
    width: int,
    height: int,
    proxy_scores: Mapping[str, float] | None,
) -> dict[str, Any]:
    val_frames = tuple(split["validation"])
    val_set = set(val_frames)
    observer_frames = [frame for frame in observer_row["frames"] if int(frame["frame"]) in val_set]
    baseline_frames = [frame for frame in baseline_row["frames"] if int(frame["frame"]) in val_set]
    truth = _truth_counts(gt_path, val_frames)
    baseline_counts = {int(frame["frame"]): len(frame.get("global") or ()) for frame in baseline_frames}
    baseline_mae = count_mae(baseline_counts, truth, val_frames)
    baseline_preds = [_prediction(row) for frame in baseline_frames for row in frame.get("global") or ()]
    truth_boxes = [row for row in parse_mot17_gt(gt_path) if row.frame in val_set]
    baseline_precision = _precision(baseline_preds, truth_boxes)

    level = disposition["level"]
    if level in {"L2", "L4"} or structure.get("abstain"):
        return {
            "camera_id": camera_id,
            "level": level,
            "intention": "B0",
            "verdict": "hold_or_escalate" if level in {"L2", "L4"} else "abstain_fallback",
            "validation_gate": {"verdict": "fail", "passed": False, "reason": level},
            "baseline_mae": baseline_mae,
            "mae": baseline_mae,
            "mae_relative_gain": 0.0,
            "precision_change": 0.0,
            "confirm_opened": False,
        }

    cells = apply_patch(structure.get("cells") or (), disposition.get("patch"))
    pred_counts = materialize_counts(observer_frames, cells, width=width, height=height)
    mae = count_mae(pred_counts, truth, val_frames)
    compiled_preds = []
    for frame in observer_frames:
        compiled_preds.extend(_frame_predictions(frame, cells, width, height))
    precision = _precision(compiled_preds, truth_boxes)
    tile_cost = sum(1 for row in cells if row["type"] == "EXPERT") / 4.0
    gate = gate_validation(
        mae=mae,
        baseline_mae=baseline_mae,
        precision_change=precision - baseline_precision,
        tile_cost_ratio=tile_cost,
        oom=False,
    )
    m0_id = None
    if proxy_scores:
        m0_id = min(proxy_scores, key=lambda key: (-proxy_scores[key], key))
    return {
        "camera_id": camera_id,
        "level": level,
        "cells": list(cells),
        "intention": "compiled" if gate["passed"] else "B0",
        "verdict": gate["verdict"],
        "validation_gate": gate,
        "baseline_mae": baseline_mae,
        "mae": mae if gate["passed"] else baseline_mae,
        "mae_relative_gain": gate["mae_relative_gain"] if gate["passed"] else 0.0,
        "precision_change": gate["precision_change"] if gate["passed"] else 0.0,
        "m0_proxy_top": m0_id,
        "confirm_opened": False,
        "primary": camera_id in PRIMARY_CAMERAS,
    }


def prepare_run(
    *,
    predictions_path: Path,
    pilot_root: Path,
    output_dir: Path,
    proxy_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    index = _load_prediction_index(predictions_path)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8")) if proxy_path else None
    cameras: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        available = [config_id for cam, config_id in index if cam == camera]
        observer = choose_observer_config(available)
        observer_row = index[(camera, observer)]
        frames = [int(row["frame"]) for row in observer_row["frames"]]
        split = temporal_protocol_split(frames)
        sequence_id = observer_row["frames"][0]["sequence_id"]
        assets = prepare_camera_assets(
            camera_id=camera,
            observer_row=observer_row,
            split=split,
            sequence_id=sequence_id,
            pilot_root=pilot_root,
            output_dir=output_dir / "media",
        )
        assets["observer_config_id"] = observer
        cameras[camera] = assets
    payload = {
        "schema_version": "sceneconfig.roi-structure-valgate.prepare.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "collage_used": False,
        "hold_ratio": HOLD_RATIO,
        "history_count": HISTORY_COUNT,
        "validation_count": VAL_COUNT,
        "proposal_frames_per_camera": PROPOSAL_FRAMES,
        "predictions_sha256": _sha256_file(predictions_path),
        "proxy_sha256": _sha256_file(proxy_path) if proxy_path else None,
        "cameras": cameras,
    }
    (output_dir / "PRE_VAL_SEAL.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def finalize_run(
    *,
    prepare_path: Path,
    predictions_path: Path,
    response_dir: Path,
    pilot_root: Path,
    output_path: Path,
    proxy_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepare.get("confirm_opened") is not False:
        raise ValueError("confirm_already_opened")
    index = _load_prediction_index(predictions_path)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8")) if proxy_path else None
    per_camera = {}
    for camera in EXPECTED_CAMERAS:
        assets = prepare["cameras"][camera]
        split = {key: tuple(value) for key, value in assets["split"].items()}
        observer = assets["observer_config_id"]
        observer_row = index[(camera, observer)]
        baseline_row = index.get((camera, "B0"), observer_row)
        structure_error = None
        structure: dict[str, Any] | None = None
        try:
            structure = parse_structure((response_dir / f"{camera}.structure.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            structure_error = str(exc)
            structure = {"abstain": True, "cells": ()}
        review_error = None
        review = None
        try:
            review = parse_review((response_dir / f"{camera}.review.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            review_error = str(exc)
        disposition = assign_disposition(
            review,
            history_frames=assets["proposal_frames"],
            parse_error=structure_error or review_error,
            proposal_abstain=bool(structure.get("abstain")),
        )
        if disposition["level"] == "L3":
            try:
                apply_patch(structure.get("cells") or (), disposition.get("patch"))
            except ValueError as exc:
                disposition = {"level": "L4", "patch": None, "reason": str(exc)}
        proxy_scores = None
        if proxy:
            proxy_scores = {
                row["config_id"]: float(row["proxy_utility"])
                for row in proxy["per_camera"][camera]
            }
        gt_path = pilot_root / "sequences" / assets["sequence_id"] / "gt.txt"
        width, height = 1280, 720
        per_camera[camera] = evaluate_camera(
            camera_id=camera,
            observer_row=observer_row,
            baseline_row=baseline_row,
            split=split,
            structure=structure,
            disposition=disposition,
            gt_path=gt_path,
            width=width,
            height=height,
            proxy_scores=proxy_scores,
        )
        per_camera[camera]["structure"] = structure
        per_camera[camera]["disposition"] = disposition
        per_camera[camera]["proposal_frames"] = assets["proposal_frames"]

    primary = [per_camera[camera] for camera in PRIMARY_CAMERAS]
    primary_pass = sum(row["verdict"] == "pass" for row in primary)
    visual_runtime = scan_visual_logs(response_dir, EXPECTED_CAMERAS)
    result = {
        "schema_version": "sceneconfig.roi-structure-valgate.result.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "visual_protocol": "dense_native_frames",
        "collage_used": False,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "visual_runtime": visual_runtime,
        "h_b1": {
            "primary_cameras": list(PRIMARY_CAMERAS),
            "primary_pass_count": primary_pass,
            "both_legal_reject": all(row["verdict"] != "pass" for row in primary),
            "passed": False,
            "reason": "confirmation_closed_until_both_primary_verdicts_sealed",
        },
        "per_camera": per_camera,
        "next_gate": "OPEN_CONFIRM_ONLY_IF_NOT_BOTH_PRIMARY_REJECT" if not all(row["verdict"] != "pass" for row in primary) else "STOP_WRITE_LEGAL_REJECT",
    }
    if all(row["verdict"] != "pass" for row in primary):
        result["h_b1"]["reason"] = "both_primary_cameras_failed_or_held"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--pilot-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--proxy", type=Path)
    review_prompt_cmd = sub.add_parser("write-review-prompt")
    review_prompt_cmd.add_argument("--camera", required=True)
    review_prompt_cmd.add_argument("--frames", type=int, nargs="+", required=True)
    review_prompt_cmd.add_argument("--structure", type=Path, required=True)
    review_prompt_cmd.add_argument("--output", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--prepare", type=Path, required=True)
    finalize.add_argument("--predictions", type=Path, required=True)
    finalize.add_argument("--response-dir", type=Path, required=True)
    finalize.add_argument("--pilot-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare_run(
            predictions_path=args.predictions,
            pilot_root=args.pilot_root,
            output_dir=args.output_dir,
            proxy_path=args.proxy,
        )
        print(json.dumps({"cameras": list(payload["cameras"]), "confirm_opened": False}, sort_keys=True))
    elif args.action == "write-review-prompt":
        try:
            structure = parse_structure(args.structure.read_text(encoding="utf-8"))
            cells = structure.get("cells") or ()
        except ValueError:
            cells = ()
        args.output.write_text(review_prompt(args.camera, args.frames, cells), encoding="utf-8")
    else:
        result = finalize_run(
            prepare_path=args.prepare,
            predictions_path=args.predictions,
            response_dir=args.response_dir,
            pilot_root=args.pilot_root,
            output_path=args.output,
            proxy_path=args.proxy,
        )
        print(json.dumps({"next_gate": result["next_gate"], "h_b1": result["h_b1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
