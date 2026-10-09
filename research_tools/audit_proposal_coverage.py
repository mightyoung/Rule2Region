"""Audit temporal/frame and spatial coverage of a label-blind ROI proposal route."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evidencegate.cascade import Box as CascadeBox
from research_tools.budgeted_cascade import _ambiguity_score, tight_candidate_rois
from research_tools.cascade_experiment import Box, Prediction, immutable_json_write, parse_mot17_gt
from research_tools.local_oracle_sweep import load_reference_frames, motion_residual_boxes
from research_tools.run_cascade_successor import _heuristic_rois


def _prediction(row: dict[str, Any]) -> Prediction:
    return Prediction(int(row["frame"]), Box(**row["box"]), float(row["confidence"]), str(row["source"]))


def audit_receipt(receipt: dict[str, Any], frames_dir: Path, gt_path: Path) -> dict[str, Any]:
    route = str(receipt["route"])
    if route not in {"heuristic", "motion"}:
        raise ValueError("proposal_audit_requires_label_blind_route")
    baseline = {
        int(row["frame"]): tuple(_prediction(item) for item in row["predictions"])
        for row in receipt["receipts"]["yolo11s_global"]
    }
    frame_ids = tuple(sorted(baseline))
    selected = set(int(value) for value in receipt["selected_frames"])
    truth = parse_mot17_gt(gt_path)
    by_truth: dict[int, list[Any]] = {}
    for item in truth:
        if item.frame in baseline:
            by_truth.setdefault(item.frame, []).append(item)
    missed = {
        frame: tuple(
            item.box for item in by_truth.get(frame, ())
            if not any(item.box.iou(pred.box) >= .5 for pred in baseline[frame])
        )
        for frame in frame_ids
    }
    frames = {
        frame: np.asarray(Image.open(frames_dir / f"{frame:06d}.jpg").convert("RGB"))
        for frame in frame_ids
    }
    references = load_reference_frames(frames_dir, frame_ids) if route == "motion" else None
    proposal_boxes: dict[int, tuple[Box, ...]] = {}
    motion_energy: dict[int, float] = {}
    for frame in frame_ids:
        if route == "heuristic":
            proposal_boxes[frame] = tuple(pred.box for pred in baseline[frame])
        else:
            proposal_boxes[frame], motion_energy[frame] = motion_residual_boxes(frames[frame], references[frame])  # type: ignore[index]
    candidates = {
        frame: tight_candidate_rois(
            frames[frame].shape[1], frames[frame].shape[0], proposal_boxes[frame],
            prefix="h" if route == "heuristic" else "m", expansion=0.0,
        )
        for frame in frame_ids
    }
    if route == "heuristic":
        roi = {frame: _heuristic_rois(baseline[frame], candidates[frame], 1)[0] for frame in frame_ids}
        score = {frame: _ambiguity_score(baseline[frame], candidates[frame]) for frame in frame_ids}
    else:
        roi = {
            frame: max(
                candidates[frame],
                key=lambda item: (
                    float(np.mean(np.abs(frames[frame][int(item.y1):int(item.y2), int(item.x1):int(item.x2)].astype(np.float32) - references[frame][int(item.y1):int(item.y2), int(item.x1):int(item.x2)].astype(np.float32)))),  # type: ignore[index]
                    item.roi_id,
                ),
            )
            for frame in frame_ids
        }
        score = motion_energy
    total_missed = sum(len(missed[frame]) for frame in frame_ids)
    selected_missed = sum(len(missed[frame]) for frame in selected)
    covered = sum(
        selected_roi.contains_center(CascadeBox(*box.coords))
        for frame in selected
        for box in missed[frame]
        for selected_roi in (roi[frame],)
    )
    return {
        "sequence_id": receipt["sequence_id"],
        "route": route,
        "frames": len(frame_ids),
        "selected_frames": len(selected),
        "total_missed_persons": total_missed,
        "missed_persons_on_selected_frames": selected_missed,
        "covered_missed_persons": covered,
        "temporal_capture_recall": selected_missed / total_missed if total_missed else 0.0,
        "spatial_recall_given_selected": covered / selected_missed if selected_missed else 0.0,
        "joint_proposal_recall": covered / total_missed if total_missed else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or len({row["route"] for row in rows}) != 1:
        raise ValueError("proposal_audit_rows_invalid")
    return {
        "schema_version": "fieldpack.proposal-coverage-audit.v1",
        "evidence_status": "ENGINEERING_DIAGNOSTIC_NOT_FORMAL_STATISTICS",
        "route": rows[0]["route"],
        "sequence_count": len(rows),
        "mean_temporal_capture_recall": statistics.mean(row["temporal_capture_recall"] for row in rows),
        "mean_spatial_recall_given_selected": statistics.mean(row["spatial_recall_given_selected"] for row in rows),
        "mean_joint_proposal_recall": statistics.mean(row["joint_proposal_recall"] for row in rows),
        "sequences": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit label-blind proposal coverage")
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for receipt_path in sorted(args.receipt_root.glob("VIRAT*.json")):
        sequence = receipt_path.stem
        sequence_dir = args.pilot_root / "sequences" / sequence
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        rows.append(audit_receipt(receipt, sequence_dir / "img1", sequence_dir / "gt.txt"))
    payload = summarize(rows)
    digest = immutable_json_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest, **{key: payload[key] for key in payload if key.startswith("mean_")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
