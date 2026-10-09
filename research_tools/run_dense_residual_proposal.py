"""Exploratory raw-head residual proposal sentinel on an already exposed split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.run_missed_person_proposal_baseline import FrameRecord, evaluate, load_records
from research_tools.yolo_dense_person_score import FrozenYoloDensePersonScore


def mask_existing_predictions(
    score: np.ndarray,
    record: FrameRecord,
    *,
    expansion: float = .10,
) -> np.ndarray:
    if score.ndim != 2 or expansion < 0:
        raise ValueError("invalid_dense_residual_mask_input")
    masked = score.copy()
    height, width = score.shape
    for box, _confidence in record.predictions:
        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1
        x1 = box.x1 - expansion * box_width
        x2 = box.x2 + expansion * box_width
        y1 = box.y1 - expansion * box_height
        y2 = box.y2 + expansion * box_height
        gx1 = max(0, min(width - 1, int(np.floor(x1 / record.width * width))))
        gx2 = max(gx1 + 1, min(width, int(np.ceil(x2 / record.width * width))))
        gy1 = max(0, min(height - 1, int(np.floor(y1 / record.height * height))))
        gy2 = max(gy1 + 1, min(height, int(np.ceil(y2 / record.height * height))))
        masked[gy1:gy2, gx1:gx2] = 0.0
    return masked


def run(
    records: list[FrameRecord],
    dense_scores: np.ndarray,
    *,
    expansion: float = .10,
    peak_count: int = 1,
    total_area: float = .095,
    budget: float = .20,
) -> dict[str, Any]:
    if dense_scores.shape != (len(records), 23, 40):
        raise ValueError("dense_scores_shape_mismatch")
    residual = np.stack([
        mask_existing_predictions(score, record, expansion=expansion)
        for score, record in zip(dense_scores, records)
    ])
    result = evaluate(
        records,
        residual,
        peak_count=peak_count,
        total_area=total_area,
        budget=budget,
    )
    return {
        "schema_version": "fieldpack.dense-residual-proposal-exploratory.v1",
        "evidence_status": "EXPLORATORY_MECHANISM_SENTINEL_NOT_CONFIRMATORY",
        "method": "pre_nms_person_logits_masked_by_existing_boxes",
        "configuration": {
            "expansion": expansion,
            "peak_count": peak_count,
            "total_area": total_area,
            "frame_budget": budget,
            "grid": [23, 40],
        },
        "online_inputs": ["frozen_yolo11s_raw_person_logits", "archived_yolo11s_predictions"],
        "gt_used_online": False,
        "strong_model_called": False,
        "result": result,
        "joint_proposal_recall": result["joint_proposal_recall"],
        "mean_roi_area": result["mean_roi_area"],
        "max_roi_area": result["max_roi_area"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run raw-head residual proposal sentinel")
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.receipt_root, args.pilot_root)
    with FrozenYoloDensePersonScore(args.small_model, device=args.device) as extractor:
        dense = extractor.extract(
            (record.rgb for record in records),
            output_width=40,
            output_height=23,
        )
    payload = run(records, dense)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("joint_proposal_recall", "mean_roi_area", "max_roi_area")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
