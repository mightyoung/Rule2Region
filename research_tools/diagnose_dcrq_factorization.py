"""Post-failure diagnostic: separate DCRQ frame utility from static spatial anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.missed_person_proposal import fit_pointwise_ridge, predict_pointwise_ridge
from research_tools.run_dcrq_development_gate import (
    EXPECTED_CAMERAS,
    LoadedObservations,
    _indices,
    _standardize_fit,
    _targets,
    load_observations,
)
from research_tools.run_missed_person_proposal_baseline import evaluate


def factorized_scores(frame_scores: np.ndarray, spatial_scores: np.ndarray) -> np.ndarray:
    """Encode one map's frame utility and another map's spatial argmax."""
    if frame_scores.shape != spatial_scores.shape or frame_scores.ndim != 3:
        raise ValueError("factorized_score_shape_mismatch")
    utility = frame_scores.max(axis=(1, 2), keepdims=True)
    anchored = spatial_scores - spatial_scores.max(axis=(1, 2), keepdims=True)
    return (anchored + utility).astype(np.float32)


def diagnose(loaded: LoadedObservations) -> dict[str, Any]:
    records = loaded.records
    targets = _targets(records, *loaded.online_features.shape[-2:])
    rows: list[dict[str, Any]] = []
    for held_out in EXPECTED_CAMERAS:
        train_indices = _indices(records, held_out, equal=False)
        evaluation_indices = _indices(records, held_out, equal=True)
        fit_features, evaluation_features = _standardize_fit(
            loaded.online_features[train_indices], loaded.online_features[evaluation_indices]
        )
        weights = fit_pointwise_ridge(fit_features, targets[train_indices], l2=1e-2, positive_weight=20.0)
        dcrq = predict_pointwise_ridge(evaluation_features, weights)
        dcrq = np.where(loaded.occupancy[evaluation_indices] < .5, dcrq, 0.0)
        static = loaded.static_scores[evaluation_indices]
        hybrid = factorized_scores(dcrq, static)
        evaluation_records = [records[index] for index in evaluation_indices]
        kwargs = {"peak_count": 1, "total_area": .095, "budget": .20}
        rows.append({
            "camera": held_out,
            "dcrq": evaluate(evaluation_records, dcrq, **kwargs),
            "static": evaluate(evaluation_records, static, **kwargs),
            "factorized_dcrq_frames_static_anchor": evaluate(evaluation_records, hybrid, **kwargs),
        })
    means = {
        key: float(np.mean([row[key]["joint_proposal_recall"] for row in rows]))
        for key in ("dcrq", "static", "factorized_dcrq_frames_static_anchor")
    }
    hybrid_spatial = float(np.mean([
        row["factorized_dcrq_frames_static_anchor"]["spatial_recall_given_selected"] for row in rows
    ]))
    static_spatial = float(np.mean([row["static"]["spatial_recall_given_selected"] for row in rows]))
    return {
        "schema_version": "fieldpack.dcrq-factorization-diagnostic.v1",
        "evidence_status": "POST_FAILURE_EXPLORATORY_DIAGNOSTIC_NOT_A_GATE",
        "configuration_search": False,
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
        "description": "DCRQ heatmap max selects frames; same-frame static raw-head argmax selects the single ROI center",
        "per_camera": rows,
        "mean_joint_proposal_recall": means,
        "hybrid_relative_to_static": means["factorized_dcrq_frames_static_anchor"] / means["static"] if means["static"] else 0.0,
        "mean_spatial_recall_given_selected": {
            "factorized": hybrid_spatial,
            "static": static_spatial,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    payload = diagnose(load_observations(args.cache_root, args.pilot_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "mean_joint_proposal_recall": payload["mean_joint_proposal_recall"],
        "hybrid_relative_to_static": payload["hybrid_relative_to_static"],
        "mean_spatial_recall_given_selected": payload["mean_spatial_recall_given_selected"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
