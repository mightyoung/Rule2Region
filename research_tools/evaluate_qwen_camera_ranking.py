from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    no_worse = (
        left["precision"] >= right["precision"]
        and left["recall"] >= right["recall"]
        and left["count_mae"] <= right["count_mae"]
        and left["p95_ms"] <= right["p95_ms"]
    )
    strictly_better = (
        left["precision"] > right["precision"]
        or left["recall"] > right["recall"]
        or left["count_mae"] < right["count_mae"]
        or left["p95_ms"] < right["p95_ms"]
    )
    return no_worse and strictly_better


def evaluate_camera_ranking(catalog_path: Path, response_path: Path) -> dict[str, Any]:
    catalog_text = catalog_path.read_text(encoding="utf-8")
    if "evaluation_metrics" in catalog_text or "evaluation_frames" in catalog_text:
        raise ValueError("heldout_evaluation_leaked_into_ranking_evaluator")
    catalog = json.loads(catalog_text)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    legal = catalog["eligible_config_ids"]
    ranked = response["ranked_config_ids"]
    if response["abstain"] or not ranked:
        raise ValueError("ranking_response_abstained")
    if len(ranked) != len(set(ranked)) or not set(ranked) <= set(legal):
        raise ValueError("ranking_response_contains_illegal_config")
    metrics: dict[str, dict[str, float]] = {}
    for config in catalog["configs"]:
        config_id = config["config_id"]
        if config_id not in legal:
            continue
        replay = config["camera_replay"]
        calibration = replay["calibration_metrics"]
        metrics[config_id] = {
            "precision": float(calibration["precision"]),
            "recall": float(calibration["recall"]),
            "count_mae": float(calibration["count_mae"]),
            "p95_ms": float(replay["p95_ms"]),
        }
    dominated_by = {
        config_id: sorted(other for other in legal if other != config_id and _dominates(metrics[other], values))
        for config_id, values in metrics.items()
    }
    frontier = [config_id for config_id in legal if not dominated_by[config_id]]
    latency_minimizer = min(legal, key=lambda config_id: (metrics[config_id]["p95_ms"], config_id))
    top1 = ranked[0]
    return {
        "schema_version": "sceneconfig.qwen-camera-calibration-ranking-audit.v1",
        "camera_id": catalog["camera_id"],
        "evidence_status": "DEVELOPMENT_ONLY_CALIBRATION_ONLY",
        "holdout_opened": False,
        "qwen_ranked_config_ids": ranked,
        "qwen_top1": top1,
        "pareto_frontier_config_ids": frontier,
        "dominated_by": dominated_by,
        "ranked_dominated_config_ids": [config_id for config_id in ranked if dominated_by[config_id]],
        "deterministic_latency_minimizer": latency_minimizer,
        "qwen_top1_is_pareto_valid": top1 in frontier,
        "qwen_top1_matches_latency_minimizer": top1 == latency_minimizer,
        "incremental_value_established": False,
        "incremental_value_reason": "No pre-registered scalar utility distinguishes the Pareto frontier, and Qwen top-1 equals the deterministic latency minimizer.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    result = evaluate_camera_ranking(args.catalog, args.response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
