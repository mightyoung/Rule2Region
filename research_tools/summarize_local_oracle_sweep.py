"""Validate and summarize the six-sequence local Oracle factorial matrix."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from research_tools.cascade_experiment import immutable_json_write


METRICS = ("count_mae", "ap50", "recall", "small_object_recall", "model_input_tensor_ratio_to_small_global", "model_compute_total_ms")


def load_receipts(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("VIRAT*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") in {"fieldpack.local-oracle-sweep.v1", "fieldpack.local-route-transfer.v1"}:
            rows.append(payload)
    if not rows:
        raise ValueError("local_oracle_sweep_receipts_missing")
    return rows


def summarize_local_oracle_sweep(
    receipts: Iterable[dict[str, Any]],
    *,
    expected_sequence_count: int,
    expected_roi_sizes: Iterable[int],
    expected_thresholds: Iterable[float],
    expected_fusions: Iterable[str],
    route: str = "oracle",
) -> dict[str, Any]:
    rows = list(receipts)
    if len(rows) != expected_sequence_count or len({row["sequence_id"] for row in rows}) != expected_sequence_count:
        raise ValueError("local_oracle_sequence_count_mismatch")
    expected_sizes = sorted(int(value) for value in expected_roi_sizes)
    expected_conf = sorted(float(value) for value in expected_thresholds)
    expected_fusion = sorted(str(value) for value in expected_fusions)
    metadata = {
        (tuple(row["roi_sizes"]), tuple(row["confidence_thresholds"]), tuple(row["fusions"]), row["budget"], row["context_expansion"], row["global_imgsz"], row["frames"], row["seed"])
        for row in rows
    }
    if len(metadata) != 1:
        raise ValueError("local_oracle_metadata_mismatch")
    first = rows[0]
    if first["roi_sizes"] != expected_sizes or first["confidence_thresholds"] != expected_conf or first["fusions"] != expected_fusion:
        raise ValueError("local_oracle_factor_set_mismatch")
    if route not in {"oracle", "heuristic", "motion"}:
        raise ValueError("local_route_invalid")
    if any(row.get("route", "oracle") != route for row in rows):
        raise ValueError("local_route_mismatch")
    expected_conditions = {"yolo11s_global"} | {
        f"{route}_i{size}_c{int(round(conf * 100)):02d}_{fusion}"
        for size in expected_sizes for conf in expected_conf for fusion in expected_fusion
    }
    if any(set(row["conditions"]) != expected_conditions for row in rows):
        raise ValueError("local_oracle_condition_matrix_incomplete")
    conditions = {
        condition: {metric: float(statistics.mean(float(row["conditions"][condition][metric]) for row in rows)) for metric in METRICS}
        for condition in sorted(expected_conditions)
    }
    baseline = conditions["yolo11s_global"]["count_mae"]
    candidates = [(values["count_mae"], -values["ap50"], name) for name, values in conditions.items() if name != "yolo11s_global"]
    _, _, winner = min(candidates)
    improvement = (baseline - conditions[winner]["count_mae"]) / baseline if baseline else 0.0
    ap50_delta = conditions[winner]["ap50"] - conditions["yolo11s_global"]["ap50"]
    mae_passed = improvement >= .10
    ap50_passed = ap50_delta >= 0.0
    return {
        "schema_version": "fieldpack.local-oracle-sweep-summary.v1",
        "evidence_status": "ENGINEERING_PILOT_NOT_FORMAL_STATISTICS",
        "route": route,
        "route_scope": "GT_SELECTED_UPPER_BOUND_NOT_DEPLOYABLE" if route == "oracle" else "LABEL_BLIND_DEPLOYABLE_CANDIDATE",
        "weighting": "equal_sequence_means",
        "sequence_count": len(rows),
        "factor_contract": {"roi_sizes": expected_sizes, "confidence_thresholds": expected_conf, "fusions": expected_fusion, "context_expansion": 0.0, "budget": .20},
        "conditions": conditions,
        "winner": winner,
        "winner_mae_improvement_fraction": improvement,
        "winner_ap50_delta": ap50_delta,
        "quality_gate": {
            "mae_improvement_threshold": .10,
            "mae_passed": mae_passed,
            "ap50_non_degradation_threshold": 0.0,
            "ap50_passed": ap50_passed,
            "passed": mae_passed and ap50_passed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize local Oracle factorial sweep")
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sequence-count", type=int, default=6)
    parser.add_argument("--roi-sizes", default="384,512,640"); parser.add_argument("--confidence-thresholds", default="0.10,0.25,0.50"); parser.add_argument("--fusions", default="nms,wbf")
    parser.add_argument("--route", choices=("oracle", "heuristic", "motion"), default="oracle")
    args = parser.parse_args()
    payload = summarize_local_oracle_sweep(
        load_receipts(args.root), expected_sequence_count=args.expected_sequence_count,
        expected_roi_sizes=(int(value) for value in args.roi_sizes.split(",")),
        expected_thresholds=(float(value) for value in args.confidence_thresholds.split(",")),
        expected_fusions=(value.strip() for value in args.fusions.split(",")),
        route=args.route,
    )
    digest = immutable_json_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest, "winner": payload["winner"], "gate": payload["quality_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
