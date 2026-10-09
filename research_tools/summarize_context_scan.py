"""Aggregate a bounded context-expansion scan without treating frames as replicates."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from research_tools.cascade_experiment import immutable_json_write


SUMMARY_KEYS = (
    "count_mae",
    "ap50",
    "recall",
    "small_object_recall",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_fps",
    "model_input_tensor_ratio_to_small_global",
    "budget_fraction",
    "selected_frames",
    "total_compute_ms",
    "local_batch_total_ms",
)


def _mean(values: Iterable[float]) -> float:
    return float(statistics.mean(values))


def load_scan_receipts(root: Path) -> list[dict[str, Any]]:
    receipts = []
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "fieldpack.budgeted-cascade.v1":
            receipts.append(payload)
    if not receipts:
        raise ValueError("context_scan_receipts_missing")
    identities = [(float(item["context_expansion"]), str(item["sequence_id"])) for item in receipts]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_context_sequence_receipt")
    return receipts


def summarize_context_scan(
    receipts: Iterable[dict[str, Any]],
    *,
    expected_expansions: Iterable[float] | None = None,
    expected_sequence_count: int | None = None,
    expected_receipt_count: int | None = None,
) -> dict[str, Any]:
    rows = list(receipts)
    grouped: dict[float, list[dict[str, Any]]] = {}
    for receipt in rows:
        grouped.setdefault(float(receipt["context_expansion"]), []).append(receipt)
    if expected_expansions is not None:
        expected = {round(float(value), 9) for value in expected_expansions}
        observed = {round(value, 9) for value in grouped}
        if observed != expected:
            raise ValueError("context_scan_expansion_set_mismatch")
    if expected_receipt_count is not None and len(rows) != expected_receipt_count:
        raise ValueError("context_scan_receipt_count_mismatch")
    sequence_sets = {frozenset(str(item["sequence_id"]) for item in members) for members in grouped.values()}
    if len(sequence_sets) != 1:
        raise ValueError("context_scan_sequence_set_mismatch")
    sequence_count = len(next(iter(sequence_sets)))
    if expected_sequence_count is not None and sequence_count != expected_sequence_count:
        raise ValueError("context_scan_sequence_count_mismatch")
    metadata = {
        (
            int(item["frames"]), int(item["seed"]), tuple(float(v) for v in item["budgets"]),
            int(item["batch_size"]), int(item["global_imgsz"]), int(item["roi_imgsz"]),
        )
        for item in rows
    }
    if len(metadata) != 1:
        raise ValueError("context_scan_metadata_mismatch")
    expansions: dict[str, Any] = {}
    for expansion, members in sorted(grouped.items()):
        condition_names = set(members[0]["conditions"])
        if any(set(item["conditions"]) != condition_names for item in members):
            raise ValueError("context_scan_condition_mismatch")
        conditions: dict[str, Any] = {}
        for condition in sorted(condition_names):
            summary = {
                key: _mean(float(item["conditions"][condition][key]) for item in members)
                for key in SUMMARY_KEYS
            }
            selected_rois = [
                row["selected_roi"]
                for item in members
                for row in item["receipts"][condition]
                if row.get("selected_roi") is not None
            ]
            summary["selected_roi_count"] = len(selected_rois)
            summary["mean_core_area_ratio"] = (
                _mean(float(roi["core_area_ratio"]) for roi in selected_rois) if selected_rois else 0.0
            )
            summary["mean_crop_area_ratio"] = (
                _mean(float(roi["crop_area_ratio"]) for roi in selected_rois) if selected_rois else 0.0
            )
            summary["max_crop_area_ratio"] = (
                max(float(roi["crop_area_ratio"]) for roi in selected_rois) if selected_rois else 0.0
            )
            conditions[condition] = summary
        expansions[f"{expansion:.2f}"] = {
            "sequence_count": len(members),
            "sequence_ids": sorted(str(item["sequence_id"]) for item in members),
            "conditions": conditions,
        }
    return {
        "schema_version": "fieldpack.context-scan-summary.v1",
        "evidence_status": "ENGINEERING_PILOT_NOT_FORMAL_STATISTICS",
        "weighting": "equal_sequence_means",
        "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END",
        "oracle_scope": "GT_SELECTED_UPPER_BOUND_NOT_PAIRED_WITH_HEURISTIC",
        "receipt_count": len(rows),
        "expected_contract": {
            "expansions": sorted(float(value) for value in expected_expansions) if expected_expansions is not None else None,
            "sequence_count": expected_sequence_count,
            "receipt_count": expected_receipt_count,
        },
        "expansions": expansions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize tight-ROI context expansion scan")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-expansions", default="")
    parser.add_argument("--expected-sequence-count", type=int)
    parser.add_argument("--expected-receipt-count", type=int)
    args = parser.parse_args()
    expected_expansions = tuple(float(value) for value in args.expected_expansions.split(",") if value.strip())
    payload = summarize_context_scan(
        load_scan_receipts(args.root),
        expected_expansions=expected_expansions or None,
        expected_sequence_count=args.expected_sequence_count,
        expected_receipt_count=args.expected_receipt_count,
    )
    digest = immutable_json_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
