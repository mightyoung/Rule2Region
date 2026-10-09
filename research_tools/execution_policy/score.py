"""E1 three-way scores from an aligned dual ledger (S1: filter, not crop)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import GroundTruth, Prediction, compute_metrics
from research_tools.execution_policy.card import PolicyCard, latency_proxy
from research_tools.execution_policy.compile import apply_card, h1_pass
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.roi_structure_protocol import count_mae


def _precision(preds: Sequence[Prediction], truth: Sequence[GroundTruth]) -> float:
    if not truth and not preds:
        return 1.0
    if not truth:
        return 0.0
    return float(compute_metrics(list(preds), list(truth))["precision"])


def _arm_metrics(
    frames: Sequence[int],
    pred_by_frame: Mapping[int, Sequence[Prediction]],
    truth: Sequence[GroundTruth],
) -> dict[str, float]:
    allowed = set(frames)
    truth_boxes = [row for row in truth if row.frame in allowed]
    truth_counts = {frame: sum(1 for row in truth_boxes if row.frame == frame) for frame in frames}
    pred_counts = {frame: len(pred_by_frame.get(frame, ())) for frame in frames}
    preds = [row for frame in frames for row in pred_by_frame.get(frame, ())]
    return {
        "mae": count_mae(pred_counts, truth_counts, frames),
        "precision": _precision(preds, truth_boxes),
        "pred_box_count": float(len(preds)),
        "truth_box_count": float(len(truth_boxes)),
    }


def compare_three(
    card: PolicyCard,
    ledgers: Mapping[int, FrameLedger],
    frames: Sequence[int],
    truth: Sequence[GroundTruth],
    *,
    compiled_time_ms: float = 0.0,
    b26_time_ms: float = 1.0,
    scoring: str = "ledger_filter",
) -> dict[str, object]:
    if scoring != "ledger_filter":
        raise ValueError("e1_only_supports_ledger_filter_s1")
    b11 = {frame: ledgers[frame].yolo11s for frame in frames}
    b26 = {frame: ledgers[frame].yolo26 for frame in frames}
    compiled = {frame: apply_card(card, ledgers[frame]).for_metrics() for frame in frames}
    b11_m = _arm_metrics(frames, b11, truth)
    b26_m = _arm_metrics(frames, b26, truth)
    compiled_m = _arm_metrics(frames, compiled, truth)
    proxy = latency_proxy(card)
    gate = h1_pass(
        compiled_mae=compiled_m["mae"],
        b11_mae=b11_m["mae"],
        precision_change_vs_b11=compiled_m["precision"] - b11_m["precision"],
        compiled_proxy=proxy,
        compiled_time_ms=compiled_time_ms,
        b26_time_ms=b26_time_ms,
    )
    return {
        "schema_version": "sceneconfig.execution-policy.e1.v1",
        "scoring": "ledger_filter_s1",
        "val_score": "crop_infer_not_run",
        "b11": b11_m,
        "b26": b26_m,
        "compiled": {**compiled_m, "latency_proxy": proxy},
        "h1": gate,
    }


def compare_camera(
    card: PolicyCard,
    sources: Sequence[Mapping[str, Any]],
    *,
    frames_key: str = "history",
    compiled_time_ms: float = 0.0,
    b26_time_ms: float = 1.0,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("camera_sources_empty")
    scored = [
        compare_three(
            card,
            source["ledgers"],
            list(source[frames_key]),
            source["truth"],
            compiled_time_ms=compiled_time_ms,
            b26_time_ms=b26_time_ms,
        )
        for source in sources
    ]

    def mean_arm(key: str) -> dict[str, float]:
        return {
            "mae": sum(float(row[key]["mae"]) for row in scored) / len(scored),
            "precision": sum(float(row[key]["precision"]) for row in scored) / len(scored),
            "pred_box_count": sum(float(row[key]["pred_box_count"]) for row in scored),
            "truth_box_count": sum(float(row[key]["truth_box_count"]) for row in scored),
        }

    compiled_m = mean_arm("compiled")
    b11_m = mean_arm("b11")
    b26_m = mean_arm("b26")
    proxy = latency_proxy(card)
    compiled_m["latency_proxy"] = proxy
    gate = h1_pass(
        compiled_mae=compiled_m["mae"],
        b11_mae=b11_m["mae"],
        precision_change_vs_b11=compiled_m["precision"] - b11_m["precision"],
        compiled_proxy=proxy,
        compiled_time_ms=compiled_time_ms,
        b26_time_ms=b26_time_ms,
    )
    return {
        "schema_version": "sceneconfig.execution-policy.e1.v1",
        "scoring": "ledger_filter_s1",
        "scope": "camera",
        "val_score": "crop_infer_not_run",
        "b11": b11_m,
        "b26": b26_m,
        "compiled": compiled_m,
        "h1": gate,
        "sequences": [str(source["sequence"]) for source in sources],
    }
