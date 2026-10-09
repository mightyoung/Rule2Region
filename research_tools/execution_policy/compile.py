"""Partition boxes by region. YOLO11s and YOLO26 stay in separate lists."""
from __future__ import annotations

from dataclasses import dataclass

from research_tools.cascade_experiment import Prediction
from research_tools.execution_policy.card import PolicyCard, RegionCard, latency_proxy
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.residual_query import box_center, nms


NMS_IOU = 0.50
B11_PROXY = 0.25
B26_PROXY = 1.0
GLOBAL_CONF = 0.25


@dataclass(frozen=True)
class PartitionedCount:
    yolo11s: tuple[Prediction, ...]
    yolo26: tuple[Prediction, ...]

    def count(self) -> int:
        return len(self.yolo11s) + len(self.yolo26)

    def for_metrics(self) -> tuple[Prediction, ...]:
        return (*self.yolo11s, *self.yolo26)


def _owner(pred: Prediction, card: PolicyCard) -> RegionCard | None:
    x, y = box_center(pred.box)
    for region in card.regions:
        box = region.box
        if box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
            return region
    return None


def apply_card(card: PolicyCard, ledger: FrameLedger) -> PartitionedCount:
    kept_11: list[Prediction] = []
    for pred in ledger.yolo11s:
        owner = _owner(pred, card)
        if owner is None or owner.role == "exclude":
            continue
        if owner.model == "yolo11s" and pred.confidence >= owner.conf:
            kept_11.append(pred)
    kept_26: list[Prediction] = []
    for pred in ledger.yolo26:
        owner = _owner(pred, card)
        if owner is None or owner.role == "exclude" or owner.model != "yolo26":
            continue
        if pred.confidence >= owner.conf:
            kept_26.append(pred)
    return PartitionedCount(nms(kept_11, iou=NMS_IOU), nms(kept_26, iou=NMS_IOU))


def h1_pass(
    *,
    compiled_mae: float,
    b11_mae: float,
    precision_change_vs_b11: float,
    compiled_proxy: float,
    compiled_time_ms: float,
    b26_time_ms: float,
    oom: bool = False,
) -> dict[str, object]:
    mae_change = 0.0 if b11_mae == 0 else (b11_mae - compiled_mae) / b11_mae
    passed = (
        compiled_proxy < B26_PROXY
        and compiled_time_ms < b26_time_ms
        and mae_change >= -0.05
        and precision_change_vs_b11 >= -0.05
        and not oom
    )
    return {
        "passed": passed,
        "mae_change_vs_b11": mae_change,
        "precision_change_vs_b11": precision_change_vs_b11,
        "compiled_proxy": compiled_proxy,
        "b11_proxy": B11_PROXY,
        "b26_proxy": B26_PROXY,
        "verdict": "pass" if passed else "fail",
    }


def card_proxy(card: PolicyCard) -> float:
    return latency_proxy(card)
