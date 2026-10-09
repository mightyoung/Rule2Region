"""History K=3 loop with a rule-based stand-in main agent (E3, no VLM)."""
from __future__ import annotations

from typing import Mapping, Sequence

from research_tools.cascade_experiment import GroundTruth
from research_tools.execution_policy.card import PolicyCard, parse_policy_card
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.execution_policy.score import compare_three


K = 3


def initial_card() -> PolicyCard:
    return parse_policy_card(
        {
            "regions": [
                {"region_id": "far", "xyxy": [0, 0, 1280, 360], "model": "yolo26", "conf": 0.25},
                {"region_id": "near", "xyxy": [0, 360, 1280, 720], "model": "yolo11s", "conf": 0.25},
            ]
        }
    )


def apply_one_class_change(card: PolicyCard, round_index: int) -> PolicyCard:
    payload = {
        "regions": [
            {
                "region_id": row.region_id,
                "xyxy": list(row.box.coords),
                "model": row.model,
                "conf": row.conf,
                "role": row.role,
            }
            for row in card.regions
        ]
    }
    if round_index == 1:
        for row in payload["regions"]:
            if row["region_id"] == "far":
                row["conf"] = 0.15
                break
    elif round_index == 2:
        for row in payload["regions"]:
            if row["region_id"] == "near":
                row["conf"] = 0.35
                break
    return parse_policy_card(payload)


def run_history_loop(
    ledgers: Mapping[int, FrameLedger],
    history_frames: Sequence[int],
    truth: Sequence[GroundTruth],
) -> dict[str, object]:
    card = initial_card()
    rounds: list[dict[str, object]] = []
    frozen = None
    for index in range(K):
        if index > 0 and frozen is None:
            card = apply_one_class_change(card, index)
        scored = compare_three(
            card,
            ledgers,
            history_frames,
            truth,
            compiled_time_ms=0.4,
            b26_time_ms=1.0,
        )
        rounds.append({"round": index, "card_regions": len(card.regions), "h1": scored["h1"]})
        if scored["h1"]["passed"]:
            frozen = card
            break
    if frozen is None:
        frozen = card
    return {
        "schema_version": "sceneconfig.execution-policy.e3-loop.v1",
        "k": K,
        "rounds": rounds,
        "intention_card": {
            "regions": [
                {
                    "region_id": row.region_id,
                    "xyxy": list(row.box.coords),
                    "model": row.model,
                    "conf": row.conf,
                    "role": row.role,
                }
                for row in frozen.regions
            ]
        },
        "early_stop": any(row["h1"]["passed"] for row in rounds[:-1]) if rounds else False,
        "scoring": "ledger_filter_s1",
    }
