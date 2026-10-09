"""Knob pick and online conf for the A/F arms. No GPU."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from research_tools.execution_policy.compile import h1_pass


CONFS = (0.15, 0.25, 0.35)


def pick_knob(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no_knob_candidates")
    ranked: list[tuple[Any, ...]] = []
    for row in candidates:
        gate = h1_pass(
            compiled_mae=float(row["mae"]),
            b11_mae=float(row["b11_mae"]),
            precision_change_vs_b11=float(row["precision"]) - float(row["b11_precision"]),
            compiled_proxy=float(row.get("proxy") or (1.0 if row["model"] == "yolo26" else 0.25)),
            compiled_time_ms=float(row["time_ms"]),
            b26_time_ms=float(row["b26_time_ms"]),
        )
        ranked.append(
            (
                bool(gate["passed"]),
                -float(row["mae"]),
                float(row["precision"]),
                -float(row["time_ms"]),
                dict(row),
                gate,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    winner = ranked[0]
    out = dict(winner[4])
    out["h1"] = winner[5]
    return out


def next_conf(current: float, prev_count: float, history_mean: float) -> float:
    steps = list(CONFS)
    index = steps.index(current) if current in steps else 1
    if history_mean > 0 and prev_count > history_mean * 1.5 and index < len(steps) - 1:
        return steps[index + 1]
    if prev_count == 0 and history_mean > 0.5 and index > 0:
        return steps[index - 1]
    return current
