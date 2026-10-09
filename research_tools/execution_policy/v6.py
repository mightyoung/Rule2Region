"""v6 conservative hard-path gate. Same h1_pass floor; no theoretical claims."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from research_tools.execution_policy.stack import MAE_TOLERANCE, decide_hard_path

HISTORY_BLOCKS = 4
METHODS = ("always_closed", "always_open", "mae_only", "v5_point", "v6_conservative")
ADMISSION_CELLS = ("false_admission", "true_admission", "true_rejection", "false_rejection")


def split_contiguous_blocks(frames: Sequence[int], n: int = HISTORY_BLOCKS) -> tuple[tuple[int, ...], ...]:
    ordered = [int(frame) for frame in frames]
    if n < 1 or len(ordered) < n:
        raise ValueError("insufficient_history_for_v6_blocks")
    if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
        raise ValueError("history_frames_not_strictly_increasing")
    size = len(ordered) // n
    if size < 1:
        raise ValueError("insufficient_history_for_v6_blocks")
    blocks: list[tuple[int, ...]] = []
    for index in range(n):
        start = index * size
        end = len(ordered) if index == n - 1 else (index + 1) * size
        chunk = tuple(ordered[start:end])
        if not chunk:
            raise ValueError("insufficient_history_for_v6_blocks")
        blocks.append(chunk)
    return tuple(blocks)


def conservative_bounds(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(blocks) < HISTORY_BLOCKS:
        return {
            "passed": False,
            "reason": "insufficient_history_blocks",
            "block_count": len(blocks),
            "mae_change_lower_bound": None,
            "precision_change_lower_bound": None,
            "time_ms_upper_bound": None,
            "b26_time_ms_lower_bound": None,
        }
    mae_bound = min(float(row["mae_change_vs_b11"]) for row in blocks)
    precision_bound = min(float(row["precision_change_vs_b11"]) for row in blocks)
    time_upper = max(float(row["time_ms"]) for row in blocks)
    b26_time_lower = min(float(row["b26_time_ms"]) for row in blocks)
    time_ok = all(float(row["time_ms"]) < float(row["b26_time_ms"]) for row in blocks)
    passed = mae_bound >= MAE_TOLERANCE and precision_bound >= MAE_TOLERANCE and time_ok
    if mae_bound < MAE_TOLERANCE:
        reason = "mae_bound_below_floor"
    elif precision_bound < MAE_TOLERANCE:
        reason = "precision_bound_below_floor"
    elif not time_ok:
        reason = "time_not_strictly_below_b26"
    else:
        reason = "all_conservative_bounds_pass"
    return {
        "passed": passed,
        "reason": reason,
        "block_count": len(blocks),
        "mae_change_lower_bound": mae_bound,
        "precision_change_lower_bound": precision_bound,
        "time_ms_upper_bound": time_upper,
        "b26_time_ms_lower_bound": b26_time_lower,
    }


def conservative_ok(blocks: Sequence[Mapping[str, Any]]) -> bool:
    return bool(conservative_bounds(blocks)["passed"])


def decide_v6_hard(
    *,
    point_open: bool,
    conservative: bool,
    foci_count: int,
    knob_model: str,
) -> bool:
    if foci_count <= 0 or knob_model != "yolo11s":
        return False
    return bool(point_open) and bool(conservative)


def admission_cell(*, opened: bool, future_opened_pass: bool) -> str:
    if opened and not future_opened_pass:
        return "false_admission"
    if opened and future_opened_pass:
        return "true_admission"
    if (not opened) and (not future_opened_pass):
        return "true_rejection"
    return "false_rejection"


def count_cells(cells: Sequence[str]) -> dict[str, int]:
    unknown = sorted(set(cells) - set(ADMISSION_CELLS))
    if unknown:
        raise ValueError(f"unknown_admission_cells:{','.join(unknown)}")
    return {key: sum(1 for cell in cells if cell == key) for key in ADMISSION_CELLS}


def policy_yolo26_calls(*, frame_count: int, knob_model: str, opened: bool, foci_count: int) -> int:
    if frame_count < 0 or foci_count < 0:
        raise ValueError("negative_policy_call_inputs")
    full_frame_calls = frame_count if knob_model == "yolo26" else 0
    crop_calls = frame_count * foci_count if opened and knob_model == "yolo11s" else 0
    return full_frame_calls + crop_calls


def summarize_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = [str(row["cell"]) for row in rows]
    passed = sum(1 for row in rows if bool(row["future_h1_passed"]))
    count = len(rows)
    return {
        **count_cells(cells),
        "view_count": count,
        "future_h1_pass_count": passed,
        "future_h1_pass_rate": (passed / count) if count else None,
        "policy_model_time_ms_mean": (
            sum(float(row["policy_model_time_ms_mean"]) for row in rows) / count if count else None
        ),
        "yolo26_call_count": sum(int(row["yolo26_call_count"]) for row in rows),
    }


def point_open_from_rehearsal(
    *,
    foci_count: int,
    knob_model: str,
    closed_mae: float,
    opened: Mapping[str, Any],
) -> bool:
    return decide_hard_path(
        foci_count=foci_count,
        knob_model=knob_model,
        closed_mae=closed_mae,
        opened=opened,
    )
