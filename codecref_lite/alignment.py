from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class CandidateAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class AlignedCandidate:
    candidate_seconds: float
    decoded_pts_seconds: float
    decoded_index: int
    absolute_error_seconds: float


def candidate_times(
    start_seconds: float, duration_seconds: float = 4.0, fps: float = 2.0
) -> tuple[float, ...]:
    if not all(
        math.isfinite(value) for value in (start_seconds, duration_seconds, fps)
    ):
        raise ValueError("candidate time inputs must be finite")
    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    raw_count = duration_seconds * fps
    count = int(round(raw_count))
    if count <= 0 or not math.isclose(raw_count, count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("duration multiplied by fps must be a positive integer")
    return tuple(start_seconds + index / fps for index in range(count))


def align_candidate_pts(
    candidates: Sequence[float],
    decoded_pts: Sequence[float],
    source_frame_period_seconds: float,
) -> tuple[AlignedCandidate, ...]:
    if source_frame_period_seconds <= 0:
        raise ValueError("source frame period must be positive")
    if not candidates or not all(
        math.isfinite(value) for value in (*candidates, *decoded_pts)
    ):
        raise ValueError(
            "candidate and decoded PTS values must be finite and non-empty"
        )
    if any(left >= right for left, right in zip(candidates, candidates[1:])):
        raise ValueError("candidate times must be strictly increasing")
    ordered_decoded = sorted(
        enumerate(decoded_pts), key=lambda item: (item[1], item[0])
    )
    if any(
        left[1] == right[1] for left, right in zip(ordered_decoded, ordered_decoded[1:])
    ):
        raise ValueError("decoded PTS values must be unique")
    tolerance = source_frame_period_seconds / 2.0

    def solution_key(option: tuple[float, tuple[int, ...]]) -> tuple[object, ...]:
        return (
            option[0],
            tuple(ordered_decoded[index][1] for index in option[1]),
            tuple(ordered_decoded[index][0] for index in option[1]),
        )

    states: list[tuple[float, tuple[int, ...]] | None] = [None] * (len(candidates) + 1)
    states[0] = (0.0, ())
    for decoded_index, (_, pts) in enumerate(ordered_decoded):
        for candidate_index in range(len(candidates) - 1, -1, -1):
            state = states[candidate_index]
            if state is None:
                continue
            error = abs(pts - candidates[candidate_index])
            if error > tolerance + 1e-12:
                continue
            proposal = (state[0] + error, (*state[1], decoded_index))
            current = states[candidate_index + 1]
            if current is None or solution_key(proposal) < solution_key(current):
                states[candidate_index + 1] = proposal

    solution = states[len(candidates)]
    if solution is None:
        raise CandidateAlignmentError("CANDIDATE_ALIGNMENT_FAILED")
    return tuple(
        AlignedCandidate(
            candidate_seconds=candidate,
            decoded_pts_seconds=ordered_decoded[assignment][1],
            decoded_index=ordered_decoded[assignment][0],
            absolute_error_seconds=abs(ordered_decoded[assignment][1] - candidate),
        )
        for candidate, assignment in zip(candidates, solution[1], strict=True)
    )
