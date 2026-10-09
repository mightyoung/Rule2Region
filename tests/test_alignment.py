import pytest

from codecref_lite.alignment import (
    CandidateAlignmentError,
    align_candidate_pts,
    candidate_times,
)


def test_candidate_times_for_four_seconds_at_two_fps() -> None:
    assert candidate_times(10.0) == (10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5)


def test_alignment_uses_pts_and_unique_frames() -> None:
    decoded = [0.0, 0.04, 0.48, 0.52, 1.0]
    aligned = align_candidate_pts([0.0, 0.5, 1.0], decoded, 0.04)
    assert [item.decoded_pts_seconds for item in aligned] == [0.0, 0.48, 1.0]
    assert len({item.decoded_index for item in aligned}) == 3


def test_alignment_fails_without_frame_inside_tolerance() -> None:
    with pytest.raises(CandidateAlignmentError):
        align_candidate_pts([0.5], [0.4], 0.04)


def test_alignment_finds_global_ordered_solution() -> None:
    aligned = align_candidate_pts([0.05, 0.1], [0.0, 0.06], 0.1)
    assert [item.decoded_pts_seconds for item in aligned] == [0.0, 0.06]


def test_alignment_handles_long_decoded_sequence_without_recursion() -> None:
    decoded = [index / 30 for index in range(3000)]
    aligned = align_candidate_pts([0.0, 0.5, 1.0, 1.5], decoded, 1 / 30)
    assert [item.decoded_pts_seconds for item in aligned] == [0.0, 0.5, 1.0, 1.5]
