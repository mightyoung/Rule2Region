import pytest

from research_tools.prepare_virat_microclips import contiguous_midpoint_indices, select_for_cameras


def test_contiguous_midpoint_indices_are_causal_clip_compatible() -> None:
    assert contiguous_midpoint_indices(100, 8) == tuple(range(46, 54))
    assert contiguous_midpoint_indices(8, 8) == tuple(range(8))


def test_select_for_cameras_is_label_blind_and_deterministic() -> None:
    members = [
        f"videos/VIRAT_S_{camera:06d}_{clip:02d}_000000_000100.mp4"
        for camera in (4, 111, 113)
        for clip in range(4)
    ]
    selected = select_for_cameras(
        members, ("VIRAT_S_000004", "VIRAT_S_000111"), videos_per_camera=2
    )
    assert {camera for camera, _ in selected} == {"VIRAT_S_000004", "VIRAT_S_000111"}
    assert all("000113" not in member for _, member in selected)


def test_contiguous_midpoint_rejects_short_video() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        contiguous_midpoint_indices(7, 8)
