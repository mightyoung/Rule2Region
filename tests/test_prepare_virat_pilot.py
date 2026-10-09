from research_tools.prepare_virat_pilot import (
    _person_rows,
    annotation_member_map,
    evenly_spaced_indices,
    select_video_members,
)


def test_evenly_spaced_indices_are_deterministic_midpoints() -> None:
    assert evenly_spaced_indices(10, 2) == (2, 7)
    assert evenly_spaced_indices(26, 3) == (4, 12, 21)


def test_select_video_members_is_label_blind_and_camera_blocked() -> None:
    members = [
        f"videos/VIRAT_S_{camera:06d}_{clip:02d}_000000_000100.mp4"
        for camera in range(5)
        for clip in range(4)
    ]
    selected = select_video_members(members, camera_count=3, videos_per_camera=2)
    assert selected == (
        ("VIRAT_S_000000", "videos/VIRAT_S_000000_00_000000_000100.mp4"),
        ("VIRAT_S_000000", "videos/VIRAT_S_000000_02_000000_000100.mp4"),
        ("VIRAT_S_000002", "videos/VIRAT_S_000002_00_000000_000100.mp4"),
        ("VIRAT_S_000002", "videos/VIRAT_S_000002_02_000000_000100.mp4"),
        ("VIRAT_S_000004", "videos/VIRAT_S_000004_00_000000_000100.mp4"),
        ("VIRAT_S_000004", "videos/VIRAT_S_000004_02_000000_000100.mp4"),
    )


def test_select_video_members_excludes_cameras_before_midpoint_selection() -> None:
    members = [
        f"videos/VIRAT_S_{camera:06d}_{clip:02d}_000000_000100.mp4"
        for camera in range(5)
        for clip in range(2)
    ]
    selected = select_video_members(
        members,
        camera_count=1,
        videos_per_camera=2,
        excluded_cameras=("VIRAT_S_000002",),
    )
    assert {camera for camera, _ in selected} == {"VIRAT_S_000001"}


def test_annotation_member_map_prefers_stable_first_path() -> None:
    members = [
        "annotations/VIRAT_S_000001_00_000000_000100.viratdata.objects.txt",
        "duplicates/VIRAT_S_000001_00_000000_000100.viratdata.objects.txt",
    ]
    mapping = annotation_member_map(members)
    assert mapping["VIRAT_S_000001_00_000000_000100"] == members[0]


def test_person_rows_map_zero_based_virat_track_ids_to_positive_mot_ids() -> None:
    rows = _person_rows("0 1 7 1 2 3 4 1\n", {7})
    assert rows[7] == ["8,1,1.0,2.0,3.0,4.0,1,1,1"]
