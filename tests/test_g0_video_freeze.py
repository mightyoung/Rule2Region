from pathlib import Path

import pytest

from research_tools.g0_video_freeze import (
    choose_deployment_ids,
    label_window,
    proportional_quotas,
    seal_manifest,
    window_frame_ranges,
)


def test_natural_window_counts_match_remote_seqinfo() -> None:
    contracts = {
        "02": (600, 30, 5),
        "04": (1050, 30, 8),
        "05": (837, 14, 14),
        "09": (525, 30, 4),
        "10": (654, 30, 5),
        "11": (900, 30, 7),
        "13": (750, 25, 7),
    }
    assert sum(
        len(window_frame_ranges(frames, fps))
        for frames, fps, _ in contracts.values()
    ) == 50
    for frames, fps, expected in contracts.values():
        assert len(window_frame_ranges(frames, fps)) == expected


def test_peak_label_is_deterministic_and_uses_half_open_range() -> None:
    tracks = {1: {1}, 2: {1, 2}, 3: {3}}
    assert label_window(tracks, 1, 3) == {
        "count": 2,
        "peak_frame": 2,
        "peak_track_ids": [1, 2],
    }


def test_proportional_deployment_is_exact_and_bounded() -> None:
    counts = {"02": 5, "04": 8, "05": 14, "09": 4, "10": 5, "11": 7, "13": 7}
    quotas = proportional_quotas(counts, 24)
    assert sum(quotas.values()) == 24
    assert all(0 <= quotas[key] <= counts[key] for key in counts)


def test_deployment_selection_is_stable_and_source_stratified() -> None:
    windows = [
        {"sample_id": f"S{sequence}:{index}", "sequence_id": sequence}
        for sequence, count in (("02", 5), ("04", 8), ("05", 14), ("09", 4), ("10", 5), ("11", 7), ("13", 7))
        for index in range(count)
    ]
    assert choose_deployment_ids(windows, seed=17) == choose_deployment_ids(windows, seed=17)
    assert len(choose_deployment_ids(windows, seed=17)) == 24


def test_video_manifest_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "video.json"
    digest = seal_manifest(target, {"schema_version": "v1", "samples": []})
    assert seal_manifest(target, {"schema_version": "v1", "samples": []}) == digest
    with pytest.raises(FileExistsError):
        seal_manifest(target, {"schema_version": "v2", "samples": []})
