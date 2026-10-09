import numpy as np

from research_tools.cascade_experiment import Box, Prediction
from research_tools.run_scene_adaptive_small_track_ablation import (
    SceneFrame,
    fit_camera_scale_prior,
    fixed_tiles,
    nms,
    recover_one_frame_gaps,
    selected_tiles,
    static_two_tile_oracle,
)


def test_fixed_tiles_cover_image_with_unique_cores() -> None:
    tiles = fixed_tiles(1280, 720)
    assert len(tiles) == 4
    assert sum(tile.core.area for tile in tiles) == 1280 * 720
    assert all(tile.crop.area >= tile.core.area for tile in tiles)


def test_nms_prefers_higher_confidence_across_global_and_tile() -> None:
    low = Prediction(1, Box(0, 0, 10, 10), .5, "global")
    high = Prediction(1, Box(0, 0, 10, 10), .9, "tile")
    assert nms((low, high)) == (high,)


def test_scale_prior_routes_sparse_or_small_bands_without_gt() -> None:
    record = SceneFrame("seq", "VIRAT_S_010004", 1, 1280, 720, None, (Prediction(1, Box(0, 10, 10, 30), .8),), ())  # type: ignore[arg-type]
    prior = fit_camera_scale_prior([record])
    assert all(row["route_high_resolution"] for row in prior["VIRAT_S_010004"])
    assert selected_tiles(prior, "VIRAT_S_010004", 1280, 720)


def test_gap_recovery_requires_support_on_both_sides_and_avoids_duplicate() -> None:
    previous = Prediction(1, Box(0, 0, 10, 10), .9)
    future = Prediction(3, Box(1, 0, 11, 10), .8)
    recovered = recover_one_frame_gaps({1: (previous,), 2: (), 3: (future,)})
    assert len(recovered[2]) == 1
    assert recovered[2][0].source == "track_gap_fill"
    current = Prediction(2, Box(.5, 0, 10.5, 10), .7)
    no_duplicate = recover_one_frame_gaps({1: (previous,), 2: (current,), 3: (future,)})
    assert no_duplicate[2] == (current,)


def test_gap_recovery_does_not_fill_without_contiguous_frames() -> None:
    rows = {1: (Prediction(1, Box(0, 0, 10, 10), .9),), 3: (), 5: (Prediction(5, Box(0, 0, 10, 10), .9),)}
    assert recover_one_frame_gaps(rows)[3] == ()


def test_static_oracle_is_bounded_to_two_tiles_per_camera() -> None:
    records = []
    rows = {}
    for camera in ("VIRAT_S_010004", "VIRAT_S_010111", "VIRAT_S_010112", "VIRAT_S_010204"):
        record = SceneFrame(camera, camera, 1, 100, 100, None, (), ())  # type: ignore[arg-type]
        records.append(record)
        rows[(camera, 1)] = (Prediction(1, Box(0, 0, 10, 10), .9, "yolo26_hr_tile:t0"),)
    chosen, output = static_two_tile_oracle(records, rows)
    assert all(len(value) <= 2 for value in chosen.values())
    assert len(output) == 4
