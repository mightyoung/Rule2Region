import json
from pathlib import Path

import pytest
from PIL import Image

from research_tools.select_dense_detection_frames import (
    OBSERVER_PRIORITY,
    choose_observer_config,
    materialize_native_scene_pack,
    score_frame,
    select_dense_frames,
    select_from_prediction_ledger,
)


def test_dense_selection_ranks_by_box_count_without_random_or_time_stride() -> None:
    rows = [
        {"frame": 10, "merged": [{}] * 1, "tile_raw": [{}] * 4, "global": [{}] * 1},
        {"frame": 11, "merged": [{}] * 5, "tile_raw": [{}] * 2, "global": [{}] * 1},
        {"frame": 12, "merged": [{}] * 5, "tile_raw": [{}] * 3, "global": [{}] * 1},
        {"frame": 13, "merged": [{}] * 2, "tile_raw": [{}] * 1, "global": [{}] * 2},
    ]
    selected = select_dense_frames(rows, allowed_frames={10, 11, 12, 13}, k=2)
    assert [row.frame for row in selected] == [12, 11]
    assert "seed" not in select_dense_frames.__code__.co_varnames


def test_dense_selection_excludes_evaluation_frames() -> None:
    rows = [
        {"frame": 1, "merged": [{}] * 9, "tile_raw": [], "global": []},
        {"frame": 2, "merged": [{}] * 3, "tile_raw": [], "global": []},
        {"frame": 3, "merged": [{}] * 2, "tile_raw": [], "global": []},
    ]
    selected = select_dense_frames(rows, allowed_frames={2, 3}, k=2)
    assert [row.frame for row in selected] == [2, 3]


def test_dense_selection_fails_closed_on_short_calibration_pool() -> None:
    rows = [{"frame": 1, "merged": [{}], "tile_raw": [], "global": []}]
    with pytest.raises(ValueError, match="insufficient_calibration_frames"):
        select_dense_frames(rows, allowed_frames={1}, k=2)


def test_observer_priority_prefers_full_yolo26_tile_config() -> None:
    assert choose_observer_config(["B0", "C04", "C15", "C16"]) == "C15"
    assert choose_observer_config(["B0", "C00", "C16"]) == "C16"
    assert OBSERVER_PRIORITY[0] == "C15"


def test_native_overlay_keeps_yolo_xyxy_in_same_pixel_space(tmp_path: Path) -> None:
    source = tmp_path / "000196.jpg"
    Image.new("RGB", (1280, 720), (10, 20, 30)).save(source, format="JPEG")
    box = {"x1": 100.0, "y1": 200.0, "x2": 180.0, "y2": 320.0}
    output = tmp_path / "overlay.jpg"
    receipt = materialize_native_scene_pack(
        source_path=source,
        output_clean=tmp_path / "clean.jpg",
        output_overlay=output,
        boxes=[{"box": box, "confidence": 0.9, "source": "yolo11s_global"}],
        box_id_prefix="b",
    )
    with Image.open(output) as image:
        assert image.size == (1280, 720)
        band = [image.getpixel((x, y)) for x in range(120, 161) for y in range(197, 204)]
        assert any(pixel[0] > 180 and pixel[1] > 140 for pixel in band)
        corner = image.getpixel((0, 0))
        assert corner[0] < 40 and corner[2] < 50
    assert receipt["boxes"][0]["xyxy"] == [100.0, 200.0, 180.0, 320.0]
    assert receipt["transform"] == "identity"
    assert receipt["width"] == 1280 and receipt["height"] == 720
    assert receipt["collage"] is False


def test_select_from_ledger_uses_calibration_only_and_records_counts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "camera_id": "VIRAT_S_010204",
            "config_id": "C04",
            "frames": [
                {
                    "camera_id": "VIRAT_S_010204",
                    "sequence_id": "VIRAT_S_010204_03_000606_000632",
                    "frame": 196,
                    "global": [{}] * 4,
                    "tile_raw": [{}],
                    "merged": [{}] * 4,
                },
                {
                    "camera_id": "VIRAT_S_010204",
                    "sequence_id": "VIRAT_S_010204_03_000606_000632",
                    "frame": 185,
                    "global": [{}] * 20,
                    "tile_raw": [{}] * 20,
                    "merged": [{}] * 20,
                },
                {
                    "camera_id": "VIRAT_S_010204",
                    "sequence_id": "VIRAT_S_010204_03_000606_000632",
                    "frame": 199,
                    "global": [{}] * 3,
                    "tile_raw": [],
                    "merged": [{}] * 3,
                },
            ],
        },
        {
            "camera_id": "VIRAT_S_010204",
            "config_id": "B0",
            "frames": [
                {
                    "camera_id": "VIRAT_S_010204",
                    "sequence_id": "VIRAT_S_010204_03_000606_000632",
                    "frame": 196,
                    "global": [{}],
                    "tile_raw": [],
                    "merged": [{}],
                }
            ],
        },
    ]
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    split = {
        "cameras": {
            "VIRAT_S_010204": {
                "sequence_id": "VIRAT_S_010204_03_000606_000632",
                "calibration_frames": [196, 199],
                "evaluation_frames": [185],
            }
        }
    }
    selected = select_from_prediction_ledger(
        predictions_path=predictions,
        frame_split=split,
        camera_id="VIRAT_S_010204",
        k=2,
    )
    assert selected["observer_config_id"] == "C04"
    assert [row.frame for row in selected["frames"]] == [196, 199]
    assert 185 not in {row.frame for row in selected["frames"]}
    assert score_frame(rows[0]["frames"][0]).merged_count == 4
