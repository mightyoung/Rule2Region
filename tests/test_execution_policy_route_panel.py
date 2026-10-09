import numpy as np

from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.card import parse_policy_card
from research_tools.execution_policy.crop_infer import adaptive_imgsz, exclusive_mix, mosaic_to_preds, pack_mosaic
from research_tools.execution_policy.route_select import next_conf, pick_knob


def test_adaptive_imgsz_follows_side_and_caps() -> None:
    assert adaptive_imgsz(Box(0, 0, 80, 130)) == 160
    assert adaptive_imgsz(Box(0, 0, 10, 10)) == 32
    assert adaptive_imgsz(Box(0, 0, 900, 700)) == 640


def test_pack_mosaic_and_map_back() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[10:50, 10:40] = 20
    card = parse_policy_card(
        {
            "regions": [
                {"region_id": "a", "xyxy": [10, 10, 40, 50], "model": "yolo26", "conf": 0.25},
                {"region_id": "b", "xyxy": [200, 100, 260, 180], "model": "yolo26", "conf": 0.25},
            ]
        }
    )
    packed = pack_mosaic(image, card.regions)
    assert packed is not None
    canvas, placements = packed
    assert canvas.shape[0] >= 40
    assert len(placements) == 2
    first = placements[0]
    local = Box(first["dx"] + 2, first["dy"] + 2, first["dx"] + 12, first["dy"] + 12)
    preds = mosaic_to_preds([(local, 0.9)], placements, 1, conf=0.25, source="yolo26")
    assert len(preds) == 1
    assert preds[0].box.x1 >= 10


def test_exclusive_mix_drops_11s_inside_26_focus() -> None:
    card = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 100, 100], "model": "yolo26", "conf": 0.25}]}
    )
    inside = Prediction(1, Box(10, 10, 20, 40), 0.9, "yolo11s")
    outside = Prediction(1, Box(200, 200, 230, 260), 0.8, "yolo11s")
    crop26 = Prediction(1, Box(12, 12, 22, 42), 0.7, "yolo26")
    mixed = exclusive_mix((inside, outside), (crop26,), card.regions)
    assert [row.box.x1 for row in mixed.yolo11s] == [200]
    assert mixed.yolo26[0].box.x1 == 12


def test_pick_knob_prefers_h1_then_mae() -> None:
    picked = pick_knob(
        [
            {
                "model": "yolo26",
                "imgsz": 640,
                "conf": 0.25,
                "mae": 4.0,
                "precision": 0.8,
                "time_ms": 40.0,
                "b11_mae": 5.0,
                "b11_precision": 0.8,
                "b26_time_ms": 30.0,
                "proxy": 1.0,
            },
            {
                "model": "yolo11s",
                "imgsz": 320,
                "conf": 0.25,
                "mae": 5.1,
                "precision": 0.79,
                "time_ms": 8.0,
                "b11_mae": 5.0,
                "b11_precision": 0.8,
                "b26_time_ms": 30.0,
                "proxy": 0.25,
            },
        ]
    )
    assert picked["model"] == "yolo11s"
    assert picked["h1"]["passed"] is True


def test_next_conf_raises_on_busy_and_drops_on_empty() -> None:
    assert next_conf(0.25, 10, 4) == 0.35
    assert next_conf(0.25, 0, 3) == 0.15
    assert next_conf(0.25, 4, 4) == 0.25
