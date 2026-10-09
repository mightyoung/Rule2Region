import numpy as np

from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.card import parse_policy_card
from research_tools.execution_policy.crop_infer import _clip_mapped, crop_region, infer_batched_yolo26_frame
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import view_id
from research_tools.residual_query import keep_owned, map_crop_box


def test_view_id_merges_parking_lot() -> None:
    assert view_id("VIRAT_S_010112") == "VIRAT_S_010111"
    assert view_id("VIRAT_S_010004") == "VIRAT_S_010004"


def test_crop_region_rejects_tiny_and_maps_back() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert crop_region(image, Box(10, 10, 11, 11)) is None
    crop, roi = crop_region(image, Box(100, 200, 180, 320))
    assert crop.shape[0] == 120
    assert crop.shape[1] == 80
    mapped = map_crop_box(Box(5, 6, 20, 30), roi)
    assert mapped.coords == (105, 206, 120, 230)
    clipped = _clip_mapped(Box(-4, -2, 10, 8))
    assert clipped is not None
    assert clipped.x1 == 0
    pred = Prediction(1, mapped, 0.4, "yolo26")
    assert keep_owned(pred, Box(100, 200, 180, 320)) is True


def test_load_view_cards_parses_e4_payload() -> None:
    cards = load_view_cards(
        {
            "per_camera": {
                "VIRAT_S_010111": {
                    "intention_card": {
                        "regions": [
                            {
                                "region_id": "cam26_1",
                                "xyxy": [916, 295, 1003, 407],
                                "model": "yolo26",
                                "conf": 0.25,
                                "role": "count",
                            }
                        ]
                    }
                },
                "VIRAT_S_010004": {"intention_card": None},
            }
        }
    )
    card = cards["VIRAT_S_010111"]
    assert len(card.regions) == 1
    assert card.regions[0].model == "yolo26"
    assert "VIRAT_S_010004" not in cards
    parsed = parse_policy_card(
        {
            "regions": [
                {"region_id": "a", "xyxy": [0, 0, 10, 10], "model": "yolo26", "conf": 0.25},
                {"region_id": "b", "xyxy": [20, 0, 30, 10], "model": "yolo26", "conf": 0.25},
            ]
        }
    )
    assert parsed.regions[0].box.x2 == 10


class _FakeTensor:
    def __init__(self, values) -> None:
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)


class _FakeResult:
    def __init__(self, xyxy, conf, cls) -> None:
        self.boxes = _FakeBoxes(xyxy, conf, cls)


class _FakeBatchModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [
            _FakeResult(
                [[5, 6, 20, 30], [30, 10, 45, 35], [55, 10, 65, 30]],
                [0.4, 0.3, 0.9],
                [0, 0, 0],
            ),
            _FakeResult([[4, 8, 14, 28], [20, 5, 30, 25]], [0.2, 0.1], [0, 0]),
        ]


def test_batched_yolo26_inference_calls_predict_once_and_filters_per_roi() -> None:
    card = parse_policy_card(
        {
            "regions": [
                {"region_id": "a", "xyxy": [100, 200, 150, 250], "model": "yolo26", "conf": 0.35},
                {"region_id": "b", "xyxy": [300, 400, 350, 450], "model": "yolo26", "conf": 0.15},
            ]
        }
    )
    model = _FakeBatchModel()

    counted, elapsed_ms = infer_batched_yolo26_frame(
        model,
        card,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        7,
        device="1",
        imgsz=320,
    )

    assert len(model.calls) == 1
    assert len(model.calls[0]["source"]) == 2
    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["conf"] == 0.15
    assert counted.yolo11s == ()
    assert [(row.box.coords, row.confidence) for row in counted.yolo26] == [
        ((105.0, 206.0, 120.0, 230.0), 0.4),
        ((304.0, 408.0, 314.0, 428.0), 0.2),
    ]
    assert elapsed_ms >= 0.0


def test_batched_yolo26_inference_can_override_all_roi_confidence() -> None:
    card = parse_policy_card(
        {
            "regions": [
                {"region_id": "a", "xyxy": [100, 200, 150, 250], "model": "yolo26", "conf": 0.35},
                {"region_id": "b", "xyxy": [300, 400, 350, 450], "model": "yolo26", "conf": 0.15},
            ]
        }
    )
    model = _FakeBatchModel()

    counted, _ = infer_batched_yolo26_frame(
        model,
        card,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        7,
        device="1",
        conf=0.25,
    )

    assert model.calls[0]["conf"] == 0.25
    assert [(row.box.coords, row.confidence) for row in counted.yolo26] == [
        ((105.0, 206.0, 120.0, 230.0), 0.4),
        ((130.0, 210.0, 145.0, 235.0), 0.3),
    ]
