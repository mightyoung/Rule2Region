from pathlib import Path

import numpy as np
from PIL import Image

import research_tools.generate_yolo11s_receipts as module
from research_tools.cascade_experiment import Box, Prediction


class _Detector:
    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, _frame, *, frame_index):
        return (Prediction(frame_index, Box(1, 2, 3, 4), .5, "yolo11s"),)

    def predict_batch(self, frames, *, frame_indices):
        assert len(frames) == len(frame_indices)
        return tuple(
            (Prediction(frame, Box(1, 2, 3, 4), .5, "yolo11s"),)
            for frame in frame_indices
        )


def test_receipt_generator_calls_only_small_detector(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "img1"
    frames.mkdir()
    for frame in (1, 2):
        Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(frames / f"{frame:06d}.jpg")
    monkeypatch.setattr(module, "UltralyticsYOLOAdapter", _Detector)
    payload = module.generate_receipt(frames, tmp_path / "yolo11s.pt", sequence_id="VIRAT_S_010111_02_x")
    assert payload["strong_model_called"] is False
    assert payload["frames"] == 2
    assert len(payload["receipts"]["yolo11s_global"]) == 2
