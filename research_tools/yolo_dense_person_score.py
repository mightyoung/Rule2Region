"""Extract pre-threshold, pre-NMS YOLO person score maps from a frozen model."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def split_dense_scores(scores: np.ndarray, shapes: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, ...]:
    """Split flattened BxN scores using the detector feature-map geometry."""
    if scores.ndim != 2:
        raise ValueError("dense_scores_expected_bn")
    expected = sum(height * width for height, width in shapes)
    if scores.shape[1] != expected:
        raise ValueError("dense_score_geometry_mismatch")
    result = []
    start = 0
    for height, width in shapes:
        end = start + height * width
        result.append(scores[:, start:end].reshape(scores.shape[0], height, width))
        start = end
    return tuple(result)


def _actual_output(captured: list[Any], expected_batch: int) -> dict[str, Any]:
    for output in reversed(captured):
        if not isinstance(output, (tuple, list)) or len(output) < 2 or not isinstance(output[1], dict):
            continue
        raw = output[1]
        scores = raw.get("scores")
        feats = raw.get("feats")
        if hasattr(scores, "shape") and scores.shape[0] == expected_batch and isinstance(feats, list):
            return raw
    raise ValueError("dense_head_actual_output_not_found")


class FrozenYoloDensePersonScore:
    """Return aligned person-probability maps before confidence filtering/NMS."""

    def __init__(self, model_path: str, *, device: str = "0", imgsz: int = 640) -> None:
        from ultralytics import YOLO  # type: ignore

        self.yolo = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self._captured: list[Any] = []
        self._hook = self.yolo.model.model[-1].register_forward_hook(
            lambda _module, _inputs, output: self._captured.append(output)
        )

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "FrozenYoloDensePersonScore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def extract(
        self,
        images: Iterable[np.ndarray],
        *,
        output_width: int,
        output_height: int,
        batch_size: int = 8,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        multiscale = self.extract_multiscale(
            images,
            output_width=output_width,
            output_height=output_height,
            batch_size=batch_size,
        )
        return multiscale.max(axis=1)

    def extract_multiscale(
        self,
        images: Iterable[np.ndarray],
        *,
        output_width: int,
        output_height: int,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return one aligned pre-NMS person-probability channel per head scale."""
        import torch
        import torch.nn.functional as functional

        image_list = list(images)
        outputs: list[np.ndarray] = []
        for start in range(0, len(image_list), batch_size):
            chunk = image_list[start : start + batch_size]
            self._captured.clear()
            with torch.inference_mode():
                self.yolo.predict(
                    source=chunk,
                    device=self.device,
                    imgsz=self.imgsz,
                    batch=len(chunk),
                    verbose=False,
                )
            raw = _actual_output(self._captured, len(chunk))
            logits = raw["scores"][:, 0].detach()
            shapes = tuple((int(feature.shape[-2]), int(feature.shape[-1])) for feature in raw["feats"])
            boundaries = []
            cursor = 0
            for height, width in shapes:
                end = cursor + height * width
                boundaries.append(logits[:, cursor:end].reshape(len(chunk), height, width))
                cursor = end
            if cursor != logits.shape[1]:
                raise ValueError("dense_head_score_geometry_mismatch")
            for image_index, image in enumerate(chunk):
                height, width = image.shape[:2]
                gain = min(self.imgsz / width, self.imgsz / height)
                resized_width, resized_height = round(width * gain), round(height * gain)
                pad_x = (self.imgsz - resized_width) / 2.0
                pad_y = (self.imgsz - resized_height) / 2.0
                aligned = []
                for scale, (feature_height, feature_width) in zip(boundaries, shapes):
                    probability = scale[image_index].sigmoid()
                    x0 = max(0, min(feature_width - 1, int(np.floor(pad_x / self.imgsz * feature_width))))
                    x1 = max(x0 + 1, min(feature_width, int(np.ceil((pad_x + resized_width) / self.imgsz * feature_width))))
                    y0 = max(0, min(feature_height - 1, int(np.floor(pad_y / self.imgsz * feature_height))))
                    y1 = max(y0 + 1, min(feature_height, int(np.ceil((pad_y + resized_height) / self.imgsz * feature_height))))
                    cropped = probability[y0:y1, x0:x1][None, None]
                    aligned.append(
                        functional.interpolate(
                            cropped,
                            size=(output_height, output_width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                    )
                outputs.append(torch.stack(aligned).cpu().numpy().astype(np.float32))
        return np.stack(outputs)
