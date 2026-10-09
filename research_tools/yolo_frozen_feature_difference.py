"""Frozen YOLO multi-scale feature-difference extraction with lazy imports."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _feature_tensor(output: Any) -> Any:
    if hasattr(output, "ndim") and output.ndim == 4:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if hasattr(item, "ndim") and item.ndim == 4:
                return item
    raise ValueError("hook_output_has_no_4d_tensor")


def _actual_batch(captured: list[Any], expected: int) -> Any:
    """Discard square warm-up hooks and concatenate the final inference shape."""
    if not captured:
        raise ValueError("feature_hook_captured_nothing")
    spatial_shape = captured[-1].shape[-2:]
    compatible = [tensor for tensor in captured if tensor.shape[-2:] == spatial_shape]
    import torch
    batch = torch.cat(compatible, dim=0)
    if batch.shape[0] < expected:
        raise ValueError("feature_hook_batch_too_small")
    return batch[-expected:]


class FrozenYoloFeatureDifference:
    """Capture three frozen backbone scales and return pairwise L1 maps."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "0",
        imgsz: int = 640,
        layer_indices: tuple[int, ...] = (4, 6, 9),
    ) -> None:
        from ultralytics import YOLO  # type: ignore

        self.yolo = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.layer_indices = layer_indices
        layers = self.yolo.model.model
        if max(layer_indices) >= len(layers):
            raise ValueError("feature_layer_index_out_of_range")
        self._captured: dict[int, list[Any]] = {index: [] for index in layer_indices}
        self._hooks = [
            layers[index].register_forward_hook(self._make_hook(index)) for index in layer_indices
        ]

    def _make_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._captured[index].append(_feature_tensor(output).detach())
        return hook

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "FrozenYoloFeatureDifference":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def extract_pairs(
        self,
        pairs: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        output_width: int,
        output_height: int,
        batch_pairs: int = 4,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        pair_list = list(pairs)
        outputs: list[np.ndarray] = []
        for start in range(0, len(pair_list), batch_pairs):
            chunk = pair_list[start:start + batch_pairs]
            flat = [image for pair in chunk for image in pair]
            for values in self._captured.values():
                values.clear()
            with torch.inference_mode():
                self.yolo.predict(
                    source=flat, device=self.device, imgsz=self.imgsz,
                    batch=len(flat), verbose=False,
                )
            tensors = {
                index: _actual_batch(self._captured[index], len(flat))
                for index in self.layer_indices
            }
            if any(tensor.shape[0] != len(flat) for tensor in tensors.values()):
                raise ValueError("feature_hook_batch_mismatch")
            for pair_index, (current, _reference) in enumerate(chunk):
                height, width = current.shape[:2]
                gain = min(self.imgsz / width, self.imgsz / height)
                resized_width, resized_height = round(width * gain), round(height * gain)
                pad_x = (self.imgsz - resized_width) / 2.0
                pad_y = (self.imgsz - resized_height) / 2.0
                maps = []
                for index in self.layer_indices:
                    tensor = tensors[index]
                    current_feature = tensor[2 * pair_index]
                    reference_feature = tensor[2 * pair_index + 1]
                    difference = (current_feature - reference_feature).abs().mean(dim=0, keepdim=True)
                    feature_height, feature_width = difference.shape[-2:]
                    x0 = max(0, min(feature_width - 1, int(np.floor(pad_x / self.imgsz * feature_width))))
                    x1 = max(x0 + 1, min(feature_width, int(np.ceil((pad_x + resized_width) / self.imgsz * feature_width))))
                    y0 = max(0, min(feature_height - 1, int(np.floor(pad_y / self.imgsz * feature_height))))
                    y1 = max(y0 + 1, min(feature_height, int(np.ceil((pad_y + resized_height) / self.imgsz * feature_height))))
                    cropped = difference[:, y0:y1, x0:x1][None, ...]
                    resized = functional.interpolate(
                        cropped, size=(output_height, output_width), mode="bilinear", align_corners=False
                    )[0, 0]
                    scale = resized.amax().clamp_min(1e-6)
                    maps.append((resized / scale).cpu().numpy().astype(np.float32))
                outputs.append(np.stack(maps))
        return np.stack(outputs)
