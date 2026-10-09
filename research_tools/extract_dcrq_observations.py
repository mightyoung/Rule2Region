"""Cache label-blind frozen-YOLO observations for DCRQ experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.yolo_dense_person_score import _actual_output


def robust_quantile_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize each map without camera-level fitted statistics."""
    if values.ndim < 2:
        raise ValueError("robust_normalization_requires_spatial_maps")
    flat = values.reshape(*values.shape[:-2], -1)
    q10 = np.quantile(flat, .10, axis=-1, keepdims=True)
    q50 = np.quantile(flat, .50, axis=-1, keepdims=True)
    q90 = np.quantile(flat, .90, axis=-1, keepdims=True)
    scale = np.maximum(q90 - q10, 1e-6)
    normalized = np.clip((flat - q50) / scale, -3.0, 3.0)
    return normalized.reshape(values.shape).astype(np.float32)


def normalized_categorical_entropy(probabilities: np.ndarray, *, axis: int) -> np.ndarray:
    if probabilities.shape[axis] < 2:
        raise ValueError("entropy_requires_at_least_two_bins")
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=axis)
    return (entropy / math.log(probabilities.shape[axis])).astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ultralytics_bgr(path: Path) -> np.ndarray:
    """Decode with PIL, then satisfy Ultralytics' BGR contract for NumPy inputs."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def _align(
    tensor: Any,
    *,
    source_width: int,
    source_height: int,
    imgsz: int,
    output_width: int,
    output_height: int,
) -> Any:
    import torch.nn.functional as functional

    feature_height, feature_width = tensor.shape[-2:]
    gain = min(imgsz / source_width, imgsz / source_height)
    resized_width, resized_height = round(source_width * gain), round(source_height * gain)
    pad_x = (imgsz - resized_width) / 2.0
    pad_y = (imgsz - resized_height) / 2.0
    x0 = max(0, min(feature_width - 1, int(np.floor(pad_x / imgsz * feature_width))))
    x1 = max(x0 + 1, min(feature_width, int(np.ceil((pad_x + resized_width) / imgsz * feature_width))))
    y0 = max(0, min(feature_height - 1, int(np.floor(pad_y / imgsz * feature_height))))
    y1 = max(y0 + 1, min(feature_height, int(np.ceil((pad_y + resized_height) / imgsz * feature_height))))
    cropped = tensor[..., y0:y1, x0:x1][None, None]
    return functional.interpolate(cropped, size=(output_height, output_width), mode="bilinear", align_corners=False)[0, 0]


def _occupancy(predictions: tuple[Prediction, ...], width: int, height: int, output_width: int, output_height: int) -> np.ndarray:
    result = np.zeros((output_height, output_width), dtype=np.float32)
    for prediction in predictions:
        box = prediction.box
        x0 = max(0, min(output_width - 1, int(np.floor(box.x1 / width * output_width))))
        x1 = max(x0 + 1, min(output_width, int(np.ceil(box.x2 / width * output_width))))
        y0 = max(0, min(output_height - 1, int(np.floor(box.y1 / height * output_height))))
        y1 = max(y0 + 1, min(output_height, int(np.ceil(box.y2 / height * output_height))))
        result[y0:y1, x0:x1] = 1.0
    return result


class DcrqObservationExtractor:
    def __init__(self, model_path: str, *, device: str = "0", imgsz: int = 640) -> None:
        from ultralytics import YOLO  # type: ignore

        self.yolo = YOLO(model_path)
        self.device = device
        self.imgsz = imgsz
        self.captured: list[Any] = []
        self.hook = self.yolo.model.model[-1].register_forward_hook(
            lambda _module, _inputs, output: self.captured.append(output)
        )

    def close(self) -> None:
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def __enter__(self) -> "DcrqObservationExtractor":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def extract_sequence(
        self,
        frames_dir: Path,
        *,
        output_width: int = 40,
        output_height: int = 23,
        batch_size: int = 8,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        import torch

        frame_ids = tuple(sorted(int(path.stem) for path in frames_dir.glob("*.jpg") if path.stem.isdigit()))
        if not frame_ids:
            raise ValueError("dcrq_observation_frames_missing")
        person_rows: list[np.ndarray] = []
        normalized_rows: list[np.ndarray] = []
        entropy_rows: list[np.ndarray] = []
        occupancy_rows: list[np.ndarray] = []
        receipts: list[dict[str, Any]] = []
        for start in range(0, len(frame_ids), batch_size):
            chunk_ids = frame_ids[start:start + batch_size]
            images = [load_ultralytics_bgr(frames_dir / f"{frame:06d}.jpg") for frame in chunk_ids]
            self.captured.clear()
            with torch.inference_mode():
                results = self.yolo.predict(source=images, device=self.device, imgsz=self.imgsz, batch=len(images), verbose=False)
            raw = _actual_output(self.captured, len(images))
            score_logits = raw["scores"][:, 0].detach()
            box_logits = raw["boxes"].detach()
            shapes = tuple((int(feature.shape[-2]), int(feature.shape[-1])) for feature in raw["feats"])
            cursor = 0
            person_scales = []
            entropy_scales = []
            for feature_height, feature_width in shapes:
                end = cursor + feature_height * feature_width
                person_scales.append(score_logits[:, cursor:end].reshape(len(images), feature_height, feature_width).sigmoid())
                box_scale = box_logits[:, :, cursor:end].reshape(len(images), 4, 16, feature_height, feature_width)
                probability = box_scale.softmax(dim=2).clamp_min(1e-12)
                entropy_scales.append((-(probability * probability.log()).sum(dim=2).mean(dim=1) / math.log(16.0)))
                cursor = end
            if cursor != score_logits.shape[1]:
                raise ValueError("dcrq_raw_head_geometry_mismatch")
            for index, (frame_id, image, result) in enumerate(zip(chunk_ids, images, results)):
                height, width = image.shape[:2]
                aligned_person = np.stack([
                    _align(scale[index], source_width=width, source_height=height, imgsz=self.imgsz, output_width=output_width, output_height=output_height).cpu().numpy().astype(np.float32)
                    for scale in person_scales
                ])
                aligned_entropy = np.stack([
                    _align(scale[index], source_width=width, source_height=height, imgsz=self.imgsz, output_width=output_width, output_height=output_height).cpu().numpy().astype(np.float32)
                    for scale in entropy_scales
                ])
                boxes = getattr(result, "boxes", None)
                predictions: list[Prediction] = []
                if boxes is not None:
                    for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()):
                        if int(cls) == 0:
                            predictions.append(Prediction(frame_id, Box(*map(float, coords)), float(confidence), "yolo11s"))
                prediction_tuple = tuple(predictions)
                person_rows.append(aligned_person)
                normalized_rows.append(robust_quantile_normalize(aligned_person))
                entropy_rows.append(aligned_entropy)
                occupancy_rows.append(_occupancy(prediction_tuple, width, height, output_width, output_height)[None])
                receipts.append({"frame": frame_id, "predictions": [asdict(prediction) for prediction in prediction_tuple]})
        return {
            "frame_ids": np.asarray(frame_ids, dtype=np.int32),
            "person_probability": np.stack(person_rows),
            "person_robust_normalized": np.stack(normalized_rows),
            "dfl_entropy": np.stack(entropy_rows),
            "detection_occupancy": np.stack(occupancy_rows),
        }, receipts


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache frozen YOLO observations for DCRQ")
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    args.output.mkdir(parents=True)
    sequence_rows = []
    with DcrqObservationExtractor(args.small_model, device=args.device) as extractor:
        for sequence_dir in sorted((args.pilot_root / "sequences").iterdir()):
            if not sequence_dir.is_dir():
                continue
            arrays, receipts = extractor.extract_sequence(sequence_dir / "img1", batch_size=args.batch_size)
            cache_path = args.output / f"{sequence_dir.name}.npz"
            receipt_path = args.output / f"{sequence_dir.name}.json"
            np.savez_compressed(cache_path, **arrays)
            receipt_path.write_text(json.dumps({
                "schema_version": "fieldpack.dcrq-yolo11s-receipt.v1",
                "sequence_id": sequence_dir.name,
                "frames": len(receipts),
                "numpy_input_color": "BGR_CONVERTED_FROM_PIL_RGB",
                "strong_model_called": False,
                "gt_used_online": False,
                "receipts": {"yolo11s_global": receipts},
            }, sort_keys=True) + "\n", encoding="utf-8")
            sequence_rows.append({
                "sequence_id": sequence_dir.name,
                "frames": len(receipts),
                "cache": cache_path.name,
                "cache_sha256": _sha256(cache_path),
                "receipt": receipt_path.name,
                "receipt_sha256": _sha256(receipt_path),
            })
    manifest = {
        "schema_version": "fieldpack.dcrq-observation-cache.v1",
        "online_channels": ["person_probability", "person_robust_normalized", "dfl_entropy", "detection_occupancy"],
        "grid": [23, 40],
        "numpy_input_color": "BGR_CONVERTED_FROM_PIL_RGB",
        "strong_model_called": False,
        "gt_used_online": False,
        "sequences": sequence_rows,
        "frames": sum(row["frames"] for row in sequence_rows),
    }
    (args.output / "cache-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sequences": len(sequence_rows), "frames": manifest["frames"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
