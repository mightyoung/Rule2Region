from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from codecref_lite.schemas import CandidateFeature
from codecref_lite.selectors import (
    codec_guided_select,
    dense_select,
    random_select,
    rgb_difference_select,
    uniform_select,
)

from .ir import VisualEvidenceIR, hash_bytes

Encoder = Callable[..., bytes]
Region = tuple[int, int, int, int]
_HASH_LENGTH = 64


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != _HASH_LENGTH or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{name}_must_be_sha256")
    return value.lower()


def _array(value: np.ndarray, name: str = "frame") -> np.ndarray:
    result = np.asarray(value)
    if result.ndim not in (2, 3) or result.shape[0] <= 0 or result.shape[1] <= 0:
        raise ValueError(f"{name}_must_be_nonempty_image")
    if result.dtype == object:
        raise ValueError(f"{name}_dtype_invalid")
    return result


def _source_hash(frames: Sequence[np.ndarray], supplied: str | None) -> str:
    if supplied is not None:
        return _sha(supplied, "source_sha256")
    digest = hashlib.sha256()
    for frame in frames:
        arr = _array(frame)
        digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _encode(arr: np.ndarray, encoder: Encoder | None, *, quality: int = 75) -> bytes:
    if encoder is None:
        raise ValueError("jpeg_encoder_required")
    try:
        payload = encoder(arr, quality=quality)
    except TypeError:
        payload = encoder(arr, quality)
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError("encoder_must_return_nonempty_bytes")
    return bytes(payload)


def _region(region: Region | None, width: int, height: int) -> Region | None:
    if region is None:
        return None
    if len(region) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in region):
        raise ValueError("region_must_be_integer_xywh")
    x, y, w, h = region
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("region_out_of_bounds")
    return (x, y, w, h)


def _crop(frame: np.ndarray, region: Region) -> np.ndarray:
    x, y, w, h = region
    return frame[y : y + h, x : x + w]


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    y = np.minimum((np.arange(height) * frame.shape[0] / height).astype(int), frame.shape[0] - 1)
    x = np.minimum((np.arange(width) * frame.shape[1] / width).astype(int), frame.shape[1] - 1)
    return frame[np.ix_(y, x)]


def _half_area_shape(height: int, width: int) -> tuple[int, int]:
    target = max(1, int(round(height * width * 0.5)))
    new_height = max(1, int(round(height * math.sqrt(0.5))))
    new_width = max(1, min(width, int(round(target / new_height))))
    # Correct rounding while remaining in image bounds.
    while new_height * new_width < target and new_width < width:
        new_width += 1
    return min(height, new_height), new_width


@dataclass(frozen=True)
class PluginResult:
    modality: str
    condition: str
    source_sha256: str
    derived_sha256: str
    selected_frame_indices: tuple[int, ...]
    region: Region | None
    encoded_bytes: int
    decoded_pixels: int
    model_pixels: int
    metadata: dict[str, Any]
    evidence: VisualEvidenceIR
    model_frames: tuple[np.ndarray, ...] = field(repr=False, compare=False)

    @property
    def visual_ir(self) -> VisualEvidenceIR:
        return self.evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "condition": self.condition,
            "source_sha256": self.source_sha256,
            "derived_sha256": self.derived_sha256,
            "selected_frame_indices": list(self.selected_frame_indices),
            "region": list(self.region) if self.region else None,
            "encoded_bytes": self.encoded_bytes,
            "decoded_pixels": self.decoded_pixels,
            "model_pixels": self.model_pixels,
            "metadata": dict(self.metadata),
            "evidence": self.evidence.to_dict(),
        }


def _result(modality: str, condition: str, source: str, frames: Sequence[np.ndarray], selected: tuple[int, ...], region: Region | None, payload: bytes, model_pixels: int, metadata: dict[str, Any], *, encoded_bytes_override: int | None = None, decoded_pixels_override: int | None = None) -> PluginResult:
    checked_frames = tuple(np.ascontiguousarray(_array(frame)) for frame in frames)
    derived = hash_bytes(payload)
    decoded = sum(int(frame.shape[0] * frame.shape[1]) for frame in checked_frames) if decoded_pixels_override is None else decoded_pixels_override
    if decoded < 0:
        raise ValueError("decoded_pixels_must_be_nonnegative")
    width, height = checked_frames[0].shape[1], checked_frames[0].shape[0]
    encoded_bytes = len(payload) if encoded_bytes_override is None else encoded_bytes_override
    ir = VisualEvidenceIR(modality, source, source, derived, condition, width, height, selected[0], encoded_bytes, decoded, model_pixels, stage="plugin")
    return PluginResult(modality, condition, source, derived, selected, region, encoded_bytes, decoded, model_pixels, metadata, ir, checked_frames)


def image_native(image: np.ndarray, *, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    frame = _array(image)
    source = _source_hash((frame,), source_sha256)
    payload = _encode(frame, encoder) if encoder else np.ascontiguousarray(frame).tobytes()
    return _result("image", "native", source, (frame,), (0,), None, payload, frame.shape[0] * frame.shape[1], {"transform": "native"})


def image_jpeg(image: np.ndarray, *, source_sha256: str | None = None, encoder: Encoder | None = None, quality: int = 75) -> PluginResult:
    frame = _array(image)
    if not 1 <= quality <= 100:
        raise ValueError("jpeg_quality_out_of_range")
    source = _source_hash((frame,), source_sha256)
    if encoder is None:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=quality, optimize=False, progressive=False, subsampling=2)
        payload = buffer.getvalue()
    else:
        payload = _encode(frame, encoder, quality=quality)
    try:
        with Image.open(io.BytesIO(payload)) as decoded_image:
            decoded_frame = np.asarray(decoded_image.convert("RGB"))
    except Exception as exc:
        raise ValueError("jpeg_payload_cannot_be_decoded") from exc
    return _result("image", "jpeg", source, (decoded_frame,), (0,), None, payload, frame.shape[0] * frame.shape[1], {"transform": "jpeg", "quality": quality, "model_input": "decoded_jpeg"}, decoded_pixels_override=frame.shape[0] * frame.shape[1])


def image_resize50(image: np.ndarray, *, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    frame = _array(image)
    source = _source_hash((frame,), source_sha256)
    height, width = _half_area_shape(frame.shape[0], frame.shape[1])
    resized = _resize_nearest(frame, height, width)
    payload = _encode(resized, encoder) if encoder else np.ascontiguousarray(resized).tobytes()
    return _result("image", "resize50", source, (resized,), (0,), None, payload, height * width, {"transform": "resize", "scale_area": 0.5, "source_shape": list(frame.shape), "derived_shape": list(resized.shape)}, decoded_pixels_override=frame.shape[0] * frame.shape[1])


def image_random_crop50(image: np.ndarray, *, source_sha256: str | None = None, seed: int = 0, encoder: Encoder | None = None) -> PluginResult:
    frame = _array(image)
    source = _source_hash((frame,), source_sha256)
    h, w = frame.shape[:2]
    crop_h = max(1, int(round(h * 0.5)))
    crop_w = w
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{source}:{seed}".encode()).digest()[:8], "big"))
    y = rng.randrange(0, h - crop_h + 1)
    region = (0, y, crop_w, crop_h)
    cropped = _crop(frame, region)
    payload = _encode(cropped, encoder) if encoder else np.ascontiguousarray(cropped).tobytes()
    return _result("image", "random_crop50", source, (cropped,), (0,), region, payload, crop_h * crop_w, {"transform": "random_crop", "seed": seed}, decoded_pixels_override=h * w)


def _query_crop50_region(region: Region, width: int, height: int) -> Region:
    """Build a near-half-area crop that contains the provider region."""
    x, y, region_width, region_height = region
    target = max(1, int(round(width * height * 0.5)))
    if region_width * region_height > target:
        raise ValueError("query_region_exceeds_crop50_budget")
    candidates: list[tuple[int, float, int, int]] = []
    source_aspect = width / height
    for crop_height in range(region_height, height + 1):
        crop_width = max(region_width, int(math.ceil(target / crop_height)))
        if crop_width <= width:
            candidates.append((abs(crop_width * crop_height - target), abs(crop_width / crop_height - source_aspect), crop_width, crop_height))
    if not candidates:
        raise ValueError("query_region_cannot_fit_crop50_budget")
    _, _, crop_width, crop_height = min(candidates)
    center_x = x + region_width / 2
    center_y = y + region_height / 2
    crop_x = min(max(0, int(round(center_x - crop_width / 2))), width - crop_width)
    crop_y = min(max(0, int(round(center_y - crop_height / 2))), height - crop_height)
    return crop_x, crop_y, crop_width, crop_height


def image_query_aware_crop50(image: np.ndarray, *, region: Region | None, provider: str | None, provenance_hash: str | None, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    frame = _array(image)
    if not provider or provider.strip().lower() in {"unknown", "gt", "ground_truth", "ground-truth", "annotation"}:
        raise ValueError("query_provider_not_allowed")
    if provenance_hash is None:
        raise ValueError("query_provenance_required")
    _sha(provenance_hash, "query_provenance_hash")
    checked = _region(region, frame.shape[1], frame.shape[0])
    if checked is None:
        raise ValueError("query_region_required")
    crop_region = _query_crop50_region(checked, frame.shape[1], frame.shape[0])
    cropped = _crop(frame, crop_region)
    source = _source_hash((frame,), source_sha256)
    payload = _encode(cropped, encoder) if encoder else np.ascontiguousarray(cropped).tobytes()
    return _result("image", "query_aware_crop50", source, (cropped,), (0,), crop_region, payload, crop_region[2] * crop_region[3], {"transform": "query_aware_crop", "provider": provider, "provenance_hash": provenance_hash.lower(), "provider_region": list(checked), "target_area_fraction": 0.5}, decoded_pixels_override=frame.shape[0] * frame.shape[1])


def image_center_crop50(image: np.ndarray, *, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    frame = _array(image)
    height, width = frame.shape[:2]
    provider_region = (min(width - 1, width // 2), min(height - 1, height // 2), 1, 1)
    crop_region = _query_crop50_region(provider_region, width, height)
    cropped = _crop(frame, crop_region)
    source = _source_hash((frame,), source_sha256)
    payload = _encode(cropped, encoder) if encoder else np.ascontiguousarray(cropped).tobytes()
    return _result(
        "image",
        "center_crop50",
        source,
        (cropped,),
        (0,),
        crop_region,
        payload,
        crop_region[2] * crop_region[3],
        {
            "transform": "global_center_crop",
            "provider": "predeclared_global_center_fallback_v1",
            "provider_region": list(provider_region),
            "input_dependence": "image_dimensions_only",
            "target_area_fraction": 0.5,
            "query_aware": False,
        },
        decoded_pixels_override=height * width,
    )


def video_select(frames: Sequence[np.ndarray], *, condition: str, budget: int | None = None, global_seed: int = 0, window_id: str = "window", codec_features: Sequence[CandidateFeature] | None = None, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    if not frames:
        raise ValueError("video_frames_required")
    checked = tuple(_array(frame, "video_frame") for frame in frames)
    shape = checked[0].shape
    if any(frame.shape != shape for frame in checked):
        raise ValueError("video_frame_dimensions_mismatch")
    count = len(checked)
    condition = condition.lower()
    if condition == "resize50":
        return video_resize50(checked, source_sha256=source_sha256, encoder=encoder)
    if condition == "dense":
        selected = dense_select(count)
    elif condition in {"uniform", "uniform50"}:
        selected = uniform_select(count, budget or count)
    elif condition in {"random", "random50"}:
        selected = random_select(count, budget or count, global_seed, window_id)
    elif condition in {"rgb_difference", "rgb-difference", "rgb_difference50", "rgb-difference50"}:
        selected = rgb_difference_select(checked, budget or count)
    elif condition in {"codec", "codec_guided", "codec-guided", "codec_guided50", "codec-guided50"}:
        if codec_features is None:
            raise ValueError("codec_features_required")
        selected = codec_guided_select(codec_features, budget or count)
    else:
        raise ValueError("unknown_video_condition")
    if any(index < 0 or index >= count for index in selected):
        raise ValueError("selected_frame_index_out_of_range")
    source = _source_hash(checked, source_sha256)
    chosen = tuple(checked[index] for index in selected)
    payload = b"".join(_encode(frame, encoder) if encoder else np.ascontiguousarray(frame).tobytes() for frame in chosen)
    pixels = sum(int(frame.shape[0] * frame.shape[1]) for frame in chosen)
    decoded_pixels = sum(int(frame.shape[0] * frame.shape[1]) for frame in checked)
    return _result("video", condition, source, chosen, selected, None, payload, pixels, {"selected_frame_indices": list(selected), "frame_count": count}, decoded_pixels_override=decoded_pixels)


def video_resize50(frames: Sequence[np.ndarray], *, source_sha256: str | None = None, encoder: Encoder | None = None) -> PluginResult:
    if not frames:
        raise ValueError("video_frames_required")
    checked = tuple(_array(frame, "video_frame") for frame in frames)
    if any(frame.shape != checked[0].shape for frame in checked):
        raise ValueError("video_frame_dimensions_mismatch")
    height, width = _half_area_shape(checked[0].shape[0], checked[0].shape[1])
    resized = tuple(_resize_nearest(frame, height, width) for frame in checked)
    source = _source_hash(checked, source_sha256)
    payload = b"".join(_encode(frame, encoder) if encoder else np.ascontiguousarray(frame).tobytes() for frame in resized)
    decoded_pixels = sum(int(frame.shape[0] * frame.shape[1]) for frame in checked)
    return _result("video", "resize50", source, resized, tuple(range(len(resized))), None, payload, len(resized) * height * width, {"transform": "resize", "source_shape": list(checked[0].shape), "derived_shape": list(resized[0].shape)}, decoded_pixels_override=decoded_pixels)


def video_bitrate_only(frames: Sequence[np.ndarray], *, reduced_frames: Sequence[np.ndarray] | None, original_encoded_bytes: int, reduced_encoded_bytes: int, source_sha256: str | None = None) -> PluginResult:
    if original_encoded_bytes <= 0 or reduced_encoded_bytes < 0 or reduced_encoded_bytes > original_encoded_bytes:
        raise ValueError("encoded_byte_budget_invalid")
    checked = tuple(_array(frame, "video_frame") for frame in frames)
    if reduced_frames is None:
        raise ValueError("reduced_frames_required_for_bitrate_only_model_input")
    checked_reduced = tuple(_array(frame, "reduced_video_frame") for frame in reduced_frames)
    if len(checked_reduced) != len(checked) or any(reduced.shape != source.shape for source, reduced in zip(checked, checked_reduced)):
        raise ValueError("reduced_frames_must_match_source_count_and_shape")
    source = _source_hash(checked, source_sha256)
    pixels = sum(int(frame.shape[0] * frame.shape[1]) for frame in checked_reduced)
    reduction = original_encoded_bytes - reduced_encoded_bytes
    metadata = {"frame_count": len(checked), "original_encoded_bytes": original_encoded_bytes, "encoded_byte_reduction": reduction, "model_pixels_unchanged": True, "token_reduction_claim": False, "model_input": "decoded_reduced_bitrate_video"}
    payload = b"".join(np.ascontiguousarray(frame).tobytes() for frame in checked_reduced)
    return _result("video", "bitrate_only", source, checked_reduced, tuple(range(len(checked_reduced))), None, payload, pixels, metadata, encoded_bytes_override=reduced_encoded_bytes, decoded_pixels_override=pixels)


# Readable aliases used by runners that spell the percentage in the function
# name rather than in the condition string.
image_resize_50 = image_resize50
image_random_crop_50 = image_random_crop50
image_query_aware_crop_50 = image_query_aware_crop50
image_center_crop_50 = image_center_crop50
video_bitrate = video_bitrate_only
