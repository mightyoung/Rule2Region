from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import av
import numpy as np
from PIL import Image

from codecref_lite.schemas import CandidateFeature

from .plugins import (
    PluginResult,
    image_center_crop50,
    image_jpeg,
    image_native,
    image_random_crop50,
    image_resize50,
    video_bitrate_only,
    video_resize50,
    video_select,
)


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def decode_video_rgb(path: str | Path) -> tuple[np.ndarray, ...]:
    with av.open(str(path)) as container:
        frames = tuple(np.asarray(frame.to_image().convert("RGB")) for frame in container.decode(video=0))
    if not frames:
        raise ValueError("decoded_video_has_no_frames")
    return frames


def load_candidate_features(path: str | Path) -> tuple[CandidateFeature, ...]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(CandidateFeature(**row) for row in rows)


def prepare_image_input(sample: dict[str, Any], condition: str, *, seed: int = 20260810) -> PluginResult:
    frame = load_rgb_image(sample["source"]["path"])
    source_hash = sample["source"]["sha256"]
    if condition == "native":
        return image_native(frame, source_sha256=source_hash)
    if condition == "jpeg":
        return image_jpeg(frame, source_sha256=source_hash, quality=75)
    if condition == "resize50":
        return image_resize50(frame, source_sha256=source_hash)
    if condition == "random_crop50":
        return image_random_crop50(frame, source_sha256=source_hash, seed=seed)
    if condition == "center_crop50":
        return image_center_crop50(frame, source_sha256=source_hash)
    raise ValueError(f"unsupported_image_smoke_condition:{condition}")


def prepare_video_input(
    source_frames: Sequence[np.ndarray],
    condition: str,
    *,
    source_sha256: str,
    codec_features: Sequence[CandidateFeature] | None = None,
    bitrate_frames: Sequence[np.ndarray] | None = None,
    original_encoded_bytes: int | None = None,
    reduced_encoded_bytes: int | None = None,
    seed: int = 20260810,
    window_id: str = "development-video",
) -> PluginResult:
    count = len(source_frames)
    budget = max(1, count // 2)
    if condition == "dense":
        return video_select(source_frames, condition="dense", source_sha256=source_sha256)
    if condition in {"uniform50", "random50", "rgb_difference50", "codec_guided50"}:
        return video_select(
            source_frames,
            condition=condition,
            budget=budget,
            global_seed=seed,
            window_id=window_id,
            codec_features=codec_features,
            source_sha256=source_sha256,
        )
    if condition == "resize50":
        return video_resize50(source_frames, source_sha256=source_sha256)
    if condition == "bitrate_only":
        if original_encoded_bytes is None or reduced_encoded_bytes is None:
            raise ValueError("bitrate_byte_receipts_required")
        return video_bitrate_only(
            source_frames,
            reduced_frames=bitrate_frames,
            original_encoded_bytes=original_encoded_bytes,
            reduced_encoded_bytes=reduced_encoded_bytes,
            source_sha256=source_sha256,
        )
    raise ValueError(f"unsupported_video_smoke_condition:{condition}")
