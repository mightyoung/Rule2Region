from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from evidencegate.manifest import manifest_sha256


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGAL_STATES = frozenset({"sealed", "unsealed", "not_required"})
_EVIDENCE_STATES = frozenset({"verified", "unverified", "absent", "incomplete"})


def _nonempty(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field}_required")
    return normalized


def _sha256(value: str, field: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field}_invalid")
    return normalized


@dataclass(frozen=True)
class DatasetAdmission:
    dataset_id: str
    official_url: str
    agreement_execution: str
    processing_permission: str
    video_inventory: str
    annotation_mapping: str
    source_hashes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _nonempty(self.dataset_id, "dataset_id"))
        if not self.official_url.startswith("https://"):
            raise ValueError("official_url_must_use_https")
        for field in ("agreement_execution", "processing_permission"):
            if getattr(self, field) not in _LEGAL_STATES:
                raise ValueError(f"{field}_invalid")
        for field in ("video_inventory", "annotation_mapping", "source_hashes"):
            if getattr(self, field) not in _EVIDENCE_STATES:
                raise ValueError(f"{field}_invalid")

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        for field in ("agreement_execution", "processing_permission"):
            value = getattr(self, field)
            if value == "unsealed":
                blockers.append(f"{field}:{value}")
        for field in ("video_inventory", "annotation_mapping", "source_hashes"):
            value = getattr(self, field)
            if value != "verified":
                blockers.append(f"{field}:{value}")
        return tuple(blockers)

    @property
    def status(self) -> str:
        return "READY" if not self.blockers else "BLOCKED"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status
        result["blockers"] = list(self.blockers)
        return result


@dataclass(frozen=True)
class PersonBox:
    track_id: str
    x: float
    y: float
    width: float
    height: float
    occlusion: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonempty(self.track_id, "track_id"))
        if not all(math.isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("person_box_non_finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("person_box_non_positive_extent")

    def clipped_xyxy(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        x1 = min(float(image_width), max(0.0, self.x))
        y1 = min(float(image_height), max(0.0, self.y))
        x2 = min(float(image_width), max(0.0, self.x + self.width))
        y2 = min(float(image_height), max(0.0, self.y + self.height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person_box_outside_frame")
        return x1, y1, x2, y2


@dataclass(frozen=True)
class IndustrialFrameRecord:
    dataset_id: str
    camera_id: str
    source_video_id: str
    frame_index: int
    timestamp_seconds: float
    image_width: int
    image_height: int
    persons: tuple[PersonBox, ...]

    def __post_init__(self) -> None:
        for field in ("dataset_id", "camera_id", "source_video_id"):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        if self.frame_index < 0:
            raise ValueError("frame_index_negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_invalid")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_extent_invalid")
        for person in self.persons:
            person.clipped_xyxy(self.image_width, self.image_height)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "camera_id": self.camera_id,
            "source_video_id": self.source_video_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "person_count": self.person_count,
            "persons": [
                {
                    "track_id": person.track_id,
                    "bbox_xyxy": list(person.clipped_xyxy(self.image_width, self.image_height)),
                    "occlusion": person.occlusion,
                }
                for person in self.persons
            ],
        }


@dataclass(frozen=True)
class PilotVideo:
    dataset_id: str
    camera_id: str
    source_video_id: str
    total_frames: int
    fps: float
    video_sha256: str
    annotation_sha256: str

    def __post_init__(self) -> None:
        for field in ("dataset_id", "camera_id", "source_video_id"):
            object.__setattr__(self, field, _nonempty(getattr(self, field), field))
        if self.total_frames <= 0:
            raise ValueError("total_frames_must_be_positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps_must_be_positive")
        object.__setattr__(self, "video_sha256", _sha256(self.video_sha256, "video_sha256"))
        object.__setattr__(
            self,
            "annotation_sha256",
            _sha256(self.annotation_sha256, "annotation_sha256"),
        )


def select_evenly_spaced_frames(
    *,
    total_frames: int,
    count: int,
    minimum_gap_frames: int,
) -> tuple[int, ...]:
    if total_frames <= 0 or count <= 0:
        raise ValueError("frame_extent_and_count_must_be_positive")
    if minimum_gap_frames < 0:
        raise ValueError("minimum_gap_frames_negative")
    if count > total_frames or (count - 1) * minimum_gap_frames > total_frames - 1:
        raise ValueError("insufficient_frames_for_gap")

    selected = tuple(
        ((2 * index + 1) * total_frames - 1) // (2 * count)
        for index in range(count)
    )
    if any(right - left < minimum_gap_frames for left, right in zip(selected, selected[1:])):
        raise ValueError("insufficient_frames_for_gap")
    return selected


def build_pilot_manifest(
    videos: Iterable[PilotVideo],
    *,
    frames_per_video: int,
    minimum_gap_seconds: float,
) -> dict[str, Any]:
    if frames_per_video <= 0:
        raise ValueError("frames_per_video_must_be_positive")
    if not math.isfinite(minimum_gap_seconds) or minimum_gap_seconds < 0:
        raise ValueError("minimum_gap_seconds_invalid")

    ordered = tuple(sorted(videos, key=lambda item: item.source_video_id))
    if not ordered:
        raise ValueError("pilot_requires_videos")
    video_ids = [video.source_video_id for video in ordered]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("duplicate_source_video_id")

    samples: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for video in ordered:
        minimum_gap_frames = math.ceil(minimum_gap_seconds * video.fps)
        frame_indices = select_evenly_spaced_frames(
            total_frames=video.total_frames,
            count=frames_per_video,
            minimum_gap_frames=minimum_gap_frames,
        )
        sources.append(asdict(video))
        for frame_index in frame_indices:
            samples.append(
                {
                    "sample_id": f"{video.source_video_id}:frame:{frame_index}",
                    "dataset_id": video.dataset_id,
                    "camera_id": video.camera_id,
                    "source_video_id": video.source_video_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / video.fps, 9),
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": "fieldpack.g0-pilot.v1",
        "selection_policy": "deterministic_equal_temporal_bins_midpoint",
        "frames_per_video": frames_per_video,
        "minimum_gap_seconds": minimum_gap_seconds,
        "independent_camera_count": len({video.camera_id for video in ordered}),
        "source_video_count": len(ordered),
        "sample_count": len(samples),
        "sources": sources,
        "samples": samples,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest
