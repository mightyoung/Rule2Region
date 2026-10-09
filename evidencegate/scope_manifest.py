from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class VideoSample:
    sample_id: str
    sequence_id: str
    stratum: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("video_sample_id_required")
        if not self.sequence_id.strip():
            raise ValueError("video_sequence_id_required")
        if self.stratum not in {"STATIC_PRIMARY", "MOVING_CAMERA_STRESS"}:
            raise ValueError("invalid_video_stratum")


@dataclass(frozen=True)
class ScopeManifest:
    """Frozen pre-registration binding matrix units to real dataset IDs."""

    video_samples: tuple[VideoSample, ...]
    formal_image_ids: tuple[str, ...]
    video_deployment_ids: tuple[str, ...]
    image_deployment_ids: tuple[str, ...]
    development_image_ids: tuple[str, ...]
    development_video_fixture_path: str
    development_video_fixture_sha256: str
    video_source_manifest_sha256: str
    image_source_manifest_sha256: str

    def __post_init__(self) -> None:
        videos = tuple(
            v
            if isinstance(v, VideoSample)
            else VideoSample(
                str(v["sample_id"]),
                str(v["sequence_id"]),
                str(v["stratum"]),
            )
            for v in self.video_samples
        )
        images = tuple(self.formal_image_ids)
        video_deploy = tuple(self.video_deployment_ids)
        image_deploy = tuple(self.image_deployment_ids)
        development_images = tuple(self.development_image_ids)
        object.__setattr__(self, "video_samples", videos)
        object.__setattr__(self, "formal_image_ids", images)
        object.__setattr__(self, "video_deployment_ids", video_deploy)
        object.__setattr__(self, "image_deployment_ids", image_deploy)
        object.__setattr__(self, "development_image_ids", development_images)
        video_ids = tuple(v.sample_id for v in videos)
        if len(videos) != 50 or len(set(video_ids)) != 50:
            raise ValueError("scope_manifest_requires_exactly_50_unique_video_samples")
        if sum(v.stratum == "STATIC_PRIMARY" for v in videos) != 17 or sum(v.stratum == "MOVING_CAMERA_STRESS" for v in videos) != 33:
            raise ValueError("scope_manifest_requires_17_static_and_33_moving_video_samples")
        if len(images) != 48 or len(set(images)) != 48 or any(not x.strip() for x in images):
            raise ValueError("scope_manifest_requires_exactly_48_unique_image_ids")
        if len(video_deploy) != 24 or len(set(video_deploy)) != 24 or not set(video_deploy) <= set(video_ids):
            raise ValueError("video_deployment_ids_must_be_24_member_subset")
        if len(image_deploy) != 12 or len(set(image_deploy)) != 12 or not set(image_deploy) <= set(images):
            raise ValueError("image_deployment_ids_must_be_12_member_subset")
        if (
            len(development_images) != 12
            or len(set(development_images)) != 12
            or set(development_images) & set(images)
        ):
            raise ValueError("development_image_ids_must_be_12_unique_and_formal_disjoint")
        if not self.development_video_fixture_path.strip():
            raise ValueError("development_video_fixture_path_required")
        for name in (
            "development_video_fixture_sha256",
            "video_source_manifest_sha256",
            "image_source_manifest_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
                raise ValueError(f"{name}_must_be_sha256")

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(v.sample_id for v in self.video_samples)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self.formal_image_ids

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScopeManifest":
        videos = tuple(
            v
            if isinstance(v, VideoSample)
            else VideoSample(
                str(v["sample_id"]),
                str(v["sequence_id"]),
                str(v["stratum"]),
            )
            for v in value["video_samples"]
        )
        return cls(
            videos,
            tuple(value["formal_image_ids"]),
            tuple(value["video_deployment_ids"]),
            tuple(value["image_deployment_ids"]),
            tuple(value["development_image_ids"]),
            str(value["development_video_fixture_path"]),
            str(value["development_video_fixture_sha256"]),
            str(value["video_source_manifest_sha256"]),
            str(value["image_source_manifest_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_samples": [
                {
                    "sample_id": video.sample_id,
                    "sequence_id": video.sequence_id,
                    "stratum": video.stratum,
                }
                for video in self.video_samples
            ],
            "formal_image_ids": list(self.formal_image_ids),
            "video_deployment_ids": list(self.video_deployment_ids),
            "image_deployment_ids": list(self.image_deployment_ids),
            "development_image_ids": list(self.development_image_ids),
            "development_video_fixture_path": self.development_video_fixture_path,
            "development_video_fixture_sha256": self.development_video_fixture_sha256,
            "video_source_manifest_sha256": self.video_source_manifest_sha256,
            "image_source_manifest_sha256": self.image_source_manifest_sha256,
        }
