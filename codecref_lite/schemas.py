from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MotionVector:
    motion_x: int
    motion_y: int
    motion_scale: int
    width: int
    height: int
    dst_x: int = 0
    dst_y: int = 0


@dataclass(frozen=True)
class CandidateFeature:
    index: int
    pts_seconds: float
    packet_bpp: float
    mv_density: float
    mv_mean: float
    mv_energy: float
    frame_type: str
    key_frame: bool = False
    mv_missing: bool = False
    packet_attribution_supported: bool = True
    pts_supported: bool = True
    mv_side_data_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackObservation:
    source_video_id: str
    track_id: str
    category: str
    pts_seconds: float
    x: float
    y: float
    width: float
    height: float
    ignored: bool = False
    absent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
