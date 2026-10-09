from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
from av.video.frame import PictureType

from .alignment import align_candidate_pts
from .features import build_candidate_feature
from .schemas import CandidateFeature, MotionVector


@dataclass(frozen=True)
class CodecProbe:
    path: str
    codec_name: str
    width: int
    height: int
    average_rate: float
    time_base: float
    frames: tuple[CandidateFeature, ...]


def _motion_vectors(side_data_items: Iterable[Any]) -> tuple[list[MotionVector], bool]:
    for side_data in side_data_items:
        if "MOTION_VECTOR" not in str(side_data.type).upper():
            continue
        array = side_data.to_ndarray()
        names = set(array.dtype.names or ())
        required = {
            "w",
            "h",
            "dst_x",
            "dst_y",
            "motion_x",
            "motion_y",
            "motion_scale",
        }
        if not required.issubset(names):
            return [], False
        vectors = [
            MotionVector(
                motion_x=int(row["motion_x"]),
                motion_y=int(row["motion_y"]),
                motion_scale=int(row["motion_scale"]),
                width=int(row["w"]),
                height=int(row["h"]),
                dst_x=int(row["dst_x"]),
                dst_y=int(row["dst_y"]),
            )
            for row in array
        ]
        return vectors, True
    return [], True


def probe_video(path: str | Path, *, export_motion_vectors: bool = True) -> CodecProbe:
    video_path = Path(path)
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("video stream not found")
        stream = container.streams.video[0]
        if export_motion_vectors:
            stream.codec_context.options = {"flags2": "+export_mvs"}
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        average_rate = float(stream.average_rate) if stream.average_rate else 0.0
        time_base = float(stream.time_base)
        codec_name = str(stream.codec_context.name)
        frames: list[CandidateFeature] = []
        packet_sizes: dict[int, int] = {}
        ambiguous_packet_pts: set[int] = set()
        decoded_frames: list[Any] = []

        for packet in container.demux(stream):
            if packet.pts is not None and packet.size > 0:
                if packet.pts in packet_sizes:
                    ambiguous_packet_pts.add(packet.pts)
                else:
                    packet_sizes[packet.pts] = packet.size
            decoded_frames.extend(packet.decode())

        for index, frame in enumerate(decoded_frames):
            pts_supported = frame.pts is not None and frame.time_base is not None
            pts_seconds = (
                float(frame.pts * frame.time_base) if pts_supported else float("nan")
            )
            packet_size = packet_sizes.get(frame.pts) if frame.pts is not None else None
            packet_supported = (
                isinstance(packet_size, int)
                and packet_size >= 0
                and frame.pts not in ambiguous_packet_pts
            )
            vectors, side_data_valid = _motion_vectors(frame.side_data)
            frame_type = (
                PictureType(frame.pict_type).name if frame.pict_type else "UNKNOWN"
            )
            mv_missing = frame_type.upper() != "I" and not vectors
            frames.append(
                build_candidate_feature(
                    index=index,
                    pts_seconds=pts_seconds,
                    packet_size_bytes=packet_size if packet_supported else 0,
                    frame_width=width,
                    frame_height=height,
                    vectors=vectors,
                    frame_type=frame_type,
                    key_frame=bool(frame.key_frame),
                    mv_missing=mv_missing,
                    packet_attribution_supported=packet_supported,
                    pts_supported=pts_supported,
                    mv_side_data_valid=side_data_valid,
                )
            )

    return CodecProbe(
        path=str(video_path),
        codec_name=codec_name,
        width=width,
        height=height,
        average_rate=average_rate,
        time_base=time_base,
        frames=tuple(frames),
    )


def aligned_candidate_features(
    probe: CodecProbe,
    candidate_seconds: Iterable[float],
    source_frame_period_seconds: float,
) -> tuple[CandidateFeature, ...]:
    candidate_values = tuple(candidate_seconds)
    decoded_pts = tuple(frame.pts_seconds for frame in probe.frames)
    aligned = align_candidate_pts(
        candidate_values, decoded_pts, source_frame_period_seconds
    )
    return tuple(probe.frames[item.decoded_index] for item in aligned)
