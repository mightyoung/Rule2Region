from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np

from codecref_lite.alignment import candidate_times
from codecref_lite.codec import aligned_candidate_features, probe_video
from codecref_lite.io import sha256_file
from codecref_lite.selectors import codec_guided_select


def encode_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"bf": "2", "g": "12", "crf": "23", "preset": "fast"}
        for index in range(48):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            left = index % 48
            image[20:32, left : min(left + 12, 64), 0] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()

    encode_fixture(arguments.video)
    probe = probe_video(arguments.video)
    candidates = aligned_candidate_features(probe, candidate_times(0.0), 1 / 12)
    selected = codec_guided_select(candidates, 4)
    frame_types = Counter(frame.frame_type for frame in probe.frames)
    report = {
        "artifact_role": "synthetic_codec_contract_fixture_not_research_result",
        "video": str(arguments.video),
        "video_sha256": sha256_file(arguments.video),
        "pyav_version": av.__version__,
        "ffmpeg_library_versions": av.library_versions,
        "codec_name": probe.codec_name,
        "width": probe.width,
        "height": probe.height,
        "average_rate": probe.average_rate,
        "time_base": probe.time_base,
        "decoded_frames": len(probe.frames),
        "frame_types": dict(sorted(frame_types.items())),
        "packet_attribution_supported_frames": sum(
            frame.packet_attribution_supported for frame in probe.frames
        ),
        "motion_vector_supported_non_i_frames": sum(
            not frame.mv_missing for frame in probe.frames if frame.frame_type != "I"
        ),
        "candidate_pts": [frame.pts_seconds for frame in candidates],
        "codec_selected_indices": list(selected),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
