from pathlib import Path

import av
import numpy as np

from codecref_lite.alignment import candidate_times
from codecref_lite.codec import aligned_candidate_features, probe_video
from codecref_lite.selectors import codec_guided_select


def _encode_bframe_fixture(path: Path) -> None:
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


def test_pyav_cfr_bframe_codec_contract(tmp_path: Path) -> None:
    video_path = tmp_path / "cfr-bframes.mp4"
    _encode_bframe_fixture(video_path)
    probe = probe_video(video_path)
    assert probe.codec_name == "h264"
    assert len(probe.frames) == 48
    assert any(frame.frame_type == "B" for frame in probe.frames)
    assert all(frame.pts_supported for frame in probe.frames)
    assert all(frame.packet_attribution_supported for frame in probe.frames)
    assert any(
        not frame.mv_missing for frame in probe.frames if frame.frame_type != "I"
    )
    selected = aligned_candidate_features(probe, candidate_times(0.0), 1 / 12)
    assert len(selected) == 8
    assert [round(frame.pts_seconds, 3) for frame in selected] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]
    assert len(codec_guided_select(selected, 4)) == 4
