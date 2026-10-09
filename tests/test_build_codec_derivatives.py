import json
from pathlib import Path

import av
import numpy as np
import pytest

from research_tools.remote import build_codec_derivatives as codec


def sample_window() -> dict[str, object]:
    return {
        "sample_id": "MOT17-02:000001-000120",
        "frame_tree_sha256": "a" * 64,
        "start_frame_inclusive": 1,
        "frame_count": 120,
    }


def sequence_receipt(tmp_path: Path) -> dict[str, object]:
    return {"fps": 30, "seqinfo_path": str(tmp_path / "MOT17-02" / "seqinfo.ini")}


def test_encode_command_freezes_declared_h264_contract(tmp_path):
    output = tmp_path / "codec_guided_crf18.mp4"
    command = codec.encode_command(sample_window(), sequence_receipt(tmp_path), output, 18)
    assert command[0] == "ffmpeg"
    assert command[command.index("-profile:v") + 1] == "high"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-preset") + 1] == "medium"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-frames:v") + 1] == "120"
    assert command[command.index("-x264-params") + 1] == "keyint=60:min-keyint=60:scenecut=0:bframes=2:ref=3"


def test_encode_resume_requires_matching_immutable_receipt(tmp_path, monkeypatch):
    output = tmp_path / "codec_guided_crf18.mp4"

    def fake_run(command, check):
        assert check is True
        Path(command[-1]).write_bytes(b"encoded-h264")

    monkeypatch.setattr(codec.subprocess, "run", fake_run)
    command, command_hash = codec.encode(sample_window(), sequence_receipt(tmp_path), output, 18)
    receipt_path = output.with_suffix(".encode.json")
    sealed = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert sealed["command_sha256"] == command_hash
    assert sealed["output_sha256"] == codec.sha256_file(output)

    monkeypatch.setattr(codec.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("resume re-encoded"))
    assert codec.encode(sample_window(), sequence_receipt(tmp_path), output, 18) == (command, command_hash)

    output.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="derivative_content_conflict"):
        codec.encode(sample_window(), sequence_receipt(tmp_path), output, 18)


def test_encode_rejects_orphan_and_identity_conflict(tmp_path, monkeypatch):
    output = tmp_path / "codec_guided_crf18.mp4"
    output.write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="orphan_derivative_without_receipt"):
        codec.encode(sample_window(), sequence_receipt(tmp_path), output, 18)

    output.unlink()

    def fake_run(command, check):
        Path(command[-1]).write_bytes(b"encoded-h264")

    monkeypatch.setattr(codec.subprocess, "run", fake_run)
    codec.encode(sample_window(), sequence_receipt(tmp_path), output, 18)
    with pytest.raises(RuntimeError, match="derivative_identity_conflict"):
        codec.encode({**sample_window(), "frame_count": 119}, sequence_receipt(tmp_path), output, 18)


def test_write_atomic_refuses_manifest_overwrite(tmp_path):
    target = tmp_path / "manifest.json"
    codec.write_atomic(target, b"one\n")
    assert codec.write_atomic(target, b"one\n") == codec.sha256_bytes(b"one\n")
    with pytest.raises(RuntimeError, match="immutable_conflict"):
        codec.write_atomic(target, b"two\n")


def test_probe_reports_real_bframe_packet_and_mv_receipts(tmp_path):
    video = tmp_path / "bframes.mp4"
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"bf": "2", "g": "12", "crf": "23", "preset": "fast"}
        for index in range(24):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[20:32, index : min(index + 12, 64), 0] = 255
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    summary, features = codec.probe_video(video, include_features=True)
    assert summary["decoded_frames"] == 24
    assert summary["pts_monotonic"] is True
    assert summary["packet_attribution_supported_frames"] == 24
    assert 0 < summary["valid_mv_side_data_frames"] <= summary["motion_vector_frames"]
    assert any(item["frame_type"] == "B" for item in features)
    assert all(item["packet_attribution_supported"] for item in features)
