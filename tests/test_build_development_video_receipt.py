from pathlib import Path

from research_tools.remote.build_development_video_receipt import transcode_command


def test_development_transcode_matches_frozen_codec_contract(tmp_path):
    command = transcode_command(Path("fixture.mp4"), tmp_path / "crf32.mp4", fps=12, frames=48, crf=32)
    assert command[command.index("-frames:v") + 1] == "48"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-profile:v") + 1] == "high"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-crf") + 1] == "32"
    assert command[command.index("-x264-params") + 1] == "keyint=24:min-keyint=24:scenecut=0:bframes=2:ref=3"
