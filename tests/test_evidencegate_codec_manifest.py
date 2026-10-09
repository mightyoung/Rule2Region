import hashlib
import json
from pathlib import Path

import pytest

from evidencegate.codec_manifest import validate_codec_derivative_manifest
from evidencegate.execution_config import compile_execution_config
from evidencegate.manifest import canonical_json, manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
G0_DIR = ROOT / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g0"
VIDEO_SCOPE = G0_DIR / "g0-video-scope.json"
EXECUTION_SCOPE = G0_DIR / "g0-execution-scope-v2.json"


def _command(source: dict, fps: int, crf: int, output: str) -> list[str]:
    gop = 2 * fps
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-fflags", "+bitexact",
        "-framerate", str(fps), "-start_number", str(source["start_frame_inclusive"]),
        "-i", "/dataset/img1/%06d.jpg", "-frames:v", str(source["frame_count"]), "-an",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "medium",
        "-crf", str(crf), "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=2:ref=3",
        "-r", str(fps), "-fps_mode", "cfr", "-map_metadata", "-1", "-movflags", "+faststart",
        "-y", output + ".partial.mp4",
    ]


def _write_manifest(path: Path, *, corrupt_probe: bool = False) -> dict:
    scope = json.loads(VIDEO_SCOPE.read_text(encoding="utf-8"))
    entries = []
    for source in scope["samples"]:
        suffix = source["sequence_id"].split("-")[-1]
        fps = int(scope["sequence_receipts"][suffix]["fps"])
        derivatives = {}
        for condition, crf in (("codec_guided_crf18", 18), ("bitrate_only_crf32", 32)):
            output = f"/share/codec/{source['sample_id']}/{condition}.mp4"
            command = _command(source, fps, crf, output)
            item = {
                "path": output,
                "sha256": "a" * 64,
                "bytes": 100,
                "crf": crf,
                "command": command,
                "command_sha256": hashlib.sha256(canonical_json(command)).hexdigest(),
                "probe": {
                    "codec_name": "h264",
                    "decoded_frames": source["frame_count"],
                    "pts_monotonic": not corrupt_probe,
                    "packet_attribution_supported_frames": source["frame_count"],
                    "motion_vector_frames": source["frame_count"] - 1,
                    "valid_mv_side_data_frames": source["frame_count"] - 1,
                    "time_base": "1/15360",
                },
            }
            if condition == "codec_guided_crf18":
                item.update({
                    "feature_path": output + ".features.json",
                    "feature_sha256": "b" * 64,
                    "feature_count": source["frame_count"],
                })
            derivatives[condition] = item
        entries.append({
            "sample_id": source["sample_id"],
            "sequence_id": source["sequence_id"],
            "camera_motion_stratum": source["camera_motion_stratum"],
            "source_frame_tree_sha256": source["frame_tree_sha256"],
            "frame_count": source["frame_count"],
            "fps": fps,
            "derivatives": derivatives,
        })
    manifest = {
        "schema_version": "evidencegate.codec-derivatives.v1",
        "source_scope_file_sha256": hashlib.sha256(VIDEO_SCOPE.read_bytes()).hexdigest(),
        "source_scope_manifest_sha256": scope["manifest_sha256"],
        "pyav_version": "17.0.0",
        "ffmpeg_version": "ffmpeg version 6.1.4 Copyright",
        "entries": entries,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    path.write_bytes(canonical_json(manifest))
    return manifest


def test_codec_manifest_verifier_closes_all_50_windows_and_binds_execution_config(tmp_path):
    path = tmp_path / "codec-manifest.json"
    _write_manifest(path)
    file_hash = validate_codec_derivative_manifest(path, VIDEO_SCOPE)
    assert file_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    config = compile_execution_config(
        EXECUTION_SCOPE,
        query_provider_resolution_sha256="d" * 64,
        runtime_environment_receipt_sha256="e" * 64,
        development_video_manifest_sha256="f" * 64,
        model_cache_index_sha256="1" * 64,
        g2_output_index_sha256="2" * 64,
        prompt_adjustment_receipt_sha256="3" * 64,
        codec_derivative_manifest_path=path,
        video_scope_path=VIDEO_SCOPE,
    )
    assert config.codec_derivative_manifest_sha256 == file_hash


def test_codec_manifest_verifier_rejects_semantically_invalid_probe(tmp_path):
    path = tmp_path / "codec-manifest.json"
    _write_manifest(path, corrupt_probe=True)
    with pytest.raises(ValueError, match="codec_probe_invalid"):
        validate_codec_derivative_manifest(path, VIDEO_SCOPE)
