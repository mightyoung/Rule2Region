from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import canonical_json, manifest_sha256


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name}_must_be_sha256")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_value(command: list[Any], flag: str) -> str:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"codec_command_missing:{flag}") from exc


def validate_codec_derivative_manifest(
    manifest_path: str | Path,
    video_scope_path: str | Path,
    *,
    verify_files: bool = False,
) -> str:
    """Validate the frozen 50-window codec manifest and return its file SHA-256."""
    manifest_file = Path(manifest_path)
    scope_file = Path(video_scope_path)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    scope = json.loads(scope_file.read_bytes())
    if manifest.get("schema_version") != "evidencegate.codec-derivatives.v1":
        raise ValueError("codec_manifest_schema_invalid")
    if manifest.get("source_scope_file_sha256") != _file_sha256(scope_file):
        raise ValueError("codec_manifest_scope_file_hash_mismatch")
    if manifest.get("source_scope_manifest_sha256") != scope.get("manifest_sha256"):
        raise ValueError("codec_manifest_scope_identity_mismatch")
    material = dict(manifest)
    embedded_hash = _require_sha256(material.pop("manifest_sha256", ""), "codec_manifest_identity")
    if manifest_sha256(material) != embedded_hash:
        raise ValueError("codec_manifest_identity_mismatch")
    if not str(manifest.get("ffmpeg_version", "")).startswith("ffmpeg version 6.1.4"):
        raise ValueError("codec_manifest_ffmpeg_version_mismatch")
    if not str(manifest.get("pyav_version", "")).startswith("17."):
        raise ValueError("codec_manifest_pyav_version_mismatch")

    expected = {str(item["sample_id"]): item for item in scope["samples"]}
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ValueError("codec_manifest_requires_50_entries")
    actual_ids = [str(item.get("sample_id")) for item in entries]
    if len(set(actual_ids)) != 50 or set(actual_ids) != set(expected):
        raise ValueError("codec_manifest_sample_coverage_mismatch")

    for entry in entries:
        sample_id = str(entry["sample_id"])
        source = expected[sample_id]
        if entry.get("sequence_id") != source["sequence_id"]:
            raise ValueError(f"codec_sequence_mismatch:{sample_id}")
        if entry.get("camera_motion_stratum") != source["camera_motion_stratum"]:
            raise ValueError(f"codec_stratum_mismatch:{sample_id}")
        if entry.get("source_frame_tree_sha256") != source["frame_tree_sha256"]:
            raise ValueError(f"codec_source_hash_mismatch:{sample_id}")
        if int(entry.get("frame_count", -1)) != int(source["frame_count"]):
            raise ValueError(f"codec_frame_count_mismatch:{sample_id}")
        derivatives = entry.get("derivatives", {})
        if set(derivatives) != {"codec_guided_crf18", "bitrate_only_crf32"}:
            raise ValueError(f"codec_condition_coverage_mismatch:{sample_id}")
        for condition, expected_crf in (("codec_guided_crf18", 18), ("bitrate_only_crf32", 32)):
            item = derivatives[condition]
            output_hash = _require_sha256(item.get("sha256", ""), f"codec_output:{sample_id}:{condition}")
            if int(item.get("bytes", 0)) <= 0 or int(item.get("crf", -1)) != expected_crf:
                raise ValueError(f"codec_output_receipt_invalid:{sample_id}:{condition}")
            command = item.get("command")
            if not isinstance(command, list) or hashlib.sha256(canonical_json(command)).hexdigest() != item.get("command_sha256"):
                raise ValueError(f"codec_command_hash_mismatch:{sample_id}:{condition}")
            required_values = {
                "-c:v": "libx264",
                "-profile:v": "high",
                "-pix_fmt": "yuv420p",
                "-preset": "medium",
                "-crf": str(expected_crf),
                "-frames:v": str(source["frame_count"]),
                "-r": str(entry["fps"]),
                "-fps_mode": "cfr",
            }
            for flag, value in required_values.items():
                if _command_value(command, flag) != value:
                    raise ValueError(f"codec_command_value_mismatch:{sample_id}:{condition}:{flag}")
            gop = 2 * int(entry["fps"])
            if _command_value(command, "-x264-params") != f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=2:ref=3":
                raise ValueError(f"codec_x264_contract_mismatch:{sample_id}:{condition}")
            probe = item.get("probe", {})
            if (
                probe.get("codec_name") != "h264"
                or int(probe.get("decoded_frames", -1)) != int(source["frame_count"])
                or probe.get("pts_monotonic") is not True
                or int(probe.get("packet_attribution_supported_frames", -1)) != int(source["frame_count"])
                or int(probe.get("motion_vector_frames", 0)) <= 0
                or int(probe.get("valid_mv_side_data_frames", -1)) != int(probe.get("motion_vector_frames", -2))
                or not str(probe.get("time_base", ""))
            ):
                raise ValueError(f"codec_probe_invalid:{sample_id}:{condition}")
            if condition == "codec_guided_crf18":
                _require_sha256(item.get("feature_sha256", ""), f"codec_features:{sample_id}")
                if int(item.get("feature_count", -1)) != int(source["frame_count"]):
                    raise ValueError(f"codec_feature_count_mismatch:{sample_id}")
            elif any(key in item for key in ("feature_path", "feature_sha256", "feature_count")):
                raise ValueError(f"bitrate_only_feature_leak:{sample_id}")
            if verify_files:
                output_path = Path(str(item["path"]))
                if not output_path.is_file() or _file_sha256(output_path) != output_hash or output_path.stat().st_size != int(item["bytes"]):
                    raise ValueError(f"codec_output_file_mismatch:{sample_id}:{condition}")
                if condition == "codec_guided_crf18":
                    feature_path = Path(str(item["feature_path"]))
                    if not feature_path.is_file() or _file_sha256(feature_path) != item["feature_sha256"]:
                        raise ValueError(f"codec_feature_file_mismatch:{sample_id}")
    return hashlib.sha256(manifest_bytes).hexdigest()
