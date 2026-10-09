import hashlib
import json
from pathlib import Path

from research_tools.validate_virat_videos00_media_audit import (
    EXPECTED_ARCHIVE_SHA256,
    validate,
)


def _payload() -> dict:
    videos = []
    for scene_id, count in (("0000", 5), ("0001", 2), ("0002", 39)):
        for index in range(count):
            videos.append(
                {
                    "scene_id": scene_id,
                    "source_video_id": f"VIRAT_S_{scene_id}{index:02d}",
                    "annotation_matched": True,
                    "checks_passed": True,
                    "media": {
                        "sha256": hashlib.sha256(f"{scene_id}-{index}".encode()).hexdigest(),
                        "opened": True,
                        "metadata_valid": True,
                        "first_frame_decodable": True,
                        "last_frame_decodable": True,
                    },
                    "annotation": {"parse_ok": True, "frame_indices_in_range": True},
                }
            )
    return {
        "schema_version": "fieldpack.virat-videos00-media-audit.v1",
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "scene_count": 3,
        "video_count": 46,
        "all_checks_passed": True,
        "raw_media_extracted": True,
        "gpu_used": False,
        "model_inference": False,
        "videos": videos,
    }


def test_validate_complete_bundle_and_manifest(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_payload()), encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  ./result.json\n", encoding="utf-8")

    result = validate(result_path)

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["scene_video_counts"] == {
        "VIRAT_S_0000": 5,
        "VIRAT_S_0001": 2,
        "VIRAT_S_0002": 39,
    }
    assert result["manifest_detail"] == "verified"


def test_validate_detects_manifest_tampering(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_payload()), encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  ./result.json\n", encoding="utf-8")

    result = validate(result_path)

    assert result["passed"] is False
    assert result["checks"]["manifest"] is False
    assert result["manifest_detail"] == "hash_mismatch_result.json"


def test_media_check_stays_true_when_only_annotation_check_fails(tmp_path: Path) -> None:
    payload = _payload()
    payload["all_checks_passed"] = False
    payload["videos"][0]["annotation_matched"] = False
    payload["videos"][0]["annotation"] = None
    payload["videos"][0]["checks_passed"] = False
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate(result_path)

    assert result["passed"] is False
    assert result["checks"]["all_media_decodable"] is True
    assert result["checks"]["all_annotations_mapped_and_in_range"] is False
    assert result["checks"]["manifest"] is False
    assert result["manifest_detail"] == "not_present"
