from pathlib import Path

import cv2
import numpy as np
import pytest

from research_tools.data_audit.audit_virat_videos00_media import (
    SCHEMA_VERSION,
    audit,
    parse_annotation,
)


def _write_video(path: Path, *, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    for value in range(frames):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()


def test_audit_checks_three_scenes_media_and_annotations(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    for index, scene_id in enumerate(("0000", "0001", "0002")):
        source_id = f"VIRAT_S_{scene_id}01_00_000000_000004"
        _write_video(media_root / scene_id / f"{source_id}.mp4")
        (annotation_root / f"{source_id}.viratdata.objects.txt").write_text(
            "0 4 0 1 2 3 4 1\n0 4 3 2 3 4 5 1\n1 1 2 1 1 2 2 2\n",
            encoding="utf-8",
        )

    output = tmp_path / "result.json"
    result = audit(
        media_root=media_root,
        annotations_root=annotation_root,
        output_json=output,
        archive_path="/datasets/videos-00.zip",
        archive_sha256="a" * 64,
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["evidence_scope"] == "DEVELOPMENT_ONLY"
    assert result["gpu_used"] is False
    assert result["model_inference"] is False
    assert result["raw_media_extracted"] is True
    assert result["scene_count"] == 3
    assert result["video_count"] == 3
    assert result["all_checks_passed"] is True
    assert result["summary"]["scene_count"] == 3
    assert result["summary"]["video_count"] == 3
    assert result["summary"]["annotation_row_count"] == 9
    assert result["summary"]["person_row_count"] == 6
    assert result["summary"]["all_checks_passed"] is True
    assert all(row["media"]["first_frame_decodable"] for row in result["videos"])
    assert all(row["media"]["last_frame_decodable"] for row in result["videos"])
    assert output.exists()


def test_parse_annotation_reports_malformed_and_out_of_range_rows(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.viratdata.objects.txt"
    annotation.write_text("0 1 5 1 2 3 4 1\nnot eight columns\n", encoding="utf-8")

    result = parse_annotation(annotation, total_frames=4)

    assert result["row_count"] == 1
    assert result["malformed_line_numbers"] == [2]
    assert result["parse_ok"] is False
    assert result["frame_indices_in_range"] is False


def test_audit_rejects_scene_set_other_than_videos00_scope(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    _write_video(media_root / "VIRAT_S_000001_00_000000_000004.mp4")

    with pytest.raises(ValueError, match="unexpected_scene_set"):
        audit(
            media_root=media_root,
            annotations_root=annotation_root,
            output_json=tmp_path / "result.json",
            archive_path="videos-00.zip",
            archive_sha256="b" * 64,
        )
