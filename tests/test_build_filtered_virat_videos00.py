import hashlib
import json
from pathlib import Path

from research_tools.data_audit.build_filtered_virat_videos00 import build


def test_builds_38_video_derivative_and_excludes_eight(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    annotations_root = tmp_path / "annotations"
    media_root.mkdir()
    annotations_root.mkdir()
    rows = []
    for index in range(46):
        source_id = f"VIRAT_S_0002{index:02d}"
        media = media_root / f"{source_id}.mp4"
        media.write_bytes(f"video-{index}".encode())
        matched = index < 38
        annotation_name = f"{source_id}.viratdata.objects.txt"
        if matched:
            (annotations_root / annotation_name).write_text("0 1 0 0 0 1 1 1\n", encoding="utf-8")
        rows.append(
            {
                "scene_id": "0002",
                "source_video_id": source_id,
                "media_relative_path": media.name,
                "annotation_relative_path": annotation_name if matched else None,
                "annotation_matched": matched,
                "media": {"sha256": hashlib.sha256(media.read_bytes()).hexdigest(), "frame_count": 2},
                "annotation": {
                    "sha256": hashlib.sha256(b"annotation").hexdigest(),
                    "frame_indices_in_range": True,
                    "maximum_frame_index": 0,
                } if matched else None,
            }
        )
    audit = tmp_path / "result.json"
    audit.write_text(json.dumps({
        "schema_version": "fieldpack.virat-videos00-media-audit.v1",
        "archive_sha256": "a" * 64,
        "videos": rows,
    }), encoding="utf-8")

    output = tmp_path / "filtered"
    result = build(
        audit_result=audit,
        media_root=media_root,
        annotations_root=annotations_root,
        output=output,
    )

    assert result["included_video_count"] == 38
    assert result["excluded_video_count"] == 8
    assert result["gpu_experiment_allowed"] is False
    assert len(list((output / "videos_original").glob("*.mp4"))) == 38
    assert len(list((output / "annotations").glob("*.objects.txt"))) == 38
