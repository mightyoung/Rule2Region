"""Build a recoverable VIRAT videos-00 derivative containing labeled videos only."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fieldpack.virat-videos00-labeled-derivative.v1"


def build(*, audit_result: Path, media_root: Path, annotations_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    audit = json.loads(audit_result.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "fieldpack.virat-videos00-media-audit.v1":
        raise ValueError("unexpected_audit_schema")
    rows = audit.get("videos")
    if not isinstance(rows, list) or len(rows) != 46:
        raise ValueError("expected_46_audited_videos")

    included = [row for row in rows if row.get("annotation_matched") is True]
    excluded = [row for row in rows if row.get("annotation_matched") is not True]
    if len(included) != 38 or len(excluded) != 8:
        raise ValueError("expected_38_included_and_8_excluded")

    output.mkdir(parents=True)
    try:
        media_output = output / "videos_original"
        annotation_output = output / "annotations"
        media_output.mkdir()
        annotation_output.mkdir()
        manifest_rows: list[dict[str, Any]] = []
        for row in included:
            source_id = str(row["source_video_id"])
            source_media = media_root / str(row["media_relative_path"])
            annotation_relative = row.get("annotation_relative_path")
            if not source_media.is_file() or not isinstance(annotation_relative, str):
                raise FileNotFoundError(f"missing_included_source:{source_id}")
            source_annotation = annotations_root / annotation_relative
            if not source_annotation.is_file():
                raise FileNotFoundError(f"missing_included_annotation:{source_id}")
            media_target = media_output / source_media.name
            annotation_target = annotation_output / source_annotation.name
            os.link(source_media, media_target)
            shutil.copy2(source_annotation, annotation_target)
            manifest_rows.append(
                {
                    "scene_id": row["scene_id"],
                    "source_video_id": source_id,
                    "video_sha256": row["media"]["sha256"],
                    "annotation_sha256": row["annotation"]["sha256"],
                    "strict_annotation_frame_range": row["annotation"]["frame_indices_in_range"],
                    "maximum_annotation_frame": row["annotation"]["maximum_frame_index"],
                    "video_frame_count": row["media"]["frame_count"],
                }
            )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "evidence_scope": "DEVELOPMENT_ONLY",
            "construction": "hardlinked_media_and_copied_annotations",
            "source_audit_result": str(audit_result),
            "source_archive_sha256": audit["archive_sha256"],
            "included_video_count": len(included),
            "excluded_video_count": len(excluded),
            "included_scene_ids": sorted({str(row["scene_id"]) for row in included}),
            "excluded_source_video_ids": [str(row["source_video_id"]) for row in excluded],
            "annotation_endpoint_policy": "UNRESOLVED_DO_NOT_SAMPLE_ENDPOINT",
            "gpu_experiment_allowed": False,
            "model_inference_run": False,
            "videos": manifest_rows,
        }
        (output / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return payload
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-result", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        audit_result=args.audit_result,
        media_root=args.media_root,
        annotations_root=args.annotations_root,
        output=args.output,
    )
    print(json.dumps({key: result[key] for key in ("included_video_count", "excluded_video_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
