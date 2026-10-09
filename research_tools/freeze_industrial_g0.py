from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from codecref_lite.datasets.industrial_g0 import (
    DatasetAdmission,
    PilotVideo,
    build_pilot_manifest,
)
from evidencegate.manifest import manifest_sha256, write_immutable_manifest


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_g0_pilot(
    admission_payload: dict[str, Any],
    inventory_payload: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    admissions: dict[str, DatasetAdmission] = {}
    for item in admission_payload.get("datasets", []):
        material = {key: value for key, value in item.items() if key != "role"}
        admission = DatasetAdmission(**material)
        if admission.dataset_id in admissions:
            raise ValueError(f"duplicate_dataset_admission:{admission.dataset_id}")
        admissions[admission.dataset_id] = admission

    videos = tuple(PilotVideo(**item) for item in inventory_payload.get("videos", []))
    if not videos:
        raise ValueError("video_inventory_empty")
    for dataset_id in sorted({video.dataset_id for video in videos}):
        admission = admissions.get(dataset_id)
        if admission is None:
            raise ValueError(f"dataset_admission_missing:{dataset_id}")
        if admission.status != "READY":
            raise PermissionError(
                f"dataset_admission_blocked:{dataset_id}:{','.join(admission.blockers)}"
            )

    manifest = build_pilot_manifest(
        videos,
        frames_per_video=int(inventory_payload["frames_per_video"]),
        minimum_gap_seconds=float(inventory_payload["minimum_gap_seconds"]),
    )
    manifest["admissions"] = [
        admissions[dataset_id].to_dict()
        for dataset_id in sorted({video.dataset_id for video in videos})
    ]
    embedded_identity = manifest.pop("manifest_sha256")
    manifest["upstream_selection_manifest_sha256"] = embedded_identity
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    artifact = write_immutable_manifest(output_path, manifest)
    return {
        "status": "FROZEN",
        "output_path": artifact.path,
        "output_file_sha256": artifact.sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "sample_count": manifest["sample_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a fail-closed FieldPack G0 industrial pilot manifest."
    )
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze_g0_pilot(
        _load_json(args.admission),
        _load_json(args.inventory),
        args.output,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
