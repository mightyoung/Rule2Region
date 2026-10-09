"""Fail-closed remote dataset admission audit for SceneConfig.

The audit reads archives in place.  It does not extract datasets or inspect
model outputs, and it never selects confirmation cameras from annotation
quality.  Its only job is to prove archive identity and video/annotation
mapping availability before an experiment manifest can be frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VIRAT_CAMERA = re.compile(r"^(VIRAT_S_\d{6})_")
CHOKE_NESTED = re.compile(r"^(P[12][EL]_S\d+_C[123])\.tar\.xz$")
CHOKE_XML = re.compile(r"^(P[12][EL]_S\d+_C[123])(?:\.\d+)?\.xml$")
DEV_CAMERAS = {
    "VIRAT_S_010004",
    "VIRAT_S_010111",
    "VIRAT_S_010112",
    "VIRAT_S_010113",
    "VIRAT_S_010204",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("algorithm") != "sha256":
        raise ValueError("expected_hash_manifest_must_use_sha256")
    return {str(row["path"]): row for row in payload["entries"]}


def verify_identity(dataset_root: Path, expected: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for relative, row in sorted(expected.items()):
        target = dataset_root / relative
        if not target.is_file():
            raise FileNotFoundError(f"missing_archive:{relative}")
        observed_bytes = target.stat().st_size
        if observed_bytes != int(row["bytes"]):
            raise ValueError(f"archive_size_mismatch:{relative}")
        observed_sha = sha256(target)
        if observed_sha != str(row["sha256"]).lower():
            raise ValueError(f"archive_sha256_mismatch:{relative}")
        receipts.append({"path": relative, "bytes": observed_bytes, "sha256": observed_sha})
    return receipts


def audit_virat(virat_root: Path) -> dict[str, Any]:
    video_zip = virat_root / "videos-01.zip"
    annotation_root = virat_root / "annotations"
    if not video_zip.is_file() or not annotation_root.is_dir():
        raise FileNotFoundError("virat_archive_or_annotation_directory_missing")
    with zipfile.ZipFile(video_zip) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise ValueError(f"virat_zip_crc_failed:{broken}")
        video_stems = {
            Path(name).stem
            for name in archive.namelist()
            if name.lower().endswith((".mp4", ".avi"))
        }
    object_stems = {
        path.name.removesuffix(".viratdata.objects.txt")
        for path in annotation_root.glob("*.viratdata.objects.txt")
    }
    mapped = sorted(video_stems & object_stems)
    missing_objects = sorted(video_stems - object_stems)
    by_camera: dict[str, list[str]] = defaultdict(list)
    for stem in mapped:
        match = VIRAT_CAMERA.match(stem)
        if match:
            by_camera[match.group(1)].append(stem)
    eligible = sorted(
        camera for camera, videos in by_camera.items()
        if len(videos) >= 2 and camera not in DEV_CAMERAS
    )
    if len(eligible) < 6:
        raise ValueError("fewer_than_six_eligible_virat_cameras")
    return {
        "video_count": len(video_stems),
        "object_annotation_count": len(object_stems),
        "mapped_video_count": len(mapped),
        "missing_object_annotation_count": len(missing_objects),
        "missing_object_annotations": missing_objects,
        "camera_group_count": len(by_camera),
        "eligible_confirmation_camera_count": len(eligible),
        "eligible_confirmation_cameras": eligible,
        "development_cameras_excluded": sorted(DEV_CAMERAS),
        "zip_crc_verified": True,
    }


def _nested_image_count(outer: tarfile.TarFile, member: tarfile.TarInfo) -> int:
    raw = outer.extractfile(member)
    if raw is None:
        raise ValueError(f"nested_archive_unreadable:{member.name}")
    count = 0
    with raw, tarfile.open(fileobj=raw, mode="r|xz") as nested:
        for row in nested:
            if row.isfile() and row.name.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    if count == 0:
        raise ValueError(f"nested_archive_has_no_images:{member.name}")
    return count


def audit_chokepoint(choke_root: Path, *, verify_nested: bool) -> dict[str, Any]:
    scenario_archives = sorted(path for path in choke_root.glob("P*.tar.xz") if path.is_file())
    if len(scenario_archives) != 18:
        raise ValueError(f"expected_18_chokepoint_scenarios:observed_{len(scenario_archives)}")
    groundtruth = choke_root / "groundtruth.tar.xz"
    if not groundtruth.is_file():
        raise FileNotFoundError("chokepoint_groundtruth_missing")

    xml_by_camera: Counter[str] = Counter()
    xml_count = 0
    with tarfile.open(groundtruth, mode="r:xz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            xml_count += 1
            basename = Path(member.name).name
            match = CHOKE_XML.match(basename)
            if not match:
                raise ValueError(f"unrecognized_chokepoint_xml:{basename}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable_chokepoint_xml:{basename}")
            with handle:
                ET.parse(handle)
            xml_by_camera[match.group(1)] += 1

    nested_rows: list[dict[str, Any]] = []
    view_units: set[str] = set()
    for scenario in scenario_archives:
        with tarfile.open(scenario, mode="r:xz") as archive:
            members = [row for row in archive.getmembers() if row.isfile()]
            if len(members) != 3:
                raise ValueError(f"scenario_does_not_contain_three_cameras:{scenario.name}")
            for member in members:
                basename = Path(member.name).name
                match = CHOKE_NESTED.match(basename)
                if not match:
                    raise ValueError(f"unrecognized_nested_camera_archive:{basename}")
                camera = match.group(1)
                parts = camera.split("_")
                view_units.add(f"{parts[0]}_{parts[2]}")
                nested_rows.append({
                    "scenario_archive": scenario.name,
                    "camera_archive": basename,
                    "camera_id": camera,
                    "xml_count": xml_by_camera[camera],
                    "image_count": _nested_image_count(archive, member) if verify_nested else None,
                })
    mapped_nested = sum(row["xml_count"] >= 1 for row in nested_rows)
    unmapped_nested = sorted(row["camera_id"] for row in nested_rows if row["xml_count"] == 0)
    expected_unmapped = sorted(
        f"P2{direction}_S5_C{camera}"
        for direction in ("E", "L")
        for camera in range(1, 4)
    )
    if len(nested_rows) != 54 or len(view_units) < 6:
        raise ValueError("chokepoint_camera_or_view_unit_count_failed")
    if xml_count != 72 or mapped_nested != 48 or unmapped_nested != expected_unmapped:
        raise ValueError("chokepoint_annotation_coverage_contract_failed")
    return {
        "scenario_archive_count": len(scenario_archives),
        "nested_camera_archive_count": len(nested_rows),
        "groundtruth_xml_count": xml_count,
        "mapped_nested_camera_count": mapped_nested,
        "unlabeled_nested_camera_count": len(unmapped_nested),
        "unlabeled_nested_cameras_excluded": unmapped_nested,
        "view_unit_count": len(view_units),
        "view_units": sorted(view_units),
        "nested_payload_integrity_verified": verify_nested,
        "nested_cameras": nested_rows,
    }


def run(
    dataset_root: Path,
    expected_hashes: Path,
    *,
    verify_nested: bool = False,
) -> dict[str, Any]:
    expected = load_expected(expected_hashes)
    identities = verify_identity(dataset_root, expected)
    virat = audit_virat(dataset_root / "VIRAT Ground Dataset")
    chokepoint = audit_chokepoint(dataset_root / "CHOKEPOINT", verify_nested=verify_nested)
    return {
        "schema_version": "sceneconfig.dataset-admission.v1",
        "status": "passed",
        "confirmation_manifest_frozen": False,
        "annotation_quality_used_for_camera_selection": False,
        "archive_identity": identities,
        "virat": virat,
        "chokepoint": chokepoint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-nested", action="store_true")
    args = parser.parse_args()
    payload = run(args.dataset_root, args.expected_hashes, verify_nested=args.verify_nested)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": payload["status"],
        "virat_eligible_cameras": payload["virat"]["eligible_confirmation_camera_count"],
        "chokepoint_views": payload["chokepoint"]["view_unit_count"],
        "nested_verified": payload["chokepoint"]["nested_payload_integrity_verified"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
