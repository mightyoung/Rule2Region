from __future__ import annotations

import io
import json
import lzma
import tarfile
import zipfile
from pathlib import Path

from research_tools.sceneconfig_d0_audit import audit_chokepoint, audit_virat


def _inner_archive() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:xz") as archive:
        payload = b"jpeg"
        info = tarfile.TarInfo("0001.jpg")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def test_chokepoint_mapping_and_nested_integrity(tmp_path: Path) -> None:
    root = tmp_path / "CHOKEPOINT"
    root.mkdir()
    xml_buffer = io.BytesIO()
    with tarfile.open(fileobj=xml_buffer, mode="w:xz") as archive:
        for portal in (1, 2):
            for direction in ("E", "L"):
                session_count = 4 if portal == 1 else 5
                labeled_sessions = session_count - 1 if portal == 2 else session_count
                for session in range(1, labeled_sessions + 1):
                    for camera in range(1, 4):
                        suffixes = (".1", ".2") if portal == 2 else ("",)
                        for suffix in suffixes:
                            name = f"P{portal}{direction}_S{session}_C{camera}{suffix}.xml"
                            payload = b"<dataset/>"
                            info = tarfile.TarInfo(f"groundtruth/{name}")
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
    (root / "groundtruth.tar.xz").write_bytes(xml_buffer.getvalue())
    nested = _inner_archive()
    for portal in (1, 2):
        for direction in ("E", "L"):
            session_count = 4 if portal == 1 else 5
            for session in range(1, session_count + 1):
                with tarfile.open(root / f"P{portal}{direction}_S{session}.tar.xz", mode="w:xz") as archive:
                    for camera in range(1, 4):
                        info = tarfile.TarInfo(f"P{portal}{direction}_S{session}_C{camera}.tar.xz")
                        info.size = len(nested)
                        archive.addfile(info, io.BytesIO(nested))
    result = audit_chokepoint(root, verify_nested=True)
    assert result["scenario_archive_count"] == 18
    assert result["nested_camera_archive_count"] == 54
    assert result["groundtruth_xml_count"] == 72
    assert result["mapped_nested_camera_count"] == 48
    assert result["unlabeled_nested_camera_count"] == 6
    assert result["view_unit_count"] == 12


def test_virat_requires_six_non_development_camera_groups(tmp_path: Path) -> None:
    root = tmp_path / "VIRAT Ground Dataset"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    with zipfile.ZipFile(root / "videos-01.zip", mode="w") as archive:
        for camera in range(6):
            for sequence in range(2):
                stem = f"VIRAT_S_{200000 + camera:06d}_{sequence:02d}_000000_000100"
                archive.writestr(f"videos_original/{stem}.mp4", b"video")
                (annotations / f"{stem}.viratdata.objects.txt").write_text("", encoding="utf-8")
    result = audit_virat(root)
    assert result["eligible_confirmation_camera_count"] == 6
    assert result["mapped_video_count"] == 12
    assert result["zip_crc_verified"] is True
