import json
import tarfile
import zipfile
from pathlib import Path

from research_tools.data_audit.build_dataset_expansion_inventory import build_inventory


def test_metadata_inventory_builds_balanced_candidates(tmp_path: Path) -> None:
    virat = tmp_path / "virat"
    choke = tmp_path / "choke"
    virat.mkdir(); choke.mkdir()
    names = []
    for scene in range(6):
        for segment in range(4):
            names.append(f"VIRAT_S_02{scene:02d}00_{segment:02d}_{segment * 40:06d}_{segment * 40 + 40:06d}")
    with zipfile.ZipFile(virat / "videos-01.zip", "w") as archive:
        for name in names:
            archive.writestr(f"videos/{name}.mp4", b"")
    with zipfile.ZipFile(virat / "annotations-and-docs.zip", "w") as archive:
        for name in names:
            archive.writestr(f"annotations/{name}.viratdata.objects.txt", b"")
    for group, sessions in {"P1E": 4, "P1L": 4, "P2E": 5, "P2L": 5}.items():
        for session in range(1, sessions + 1):
            (choke / f"{group}_S{session}.tar.xz").write_bytes(b"")
    xml_root = tmp_path / "xml"
    xml_root.mkdir()
    for group in ("P1E", "P1L", "P2E", "P2L"):
        for session in range(1, 5):
            for camera in (1, 2, 3):
                (xml_root / f"{group}_S{session}_C{camera}.xml").write_text("<root/>", encoding="utf-8")
    with tarfile.open(choke / "groundtruth.tar.xz", "w:xz") as archive:
        for path in xml_root.iterdir():
            archive.add(path, arcname=f"groundtruth/{path.name}")
    result = build_inventory(virat_root=virat, chokepoint_root=choke)
    assert result["virat"]["structural_candidate_count"] == 6
    assert result["chokepoint"]["structural_candidate_count"] == 12
    assert len(result["chokepoint"]["recommended_six"]) == 6
    assert result["chokepoint"]["independent_location_count"] == 2
    assert result["readiness"]["structural_target_met"] is True
    assert result["readiness"]["gpu_experiment_allowed"] is False
