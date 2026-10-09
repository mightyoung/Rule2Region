import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from research_tools.prepare_expanded_fixed_camera import (
    _explicit_choke_members,
    _extract_choke_frames,
    parse_chokepoint_xml,
    parse_virat_objects,
    prepare,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jpeg(path: Path, *, size: tuple[int, int] = (32, 24)) -> None:
    Image.new("RGB", size, color=(10, 20, 30)).save(path, format="JPEG")


def _virat_manifest(media: Path, annotation: Path) -> dict:
    return {
        "schema_version": "fieldpack.formal-split.v1",
        "status": "frozen",
        "scenes": [{
            "scene_id": "0000",
            "dataset": "VIRAT",
            "source": "videos-00.zip",
            "media": media.name,
            "annotation": annotation.name,
            "media_sha256": _sha256(media),
            "annotation_sha256": _sha256(annotation),
            "frame_count": 20,
            "fps": 30.0,
            "segments": [
                {"id": "H1", "start": 2, "end": 3},
                {"id": "F1", "start": 12, "end": 13},
            ],
        }],
    }


def test_virat_terminal_boundary_is_dropped_and_later_frame_is_rejected() -> None:
    rows = parse_virat_objects(
        "1 1 9 10 20 30 40 1\n2 1 10 1 2 3 4 1\n",
        {9},
        frame_count=10,
        source_name="objects.txt",
    )
    assert rows[9][0]["bbox_xywh"] == [10.0, 20.0, 30.0, 40.0]
    with pytest.raises(ValueError, match="virat_annotation_frame_out_of_range"):
        parse_virat_objects(
            "1 1 11 10 20 30 40 1\n",
            {9},
            frame_count=10,
            source_name="objects.txt",
        )


def test_chokepoint_xml_produces_person_ids_and_eye_points_without_boxes() -> None:
    payload = b"""<dataset><frame number="112"><person id="7"><leftEye x="10" y="20"/><rightEye x="14" y="22"/></person></frame><frame number="113"><person id="8"><leftEye x="1" y="2"/></person></frame></dataset>"""
    rows = parse_chokepoint_xml(
        payload,
        {112},
        timeline_frame_count=4410,
        source_name="P1E_S1_C1.xml",
    )
    assert rows == {112: [{
        "person_id": "7",
        "eye_points": [
            {"label": "leftEye", "x": 10.0, "y": 20.0},
            {"label": "rightEye", "x": 14.0, "y": 22.0},
        ],
        "bbox_xywh": None,
        "source_annotation": "P1E_S1_C1.xml",
    }]}


def test_hash_mismatch_fails_before_output_creation(tmp_path: Path) -> None:
    media_root = tmp_path / "videos-00"
    annotation_root = tmp_path / "annotations"
    media_root.mkdir()
    annotation_root.mkdir()
    media = media_root / "sample.mp4"
    annotation = annotation_root / "sample.txt"
    media.write_bytes(b"media")
    annotation.write_text("", encoding="utf-8")
    manifest = _virat_manifest(media, annotation)
    manifest["scenes"][0]["media_sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="sha256_mismatch"):
        prepare(
            manifest_path,
            virat_media_roots={"videos-00": media_root},
            virat_annotations_root=annotation_root,
            chokepoint_root=tmp_path,
            output=output,
        )
    assert not output.exists()


def test_prepare_decodes_only_frozen_frames_and_records_roles(tmp_path: Path) -> None:
    media_root = tmp_path / "videos-00"
    annotation_root = tmp_path / "annotations"
    media_root.mkdir()
    annotation_root.mkdir()
    media = media_root / "sample.mp4"
    annotation = annotation_root / "sample.txt"
    media.write_bytes(b"media")
    annotation.write_text("1 1 2 10 20 30 40 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_virat_manifest(media, annotation)), encoding="utf-8")
    decoded: list[int] = []

    def decoder(_media: Path, frames: tuple[int, ...], destinations: dict[int, Path]) -> dict[str, int]:
        decoded.extend(frames)
        for frame in frames:
            destinations[frame].parent.mkdir(parents=True, exist_ok=True)
            destinations[frame].write_bytes(f"jpeg-{frame}".encode())
        return {"width": 800, "height": 600}

    output = tmp_path / "prepared"
    receipt = prepare(
        manifest_path,
        virat_media_roots={"videos-00": media_root},
        virat_annotations_root=annotation_root,
        chokepoint_root=tmp_path,
        output=output,
        decode_frames=decoder,
    )
    assert decoded == [2, 3, 12, 13]
    assert receipt["prepared_frame_count"] == 4
    truth = json.loads((output / "scenes" / "0000" / "truth.json").read_text(encoding="utf-8"))
    assert [(row["frame_index"], row["role"]) for row in truth["frames"]] == [
        (2, "history"), (3, "history"), (12, "future"), (13, "future")
    ]
    assert truth["frames"][0]["truth_count"] == 1
    assert truth["frames"][0]["truth_boxes"][0]["bbox_xywh"] == [10.0, 20.0, 30.0, 40.0]
    with pytest.raises(FileExistsError, match="refuse_existing_output"):
        prepare(
            manifest_path,
            virat_media_roots={"videos-00": media_root},
            virat_annotations_root=annotation_root,
            chokepoint_root=tmp_path,
            output=output,
            decode_frames=decoder,
        )


def test_sparse_chokepoint_members_use_original_frame_axis(tmp_path: Path) -> None:
    source = tmp_path / "nested-source"
    (source / "P1E_S1_C1").mkdir(parents=True)
    for frame in (1, 112, 4409):
        _write_jpeg(source / "P1E_S1_C1" / f"{frame:08d}.jpg")
    nested = tmp_path / "camera.tar.xz"
    with tarfile.open(nested, "w:xz") as archive:
        archive.add(source / "P1E_S1_C1", arcname="P1E_S1_C1")
    scene = {
        "scene_id": "P1",
        "media_timeline": "sparse_jpeg_names_on_original_frame_axis",
        "member_name_template": "P1E_S1_C1/{frame_index:08d}.jpg",
    }
    mapping = _explicit_choke_members(scene, [112])
    destination = tmp_path / "selected.jpg"
    dimensions = _extract_choke_frames(nested, mapping, {112: destination}, expected_member_count=3)
    assert dimensions == {"width": 32, "height": 24}
    with Image.open(destination) as image:
        assert image.size == (32, 24)
    assert not (tmp_path / "00000001.jpg").exists()


def test_chokepoint_contiguous_manifest_without_member_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="sparse_timeline_contract_required"):
        _explicit_choke_members({"scene_id": "P1"}, [112])


def test_prepare_chokepoint_emits_count_and_eye_midpoint_truth(tmp_path: Path) -> None:
    choke_root = tmp_path / "choke"
    image_root = tmp_path / "images" / "P1E_S1_C1"
    choke_root.mkdir()
    image_root.mkdir(parents=True)
    _write_jpeg(image_root / "00000010.jpg", size=(64, 48))
    nested = tmp_path / "P1E_S1_C1.tar.xz"
    with tarfile.open(nested, "w:xz") as archive:
        archive.add(image_root, arcname="P1E_S1_C1")
    outer = choke_root / "P1E_S1.tar.xz"
    with tarfile.open(outer, "w:xz") as archive:
        archive.add(nested, arcname=nested.name)
    xml = tmp_path / "P1E_S1_C1.xml"
    xml.write_bytes(
        b'<dataset><frame number="10"><person id="7"><leftEye x="10" y="20"/>'
        b'<rightEye x="14" y="22"/></person></frame></dataset>'
    )
    groundtruth = choke_root / "groundtruth.tar.xz"
    with tarfile.open(groundtruth, "w:xz") as archive:
        archive.add(xml, arcname=f"groundtruth/{xml.name}")
    manifest = {
        "schema_version": "fieldpack.formal-split.v1",
        "status": "frozen",
        "scene_count": 1,
        "segment_count_per_scene": 1,
        "frames_per_segment": 1,
        "scenes": [{
            "scene_id": "P1",
            "dataset": "ChokePoint",
            "source": outer.name,
            "media": nested.name,
            "annotations": [xml.name],
            "media_sha256": _sha256(nested),
            "annotation_sha256": [_sha256(xml)],
            "frame_count": 1,
            "timeline_frame_count": 20,
            "media_member_count": 1,
            "media_timeline": "sparse_jpeg_names_on_original_frame_axis",
            "member_name_template": "P1E_S1_C1/{frame_index:08d}.jpg",
            "fps": 30.0,
            "segments": [{"id": "H1", "start": 10, "end": 10}],
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare(
        manifest_path,
        virat_media_roots={},
        virat_annotations_root=tmp_path,
        chokepoint_root=choke_root,
        output=output,
    )
    truth = json.loads((output / "scenes" / "P1" / "truth.json").read_text(encoding="utf-8"))
    frame = truth["frames"][0]
    assert frame["truth_count"] == 1
    assert frame["truth_boxes"] == []
    assert frame["truth_points"] == [{"person_id": "7", "x": 12.0, "y": 21.0}]
    prepared = json.loads((output / "prepared-manifest.json").read_text(encoding="utf-8"))
    assert prepared["scenes"][0]["metric_family"] == "count_only_eye_point_focus"
    assert prepared["scenes"][0]["width"] == 64
    assert prepared["scenes"][0]["height"] == 48
