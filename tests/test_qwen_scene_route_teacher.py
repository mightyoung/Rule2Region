import json
from pathlib import Path

import pytest
from PIL import Image

from research_tools.qwen_scene_route_teacher import load_profile, make_contact_sheet, parse_route_output
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS


def test_parse_route_output_accepts_only_bounded_canonical_contract() -> None:
    parsed = parse_route_output(json.dumps({"complex_tiles": ["t0", "t2"], "reasons": {"t0": ["far_small_person_zone"], "t2": ["occlusion_zone"]}}))
    assert parsed["complex_tiles"] == ["t0", "t2"]
    with pytest.raises(ValueError):
        parse_route_output("```json\n{}\n```")
    with pytest.raises(ValueError):
        parse_route_output(json.dumps({"complex_tiles": ["t4"], "reasons": {"t4": ["far_small_person_zone"]}}))
    with pytest.raises(ValueError):
        parse_route_output(json.dumps({"complex_tiles": ["t1", "t0"], "reasons": {"t0": ["occlusion_zone"], "t1": ["occlusion_zone"]}}))
    with pytest.raises(ValueError):
        parse_route_output(json.dumps({"complex_tiles": ["t1"], "reasons": {"t1": ["no_special_risk"]}}))


def test_contact_sheet_is_deterministic_shape_and_hashes_inputs(tmp_path: Path) -> None:
    sources = []
    for index in range(4):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (80, 40), (index * 20, 0, 0)).save(path)
        sources.append(path)
    output = tmp_path / "sheet.jpg"
    receipt = make_contact_sheet(sources, output, panel_size=(100, 50))
    with Image.open(output) as image:
        assert image.size == (200, 100)
    assert len(receipt["history_image_sha256"]) == 4
    assert len(receipt["contact_sheet_sha256"]) == 64


def test_load_profile_fails_closed_on_missing_camera(tmp_path: Path) -> None:
    profile = {
        "schema_version": "human-count.qwen-scene-route-profile.v1", "status": "passed",
        "history_gt_used": False, "evaluation_frames_used": False,
        "sealed_camera_used": False, "online_qwen_called": False,
        "cameras": {camera: {"complex_tiles": [], "reasons": {}} for camera in EXPECTED_CAMERAS},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert tuple(load_profile(path)["cameras"]) == EXPECTED_CAMERAS
    profile["cameras"].pop(EXPECTED_CAMERAS[0])
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        load_profile(path)
