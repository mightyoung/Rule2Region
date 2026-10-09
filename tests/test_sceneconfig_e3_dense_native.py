import json
from pathlib import Path

from research_tools.sceneconfig_e3_qwen_validation import prepare_dense_native_prompts


def _proxy(tmp_path: Path) -> Path:
    cards = [
        {
            "config_id": f"C{index:02d}",
            "tiles": ["t0"] if index else [],
            "tile_confidence": 0.25,
        }
        for index in range(24)
    ]
    payload = {
        "holdout_opened": False,
        "evaluation_gt_used_for_proxy": False,
        "panel": cards,
    }
    path = tmp_path / "proxy-ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dense_native_prompt_forbids_contact_sheet_and_lists_unstitched_frames(tmp_path: Path) -> None:
    selection = {
        "schema_version": "sceneconfig.dense-native-frame-selection.v1",
        "collage_used": False,
        "cameras": {
            "VIRAT_S_010004": {
                "observer_config_id": "C15",
                "frames": [
                    {"frame": 246, "merged_count": 12, "overlay_name": "VIRAT_S_010004/000246.overlay.jpg"},
                    {"frame": 249, "merged_count": 11, "overlay_name": "VIRAT_S_010004/000249.overlay.jpg"},
                ],
            },
            "VIRAT_S_010111": {
                "observer_config_id": "C16",
                "frames": [
                    {"frame": 1326, "merged_count": 1, "overlay_name": "VIRAT_S_010111/001326.overlay.jpg"},
                    {"frame": 1329, "merged_count": 1, "overlay_name": "VIRAT_S_010111/001329.overlay.jpg"},
                ],
            },
            "VIRAT_S_010112": {
                "observer_config_id": "C16",
                "frames": [
                    {"frame": 875, "merged_count": 1, "overlay_name": "VIRAT_S_010112/000875.overlay.jpg"},
                    {"frame": 878, "merged_count": 1, "overlay_name": "VIRAT_S_010112/000878.overlay.jpg"},
                ],
            },
            "VIRAT_S_010204": {
                "observer_config_id": "C04",
                "frames": [
                    {"frame": 196, "merged_count": 4, "overlay_name": "VIRAT_S_010204/000196.overlay.jpg"},
                    {"frame": 199, "merged_count": 4, "overlay_name": "VIRAT_S_010204/000199.overlay.jpg"},
                ],
            },
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    output_dir = tmp_path / "prompts"
    prepare_dense_native_prompts(_proxy(tmp_path), selection_path, output_dir)
    prompt = (output_dir / "VIRAT_S_010204.round1.txt").read_text(encoding="utf-8")
    assert "The attached contact sheet" not in prompt
    assert "拼图" not in prompt
    assert "not a contact sheet or collage" in prompt
    assert "unstitched" in prompt
    assert "native 1280x720" in prompt or "native frame" in prompt
    assert "frame 196" in prompt and "frame 199" in prompt
    assert "Image 1" in prompt and "Image 2" in prompt
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["collage_used"] is False
    assert manifest["selection_rule"] == "yolo11s_yolo26_merged_box_count_desc"
    assert manifest["schema_version"] == "sceneconfig.e3-prompt-manifest.v2"
