import json
from pathlib import Path

import pytest
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.execution_policy.browse import FileBrowser, PathEscapesContext
from research_tools.execution_policy.card import FRAME_HEIGHT, FRAME_WIDTH, parse_policy_card
from research_tools.execution_policy.context import write_agent_context
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.execution_policy.prompts import VISUAL_SYSTEM_PROMPT
from research_tools.execution_policy.run_e4 import (
    group_clips_by_camera,
    next_turn_prompt,
    other_history_fewshot,
    read_observer_image,
    require_compiler_frame,
    subagent_prompt,
)


def test_subagent_prompt_gives_rules_and_dir_not_observer_boxes(tmp_path: Path) -> None:
    path = tmp_path / "000250.jpg"
    Image.new("RGB", (int(FRAME_WIDTH), int(FRAME_HEIGHT)), color=(12, 34, 56)).save(path, format="JPEG")
    info = read_observer_image(path)
    require_compiler_frame(info)
    context = tmp_path / "context"
    context.mkdir()
    text = subagent_prompt(
        "VIRAT_S_010004",
        250,
        info,
        context,
        observer_sequence="VIRAT_S_010004_01",
        sequences=["VIRAT_S_010004_01", "VIRAT_S_010004_07"],
    )
    assert "计数调度规则" in text
    assert "decoded pixels 1280x720" in text
    assert str(context) in text
    assert "list_skills" in text
    assert "load_skill" in text
    assert "Observer-frame detector counts" in text
    assert "One card for this physical camera view" in text
    assert "must not overlap" in text
    assert "merge them into one union box" in text
    assert "VIRAT_S_010004_01" in text
    assert "VIRAT_S_010004_07" in text
    assert "Few-shot" not in text
    assert "[0,360,1280,720]" not in text
    assert "yolo11s=[[50,410,80,520]]" not in text
    assert "yolo11s=3" not in text
    assert "yolo26=5" not in text
    lot = subagent_prompt(
        "VIRAT_S_010111",
        379,
        info,
        context,
        observer_sequence="VIRAT_S_010111_07",
        sequences=["VIRAT_S_010111_02", "VIRAT_S_010112_00"],
        camera_ids=["VIRAT_S_010111", "VIRAT_S_010112"],
    )
    assert "same parking-lot camera" in lot
    assert "VIRAT_S_010111" in lot
    assert "VIRAT_S_010112" in lot


def test_system_prompt_is_thin_and_defers_file_use_to_skills() -> None:
    assert "list_skills" in VISUAL_SYSTEM_PROMPT
    assert "load_skill" in VISUAL_SYSTEM_PROMPT
    assert "file-browse" in VISUAL_SYSTEM_PROMPT
    assert "count-schedule" in VISUAL_SYSTEM_PROMPT
    assert "history_boxes/" not in VISUAL_SYSTEM_PROMPT
    assert "count_schedule_skill.md" not in VISUAL_SYSTEM_PROMPT
    assert "A tool JSON is executed by the harness" not in VISUAL_SYSTEM_PROMPT


def test_fewshot_comes_from_other_history_not_observer() -> None:
    ledgers = {
        1: FrameLedger(1, (Prediction(1, Box(10, 10, 20, 20), 0.9, "yolo11s"),), ()),
        2: FrameLedger(
            2,
            (Prediction(2, Box(30, 30, 40, 40), 0.99, "yolo11s"),),
            (Prediction(2, Box(5, 5, 8, 8), 0.8, "yolo26"),),
        ),
    }
    fewshot = other_history_fewshot([1, 2], 2, ledgers)
    assert fewshot["frame"] == 1
    assert fewshot["yolo11s"] == [[10, 10, 20, 20]]
    assert [30, 30, 40, 40] not in fewshot["yolo11s"]


def test_context_writes_other_history_boxes_and_withholds_observer(tmp_path: Path) -> None:
    ledgers = {
        1: FrameLedger(
            1,
            (Prediction(1, Box(10, 10, 20, 20), 0.9, "yolo11s"),),
            (
                Prediction(1, Box(10, 10, 20, 20), 0.85, "yolo26"),
                Prediction(1, Box(100, 10, 140, 50), 0.8, "yolo26"),
            ),
        ),
        2: FrameLedger(2, (Prediction(2, Box(30, 30, 40, 40), 0.8, "yolo11s"),), (Prediction(2, Box(5, 5, 8, 8), 0.7, "yolo26"),)),
        3: FrameLedger(3, (), (Prediction(3, Box(200, 10, 240, 50), 0.9, "yolo26"),)),
    }
    truth = [
        GroundTruth(1, 1, Box(10, 10, 20, 20)),
        GroundTruth(1, 2, Box(100, 10, 140, 50)),
        GroundTruth(2, 1, Box(30, 30, 40, 40)),
        GroundTruth(3, 1, Box(200, 10, 240, 50)),
    ]
    root = tmp_path / "ctx"
    meta = write_agent_context(
        root,
        camera_id="CAM",
        observer_sequence="SEQ",
        observer=2,
        sources=[
            {
                "sequence": "SEQ",
                "camera_id": "CAM",
                "history": [1, 2],
                "validation": [3],
                "ledgers": ledgers,
                "truth": truth,
            }
        ],
    )
    assert meta["observer_boxes_withheld"] is True
    assert (root / "history_boxes" / "SEQ__000001.yolo11s.json").is_file()
    digest = json.loads((root / "history_boxes" / "_digest.json").read_text(encoding="utf-8"))
    focus = json.loads((root / "history_boxes" / "_focus.json").read_text(encoding="utf-8"))
    assert digest["observer_frame_withheld"] == 2
    assert digest["streams_mixed"] is False
    assert [row["frame"] for row in digest["frames"]] == [1]
    assert focus["scope"] == "camera"
    assert focus["regions_must_not_overlap"] is True
    assert focus["overlap_policy"] == "merge"
    assert focus["evidence_box_count"] == 1
    assert "SEQ:2" not in focus["evidence_frames"]
    assert "SEQ:3" not in focus["evidence_frames"]
    assert len(focus["proposed_regions"]) == 1
    assert focus["proposed_regions"][0]["model"] == "yolo26"
    assert focus["proposed_regions"][0]["xyxy"][0] <= 100
    assert focus["proposed_regions"][0]["xyxy"][2] >= 140
    assert not (root / "history_boxes" / "SEQ__000002.yolo11s.json").exists()
    assert not (root / "history_boxes" / "SEQ__000003.yolo11s.json").exists()
    assert (root / "rules" / "counting_rule_card.txt").read_text(encoding="utf-8").startswith("计数调度规则")
    assert "merge them into one union box" in (root / "rules" / "menu.txt").read_text(encoding="utf-8")
    assert meta["skills"] == ["file-browse", "count-schedule"]
    assert (root / "skills" / "file-browse" / "SKILL.md").is_file()
    skill_text = (root / "skills" / "count-schedule" / "SKILL.md").read_text(encoding="utf-8")
    assert "physical camera view once" in skill_text
    assert "VIRAT_S_010111" in skill_text
    assert "must not overlap" in skill_text
    assert (root / "run_status" / "b11_history.json").is_file()
    browser = FileBrowser(root)
    blocked = browser.call("read_file", "history_boxes/SEQ__000001.yolo11s.json")
    assert blocked["ok"] is False
    assert blocked["error"] == "skill_not_loaded"
    listed_skills = browser.call("list_skills", "")
    assert {row["name"] for row in listed_skills["skills"]} == {"file-browse", "count-schedule"}
    loaded = browser.call("load_skill", "skills/file-browse/SKILL.md")
    assert loaded["ok"] is True
    assert loaded["skill"] == "file-browse"
    assert "list_dir" in loaded["text"]
    listed = browser.list_dir("history_boxes")
    assert listed["ok"] is True
    names = {row["name"] for row in listed["entries"]}
    assert names == {"_digest.json", "_focus.json"}
    assert listed["omitted_frame_files"] >= 1
    unseen = browser.call("read_file", "history_boxes/SEQ__000001.yolo11s.json")
    assert unseen["ok"] is False
    assert unseen["error"] == "path_not_in_listing"
    seen = browser.call("read_file", "history_boxes/_focus.json")
    assert seen["ok"] is True
    reread = browser.call("read_file", "history_boxes/_focus.json")
    assert reread["ok"] is False
    assert reread["error"] == "already_read"
    extra = browser.call("read_file", "history_boxes/_digest.json")
    assert extra["ok"] is False
    assert extra["error"] == "history_boxes_already_read"
    memory = browser.memory()
    assert "file-browse" in memory["loaded_skills"]
    assert "history_boxes/_focus.json" in memory["read_boxes"]
    assert "history_boxes/SEQ__000001.yolo11s.json" not in memory["listed_files"]
    follow = next_turn_prompt(
        "BASE",
        [{"action": "read_file", "path": "history_boxes/_focus.json", "ok": True}],
        seen,
        browser,
        "TAIL",
    )
    assert "history_boxes/_focus.json" in follow
    assert "proposed_regions" in follow
    assert "text_omitted" not in follow
    assert "SEQ__000001.yolo11s.json" not in follow
    reread = next_turn_prompt(
        "BASE",
        [
            {"action": "read_file", "path": "history_boxes/_focus.json", "ok": True},
            {"action": "read_file", "path": "history_boxes/_focus.json", "ok": True},
        ],
        seen,
        browser,
        "TAIL",
    )
    assert "text_omitted" in reread
    assert "proposed_regions" not in reread
    with pytest.raises(PathEscapesContext):
        browser.resolve("../secret.json")
    assert browser.missing_for_submit() == ["count-schedule"]
    browser.call("load_skill", "count-schedule")
    assert browser.missing_for_submit() == []


def test_context_aggregates_two_sequences_of_one_camera(tmp_path: Path) -> None:
    ledgers_a = {
        1: FrameLedger(
            1,
            (Prediction(1, Box(10, 10, 20, 20), 0.9, "yolo11s"),),
            (
                Prediction(1, Box(10, 10, 20, 20), 0.85, "yolo26"),
                Prediction(1, Box(100, 10, 140, 50), 0.8, "yolo26"),
            ),
        ),
        2: FrameLedger(2, (Prediction(2, Box(30, 30, 40, 40), 0.8, "yolo11s"),), (Prediction(2, Box(5, 5, 8, 8), 0.7, "yolo26"),)),
        3: FrameLedger(3, (), (Prediction(3, Box(200, 10, 240, 50), 0.9, "yolo26"),)),
    }
    ledgers_b = {
        1: FrameLedger(
            1,
            (),
            (
                Prediction(1, Box(110, 12, 150, 52), 0.82, "yolo26"),
                Prediction(1, Box(500, 10, 540, 50), 0.81, "yolo26"),
            ),
        ),
        2: FrameLedger(2, (), (Prediction(2, Box(500, 10, 540, 50), 0.8, "yolo26"),)),
        3: FrameLedger(3, (), ()),
    }
    truth_a = [
        GroundTruth(1, 1, Box(10, 10, 20, 20)),
        GroundTruth(1, 2, Box(100, 10, 140, 50)),
        GroundTruth(2, 1, Box(30, 30, 40, 40)),
        GroundTruth(3, 1, Box(200, 10, 240, 50)),
    ]
    truth_b = [
        GroundTruth(1, 1, Box(110, 12, 150, 52)),
        GroundTruth(1, 2, Box(500, 10, 540, 50)),
        GroundTruth(2, 1, Box(500, 10, 540, 50)),
    ]
    root = tmp_path / "cam"
    meta = write_agent_context(
        root,
        camera_id="CAM",
        observer_sequence="A",
        observer=2,
        sources=[
            {
                "sequence": "A",
                "camera_id": "CAM",
                "history": [1, 2],
                "validation": [3],
                "ledgers": ledgers_a,
                "truth": truth_a,
            },
            {
                "sequence": "B",
                "camera_id": "CAM",
                "history": [1, 2],
                "validation": [3],
                "ledgers": ledgers_b,
                "truth": truth_b,
            },
        ],
    )
    assert meta["sequences"] == ["A", "B"]
    assert (root / "history_boxes" / "A__000001.yolo26.json").is_file()
    assert (root / "history_boxes" / "B__000001.yolo26.json").is_file()
    assert (root / "history_boxes" / "B__000002.yolo26.json").is_file()
    assert not (root / "history_boxes" / "A__000002.yolo26.json").exists()
    assert not (root / "history_boxes" / "A__000003.yolo26.json").exists()
    assert not (root / "history_boxes" / "B__000003.yolo26.json").exists()
    focus = json.loads((root / "history_boxes" / "_focus.json").read_text(encoding="utf-8"))
    assert focus["camera_id"] == "CAM"
    assert set(focus["evidence_sequences"]) == {"A", "B"}
    assert "A:2" not in focus["evidence_frames"]
    assert "A:3" not in focus["evidence_frames"]
    assert "B:3" not in focus["evidence_frames"]
    ids = {row["region_id"] for row in focus["proposed_regions"]}
    assert ids <= {"cam26_1", "cam26_2"}
    assert len(focus["proposed_regions"]) == 2
    boxes = [row["xyxy"] for row in focus["proposed_regions"]]
    assert any(box[0] <= 100 and box[2] >= 150 for box in boxes)
    assert any(box[0] <= 500 and box[2] >= 540 for box in boxes)


def test_group_clips_by_camera() -> None:
    groups = group_clips_by_camera(
        {
            "VIRAT_S_010004_07": {"camera_id": "VIRAT_S_010004"},
            "VIRAT_S_010004_01": {"camera_id": "VIRAT_S_010004"},
            "VIRAT_S_010204_03": {"camera_id": "VIRAT_S_010204"},
            "VIRAT_S_010111_07": {"camera_id": "VIRAT_S_010111"},
            "VIRAT_S_010112_00": {"camera_id": "VIRAT_S_010112"},
            "VIRAT_S_010111_02": {"camera_id": "VIRAT_S_010111"},
            "VIRAT_S_010112_01": {"camera_id": "VIRAT_S_010112"},
        }
    )
    assert groups == {
        "VIRAT_S_010004": ["VIRAT_S_010004_01", "VIRAT_S_010004_07"],
        "VIRAT_S_010111": [
            "VIRAT_S_010111_02",
            "VIRAT_S_010111_07",
            "VIRAT_S_010112_00",
            "VIRAT_S_010112_01",
        ],
        "VIRAT_S_010204": ["VIRAT_S_010204_03"],
    }


def test_partial_size_prompt_cards_are_inside_1280x720() -> None:
    first = parse_policy_card(
        {
            "regions": [
                {"region_id": "H1", "xyxy": [0, 380, 1280, 720], "model": "yolo11s", "conf": 0.25, "role": "count"},
                {"region_id": "H2", "xyxy": [0, 0, 1280, 380], "model": "yolo26", "conf": 0.15, "role": "count"},
            ]
        }
    )
    assert first.regions[0].box.coords[-1] == 720


def test_require_compiler_frame_rejects_resized_file(tmp_path: Path) -> None:
    path = tmp_path / "small.jpg"
    Image.new("RGB", (640, 360), color=(1, 2, 3)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="observer_size_mismatch"):
        require_compiler_frame(read_observer_image(path))
