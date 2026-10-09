import json

from PIL import Image

from research_tools.visual_input_health import (
    compile_structure_from_judgments,
    crop_detection,
    identity_matches,
    parse_identity,
    parse_mtmd_log,
    score_visual_health,
    select_box_crops,
)


SAMPLE_LOG = """
encoding mtmd batch, n_chunks = 1 (done = 1, total = 9)
mtmd batch encoding done in 331 ms
find_slot: non-consecutive token position 4 after 3 for sequence 0 with 512 new tokens
DEPRECATED: argument '--image' specified multiple times, use comma-separated values instead (only last value will be used)
"""


def test_log_parser_does_not_treat_nine_chunks_as_proof_of_eight_files() -> None:
    parsed = parse_mtmd_log(SAMPLE_LOG)
    assert parsed["encoder_ran"] is True
    assert parsed["max_total_chunks"] == 9
    assert parsed["token_position_warning"] is True
    score = score_visual_health(parsed, expected_images=8, identity_ok=None)
    assert score["healthy"] is False
    assert "chunk_count_compatible_with_n_images_or_single_image_tiles" in score["reasons"]


def test_identity_requires_first_and_last_markers_for_multi_image() -> None:
    parsed = parse_identity(json.dumps({"markers_seen": ["K7Q", "Z9Q"], "first_marker": "K7Q", "last_marker": "Z9Q"}))
    assert identity_matches(parsed, ["K7Q", "Z9Q"])
    wrong = parse_identity(json.dumps({"markers_seen": ["Z9Q"], "first_marker": "Z9Q", "last_marker": "Z9Q"}))
    assert identity_matches(wrong, ["K7Q", "Z9Q"]) is False


def test_crop_stays_inside_source_and_compile_uses_cell_votes() -> None:
    image = Image.new("RGB", (1280, 720), (10, 10, 10))
    crop = crop_detection(image, {"x1": 0, "y1": 0, "x2": 10, "y2": 10})
    assert crop.size[0] >= 32 and crop.size[1] >= 32
    frames = [
        {
            "frame": 10,
            "merged": [
                {"box": {"x1": 10, "y1": 10, "x2": 40, "y2": 40}, "confidence": 0.9, "source": "yolo11s_global"},
                {"box": {"x1": 700, "y1": 400, "x2": 740, "y2": 480}, "confidence": 0.8, "source": "yolo26_hr_tile:t3"},
            ],
        }
    ]
    crops = select_box_crops(frames, width=1280, height=720, k=2)
    assert {row["cell"] for row in crops} == {"t0", "t3"}
    structure = compile_structure_from_judgments(
        crops,
        [
            {"box_id": crops[0]["box_id"], "judgment": "PERSON", "cell_advice": "INCLUDE", "rationale": "a"},
            {"box_id": crops[1]["box_id"], "judgment": "FP", "cell_advice": "EXCLUDE", "rationale": "b"},
        ],
    )
    types = {row["cell"]: row["type"] for row in structure["cells"]}
    assert "INCLUDE" in types.values() or "EXPERT" in types.values()
    assert types.get("t3") == "EXCLUDE"
