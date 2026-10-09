import json

import pytest

from research_tools.cascade_experiment import Box
from research_tools.roi_structure_protocol import (
    apply_patch,
    assign_disposition,
    gate_validation,
    materialize_counts,
    parse_review,
    parse_structure,
    scan_visual_logs,
    temporal_protocol_split,
)


def test_temporal_split_is_contiguous_and_keeps_ten_validation_frames() -> None:
    split = temporal_protocol_split(range(100, 132))
    assert split["history"] == tuple(range(100, 116))
    assert split["validation"] == tuple(range(116, 126))
    assert split["confirm"] == tuple(range(126, 132))
    assert not set(split["history"]) & set(split["validation"])


def test_structure_parser_rejects_collage_style_and_over_expert() -> None:
    parsed = parse_structure(json.dumps({"abstain": False, "cells": [{"cell": "t3", "type": "EXPERT"}]}))
    assert parsed["cells"][0]["type"] == "EXPERT"
    with pytest.raises(ValueError):
        parse_structure(json.dumps({"abstain": True, "cells": [{"cell": "t0", "type": "INCLUDE"}]}))
    with pytest.raises(ValueError):
        parse_structure(
            json.dumps(
                {
                    "abstain": False,
                    "cells": [
                        {"cell": "t0", "type": "EXPERT"},
                        {"cell": "t1", "type": "EXPERT"},
                        {"cell": "t2", "type": "EXPERT"},
                    ],
                }
            )
        )


def test_disposition_forces_hold_and_allows_one_patch() -> None:
    review = parse_review(
        json.dumps({"level": "L0", "suspect_frame_ids": [1, 2, 3], "patch": None, "rationale": "x"})
    )
    hold = assign_disposition(review, history_frames=[1, 2, 3, 4, 5, 6, 7, 8])
    assert hold["level"] == "L2"
    patched = assign_disposition(
        parse_review(
            json.dumps(
                {
                    "level": "L3",
                    "suspect_frame_ids": [1],
                    "patch": {"cell": "t3", "type": "EXCLUDE"},
                    "rationale": "fp",
                }
            )
        ),
        history_frames=[1, 2, 3, 4],
    )
    assert patched["level"] == "L3"
    assert patched["patch"]["cell"] == "t3"
    leaked = assign_disposition(
        parse_review(json.dumps({"level": "L1", "suspect_frame_ids": [99], "patch": None, "rationale": "x"})),
        history_frames=[1, 2],
    )
    assert leaked["level"] == "L4"


def test_patch_cannot_create_include_exclude_or_third_expert() -> None:
    with pytest.raises(ValueError):
        apply_patch(({"cell": "t0", "type": "INCLUDE"},), {"cell": "t0", "type": "EXCLUDE"})
    with pytest.raises(ValueError):
        apply_patch(
            ({"cell": "t0", "type": "EXPERT"}, {"cell": "t1", "type": "EXPERT"}),
            {"cell": "t2", "type": "EXPERT"},
        )


def test_disposition_honors_explicit_escalate_and_rejects_patchless_l3() -> None:
    escalated = assign_disposition(
        parse_review(json.dumps({"level": "L4", "suspect_frame_ids": [], "patch": None, "rationale": "unsure"})),
        history_frames=[1, 2, 3, 4],
    )
    assert escalated["level"] == "L4"
    patchless = assign_disposition(
        parse_review(json.dumps({"level": "L3", "suspect_frame_ids": [1], "patch": None, "rationale": "no patch"})),
        history_frames=[1, 2, 3, 4],
    )
    assert patchless["level"] == "L4"


def test_visual_logs_record_token_position_warnings_without_ignoring_them(tmp_path) -> None:
    log = tmp_path / "VIRAT_S_010004.structure.log"
    log.write_text("decode: non-consecutive token position 12 vs 9\n", encoding="utf-8")
    (tmp_path / "VIRAT_S_010004.review.log").write_text("ok\n", encoding="utf-8")
    scanned = scan_visual_logs(tmp_path, ["VIRAT_S_010004"])
    assert scanned["visual_calls"] == 2
    assert scanned["token_position_warning_calls"] == 1
    assert scanned["token_position_warning_ignored"] is False


def test_materialize_counts_excludes_one_cell_and_can_add_expert_tiles() -> None:
    frame = {
        "frame": 10,
        "global": [{"box": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}, "confidence": 0.9, "source": "yolo11s_global"}],
        "tile_raw": [
            {"box": {"x1": 700, "y1": 400, "x2": 740, "y2": 480}, "confidence": 0.8, "source": "yolo26_hr_tile:t3"}
        ],
    }
    excluded = materialize_counts([frame], [{"cell": "t0", "type": "EXCLUDE"}], width=1280, height=720)
    assert excluded[10] == 0
    expert = materialize_counts([frame], [{"cell": "t3", "type": "EXPERT"}], width=1280, height=720)
    assert expert[10] == 2
    _ = Box


def test_validation_gate_uses_relative_mae_and_precision_floor() -> None:
    passed = gate_validation(mae=0.8, baseline_mae=1.0, precision_change=-0.01, tile_cost_ratio=0.25, oom=False)
    assert passed["passed"] is True
    failed = gate_validation(mae=0.99, baseline_mae=1.0, precision_change=0.0, tile_cost_ratio=0.25, oom=False)
    assert failed["passed"] is False
