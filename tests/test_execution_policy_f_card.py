from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.execution_policy.card import parse_policy_card
from research_tools.execution_policy.f_card import (
    assert_history_only,
    overlay_promoted_cards,
    propose_idle_cards,
    replay,
    split_history_sources,
)
from research_tools.execution_policy.ledger import FrameLedger
import pytest


def _source(sequence: str, history: list[int], validation: list[int], ledgers: dict[int, FrameLedger], truth) -> dict:
    return {
        "sequence": sequence,
        "camera_id": sequence.split("_01_")[0] if "_01_" in sequence else sequence[:14],
        "history": history,
        "validation": validation,
        "ledgers": ledgers,
        "truth": truth,
    }


def test_split_interleaves_frames_across_clips_and_drops_validation() -> None:
    a = _source("VIRAT_S_010004_01", [1, 2, 3, 99], [99], {}, ())
    b = _source("VIRAT_S_010004_07", [10, 11, 99], [99], {}, ())
    compile_rows, promote_rows = split_history_sources([a, b])
    compile_by_seq = {row["sequence"]: row["history"] for row in compile_rows}
    promote_by_seq = {row["sequence"]: row["history"] for row in promote_rows}
    assert "VIRAT_S_010004_01" in compile_by_seq and "VIRAT_S_010004_07" in compile_by_seq
    assert "VIRAT_S_010004_01" in promote_by_seq and "VIRAT_S_010004_07" in promote_by_seq
    assert compile_by_seq["VIRAT_S_010004_01"] == [1, 3]
    assert promote_by_seq["VIRAT_S_010004_01"] == [2]
    assert compile_by_seq["VIRAT_S_010004_07"] == [11]
    assert promote_by_seq["VIRAT_S_010004_07"] == [10]
    assert 99 not in compile_by_seq["VIRAT_S_010004_01"]
    assert 99 not in promote_by_seq.get("VIRAT_S_010004_07", [])
    single = _source("ONLY", [1, 2, 3, 4, 8], [8], {}, ())
    compile_one, promote_one = split_history_sources([single])
    assert compile_one[0]["history"] == [1, 3]
    assert promote_one[0]["history"] == [2, 4]


def test_history_overlapping_validation_is_illegal() -> None:
    with pytest.raises(ValueError, match="history_contains_validation"):
        assert_history_only([_source("CAM", [1, 2], [2], {}, ())])


def test_rebuild_foci_ignores_promote_and_validation_boxes() -> None:
    old = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 80, 80], "model": "yolo26", "conf": 0.25}]}
    )
    compile_ledger = {
        1: FrameLedger(
            1,
            (),
            (Prediction(1, Box(10, 10, 40, 70), 0.9, "yolo26"),),
        )
    }
    promote_ledger = {
        2: FrameLedger(
            2,
            (),
            (Prediction(2, Box(400, 400, 460, 500), 0.9, "yolo26"),),
        )
    }
    val_ledger = {
        9: FrameLedger(
            9,
            (),
            (Prediction(9, Box(800, 400, 860, 500), 0.9, "yolo26"),),
        )
    }
    compile = _source(
        "VIRAT_S_010004_01",
        [1],
        [9],
        {**compile_ledger, **val_ledger},
        (GroundTruth(1, 1, Box(10, 10, 40, 70)), GroundTruth(9, 9, Box(800, 400, 860, 500))),
    )
    promote = _source(
        "VIRAT_S_010004_07",
        [2],
        [9],
        {**promote_ledger, **val_ledger},
        (GroundTruth(2, 2, Box(400, 400, 460, 500)),),
    )
    names = dict(propose_idle_cards(old, [compile]))
    rebuilt = names["rebuild_foci"]
    xs = [row.box.x1 for row in rebuilt.regions]
    assert any(x < 100 for x in xs)
    assert all(x < 300 for x in xs)
    result = replay(view_sources={"VIRAT_S_010004": [compile, promote]}, cards={"VIRAT_S_010004": old})
    assert result["official_h1"] is False
    assert result["val_frames_used"] is False
    assert result["sealed_camera_used"] is False
    assert "9" not in "".join(result["per_view"]["VIRAT_S_010004"]["promote_frames"])
    assert "VIRAT_S_010113" not in str(result)


def test_holdout_gain_is_rejected_when_compile_history_gets_worse() -> None:
    old = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 200, 200], "model": "yolo26", "conf": 0.15}]}
    )
    compile = _source(
        "A_01",
        [1],
        [9],
        {1: FrameLedger(1, (), (Prediction(1, Box(10, 10, 40, 70), 0.9, "yolo26"),))},
        (GroundTruth(1, 1, Box(10, 10, 40, 70)),),
    )
    promote = _source(
        "A_07",
        [2],
        [9],
        {
            2: FrameLedger(
                2,
                (),
                (
                    Prediction(2, Box(10, 10, 40, 70), 0.9, "yolo26"),
                    Prediction(2, Box(50, 50, 80, 90), 0.2, "yolo26"),
                ),
            )
        },
        (),
    )
    result = replay(view_sources={"CAM": [compile, promote]}, cards={"CAM": old})
    row = result["per_view"]["CAM"]
    by_name = {item["name"]: item for item in row["tried"]}
    assert "a_full_11s" not in by_name
    assert "a_full_11s_035" not in by_name
    assert by_name["drop_26"]["reason"] == "worse_on_compile_history"
    assert row["promoted"] is True
    assert row["candidate"] in {"raise_26_conf", "rebuild_foci"}


def test_idle_replay_promotes_when_holdout_improves_and_compile_holds() -> None:
    old = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 200, 200], "model": "yolo26", "conf": 0.15}]}
    )
    empty_26 = {
        1: FrameLedger(1, (), (Prediction(1, Box(10, 10, 40, 70), 0.2, "yolo26"),)),
        2: FrameLedger(2, (), (Prediction(2, Box(10, 10, 40, 70), 0.2, "yolo26"),)),
    }
    compile = _source("A_01", [1], [9], empty_26, ())
    promote = _source("A_07", [2], [9], empty_26, ())
    result = replay(view_sources={"CAM": [compile, promote]}, cards={"CAM": old})
    row = result["per_view"]["CAM"]
    assert row["promoted"] is True
    assert row["candidate"] in {"drop_26", "raise_26_conf", "rebuild_foci"}
    names = [item["name"] for item in row["tried"]]
    assert "a_full_11s" not in names
    assert "a_full_11s_035" not in names


def test_sealed_camera_is_refused() -> None:
    old = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 80, 80], "model": "yolo11s", "conf": 0.25}]}
    )
    ledgers = {1: FrameLedger(1, (), ()), 2: FrameLedger(2, (), ())}
    compile = _source("VIRAT_S_010113_01", [1], [3], ledgers, ())
    promote = _source("VIRAT_S_010113_02", [2], [3], ledgers, ())
    with pytest.raises(ValueError, match="sealed_camera"):
        replay(view_sources={"VIRAT_S_010113": [compile, promote]}, cards={"VIRAT_S_010113": old})


def test_overlay_promoted_cards_only_replaces_winners() -> None:
    old = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 80, 80], "model": "yolo26", "conf": 0.25}]}
    )
    other = parse_policy_card(
        {"regions": [{"region_id": "near", "xyxy": [100, 100, 180, 180], "model": "yolo26", "conf": 0.25}]}
    )
    payload = {
        "per_view": {
            "CAM": {
                "promoted": True,
                "kept_card": {
                    "regions": [{"region_id": "full11s", "xyxy": [0, 0, 1280, 720], "model": "yolo11s", "conf": 0.35}]
                },
            },
            "OTHER": {"promoted": False, "kept_card": {"regions": []}},
        }
    }
    out = overlay_promoted_cards({"CAM": old, "OTHER": other}, payload)
    assert out["CAM"].regions[0].model == "yolo11s"
    assert out["OTHER"] is other


def test_runner_stays_on_idle_ledger_filter() -> None:
    text = open("research_tools/execution_policy/run_f_card.py", encoding="utf-8").read()
    assert "ultralytics" not in text
    assert "validation_frames" in text
    assert "compare_three" not in text
    assert "official_h1" in text
    assert "load_clips" in text
