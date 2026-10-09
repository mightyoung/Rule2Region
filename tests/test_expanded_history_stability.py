from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.expanded_history_stability import (
    FrameObservation,
    ExperimentInputError,
    _prepared_rows,
    focus_cell_ids,
    main,
    run_experiment,
)


def _frame(
    segment: str,
    frame: int,
    *,
    scene: str = "0000",
    dataset: str = "VIRAT",
    predictions: tuple[Prediction, ...] = (),
    truth_boxes: tuple[Box, ...] = (),
    truth_count: int | None = None,
    truth_points: tuple[tuple[float, float], ...] = (),
) -> FrameObservation:
    truths = tuple(GroundTruth(frame, index + 1, box) for index, box in enumerate(truth_boxes))
    return FrameObservation(
        scene_id=scene,
        dataset=dataset,
        segment_id=segment,
        frame=frame,
        width=400,
        height=400,
        predictions=predictions,
        truth=truths,
        truth_count=len(truths) if truth_count is None else truth_count,
        latency_ms=2.0,
        truth_points=truth_points,
    )


def _six_segments(*, future_score: float = 0.10, future_has_truth: bool = True) -> tuple[FrameObservation, ...]:
    rows = []
    box = Box(100, 0, 140, 80)
    for index, segment in enumerate(("H1", "H2", "H3", "H4", "F1", "F2"), 1):
        score = future_score if segment.startswith("F") else 0.10
        truth = () if segment.startswith("F") and not future_has_truth else (box,)
        rows.append(_frame(segment, index, predictions=(Prediction(index, box, score, "yolo11s"),), truth_boxes=truth))
    return tuple(rows)


def test_focus_grid_uses_half_open_boundaries_and_stable_ranking() -> None:
    # Given: misses on an exact cell boundary and two cells tied on count.
    left = _frame("H1", 1, truth_boxes=(Box(80, 10, 120, 50), Box(10, 10, 50, 50)))
    right = _frame("H1", 2, truth_boxes=(Box(80, 10, 120, 50), Box(180, 10, 220, 50)))

    # When: the fixed 4x4 miss grid is compiled.
    selected = focus_cell_ids((left, right))

    # Then: x=100 belongs to column 1 and ties use cell id order.
    assert selected == ("g01", "g00", "g02")


def test_future_labels_cannot_change_history_selection() -> None:
    # Given: identical history with opposite future outcomes.
    passing = _six_segments(future_score=0.10)
    failing = _six_segments(future_score=0.10, future_has_truth=False)

    # When: both experiments are evaluated.
    pass_result = run_experiment(passing)
    fail_result = run_experiment(failing)

    # Then: selection is frozen from history while only future verdict changes.
    for horizon in ("T1", "T2", "T4"):
        assert pass_result["scenes"]["0000"][horizon]["selected"] == fail_result["scenes"]["0000"][horizon]["selected"]
        assert pass_result["scenes"]["0000"][horizon]["focus_cell_ids"] == fail_result["scenes"]["0000"][horizon]["focus_cell_ids"]
    assert pass_result["scenes"]["0000"]["T4"]["selected_future_pass"] is True
    assert fail_result["scenes"]["0000"]["T4"]["selected_future_pass"] is False


def test_horizons_use_only_the_first_one_two_or_four_history_segments() -> None:
    # Given: the complete six-segment formal order.
    observations = _six_segments()

    # When: the three history horizons are run.
    result = run_experiment(observations)

    # Then: each horizon exposes its exact frozen prefix and all 15 candidates.
    scene = result["scenes"]["0000"]
    assert scene["T1"]["history_segment_ids"] == ["H1"]
    assert scene["T2"]["history_segment_ids"] == ["H1", "H2"]
    assert scene["T4"]["history_segment_ids"] == ["H1", "H2", "H3", "H4"]
    assert len(scene["T4"]["candidates"]) == 15
    assert {row["name"] for row in scene["T4"]["candidates"]} == {
        f"o{outside:02d}_f{focus:02d}"
        for outside in (15, 25, 35)
        for focus in (5, 10, 15, 25, 35)
    }


def test_chokepoint_uses_count_precision_and_low_confidence_focus() -> None:
    # Given: ChokePoint count truth without boxes and one low-confidence person proposal.
    rows = tuple(
        _frame(
            segment,
            index,
            scene="P1",
            dataset="ChokePoint",
            predictions=(Prediction(index, Box(210, 210, 250, 290), 0.10, "yolo11s"),),
            truth_count=1,
            truth_points=((230, 250),),
        )
        for index, segment in enumerate(("H1", "H2", "H3", "H4", "F1", "F2"), 1)
    )

    # When: the history-derived threshold experiment runs.
    result = run_experiment(rows)

    # Then: no eye-derived boxes are needed and the count metric contract is explicit.
    scene = result["scenes"]["P1"]
    assert scene["metric_type"] == "COUNT_MAE_AND_COUNT_PRECISION"
    assert scene["T1"]["focus_cell_ids"] == ["g22"]
    assert scene["T1"]["selected_future"]["F1"]["precision"] == 1.0


def test_zero_baseline_mae_rejects_count_undercount_with_unchanged_precision() -> None:
    # Given: a perfect count baseline where a higher threshold drops one of two people.
    rows = tuple(
        _frame(
            segment,
            index,
            scene="P1",
            dataset="ChokePoint",
            predictions=(
                Prediction(index, Box(10, 10, 40, 80), 0.30, "yolo11s"),
                Prediction(index, Box(110, 10, 140, 80), 0.20, "yolo11s"),
            ),
            truth_count=2,
        )
        for index, segment in enumerate(("H1", "H2", "H3", "H4", "F1", "F2"), 1)
    )

    # When: the 0.25 outside threshold is evaluated against the perfect baseline.
    candidate = next(row for row in run_experiment(rows)["scenes"]["P1"]["T1"]["candidates"] if row["name"] == "o25_f05")

    # Then: zero-baseline MAE is fail-closed even though count precision stays one.
    assert candidate["history"]["H1"]["precision_change_vs_b11"] == 0.0
    assert candidate["history"]["H1"]["zero_baseline_mae_noninferior"] is False
    assert candidate["all_history_pass"] is False


def test_summary_counts_false_activation_stable_pass_and_oracle_intersection() -> None:
    # Given: one stable VIRAT scene and one ChokePoint scene that fails in future.
    stable = _six_segments()
    unstable = tuple(
        _frame(
            row.segment_id,
            row.frame + 10,
            scene="P1",
            dataset="ChokePoint",
            predictions=(Prediction(row.frame + 10, Box(10, 10, 40, 80), 0.10, "yolo11s"),),
            truth_count=0 if row.segment_id.startswith("F") else 1,
            truth_points=() if row.segment_id.startswith("F") else ((25, 45),),
        )
        for row in _six_segments()
    )

    # When: scene and dataset summaries are aggregated.
    summary = run_experiment((*stable, *unstable))["summary"]["T4"]

    # Then: stable and false-activation outcomes are counted independently.
    assert summary["scene_count"] == 2
    assert summary["history_selected_scene_count"] == 2
    assert summary["stable_future_pass_scene_count"] == 1
    assert summary["false_activation_scene_count"] == 1
    assert summary["by_dataset"]["VIRAT"]["stable_future_pass_scene_count"] == 1
    assert summary["by_dataset"]["ChokePoint"]["false_activation_scene_count"] == 1


def test_cli_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    # Given: a prepared manifest and an existing output path.
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"scenes": []}), encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text("sealed", encoding="utf-8")

    # When / Then: immutable output protection fires before model loading.
    with pytest.raises(FileExistsError, match="refuse_existing_output"):
        main(["--prepared-manifest", str(manifest), "--prepared-root", str(tmp_path), "--yolo11s", "model.pt", "--output", str(output)])


def test_prepared_fixture_verifies_hashes_and_converts_xywh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given: one real prepared JPEG, aligned truth receipt, and a detector package fake.
    root = tmp_path / "prepared"
    image_path = root / "scenes" / "0000" / "frames" / "00000001.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), "white").save(image_path)
    truth_path = root / "scenes" / "0000" / "truth.json"
    truth = {
        "scene_id": "0000",
        "dataset": "VIRAT",
        "frames": [{
            "frame_index": 1,
            "segment_id": "H1",
            "role": "history",
            "truth_count": 1,
            "truth_boxes": [{"track_id": 9, "bbox_xywh": [1, 2, 2, 1]}],
            "truth_points": [{"track_id": 9, "x": 2, "y": 2.5}],
        }],
    }
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = root / "prepared-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "fieldpack.prepared-fixed-camera.v1",
        "scenes": [{
            "scene_id": "0000",
            "dataset": "VIRAT",
            "width": 4,
            "height": 3,
            "truth_sha256": digest(truth_path),
            "prepared_frames": [{
                "frame_index": 1,
                "segment_id": "H1",
                "role": "history",
                "path": "scenes/0000/frames/00000001.jpg",
                "sha256": digest(image_path),
            }],
        }],
    }), encoding="utf-8")
    calls = []

    class FakeYolo:
        def __init__(self, _path: str) -> None:
            pass

        def predict(self, **_arguments) -> list[SimpleNamespace]:
            calls.append(1)
            return [SimpleNamespace(boxes=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYolo))

    # When: the runner consumes the prepared adapter contract.
    rows = _prepared_rows(manifest, root, Path("yolo11s.pt"), "cpu")

    # Then: inference occurs once, xywh becomes xyxy, and later hash drift is rejected.
    assert calls == [1]
    assert rows[0].truth[0].box == Box(1, 2, 3, 3)
    assert rows[0].truth_points == ((2.0, 2.5),)
    image_path.write_bytes(b"tampered")
    with pytest.raises(ExperimentInputError, match="prepared_frame_hash_mismatch"):
        _prepared_rows(manifest, root, Path("yolo11s.pt"), "cpu")
