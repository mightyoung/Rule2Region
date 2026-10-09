import pytest

from research_tools.summarize_local_oracle_sweep import summarize_local_oracle_sweep


def _receipt(sequence, *, complete=True):
    sizes = [384, 512]
    thresholds = [.1, .5]
    fusions = ["nms", "wbf"]
    conditions = {
        "yolo11s_global": {"count_mae": 5, "ap50": .15, "recall": .2, "small_object_recall": .1, "model_input_tensor_ratio_to_small_global": 1, "model_compute_total_ms": 10}
    }
    for size in sizes:
        for conf in thresholds:
            for fusion in fusions:
                conditions[f"oracle_i{size}_c{int(conf*100):02d}_{fusion}"] = {
                    "count_mae": 4 if size == 512 else 4.8,
                    "ap50": .16,
                    "recall": .21,
                    "small_object_recall": .11,
                    "model_input_tensor_ratio_to_small_global": 1.1,
                    "model_compute_total_ms": 20,
                }
    if not complete:
        conditions.pop("oracle_i512_c50_wbf")
    return {"sequence_id": sequence, "roi_sizes": sizes, "confidence_thresholds": thresholds, "fusions": fusions, "budget": .2, "context_expansion": 0, "global_imgsz": 640, "frames": 20, "seed": 7, "conditions": conditions}


def test_local_oracle_summary_selects_winner_and_applies_gate():
    result = summarize_local_oracle_sweep(
        (_receipt("a"), _receipt("b")), expected_sequence_count=2,
        expected_roi_sizes=(384, 512), expected_thresholds=(.1, .5), expected_fusions=("nms", "wbf"),
    )
    assert result["winner"].startswith("oracle_i512")
    assert result["winner_mae_improvement_fraction"] == pytest.approx(.2)
    assert result["winner_ap50_delta"] == pytest.approx(.01)
    assert result["quality_gate"]["passed"] is True


def test_local_oracle_summary_rejects_incomplete_matrix():
    with pytest.raises(ValueError, match="matrix_incomplete"):
        summarize_local_oracle_sweep(
            (_receipt("a", complete=False),), expected_sequence_count=1,
            expected_roi_sizes=(384, 512), expected_thresholds=(.1, .5), expected_fusions=("nms", "wbf"),
        )
