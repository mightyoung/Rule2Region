import pytest

from research_tools.summarize_context_scan import summarize_context_scan


def _receipt(expansion, sequence, mae, crop):
    summary = {
        "count_mae": mae,
        "ap50": .2,
        "recall": .3,
        "small_object_recall": .1,
        "latency_p50_ms": 10,
        "latency_p95_ms": 20,
        "throughput_fps": 50,
        "model_input_tensor_ratio_to_small_global": 1.072,
        "budget_fraction": .2,
        "selected_frames": 4,
        "total_compute_ms": 100,
        "local_batch_total_ms": 20,
    }
    roi = {
        "core_area_ratio": .05,
        "crop_area_ratio": crop,
        "context_expansion": expansion,
    }
    return {
        "schema_version": "fieldpack.budgeted-cascade.v1",
        "context_expansion": expansion,
        "sequence_id": sequence,
        "frames": 20,
        "seed": 7,
        "budgets": [.2],
        "batch_size": 8,
        "global_imgsz": 640,
        "roi_imgsz": 384,
        "conditions": {"oracle_budget_20": summary},
        "receipts": {"oracle_budget_20": [{"selected_roi": roi}]},
    }


def test_context_scan_uses_equal_sequence_means_and_reports_crop_area():
    result = summarize_context_scan((
        _receipt(.0, "a", 4, .05),
        _receipt(.0, "b", 6, .05),
        _receipt(.3, "a", 3, .08),
        _receipt(.3, "b", 5, .09),
    ))
    assert result["weighting"] == "equal_sequence_means"
    assert result["expansions"]["0.00"]["conditions"]["oracle_budget_20"]["count_mae"] == 5
    assert result["expansions"]["0.30"]["conditions"]["oracle_budget_20"]["mean_crop_area_ratio"] == pytest.approx(.085)


def test_context_scan_rejects_condition_mismatch():
    left = _receipt(.15, "a", 4, .06)
    right = _receipt(.15, "b", 4, .06)
    right["conditions"]["extra"] = right["conditions"]["oracle_budget_20"]
    right["receipts"]["extra"] = []
    with pytest.raises(ValueError, match="condition_mismatch"):
        summarize_context_scan((left, right))


def test_context_scan_requires_paired_sequence_sets():
    with pytest.raises(ValueError, match="sequence_set_mismatch"):
        summarize_context_scan((
            _receipt(.0, "a", 4, .05),
            _receipt(.15, "b", 4, .06),
        ))


def test_context_scan_requires_complete_expected_matrix():
    receipts = (
        _receipt(.0, "a", 4, .05),
        _receipt(.15, "a", 4, .06),
    )
    with pytest.raises(ValueError, match="expansion_set_mismatch"):
        summarize_context_scan(receipts, expected_expansions=(0, .15, .30))
    with pytest.raises(ValueError, match="receipt_count_mismatch"):
        summarize_context_scan(receipts, expected_receipt_count=3)
    with pytest.raises(ValueError, match="sequence_count_mismatch"):
        summarize_context_scan(receipts, expected_sequence_count=6)
