from __future__ import annotations

from copy import deepcopy

from research_tools.model_compatibility.harness import (
    FIXED_MODELS,
    validate_matrix,
    validate_receipt,
)


def _passing_receipt(family: str = "llava_onevision_05b") -> dict:
    spec = FIXED_MODELS[family]
    return {
        "schema_version": "codecchange.model-compatibility.v1",
        "status": "passed",
        "family": family,
        "repo_id": spec.repo_id,
        "revision_requested": spec.revision,
        "revision_loaded": spec.revision,
        "precision": "bf16",
        "input_manifest": [{"sha256": "a" * 64}],
        "environment": {"cuda_available": True},
        "weight_manifest": [{"sha256": "b" * 64}],
        "processor": {"input_keys": ["pixel_values_videos"]},
        "generation": {
            "parser_valid": True,
            "parsed_count": 1,
            "kv_cache_dtypes": ["torch.bfloat16"],
        },
        "dtype_inventory": {
            "parameter_numel_by_dtype": {
                "vision": {"torch.bfloat16": 1},
                "projector": {"torch.bfloat16": 1},
                "language": {"torch.bfloat16": 1},
            }
        },
        "peak_vram_bytes": 1,
        "checks": {"fixed_revision": True, "parser_valid": True},
    }


def test_fixed_revisions_are_full_commit_hashes() -> None:
    assert set(FIXED_MODELS) == {"llava_onevision_05b", "qwen25_vl_3b"}
    for spec in FIXED_MODELS.values():
        assert len(spec.revision) == 40
        int(spec.revision, 16)


def test_valid_receipt_passes() -> None:
    assert validate_receipt(_passing_receipt()) == []


def test_model_substitution_fails() -> None:
    receipt = _passing_receipt()
    receipt["repo_id"] = "substituted/model"
    assert "repo_id does not match frozen contract" in validate_receipt(receipt)


def test_revision_drift_fails() -> None:
    receipt = _passing_receipt()
    receipt["revision_loaded"] = "0" * 40
    assert "loaded revision does not match requested frozen revision" in validate_receipt(
        receipt
    )


def test_loader_only_receipt_fails() -> None:
    receipt = _passing_receipt()
    receipt["generation"]["parser_valid"] = False
    assert "generation parser did not pass" in validate_receipt(receipt)


def test_missing_vram_fails() -> None:
    receipt = _passing_receipt()
    receipt["peak_vram_bytes"] = 0
    assert "peak VRAM was not observed" in validate_receipt(receipt)


def test_failed_check_fails() -> None:
    receipt = deepcopy(_passing_receipt())
    receipt["checks"]["language_weight_nf4_only"] = False
    assert "one or more compatibility checks failed" in validate_receipt(receipt)


def test_full_two_model_two_precision_matrix_passes() -> None:
    receipts = []
    for family in FIXED_MODELS:
        for precision in ("bf16", "nf4"):
            receipt = _passing_receipt(family)
            receipt["precision"] = precision
            receipts.append(receipt)
    assert validate_matrix(receipts) == []


def test_missing_matrix_cell_fails() -> None:
    receipts = []
    for family in FIXED_MODELS:
        receipt = _passing_receipt(family)
        receipt["precision"] = "bf16"
        receipts.append(receipt)
    assert any("missing fixed compatibility cells" in item for item in validate_matrix(receipts))
