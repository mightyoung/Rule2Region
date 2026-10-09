import json

import pytest

from evidencegate.manifest import canonical_json, manifest_sha256
from evidencegate.query_provider import global_center_provider_region, validate_query_provider_resolution


def test_global_center_provider_is_dimensions_only_and_bounded():
    assert global_center_provider_region(12, 10) == (6, 5, 1, 1)
    assert global_center_provider_region(1, 1) == (0, 0, 1, 1)
    with pytest.raises(ValueError, match="dimensions"):
        global_center_provider_region(0, 10)


def test_query_provider_resolution_rejects_attention_or_runtime_selection(tmp_path):
    payload = {
        "schema_version": "evidencegate.g1.query-provider-resolution.v1",
        "decision": "PREDECLARED_FALLBACK",
        "selected_provider": "global_center_crop50",
        "reporting_label": "global_center_crop50_fallback_not_query_aware",
        "input_dependence": "image_dimensions_only",
        "extra_lmm_calls": 0,
        "formal_samples_may_select_provider": False,
        "ground_truth_used": False,
        "native_attention_verdict": "REJECTED_FOR_G2",
    }
    payload["resolution_sha256"] = manifest_sha256(payload)
    path = tmp_path / "query.json"
    path.write_bytes(canonical_json(payload))
    assert len(validate_query_provider_resolution(path)) == 64

    payload["formal_samples_may_select_provider"] = True
    payload["resolution_sha256"] = manifest_sha256({key: value for key, value in payload.items() if key != "resolution_sha256"})
    path.write_bytes(canonical_json(payload))
    with pytest.raises(ValueError, match="contract"):
        validate_query_provider_resolution(path)
