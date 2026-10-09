import copy

import pytest

from research_tools.build_qwen_candidate_config_catalog import build_catalog, validate_catalog


def test_catalog_recovers_registered_c00_c05_operator_definitions() -> None:
    catalog = build_catalog()
    configs = {row["config_id"]: row for row in catalog["configs"]}
    assert configs["C00"]["operator"]["selected_tiles"] == []
    assert [row["tile_id"] for row in configs["C01"]["operator"]["selected_tiles"]] == ["t0"]
    assert [row["tile_id"] for row in configs["C04"]["operator"]["selected_tiles"]] == ["t3"]
    assert [row["tile_id"] for row in configs["C05"]["operator"]["selected_tiles"]] == ["t0", "t1"]
    assert configs["C05"]["resource_evidence"]["tile_cost_ratio_proxy"] == 0.5


def test_catalog_is_deterministic_and_does_not_invent_measured_resources() -> None:
    first = build_catalog()
    second = build_catalog()
    assert first == second
    assert all(row["resource_evidence"]["measured_p95_ms"] is None for row in first["configs"])
    assert all(row["resource_evidence"]["measured_peak_vram_mib"] is None for row in first["configs"])


def test_catalog_hash_tampering_is_rejected() -> None:
    catalog = build_catalog()
    tampered = copy.deepcopy(catalog)
    tampered["configs"][1]["operator"]["tile_confidence"] = 0.99
    with pytest.raises(ValueError, match="sha256"):
        validate_catalog(tampered)
