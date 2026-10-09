from research_tools.sceneconfig_e4_m0_deployability import evaluate_gate, select_m0_configs


def test_m0_selection_uses_proxy_utility_only() -> None:
    proxy = {
        "panel": [{"config_id": "C00", "tiles": []}, {"config_id": "C01", "tiles": ["t0"]}],
        "per_camera": {
            camera: [{"config_id": "C00", "proxy_utility": 0.0}, {"config_id": "C01", "proxy_utility": 1.0}]
            for camera in ("VIRAT_S_010004", "VIRAT_S_010111", "VIRAT_S_010112", "VIRAT_S_010204")
        },
    }
    assert all(row["config_id"] == "C01" for row in select_m0_configs(proxy).values())


def test_e4_gate_is_per_camera_for_latency_and_macro_for_precision() -> None:
    rows = {
        "a": {"latency_ratio_p95": 2.0, "relative_count_mae_improvement": .2, "precision_change": -.04},
        "b": {"latency_ratio_p95": 3.1, "relative_count_mae_improvement": .2, "precision_change": -.04},
    }
    gate = evaluate_gate(rows, oom=False)
    assert gate["macro_precision_change_ge_minus_0_05"] is True
    assert gate["all_camera_p95_latency_ratio_le_3"] is False
    assert gate["passed"] is False
