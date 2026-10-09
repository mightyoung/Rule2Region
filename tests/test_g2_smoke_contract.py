from pathlib import Path

from research_tools.g2_smoke import FAULT_EXPECTED, _validate_execution_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g1/g1-execution-config-v6.json"


def test_g2_runner_accepts_only_final_v6_and_closed_smoke_identities():
    config, file_hash = _validate_execution_config(CONFIG)
    assert len(file_hash) == 64
    assert len(config["smoke_units"]) == 36
    assert len({unit["unit_id"] for unit in config["smoke_units"]}) == 36
    conditions = {unit["condition"] for unit in config["smoke_units"]}
    assert "center_crop50" in conditions
    assert "query_aware_crop50" not in conditions
    assert config["prompts"]["video"]["parser"] == "exact_json_single_count_0_99"
    assert config["runtime"]["allow_model_download_during_g2"] is False
    fault_modes = {unit["fault_mode"] for unit in config["smoke_units"] if unit["fault_mode"]}
    assert fault_modes == set(FAULT_EXPECTED)
