import json
import hashlib
from pathlib import Path

from evidencegate.execution_config import compile_execution_config


ROOT = Path(__file__).resolve().parents[1]
G0_SCOPE = ROOT / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g0/g0-execution-scope-v2.json"


def test_real_g0_scope_compiles_deterministic_execution_identities():
    assert hashlib.sha256(G0_SCOPE.read_bytes()).hexdigest() == "5c246d3b348169130055059456170b00b4300e0c6d1ab451ec244f7fbc58441f"
    config = compile_execution_config(G0_SCOPE, codec_derivative_manifest_sha256="a" * 64, query_provider_resolution_sha256="d" * 64, runtime_environment_receipt_sha256="e" * 64, development_video_manifest_sha256="f" * 64, model_cache_index_sha256="1" * 64, g2_output_index_sha256="2" * 64, prompt_adjustment_receipt_sha256="3" * 64)
    repeated = compile_execution_config(G0_SCOPE, codec_derivative_manifest_sha256="a" * 64, query_provider_resolution_sha256="d" * 64, runtime_environment_receipt_sha256="e" * 64, development_video_manifest_sha256="f" * 64, model_cache_index_sha256="1" * 64, g2_output_index_sha256="2" * 64, prompt_adjustment_receipt_sha256="3" * 64)
    assert config.sha256 == repeated.sha256
    assert len(config.latency_cells) == 64
    assert len({cell.cell_id for cell in config.latency_cells}) == 64
    assert sum(cell.warmup_attempts + cell.steady_attempts for cell in config.latency_cells) == 896
    assert len(config.smoke_units) == 36
    assert len({unit.unit_id for unit in config.smoke_units}) == 36


def test_bf16_and_nf4_have_distinct_canonical_model_layer_identities():
    config = compile_execution_config(G0_SCOPE, codec_derivative_manifest_sha256="b" * 64, query_provider_resolution_sha256="d" * 64, runtime_environment_receipt_sha256="e" * 64, development_video_manifest_sha256="f" * 64, model_cache_index_sha256="1" * 64, g2_output_index_sha256="2" * 64, prompt_adjustment_receipt_sha256="3" * 64)
    layers = {layer.model_key: layer for layer in config.model_layers}
    assert layers["0.5b-bf16"].repository_id == layers["0.5b-nf4"].repository_id
    assert layers["0.5b-bf16"].model_layer_identity != layers["0.5b-nf4"].model_layer_identity
    assert layers["0.5b-nf4"].quantization_config["llm_int8_skip_modules"] == ["vision_tower", "multi_modal_projector", "lm_head"]
    assert layers["3b-nf4"].quantization_config["llm_int8_skip_modules"] == ["visual", "lm_head"]


def test_execution_config_serializes_without_nonfinite_or_implicit_fields():
    config = compile_execution_config(G0_SCOPE, codec_derivative_manifest_sha256="c" * 64, query_provider_resolution_sha256="d" * 64, runtime_environment_receipt_sha256="e" * 64, development_video_manifest_sha256="f" * 64, model_cache_index_sha256="1" * 64, g2_output_index_sha256="2" * 64, prompt_adjustment_receipt_sha256="3" * 64)
    payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
    assert payload["query_provider"]["extra_lmm_calls"] == 0
    assert payload["query_provider"]["mode"] == "global_center_crop50"
    assert payload["query_provider"]["reporting_label"].endswith("not_query_aware")
    assert payload["gpu_reservation"]["max_active_attempts_per_gpu"] == 1
    assert payload["codec"]["runtime_encoder_substitution_allowed"] is False
    assert payload["raw_store"]["output_root"] == "/srv/project-assets/evidencegate-v2"
    assert payload["runtime"]["model_cache_index_sha256"] == "1" * 64
    assert payload["runtime"]["g2_output_index_sha256"] == "2" * 64
    assert payload["runtime"]["prompt_adjustment_receipt_sha256"] == "3" * 64
    assert "0 to 99" not in payload["prompts"]["video"]["template"]
    assert "maximum number" in payload["prompts"]["video"]["template"]
    assert payload["runtime"]["formal_condition_resolution"]["video"] == {
        "native": "dense",
        "reduced": "codec_guided50",
        "jpeg": "bitrate_only",
    }
    assert payload["runtime"]["formal_condition_resolution"]["image"]["query_aware_crop50"] == "center_crop50"
    assert payload["runtime"]["input_transforms"]["image_jpeg_quality"] == 75
    assert payload["limits"]["total_call_hard_cap"] == 2107
    assert payload["limits"]["quality_success_minimum"] == 937
    assert payload["prompts"]["image"]["parser"] == "exact_case_insensitive_yes_or_no"
    assert payload["prompts"]["video"]["parser"] == "exact_json_single_count_0_99"
    assert payload["runtime"]["allow_model_download_during_g2"] is False
    assert payload["runtime"]["cuda_visible_device_count"] == 1
