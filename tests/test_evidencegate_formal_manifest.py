from pathlib import Path

from evidencegate.formal_manifest import compile_formal_manifest, validate_formal_manifest


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g0/g0-execution-scope-v2.json"
CONFIG = ROOT / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g2/g2-formal-execution-config-v8.json"


def test_formal_manifest_preserves_unit_ids_and_resolves_plugin_conditions():
    manifest = compile_formal_manifest(SCOPE, CONFIG)
    validate_formal_manifest(manifest)
    assert len(manifest["units"]) == 986
    assert len({unit["unit_id"] for unit in manifest["units"]}) == 986
    assert all(unit["resolved_condition"] != "query_aware_crop50" for unit in manifest["units"])
    assert all(unit["resolved_condition"] not in {"native", "reduced", "jpeg"} for unit in manifest["units"] if unit["family"] == "deployment" and unit["modality"] == "video")
    resolved = {(unit["modality"], unit["condition"], unit["resolved_condition"]) for unit in manifest["units"]}
    assert ("video", "native", "dense") in resolved
    assert ("video", "reduced", "codec_guided50") in resolved
    assert ("video", "jpeg", "bitrate_only") in resolved
    assert ("image", "reduced", "center_crop50") in resolved
