import json
from pathlib import Path

from research_tools.evaluate_qwen_text_role_matrix import evaluate


def _write_matrix(root: Path, *, contaminate_business: bool = False) -> None:
    rows = []
    role_ids = {
        "algorithm_diagnostician": ["detector_ledger_0", "track_ledger_0"],
        "business_expert": ["business_card_0"],
        "deployment_expert": ["resource_ledger_0"],
    }
    for role, ids in role_ids.items():
        for repeat in range(3):
            case_id = f"{role}_r{repeat}"
            rows.append({"case_id": case_id, "role": role, "repeat": repeat, "launcher_exit_code": 0})
            case_dir = root / case_id
            case_dir.mkdir(parents=True)
            findings = ["stable bounded finding"]
            if contaminate_business and role == "business_expert":
                findings = ["small_box_ratio is 0.42"]
            payload = {
                "role": role,
                "consumed_input_ids": ids,
                "findings": findings,
                "constraints": [],
                "uncertainties": [],
                "summary": "bounded",
            }
            (case_dir / "extracted-response.json").write_text(json.dumps(payload), encoding="utf-8")
            (case_dir / "validation.json").write_text(json.dumps({"schema_valid": True}), encoding="utf-8")
            (case_dir / "meta.json").write_text(json.dumps({"exit_code": 0, "wall_ms": 1}), encoding="utf-8")
            (case_dir / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    (root / "case-manifest.json").write_text(json.dumps({"cases": rows}), encoding="utf-8")


def test_matrix_passes_for_exact_isolated_repeats(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    result = evaluate(tmp_path)
    assert result["passed"] is True
    assert all(result["exact_repeat_match"].values())


def test_matrix_rejects_cross_role_marker(tmp_path: Path) -> None:
    _write_matrix(tmp_path, contaminate_business=True)
    result = evaluate(tmp_path)
    assert result["passed"] is False
    assert result["role_isolation_pass"] is False
