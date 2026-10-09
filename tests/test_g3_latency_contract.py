from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencegate.runner import AttemptArtifactStore
from evidencegate.terminals import TerminalState
from research_tools.g3_latency import (
    expand_latency_attempts,
    latency_summary,
    load_latency_attempts,
)


ROOT = Path(__file__).resolve().parents[1]
GOAL = (
    ROOT
    / ".omx"
    / "goals"
    / "autoresearch"
    / "execute-evidencegate-engineering-paper-plan-v2-f"
    / "artifacts"
)


def _real_attempts() -> list[dict[str, object]]:
    scope = json.loads(
        (GOAL / "g0" / "g0-execution-scope-v2.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (GOAL / "g2" / "g2-formal-execution-config-v8.json").read_text(encoding="utf-8")
    )
    return expand_latency_attempts(scope, config)


def test_real_latency_cells_expand_to_frozen_896_attempts() -> None:
    attempts = _real_attempts()
    assert len(attempts) == 896
    assert len({item["cell_id"] for item in attempts}) == 64
    for cell_id in {item["cell_id"] for item in attempts}:
        block = [item for item in attempts if item["cell_id"] == cell_id]
        assert [item["attempt_index"] for item in block] == list(range(14))
        assert [item["latency_kind"] for item in block].count("warmup") == 4
        assert [item["latency_kind"] for item in block].count("steady") == 10


def test_latency_summary_excludes_warmup_and_reports_cell_completeness() -> None:
    attempts = _real_attempts()[:14]
    grouped = {}
    for item in attempts:
        key = (item["unit_id"], item["attempt_index"])
        grouped[key] = [
            {
                "unit_id": item["unit_id"],
                "attempt_index": item["attempt_index"],
                "attempt_id": f"a-{item['attempt_index']}",
                "retry_index": 0,
                "terminal_state": TerminalState.SUCCESS.value,
                "result": {"model_path_seconds": float(item["attempt_index"])},
            }
        ]
    summary = latency_summary(attempts, grouped)
    assert summary["terminal_verdict"] == "PASS"
    assert summary["latency_terminal_coverage"] == 14
    assert summary["latency_cells_with_all_steady_successes"] == 1
    assert summary["cell_metrics"][0]["steady_successes"] == 10
    assert summary["cell_metrics"][0]["median_seconds"] == pytest.approx(8.5)


def test_latency_summary_fails_terminal_coverage_without_inventing_values() -> None:
    attempts = _real_attempts()[:14]
    summary = latency_summary(attempts, {})
    assert summary["terminal_verdict"] == "FAIL"
    assert summary["latency_terminal_coverage"] == 0
    assert summary["cell_metrics"][0]["median_seconds"] is None


def test_latency_attempt_resume_binds_frozen_attempt_index(tmp_path: Path) -> None:
    attempts = _real_attempts()[:1]
    item = attempts[0]
    store = AttemptArtifactStore(tmp_path)
    record = {
        **item,
        "attempt_id": f"g3l-{item['cell_id']}-a00-r0",
        "retry_index": 0,
        "terminal_state": TerminalState.SUCCESS.value,
        "config_file_sha256": "a" * 64,
        "formal_manifest_file_sha256": "b" * 64,
    }
    store.write(record, attempt_id=record["attempt_id"])
    grouped = load_latency_attempts(store, attempts, "a" * 64, "b" * 64)
    assert grouped[(item["unit_id"], 0)][0]["attempt_id"] == record["attempt_id"]
    with pytest.raises(ValueError, match="latency_attempt_frozen_identity_mismatch"):
        load_latency_attempts(store, attempts, "c" * 64, "b" * 64)
