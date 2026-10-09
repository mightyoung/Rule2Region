import json
from pathlib import Path

import pytest

from evidencegate.runner import (
    AttemptArtifactStore,
    EvidenceRunner,
    SingleScheduler,
    parse_count,
    parse_yes_no,
    write_content_addressed_jsonl,
    main,
)
from evidencegate.matrix import ExperimentUnit, MatrixPlan
from evidencegate.scope import ScopeVerdict
from evidencegate.terminals import TerminalState


def test_strict_parsers():
    assert parse_count("0") == 0 and parse_count(" 12 ") == 12
    assert parse_yes_no(" YES ") is True and parse_yes_no("no") is False
    for value in ("-1", "+1", "1.0", "01", "one"):
        with pytest.raises(ValueError):
            parse_count(value)
    with pytest.raises(ValueError):
        parse_yes_no("maybe")


def test_attempt_store_atomic_commit_and_recovery(tmp_path: Path):
    store = AttemptArtifactStore(tmp_path)
    artifact = store.write({"unit_id": "u1", "state": "success"}, attempt_id="a1")
    assert Path(artifact.path).exists()
    assert "a1" in store.wal_path.read_text(encoding="utf-8")
    store.write({"unit_id": "u1", "state": "success"}, attempt_id="a1")
    with pytest.raises(FileExistsError):
        store.write({"unit_id": "u1", "state": "crash"}, attempt_id="a1")
    orphan = store.attempts / "orphan.json.partial"
    orphan.write_text("not-json", encoding="utf-8")
    corrupted = store.attempts / "corrupted.json"
    corrupted.write_text("{broken", encoding="utf-8")
    report = store.recover()
    assert report["quarantined_partials"] == 1 and report["quarantined_final"] == 1


def test_content_addressed_snapshot_is_immutable(tmp_path: Path):
    path = write_content_addressed_jsonl(tmp_path, [{"b": 2, "a": 1}])
    assert path.name.endswith(".jsonl")
    assert write_content_addressed_jsonl(tmp_path, [{"a": 1, "b": 2}]) == path


def test_single_scheduler_lease_and_gpu_state(tmp_path: Path):
    scheduler = SingleScheduler(tmp_path)
    scheduler.acquire()
    scheduler.reserve_gpu()
    with pytest.raises(RuntimeError):
        scheduler.reserve_gpu()
    scheduler.hard_stop()
    with pytest.raises(RuntimeError):
        scheduler.reserve_gpu()
    scheduler.release()
    assert not (tmp_path / "scheduler.lease").exists()


def test_runner_closes_remaining_units_and_preserves_retry_unit_id(tmp_path: Path):
    units = tuple(ExperimentUnit(f"u{i}", "quality", "image", f"s{i}", "native") for i in range(3))
    plan = MatrixPlan(ScopeVerdict.IMAGE_ONLY, units, (), (), 0)
    runner = EvidenceRunner(tmp_path)
    terminals = runner.run(plan, max_calls=1)
    assert {item.state for item in terminals} == {TerminalState.SUCCESS, TerminalState.NOT_RUN_DUE_TO_HARD_STOP}
    retry = runner.record_attempt("u0", TerminalState.TIMEOUT, retry_index=1, attempt_id="retry-u0")
    assert retry.unit_id == "u0"


def test_phase_ledger_combined_flag(tmp_path: Path):
    runner = EvidenceRunner(tmp_path)
    record = runner.ledger.record("vision+prefill", combined=True, visual_tokens=2, prefill_tokens=3)
    assert record.combined is True


def test_recovery_commits_orphan_and_fails_closed_on_missing_artifact(tmp_path: Path):
    store = AttemptArtifactStore(tmp_path)
    artifact = store.write({"unit_id": "u"}, attempt_id="a")
    store.wal_path.write_bytes(b"")
    recovered = store.recover()
    assert recovered["orphan_final"] == 1
    assert not recovered["fail_closed"]
    Path(artifact.path).unlink()
    recovered = store.recover()
    assert recovered["missing_artifacts"] == ["a"]
    assert recovered["fail_closed"]


def test_gpu_reservation_uses_conservative_cap_and_crash_charge(tmp_path: Path):
    scheduler = SingleScheduler(tmp_path)
    scheduler.acquire()
    scheduler.reserve_gpu(10.0, 10.0, "a")
    scheduler.settle_gpu(None, crash_unknown=True)
    assert scheduler.consumed_seconds == 10.0
    with pytest.raises(RuntimeError, match="cap_would_be_exceeded"):
        scheduler.reserve_gpu(0.1, 10.0, "b")
    scheduler.release()


def test_hard_stop_closes_each_remaining_latency_attempt(tmp_path: Path):
    repeated = tuple(ExperimentUnit("cell", "latency", "image", "s", f"attempt-{i}", attempt_index=i) for i in range(4))
    plan = MatrixPlan(ScopeVerdict.BROAD, (), repeated, (), 0)
    runner = EvidenceRunner(tmp_path)
    terminals = runner.run(plan, max_calls=1)
    terminal = next(item for item in terminals if item.unit_id == "cell")
    assert len(terminal.attempt_ids) == 4
    assert terminal.state is TerminalState.NOT_RUN_DUE_TO_HARD_STOP


def test_cli_consumes_real_g0_scope_manifest(capsys, tmp_path: Path):
    scope = Path(__file__).resolve().parents[1] / ".omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g0/g0-execution-scope-v2.json"
    assert main(["--scope", str(scope), "--output-root", str(tmp_path), "--phase", "development"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "BROAD"
    assert result["total_calls"] == 2107
