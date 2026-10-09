from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .ledger import ResourceRecord, StageResourceLedger
from .manifest import canonical_json, manifest_sha256
from .matrix import ExperimentUnit, MatrixPlan, compile_matrix
from .scope import ScopeVerdict
from .scope_manifest import ScopeManifest, VideoSample
from .terminals import AttemptRecord, TerminalRecord, TerminalState, aggregate_terminals, new_attempt_id


def parse_count(value: str) -> int:
    """Parse only a non-negative decimal count; no fallback/default coercion."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", value.strip()):
        raise ValueError("count_must_be_nonnegative_decimal_integer")
    return int(value)


def parse_yes_no(value: str) -> bool:
    if not isinstance(value, str) or value.strip().lower() not in {"yes", "no"}:
        raise ValueError("value_must_be_exactly_yes_or_no")
    return value.strip().lower() == "yes"


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArtifactConflictError(FileExistsError):
    pass


@dataclass(frozen=True)
class AttemptArtifact:
    attempt_id: str
    unit_id: str
    path: str
    sha256: str


class AttemptArtifactStore:
    """Crash-safe attempt JSON store with a fsynced commit WAL."""

    def __init__(self, output_root: str | os.PathLike[str]) -> None:
        self.root = Path(output_root)
        self.attempts = self.root / "attempts"
        self.wal_path = self.root / "commit.wal.jsonl"
        self.corrupt = self.root / "recovery-corrupt"
        self.attempts.mkdir(parents=True, exist_ok=True)

    def _path(self, attempt_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id):
            raise ValueError("attempt_id_contains_invalid_characters")
        return self.attempts / f"{attempt_id}.json"

    def write(self, record: Mapping[str, Any], *, attempt_id: str | None = None) -> AttemptArtifact:
        payload = dict(record)
        aid = str(attempt_id or payload.get("attempt_id") or new_attempt_id())
        uid = payload.get("unit_id")
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("attempt_unit_id_required")
        payload["attempt_id"] = aid
        blob = canonical_json(payload)
        digest = hashlib.sha256(blob).hexdigest()
        target = self._path(aid)
        if target.exists():
            if target.read_bytes() != blob:
                raise ArtifactConflictError(f"attempt_exists_with_different_content:{aid}")
            if aid not in self._committed_attempt_ids():
                self._append_wal({"event": "recovery_commit", "attempt_id": aid, "unit_id": uid, "sha256": digest, "path": str(target.name)})
            return AttemptArtifact(aid, uid, str(target), digest)
        partial = target.with_suffix(target.suffix + ".partial")
        with partial.open("wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(partial, target)
        except FileExistsError:
            if target.read_bytes() != blob:
                raise ArtifactConflictError(f"attempt_exists_with_different_content:{aid}")
        finally:
            partial.unlink(missing_ok=True)
        _fsync_dir(target.parent)
        self._append_wal({"event": "commit", "attempt_id": aid, "unit_id": uid, "sha256": digest, "path": str(target.name)})
        return AttemptArtifact(aid, uid, str(target), digest)

    def _append_wal(self, event: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = canonical_json(dict(event))
        with self.wal_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(self.root)

    def _committed_attempt_ids(self) -> set[str]:
        if not self.wal_path.exists():
            return set()
        committed: set[str] = set()
        for line in self.wal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") in {"commit", "recovery_commit"}:
                committed.add(str(event["attempt_id"]))
        return committed

    def recover(self) -> dict[str, Any]:
        """Quarantine malformed/orphan partials and report WAL inconsistencies."""
        self.corrupt.mkdir(parents=True, exist_ok=True)
        quarantined_partials = 0
        quarantined_final = 0
        for partial in self.attempts.glob("*.partial"):
            os.replace(partial, self.corrupt / partial.name)
            quarantined_partials += 1
        for final in self.attempts.glob("*.json"):
            try:
                decoded = json.loads(final.read_text(encoding="utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                os.replace(final, self.corrupt / final.name)
                quarantined_final += 1
        wal_events: list[dict[str, Any]] = []
        wal_errors = 0
        if self.wal_path.exists():
            for line in self.wal_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("event") not in {"commit", "recovery_commit"}:
                        raise ValueError
                    wal_events.append(event)
                except (ValueError, UnicodeError, json.JSONDecodeError):
                    wal_errors += 1
        committed = {str(item.get("attempt_id")) for item in wal_events}
        orphan_final = 0
        for final in self.attempts.glob("*.json"):
            if final.stem not in committed:
                decoded = json.loads(final.read_text(encoding="utf-8"))
                digest = hashlib.sha256(final.read_bytes()).hexdigest()
                self._append_wal({"event": "recovery_commit", "attempt_id": final.stem, "unit_id": decoded["unit_id"], "sha256": digest, "path": final.name})
                orphan_final += 1
        final_ids = {path.stem for path in self.attempts.glob("*.json")}
        missing_artifacts = sorted(committed - final_ids)
        return {"quarantined_partials": quarantined_partials, "quarantined_final": quarantined_final, "quarantined": quarantined_partials + quarantined_final, "wal_errors": wal_errors, "orphan_final": orphan_final, "committed_attempts": len(committed) + orphan_final, "missing_artifacts": missing_artifacts, "fail_closed": bool(wal_errors or missing_artifacts)}


def write_content_addressed_jsonl(output_root: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]) -> Path:
    lines = b"".join(canonical_json(dict(record)) for record in records)
    digest = hashlib.sha256(lines).hexdigest()
    directory = Path(output_root) / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.jsonl"
    if target.exists():
        if target.read_bytes() != lines:
            raise ArtifactConflictError(f"snapshot_exists_with_different_content:{digest}")
        return target
    partial = target.with_suffix(".jsonl.partial")
    with partial.open("wb") as stream:
        stream.write(lines)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(partial, target)
    except FileExistsError:
        if target.read_bytes() != lines:
            raise ArtifactConflictError(f"snapshot_exists_with_different_content:{digest}")
    finally:
        partial.unlink(missing_ok=True)
    _fsync_dir(directory)
    return target


class LeaseState(str, Enum):
    FREE = "free"
    HELD = "held"
    RELEASED = "released"


@dataclass(frozen=True)
class SchedulerLease:
    lease_id: str
    owner: str
    state: LeaseState
    path: str


class SingleScheduler:
    def __init__(self, output_root: str | os.PathLike[str]) -> None:
        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease_path = self.root / "scheduler.lease"
        self.gpu_budget_path = self.root / "gpu-budget.json"
        self.lease: SchedulerLease | None = None
        self.gpu_reserved = False
        self.reservation_id: str | None = None
        self.reserved_seconds = 0.0
        self.consumed_seconds = 0.0
        if self.gpu_budget_path.exists():
            state = json.loads(self.gpu_budget_path.read_text(encoding="utf-8"))
            self.consumed_seconds = float(state.get("consumed_seconds", 0.0))
            if state.get("reservation_id"):
                raise RuntimeError("unreconciled_gpu_reservation_requires_audit")
        self.hard_stopped = False

    def acquire(self, owner: str = "evidencegate") -> SchedulerLease:
        lease_id = uuid.uuid4().hex
        payload = canonical_json({"lease_id": lease_id, "owner": owner, "host": socket.gethostname(), "created_at": time.time()})
        try:
            descriptor = os.open(str(self.lease_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("scheduler_lease_already_held") from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_dir(self.root)
        self.lease = SchedulerLease(lease_id, owner, LeaseState.HELD, str(self.lease_path))
        return self.lease

    def _write_gpu_budget(self) -> None:
        payload = canonical_json({"consumed_seconds": self.consumed_seconds, "reservation_id": self.reservation_id, "reserved_seconds": self.reserved_seconds})
        temporary = self.gpu_budget_path.with_suffix(".json.partial")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.gpu_budget_path)
        _fsync_dir(self.root)

    def reserve_gpu(self, reserved_seconds: float = 0.0, cap_seconds: float = float("inf"), attempt_id: str | None = None) -> str:
        if self.lease is None or self.lease.state is not LeaseState.HELD:
            raise RuntimeError("scheduler_lease_required")
        if self.hard_stopped or self.gpu_reserved:
            raise RuntimeError("gpu_reservation_unavailable")
        if reserved_seconds < 0 or cap_seconds < 0:
            raise ValueError("gpu_seconds_must_be_nonnegative")
        if self.consumed_seconds + reserved_seconds > cap_seconds:
            raise RuntimeError("gpu_hour_cap_would_be_exceeded")
        self.gpu_reserved = True
        self.reservation_id = attempt_id or uuid.uuid4().hex
        self.reserved_seconds = float(reserved_seconds)
        self._write_gpu_budget()
        return self.reservation_id

    def release_gpu(self) -> None:
        self.gpu_reserved = False
        self.reservation_id = None
        self.reserved_seconds = 0.0
        self._write_gpu_budget()

    def settle_gpu(self, actual_seconds: float | None, *, crash_unknown: bool = False) -> None:
        if not self.gpu_reserved:
            raise RuntimeError("gpu_reservation_required")
        if actual_seconds is not None and actual_seconds < 0:
            raise ValueError("actual_gpu_seconds_must_be_nonnegative")
        charge = self.reserved_seconds if crash_unknown or actual_seconds is None else min(float(actual_seconds), self.reserved_seconds)
        self.consumed_seconds += charge
        self.release_gpu()

    def hard_stop(self) -> None:
        self.hard_stopped = True
        if self.gpu_reserved:
            self.settle_gpu(None, crash_unknown=True)

    def release(self) -> None:
        self.release_gpu()
        if self.lease is not None:
            self.lease = SchedulerLease(self.lease.lease_id, self.lease.owner, LeaseState.RELEASED, self.lease.path)
        try:
            self.lease_path.unlink()
        except FileNotFoundError:
            pass
        _fsync_dir(self.root)


class PhaseLedger:
    def __init__(self) -> None:
        self._ledger = StageResourceLedger()

    def record(self, stage: str, *, combined: bool = False, **resources: int | float) -> ResourceRecord:
        return self._ledger.record(stage, combined=combined, **resources)

    @property
    def records(self) -> tuple[ResourceRecord, ...]:
        return self._ledger.records

    def to_dict(self) -> dict[str, Any]:
        return self._ledger.to_dict()


class EvidenceRunner:
    def __init__(self, output_root: str | os.PathLike[str], adapter: Callable[[ExperimentUnit], Mapping[str, Any]] | None = None) -> None:
        self.root = Path(output_root)
        self.store = AttemptArtifactStore(self.root)
        self.scheduler = SingleScheduler(self.root)
        self.adapter = adapter or (lambda unit: {"unit_id": unit.unit_id, "state": TerminalState.SUCCESS.value})
        self.attempts: list[AttemptRecord] = []
        self.ledger = PhaseLedger()

    def record_attempt(self, unit_id: str, state: TerminalState | str, *, payload: Mapping[str, Any] | None = None, retry_index: int = 0, attempt_id: str | None = None) -> AttemptRecord:
        status = state if isinstance(state, TerminalState) else TerminalState(state)
        aid = attempt_id or new_attempt_id()
        record = dict(payload or {})
        record.update({"unit_id": unit_id, "attempt_id": aid, "state": status.value, "retry_index": retry_index})
        self.store.write(record, attempt_id=aid)
        attempt = AttemptRecord(unit_id, aid, status, retry_index)
        self.attempts.append(attempt)
        return attempt

    def close_remaining(self, units: Iterable[ExperimentUnit], *, skip_existing: bool = True) -> tuple[AttemptRecord, ...]:
        closed: list[AttemptRecord] = []
        for unit in units:
            if skip_existing and any(item.unit_id == unit.unit_id for item in self.attempts):
                continue
            closed.append(self.record_attempt(unit.unit_id, TerminalState.NOT_RUN_DUE_TO_HARD_STOP, payload={"attempt_index": unit.attempt_index}))
        return tuple(closed)

    def run(self, plan: MatrixPlan, *, phase: str = "development", max_calls: int | None = None, gpu_hour_cap: float = 24.0, reserved_seconds_per_attempt: float = 0.0) -> tuple[TerminalRecord, ...]:
        self.scheduler.acquire(owner=f"evidencegate:{phase}")
        units = (*plan.quality_units, *plan.latency_attempts, *plan.smoke_units)
        started_calls = 0
        remaining: tuple[ExperimentUnit, ...] = ()
        try:
            for index, unit in enumerate(units):
                if self.scheduler.hard_stopped or (max_calls is not None and started_calls >= max_calls):
                    self.scheduler.hard_stop()
                    remaining = units[index:]
                    break
                attempt_id = new_attempt_id()
                try:
                    self.scheduler.reserve_gpu(reserved_seconds_per_attempt, gpu_hour_cap * 3600.0, attempt_id)
                except RuntimeError as error:
                    if str(error) != "gpu_hour_cap_would_be_exceeded":
                        raise
                    self.scheduler.hard_stop()
                    remaining = units[index:]
                    break
                started_at = time.monotonic()
                try:
                    result = dict(self.adapter(unit))
                    state = TerminalState(result.pop("state", TerminalState.SUCCESS.value))
                except TimeoutError:
                    result, state = {}, TerminalState.TIMEOUT
                except MemoryError:
                    result, state = {}, TerminalState.OOM
                except Exception:
                    result, state = {}, TerminalState.CRASH
                elapsed = time.monotonic() - started_at
                self.scheduler.settle_gpu(elapsed)
                result.update({"attempt_index": unit.attempt_index, "phase": phase, "gpu_seconds": elapsed})
                self.record_attempt(unit.unit_id, state, payload=result, attempt_id=attempt_id)
                started_calls += 1
            if remaining:
                self.close_remaining(remaining, skip_existing=False)
            return aggregate_terminals(self.attempts)
        finally:
            self.scheduler.release()


def _fixture_manifest() -> ScopeManifest:
    digest = hashlib.sha256(b"evidencegate-fixture").hexdigest()
    return ScopeManifest(
        tuple(VideoSample(f"fixture-video-{i:02d}", f"fixture-sequence-{i:02d}", "STATIC_PRIMARY" if i < 17 else "MOVING_CAMERA_STRESS") for i in range(50)),
        tuple(f"fixture-image-{i:02d}" for i in range(48)),
        tuple(f"fixture-video-{i:02d}" for i in range(24)),
        tuple(f"fixture-image-{i:02d}" for i in range(12)),
        tuple(f"fixture-development-image-{i:02d}" for i in range(12)),
        "fixture/video.mp4",
        digest,
        digest,
        digest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EvidenceGate manifest-driven runner")
    parser.add_argument("--scope", required=True, help="frozen G0 scope JSON path; enum values require --fixture")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", default="development")
    parser.add_argument("--fixture", action="store_true", help="compile and report the deterministic fixture matrix")
    args = parser.parse_args(argv)
    scope_path = Path(args.scope)
    if scope_path.is_file():
        raw = json.loads(scope_path.read_text(encoding="utf-8"))
        verdict = ScopeVerdict(str(raw["scope_verdict"]).upper())
        manifest = ScopeManifest.from_mapping(raw["scope_manifest"])
        plan = compile_matrix(verdict, manifest)
        gates = raw.get("gates", {})
        if gates and (plan.quality_count, plan.latency_count, plan.smoke_count, plan.retry_budget, plan.total_calls) != (
            gates["quality_logic_units"], gates["latency_attempts"], gates["smoke_call_cap"], gates["retry_attempt_budget"], gates["total_call_hard_cap"]
        ):
            raise ValueError("scope_plan_gate_mismatch")
    elif not args.fixture:
        raise ValueError("scope_path_required_without_fixture")
    elif args.scope == ScopeVerdict.STOP.value:
        plan = MatrixPlan(ScopeVerdict.STOP, (), (), (), 0)
    else:
        plan = compile_matrix(args.scope, _fixture_manifest())
    print(json.dumps({"scope": plan.scope.value, "phase": args.phase, "quality": plan.quality_count, "latency": plan.latency_count, "smoke": plan.smoke_count, "retry_budget": plan.retry_budget, "total_calls": plan.total_calls}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
