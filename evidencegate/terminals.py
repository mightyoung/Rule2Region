from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class TerminalState(str, Enum):
    SUCCESS = "success"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    OOM = "OOM"
    CRASH = "crash"
    NOT_RUN_DUE_TO_HARD_STOP = "not_run_due_to_hard_stop"


@dataclass(frozen=True)
class AttemptRecord:
    unit_id: str
    attempt_id: str
    state: TerminalState
    retry_index: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.unit_id.strip() or not self.attempt_id.strip():
            raise ValueError("unit_id_and_attempt_id_required")
        if self.retry_index < 0:
            raise ValueError("retry_index_must_be_nonnegative")


@dataclass(frozen=True)
class TerminalRecord:
    unit_id: str
    state: TerminalState
    attempt_ids: tuple[str, ...]
    aggregate: bool = True

    def __post_init__(self) -> None:
        if not self.unit_id.strip() or not self.attempt_ids:
            raise ValueError("terminal_requires_unit_and_attempts")
        if len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("attempt_ids_must_be_unique")
        if not self.aggregate:
            raise ValueError("terminal_record_must_be_aggregate")


def new_attempt_id() -> str:
    return uuid.uuid4().hex


def aggregate_terminals(attempts: Iterable[AttemptRecord]) -> tuple[TerminalRecord, ...]:
    grouped: dict[str, list[AttemptRecord]] = {}
    seen: set[str] = set()
    for attempt in attempts:
        if attempt.attempt_id in seen:
            raise ValueError("duplicate_attempt_id")
        seen.add(attempt.attempt_id)
        grouped.setdefault(attempt.unit_id, []).append(attempt)
    records: list[TerminalRecord] = []
    for unit_id in sorted(grouped):
        values = grouped[unit_id]
        # Last recorded attempt is authoritative; all prior attempts remain in
        # the aggregate for retry-preserving provenance.
        records.append(TerminalRecord(unit_id, values[-1].state, tuple(v.attempt_id for v in values)))
    return tuple(records)
