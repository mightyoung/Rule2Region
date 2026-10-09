from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class StageRecord:
    stage: str
    elapsed_ms: float
    status: str
    error: str | None = None


class RunLedger:
    def __init__(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.metadata = dict(metadata or {})
        self._stages: list[StageRecord] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        try:
            yield
        except Exception as error:
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            self._stages.append(StageRecord(name, elapsed_ms, "failed", repr(error)))
            raise
        else:
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            self._stages.append(StageRecord(name, elapsed_ms, "passed"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "stages": [asdict(stage) for stage in self._stages],
            "elapsed_ms": sum(stage.elapsed_ms for stage in self._stages),
            "status": (
                "failed"
                if any(stage.status == "failed" for stage in self._stages)
                else "passed"
            ),
        }
