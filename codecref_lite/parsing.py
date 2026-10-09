from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CountParseResult:
    valid: bool
    count: int | None
    error: str | None


def parse_count_json(text: str) -> CountParseResult:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return CountParseResult(False, None, f"invalid_json:{error.msg}")
    if not isinstance(value, dict) or set(value) != {"count"}:
        return CountParseResult(False, None, "schema_mismatch")
    count = value["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        return CountParseResult(False, None, "count_not_integer")
    if count < 0:
        return CountParseResult(False, None, "count_negative")
    return CountParseResult(True, count, None)
