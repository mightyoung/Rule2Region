"""Idle F_card replay from dual-ledger history. No GPU, no validation scoring."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from research_tools.execution_policy.f_card import replay
from research_tools.execution_policy.run_crop_infer import load_view_cards
from research_tools.execution_policy.run_e4 import load_dual_ledger, view_id
from research_tools.route_a_clip_commissioning import load_clips
from research_tools.run_dcrq_development_gate import SEALED_CAMERA


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    dual_ledger_path: Path,
    e4_result: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    clips = load_clips(cache_root, pilot_root)
    if any(SEALED_CAMERA in sequence or str(clip["camera_id"]) == SEALED_CAMERA for sequence, clip in clips.items()):
        raise ValueError("sealed_camera_present")
    dual = load_dual_ledger(dual_ledger_path)
    cards = load_view_cards(json.loads(e4_result.read_text(encoding="utf-8")))
    view_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence, clip in sorted(clips.items()):
        view = view_id(str(clip["camera_id"]))
        view_sources[view].append(
            {
                "sequence": sequence,
                "camera_id": clip["camera_id"],
                "history": list(clip["history_frames"]),
                "validation": list(clip["validation_frames"]),
                "ledgers": dual[sequence],
                "truth": clip["truth"],
            }
        )
    result = replay(view_sources=view_sources, cards=cards)
    dumped = json.dumps(result, indent=2, sort_keys=True)
    if SEALED_CAMERA in dumped:
        raise ValueError("sealed_camera_leaked")
    (output_dir / "result.json").write_text(dumped + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--dual-ledger", type=Path, required=True)
    parser.add_argument("--e4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        dual_ledger_path=args.dual_ledger,
        e4_result=args.e4_result,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "official_h1": False,
                "plane": "idle",
                "promoted_view_count": result["promoted_view_count"],
                "view_count": result["view_count"],
                "time_basis": result["time_basis"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
