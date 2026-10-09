from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .admission import validate_admission_receipt
from .io import read_jsonl
from .schemas import CandidateFeature
from .selectors import codec_guided_select, random_select, uniform_select


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codecref-lite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    admission = subparsers.add_parser("validate-admission")
    admission.add_argument("receipt", type=Path)
    admission.add_argument("--skip-file-hashes", action="store_true")

    selection = subparsers.add_parser("select")
    selection.add_argument("method", choices=("uniform", "random", "codec"))
    selection.add_argument("--candidate-count", type=int, default=8)
    selection.add_argument("--budget", type=int, default=4)
    selection.add_argument("--window-id", default="window-0")
    selection.add_argument("--seed", type=int, default=17)
    selection.add_argument("--features", type=Path)
    return parser


def _candidate_from_dict(value: dict[str, object]) -> CandidateFeature:
    return CandidateFeature(
        index=int(value["index"]),
        pts_seconds=float(value["pts_seconds"]),
        packet_bpp=float(value["packet_bpp"]),
        mv_density=float(value["mv_density"]),
        mv_mean=float(value["mv_mean"]),
        mv_energy=float(value["mv_energy"]),
        frame_type=str(value["frame_type"]),
        key_frame=bool(value.get("key_frame", False)),
        mv_missing=bool(value.get("mv_missing", False)),
        packet_attribution_supported=bool(
            value.get("packet_attribution_supported", True)
        ),
        pts_supported=bool(value.get("pts_supported", True)),
        mv_side_data_valid=bool(value.get("mv_side_data_valid", True)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "validate-admission":
        receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
        decision = validate_admission_receipt(
            receipt,
            verify_files=not arguments.skip_file_hashes,
            base_dir=arguments.receipt.parent,
        )
        print(json.dumps({"admitted": decision.admitted, "errors": decision.errors}))
        return 0 if decision.admitted else 2

    if arguments.method == "uniform":
        selected = uniform_select(arguments.candidate_count, arguments.budget)
    elif arguments.method == "random":
        selected = random_select(
            arguments.candidate_count,
            arguments.budget,
            arguments.seed,
            arguments.window_id,
        )
    else:
        if arguments.features is None:
            raise SystemExit("--features is required for codec selection")
        features = [_candidate_from_dict(row) for row in read_jsonl(arguments.features)]
        selected = codec_guided_select(features, arguments.budget)
    print(json.dumps({"selected_indices": selected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
