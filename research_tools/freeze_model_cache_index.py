from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegate.manifest import write_immutable_manifest
from evidencegate.model_snapshot import build_model_cache_index, validate_model_cache_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate fixed model snapshots and freeze their runtime index.")
    parser.add_argument("--model", nargs=2, action="append", metavar=("FAMILY", "SNAPSHOT_ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {family: Path(root) for family, root in args.model}
    if len(models) != len(args.model):
        raise ValueError("duplicate_model_family")
    index = build_model_cache_index(models)
    validate_model_cache_index(index)
    artifact = write_immutable_manifest(args.output, index)
    print(json.dumps({
        "path": artifact.path,
        "file_sha256": artifact.sha256,
        "manifest_sha256": index["manifest_sha256"],
        "models": sorted(index["models"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
