from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegate.manifest import write_immutable_manifest
from evidencegate.model_snapshot import build_model_snapshot_manifest, validate_model_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_model_snapshot_manifest(args.root, args.repo_id, args.revision)
    validate_model_snapshot(args.root, manifest)
    artifact = write_immutable_manifest(args.output, manifest)
    print(json.dumps({"path": artifact.path, "file_sha256": artifact.sha256, "files": len(manifest["files"]), "total_bytes": manifest["total_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
