from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidencegate.formal_manifest import compile_formal_manifest, validate_formal_manifest
from evidencegate.manifest import write_immutable_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and freeze the G3 formal quality manifest.")
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_formal_manifest(args.scope, args.config)
    validate_formal_manifest(manifest)
    artifact = write_immutable_manifest(args.output, manifest)
    print(json.dumps({
        "path": artifact.path,
        "file_sha256": artifact.sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "quality_units": len(manifest["units"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
