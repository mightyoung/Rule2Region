import argparse
import json
from pathlib import Path
from .contract import validate_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate metadata only; does not verify media or onsite truth")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    print(json.dumps({"valid": True, "purpose": manifest["purpose"], "samples": len(manifest["samples"])}))


if __name__ == "__main__":
    main()
