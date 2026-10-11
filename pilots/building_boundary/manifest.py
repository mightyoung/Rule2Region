"""Hash/check only this pilot's publishable text artifacts; no external assets."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT/'MANIFEST.json'


def entries():
    return {p.name:{'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(ROOT.iterdir()) if p.is_file() and p != MANIFEST
            and p.suffix in {'.py','.json','.md','.html'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    actual=entries()
    if args.check:
        frozen=json.loads(MANIFEST.read_text())['files']
        if actual != frozen:
            raise SystemExit('Manifest mismatch: '+', '.join(sorted(k for k in actual.keys() | frozen.keys() if actual.get(k)!=frozen.get(k))))
        print(f'Manifest verified: {len(actual)} files')
    else:
        result={'scope':'self_authored_public_pilot_only_no_raw_media',
                'base_main_sha':'cf22ddd8e46ee83ddd9565b058e1df4e362377ea',
                'protocol_freeze_commit':'59d8faf',
                'manifest_excludes_itself':True, 'files':actual}
        MANIFEST.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(f'Manifest frozen: {len(actual)} files')

if __name__=='__main__':
    main()
