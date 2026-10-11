"""Manifest only incremental self-authored text artifacts; baseline remains immutable."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();path=ROOT/'MANIFEST.json'
    files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(ROOT.iterdir()) if p.is_file() and p!=path and p.suffix in {'.py','.json','.md','.html'}}
    if args.check:
        assert files==json.loads(path.read_text())['files'],'Incremental manifest mismatch'
        print(f'v2 manifest verified: {len(files)} files')
    else:
        result={'scope':'incremental_synthetic_admission_revision_only',
            'baseline_commit':'40f0240b6929c13c6ce9bad62b25a5111ec49e85',
            'baseline_manifest_sha256':hashlib.sha256((ROOT.parent/'building_boundary/MANIFEST.json').read_bytes()).hexdigest(),
            'protocol_v2_freeze_commit':'af2adce','files':files,'manifest_excludes_itself':True}
        path.write_text(json.dumps(result,indent=2)+'\n');print(f'v2 manifest frozen: {len(files)} files')

if __name__=='__main__':main()
