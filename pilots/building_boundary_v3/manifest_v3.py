import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    target=ROOT/'MANIFEST.json'
    files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(ROOT.iterdir()) if p.is_file() and p!=target and p.suffix in {'.py','.json','.md','.html','.txt'}}
    if args.check:
        assert files==json.loads(target.read_text())['files'],'v3 manifest mismatch'
        print(f'v3 manifest verified: {len(files)} files')
    else:
        target.write_text(json.dumps({'scope':'bounded_I1_M1_synthetic_revision',
            'baseline_commit':'850b4f3f3f9fc1ac955b16eaa070493a2e453b4d','files':files,
            'manifest_excludes_itself':True,'new_head_independent_review':'PENDING'},indent=2)+'\n')
        print(f'v3 manifest frozen: {len(files)} files')

if __name__=='__main__':main()
