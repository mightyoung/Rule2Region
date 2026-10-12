"""Operator-only, explicit pinned download; never called by run/prepare."""
import argparse,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
from handoff import ROOT,outside_git,require,sha
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();dst=outside_git(a.model)
m=json.loads((ROOT/'model_manifest.json').read_text())
snapshot_download(repo_id=m['model_id'],revision=m['revision'],local_dir=dst,allow_patterns=[x['name'] for x in m['model_snapshot_files']],token=False)
for x in m['model_snapshot_files']:
 f=dst/x['name'];require(f.stat().st_size==x['bytes'] and sha(f)==x['sha256'],'model mismatch '+x['name'])
print('Pinned 9 model files verified; no inference run')
