"""New private archive with required originals; never overwrite an old bundle.
Production CLI uses pinned source hashes. Tests inject explicitly synthetic bytes.
"""
import argparse,hashlib,io,json,sys,tarfile
from pathlib import Path
from audit_returns import BASE,require
from check_private_return import ARTIFACTS,LICENSE_SHA
REPO=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pack(root,run_name,out,*,source_spec=None,license_sha=LICENSE_SHA,allow_partial=False):
 root=Path(root);out=Path(out);require(run_name in {'cpu-matrix','gpu-matrix'},'unknown matrix')
 require(not out.exists(),'archive exists; choose new name, preserve old evidence')
 sources=source_spec if source_spec is not None else json.loads((REPO/BASE/'input_manifest.json').read_text())['frames']
 paths=[]
 for x in sources:
  rel='data/'+x['id']+'.jpg';p=root/rel;require(p.is_file() and not p.is_symlink(),'missing/nonregular original: '+rel);require(sha(p)==x['sha256'],'source hash mismatch: '+rel);paths.append(rel)
 license_rel='data/SOURCE_LICENSE_ORIGINAL.html';p=root/license_rel;require(p.is_file() and not p.is_symlink(),'missing original license');require(sha(p)==license_sha,'license hash mismatch');paths.append(license_rel)
 required=[run_name+'/'+f for f in ['RUN_PRIVATE.json','MODEL_ENV_PRIVATE.json']]
 for x in sources:
  for arm in ['combined','door_only','doorway_only']:
   required.extend(run_name+'/'+x['id']+'--'+arm+'/'+f for f in ARTIFACTS)
 missing=[]
 for rel in required:
  p=root/rel
  if not p.is_file():missing.append(rel);continue
  require(not p.is_symlink(),'symlink artifact rejected: '+rel);paths.append(rel)
 require(allow_partial or not missing,'incomplete matrix; preserve failure and use explicit --allow-partial')
 for rel in [run_name+'/PUBLIC_AGGREGATE.json','environment.freeze.txt']+[run_name+'.server-logs/'+f for f in ['environment.freeze.txt','pip-check.log','run-resource.txt','run.stdout.log','run.stderr.log','validate-resource.txt','validate.stdout.log','validate.stderr.log']]:
  p=root/rel
  if p.is_file():require(not p.is_symlink(),'symlink log rejected');paths.append(rel)
 manifest={'schema_version':1,'kind':'private_return_archive_inventory','matrix':run_name,'complete_required_file_set':not missing,'allow_partial':allow_partial,'missing':missing,'human_GT':False,'decision':'unknown','artifact_contents_validated':False,'files':[{'path':rel,'bytes':(root/rel).stat().st_size,'sha256':sha(root/rel)} for rel in paths]}
 # Exclusive create prevents racing or silently replacing an older archive.
 with out.open('xb') as stream:
  with tarfile.open(fileobj=stream,mode='w:gz') as archive:
   for rel in paths:archive.add(root/rel,arcname=rel,recursive=False)
   b=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('RETURN_ARCHIVE_MANIFEST_PRIVATE.json');info.size=len(b);archive.addfile(info,io.BytesIO(b))
 return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-name',choices=['cpu-matrix','gpu-matrix'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
 sys.path.insert(0,str(REPO/BASE));import handoff
 handoff.outside_git(a.out);m=pack(a.root,a.run_name,a.out,allow_partial=a.allow_partial)
 print(json.dumps({'archive_sha256':sha(a.out),'required_files_complete':m['complete_required_file_set'],'missing_count':len(m['missing']),'content_acceptance':'not_run','decision':'unknown'}))
