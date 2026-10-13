"""Offline inspection of an already-unpacked authorized private return.
Does not extract archives, fetch, execute logs, run forward passes or create GT.
"""
import argparse,hashlib,json,sys,shutil,tempfile
from pathlib import Path
from audit_returns import BASE,audit,require,digest
ARTIFACTS=['SOURCE_PRIVATE.json','stdout.log','stderr.log','DOOR_RAW_OUTPUT.safetensors','PROCESSED_INPUT.safetensors','DOOR_PROBE_RECEIPT_PRIVATE.json','AI_PENDING_HUMAN.png']
LICENSE_SHA='d18e672f3d402615178b8f586d587d5071259c819b13db9558c4e725ad76fcb6'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def inspect(repo,run,name,data=None,license_path=None,model=None):
 public,receipts,aggregates=audit(repo);require(name in ['cpu-matrix','gpu-matrix'],'only full returned matrices')
 missing=[];checks=[];review=[];run=Path(run)
 for f in ['RUN_PRIVATE.json','MODEL_ENV_PRIVATE.json']:
  if not (run/f).is_file():missing.append('run/'+f)
 model_manifest=json.loads((repo/BASE/'model_manifest.json').read_text());sources=json.loads((repo/BASE/'input_manifest.json').read_text())['frames']
 for x in sources:
  if data is None or not (Path(data)/(x['id']+'.jpg')).is_file():missing.append('original/'+x['id']+'.jpg')
  else:require(sha(Path(data)/(x['id']+'.jpg'))==x['sha256'],'original source bytes mismatch');checks.append('original/'+x['id'])
 if license_path is None or not Path(license_path).is_file():missing.append('SOURCE_LICENSE_ORIGINAL.html')
 else:require(sha(license_path)==LICENSE_SHA,'license bytes mismatch');checks.append('original source license')
 if model is not None:
  for x in model_manifest['model_snapshot_files']:
   p=Path(model)/x['name']
   if not p.is_file():missing.append('model/'+x['name'])
   else:require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'model bytes mismatch '+x['name'])
  checks.append('local model snapshot checked where present; not proof of model used by server')
 for (rn,sid,arm),(r,b,row) in receipts.items():
  if rn!=name:continue
  t=run/(sid+'--'+arm)
  for f in ARTIFACTS:
   if not (t/f).is_file():missing.append(sid+'--'+arm+'/'+f)
  p=t/'DOOR_PROBE_RECEIPT_PRIVATE.json'
  if p.is_file():require(p.read_bytes()==b,'private receipt differs from pinned public bytes');checks.append(sid+'--'+arm+'/receipt byte identity')
  p=t/'DOOR_RAW_OUTPUT.safetensors'
  if p.is_file():require(sha(p)==r['raw_tensor_sha256'],'raw tensor differs from declared public hash');checks.append(sid+'--'+arm+'/raw tensor sha')
  for c in r['model_candidates_after_nms']:
   review.append({'source_id':sid,'arm':arm,'prediction_id':c['prediction_id'],'receipt_sha256':row['sha256'],'source_sha256':r['source_sha256'],'AI_bbox_xyxy_original_unclipped':c['bbox_xyxy_original_unclipped'],'AI_score':c['score'],'AI_text_label':c['text_label'],'review_status':'unreviewed','reviewer':None,'reviewed_utc':None,'viewed_original_sha256':None,'seen_AI_candidates':None,'physical_door_identity':'unknown','top_visibility':'unknown','left_visibility':'unknown','right_visibility':'unknown','bottom_visibility':'unknown','threshold_location':'unknown','full_opening_geometry_confirmed':False,'S':'unknown','B':'unknown','events':'unknown','trajectory':'unknown','direction':'unknown','coordinates_added_by_human':None,'note':None})
 result={'schema_version':1,'run_name':name,'status':'incomplete_private_return','missing':missing,'checks':checks,'raw_and_processed_tensor_validation':'not_run_missing_bytes','model_snapshot_bytes_status':'not_supplied' if model is None else 'checked_files_present_only','public_receipts_verified':True,'human_GT':False,'decision':'unknown','accuracy_metrics':None,'scientific_ROI_admission':False,'review_records':len(review)}
 if not missing:
  sys.path.insert(0,str(repo/BASE));import handoff
  # Original return is read-only: the original validator writes only into a scratch copy.
  with tempfile.TemporaryDirectory(prefix='rule2region-validation-') as temp:
   scratch=Path(temp)/'run';shutil.copytree(run,scratch)
   handoff.validate(type('Args',(),{'out':str(scratch)})())
   generated=json.loads((scratch/'PUBLIC_AGGREGATE.json').read_text())
   require(generated==aggregates[name],'regenerated aggregate differs from pinned returned aggregate')
  result.update(status='private_bytes_and_original_validator_checks_passed',raw_and_processed_tensor_validation='passed; normal padded -inf logits allowed',model_snapshot_bytes_status='not_supplied' if model is None else '9 pinned files verified')
 return result,review
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--run-name',choices=['cpu-matrix','gpu-matrix'],required=True);p.add_argument('--source-data',type=Path);p.add_argument('--license',type=Path);p.add_argument('--model',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 # Reports contain private paths/AI candidates and must remain outside Git.
 sys.path.insert(0,str(a.repo/BASE));import handoff
 handoff.outside_git(a.out);a.out.mkdir(parents=True,exist_ok=False)
 try:
  result,review=inspect(a.repo.resolve(),a.run,a.run_name,a.source_data,a.license,a.model)
  (a.out/'PRIVATE_INTEGRITY_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
  (a.out/'HUMAN_REVIEW_TEMPLATE_PRIVATE.json').write_text(json.dumps({'schema_version':1,'human_GT':False,'AI_assisted_review_not_blind_GT':True,'records':review},indent=2)+'\n')
  print(json.dumps({'status':result['status'],'missing_count':len(result['missing']),'decision':'unknown'}))
  if result['missing']:sys.exit(2)
 except (ValueError,FileNotFoundError) as e:
  (a.out/'PRIVATE_FAILURE.json').write_text(json.dumps({'status':'rejected','reason':str(e),'decision':'unknown'},indent=2)+'\n');raise
