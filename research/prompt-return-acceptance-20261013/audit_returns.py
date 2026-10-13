"""Offline audit of pinned Git return objects; never fetch or run a model."""
import argparse,hashlib,json,math,subprocess
from pathlib import Path
BASE='research/gpu-prompt-boundary-handoff-20261012/'
PINS={'cpu-matrix':'2b85946e48cdc46ca780d2659ab8cffa3cb664c8','gpu-matrix':'c09db0d147ed1d1a9426f14dfc2337ea08a8da80','receipts':'7151e4b321f2cda87d9c99ace729c39e16015c40'}
AGG={'cpu-matrix':'returns/cpu-operator-run-1/aggregate.json','gpu-matrix':'returns/operator-run-1/aggregate.json'}
def require(c,m):
 if not c:raise ValueError(m)
def digest(b):return hashlib.sha256(b).hexdigest()
def blob(repo,commit,path):return subprocess.check_output(['git','-C',str(repo),'show',commit+':'+path])
def load(b):return json.loads(b)
def audit(repo):
 src_bytes=blob(repo,'3502616f9eca5c99ca05871daa4479058fad6ef1',BASE+'input_manifest.json');sources={r['id']:r for r in load(src_bytes)['frames']}
 model=load(blob(repo,'3502616f9eca5c99ca05871daa4479058fad6ef1',BASE+'model_manifest.json'))
 prompts=load(blob(repo,'3502616f9eca5c99ca05871daa4479058fad6ef1',BASE+'experiment.json'))['prompts']
 mb=blob(repo,PINS['receipts'],BASE+'returns/sanitized-receipts/MANIFEST.json');manifest=load(mb)
 require(manifest['human_GT'] is False and manifest['decision']=='unknown' and manifest['accuracy_metrics'] is None,'receipt manifest scope drift')
 rows=[];aggregates={};private_receipts={}
 for name,run in manifest['runs'].items():
  require(name in ['cpu-matrix','gpu-matrix','cpu-smoke'],'unexpected run')
  require(run['receipt_count']==len(run['receipts']),'receipt count drift')
  expected={(s,k) for s in sources for k in prompts} if name!='cpu-smoke' else {('C1-f00000000','combined')}
  require({(x['source_id'],x['arm']) for x in run['receipts']}==expected and len(expected)==len(run['receipts']),'missing/duplicate arm')
  agg=None
  if name in AGG:
   ab=blob(repo,PINS[name],BASE+AGG[name]);agg=load(ab);aggregates[name]=agg
   require(agg['source_manifest_sha256']==digest(src_bytes) and agg['model_revision']==model['revision'],'aggregate source/model drift')
   require(agg['human_GT'] is False and agg['decision']=='unknown' and agg['accuracy_metrics'] is None,'aggregate scope drift')
   require(agg['completed_forward_calls']==9 and agg['git_commit']==run['git_commit'] and agg['device']==run['device'],'aggregate provenance drift')
   require({(x['source_id'],x['arm']) for x in agg['trials']}==expected and len(agg['trials'])==9,'aggregate matrix drift')
  for row in run['receipts']:
   require(row['path']==name+'/'+row['source_id']+'--'+row['arm']+'.json','unexpected receipt path')
   b=blob(repo,PINS['receipts'],BASE+'returns/sanitized-receipts/'+row['path']);r=load(b)
   require(len(b)==row['bytes'] and digest(b)==row['sha256'],'receipt byte/hash drift')
   require(r['source_sha256']==sources[row['source_id']]['sha256'] and r['full_original_dimensions']==[800,600],'source identity drift')
   require(r['weight_sha256']==model['weight_sha256'] and r['model_revision']==model['revision'],'weight identity drift')
   for key in ['device','prompt','kept_count','thresholded_count','forward_seconds','peak_cuda_bytes']:require(r[key]==row[key],'manifest receipt field drift '+key)
   require(r['prompt']==prompts[row['arm']] and r['device']==run['device'],'arm/device drift')
   require(r['box_threshold']==.25 and r['text_threshold']==.25 and r['classwise_nms_iou']==.7 and r['queries']==900,'configuration drift')
   require(r['human_GT'] is False and r['accuracy_metrics'] is None and r['physical_door_count'] is None,'receipt GT drift')
   for k in ['S_image_relation','B_image_relation','event_decision','trajectory','direction']:require(r[k]=='unknown','unknown changed')
   require(r['kept_count']==len(r['model_candidates_after_nms']) and r['thresholded_count']==len(r['model_candidates_before_nms']),'candidate count drift')
   edge=0
   for c in r['model_candidates_after_nms']:
    box=c['bbox_xyxy_original_unclipped'];require(len(box)==4 and all(math.isfinite(v) for v in box),'nonfinite candidate')
    require(box[2]>box[0] and box[3]>box[1] and math.isfinite(c['score']) and 0<=c['score']<=1,'invalid candidate')
    edge+=int(box[0]<=0 or box[1]<=0 or box[2]>=800 or box[3]>=600)
   if agg:
    t=next(x for x in agg['trials'] if (x['source_id'],x['arm'])==(row['source_id'],row['arm']))
    for k in ['kept_count','forward_seconds','peak_cuda_bytes']:require(t[k]==r[k],'aggregate receipt drift')
    require(t['image_edge_touch_or_overrun_count']==edge,'edge audit drift')
   rows.append({'run':name,'source_id':row['source_id'],'arm':row['arm'],'receipt_sha256':digest(b),'raw_tensor_declared_sha256':r['raw_tensor_sha256'],'kept_count':r['kept_count']})
   private_receipts[(name,row['source_id'],row['arm'])]=(r,b,row)
 require(len(rows)==19,'expected 19 receipts')
 report={'schema_version':1,'status':'public_git_bytes_and_internal_consistency_verified','pins':PINS,'receipt_manifest_sha256':digest(mb),'source_manifest_sha256':digest(src_bytes),'verified_receipts':len(rows),'CPU_GPU_count_equal_trials':sum(a['kept_count']==next(b['kept_count'] for b in aggregates['gpu-matrix']['trials'] if (a['source_id'],a['arm'])==(b['source_id'],b['arm'])) for a in aggregates['cpu-matrix']['trials']),'private_original_package_verified':False,'human_GT':False,'decision':'unknown','accuracy_metrics':None,'rows':rows,'missing_return_evidence':['operator original source JPEG bytes','operator raw/processed tensors','operator candidate PNG renders','operator RUN_PRIVATE actual argv/PID/UTC/environment and stdout/stderr/resource logs','operator retained original source license bytes','complete physical doorway/threshold human GT (not created)']}
 return report,private_receipts,aggregates
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True);a=p.parse_args();report,_,_=audit(a.repo);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'verified_receipts':19,'private_original_package_verified':False,'decision':'unknown'}))
