"""Pinned offline probe orchestration; private output is never safe to git-add."""
import argparse, datetime, hashlib, io, json, math, os, platform, subprocess, sys, tarfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def write(p, obj): Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576), b''): h.update(b)
 return h.hexdigest()
def require(condition, message):
 if not condition: raise ValueError(message)
def outside_git(path):
 p=Path(path).resolve()
 for parent in [p, *p.parents]:
  if (parent/'.git').exists():
   git=subprocess.run(['git','-C',str(parent),'rev-parse','--show-toplevel'],capture_output=True,text=True)
   require(git.returncode!=0, 'private output/input must be outside every actual git checkout')
 return p

def prepare(a):
 dst=outside_git(a.data);dst.mkdir(parents=True,exist_ok=False)
 manifest=read(ROOT/'input_manifest.json')
 require(Path(a.archive).stat().st_size==manifest['archive_bytes'] and sha(a.archive)==manifest['archive_sha256'],'archive hash/size mismatch')
 require(sha(a.license)=='d18e672f3d402615178b8f586d587d5071259c819b13db9558c4e725ad76fcb6','original license notice mismatch; obtain retained original authorized notice')
 (dst/'SOURCE_LICENSE_ORIGINAL.html').write_bytes(Path(a.license).read_bytes())
 rows={x['nested_archive']:x for x in manifest['frames']};found=[]
 with tarfile.open(a.archive,'r|xz') as outer:
  for m in outer:
   if m.name not in rows: continue
   require(m.isfile(), 'nested archive is not a regular file')
   data=outer.extractfile(m).read();row=rows[m.name]
   with tarfile.open(fileobj=io.BytesIO(data),mode='r|xz') as inner:
    for member in inner:
     if member.name!=row['member']:continue
     require(member.isfile(),'source member is not regular')
     raw=inner.extractfile(member).read();require(hashlib.sha256(raw).hexdigest()==row['sha256'],'source frame mismatch')
     (dst/(row['id']+'.jpg')).write_bytes(raw);found.append(row['id']);break
 require(set(found)=={x['id'] for x in rows.values()},'missing source frame')
 write(dst/'IMPORT_RECEIPT_PRIVATE.json', {'schema_version':1,'archive_sha256':sha(a.archive),'license_sha256':sha(a.license),'source_ids':found,'native_pts':None,'human_GT':False,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print('Prepared 3 byte-verified originals; no annotations inferred')

def preflight(a):
 import importlib.metadata as md
 import torch
 from PIL import Image
 require(platform.python_version()=='3.12.14','requires Python 3.12.14')
 expected={'transformers':'4.57.1','safetensors':'0.6.2','tokenizers':'0.22.2','huggingface-hub':'0.35.3','numpy':'2.5.4','pillow':'12.3.0'}
 versions={k:md.version(k) for k in expected};require(versions==expected,'runtime version mismatch')
 require(torch.__version__==('2.8.0+cpu' if a.device=='cpu' else '2.8.0+cu128'),'torch build mismatch')
 if a.device=='cuda':require(torch.cuda.is_available(),'CUDA unavailable; no fallback')
 model=read(ROOT/'model_manifest.json');checks=[]
 for f in model['model_snapshot_files']:
  p=Path(a.model)/f['name'];require(p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'model file mismatch: '+f['name']);checks.append({'name':f['name'],'sha256':sha(p)})
 for x in read(ROOT/'input_manifest.json')['frames']:
  p=Path(a.data)/(x['id']+'.jpg');require(sha(p)==x['sha256'],'input mismatch '+x['id'])
  with Image.open(p) as im:require(list(im.size)==x['dimensions'] and im.format=='JPEG','wrong dimensions/format')
 require(sha(Path(a.data)/'SOURCE_LICENSE_ORIGINAL.html')=='d18e672f3d402615178b8f586d587d5071259c819b13db9558c4e725ad76fcb6','missing original license')
 return {'versions':versions,'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu_name':torch.cuda.get_device_name() if a.device=='cuda' else None,'model_checks':checks}

def run(a):
 out=outside_git(a.out);out.mkdir(parents=True,exist_ok=False)
 cfg=read(ROOT/'experiment.json');manifest=read(ROOT/'input_manifest.json');model=read(ROOT/'model_manifest.json')
 run={'schema_version':1,'status':'running','kind':'real_cpu_smoke' if a.smoke else 'real_prompt_probe','device':a.device,'decision':'unknown','human_GT':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'trials':[],'manifest_hashes':{p:sha(ROOT/p) for p in ['input_manifest.json','model_manifest.json','experiment.json','forward.py','handoff.py']},'git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()}
 write(out/'RUN_PRIVATE.json',run)
 try:
  run['environment']=preflight(a);write(out/'RUN_PRIVATE.json',run)
  write(out/'MODEL_ENV_PRIVATE.json',{'configuration':{'model_revision':model['revision']},'weight_sha256':model['weight_sha256'],'model_snapshot_files':model['model_snapshot_files']})
  selections=[(x,arm,prompt) for x in manifest['frames'] for arm,prompt in cfg['prompts'].items()]
  if a.smoke:selections=selections[:1]
  for x,arm,prompt in selections:
   trial=out/(x['id']+'--'+arm);trial.mkdir();src=trial/'SOURCE_PRIVATE.json';write(src,{'source_sha256':x['sha256']})
   cmd=[sys.executable,str(ROOT/'forward.py'),'--model-dir',str(Path(a.model).resolve()),'--model-env',str(out/'MODEL_ENV_PRIVATE.json'),'--original',str(Path(a.data).resolve()/(x['id']+'.jpg')),'--source-receipt',str(src),'--output-dir',str(trial),'--device',a.device,'--prompt',prompt]
   row={'source_id':x['id'],'arm':arm,'argv':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_monotonic':time.monotonic(),'status':'running','decision':'unknown'};run['trials'].append(row);write(out/'RUN_PRIVATE.json',run)
   with (trial/'stdout.log').open('w') as so,(trial/'stderr.log').open('w') as se:
    proc=subprocess.Popen(cmd,stdout=so,stderr=se);row['pid']=proc.pid;write(out/'RUN_PRIVATE.json',run);row['exit_code']=proc.wait()
   row.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),end_monotonic=time.monotonic())
   require(row['exit_code']==0,'child failed; stop without retry')
   from PIL import Image,ImageDraw
   receipt=read(trial/'DOOR_PROBE_RECEIPT_PRIVATE.json');im=Image.open(Path(a.data)/(x['id']+'.jpg')).convert('RGB');draw=ImageDraw.Draw(im)
   for c in receipt['model_candidates_after_nms']:
    b=c['bbox_xyxy_original_unclipped'];draw.rectangle(b,outline='red',width=3);draw.text((max(0,b[0]),max(18,b[1])),c['prediction_id']+' '+c['text_label'],fill='red')
   draw.rectangle((0,0,im.width,17),fill='white');draw.text((3,2),'AI ONLY / '+x['id']+' / '+arm+' / DECISION UNKNOWN',fill='black');im.save(trial/'AI_PENDING_HUMAN.png')
   row['artifacts']={p.name:sha(p) for p in trial.iterdir() if p.is_file()};row['status']='completed';write(out/'RUN_PRIVATE.json',run)
  run['status']='completed'
 except Exception as e:
  run.update(status='failed',failure_type=type(e).__name__,failure_reason=str(e))
  raise
 finally:
  run['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(out/'RUN_PRIVATE.json',run)
 print('Completed; validate private artifacts before return')

def summarize_receipt(r):
 require(r['human_GT'] is False and r['accuracy_metrics'] is None,'unexpected GT/accuracy')
 for k in ['S_image_relation','B_image_relation','event_decision','trajectory','direction']:require(r[k]=='unknown','unknown scope changed')
 require(r['box_threshold']==.25 and r['text_threshold']==.25 and r['classwise_nms_iou']==.7,'threshold drift')
 require(r['kept_count']==len(r['model_candidates_after_nms']),'count mismatch')
 clipped=0
 for c in r['model_candidates_after_nms']:
  b=c['bbox_xyxy_original_unclipped'];require(len(b)==4 and all(math.isfinite(v) for v in b),'non-finite coordinates')
  require(math.isfinite(c['score']) and 0<=c['score']<=1,'bad score')
  require(b[2]>b[0] and b[3]>b[1],'degenerate box')
  clipped+=int(b[0]<=0 or b[1]<=0 or b[2]>=800 or b[3]>=600)
 return {'kept_count':r['kept_count'],'image_edge_touch_or_overrun_count':clipped,'forward_seconds':r['forward_seconds'],'peak_cuda_bytes':r['peak_cuda_bytes']}

def check_tensors(raw, inp):
 import torch
 require(raw['pred_boxes'].shape==(1,900,4) and raw['logits'].shape==(1,900,256),'raw shape mismatch')
 require(torch.isfinite(raw['pred_boxes']).all() and torch.isfinite(inp['pixel_values']).all(),'non-finite geometry/input')
 n=raw['attention_mask'].shape[1];mask=torch.zeros((1,256),dtype=torch.bool);mask[:,:n]=raw['attention_mask'].bool()
 mask=mask[:,None,:].expand_as(raw['logits'])
 require(torch.isfinite(raw['logits'][mask]).all(),'non-finite active token logits')
 require(torch.isneginf(raw['logits'][~mask]).all(),'unexpected padded token logits')
 require(torch.equal(raw['input_ids'],inp['input_ids']) and torch.equal(raw['attention_mask'],inp['attention_mask']),'input token mismatch')

def validate(a):
 out=Path(a.out);r=read(out/'RUN_PRIVATE.json');require(r['status']=='completed','failed/incomplete run; keep private failure, no success export')
 require(r['schema_version']==1 and r['human_GT'] is False and r['decision']=='unknown','invalid scope')
 for p,h in r['manifest_hashes'].items():require(sha(ROOT/p)==h,'code/config mismatch '+p)
 cfg=read(ROOT/'experiment.json');frames={x['id']:x for x in read(ROOT/'input_manifest.json')['frames']};model=read(ROOT/'model_manifest.json');summary=[]
 expected={(x,k) for x in frames for k in cfg['prompts']}
 if r['kind']=='real_cpu_smoke':expected={(next(iter(frames)),'combined')}
 require({(x['source_id'],x['arm']) for x in r['trials']}==expected and len(r['trials'])==len(expected),'missing/duplicate trial')
 from safetensors.torch import load_file
 import torch
 for row in r['trials']:
  require(row['status']=='completed' and row['exit_code']==0 and row['pid']>0,'bad subprocess evidence')
  require(row['end_monotonic']>=row['start_monotonic'],'bad timing')
  trial=out/(row['source_id']+'--'+row['arm'])
  require(set(row['artifacts'])=={'SOURCE_PRIVATE.json','stdout.log','stderr.log','DOOR_RAW_OUTPUT.safetensors','PROCESSED_INPUT.safetensors','DOOR_PROBE_RECEIPT_PRIVATE.json','AI_PENDING_HUMAN.png'},'missing artifacts')
  for p,h in row['artifacts'].items():require(sha(trial/p)==h,'artifact hash mismatch '+p)
  receipt=read(trial/'DOOR_PROBE_RECEIPT_PRIVATE.json');require(receipt['source_sha256']==frames[row['source_id']]['sha256'],'wrong source')
  require(receipt['prompt']==cfg['prompts'][row['arm']] and receipt['model_revision']==model['revision'] and receipt['weight_sha256']==model['weight_sha256'] and receipt['device']==r['device'],'wrong model/arm/device')
  require(receipt['raw_tensor_sha256']==sha(trial/'DOOR_RAW_OUTPUT.safetensors'),'raw tensor mismatch')
  raw=load_file(str(trial/'DOOR_RAW_OUTPUT.safetensors'));inp=load_file(str(trial/'PROCESSED_INPUT.safetensors'))
  check_tensors(raw, inp)
  ids=torch.nonzero(raw['logits'].sigmoid().max(dim=-1).values[0]>.25).flatten().tolist()
  require(ids==[x['query_index'] for x in receipt['model_candidates_before_nms']],'raw query decoding mismatch')
  summary.append({'source_id':row['source_id'],'arm':row['arm'],**summarize_receipt(receipt)})
 public={'schema_version':1,'kind':r['kind'],'device':r['device'],'git_commit':r['git_commit'],'decision':'unknown','human_GT':False,'accuracy_metrics':None,'source_manifest_sha256':r['manifest_hashes']['input_manifest.json'],'model_revision':model['revision'],'completed_forward_calls':len(summary),'trials':summary,'limitations':['no complete ROI GT','image-edge count is not physical threshold validity','no cross-camera registration','not held-out confirmation','GPU arithmetic need not equal CPU bytewise']}
 write(out/'PUBLIC_AGGREGATE.json',public);print(json.dumps({'valid':True,'forward_calls':len(summary),'public_export':'PUBLIC_AGGREGATE.json'}))

if __name__=='__main__':
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
 q=sp.add_parser('prepare');q.add_argument('--archive',required=True);q.add_argument('--license',required=True);q.add_argument('--data',required=True);q.set_defaults(fn=prepare)
 q=sp.add_parser('run');q.add_argument('--data',required=True);q.add_argument('--model',required=True);q.add_argument('--out',required=True);q.add_argument('--device',choices=['cpu','cuda'],required=True);q.add_argument('--smoke',action='store_true');q.set_defaults(fn=run)
 q=sp.add_parser('validate');q.add_argument('--out',required=True);q.set_defaults(fn=validate)
 a=p.parse_args();a.fn(a)
