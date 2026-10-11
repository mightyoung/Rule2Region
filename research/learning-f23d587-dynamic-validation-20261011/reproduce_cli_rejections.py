import copy,hashlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import argparse
parser=argparse.ArgumentParser(description='Reproduce four CLI rejections using software-only 8D tensors; no real media/model.')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
source=args.source.resolve();root=args.output_dir.resolve()
assert not root.is_relative_to(source), 'Output must be outside source checkout'
assert not root.exists(), 'Choose a new output directory; no overwrite'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='f23d587de0e8d82b4a823173359d3f34b0990d77'
root.mkdir(parents=True);(root/'logs').mkdir()
sys.path.insert(0,str(source))
import torch
from learning.contract import fingerprint
from learning.train import run
probe=root/'cli-fixture';probe.mkdir(exist_ok=True)
m=json.loads((source/'learning/data/software_manifest.json').read_text());f=json.loads((source/'learning/data/software_features.json').read_text())
for row in m['samples']:
 if row['split']!='test':
  row['input']['view_pairs']=[];row['truth']['matches']={}
 f['samples'][row['id']]['input_sha256']=fingerprint(row['input'])
mp=probe/'manifest.json';fp=probe/'features.json';cp=probe/'software-head.pt';pp=probe/'policy.json'
mp.write_text(json.dumps(m));fp.write_text(json.dumps(f))
run(SimpleNamespace(manifest=mp,features=fp,checkpoint=cp,resume=False,steps=1,seed=20261011,hidden=16,lr=.001,device='cpu',software_test=True))
state=torch.load(cp,weights_only=True)
for name,tensor in state['model'].items():
 tensor.zero_()
 if name.endswith('2.bias'):tensor[1]=2
# Deliberately forced software-only logits, never accuracy evidence.
torch.save(state,cp)
records=[]
base=[sys.executable,'-m','learning.evaluate',str(mp),str(fp),str(cp),'--software-test']
def call(name,args,code,needle=None):
 result=subprocess.run(base+args,cwd=source,capture_output=True,text=True)
 log=root/'logs'/(name+'.log');log.write_text(result.stdout+result.stderr)
 assert result.returncode==code,(name,result.returncode,result.stderr)
 if needle:assert needle in result.stderr,(name,result.stderr)
 records.append({'name':name,'command':base+args,'expected_exit':code,'actual_exit':result.returncode,'expected_error':needle,'log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
 print(json.dumps(records[-1]),flush=True)
 return result
positive=call('cli_calibration_matches_empty',['--split','calibration','--max-risk','1','--policy',str(pp)],0)
report=json.loads(positive.stdout)
assert report['thresholds']['matches'] is None and report['metrics']['matches']['status']=='not_applicable_no_candidates'
call('cli_negative_uncalibrated_test_task',['--split','test','--policy',str(pp)],1,'task not calibrated; do not score test candidates')
call('cli_negative_test_tuning',['--split','test','--max-risk','1','--policy',str(pp)],1,'test may not tune')
policy=json.loads(pp.read_text());policy['task_statuses']['relations']='not_applicable_no_candidates'
bad=probe/'bad-policy.json';bad.write_text(json.dumps(policy))
call('cli_negative_forged_task_status',['--split','test','--policy',str(bad)],1,'policy status/threshold mismatch')
# Restore equal zero logits, so all three classes tie and known classes cannot act.
for tensor in state['model'].values():tensor.zero_()
torch.save(state,cp)
tie_policy=probe/'tie-policy.json'
call('cli_negative_all_top_ties_no_coverage',['--split','calibration','--max-risk','1','--policy',str(tie_policy)],1,'no feasible calibration policy: relations')
assert not tie_policy.exists()
(root/'CLI_PROBE_RECEIPT.json').write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),'kind':'software_test_only','purpose':'Additional CLI negative behavior checks; generated small manifest and fake forced logits, no media/real encoder','positive_calibration_checks':1,'expected_rejection_checks':4,'unexpected_failures':0,'records':records},indent=2)+'\n')
