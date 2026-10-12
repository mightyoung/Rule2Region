"""One fixed door/doorway probe, local cached model, AI candidates only."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import argparse,hashlib,json,time,random
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def run(a):
 import torch,numpy as np
 from PIL import Image
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
 from safetensors.torch import save_file
 env=json.loads(a.model_env.read_text())
 for x in env['model_snapshot_files']:
  p=a.model_dir/x['name']
  if p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('snapshot mismatch')
 if sha(a.original)!=json.loads(a.source_receipt.read_text())['source_sha256']:raise ValueError('source mismatch')
 a.output_dir.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0)
 processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False,use_fast=False,token=False)
 model=AutoModelForZeroShotObjectDetection.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,token=False).to(a.device).eval()
 if a.device=='cuda':
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with Image.open(a.original) as f:image=f.convert('RGB')
 prompt=a.prompt
 inputs=processor(images=image,text=prompt,return_tensors='pt').to(a.device)
 if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter()
 with torch.inference_mode():output=model(**inputs)
 if a.device=='cuda':torch.cuda.synchronize()
 elapsed=time.perf_counter()-start
 result=processor.post_process_grounded_object_detection(output,input_ids=inputs['input_ids'],threshold=.25,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
 ids=torch.nonzero(output.logits.sigmoid().max(dim=-1).values[0]>.25).flatten().tolist()
 rows=[{'prediction_id':f'q{q:03d}','query_index':q,'score':float(score),'text_label':label,'bbox_xyxy_original_unclipped':[float(x) for x in box]} for q,score,label,box in zip(ids,result['scores'],result['text_labels'],result['boxes'])]
 def iou(x,y):
  a0,b0,a1,b1=x;c0,d0,c1,d1=y
  inter=max(0,min(a1,c1)-max(a0,c0))*max(0,min(b1,d1)-max(b0,d0));union=max(0,a1-a0)*max(0,b1-b0)+max(0,c1-c0)*max(0,d1-d0)-inter
  return inter/union if union>0 else 0
 kept=[];suppressed=[]
 for r in sorted(rows,key=lambda x:(-x['score'],x['prediction_id'])):
  hit=next((k for k in kept if k['text_label']==r['text_label'] and iou(k['bbox_xyxy_original_unclipped'],r['bbox_xyxy_original_unclipped'])>.7),None)
  if hit:suppressed.append({'prediction_id':r['prediction_id'],'by':hit['prediction_id'],'iou':iou(hit['bbox_xyxy_original_unclipped'],r['bbox_xyxy_original_unclipped'])})
  else:kept.append(r)
 save_file({'logits':output.logits.cpu().contiguous(),'pred_boxes':output.pred_boxes.cpu().contiguous(),'input_ids':inputs['input_ids'].cpu().contiguous(),'attention_mask':inputs['attention_mask'].cpu().contiguous()},str(a.output_dir/'DOOR_RAW_OUTPUT.safetensors'))
 save_file({k:v.cpu().contiguous() for k,v in inputs.items()},str(a.output_dir/'PROCESSED_INPUT.safetensors'))
 receipt={'device':a.device,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,'status':'new_fixed_prompt_door_probe_completed','frame_index':0,'source_sha256':sha(a.original),'weight_sha256':env['weight_sha256'],'model_revision':env['configuration']['model_revision'],'prompt':prompt,'box_threshold':.25,'text_threshold':.25,'classwise_nms_iou':.7,'nms_convention':'same text label; suppress when IoU>0.7; score desc then prediction ID; no class-synonym merge','queries':900,'forward_seconds':elapsed,'thresholded_count':len(rows),'kept_count':len(kept),'model_candidates_before_nms':rows,'model_candidates_after_nms':kept,'suppression_decisions':suppressed,'full_original_dimensions':[image.width,image.height],'annotation_status':'AI_model_candidates_pending_human_review','physical_door_count':None,'S_image_relation':'unknown','B_image_relation':'unknown','event_decision':'unknown','trajectory':'unknown','direction':'unknown','human_GT':False,'accuracy_metrics':None,'new_forward_passes':1,'new_model_downloads':0,'new_training_runs':0,'historical_person_prompt_comparison':'not applicable: different query, not a historical-replay claim','raw_tensor_sha256':sha(a.output_dir/'DOOR_RAW_OUTPUT.safetensors')}
 (a.output_dir/'DOOR_PROBE_RECEIPT_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['model-dir','model-env','original','source-receipt','output-dir']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--prompt',required=True,choices=['door. doorway.','door.','doorway.'])
 p.add_argument('--device',required=True,choices=['cpu','cuda'])
 run(p.parse_args())
