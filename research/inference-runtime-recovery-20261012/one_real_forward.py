"""One genuine offline cached GroundingDINO forward; no new model retrieval."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import argparse,hashlib,json,random,time,importlib.metadata
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def run(args):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file,save_file
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    args.output_dir.mkdir(parents=True,exist_ok=True)
    original=json.loads(args.historical_receipt.read_text());env=json.loads(args.historical_model_env.read_text())
    for row in env['model_snapshot_files']:
        p=args.model_dir/row['name']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('cached model snapshot mismatch')
    if sha(args.original)!=original['source_sha256']:raise ValueError('source image mismatch')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0)
    t=time.perf_counter()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,use_fast=False,token=False)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,token=False).to('cpu').eval()
    load_seconds=time.perf_counter()-t
    with Image.open(args.original) as raw:image=raw.convert('RGB')
    t=time.perf_counter();inputs=processor(images=image,text='person.',return_tensors='pt');preprocess_seconds=time.perf_counter()-t
    t=time.perf_counter()
    with torch.inference_mode():outputs=model(**inputs)
    forward_seconds=time.perf_counter()-t
    result=processor.post_process_grounded_object_detection(outputs,input_ids=inputs['input_ids'],threshold=.25,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
    logits=outputs.logits.cpu().contiguous();boxes=outputs.pred_boxes.cpu().contiguous()
    if torch.isnan(logits).any() or not torch.isfinite(boxes).all():raise ValueError('invalid model tensors')
    raw_out={'logits':logits,'pred_boxes':boxes,'input_ids':inputs['input_ids'].cpu().contiguous(),'attention_mask':inputs['attention_mask'].cpu().contiguous()}
    save_file(raw_out,str(args.output_dir/'NEW_RAW_OUTPUT.safetensors'))
    save_file({k:v.cpu().contiguous() for k,v in inputs.items()},str(args.output_dir/'NEW_PROCESSED_INPUT.safetensors'))
    prior=load_file(str(args.historical_raw));finite=torch.isfinite(prior['logits'])&torch.isfinite(logits)
    old_rows=original['model_boxes_before_nms'];new_ids=torch.nonzero(logits.sigmoid().max(dim=-1).values[0]>.25).flatten().tolist()
    old_ids=[x['query_index'] for x in old_rows]
    new_boxes=result['boxes'].cpu().tolist();new_scores=result['scores'].cpu().tolist()
    same_ids=new_ids==old_ids
    score_error=max((abs(a-b['score']) for a,b in zip(new_scores,old_rows)),default=0.) if same_ids else None
    box_error=max((max(abs(a-b) for a,b in zip(x,y['bbox_xyxy_original_unclipped'])) for x,y in zip(new_boxes,old_rows)),default=0.) if same_ids else None
    comparison={'input_ids_equal':torch.equal(inputs['input_ids'].cpu(),prior['input_ids']),'processed_shape_matches_historical':list(inputs['pixel_values'].shape)==original['processed_pixel_values_shape'],'candidate_ids_equal':same_ids,'new_candidate_ids':new_ids,'historical_candidate_ids':old_ids,'scores_max_absolute_error':score_error,'boxes_max_absolute_error_pixels':box_error,'finite_mask_equal':torch.equal(torch.isfinite(logits),torch.isfinite(prior['logits'])),'finite_logits_max_absolute_error':float((logits[finite]-prior['logits'][finite]).abs().max()),'raw_boxes_max_absolute_error':float((boxes-prior['pred_boxes']).abs().max()),'raw_logits_equal':torch.equal(logits,prior['logits']),'raw_boxes_equal':torch.equal(boxes,prior['pred_boxes'])}
    receipt={'status':'new_forward_completed','frame_index':original['frame_index'],'source_sha256':sha(args.original),'model_id':env['configuration']['model_id'],'model_revision':env['configuration']['model_revision'],'weight_sha256':env['weight_sha256'],'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','safetensors','tokenizers','huggingface-hub','numpy','Pillow']},'device':'cpu','dtype':'float32','torch_threads':4,'torch_interop_threads':1,'prompt':'person.','box_threshold':.25,'text_threshold':.25,'full_original_dimensions':[image.width,image.height],'processed_pixel_values_shape':list(inputs['pixel_values'].shape),'logits_shape':list(logits.shape),'pred_boxes_shape':list(boxes.shape),'new_forward_passes':1,'new_training_runs':0,'new_model_downloads':0,'network_model_access':'offline/local_files_only/token=False/trust_remote_code=False','load_seconds':load_seconds,'preprocess_seconds':preprocess_seconds,'forward_seconds':forward_seconds,'thresholded_count':len(new_ids),'postprocessed_boxes':new_boxes,'postprocessed_scores':new_scores,'postprocessed_text_labels':result.get('text_labels',[]),'NMS':'not required for single candidate; no new NMS-removal claim','comparison_to_historical':comparison,'event_decision':'unknown','S_image_relation':'unknown','B_image_relation':'unknown','human_GT':False,'AI_annotation_status':'model_output_pending_human_review','trajectory':'unknown','direction':'unknown','accuracy_metrics':None,'raw_output_sha256':sha(args.output_dir/'NEW_RAW_OUTPUT.safetensors'),'processed_input_sha256':sha(args.output_dir/'NEW_PROCESSED_INPUT.safetensors')}
    (args.output_dir/'NEW_FORWARD_RECEIPT_PRIVATE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','frame_index','versions','forward_seconds','thresholded_count','comparison_to_historical','new_forward_passes','S_image_relation','B_image_relation']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir','original','historical-raw','historical-receipt','historical-model-env','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
