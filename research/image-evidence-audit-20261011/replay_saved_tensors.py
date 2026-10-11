"""Replay four already-authorized saved detector tensors, without a model forward.
Requires existing NumPy. Inputs are private authorized files, not bundled here.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

FRAMES=('00000000','00000142','00000262','00000312')

def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

def tensors(path):
    data=path.read_bytes()
    if len(data)<8:raise ValueError('truncated tensor file')
    size=struct.unpack('<Q',data[:8])[0]
    if not 0<size<=1048576 or 8+size>len(data):raise ValueError('invalid tensor header')
    header=json.loads(data[8:8+size]);base=8+size;result={}
    for name,shape in [('logits',[1,900,256]),('pred_boxes',[1,900,4])]:
        entry=header[name];a,z=entry['data_offsets']
        if entry['dtype']!='F32' or entry['shape']!=shape or not 0<=a<=z<=len(data)-base or z-a!=np.prod(shape)*4:
            raise ValueError('unexpected tensor contract')
        result[name]=np.frombuffer(data[base+a:base+z],dtype='<f4').reshape(shape)
    return result

def replay(original_dir,cache_dir):
    records=[]
    for frame in FRAMES:
        receipt=json.loads((cache_dir/(frame+'-receipt.json')).read_text())
        if file_sha(original_dir/(frame+'.jpg'))!=receipt['source_sha256']:raise ValueError('original source binding mismatch')
        raw=tensors(cache_dir/(frame+'-raw.safetensors'))
        scores=np.exp(-np.logaddexp(np.float32(0),-raw['logits'][0])).max(axis=-1)
        indices=np.flatnonzero(scores>np.float32(.25));expected=receipt['model_boxes_before_nms']
        if list(indices)!=[row['query_index'] for row in expected]:raise ValueError('threshold candidate mismatch')
        wh=np.array([800,600,800,600],dtype=np.float32);score_error=box_error=0.
        for index,row in zip(indices,expected):
            cx,cy,w,h=raw['pred_boxes'][0,index]
            xy=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2],dtype=np.float32)*wh
            score_error=max(score_error,abs(float(scores[index])-row['score']))
            box_error=max(box_error,float(np.max(np.abs(xy-np.array(row['bbox_xyxy_original_unclipped'])))))
        if score_error>=1e-6 or box_error>=1e-3:raise ValueError('cached score/coordinate mismatch')
        # These four cases have no thresholded box removed. No new NMS branch is claimed.
        if [row['query_index'] for row in receipt['model_boxes_after_nms']]!=list(indices):raise ValueError('unsupported NMS-removal case')
        if receipt['event_decision']!='unknown' or receipt['identity_verified'] is not False:raise ValueError('unexpected historical status')
        records.append({'frame_index':int(frame),'source_byte_binding':'match','queries':900,'original_threshold':.25,'cached_pre_nms_count':len(indices),'cached_kept_count':receipt['kept_count'],'max_score_absolute_error':score_error,'max_box_absolute_error_pixels':box_error,'event_decision':'unknown','identity_verified':False,'trajectory':'unknown','direction':'unknown'})
    return {'status':'match_saved_model_receipts','method':'Read saved F32 tensor bytes, sigmoid max and existing threshold, transform normalized boxes; compare with historical receipts. No new forward/NMS/model inference.','records':records,'new_forward_passes':0,'new_training_runs':0,'GT_accuracy_claim':None}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir',type=Path,required=True);parser.add_argument('--cache-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=replay(args.original_dir,args.cache_dir);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
