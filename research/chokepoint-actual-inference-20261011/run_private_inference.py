"""Four-frame frozen person-prompt development inference; all outputs private.

No network, remote code, credentials, tracking, ROI masks or accuracy evaluation.
Use an authorised local safetensors snapshot and original full-frame JPEGs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite_json(value):
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')
    return value


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - inter
    return inter / union if union else 0.0


def classwise_nms(rows, threshold):
    kept, decisions = [], []
    for row in sorted(rows, key=lambda r: (-r['score'], r['prediction_id'])):
        blocker = next((r for r in kept if r['label'] == row['label'] and iou(r['bbox_xyxy_original_unclipped'], row['bbox_xyxy_original_unclipped']) > threshold), None)
        decisions.append({'prediction_id': row['prediction_id'], 'kept': blocker is None,
                          'suppressed_by': None if blocker is None else blocker['prediction_id']})
        if blocker is None:
            kept.append(row)
    return kept, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Empty dedicated local cache, offline loading; never use a credential store.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', HF_HOME=str(args.output/'empty-hf-cache'))
    started = time.perf_counter()
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from safetensors.torch import save_file
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import_elapsed = time.perf_counter() - started
    configuration = json.loads(args.configuration.read_text())['detector_parameters']
    assert configuration['model_id'] == 'IDEA-Research/grounding-dino-tiny'
    assert configuration['prompt'] == 'person.'
    assert configuration['box_threshold'] == configuration['text_threshold'] == .25
    assert configuration['classwise_nms_iou'] == .7
    weight_hash = digest(args.snapshot/'model.safetensors')
    assert weight_hash == '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
    torch.set_num_threads(configuration['torch_threads'])
    torch.set_num_interop_threads(configuration['torch_interop_threads'])
    torch.manual_seed(configuration['seed'])
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.snapshot), local_files_only=True,
        trust_remote_code=False, use_fast=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(args.snapshot),
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32).to('cpu').eval()
    load_seconds = time.perf_counter() - started
    assert processor.image_processor.size == configuration['processor_size']
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'safetensors', 'tokenizers', 'huggingface-hub', 'Pillow', 'numpy')}
    model_meta = {'configuration': configuration, 'actual_versions': versions,
                  'python': platform.python_version(), 'torchvision_used': False,
                  'backend': 'PIL slow image processor; standalone frozen-convention NMS',
                  'weight_sha256': weight_hash, 'model_snapshot_files':
                  [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                   for p in sorted(args.snapshot.iterdir()) if p.is_file()],
                  'import_seconds': import_elapsed, 'processor_and_model_load_seconds': load_seconds,
                  'hardware': {'device': 'cpu', 'cuda_available': torch.cuda.is_available(),
                               'torch_threads': torch.get_num_threads(),
                               'interop_threads': torch.get_num_interop_threads()},
                  'warmup_calls': 0, 'threshold_tuning': False,
                  'door_detection_executed': False, 'accuracy_metrics': None}
    dump(args.output/'MODEL_AND_ENVIRONMENT.json', model_meta)
    summary = []
    completed_forwards = 0
    for index in (0, 142, 262, 312):
        path = args.frames/f'{index:08d}.jpg'
        row = {'frame_index': index, 'original_filename': path.name,
               'source_sha256': digest(path), 'status': 'started',
               'event_decision': 'unknown', 'identity_verified': False}
        total_start = time.perf_counter()
        try:
            prep_start = time.perf_counter()
            image = Image.open(path).convert('RGB')
            assert image.size == (800, 600)
            inputs = processor(images=image, text=configuration['prompt'], return_tensors='pt')
            preprocess_seconds = time.perf_counter()-prep_start
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            forward_seconds = time.perf_counter()-start
            timing_phase = 'first_forward_no_warmup' if completed_forwards == 0 else 'subsequent_forward_no_repeat_benchmark'
            completed_forwards += 1
            tensors = {'logits': outputs.logits.detach().contiguous().cpu(),
                       'pred_boxes': outputs.pred_boxes.detach().contiguous().cpu(),
                       'input_ids': inputs.input_ids.contiguous().cpu(),
                       'attention_mask': inputs.attention_mask.contiguous().cpu()}
            save_file(tensors, str(args.output/f'{index:08d}-raw.safetensors'))
            dump(args.output/f'{index:08d}-raw.json', {
                'frame_index': index, 'source_sha256': row['source_sha256'],
                'raw_model_tensor_shapes': {k:list(v.shape) for k,v in tensors.items()},
                'nonfinite_encoding': 'JSON strings nan/+inf/-inf; safetensors retains exact float32 bits',
                'logits': finite_json(tensors['logits'].tolist()),
                'pred_boxes_normalized_cxcywh': finite_json(tensors['pred_boxes'].tolist()),
                'input_ids': tensors['input_ids'].tolist(),
                'attention_mask': tensors['attention_mask'].tolist()})
            post_start = time.perf_counter()
            result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
                threshold=configuration['box_threshold'], text_threshold=configuration['text_threshold'],
                target_sizes=[(image.height,image.width)])[0]
            scores = outputs.logits.sigmoid().max(dim=-1).values[0]
            selected_queries = torch.nonzero(scores > configuration['box_threshold']).flatten().tolist()
            labels = result['text_labels'] if 'text_labels' in result else result['labels']
            assert len(selected_queries) == len(result['boxes'])
            boxes = []
            for q, coords, score, label in zip(selected_queries,result['boxes'].tolist(),result['scores'].tolist(),labels):
                boxes.append({'prediction_id':f'q{q:03d}', 'query_index':q,
                              'bbox_xyxy_original_unclipped':coords,'score':score,'label':label})
            kept, decisions = classwise_nms(boxes, configuration['classwise_nms_iou'])
            row.update(status='completed', original_dimensions=list(image.size),
                processed_pixel_values_shape=list(inputs.pixel_values.shape),
                tensor_shapes={k:list(v.shape) for k,v in tensors.items()},
                timing_phase=timing_phase, preprocess_seconds=preprocess_seconds,
                forward_seconds=forward_seconds, postprocess_seconds=time.perf_counter()-post_start,
                model_boxes_before_nms=boxes, model_boxes_after_nms=kept,
                nms_decisions=decisions, raw_query_count=int(outputs.pred_boxes.shape[1]),
                thresholded_count=len(boxes), kept_count=len(kept), corrections_applied=False)
            # Output derivative only: original JPEG bytes are never overwritten.
            overlay=image.copy();draw=ImageDraw.Draw(overlay)
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
            for box in kept:
                b=box['bbox_xyxy_original_unclipped']
                draw.rectangle(b,outline='#22ff77',width=3)
                label=f"MODEL {box['label']} {box['score']:.3f} {box['prediction_id']}"
                at=(max(0,b[0]),max(22,b[1]-19))
                draw.text(at,label,font=font,fill='#22ff77',stroke_width=1,stroke_fill='black')
            draw.rectangle((0,0,800,23),fill='#101a27')
            draw.text((5,3),'Actual frozen model output | person. | conf/text 0.25 | NMS 0.70 | no AI box correction',font=font,fill='white')
            overlay.save(args.output/f'{index:08d}-model-only.png')
            row['total_image_seconds_including_raw_serialization_and_overlay']=time.perf_counter()-total_start
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__,error_message=str(error),
                       traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-total_start)
        dump(args.output/f'{index:08d}-receipt.json',row)
        summary.append(row)
        dump(args.output/'ALL_FRAME_RECEIPTS.json',summary)
        print(json.dumps({'frame_index':index,'status':row['status'],
                          'kept_count':row.get('kept_count'),'forward_seconds':row.get('forward_seconds'),
                          'error':row.get('error_message')},ensure_ascii=False),flush=True)
    assert len(summary)==4
    if any(row['status']!='completed' for row in summary):
        raise SystemExit('One or more frames failed; all failure receipts retained.')


if __name__ == '__main__':
    main()
