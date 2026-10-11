"""Independent public inventory and classwise NMS audit; standard library only."""
from pathlib import Path
from fractions import Fraction
import json
import math
import sys

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / 'continuous-cache-20261010'


def load_inputs():
    def read(path):
        return json.loads(path.read_text())
    return {
        'inventory': read(PUBLIC / 'detection/nine_frame_bbox_inventory.json'),
        'old_frames': read(PUBLIC / 'cache/bbox_fixture.json'),
        'detector_records': [read(p) for p in sorted((PUBLIC / 'detection').glob('grounding-dino-tiny__*.json'))],
        'configuration': read(PUBLIC / 'detection/configuration.json'),
    }


def _unique(rows, key):
    by_id = {}
    for row in rows:
        oid = row[key]
        if oid in by_id:
            raise ValueError('duplicate:' + oid)
        by_id[oid] = row
    return by_id


def _iou(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    overlap = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    union = (x2-x1)*(y2-y1) + (u2-u1)*(v2-v1) - overlap
    return overlap / union


def calculate(inventory, old_frames, detector_records, configuration):
    params = configuration['detector_parameters']
    threshold = params['classwise_nms_iou']
    records = _unique(detector_records, 'sample_id')
    old = {f"{f['camera']}-P{f['pts']}": f for f in old_frames}
    nms_rows = []
    for record in detector_records:
        raw = _unique(record['raw_above_threshold'], 'prediction_id')
        kept = []
        suppressed = []
        for d in sorted(raw.values(), key=lambda d: (-d['confidence'], d['prediction_id'])):
            box = d['bbox_xyxy']
            if len(box) != 4 or not all(math.isfinite(x) for x in box) or not (box[2] > box[0] and box[3] > box[1]):
                raise ValueError('invalid_bbox')
            if not (params['box_threshold'] < d['confidence'] <= 1):
                raise ValueError('below_threshold_or_invalid_confidence')
            if d['canonical_class'] not in params['class_mapping'].values():
                raise ValueError('class_mapping_mismatch')
            suppressor = next((k for k in kept if k['canonical_class'] == d['canonical_class'] and _iou(k['bbox_xyxy'], box) > threshold), None)
            if suppressor is None:
                kept.append(d)
            else:
                suppressed.append({'prediction_id': d['prediction_id'],
                                   'suppressed_by': suppressor['prediction_id'],
                                   'iou': _iou(suppressor['bbox_xyxy'], box)})
        predicted = [{
            'prediction_id': d['prediction_id'], 'class': d['canonical_class'],
            'bbox_xyxy': d['bbox_xyxy'], 'confidence': d['confidence'],
        } for d in kept]
        if predicted != record['predictions']:
            raise ValueError('nms_kept_mismatch:' + record['sample_id'])
        expected = record['nms_suppressed']
        if len(expected) != len(suppressed) or any(
            a['prediction_id'] != b['prediction_id'] or a['suppressed_by'] != b['suppressed_by'] or
            not math.isclose(a['iou'], b['iou'], rel_tol=0, abs_tol=1e-12)
            for a, b in zip(expected, suppressed)
        ):
            raise ValueError('nms_suppression_mismatch:' + record['sample_id'])
        nms_rows.append({'sample_id': record['sample_id'], 'threshold_candidates':len(raw),
                         'kept':len(kept), 'suppressed':suppressed})
    frame_rows = []
    ids = set()
    times = []
    seen_records = set()
    for frame in inventory['frames']:
        sid = frame['sample_id']
        current = _unique(frame['detections'], 'prediction_id')
        if ids & current.keys():
            raise ValueError('duplicate_global_observation')
        ids.update(current)
        times.append(Fraction(frame['pts']) * Fraction(frame['time_base']))
        matched_old = matched_new = 0
        if frame['prediction_provenance'] == 'existing_frozen_prediction':
            previous = old.get(sid)
            if previous is None or (frame['pts'],frame['time_base']) != (previous['pts'],previous['time_base']):
                raise ValueError('inventory_record_mismatch:' + sid)
            source = _unique(previous['detections'], 'id')
            if source.keys() != current.keys() or any(current[oid]['bbox_xyxy'] != source[oid]['bbox'] for oid in source):
                raise ValueError('inventory_record_mismatch:' + sid)
            matched_old = len(current)
        elif frame['prediction_provenance'] == 'new_authorized_five_forward_batch':
            record = records.get(sid)
            if record is None or (frame['pts'],frame['time_base']) != (record['actual_pts'],record['time_base']):
                raise ValueError('inventory_record_mismatch:' + sid)
            source = _unique(record['predictions'], 'prediction_id')
            if source.keys() != current.keys() or any(
                any(current[oid][key] != source[oid][key] for key in ('bbox_xyxy','confidence')) for oid in source
            ):
                raise ValueError('inventory_record_mismatch:' + sid)
            matched_new = len(current)
            seen_records.add(sid)
        else:
            raise ValueError('inventory_source_unrecognized:' + sid)
        frame_rows.append({'sample_id':sid,'pts':frame['pts'],'time_base':frame['time_base'],
                           'observations':len(current),'old_matches':matched_old,'new_record_matches':matched_new,
                           'old_confidence_verifiable':False if matched_old else None,
                           'trajectory_status':'unknown','direction_continuity_status':'unknown',
                           'physical_side':'unknown','event_status':'unknown'})
    if seen_records != records.keys() or times != sorted(set(times)):
        raise ValueError('frame_coverage_or_time_grid_mismatch')
    summary = {
        'frames':len(frame_rows),'observations':len(ids),
        'old_fixture_matches':sum(f['old_matches'] for f in frame_rows),
        'five_detector_record_matches':sum(f['new_record_matches'] for f in frame_rows),
        'raw_threshold_candidates':sum(r['threshold_candidates'] for r in nms_rows),
        'nms_suppressed':sum(len(r['suppressed']) for r in nms_rows),
        'nms_kept':sum(r['kept'] for r in nms_rows),'unmatched_observations':0,
        'event_unknown':len(ids),'confirmed_real_negative_events':0,'accuracy':None,
        'new_model_calls':0,'composite_queries':0,
    }
    return {'scope':'consistency_of_already_public_inventory_and_NMS_records',
            'summary':summary,'frames':frame_rows,'nms':nms_rows,
            'limits':['No original pixels or model forwards verified',
                      'Old fixture omits confidence; 36 old confidence values cannot be cross-checked',
                      'Raw text-token scores absent; text-threshold filtering cannot be replayed',
                      'Synthetic corruptions are not independent real negative events',
                      'No association, physical side, human GT or recognition improvement inferred']}


if __name__ == '__main__':
    result = calculate(**load_inputs())
    if sys.argv[1:] == ['--generate']:
        (HERE/'coverage_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    elif sys.argv[1:]:
        raise SystemExit('Only optional --generate is supported')
    elif result != json.loads((HERE/'coverage_results.json').read_text()):
        raise SystemExit('Coverage results differ')
    print(json.dumps(result['summary'],indent=2))
