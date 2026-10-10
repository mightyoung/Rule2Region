"""Portable replay primitives; source-directory archive loader omitted."""
import sys
from pathlib import Path
import json
import hashlib
from fractions import Fraction
from collections import Counter
from datetime import datetime, timezone
from continuous_cache import ContinuousCache
from ablation import ARMS, arm_view
AUDIT = Counter()

def runtime_guard(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.spawn', 'os.exec')):
        AUDIT[event] += 1
        raise RuntimeError('External execution/network forbidden in this replay: ' + event)
    if event == 'import' and args and (str(args[0]).split('.')[0] in ('torch', 'transformers', 'ultralytics', 'requests', 'urllib', 'http', 'onnxruntime')):
        AUDIT['forbidden_model_or_network_import'] += 1
        raise RuntimeError('No model/network imports in replay')
sys.addaudithook(runtime_guard)

def run_pass(frames, associations, geometry, accepted, manifest, do_requests):
    endpoints = {d['id']: {'time': str(time_of_frame(f)), 'source_sha256': f['source_sha256']} for f in frames for d in f['detections']}
    transitions = {t['sample_id']: t for t in associations['transitions']}
    derived = {}
    checkpoints = []
    outputs = []
    events = []
    c = ContinuousCache(event_sink=events.append)
    for f in frames:
        k = (f['camera'], f['viewRevision'])
        c.register(*k, geometry)
        now = time_of_frame(f)
        c.ingest(f, now)
        transition = transitions.get(f"G420-P{f['pts']}")
        if transition:
            for p in transition['all_pairs']:
                a, b = (p['previous_prediction_id'], p['prediction_id'])
                c.add_relation(*k, a, b, 'accepted_hypothesis' if (a, b) in accepted else 'unknown', manifest['associations']['sha256'], now, endpoints=[endpoints[a], endpoints[b]])
        for did, d in c.descriptors.items():
            derived.setdefault(did, c.public(d))
        before = deepcopy_public(c.descriptors)
        stats_before = dict(c.stats)
        assert c.retrieve(None, now) is None and c.descriptors == before and (dict(c.stats) == stats_before)
        if do_requests:
            for d in f['detections']:
                q = {'camera': k[0], 'viewRevision': k[1], 'pts': f['pts'], 'time_base': f['time_base'], 'target_id': d['id'], 'ambiguity_reason': 'offline_diagnostic_forced_all_targets_not_predicted_ambiguity'}
                payload = c.retrieve(q, now)
                views = {arm: arm_view(payload, arm) for arm in ARMS}
                outputs.append({'request': q, 'label': 'caller-authored diagnostic trigger, not actual uncertainty prediction', 'cache_result': payload, 'arms': views})
        assert c.descriptors == before
        checkpoints.append({'pts': f['pts'], 'source_seconds': str(now), 'observations': len(c.obs), 'relations': len(c.rels), 'descriptors': len(c.descriptors), 'geometry_descriptors': sum((d['kind'] == 'geometry' for d in c.descriptors.values())), 'motion_descriptors': sum((d['kind'] == 'motion' for d in c.descriptors.values())), 'stats': dict(c.stats)})
    return (c, derived, checkpoints, outputs, events)

def deepcopy_public(x):
    from copy import deepcopy
    return deepcopy(x)

def time_of_frame(f):
    return Fraction(f['pts']) * Fraction(f['time_base'])

def summarize_arm(outputs, arm):
    views = [r['arms'][arm] for r in outputs]
    missing = Counter((x for v in views for x in v['missing_inputs']))
    return {'requests': len(views), 'valid_current_geometry': sum((v['current_geometry'] is not None and v['current_geometry']['geometry_status'] == 'valid' for v in views)), 'direction_segment_available': sum((v['direction_segment_available'] for v in views)), 'historical_distance_sequence_available': sum((bool(v['historical_distance_sequence']) for v in views)), 'direction_continuity_available': sum((v['direction_continuity_available'] for v in views)), 'three_point_distance_available': sum((v['three_point_distance_available'] for v in views)), 'event_unknown': sum((v['event_status'] == 'unknown' for v in views)), 'inference_missing_inputs_counts': dict(missing), 'evaluation_gaps': ['independent_event_reference', 'real_negative_events']}
