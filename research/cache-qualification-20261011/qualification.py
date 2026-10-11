"""Public-fixture cache eligibility audit; no identity or event inference."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / 'continuous-cache-20261010'
INPUTS = {
    'frames': 'cache/bbox_fixture.json',
    'inventory': 'detection/nine_frame_bbox_inventory.json',
    'relations': 'cache/relation_fixture.json',
    'descriptors': 'cache/public_descriptors.json',
}

def load_inputs():
    return {key: json.loads((PUBLIC / name).read_text()) for key, name in INPUTS.items()} | {
        'bbox_hash': hashlib.sha256((PUBLIC / INPUTS['frames']).read_bytes()).hexdigest(),
        'relation_hash': hashlib.sha256((PUBLIC / INPUTS['relations']).read_bytes()).hexdigest(),
        'geometry_hash': hashlib.sha256((PUBLIC / 'cache/geometry_fixture.json').read_bytes()).hexdigest(),
    }

from fractions import Fraction
from copy import deepcopy
import sys


def calculate(frames, inventory, relations, descriptors, bbox_hash, geometry_hash, relation_hash, query_time=None):
    # The inventory has no original-pixel hash. This join checks consistency
    # with its explicit frozen-prediction declaration, never source-image proof.
    old = {}
    for f in frames:
        for d in f['detections']:
            if d['id'] in old:
                raise ValueError('Duplicate old observation ID')
            old[d['id']] = (f, d)
    geometry = {}
    motion = {}
    for d in descriptors:
        es = d['endpoint_bindings']
        if d['kind'] == 'geometry':
            key = es[0]['observation_id']
            if key in geometry:
                raise ValueError('Duplicate geometry endpoint')
            geometry[key] = d
        elif d['kind'] == 'motion':
            key = tuple(e['observation_id'] for e in es)
            if key in motion:
                raise ValueError('Duplicate motion endpoints')
            motion[key] = d

    def bound(descriptor, ids):
        if descriptor is None or len(descriptor['endpoint_bindings']) != len(ids):
            return False
        for e, oid in zip(descriptor['endpoint_bindings'], ids):
            if oid not in old:
                return False
            f, _ = old[oid]
            if any(e.get(k) != value for k, value in {
                'observation_id': oid, 'camera': f['camera'], 'pts': f['pts'],
                'time_base': f['time_base'], 'source_sha256': bbox_hash,
                'viewRevision': 'public-geometry-v1',
            }.items()):
                return False
        if descriptor['kind'] == 'geometry':
            return descriptor.get('geometry_source_sha256') == geometry_hash
        return (descriptor.get('relation_source_sha256') == relation_hash
                and descriptor.get('association_status') == 'accepted_hypothesis')

    rows = []
    by_id = {}
    grid = []
    for f in inventory['frames']:
        t = Fraction(f['pts']) * Fraction(f['time_base'])
        grid.append(t)
        for d in f['detections']:
            oid = d['prediction_id']
            if oid in by_id:
                raise ValueError('Duplicate inventory observation ID')
            reasons = []
            if oid not in old:
                reasons.append('no_old_endpoint')
            else:
                previous_frame, previous_bbox = old[oid]
                # Bind camera through both the sample name and the ID prefix.
                if f['sample_id'] != f"{previous_frame['camera']}-P{f['pts']}" or not oid.startswith(f['sample_id'] + ':'):
                    reasons.append('sample_or_camera_mismatch')
                if f['pts'] != previous_frame['pts']:
                    reasons.append('pts_mismatch')
                if f['time_base'] != previous_frame['time_base']:
                    reasons.append('original_timebase_mismatch')
                if d['bbox_xyxy'] != previous_bbox['bbox']:
                    reasons.append('bbox_mismatch')
                if f['prediction_provenance'] != 'existing_frozen_prediction':
                    reasons.append('frozen_source_declaration_mismatch')
            qualified = not reasons
            gd = geometry.get(oid)
            geometry_ok = qualified and bound(gd, [oid])
            if qualified and not geometry_ok:
                reasons.append('geometry_source_binding_mismatch')
            now = t if query_time is None else Fraction(query_time)
            readable = geometry_ok and Fraction(gd['available_at']) <= now and t <= now
            row = {
                'observation_id': oid, 'pts': f['pts'], 'time_base': f['time_base'],
                'source_seconds': str(t), 'prediction_provenance': f['prediction_provenance'],
                'endpoint_qualified': qualified, 'geometry_reuse_eligible': geometry_ok,
                'geometry_read_eligible': readable, 'rejection_reasons': reasons,
                'trajectory_status': 'unknown', 'direction_status': 'unknown',
                'side_evidence': 'unknown', 'event_status': 'unknown',
                'original_pixel_source_verified': False,
            }
            rows.append(row)
            by_id[oid] = row
    if grid != sorted(set(grid)):
        raise ValueError('Inventory frame times must be unique and increasing')
    accepted = {tuple(pair) for pair in relations['accepted_hypothesis_pairs']}
    relation_rows = []
    accepted_motion = []
    for transition in relations['transitions']:
        for p in transition['all_pairs']:
            a, b = p['previous_prediction_id'], p['prediction_id']
            pair = (a, b)
            endpoint_ok = all(oid in by_id and by_id[oid]['endpoint_qualified'] for oid in pair)
            hypothesis = endpoint_ok and pair in accepted
            intervening = []
            adjacent = False
            if endpoint_ok:
                start, end = (Fraction(by_id[oid]['source_seconds']) for oid in pair)
                intervening = [f['pts'] for f in inventory['frames']
                               if start < Fraction(f['pts']) * Fraction(f['time_base']) < end]
                adjacent = start < end and not intervening
            md = motion.get(pair)
            motion_ok = hypothesis and bound(md, pair)
            now = Fraction(by_id[b]['source_seconds']) if query_time is None and endpoint_ok else Fraction(query_time or 0)
            readable = motion_ok and Fraction(md['available_at']) <= now and Fraction(by_id[b]['source_seconds']) <= now
            relation_rows.append({
                'previous_id': a, 'current_id': b, 'endpoint_qualified': endpoint_ok,
                'relation_status': 'accepted_hypothesis' if hypothesis else 'unknown',
                'dense_adjacent': adjacent, 'motion_read_eligible': readable,
                'intervening_pts': intervening, 'identity_verified': False,
                'event_status': 'unknown',
            })
            if motion_ok:
                record = {
                    'previous_id': a, 'current_id': b,
                    'duration_seconds': str(end - start), 'intervening_pts': intervening,
                    'dense_adjacent': adjacent, 'motion_read_eligible': readable,
                    'anchors': deepcopy(md['anchors']),
                    'trajectory_status': 'sparse_two_endpoint_engineering_hypothesis',
                    'direction_status': 'bbox_chord_only_not_body_direction',
                    'direction_continuity_available': False, 'event_status': 'unknown',
                    'identity_verified': False,
                }
                accepted_motion.append(record)
    summary = {
        'frames': len(inventory['frames']), 'observations': len(rows),
        'geometry_reuse_eligible': sum(r['geometry_reuse_eligible'] for r in rows),
        'geometry_read_eligible': sum(r['geometry_read_eligible'] for r in rows),
        'observations_without_old_endpoint': sum('no_old_endpoint' in r['rejection_reasons'] for r in rows),
        'old_relation_candidates': len(relation_rows),
        'endpoint_qualified_relation_candidates': sum(r['endpoint_qualified'] for r in relation_rows),
        'accepted_endpoint_hypotheses': sum(r['relation_status'] == 'accepted_hypothesis' for r in relation_rows),
        'accepted_dense_adjacent_links': sum(r['relation_status'] == 'accepted_hypothesis' and r['dense_adjacent'] for r in relation_rows),
        'motion_read_eligible': sum(r['motion_read_eligible'] for r in relation_rows),
        'event_unknown': len(rows), 'direction_continuity_available': 0,
        'new_model_calls': 0, 'composite_queries': 0, 'accuracy': None,
    }
    return {
        'scope': 'public_fixture_consistency_and_offline_read_qualification_only',
        'provenance_limit': 'JSON hashes and frozen-source declaration; no original-pixel binding is public',
        'query_policy': 'No ambiguous business targets supplied; no composite queries. Unconditional trajectory/direction recording only.',
        'query_time': query_time, 'summary': summary,
        'observations': rows, 'relations': relation_rows, 'accepted_motion_records': accepted_motion,
        'missing_evidence': ['original_pixels_and_provenance_mapping', 'HSV_for_new_association',
                             'automatic_physical_side', 'human_person_geometry_event_GT',
                             'confirmed_parallel_pass_and_pre_crossing_return_negative_events'],
    }


if __name__ == '__main__':
    result = calculate(**load_inputs())
    if sys.argv[1:] == ['--generate']:
        (HERE / 'qualification_results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    else:
        if sys.argv[1:]:
            raise SystemExit('Only optional --generate is supported')
        if result != json.loads((HERE / 'qualification_results.json').read_text()):
            raise SystemExit('Qualification results differ')
    print(json.dumps(result['summary'], indent=2))
