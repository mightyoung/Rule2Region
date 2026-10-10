"""Replay public fixtures only; provenance hashes bind these public JSON bytes."""
from pathlib import Path
import hashlib
import json
import sys
from copy import deepcopy
from replay import run_pass, summarize_arm, ARMS, AUDIT
HERE = Path(__file__).resolve().parent

def read(name):
    return json.loads((HERE / name).read_text())

def digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()

def calculate():
    bbox_hash = digest('bbox_fixture.json')
    geometry_hash = digest('geometry_fixture.json')
    relation_hash = digest('relation_fixture.json')
    frames = read('bbox_fixture.json')
    for frame in frames:
        frame.update(viewRevision='public-geometry-v1', source_sha256=bbox_hash,
                     model={'name': 'frozen_detector_public_bbox_fixture',
                            'provenance_kind': 'public_fixture_file_not_original_image'},
                     rgb_pointer=None)
    geometry = read('geometry_fixture.json')
    geometry['portal_source_sha256'] = geometry_hash
    geometry['sill']['source_sha256'] = geometry_hash
    geometry['sill']['source_frame_sha256'] = bbox_hash
    relations = read('relation_fixture.json')
    accepted = {tuple(pair) for pair in relations['accepted_hypothesis_pairs']}
    manifest = {'associations': {'sha256': relation_hash}}
    cache0, descriptors0, checkpoints0, _, _ = run_pass(
        frames, relations, geometry, accepted, manifest, False)
    cache, descriptors, checkpoints, outputs, _ = run_pass(
        frames, relations, geometry, accepted, manifest, True)
    assert descriptors0 == descriptors
    assert cache0.stats['retrievals'] == 0
    assert cache.stats['retrievals'] == 62
    assert cache.stats['computed_geometry'] == 62
    assert cache.stats['computed_motion'] == 1
    assert cache.stats['stored_relation'] == 531
    assert not AUDIT
    rows = []
    for output in outputs:
        rows.append({'observation_id': output['request']['target_id'],
                     'pts': output['request']['pts'],
                     'arms': {arm: {key: view[key] for key in (
                         'event_status', 'direction_segment_available',
                         'direction_continuity_available', 'three_point_distance_available',
                         'missing_inputs')} for arm, view in output['arms'].items()}})
    return {
        'public_descriptors.json': list(descriptors.values()),
        'public_four_arm_results.json': rows,
        'public_summary.json': {
            'provenance_kind': 'hashes_bind_public_fixture_files_not_original_images',
            'frames': len(frames), 'observations': 62, 'candidate_relations': 531,
            'accepted_hypothesis_relations': 1, 'unknown_relations': 530,
            'computed_geometry_descriptors': 62, 'computed_motion_descriptors': 1,
            'no_request_retrievals': 0, 'forced_diagnostic_requests': 62,
            'no_read_recomputation': descriptors0 == descriptors,
            'arms': {arm: summarize_arm(outputs, arm) for arm in ARMS},
            'accuracy': None, 'new_model_calls': 0,
            'source_clock_simulation_not_wall_clock': True,
            'original_pixel_provenance_mapping_published': False,
        },
    }

if __name__ == '__main__':
    actual = calculate()
    if sys.argv[1:] == ['--generate']:
        for name, value in actual.items():
            (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                               allow_nan=False) + '\n')
    else:
        assert not sys.argv[1:], 'Only optional --generate is supported'
        for name, value in actual.items():
            assert value == read(name), name + ' differs from public fixture replay'
    print(json.dumps({'status': 'passed', 'observations': 62, 'geometry_records': 62,
                      'motion_records': 1, 'relations': 531, 'new_model_calls': 0,
                      'scope': 'public_fixture_replay_only'}, indent=2))
