"""Geometry/schema and source-axis admission only; no media/model/event inference.

Inputs remain private. Stdlib code imports the unchanged frozen geometry cache.
Usage: python check_private_inputs.py PROPOSALS_JSON C1_AXIS_JSON
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'continuous-cache-20261010/cache'))
from continuous_cache import ContinuousCache, content_id
from vendor import roi_core


def check(proposals, axis):
    validated = 0
    for row in proposals:
        width, height = row['dimensions']
        for key in ('candidate_far_door_polygon', 'candidate_near_frame_polygon'):
            points = row[key]
            result = roi_core.polygon_validation(points, width, height)
            if not result['valid']:
                raise ValueError(result['reason'])
            # '0' is a schema probe clock, never a recovered native source time.
            ContinuousCache.validate_geometry({
                'width': width, 'height': height,
                'portal_source_sha256': content_id(points), 'available_at': '0',
                'portals': [{'reference_id': 'illustration', 'points': points}],
                'sill': None})
            validated += 1
    members = {r['frame_index'] for r in axis['members']}
    pairs = []
    for a, b in ((98, 99), (99, 112), (112, 122), (142, 262), (262, 312)):
        if a not in members or b not in members:
            raise ValueError('endpoint_absent_from_original_member_axis')
        missing = sum(i not in members for i in range(a + 1, b))
        reasons = []
        if missing:
            reasons.append('source_axis_gap')
        if b != a + 1:
            reasons.append('source_endpoints_not_adjacent')
        # Adjacency alone cannot certify identity or recover source timestamps.
        reasons += ['native_timestamps_unavailable', 'no_verified_identity_association']
        pairs.append({'from_index': a, 'to_index': b,
                      'missing_source_members_count': missing,
                      'source_adjacent': b == a + 1 and not missing,
                      'continuous_motion_eligible': False, 'reasons': reasons,
                      'event_decision': 'unknown'})
    return {'geometry_schema_checks_passed': validated,
            'physical_boundary_verified': False, 'pairs': pairs,
            'continuous_motion_eligible_count': 0,
            'identity_verified': False, 'native_timestamps_recovered': False,
            'recognition_metrics': None, 'new_model_calls': 0}


if __name__ == '__main__':
    proposals = json.loads(Path(sys.argv[1]).read_text())
    axis = json.loads(Path(sys.argv[2]).read_text())
    print(json.dumps(check(proposals, axis), indent=2))
