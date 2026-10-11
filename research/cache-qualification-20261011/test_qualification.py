"""Counterexamples to unsafe cache reuse, using public bytes only."""
import unittest
from copy import deepcopy
from qualification import calculate, load_inputs

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = load_inputs()

    def result(self):
        return calculate(**self.inputs)

    def test_public_population_and_sparse_chord_not_dense_continuity(self):
        r = self.result()
        self.assertEqual(r['summary']['observations'], 83)
        self.assertEqual(r['summary']['geometry_reuse_eligible'], 36)
        self.assertEqual(r['summary']['accepted_endpoint_hypotheses'], 1)
        self.assertEqual(r['summary']['accepted_dense_adjacent_links'], 0)
        motion = r['accepted_motion_records'][0]
        self.assertEqual(motion['duration_seconds'], '1')
        self.assertEqual(motion['intervening_pts'], [375])
        self.assertEqual(motion['event_status'], 'unknown')
        self.assertEqual(set(motion['anchors']), {'center', 'bottom_proxy'})

    def test_same_id_at_changed_pts_fails(self):
        self.inputs['inventory']['frames'][0]['pts'] += 1
        self.assertEqual(self.result()['summary']['accepted_endpoint_hypotheses'], 0)

    def test_equivalent_rational_time_is_not_original_timebase_binding(self):
        self.inputs['inventory']['frames'][0]['time_base'] = '2/60'
        self.assertEqual(self.result()['summary']['accepted_endpoint_hypotheses'], 0)

    def test_same_id_changed_bbox_fails(self):
        f = self.inputs['inventory']['frames'][0]
        next(d for d in f['detections'] if d['prediction_id'].endswith(':query003'))['bbox_xyxy'][0] += 0.01
        self.assertEqual(self.result()['summary']['accepted_endpoint_hypotheses'], 0)

    def test_changed_prediction_source_fails(self):
        self.inputs['inventory']['frames'][0]['prediction_provenance'] = 'unverified_replacement'
        self.assertEqual(self.result()['summary']['accepted_endpoint_hypotheses'], 0)

    def test_descriptor_source_cannot_rebind(self):
        for d in self.inputs['descriptors']:
            for e in d['endpoint_bindings']:
                e['source_sha256'] = '0' * 64
        r = self.result()
        self.assertEqual(r['summary']['geometry_reuse_eligible'], 0)
        self.assertEqual(len(r['accepted_motion_records']), 0)

    def test_relation_source_cannot_rebind_motion(self):
        for d in self.inputs['descriptors']:
            if d['kind'] == 'motion':
                d['relation_source_sha256'] = '0' * 64
        self.assertEqual(self.result()['accepted_motion_records'], [])

    def test_geometry_revision_cannot_rebind(self):
        for d in self.inputs['descriptors']:
            if d['kind'] == 'geometry':
                d['geometry_source_sha256'] = '0' * 64
        self.assertEqual(self.result()['summary']['geometry_reuse_eligible'], 0)

    def test_future_motion_cannot_be_read_at_previous_endpoint(self):
        self.inputs['query_time'] = '12'
        r = self.result()
        self.assertEqual(r['summary']['motion_read_eligible'], 0)
        self.assertEqual(r['summary']['accepted_endpoint_hypotheses'], 1)

    def test_unknown_pairs_never_become_accepted(self):
        self.inputs['relations']['accepted_hypothesis_pairs'] = []
        r = self.result()
        self.assertEqual(r['summary']['accepted_endpoint_hypotheses'], 0)
        self.assertEqual(r['accepted_motion_records'], [])

    def test_duplicate_id_rejected(self):
        f = self.inputs['inventory']['frames'][0]
        f['detections'].append(deepcopy(f['detections'][0]))
        with self.assertRaises(ValueError):
            self.result()

    def test_duplicate_old_id_rejected(self):
        f = self.inputs['frames'][0]
        f['detections'].append(deepcopy(f['detections'][0]))
        with self.assertRaises(ValueError):
            self.result()

    def test_no_input_mutation_or_ai_reference_dependency(self):
        before = deepcopy(self.inputs)
        r = self.result()
        self.assertEqual(self.inputs, before)
        self.assertEqual(r['summary']['composite_queries'], 0)
        self.assertEqual(r['summary']['new_model_calls'], 0)
        self.assertIsNone(r['summary']['accuracy'])

if __name__ == '__main__':
    unittest.main()
