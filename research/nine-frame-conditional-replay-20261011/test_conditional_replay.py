"""Actual public observations, no AI identities or event reference labels."""
import unittest
import json
from conditional_replay import calculate

class ReplayTests(unittest.TestCase):
    def test_result_is_JSON_native_and_roundtrip_stable(self):
        r=calculate(False)
        self.assertEqual(r,json.loads(json.dumps(r)))

    def test_every_observation_computes_even_without_queries(self):
        r=calculate(False)
        self.assertEqual(r['summary']['observations'],83)
        self.assertEqual(r['summary']['computed_geometry'],83)
        self.assertEqual(r['summary']['retrievals'],0)
        self.assertEqual(r['summary']['computed_motion'],0)
        self.assertEqual(len(r['trace']),83)

    def test_denser_grid_rejects_original_sparse_motion(self):
        r=calculate(False)
        self.assertEqual(r['legacy_relation_at_ingest']['status'],'accepted_hypothesis')
        self.assertEqual(r['legacy_relation_at_ingest']['motion_status'],'unknown_motion')
        self.assertIn('nonadjacent_frames',r['legacy_relation_at_ingest']['motion_missing_inputs'])

    def test_only_geometrically_flagged_targets_read_composite_features(self):
        r=calculate(True)
        count=r['summary']['retrievals']
        self.assertGreater(count,0)
        self.assertLess(count,83)
        self.assertEqual(count,sum(x['review_flag'] for x in r['trace']))
        self.assertTrue(all(x['review_flag'] for x in r['reads']))
        self.assertTrue(all(x['event_status']=='unknown' for x in r['trace']))

    def test_reads_leave_computation_and_expiry_unchanged(self):
        a,b=calculate(False),calculate(True)
        self.assertEqual(a['trace'],b['trace'])
        self.assertEqual(a['checkpoints'],b['checkpoints'])
        self.assertEqual(a['summary']['computed_geometry'],b['summary']['computed_geometry'])
        self.assertEqual(a['summary']['computed_motion'],b['summary']['computed_motion'])
        self.assertTrue(b['summary']['no_read_mutation'])

    def test_bounds_and_expiry_really_apply_to_new_density(self):
        r=calculate(True)
        self.assertGreater(r['summary']['observation_capacity_evictions'],0)
        self.assertGreater(r['summary']['observation_expirations'],0)
        self.assertLessEqual(max(c['live_observations'] for c in r['checkpoints']),64)
        self.assertLessEqual(max(c['live_relations'] for c in r['checkpoints']),512)
        self.assertLessEqual(max(c['live_descriptors'] for c in r['checkpoints']),128)

    def test_no_verified_identity_side_or_direction_continuity(self):
        r=calculate(True)
        for read in r['reads']:
            self.assertFalse(read['identity_verified'])
            self.assertEqual(read['arms']['G_DIR_DIST']['event_status'],'unknown')
            self.assertFalse(read['arms']['G_DIR_DIST']['direction_continuity_available'])
        self.assertEqual(r['summary']['new_model_calls'],0)
        self.assertIsNone(r['summary']['accuracy'])

if __name__=='__main__':unittest.main()
