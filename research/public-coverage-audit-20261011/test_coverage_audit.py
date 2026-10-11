"""Controlled corruptions of published fixtures; not real event negatives."""
from copy import deepcopy
import unittest
from coverage_audit import calculate, load_inputs

class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.inputs = load_inputs()

    def test_public_inventory_covers_exact_old_and_new_observations(self):
        r = calculate(**self.inputs)
        self.assertEqual(r['summary']['observations'], 83)
        self.assertEqual(r['summary']['old_fixture_matches'], 36)
        self.assertEqual(r['summary']['five_detector_record_matches'], 47)
        self.assertEqual(r['summary']['raw_threshold_candidates'], 50)
        self.assertEqual(r['summary']['nms_suppressed'], 3)
        self.assertEqual(r['summary']['nms_kept'], 47)
        self.assertEqual(r['summary']['unmatched_observations'], 0)
        self.assertEqual(r['summary']['confirmed_real_negative_events'], 0)
        self.assertIsNone(r['summary']['accuracy'])

    def test_new_bbox_cannot_be_rebound_to_same_id(self):
        self.inputs['inventory']['frames'][1]['detections'][0]['bbox_xyxy'][0] += 1
        with self.assertRaisesRegex(ValueError, 'inventory_record_mismatch'):
            calculate(**self.inputs)

    def test_new_confidence_cannot_be_rebound_to_same_id(self):
        self.inputs['inventory']['frames'][1]['detections'][0]['confidence'] += 0.01
        with self.assertRaisesRegex(ValueError, 'inventory_record_mismatch'):
            calculate(**self.inputs)

    def test_equivalent_timebase_representation_does_not_bind_original(self):
        self.inputs['inventory']['frames'][1]['time_base'] = '2/60'
        with self.assertRaisesRegex(ValueError, 'inventory_record_mismatch'):
            calculate(**self.inputs)

    def test_changed_frozen_pts_rejected(self):
        self.inputs['inventory']['frames'][0]['pts'] += 1
        with self.assertRaises(ValueError):
            calculate(**self.inputs)

    def test_duplicate_observation_rejected(self):
        f = self.inputs['inventory']['frames'][1]
        f['detections'].append(deepcopy(f['detections'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            calculate(**self.inputs)

    def test_suppressed_box_does_not_enter_kept_inventory(self):
        record = self.inputs['detector_records'][0]
        suppressed = record['nms_suppressed'][0]['prediction_id']
        raw = next(d for d in record['raw_above_threshold'] if d['prediction_id'] == suppressed)
        f = next(f for f in self.inputs['inventory']['frames'] if f['sample_id'] == record['sample_id'])
        f['detections'].append({k:raw[k] for k in ('prediction_id', 'bbox_xyxy', 'confidence')})
        with self.assertRaisesRegex(ValueError, 'inventory_record_mismatch'):
            calculate(**self.inputs)

    def test_false_nms_suppressor_rejected(self):
        self.inputs['detector_records'][0]['nms_suppressed'][0]['suppressed_by'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'nms_suppression_mismatch'):
            calculate(**self.inputs)

    def test_missing_forward_record_rejected(self):
        self.inputs['detector_records'].pop()
        with self.assertRaises(ValueError):
            calculate(**self.inputs)

    def test_old_frame_is_not_new_forward(self):
        self.inputs['inventory']['frames'][0]['prediction_provenance'] = 'new_authorized_five_forward_batch'
        with self.assertRaises(ValueError):
            calculate(**self.inputs)

    def test_changed_nms_threshold_fails_historical_consistency(self):
        self.inputs['configuration']['detector_parameters']['classwise_nms_iou'] = 1.0
        with self.assertRaises(ValueError):
            calculate(**self.inputs)

    def test_no_mutation_or_model_calls(self):
        before = deepcopy(self.inputs)
        r = calculate(**self.inputs)
        self.assertEqual(self.inputs, before)
        self.assertEqual(r['summary']['new_model_calls'], 0)
        self.assertEqual(r['summary']['composite_queries'], 0)
        self.assertEqual(r['summary']['event_unknown'], 83)
        self.assertTrue(all(f['trajectory_status'] == 'unknown' for f in r['frames']))

if __name__ == '__main__':
    unittest.main()
