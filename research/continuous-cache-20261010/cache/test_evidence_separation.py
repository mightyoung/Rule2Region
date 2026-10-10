import unittest
from continuous_cache import ContinuousCache
from ablation import arm_view
from test_continuous_cache import K,geometry,frame,request
class Tests(unittest.TestCase):
 def test_evaluation_reference_never_an_inference_requirement(self):
  c=ContinuousCache();c.register(*K,geometry());c.ingest(frame(),0);r=c.retrieve(request(),0)
  self.assertNotIn('independent_event_reference',r['missing_inputs'])
  self.assertIn('automatic_side_evidence',r['inference_missing_inputs'])
  self.assertIn('independent_event_reference',r['evaluation_gaps'])
  v=arm_view(r,'G');self.assertEqual(v['evaluation_gaps'],r['evaluation_gaps']);self.assertNotIn('independent_event_reference',v['inference_missing_inputs'])
