import unittest
from continuous_cache import ContinuousCache
from test_continuous_cache import K, geometry, frame
class Tests(unittest.TestCase):
 def test_tiny_sill_rejected_before_registration(self):
  c=ContinuousCache();g=geometry();g['sill']['points']=[[0,0],[1e-200,0]]
  with self.assertRaises(ValueError):c.register(*K,g)
  self.assertEqual(c.keys,{})
 def test_tiny_bbox_rejected_before_observation_mutation(self):
  c=ContinuousCache();c.register(*K,geometry())
  with self.assertRaises(ValueError):c.ingest(frame(bbox=[0,0,1e-200,1e-200]),0)
  self.assertEqual(c.obs,{});self.assertEqual(c.descriptors,{});self.assertEqual(c._frame_no[K],0)
 def test_tiny_portal_edge_rejected_before_registration(self):
  c=ContinuousCache();g=geometry();g['portals'][0]['points']=[[0,0],[1e-200,0],[30,30],[0,30]]
  with self.assertRaises(ValueError):c.register(*K,g)
  self.assertEqual(c.keys,{})
if __name__=='__main__':unittest.main(verbosity=2)
