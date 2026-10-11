import unittest
from continuous_cache import ContinuousCache
from test_continuous_cache import K, geometry, pair, request
try:
 from ablation import arm_view
except ImportError:
 def arm_view(payload,arm):return {}
class Tests(unittest.TestCase):
 def test_current_geometry_is_shared_and_historical_features_toggle(self):
  c=ContinuousCache();c.register(*K,geometry());pair(c);p=c.retrieve(request(1,'b'),1)
  views={a:arm_view(p,a) for a in ('G','G_DIR','G_DIST','G_DIR_DIST')}
  for a,v in views.items():self.assertIn('current_geometry',v)
  self.assertTrue(all(v['current_geometry']==views['G']['current_geometry'] for v in views.values()))
  for a,v in views.items():
   self.assertEqual(len(v['direction_segments']),1 if a in ('G_DIR','G_DIR_DIST') else 0)
   self.assertEqual(len(v['historical_distance_sequence']),2 if a in ('G_DIST','G_DIR_DIST') else 0)
   self.assertEqual(v['event_status'],'unknown');self.assertFalse(v['direction_continuity_available']);self.assertFalse(v['three_point_distance_available'])
 def test_invalid_arm_rejected(self):
  with self.assertRaises(ValueError):arm_view({},'invented')
if __name__=='__main__':unittest.main(verbosity=2)

class EdgeAvailabilityTests(unittest.TestCase):
 def test_stationary_segments_do_not_claim_direction_continuity(self):
  from test_continuous_cache import frame, relation
  c=ContinuousCache();c.register(*K,geometry());c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c);c.ingest(frame(2,'c'),2);relation(c,'b','c',2)
  v=arm_view(c.retrieve(request(2,'c'),2),'G_DIR_DIST')
  self.assertFalse(v['direction_continuity_available']);self.assertFalse(v['direction_segment_available'])
 def test_unavailable_geometry_is_not_a_usable_distance_sequence(self):
  from test_continuous_cache import frame, relation
  g=geometry();g['available_at']='100';c=ContinuousCache();c.register(*K,g);c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c);c.ingest(frame(2,'c'),2);relation(c,'b','c',2)
  v=arm_view(c.retrieve(request(2,'c'),2),'G_DIR_DIST')
  self.assertFalse(v['three_point_distance_available']);self.assertEqual(v['historical_distance_sequence'],[])
