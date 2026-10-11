import copy, hashlib, json, math, unittest
from fractions import Fraction
from continuous_cache import ContinuousCache
H='a'*64; K=('cam','v1')
def geometry(**kw):
 d={'width':100,'height':100,'portal_source_sha256':'b'*64,'available_at':'0','portals':[{'reference_id':'right','points':[[10,10],[30,10],[30,30],[10,30]]}], 'sill':{'reference_id':'fixed-right','points':[[10,30],[30,30]],'source_sha256':'c'*64,'source_frame_sha256':'d'*64,'source_pts':0,'time_base':'1/1','available_at':'0','provenance':'AI_visible_fragment_not_physical_boundary','coordinate_tolerance_px':6}}
 d.update(kw);return d
def frame(t=0,id='a',bbox=None,**kw):
 d={'camera':'cam','viewRevision':'v1','pts':t,'time_base':'1/1','source_sha256':H,'model':{'name':'frozen'},'rgb_pointer':None,'detections':[] if id is None else [{'id':id,'bbox':bbox or [10,10,14,14]}]};d.update(kw);return d
def request(t=0,id='a',**kw):
 d={'camera':'cam','viewRevision':'v1','pts':t,'time_base':'1/1','target_id':id,'ambiguity_reason':'test_only'};d.update(kw);return d
def relation(c,a='a',b='b',now=1,status='accepted_hypothesis',**kw):
 ep=[{'time':c.obs[K+(i,)]['time'],'source_sha256':c.obs[K+(i,)]['source_sha256']} for i in (a,b)]
 return c.add_relation(*K,a,b,status,H,now,endpoints=ep,**kw)
def pair(c,status='accepted_hypothesis'):
 c.ingest(frame(),0);c.ingest(frame(1,'b',[13,14,17,18]),1);relation(c,status=status)
class Tests(unittest.TestCase):
 def setUp(self):self.c=ContinuousCache();self.c.register(*K,geometry())
 def test_geometry_updates_without_ambiguity_or_retrieval(self):
  self.c.ingest(frame(),0);self.assertEqual(len(self.c.descriptors),1)
  self.assertIsNone(self.c.retrieve(None,0));self.assertEqual(self.c.stats['retrievals'],0)
 def test_nonportal_observation_still_computes(self):
  self.c.ingest(frame(bbox=[60,60,64,64]),0);r=self.c.retrieve(request(),0)
  self.assertEqual(len(r['geometry_descriptors']),1);self.assertFalse(r['geometry_descriptors'][0]['center_in_any_portal'])
 def test_exact_distances_and_no_physical_side(self):
  self.c.ingest(frame(bbox=[12,12,16,16]),0);r=self.c.retrieve(request(),0);self.assertEqual(len(r['geometry_descriptors']),1)
  d=r['geometry_descriptors'][0];v=d['anchors']['center'];p=v['portals'][0]
  self.assertEqual(p['set_distance_px'],0);self.assertEqual(p['boundary_distance_px'],4);self.assertAlmostEqual(p['vertex_mean_distance_px'],math.sqrt(72));self.assertEqual(v['sill']['signed_normal_px'],-16);self.assertEqual(v['sill']['physical_side'],'unknown');self.assertEqual(d['footcontact'],'unknown')
 def test_sill_finite_fragment_not_extrapolated(self):
  self.c.ingest(frame(bbox=[60,10,64,14]),0);r=self.c.retrieve(request(),0);self.assertEqual(len(r['geometry_descriptors']),1)
  s=r['geometry_descriptors'][0]['anchors']['center']['sill'];self.assertFalse(s['within_observed_fragment']);self.assertGreater(s['projection_t'],1);self.assertEqual(s['physical_side'],'unknown')
 def test_accepted_pair_computes_before_query(self):
  pair(self.c);self.assertEqual(len(self.c.descriptors),3);before=copy.deepcopy(self.c.descriptors)
  r=self.c.retrieve(request(1,'b'),1);self.assertEqual(len(r['motion_descriptors']),1);m=r['motion_descriptors'][0]['anchors']['center'];self.assertEqual(m['dx_px'],3);self.assertEqual(m['dy_px'],4);self.assertEqual(m['norm_px'],5);self.assertEqual(before,self.c.descriptors)
 def test_unknown_pair_never_emits_motion(self):
  pair(self.c,'unknown');r=self.c.retrieve(request(1,'b'),1);self.assertEqual(len(self.c.descriptors),2);self.assertEqual(r['motion_descriptors'],[]);self.assertEqual(r['accepted_history'],[]);self.assertEqual(len(r['unknown_candidates']),1);self.assertNotIn('direction',r['unknown_candidates'][0]);self.assertEqual(r['unknown_candidates'][0]['motion_status'],'unknown_motion')
 def test_full_endpoint_bindings_and_source_groups(self):
  pair(self.c);r=self.c.retrieve(request(1,'b'),1);self.assertEqual(len(r['motion_descriptors']),1);m=r['motion_descriptors'][0]
  self.assertEqual([x['observation_id'] for x in m['endpoint_bindings']],['a','b'])
  for ep in m['endpoint_bindings']:
   self.assertEqual(set(ep),{'observation_id','camera','viewRevision','pts','time_base','source_sha256','received_at','available_at'});self.assertEqual(ep['source_sha256'],H)
  self.assertEqual(len(m['source_groups']),2);self.assertFalse(m['identity_verified']);self.assertEqual(m['computed_at'],'1')
 def test_empty_frame_breaks_motion_and_history(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,None),1);self.c.ingest(frame(2,'b'),2);relation(self.c,now=2)
  r=self.c.retrieve(request(2,'b'),2);self.assertEqual(r['accepted_history'],[]);self.assertEqual(r['motion_descriptors'],[]);self.assertIn('nonadjacent_frames',r['missing_inputs'])
 def test_large_gap_breaks_motion(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(3,'b'),3);relation(self.c,now=3);r=self.c.retrieve(request(3,'b'),3);self.assertEqual(r['motion_descriptors'],[]);self.assertEqual(r['accepted_history'],[])
 def test_late_relation_does_not_leak_historical_query(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,'b'),1);relation(self.c,now=2)
  r=self.c.retrieve(request(1,'b'),2);self.assertEqual(r['accepted_history'],[]);self.assertEqual(r['motion_descriptors'],[])
  r=self.c.retrieve(request(1,'b',query_pts=2,query_time_base='1/1'),2);self.assertEqual(len(r['motion_descriptors']),1)
 def test_late_observation_not_visible_before_available(self):
  self.c.ingest(frame(),1)
  with self.assertRaises(ValueError):self.c.retrieve(request(),1)
  r=self.c.retrieve(request(query_pts=1,query_time_base='1/1'),1);self.assertEqual(len(r['geometry_descriptors']),1)
 def test_available_cannot_precede_event_or_exceed_receipt(self):
  for a in ('0','3'):
   with self.assertRaises(ValueError):self.c.ingest(frame(1),2,available_at=a)
  self.assertEqual(len(self.c.obs),0)
 def test_future_time_atomic_and_out_of_order_no_computation(self):
  with self.assertRaises(ValueError):self.c.ingest(frame(2),1)
  self.c.ingest(frame(2),2);d=copy.deepcopy(self.c.descriptors);self.assertEqual(self.c.ingest(frame(1,'b'),2),'out_of_order');self.assertEqual(self.c.descriptors,d)
 def test_future_query_does_not_advance(self):
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.retrieve(request(query_pts=2,query_time_base='1/1'),1)
  self.assertEqual(self.c.keys[K]['now'],0)
 def test_read_does_not_renew_expiry_or_computation(self):
  pair(self.c);self.assertEqual(len(self.c.descriptors),3);before=copy.deepcopy(self.c.descriptors);self.c.retrieve(request(1,'b'),3);self.assertEqual(before,self.c.descriptors);self.c.advance(4,K)
  self.assertEqual(len(self.c.descriptors),1);self.assertEqual(self.c.retrieve(request(1,'b'),4)['motion_descriptors'],[])
 def test_late_expired_observation_no_derived(self):
  self.assertEqual(self.c.ingest(frame(),4),'expired_frame');self.assertEqual(len(self.c.descriptors),0)
 def test_duplicate_does_not_recompute(self):
  self.c.ingest(frame(),0);self.assertEqual(len(self.c.descriptors),1);d=copy.deepcopy(self.c.descriptors);self.assertEqual(self.c.ingest(frame(),2),'duplicate');self.assertEqual(d,self.c.descriptors)
 def test_revision_change_purges_old_descriptors(self):
  pair(self.c);self.c.register('cam','v2',geometry());self.assertEqual(len(self.c.obs),0);self.assertEqual(len(self.c.descriptors),0)
  with self.assertRaises(ValueError):self.c.retrieve(request(1,'b'),1)
  with self.assertRaises(ValueError):self.c.register(*K,geometry())
 def test_same_revision_geometry_mutation_rejected(self):
  changed=geometry();changed['portals'][0]['points'][0][0]=9
  with self.assertRaises(ValueError):self.c.register(*K,changed)
 def test_repeated_id_cannot_rebind_old_descriptor(self):
  pair(self.c);self.c.advance(5,K);self.c.ingest(frame(5),5);self.c.ingest(frame(6,'b'),6)
  with self.assertRaises(ValueError):self.c.add_relation(*K,'a','b','accepted_hypothesis',H,6,endpoints=[{'time':'0','source_sha256':H},{'time':'1','source_sha256':H}])
  self.assertEqual(self.c.retrieve(request(6,'b'),6)['motion_descriptors'],[])
 def test_observation_eviction_purges_dependent_descriptors(self):
  c=ContinuousCache(per_view_obs=2,total_obs=2);c.register(*K,geometry());pair(c);c.ingest(frame(2,'c'),2);self.assertEqual(len(c.descriptors),2);self.assertEqual(c.retrieve(request(1,'b'),2)['motion_descriptors'],[])
 def test_relation_eviction_purges_motion_descriptor(self):
  c=ContinuousCache(per_view_rel=1,total_rel=1);c.register(*K,geometry());pair(c);c.ingest(frame(2,'c'),2);relation(c,'b','c',2,'unknown');self.assertEqual(len(c.descriptors),3);self.assertEqual(c.retrieve(request(1,'b'),2)['motion_descriptors'],[])
 def test_descriptor_capacity_independently_bounded(self):
  c=ContinuousCache(per_view_descriptors=1,total_descriptors=1);c.register(*K,geometry());pair(c);self.assertEqual(len(c.descriptors),1)
 def test_same_source_features_not_independent_votes(self):
  pair(self.c);r=self.c.retrieve(request(1,'b'),1);self.assertEqual(r.get('independent_feature_votes'),0);self.assertEqual(r['model_calls'],0);self.assertEqual(r['event_status'],'unknown')
 def test_snapshots_and_input_aliases_cannot_mutate_cache(self):
  f=frame();self.c.ingest(f,0);f['detections'][0]['bbox'][0]=99;r=self.c.retrieve(request(),0);self.assertEqual(len(r['geometry_descriptors']),1);r['geometry_descriptors'][0]['anchors']['center']['xy'][0]=99;r2=self.c.retrieve(request(),0);self.assertEqual(r2['geometry_descriptors'][0]['anchors']['center']['xy'],[12,12])
 def test_timebase_binding_uses_original_rational_representation(self):
  self.c.ingest(frame(pts=1,time_base='1/3'),1);r=self.c.retrieve(request(pts=2,time_base='1/6',query_pts=1,query_time_base='1/1'),1);self.assertEqual(len(r['geometry_descriptors']),1);self.assertEqual(r['geometry_descriptors'][0]['endpoint_bindings'][0]['time_base'],'1/3')
 def test_stationary_chord_has_unknown_bearing(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,'b'),1);relation(self.c);r=self.c.retrieve(request(1,'b'),1);self.assertEqual(len(r['motion_descriptors']),1);self.assertIsNone(r['motion_descriptors'][0]['anchors']['center']['bearing_degrees'])
 def test_independent_cameras_do_not_share_clock_or_descriptors(self):
  self.c.register('other','v1',geometry());self.c.ingest(frame(100),100);self.c.ingest(frame(0,'b',camera='other'),0);r=self.c.retrieve(request(0,'b',camera='other'),0);self.assertEqual(len(r['geometry_descriptors']),1);self.assertEqual(r['geometry_descriptors'][0]['endpoint_bindings'][0]['camera'],'other');self.assertEqual(len(self.c.descriptors),2)
 def test_unknown_upgrade_does_not_create_motion(self):
  pair(self.c,'unknown');self.assertEqual(relation(self.c),'relation_conflict');self.assertEqual(len(self.c.descriptors),2);self.assertEqual(self.c.retrieve(request(1,'b'),1)['motion_descriptors'],[])
 def test_newer_observation_not_returned_to_earlier_query(self):
  pair(self.c);r=self.c.retrieve(request(),1);self.assertEqual(len(r['geometry_descriptors']),1);self.assertEqual(r['geometry_descriptors'][0]['endpoint_bindings'][0]['observation_id'],'a');self.assertEqual(r['motion_descriptors'],[])
 def test_sill_unavailable_on_early_frame_stays_unknown(self):
  g=geometry();g['sill']['source_pts']=2;g['sill']['available_at']='2';c=ContinuousCache();c.register(*K,g);c.ingest(frame(),0);r=c.retrieve(request(),0);self.assertEqual(len(r['geometry_descriptors']),1);self.assertIsNone(r['geometry_descriptors'][0]['anchors']['center']['sill'])
if __name__=='__main__':unittest.main(verbosity=2)
