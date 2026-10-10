import unittest
from vendor.base_cache import Cache
H='a'*64
K=('cam','v1')
def frame(t=0, ids=('a',), **kw):
 d=dict(camera='cam',viewRevision='v1',pts=t,time_base='1/1',source_sha256=H,model={'name':'frozen-detector','version':'unknown'},detections=[{'id':x,'bbox':[0,0,2,2]} for x in ids],rgb_pointer=None)
 d.update(kw);return d
def req(t=0,target='a',**kw):
 d=dict(camera='cam',viewRevision='v1',pts=t,time_base='1/1',target_id=target,ambiguity_reason='unresolved');d.update(kw);return d
def rel(c,camera,view,a,b,status,source,now):
 k=(camera,view)
 endpoints=[{'time':c.obs.get(k+(i,),{}).get('time','0'),'source_sha256':c.obs.get(k+(i,),{}).get('source_sha256',H)} for i in (a,b)]
 return c.add_relation(camera,view,a,b,status,source,now,endpoints=endpoints)
class Tests(unittest.TestCase):
 def setUp(self): self.c=Cache();self.c.register(*K)
 def test_store_all_without_request(self):
  self.c.ingest(frame(ids=('a','b')),0);self.assertIsNone(self.c.retrieve(None,0));self.assertEqual(len(self.c.obs),2);self.assertEqual(self.c.stats['retrievals'],0)
 def test_normal_and_unknown_does_not_link(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,('b',)),1);rel(self.c,*K,'a','b','unknown',H,1)
  r=self.c.retrieve(req(1,'b'),1);self.assertEqual(r['accepted_history'],[]);self.assertEqual(len(r['unknown_candidates']),1);self.assertEqual(r['entry'],'unknown')
 def test_accepted_chain(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,('b',)),1);rel(self.c,*K,'a','b','accepted_hypothesis',H,1)
  r=self.c.retrieve(req(1,'b'),1);self.assertEqual([x['id'] for x in r['accepted_history']],['a']);self.assertFalse(r['identity_verified'])
 def test_scope_isolation(self):
  self.c.ingest(frame(),0)
  for q in [req(camera='other'),req(viewRevision='zoom2'),req(viewRevision=''),req(viewRevision='unknown')]:
   with self.assertRaises(ValueError):self.c.retrieve(q,0)
 def test_future_out_of_order_duplicate(self):
  with self.assertRaises(ValueError):self.c.ingest(frame(2),1)
  self.c.ingest(frame(2),2);self.assertEqual(self.c.ingest(frame(2),2),'duplicate')
  self.assertEqual(self.c.ingest(frame(1,('b',)),2),'out_of_order');self.assertEqual(len(self.c.obs),1)
  with self.assertRaises(ValueError):self.c.retrieve(req(3),2)
 def test_no_renewal_expiry_no_resurrection(self):
  self.c.ingest(frame(),0);self.c.retrieve(req(),2);self.assertEqual(self.c.ingest(frame(),3),'duplicate');self.assertEqual(len(self.c.obs),1)
  self.c.advance(4);self.assertEqual(len(self.c.obs),0);self.c.ingest(frame(),4);self.assertEqual(len(self.c.obs),0);self.assertEqual(self.c.stats['expired_observation'],1)
 def test_capacity_and_incident_accounting(self):
  c=Cache(per_view_obs=2,total_obs=2);c.register(*K);c.ingest(frame(),0);c.ingest(frame(1,('b',)),1);rel(c,*K,'a','b','unknown',H,1);c.ingest(frame(2,('c',)),2)
  self.assertEqual(len(c.obs),2);self.assertEqual(len(c.rels),0);self.assertEqual(c.stats['evicted_observation'],1);self.assertEqual(c.stats['endpoint_removed_relation'],1)
 def test_invalid_input_atomic(self):
  for f in [frame(viewRevision=''),frame(source_sha256=''),frame(pts=float('nan')),frame(time_base='0/30'),frame(time_base='1/0'),frame(ids=('','b')),frame(ids=('x','x')),frame(detections=[{'id':'x','bbox':[0,0,float('nan'),2]}])]:
   with self.assertRaises(ValueError):self.c.ingest(f,0)
  self.assertEqual(len(self.c.obs),0)
 def test_empty_missing_crossing(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,()),1)
  with self.assertRaises(ValueError):self.c.retrieve(req(1,'absent'),1)
  r=self.c.retrieve(req(),1);self.assertEqual(r['hidden_crossing'],'unknown');self.assertEqual(r['footcontact'],'unknown');self.assertEqual(r['safety'],'unknown')
 def test_conflict_replay_does_not_change_bbox(self):
  self.c.ingest(frame(),0);self.assertEqual(self.c.ingest(frame(detections=[{'id':'a','bbox':[0,0,3,3]}]),0),'replay_conflict');self.assertEqual(self.c.retrieve(req(),0)['current']['bbox'],[0,0,2,2])
 def test_max_keys_and_metadata_bounds(self):
  c=Cache(max_cameras=1,max_keys=1);c.register(*K)
  with self.assertRaises(ValueError):c.register('c2','v1')
  with self.assertRaises(ValueError):c.register('cam','v2')
  with self.assertRaises(ValueError):c.ingest(frame(model={'x':'a'*2000}),0)
 def test_relation_bounds(self):
  c=Cache(per_view_rel=1,total_rel=1);c.register(*K);c.ingest(frame(0,('a','c')),0);c.ingest(frame(1,('b',)),1)
  rel(c,*K,'a','b','unknown',H,1);rel(c,*K,'c','b','unknown',H,1);self.assertEqual(len(c.rels),1);self.assertEqual(c.stats['evicted_relation'],1)
  with self.assertRaises(ValueError):rel(c,*K,'b','c','unknown',H,1)
 def test_timebase_exact_and_no_clock_rewind(self):
  self.c.ingest(frame(pts=1,time_base='1/3'),1);self.assertEqual(self.c.retrieve(req(pts=2,time_base='1/6'),1)['current']['time'],'1/3')
  with self.assertRaises(ValueError):self.c.advance(0)
 def test_invalid_future_ingest_does_not_advance(self):
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.ingest(frame(100,('b',)),99)
  self.assertEqual(self.c.now,0);self.assertEqual(self.c.ingest(frame(1,('b',)),1),'stored')
 def test_missing_query_does_not_advance(self):
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.retrieve(req(99,'absent'),99)
  self.assertEqual(self.c.now,0);self.assertEqual(len(self.c.obs),1)
 def test_missing_relation_does_not_advance(self):
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):rel(self.c,*K,'a','absent','unknown',H,99)
  self.assertEqual(self.c.now,0);self.assertEqual(len(self.c.obs),1)
 def test_live_reused_id_does_not_advance(self):
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.ingest(frame(1),1)
  self.assertEqual(self.c.now,0)
 def test_no_accepted_branch_or_unknown_upgrade(self):
  self.c.ingest(frame(0,('a','b')),0);self.c.ingest(frame(1,('c','d')),1)
  rel(self.c,*K,'a','c','accepted_hypothesis',H,1)
  for a,b in [('a','d'),('b','c')]:
   with self.assertRaises(ValueError):rel(self.c,*K,a,b,'accepted_hypothesis',H,1)
  rel(self.c,*K,'b','d','unknown',H,1)
  self.assertEqual(rel(self.c,*K,'b','d','accepted_hypothesis',H,1),'relation_conflict')
  self.assertEqual(self.c.retrieve(req(1,'d'),1)['accepted_history'],[])
 def test_registered_camera_and_zoom_never_mix(self):
  self.c.register('other','v1');self.c.register('cam','v2');self.c.ingest(frame(),0)
  self.c.ingest(frame(1,('b',),camera='other'),1);self.c.ingest(frame(1,('c',),viewRevision='v2'),1)
  for ca,v,a,b in [('cam','v1','a','b'),('other','v1','a','b'),('cam','v2','a','c')]:
   with self.assertRaises(ValueError):rel(self.c,ca,v,a,b,'accepted_hypothesis',H,1)
 def test_old_query_excludes_future_and_expired_edges(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,('b',)),1);rel(self.c,*K,'a','b','accepted_hypothesis',H,1)
  self.assertEqual(self.c.retrieve(req(),1)['accepted_history'],[])
  self.c.advance(4);self.assertEqual(self.c.retrieve(req(1,'b'),4)['accepted_history'],[])
 def test_unapproved_rgb_pixels_and_blank_ambiguity(self):
  with self.assertRaises(ValueError):self.c.ingest(frame(rgb_pointer='file:///made-up.jpg'),0)
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.retrieve(req(ambiguity_reason=''),0)
 def test_global_capacity_across_views(self):
  c=Cache(total_obs=1);c.register(*K);c.register('other','v1');c.ingest(frame(),0);c.ingest(frame(1,('b',),camera='other'),1)
  self.assertEqual(len(c.obs),1);self.assertEqual(c.stats['evicted_observation'],1)
 def test_unexpected_payload_rejected(self):
  with self.assertRaises(ValueError):self.c.ingest(frame(oracle_identity='person1'),0)
  self.c.ingest(frame(),0)
  with self.assertRaises(ValueError):self.c.retrieve(req(oracle_scope='inside'),0)
 def test_stale_relation_cannot_attach_reused_ids(self):
  self.c.ingest(frame(),0);self.c.ingest(frame(1,('b',)),1)
  old=[{'time':'0','source_sha256':H},{'time':'1','source_sha256':H}]
  self.c.advance(5);self.c.ingest(frame(5),5);self.c.ingest(frame(6,('b',)),6)
  with self.assertRaises(ValueError):self.c.add_relation(*K,'a','b','accepted_hypothesis',H,6,endpoints=old)
  self.assertEqual(self.c.retrieve(req(6,'b'),6)['accepted_history'],[])
 def test_independent_camera_clock_origins(self):
  self.c.register('other','v1');self.c.ingest(frame(100),100)
  self.assertEqual(self.c.ingest(frame(0,('b',),camera='other'),0),'stored')
  self.assertEqual(len(self.c.obs),2)
  self.c.retrieve(req(0,'b',camera='other'),3)
  self.assertEqual(self.c.retrieve(req(100),100)['current']['id'],'a')
if __name__=='__main__':unittest.main(verbosity=2)
