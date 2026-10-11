"""Synthetic software contracts only; no original frames, identities or event GT."""
import unittest
from copy import deepcopy
from development_cache import DevelopmentCache
from test_continuous_cache import geometry, frame, request, relation, K, H
from ablation import arm_view

class DevelopmentTests(unittest.TestCase):
    def cache(self,**kwargs):
        c=DevelopmentCache(**kwargs);c.register(*K,geometry());return c

    def sparse(self,**kwargs):
        c=self.cache(**kwargs)
        c.ingest(frame(0,'a'),0);c.ingest(frame(1,None),1);c.ingest(frame(2,'b'),2)
        relation(c,now=2);return c

    def deleted(self,events,kind):
        return [e['deletion'] for e in events if e.get('deletion',{}).get('kind')==kind]

    def test_intervening_frame_does_not_erase_endpoint_hypothesis(self):
        c=self.sparse();r=c.retrieve(request(2,'b'),2)
        self.assertIn('accepted_endpoint_hypotheses',r)
        h=r['accepted_endpoint_hypotheses'][0]
        self.assertEqual((h['previous_id'],h['current_id']),('a','b'))
        self.assertEqual(h['source_sha256'],H)
        self.assertEqual(h['span_seconds'],'2')
        self.assertEqual(h['intervening_ingested_frames'],1)
        self.assertEqual(h['endpoint_bindings'][0]['pts'],0)
        self.assertEqual(h['motion_status'],'unknown_motion')
        self.assertIn('nonadjacent_frames',h['motion_missing_inputs'])
        self.assertEqual(h['intermediate_association_status'],'not_established')
        self.assertFalse(h['identity_verified'])
        self.assertEqual(h['direction_continuity'],'unknown')
        self.assertEqual(r['accepted_history'],[])
        self.assertEqual(r['motion_descriptors'],[])
        self.assertFalse(arm_view(r,'G_DIR_DIST')['direction_continuity_available'])
        self.assertEqual(r['event_status'],'unknown')

    def test_single_adjacent_chord_does_not_become_continuous_identity(self):
        c=self.cache();c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c)
        r=c.retrieve(request(1,'b'),1)
        self.assertIn('accepted_endpoint_hypotheses',r)
        h=r['accepted_endpoint_hypotheses'][0]
        self.assertEqual(h['intervening_ingested_frames'],0)
        self.assertEqual(h['motion_status'],'qualified_hypothesis')
        self.assertTrue(h['motion_descriptor_available'])
        self.assertEqual(h['direction_continuity'],'unknown')

    def test_large_time_gap_remains_visible_but_unqualified(self):
        c=self.cache();c.ingest(frame(),0);c.ingest(frame(3,'b'),3);relation(c,now=3)
        r=c.retrieve(request(3,'b'),3)
        self.assertIn('accepted_endpoint_hypotheses',r)
        self.assertEqual(r['accepted_endpoint_hypotheses'][0]['span_seconds'],'3')
        self.assertIn('max_time_gap_exceeded',r['accepted_endpoint_hypotheses'][0]['motion_missing_inputs'])
        self.assertEqual(r['accepted_history'],[])

    def test_unknown_candidate_is_not_an_accepted_endpoint_hypothesis(self):
        c=self.cache();c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,status='unknown')
        r=c.retrieve(request(1,'b'),1)
        self.assertIn('accepted_endpoint_hypotheses',r)
        self.assertEqual(r['accepted_endpoint_hypotheses'],[])
        self.assertEqual(len(r['unknown_candidates']),1)

    def test_late_relation_cannot_leak_into_historical_query(self):
        c=self.cache();c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,now=2)
        r=c.retrieve(request(1,'b'),2)
        self.assertIn('accepted_endpoint_hypotheses',r)
        self.assertEqual(r['accepted_endpoint_hypotheses'],[])
        r=c.retrieve(request(1,'b',query_pts=2,query_time_base='1/1'),2)
        self.assertEqual(len(r['accepted_endpoint_hypotheses']),1)

    def test_export_is_by_copy_and_does_not_recompute_or_renew(self):
        c=self.sparse();before=deepcopy((c.obs,c.rels,c.descriptors))
        r=c.retrieve(request(2,'b'),3)
        self.assertIn('accepted_endpoint_hypotheses',r)
        r['accepted_endpoint_hypotheses'][0]['endpoint_bindings'][0]['pts']=99
        self.assertEqual(before,(c.obs,c.rels,c.descriptors))
        c.advance(4,K)
        self.assertEqual(c.retrieve(request(2,'b',query_pts=4,query_time_base='1/1'),4)['accepted_endpoint_hypotheses'],[])

    def test_unrelated_target_does_not_expose_other_endpoint_hypothesis(self):
        c=self.sparse();c.ingest(frame(3,'c'),3)
        r=c.retrieve(request(3,'c'),3)
        self.assertEqual(r['accepted_endpoint_hypotheses'],[])

    def test_descriptor_eviction_keeps_relation_visible_without_motion_feature(self):
        c=self.cache(per_view_descriptors=1)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c)
        c.ingest(frame(2,'c'),2)
        r=c.retrieve(request(1,'b',query_pts=2,query_time_base='1/1'),2)
        self.assertEqual(len(r['accepted_endpoint_hypotheses']),1)
        self.assertFalse(r['accepted_endpoint_hypotheses'][0]['motion_descriptor_available'])
        self.assertEqual(r['motion_descriptors'],[])
        self.assertEqual(r['accepted_history'],[])

    def test_diagnostics_never_export_deleted_bbox_or_retain_extra_history(self):
        events=[];c=self.cache(per_view_obs=1,audit_size=2,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1)
        for e in events:
            if 'deletion' in e:
                self.assertNotIn('bbox',e['deletion'])
                self.assertFalse(e['deletion']['importance_assessed'])
        self.assertEqual(c._deletion_context,[])
        self.assertEqual(len(c.events),2)
        self.assertEqual(self.cache().po,64)

    def test_valid_query_performs_expiry_maintenance(self):
        events=[];c=self.sparse(event_sink=events.append)
        c.retrieve(request(2,'b',query_pts=4,query_time_base='1/1'),4)
        self.assertNotIn(K+('a',),c.obs)
        ds=self.deleted(events,'observation')
        self.assertTrue(ds,'expired observation must emit deletion diagnostic')
        self.assertEqual(ds[0]['reason'],'expired_observation')
        self.assertEqual(ds[0]['root_cause'],'source_ttl')
        self.assertEqual(self.deleted(events,'relation')[0]['association_class'],'accepted_hypothesis')
        self.assertEqual(self.deleted(events,'relation')[0]['root_cause'],'source_ttl')

    def test_query_reads_do_not_extend_endpoint_TTL(self):
        events=[];c=self.sparse(event_sink=events.append)
        for now in (2,3):c.retrieve(request(2,'b',query_pts=now,query_time_base='1/1'),now)
        self.assertEqual(self.deleted(events,'observation'),[])
        c.advance(4,K)
        self.assertTrue(self.deleted(events,'observation'),'reads must not renew source TTL')
        self.assertNotIn(K+('a',),c.obs)

    def test_null_or_rejected_query_does_not_perform_maintenance(self):
        c=self.sparse();before=deepcopy((c.obs,c.rels,c.keys,c.descriptors))
        self.assertIsNone(c.retrieve(None,100))
        with self.assertRaises(ValueError):c.retrieve(request(2,'b',query_pts=5,query_time_base='1/1'),4)
        self.assertEqual(before,(c.obs,c.rels,c.keys,c.descriptors))

    def test_observation_capacity_cascade_distinguishes_entity_kinds(self):
        events=[];c=self.cache(per_view_obs=2,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c)
        c.ingest(frame(2,'c'),2)
        obs=self.deleted(events,'observation');rel=self.deleted(events,'relation');desc=self.deleted(events,'descriptor')
        self.assertTrue(obs,'capacity deletion diagnostic missing')
        self.assertEqual(obs[0]['reason'],'evicted_observation')
        self.assertEqual(obs[0]['root_cause'],'capacity')
        self.assertEqual(rel[0]['association_class'],'accepted_hypothesis')
        self.assertEqual(rel[0]['root_cause'],'capacity')
        self.assertEqual({d['descriptor_kind'] for d in desc},{'geometry','motion'})
        self.assertTrue(all(d['root_cause']=='capacity' for d in desc))
        self.assertEqual(len(c.obs),2)

    def test_relation_capacity_identifies_unscored_candidate(self):
        events=[];c=self.cache(per_view_rel=1,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,status='unknown')
        c.ingest(frame(2,'c'),2);relation(c,'b','c',2)
        ds=self.deleted(events,'relation')
        self.assertTrue(ds,'relation capacity deletion diagnostic missing')
        self.assertEqual(ds[0]['association_class'],'unscored_candidate')
        self.assertEqual(ds[0]['reason'],'evicted_relation')
        self.assertEqual(ds[0]['root_cause'],'capacity')
        self.assertEqual(ds[0]['counts_after']['relations'],1)
        self.assertEqual(c.rels[K+('b','c')]['status'],'accepted_hypothesis')

    def test_relation_capacity_reports_accepted_deletion_without_priority_claim(self):
        events=[];c=self.cache(per_view_rel=1,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c)
        c.ingest(frame(2,'c'),2);relation(c,'b','c',2,'unknown')
        ds=self.deleted(events,'relation')
        self.assertTrue(ds,'accepted relation deletion must be distinguishable')
        self.assertEqual(ds[0]['association_class'],'accepted_hypothesis')
        motion=[d for d in self.deleted(events,'descriptor') if d['descriptor_kind']=='motion']
        self.assertEqual(motion[0]['reason'],'relation_removed_descriptor')
        self.assertEqual(motion[0]['root_cause'],'capacity')
        self.assertEqual(motion[0]['association_class'],'accepted_hypothesis')

    def test_descriptor_capacity_has_its_own_diagnostic(self):
        events=[];c=self.cache(per_view_descriptors=1,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1)
        ds=self.deleted(events,'descriptor')
        self.assertTrue(ds,'descriptor capacity diagnostic missing')
        self.assertEqual(ds[0]['reason'],'evicted_descriptor')
        self.assertEqual(ds[0]['descriptor_kind'],'geometry')
        self.assertEqual(ds[0]['counts_after']['descriptors'],1)
        self.assertEqual(len(c.obs),2)

    def test_sink_can_stream_deletions_beyond_bounded_audit_ring(self):
        events=[];c=self.cache(per_view_obs=1,audit_size=2,event_sink=events.append)
        for t in range(5):c.ingest(frame(t,str(t)),t)
        self.assertEqual(len(c.events),2)
        self.assertEqual(len(self.deleted(events,'observation')),4)
        self.assertGreater(c.stats['audit_ring_dropped'],0)
        self.assertEqual((c.po,c.pr,c.pd),(1,512,128))

if __name__=='__main__':unittest.main()
