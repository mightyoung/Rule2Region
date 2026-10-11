"""Synthetic boundary contracts; no scene pixels, boxes, identities or event GT."""
from pathlib import Path
import sys
import unittest
from copy import deepcopy
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cache-diagnostics-dev-20261011'))
from development_cache import DevelopmentCache
from test_continuous_cache import frame,geometry,relation,request,K
from ablation import arm_view

class Boundaries(unittest.TestCase):
    cache_class=DevelopmentCache
    def make(self,**kwargs):
        c=self.cache_class(**kwargs);c.register(*K,geometry());return c
    def pair(self,status='accepted_hypothesis',**kwargs):
        c=self.make(**kwargs);c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,status=status);return c

    def test_unknown_cannot_upgrade_by_query_or_conflicting_resubmission(self):
        c=self.pair('unknown')
        self.assertEqual(relation(c,status='accepted_hypothesis'),'relation_conflict')
        for now in (1,2,3):
            r=c.retrieve(request(1,'b',query_pts=now,query_time_base='1/1'),now)
            self.assertEqual(r['accepted_endpoint_hypotheses'],[])
            self.assertEqual(r['accepted_history'],[])
            self.assertEqual(r['motion_descriptors'],[])
            self.assertFalse(r['identity_verified'])

    def test_TTL_exact_boundary_then_one_tick_expires_without_read_renewal(self):
        c=self.pair()
        for now in (1,2,3):
            r=c.retrieve(request(1,'b',query_pts=now,query_time_base='1/1'),now)
            self.assertEqual(len(r['accepted_endpoint_hypotheses']),1)
            self.assertEqual(c.obs[K+('a',)]['_t'],0)
        r=c.retrieve(request(1,'b',query_pts=91,query_time_base='1/30'),Fraction(91,30))
        self.assertEqual(r['accepted_endpoint_hypotheses'],[])
        self.assertEqual(r['accepted_history'],[])
        self.assertEqual(r['motion_descriptors'],[])
        self.assertNotIn(K+('a',),c.obs)
        self.assertEqual(c.rels,{})
        self.assertTrue(all(K+('a',) not in d['_observations'] for d in c.descriptors.values()))

    def test_removed_endpoint_cannot_survive_in_evidence_or_descriptor(self):
        for endpoint in ('a','b'):
            with self.subTest(endpoint=endpoint):
                c=self.pair();c.remove_obs(K+(endpoint,),'explicit_test_removal')
                self.assertEqual(c.rels,{})
                self.assertTrue(all(K+(endpoint,) not in d['_observations'] for d in c.descriptors.values()))
                if endpoint=='a':
                    r=c.retrieve(request(1,'b'),1)
                    self.assertEqual(r['accepted_endpoint_hypotheses'],[])
                else:
                    with self.assertRaises(ValueError):c.retrieve(request(1,'b'),1)

    def test_sampling_density_changes_qualification_never_identity_chain(self):
        for middle_count in (0,1,2,5):
            with self.subTest(middle_count=middle_count):
                c=self.make();c.ingest(frame(),0)
                for i in range(1,middle_count+1):
                    c.ingest(frame(i,'middle'+str(i),time_base=f'1/{middle_count+1}'),Fraction(i,middle_count+1))
                c.ingest(frame(1,'b'),1);relation(c)
                r=c.retrieve(request(1,'b'),1);h=r['accepted_endpoint_hypotheses'][0]
                self.assertEqual(h['span_seconds'],'1')
                self.assertEqual(h['intervening_ingested_frames'],middle_count)
                self.assertFalse(h['identity_verified'])
                self.assertEqual(h['direction_continuity'],'unknown')
                self.assertEqual(r['event_status'],'unknown')
                if middle_count:
                    self.assertEqual(r['accepted_history'],[])
                    self.assertEqual(r['motion_descriptors'],[])
                    self.assertIn('nonadjacent_frames',h['motion_missing_inputs'])
                self.assertFalse(arm_view(r,'G_DIR_DIST')['direction_continuity_available'])

    def test_late_evidence_query_cutoff_is_not_receipt_now(self):
        c=self.make();c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,now=2)
        early=c.retrieve(request(1,'b'),2)
        self.assertEqual(early['accepted_endpoint_hypotheses'],[])
        late=c.retrieve(request(1,'b',query_pts=2,query_time_base='1/1'),2)
        self.assertEqual(len(late['accepted_endpoint_hypotheses']),1)

    def test_deletion_classes_and_root_causes_do_not_bleed_between_operations(self):
        for status,label in [('unknown','unscored_candidate'),('accepted_hypothesis','accepted_hypothesis')]:
            with self.subTest(status=status):
                stream=[];c=self.pair(status,event_sink=stream.append,per_view_obs=2)
                c.ingest(frame(2,'c'),2)
                ds=[e['deletion'] for e in stream if e.get('deletion',{}).get('kind')=='relation']
                self.assertEqual(len(ds),1)
                self.assertEqual(ds[0]['association_class'],label)
                self.assertEqual(ds[0]['root_cause'],'capacity')
                self.assertFalse(ds[0]['importance_assessed'])
                stream.clear();c.advance(5,K)
                ds=[e['deletion'] for e in stream if 'deletion' in e]
                self.assertTrue(ds)
                self.assertTrue(all(d['root_cause']=='source_ttl' for d in ds))
                self.assertEqual(c._deletion_context,[])

    def test_total_capacity_removes_other_view_dependencies_without_cross_view_export(self):
        stream=[];c=self.pair(total_obs=2,event_sink=stream.append)
        c.register('other','v1',geometry());c.ingest(frame(0,'x',camera='other'),1)
        self.assertEqual(len(c.obs),2);self.assertEqual(c.rels,{})
        self.assertTrue(all(K+('a',) not in d['_observations'] for d in c.descriptors.values()))
        r=c.retrieve(request(0,'x',camera='other',query_pts=1,query_time_base='1/1'),1)
        self.assertEqual(r['accepted_endpoint_hypotheses'],[])
        ds=[e['deletion'] for e in stream if e.get('deletion',{}).get('kind')=='relation']
        self.assertEqual(ds[0]['association_class'],'accepted_hypothesis')

    def test_export_mutation_cannot_rebind_internal_endpoint_source(self):
        c=self.pair();before=deepcopy(c.rels)
        r=c.retrieve(request(1,'b'),1)
        r['accepted_endpoint_hypotheses'][0]['endpoint_bindings'][0]['source_sha256']='0'*64
        self.assertEqual(c.rels,before)

if __name__=='__main__':unittest.main()
