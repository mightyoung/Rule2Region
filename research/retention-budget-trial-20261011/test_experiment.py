"""Synthetic policy guardrails; not a recognition evaluation."""
import unittest
from fractions import Fraction
from experimental_cache import ExperimentalCache
from test_continuous_cache import geometry,frame,relation,request,K

class Guardrails(unittest.TestCase):
    def setup_cache(self,status='accepted_hypothesis'):
        c=ExperimentalCache();c.register(*K,geometry())
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,status=status)
        return c
    def pressure(self,c):
        for i in range(80):
            t=Fraction(1100+i,1000)
            c.ingest(frame(1100+i,'ordinary'+str(i),time_base='1/1000'),t)
            self.assertLessEqual(len(c.obs),64)
            self.assertLessEqual(len(c.rels),512)
    def test_pair_survives_within_TTL_under_bounded_protection(self):
        c=self.setup_cache();self.pressure(c)
        self.assertIn(K+('a','b'),c.rels)
        self.assertEqual(len(c.obs),64)
        self.assertEqual(c.obs[K+('a',)]['_t'],0)
    def test_unknown_flood_never_protects_endpoints(self):
        c=self.setup_cache('unknown');self.pressure(c)
        self.assertNotIn(K+('a','b'),c.rels)
    def test_reads_and_protection_do_not_renew_TTL(self):
        c=self.setup_cache();self.pressure(c)
        self.assertIn(K+('b',),c.obs,'protected current endpoint must remain queryable')
        r=c.retrieve(request(1,'b',query_pts=3,query_time_base='1/1'),3)
        self.assertEqual(len(r['accepted_endpoint_hypotheses']),1)
        self.assertEqual(r['event_status'],'unknown')
        c.retrieve(request(1,'b',query_pts=301,query_time_base='1/100'),Fraction(301,100))
        self.assertNotIn(K+('a',),c.obs)
        self.assertNotIn(K+('a','b'),c.rels)
    def test_saturated_quota_never_protects_all_accepted_dependencies(self):
        c=ExperimentalCache();c.register(*K,geometry())
        for i in range(16):
            a='left'+str(i);b='right'+str(i)
            c.ingest(frame(2*i,a,time_base='1/100'),Fraction(2*i,100))
            c.ingest(frame(2*i+1,b,time_base='1/100'),Fraction(2*i+1,100))
            relation(c,a,b,Fraction(2*i+1,100))
        self.pressure(c)
        self.assertLessEqual(len(c.rels),8)
        self.assertGreater(len(c.rels),0)
        self.assertLessEqual(len(c.obs),64)
    def test_binding_mismatch_does_not_grant_protection(self):
        c=self.setup_cache()
        # Explicit adversarial internal corruption check, not an admitted source.
        c.rels[K+('a','b')]['endpoint_bindings'][0]['source_sha256']='0'*64
        self.pressure(c)
        self.assertNotIn(K+('a','b'),c.rels)

if __name__=='__main__':unittest.main()
