from copy import deepcopy
import ast
from pathlib import Path
import unittest
from baseline import v2,v1
# Red phase ran against v2 before this module existed; raw log is retained.
import pilot_v3 as gate

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.c=v1.read('manual_config.json');self.t=v1.read('topology.json');self.g=v1.read('goals.json')
        self.e=v2.read('geometry_evidence.json');self.r=v2.read('scene_reference.json')
        self.s=v2.read('initial_session.json');self.src=v2.read('initial_census.json')
        # Reviewed rectangle topology: passage through left/right faces, top/bottom barriers.
        for view in self.r['views'].values():
            view['passage_corridor']={'shape':'axis_aligned_rectangle','roi':view['doorway_support_roi'],
                'passage_axis':'x','barrier_faces':['y_min','y_max'],
                'a_face':'x_min' if view['a_anchor'][0]<view['b_anchor'][0] else 'x_max',
                'b_face':'x_max' if view['a_anchor'][0]<view['b_anchor'][0] else 'x_min',
                'review_status':'AUTHOR_REVIEWED_SYNTHETIC_CORRIDOR'}
        self.e['reference_sha256']=v1.digest(self.r)
    def reviews(self):
        cr=gate.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        return cr,gate.fixture_session_review(self.s,cr)
    def run_gate(self):
        cr,sr=self.reviews()
        return gate.execute(self.c,self.t,self.g,v1.read('observations.json')['observations'],
                            self.e,self.r,cr,self.s,self.src,sr)
    def test_short_finite_line_not_a_corridor_cut(self):
        self.c['targets']['building'][0]['line']=[[.5,.26],[.5,.30]]
        run=self.run_gate()
        self.assertEqual(run['status'],'REJECTED_OR_UNKNOWN')
        self.assertNotIn('series',run)
    def test_missing_source_id_and_revision_not_none_equal(self):
        for key in ('source_id','revision'): self.src.pop(key)
        for key in ('source_id','source_revision'): self.s.pop(key)
        self.s['source_sha256']=v1.digest(self.src)
        run=self.run_gate()
        self.assertEqual(run['status'],'REJECTED_OR_UNKNOWN')
        self.assertNotIn('series',run)
    def test_empty_wrong_type_and_nonpositive_source_fields_rejected(self):
        for bad_id,bad_revision in [('',1),('  ',1),(None,1),(42,1),('source',None),('source',0),('source',-1),('source',True),('source','1')]:
            with self.subTest(source_id=bad_id,revision=bad_revision):
                self.src['source_id']=bad_id;self.src['revision']=bad_revision
                self.s['source_id']=bad_id;self.s['source_revision']=bad_revision
                self.s['source_sha256']=v1.digest(self.src)
                self.assertEqual(self.run_gate()['status'],'REJECTED_OR_UNKNOWN')
    def test_different_full_corridor_cut_is_accepted(self):
        self.c['targets']['building'][0]['line']=[[.49,.25],[.51,.75]]
        run=self.run_gate()
        self.assertIn('series',run)
        truth=v1.ground_truth(self.t,self.g,v1.read('states_gt.json')['states'],v1.read('events_gt.json')['events'])
        self.assertTrue(all(m['max_abs_error']==0 for m in v1.score(run,truth).values()))
    def test_missing_or_unsupported_corridor_keeps_unknown(self):
        for change in ('missing','unsupported'):
            with self.subTest(change=change):
                r=deepcopy(self.r)
                view=r['views']['east_front']
                if change=='missing': view.pop('passage_corridor')
                else: view['passage_corridor']['shape']='arbitrary_polygon'
                evidence=deepcopy(self.e);evidence['reference_sha256']=v1.digest(r)
                cr=gate.fixture_config_review(self.c,self.t,self.g,evidence,r)
                sr=gate.fixture_session_review(self.s,cr)
                run=gate.execute(self.c,self.t,self.g,[],evidence,r,cr,self.s,self.src,sr)
                self.assertEqual(run['status'],'REJECTED_OR_UNKNOWN')
    def test_partial_span_entry_exit_span_and_corner_touch_fail(self):
        for line in ([[.5,.25],[.5,.6]], [[.45,.5],[.55,.5]], [[.45,.25],[.55,.75]]):
            with self.subTest(line=line):
                self.c['targets']['building'][0]['line']=line
                self.assertEqual(self.run_gate()['status'],'REJECTED_OR_UNKNOWN')
    def test_full_tilted_cut_not_original_geometry_is_accepted(self):
        self.c['targets']['building'][0]['line']=[[.46,.2],[.54,.8]]
        self.assertIn('series',self.run_gate())
    def test_stale_v2_receipts_cannot_qualify_v3(self):
        cr=v2.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        sr=v2.fixture_session_review(self.s,cr)
        run=gate.execute(self.c,self.t,self.g,[],self.e,self.r,cr,self.s,self.src,sr)
        self.assertEqual(run['status'],'REJECTED_OR_UNKNOWN')
    def test_partition_predicate_reads_no_gt_or_observation(self):
        tree=ast.parse((Path(__file__).parent/'pilot_v3.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finite_corridor_cut')
        code=ast.unparse(fn)
        for forbidden in ('observations','states_gt','events_gt','ground_truth','score','read('):
            self.assertNotIn(forbidden,code)

if __name__=='__main__':unittest.main()
