import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import pilot_v2 as p

class LayeredAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results=p.run_all()
    def setUp(self):
        self.c=p.v1.read('manual_config.json');self.t=p.v1.read('topology.json');self.g=p.v1.read('goals.json')
        self.e=p.read('geometry_evidence.json');self.r=p.read('scene_reference.json')
        self.s=p.read('initial_session.json');self.src=p.read('initial_census.json')
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.sr=p.fixture_session_review(self.s,self.cr)
    def config_reasons(self):
        return p.validate_config(self.c,self.t,self.g,self.e,self.r,self.cr)
    def session_reasons(self):
        return p.validate_session(self.s,self.src,self.cr,self.sr)
    def test_v1_three_false_admissions_are_retained(self):
        cases=self.results['counterexamples_v1']
        self.assertEqual([cases[k]['metrics']['building']['max_abs_error'] for k in
            ('reverse_line_only','displaced_valid_roi','changed_initial_old_approval')],[2,1,27])
        self.assertTrue(all(r['status']=='SYNTHETIC_ADMITTED' for r in cases.values()))
    def test_v2_rejects_all_twelve_cases_without_predictions(self):
        for name,r in self.results['failures_v2'].items():
            with self.subTest(case=name):
                self.assertEqual(r['status'],'REJECTED_OR_UNKNOWN')
                self.assertIsNone(r['metrics']);self.assertNotIn('series',r)
    def test_direction_uses_oriented_line_and_independent_region_anchors(self):
        e=self.c['targets']['building'][0];e['line'].reverse()
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertIn('building/east:DIRECTED_LINE_REGION_SIDE_CONFLICT',self.config_reasons())
        e['a_side']*=-1
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertEqual(self.config_reasons(),[])
    def test_numeric_legality_is_not_line_roi_consistency(self):
        e=self.c['targets']['building'][0];e['roi']=[.01,.01,.1,.1]
        self.assertTrue(p.v1.valid_geometry(e))
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertIn('building/east:LINE_ROI_INCONSISTENT',self.config_reasons())
    def test_line_roi_consistency_is_not_reviewed_portal_coverage(self):
        e=self.c['targets']['building'][0];e['roi']=[.49,.4,.51,.6]
        self.assertTrue(p.intersects_interior(e['line'],e['roi']))
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertIn('building/east:ROI_MISSES_REVIEWED_PORTAL_SUPPORT',self.config_reasons())
    def test_roi_tangent_and_disjoint_line_are_rejected(self):
        self.assertFalse(p.intersects_interior([[.5,.2],[.5,.8]],[.5,.1,.8,.9]))
        self.assertFalse(p.intersects_interior([[.5,.2],[.5,.8]],[.01,.01,.1,.1]))
        self.assertTrue(p.intersects_interior([[.5,.2],[.5,.8]],[.2,.1,.8,.9]))
    def test_missing_evidence_is_unknown_even_with_new_receipt(self):
        self.e=None;self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertIn('UNKNOWN_OR_STALE_SCENE_EVIDENCE',self.config_reasons())
    def test_malformed_scene_geometry_rejected_without_counting(self):
        self.r['views']['east_front']['a_anchor']=None
        self.e['reference_sha256']=p.digest(self.r)
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertIn('building/east:MALFORMED_SCENE_GEOMETRY_EVIDENCE',self.config_reasons())
    def test_missing_session_is_unknown_and_never_counts(self):
        run=p.execute(self.c,self.t,self.g,[],self.e,self.r,self.cr,None,self.src,self.sr)
        self.assertIn('UNKNOWN_INITIAL_SESSION',run['session_reasons'])
        self.assertNotIn('series',run)
    def test_config_approval_is_separate_from_initial_session(self):
        self.g['initial']={'building':30,'floor1':20,'floor2':10}
        self.s['initial']=dict(self.g['initial'])
        self.assertEqual(self.config_reasons(),[])
        self.assertIn('MISSING_OR_STALE_SESSION_QUALIFICATION',self.session_reasons())
        self.assertIn('UNKNOWN_OR_INCONSISTENT_INITIAL_SOURCE',self.session_reasons())
    def test_t0_source_and_revision_changes_invalidate_old_session(self):
        for key,value in [('t0',1),('source_sha256','new'),('source_revision',2),('revision',2)]:
            with self.subTest(field=key):
                self.s=p.read('initial_session.json');self.s[key]=value
                self.assertIn('MISSING_OR_STALE_SESSION_QUALIFICATION',self.session_reasons())
    def test_new_config_approval_invalidates_old_session_receipt(self):
        self.c['strategy']='revised_proposal'
        self.cr=p.fixture_config_review(self.c,self.t,self.g,self.e,self.r)
        self.assertEqual(self.config_reasons(),[])
        self.assertIn('MISSING_OR_STALE_SESSION_QUALIFICATION',self.session_reasons())
    def test_new_initial_source_requires_matching_values_and_new_receipt(self):
        # No hidden comparison to original GT=3. Authenticated real source accuracy is external.
        self.src['initial']={'building':30,'floor1':20,'floor2':10};self.src['revision']=2
        self.s['initial']=dict(self.src['initial']);self.s['source_revision']=2
        self.s['source_sha256']=p.digest(self.src);self.s['revision']=2
        self.assertIn('MISSING_OR_STALE_SESSION_QUALIFICATION',self.session_reasons())
        self.sr=p.fixture_session_review(self.s,self.cr)
        self.assertEqual(self.session_reasons(),[])
    def test_positive_controls_not_exact_original_geometry_or_gt_binding(self):
        for r in self.results['non_oracle_positive_controls'].values():
            self.assertEqual(r['status'],'SYNTHETIC_V2_ADMITTED')
            self.assertTrue(all(m['max_abs_error']==0 and m['event_fp']==m['event_fn']==0 for m in r['metrics'].values()))
    def test_baseline_fixed_numerical_executor_and_zero_fixture_preserved(self):
        old=json.loads((p.v1.ROOT/'results.json').read_text())
        for name in ('manual_truth','metadata_rule'):
            self.assertEqual(old['strategies'][name]['series'],self.results['strategies'][name]['series'])
            self.assertEqual(old['strategies'][name]['metrics'],self.results['strategies'][name]['metrics'])
        self.assertEqual(old['fixed_executor_sha256'],self.results['fixed_v1_executor_sha256'])
    def test_model_slots_still_not_run(self):
        for name in ('model_candidate','model_plus_metadata_or_plan'):
            self.assertEqual(self.results['strategies'][name],{'status':'NOT_RUN','metrics':None})
    def test_original_manifest_files_are_unchanged(self):
        manifest=json.loads((p.v1.ROOT/'MANIFEST.json').read_text())
        for name,e in manifest['files'].items():
            self.assertEqual(hashlib.sha256((p.v1.ROOT/name).read_bytes()).hexdigest(),e['sha256'],name)
    def test_admission_has_no_gt_read_or_scorer_calls(self):
        tree=ast.parse((p.ROOT/'pilot_v2.py').read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('validate_config','validate_session','execute','config_binding'):
                text=ast.unparse(node)
                for forbidden in ('states_gt','events_gt','ground_truth','v1.score','v1.read'):
                    self.assertNotIn(forbidden,text)

if __name__=='__main__': unittest.main()
