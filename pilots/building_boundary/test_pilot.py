import ast
from collections import Counter
from copy import deepcopy
import json
import unittest
import pilot as p


class BoundaryPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.g = p.read('topology.json'), p.read('goals.json')
        cls.states, cls.events = p.read('states_gt.json')['states'], p.read('events_gt.json')['events']
        cls.obs = p.read('observations.json')['observations']
        cls.cfg = p.read('manual_config.json')
        cls.truth = p.ground_truth(cls.t,cls.g,cls.states,cls.events)

    def run_cfg(self,cfg=None,obs=None):
        cfg = cfg or self.cfg
        return p.execute(cfg,self.t,self.g,self.obs if obs is None else obs,p.fixture_review(cfg))

    def test_independent_checkpoints_match_manual_ledger(self):
        p.verify_ledgers(self.t,self.states,self.events)
        altered = deepcopy(self.events); altered[0]['destination']='B'
        with self.assertRaises(AssertionError):
            p.verify_ledgers(self.t,self.states,altered)

    def test_gt_initial_is_nonzero_and_matches_input(self):
        for target in ('building','floor1','floor2'):
            self.assertEqual(self.truth['series'][0]['values'][target],self.g['initial'][target])
        self.assertEqual(self.g['initial']['building'],3)

    def test_fixed_counter_matches_every_checkpoint_and_event(self):
        for strategy in ('manual_truth','metadata_rule'):
            cfg = self.cfg if strategy=='manual_truth' else p.propose(self.t,self.g,strategy)
            run = self.run_cfg(cfg)
            self.assertEqual(run['status'],'SYNTHETIC_ADMITTED')
            for m in p.score(run,self.truth).values():
                self.assertEqual((m['mae'],m['event_fp'],m['event_fn']),(0,0,0))

    def test_internal_moves_conserve_building_and_floor(self):
        values = [r['values'] for r in self.run_cfg()['series']]
        for step in (2,8):
            self.assertEqual(values[step],values[step-1])
        for step in (3,4,9,11):
            self.assertEqual(values[step]['building'],values[step-1]['building'])
            self.assertNotEqual(values[step]['floor1'],values[step-1]['floor1'])

    def test_floor_and_external_conservation_all_steps(self):
        for r in self.run_cfg()['series']:
            v = r['values']
            self.assertEqual(v['building'],v['floor1']+v['floor2'])
            self.assertEqual(v['building']-3,v['net_east']+v['net_upper']+v['net_service'])

    def test_reenter_exit_are_separate_and_simultaneous_events_survive(self):
        run = self.run_cfg()
        self.assertIn([1,'east',1],run['events']['building'])
        self.assertIn([5,'east',-1],run['events']['building'])
        self.assertEqual(sum(e[0]==10 for e in run['events']['building']),2)

    def test_multi_view_owner_excludes_duplicate_camera(self):
        self.assertEqual(sum(o['step']==1 for o in self.obs),2)
        self.assertEqual(sum(e[0]==1 for e in self.run_cfg()['events']['building']),1)

    def test_retransmitted_local_observation_not_counted_twice(self):
        self.assertEqual(self.run_cfg()['series'],self.run_cfg(obs=self.obs+self.obs)['series'])

    def test_pass_by_and_outside_roi_do_not_count(self):
        clean = [o for o in self.obs if o['track_id'] not in ('pass_by','outside_roi')]
        self.assertEqual(self.run_cfg()['events'],self.run_cfg(obs=clean)['events'])

    def test_geometry_finite_line_touch_and_oblique_crossing(self):
        e = deepcopy(self.cfg['targets']['building'][0])
        self.assertIsNone(p.crossing({'prev':[.5,.5],'next':[.3,.5]},e))
        self.assertIsNone(p.crossing({'prev':[.7,.05],'next':[.3,.05]},e))
        self.assertEqual(p.crossing({'prev':[.7,.3],'next':[.3,.7]},e),1)
        e['roi'] = [.1,.1,.2,.2]
        self.assertIsNone(p.crossing({'prev':[.7,.3],'next':[.3,.7]},e))
        e['roi'] = [0,0,float('nan'),1]
        self.assertFalse(p.valid_geometry(e))

    def test_door_only_is_rejected_not_officially_scored(self):
        results = p.run_all()['strategies']['door_only']
        self.assertIsNone(results['metrics'])
        self.assertEqual(results['diagnostic']['status'],'DIAGNOSTIC_INVALID_CONFIG')
        self.assertGreater(results['diagnostic']['diagnostic_metrics']['building']['mae'],0)

    def test_nine_failure_gates_do_not_emit_scores(self):
        for name,run in p.run_all()['failures'].items():
            with self.subTest(case=name):
                self.assertEqual(run['status'],'REJECTED_OR_UNKNOWN')
                self.assertTrue(run['reasons'])
                self.assertIsNone(run['metrics'])

    def test_missing_initial_never_becomes_zero(self):
        g = deepcopy(self.g); g['initial']['building']=None
        run = p.execute(self.cfg,self.t,g,self.obs,p.fixture_review(self.cfg))
        self.assertIn('UNKNOWN_INITIAL_OCCUPANCY',run['reasons'])
        self.assertNotIn('series',run)

    def test_missing_roi_has_no_fallback(self):
        cfg = deepcopy(self.cfg); del cfg['targets']['building'][0]['roi']
        run = self.run_cfg(cfg)
        self.assertIn('building:INVALID_GEOMETRY',run['reasons'])

    def test_review_hash_binds_configuration(self):
        review = p.fixture_review(self.cfg)
        cfg = deepcopy(self.cfg); cfg['strategy']='modified'
        self.assertIn('MISSING_OR_STALE_REVIEW',p.execute(cfg,self.t,self.g,self.obs,review)['reasons'])

    def test_uncalled_models_and_real_subtask_have_null_scores(self):
        r = p.run_all()
        for name in ('model_candidate','model_plus_metadata_or_plan'):
            self.assertEqual(r['strategies'][name]['status'],'NOT_RUN')
            self.assertIsNone(r['strategies'][name]['metrics'])
        self.assertIsNone(r['real_public_subtasks']['occupancy_metrics'])
        self.assertFalse(r['real_public_subtasks']['media_acquired'])

    def test_proposer_and_executor_have_no_gt_access(self):
        tree = ast.parse((p.ROOT/'pilot.py').read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('propose','execute','crossing'):
                code = ast.unparse(node)
                self.assertNotIn('states_gt',code)
                self.assertNotIn('events_gt',code)
                for child in ast.walk(node):
                    if isinstance(child,ast.Call) and isinstance(child.func,ast.Name):
                        self.assertNotIn(child.func.id,('read','ground_truth','score','open'))

    def test_event_direction_errors_score_fp_and_fn(self):
        run = self.run_cfg(); run['events']['building'][0][2] *= -1
        m = p.score(run,self.truth)['building']
        self.assertEqual((m['event_fp'],m['event_fn']),(1,1))

    def test_scoring_rejects_incomplete_series(self):
        run = self.run_cfg(); run['series'].pop()
        with self.assertRaises(ValueError):
            p.score(run,self.truth)

    def test_shifted_checkpoints_and_out_of_range_gt_are_rejected(self):
        run = self.run_cfg(); run['series'][0]['step'] = 100
        with self.assertRaises(ValueError):
            p.score(run,self.truth)
        events = deepcopy(self.events); events[0]['step'] = 100
        with self.assertRaises(AssertionError):
            p.verify_ledgers(self.t,self.states,events)

if __name__ == '__main__':
    unittest.main()
