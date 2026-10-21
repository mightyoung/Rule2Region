"""SYNTHETIC software fixtures, never human truth or research accuracy."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

try:
    import scoring_contract as s
except ImportError:
    s = None


def box(oid, xyxy, **extra):
    return dict(id=oid, category='person', bbox=xyxy, truncated=False, **extra)


def fixture():
    binding = dict(source_sha256='a'*64, camera_id='synthetic-camera',
                   source_version='synthetic-v1', roi_version='synthetic-roi-v1',
                   policy_version='synthetic-policy-v1')
    unit = dict(unit_id='synthetic-u1', source_sha256='a'*64,
        camera_id='synthetic-camera', source_version='synthetic-v1', arm='synthetic-arm',
        width=100, height=100, split='development', lineage_ids=['synthetic-family-1'],
        input_layer='I0', task='person_roi_retention', counterfactual_family='none',
        target_answers_supplied=False, source_geometry_substitutes_observability=False,
        fixture_admission={k: True for k in ['source_authorized', 'bundle_integrity',
            'independent_full_frame_labels', 'roi_reviewed', 'camera_registered',
            'policy_bound', 'observability_reviewed', 'stable_ids_frozen',
            'preregistered_before_results']},
        roi=dict(version='synthetic-roi-v1', points=[[0,0],[50,0],[50,100],[0,100],[0,0]]),
        policy_version='synthetic-policy-v1',
        labels=dict(status='complete', annotation_complete=True,
            scope='full_frame_visible_person', fixture_reviewed=True, binding=binding,
            objects=[box('g1',[10,10,20,20]), box('g2',[40,10,60,20]),
                     box('g3',[25,30,35,40])], ignore=[]),
        predictions=[box('p1',[10,10,20,20],route='keep'),
                     box('p2',[40,10,60,20],route='suppress'),
                     box('p3',[70,10,80,20],route='suppress')])
    return dict(contract_version='person-roi-retention.v1',
        kind='synthetic_software_fixture', preregistered_units=['synthetic-u1'],units=[unit])


class ScoringContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'deterministic scorer not implemented yet')

    def score(self, p=None):
        return s.evaluate(fixture() if p is None else p)

    def metrics(self, report):
        return report['software_metrics_by_arm']['synthetic-arm']

    def test_four_denominators_are_not_interchanged(self):
        r=self.score(); self.assertEqual(r['status'],'synthetic_scored')
        m=self.metrics(r)
        for name,n,d in [('in_region_recall',1,3),('in_region_recall_loss',1,3),
                         ('boundary_retention',0,1),('outside_suppression',1,1)]:
            self.assertEqual((m[name]['numerator'],m[name]['denominator']),(n,d))
            self.assertEqual(m[name]['value'],n/d)
        self.assertIsNone(r['accuracy_metrics']); self.assertFalse(r['human_GT'])
        self.assertFalse(r['scientific_ROI_admission'])

    def test_baseline_pair_frozen_after_suppress_not_rematched(self):
        p=fixture();u=p['units'][0];u['labels']['objects']=[box('g',[10,10,20,20])]
        u['predictions']=[box('a',[10,10,20,20],route='suppress'),
                          box('b',[10,10,20,20],route='keep')]
        r=self.score(p);m=self.metrics(r)
        self.assertEqual(r['units'][0]['pairs'],[['g','a']])
        self.assertEqual(m['in_region_recall']['value'],0)
        self.assertEqual(m['in_region_recall_loss']['value'],1)

    def test_ignore_cannot_steal_normal_gt_and_excluded_from_outside(self):
        p=fixture();u=p['units'][0];u['labels']['ignore']=[
            box('ignore-in',[10,10,20,20],reason='synthetic predeclared'),
            box('ignore-out',[70,10,80,20],reason='synthetic predeclared')]
        r=self.score(p)
        self.assertIn(['g1','p1'],r['units'][0]['pairs'])
        self.assertEqual(r['units'][0]['ignored_prediction_ids'],['p3'])
        self.assertEqual(self.metrics(r)['outside_suppression']['denominator'],0)

    def test_reviewed_negative_has_no_recall_support_not_perfect_score(self):
        p=fixture();p['units'][0]['labels']['objects']=[]
        m=self.metrics(self.score(p))
        for name in ['in_region_recall','in_region_recall_loss','boundary_retention']:
            self.assertIsNone(m[name]['value']);self.assertEqual(m[name]['reason'],'no_support')
        self.assertEqual(m['outside_suppression']['value'],1)

    def test_invalid_unit_blocks_entire_preregistered_population(self):
        changes=[lambda u:u['labels'].update(annotation_complete=False),
                 lambda u:u['labels'].update(fixture_reviewed=False),
                 lambda u:u['predictions'][0].update(route='unknown'),
                 lambda u:u['predictions'][0].update(route='review'),
                 lambda u:u['predictions'][0].update(category='door'),
                 lambda u:u['predictions'][0].update(bbox=[0,0,0,1]),
                 lambda u:u['predictions'].append(copy.deepcopy(u['predictions'][0])),
                 lambda u:u.update(roi=None),lambda u:u.update(width=True),
                 lambda u:u.update(camera_id='unknown'),
                 lambda u:u['labels']['binding'].update(source_sha256='b'*64),
                 lambda u:u['labels']['ignore'].append(box('i',[1,1,2,2],reason='')),
                 lambda u:u['fixture_admission'].update(bundle_integrity=False),
                 lambda u:u.update(unrecognized_field=True)]
        for mutate in changes:
            p=fixture();mutate(p['units'][0]);r=self.score(p)
            with self.subTest(mutation=mutate):
                self.assertEqual(r['status'],'blocked')
                self.assertEqual(r['coverage']['admitted_units'],0)
                self.assertIsNone(self.metrics(r)['in_region_recall']['denominator'])

    def test_no_partial_population_or_prompt_arm_pooling(self):
        p=fixture();v=copy.deepcopy(p['units'][0]);v['unit_id']='synthetic-u2'
        v['arm']='second-arm';v['predictions'][0]['route']='suppress'
        p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
        r=self.score(p)
        self.assertEqual(len(r['software_metrics_by_arm']),2)
        self.assertEqual(self.metrics(r)['in_region_recall']['numerator'],1)
        self.assertEqual(r['software_metrics_by_arm']['second-arm']['in_region_recall']['numerator'],0)
        v['labels']['annotation_complete']=False;r=self.score(p)
        self.assertEqual(r['coverage']['admitted_units'],1)
        self.assertIsNone(self.metrics(r)['in_region_recall']['value'])
        p['preregistered_units'].append('synthetic-missing')
        self.assertEqual(self.score(p)['status'],'blocked')

    def test_transitive_lineage_and_same_image_cannot_cross_splits(self):
        p=fixture()
        for n,ids in [(2,['synthetic-family-1','synthetic-link-2']),
                      (3,['synthetic-link-2'])]:
            v=copy.deepcopy(p['units'][0]);v['unit_id']=f'synthetic-u{n}'
            v['source_sha256']=str(n)*64;v['labels']['binding']['source_sha256']=str(n)*64
            v['lineage_ids']=ids;v['split']='holdout' if n==3 else 'development'
            p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
        self.assertEqual(self.score(p)['status'],'blocked')

    def test_global_split_failure_does_not_claim_full_admission(self):
        p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='synthetic-u2',source_sha256='b'*64,split='holdout')
        v['labels']['binding']['source_sha256']='b'*64
        p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
        r=self.score(p)
        self.assertEqual(r['status'],'blocked')
        self.assertEqual(r['coverage']['locally_valid_units'],2)
        self.assertEqual(r['coverage']['admitted_units'],0)
        self.assertEqual(r['coverage']['value'],0)
        self.assertEqual(r['camera_support_by_arm'],{})
        self.assertIn('synthetic-arm',r['locally_valid_subset_camera_support_by_arm'])

    def test_arm_cannot_pool_input_layers_or_splits_or_different_populations(self):
        for change in [dict(input_layer='I1-E'),dict(split='calibration'),dict(arm='different-arm')]:
            p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='synthetic-u2',source_sha256='b'*64,lineage_ids=['synthetic-independent'])
            v['labels']['binding']['source_sha256']='b'*64;v.update(change)
            p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
            self.assertEqual(self.score(p)['status'],'blocked')

    def test_different_layer_arms_are_scoped_never_visual_gain(self):
        p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='synthetic-u2',arm='configuration',input_layer='I1-E')
        p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
        r=self.score(p);self.assertEqual(r['status'],'synthetic_scored')
        self.assertEqual(r['arm_scope']['configuration']['interpretation'],'configuration_execution_only')
        self.assertEqual(r['cross_arm_comparison'],'not_comparable_across_input_layers')

    def test_same_version_cannot_hide_cross_arm_gt_roi_or_dimension_changes(self):
        for mutation in ['gt','ignore','roi','dimensions']:
            p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='synthetic-u2',arm='second-arm')
            if mutation=='gt':v['labels']['objects'].pop()
            if mutation=='ignore':v['labels']['ignore'].append(box('i',[70,70,80,80],reason='synthetic'))
            if mutation=='roi':v['roi']['points']=[[0,0],[25,0],[25,100],[0,100],[0,0]]
            if mutation=='dimensions':v['width']=101
            p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
            with self.subTest(mutation=mutation):
                r=self.score(p);self.assertEqual(r['status'],'blocked')
                self.assertIn('shared_reference_content_mismatch',r['blocked_reason_counts'])

    def test_reference_freeze_and_micro_counts_are_order_invariant(self):
        p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='synthetic-u2',arm='second-arm')
        v['labels']['objects'].reverse();v['predictions'].reverse()
        vertices=v['roi']['points'][:-1];vertices=list(reversed(vertices[1:]+vertices[:1]))
        v['roi']['points']=vertices+[vertices[0]]
        p['units'].append(v);p['preregistered_units'].append(v['unit_id'])
        frozen=copy.deepcopy(p);a=self.score(p)
        self.assertEqual(a['status'],'synthetic_scored');self.assertEqual(p,frozen)
        p['units'].reverse();p['preregistered_units'].reverse()
        self.assertEqual(self.score(p),a)
        p=fixture();v=copy.deepcopy(p['units'][0]);v.update(unit_id='other',arm='other',split='holdout',lineage_ids=['synthetic-other'])
        p['units'].append(v);p['preregistered_units'].append('other')
        self.assertEqual(self.score(p)['status'],'blocked')

    def test_input_layers_and_counterfactual_scope(self):
        for layer in ['I0','I1-R','I1-R-no-RGB','I1-E']:
            p=fixture();p['units'][0]['input_layer']=layer;r=self.score(p)
            self.assertEqual(r['status'],'synthetic_scored')
            if layer=='I1-E':self.assertEqual(r['units'][0]['interpretation'],'configuration_execution_only')
        for patch in [dict(input_layer='unknown'),dict(task='entry_event'),
                      dict(target_answers_supplied=True),
                      dict(source_geometry_substitutes_observability=True),
                      dict(counterfactual_family='C')]:
            p=fixture();p['units'][0].update(patch)
            self.assertEqual(self.score(p)['status'],'blocked')
        p=fixture();p['units'][0].update(counterfactual_family='C',equivalent_evidence_reviewed=True)
        self.assertEqual(self.score(p)['status'],'synthetic_scored')

    def test_real_input_cannot_unlock_by_claiming_evidence_flags(self):
        p=fixture();p['kind']='real';p['human_GT']=True
        r=self.score(p);self.assertEqual(r['status'],'blocked')
        self.assertIsNone(r['accuracy_metrics']);self.assertFalse(r['scientific_ROI_admission'])

    def test_exact_iou_threshold_and_decimal_token(self):
        self.assertEqual(s.iou([0,0,2,2],[0,0,1,2]),Fraction(1,2))
        p=fixture();u=p['units'][0];u['labels']['objects']=[box('g',[0,0,2,2])]
        u['predictions']=[box('p',[0,0,1,2],route='keep')]
        self.assertEqual(self.score(p)['units'][0]['pairs'],[['g','p']])
        u['predictions'][0]['bbox'][2]=Decimal('0.9999999999999999999999999999')
        self.assertEqual(self.score(p)['units'][0]['pairs'],[])
        decoded=s.loads('{"x":0.1000000000000000000001}')
        self.assertEqual(decoded['x'],Decimal('0.1000000000000000000001'))

    def test_clipped_boxes_and_touch_do_not_invent_support(self):
        p=fixture();u=p['units'][0];u['labels']['objects']=[box('g',[-5,0,10,10])]
        u['predictions']=[box('p',[0,0,10,10],route='keep'),
                          box('touch',[50,30,60,40],route='suppress')]
        self.assertEqual(self.score(p)['status'],'blocked') # missing truncation marker
        u['labels']['objects'][0]['truncated']=True;r=self.score(p)
        self.assertEqual(r['units'][0]['pairs'],[['g','p']])
        self.assertEqual(self.metrics(r)['outside_suppression']['denominator'],0)
        audit=r['units'][0]['boxes']['gt']['g']
        self.assertEqual(audit['original'],['-5','0','10','10'])
        self.assertEqual(audit['clipped'],['0','0','10','10'])

    def test_concave_polygon_and_invalid_roi(self):
        p=fixture();u=p['units'][0]
        u['roi']['points']=[[0,0],[40,0],[40,20],[20,20],[20,40],[0,40],[0,0]]
        u['labels']['objects']=[box('g',[10,10,30,30])]
        u['predictions']=[box('p',[10,10,30,30],route='keep')]
        self.assertEqual(self.metrics(self.score(p))['boundary_retention']['value'],1)
        for polygon in [[[0,0],[40,40],[0,40],[40,0],[0,0]],
                        [[0,0],[10,0],[10,10]],
                        [[0,0],[101,0],[101,10],[0,0]]]:
            u['roi']['points']=polygon;self.assertEqual(self.score(p)['status'],'blocked')

    def test_duplicate_boxes_with_distinct_ids_are_not_deduplicated(self):
        p=fixture();u=p['units'][0];u['labels']['objects']=[box('a',[1,1,9,9]),box('b',[1,1,9,9])]
        u['predictions']=[box('a',[1,1,9,9],route='keep'),box('b',[1,1,9,9],route='keep')]
        self.assertEqual(self.score(p)['units'][0]['pairs'],[['a','a'],['b','b']])

    def test_strict_json_and_in_memory_float_fail_closed(self):
        for text in ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}']:
            with self.assertRaises(ValueError):s.loads(text)
        p=fixture();p['units'][0]['predictions'][0]['bbox'][0]=10.0
        self.assertEqual(self.score(p)['status'],'blocked')

    def test_cli_synthetic_only_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'synthetic.json';out=Path(t)/'result.json'
            source.write_text(json.dumps(fixture()))
            cmd=[sys.executable,str(Path(s.__file__)), '--input',str(source),'--out',str(out)]
            a=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(a.returncode,0,a.stderr)
            b=out.read_bytes();a=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(a.returncode,0);self.assertEqual(out.read_bytes(),b)
            source.write_text('{"kind":"real"}')
            cmd[-1]=str(Path(t)/'blocked.json');a=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(a.returncode,2,a.stderr)
            self.assertEqual(json.loads(Path(cmd[-1]).read_text())['status'],'blocked')


if __name__=='__main__':unittest.main()
