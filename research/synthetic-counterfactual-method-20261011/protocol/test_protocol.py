"""Software-only tests; no visual labels or model evaluation."""
import copy
import unittest
import protocol

class ProtocolTests(unittest.TestCase):
    def setUp(self): self.m = protocol.make_template()
    def rejects(self, edit):
        m=copy.deepcopy(self.m); edit(m)
        with self.assertRaises(ValueError): protocol.validate_manifest(m)
    def test_empty_sixteen_records_valid(self):
        protocol.validate_manifest(self.m)
        self.assertEqual(len(self.m['records']),16)
        self.assertEqual(len(set(protocol.source_components(self.m).values())),4)
        self.assertTrue(all(r['split'] is None for r in self.m['records']))
    def test_coverage_missing_record_rejected(self): self.rejects(lambda m:m['records'].pop())
    def test_duplicate_sample_rejected(self): self.rejects(lambda m:m['records'][0].update(sample_id=m['records'][1]['sample_id']))
    def test_assigned_split_rejected(self): self.rejects(lambda m:m['records'][0].update(split='train'))
    def test_training_admission_rejected(self): self.rejects(lambda m:m['training_admitted_ids'].append(m['records'][0]['sample_id']))
    def test_effect_row_rejected(self): self.rejects(lambda m:m['effect_rows'].append({'accuracy':1}))
    def test_independence_claim_rejected(self): self.rejects(lambda m:m.update(independent_sample_count=16))
    def test_unseen_image_cannot_be_reviewed(self): self.rejects(lambda m:m['records'][0]['image_evidence'].update(status='known'))
    def test_unseen_decision_rejected(self): self.rejects(lambda m:m['records'][0]['decision'].update(I0='outside'))
    def test_fabricated_geometry_truth_rejected(self): self.rejects(lambda m:m['records'][0]['source_geometry_truth'].update(endpoint_S_membership=['inside','outside']))
    def test_fabricated_policy_truth_rejected(self): self.rejects(lambda m:m['records'][0]['source_policy_truth'].update(count_boundary_crossing=True))
    def test_unknown_membership_not_false(self): self.assertEqual(protocol.execute_configuration(None),'unknown')
    def test_xor_is_software_configuration_only(self):
        for a,b,want in [('inside','inside',False),('outside','outside',False),('inside','outside',True),('outside','inside',True),('unknown','inside','unknown')]:
            self.assertEqual(protocol.execute_configuration({'B_version':'toy-software-test-v1','endpoint_B_membership':[a,b]}),want)
    def test_I0_export_has_no_source_or_family(self):
        d=protocol.export_inputs(self.m['records'][0],'I0')
        self.assertEqual(set(d),{'rgb','generic_task','neutral_door_roi'})
        self.assertNotIn('source_geometry_truth',d)
    def test_I0_extra_answer_rejected(self): self.rejects(lambda m:m['records'][0]['input'].update(target_answer='outside'))
    def test_I1R_direct_answer_rejected(self): self.rejects(lambda m:m['records'][0]['context']['I1-R'].update(endpoint_B_membership=['inside','outside']))
    def test_I1R_marked_answer_bearing_map_rejected(self):
        def edit(m):
            m['records'][0]['context']['I1-R']['map']={'reference':'toy-map','version':'v1','provenance':'runtime_available','review_status':'confirmed','contains_target_endpoint_membership':True}
        self.rejects(edit)
    def test_availability_cannot_claim_absent_input(self): self.rejects(lambda m:m['records'][0]['input_available']['I0'].update(rgb=True))
    def test_I1E_baseline_has_same_config_no_rgb(self):
        r=self.m['records'][0]
        r['context']['I1-E']={'B_version':'toy-software-test-v1','endpoint_B_membership':['inside','outside']}
        a=protocol.export_inputs(r,'I1-E');b=protocol.export_inputs(r,'I1-E-no-RGB')
        self.assertEqual(a['configuration'],b['configuration']);self.assertNotIn('rgb',b)
        self.assertEqual(protocol.execute_configuration(a['configuration']),protocol.execute_configuration(b['configuration']))
    def test_D_pair_reuses_rgb_keys_and_I0_context(self):
        for side in [0,1]:
            rs=[r for r in self.m['records'] if r['pair']['family']=='D' and r['pair']['side']==side]
            self.assertEqual(rs[0]['source_links']['rgb_reuse_id'],rs[1]['source_links']['rgb_reuse_id'])
            self.assertEqual(protocol.export_inputs(rs[0],'I0'),protocol.export_inputs(rs[1],'I0'))
    def test_D_reuse_obligation_cannot_be_removed(self):
        def edit(m):
            r=next(r for r in m['records'] if r['pair']=={'family':'D','condition':1,'side':0})
            r['source_links']['rgb_reuse_id']='other'
        self.rejects(edit)
    def test_transitive_layout_ancestry_merges_components(self):
        m=copy.deepcopy(self.m)
        m['source_registry']['planned_B']['parents']=['planned_A']
        m['source_registry']['planned_C']['parents']=['planned_B']
        groups=protocol.source_components(m)
        self.assertEqual(len(set(groups.values())),2)
        # A forged old split component must not be accepted.
        with self.assertRaises(ValueError): protocol.validate_manifest(m)
        for r in m['records']:r['component_group']=groups[r['sample_id']]
        protocol.validate_manifest(m)
    def test_reused_decoded_pixels_merge_different_layouts(self):
        m=copy.deepcopy(self.m)
        for r in [m['records'][0],m['records'][4]]:r['input']['rgb']={'reference':'toy-placeholder-not-rendered.png','decoded_rgb_sha256':'a'*64}
        groups=protocol.source_components(m)
        self.assertEqual(groups[m['records'][0]['sample_id']],groups[m['records'][4]['sample_id']])
    def test_v1_origin_must_be_declared(self):
        self.rejects(lambda m:m['source_registry']['planned_A'].update(origin='v1'))
    def test_v1_shared_origin_group_merges_with_derived_layout(self):
        m=copy.deepcopy(self.m)
        m['source_registry']['prior_v1']={'origin':'v1','parents':[],'v1_source_group':'toy-v1-source-group'}
        m['source_registry']['planned_A']['parents']=['prior_v1']
        m['source_registry']['planned_C']['parents']=['prior_v1']
        groups=protocol.source_components(m)
        self.assertEqual(groups[m['records'][0]['sample_id']],groups[m['records'][8]['sample_id']])
        for r in m['records']:r['component_group']=groups[r['sample_id']]
        protocol.validate_manifest(m)
    def test_export_rejects_unvalidated_answer_field(self):
        r=copy.deepcopy(self.m['records'][0]);r['input']['target_answer']='outside'
        with self.assertRaises(ValueError):protocol.export_inputs(r,'I0')
    def test_D_software_policy_response_does_not_change_I0(self):
        rs=[copy.deepcopy(r) for r in self.m['records'] if r['pair']['family']=='D' and r['pair']['side']==0]
        rs[0]['context']['I1-E']={'B_version':'toy-v0','endpoint_B_membership':['inside','outside']}
        rs[1]['context']['I1-E']={'B_version':'toy-v1','endpoint_B_membership':['inside','inside']}
        self.assertEqual(protocol.export_inputs(rs[0],'I0'),protocol.export_inputs(rs[1],'I0'))
        self.assertEqual([protocol.execute_configuration(r['context']['I1-E']) for r in rs],[True,False])
        self.assertEqual(rs[0]['source_geometry_truth'],rs[1]['source_geometry_truth'])
    def test_reused_RGB_stale_components_rejected_after_availability_updated(self):
        m=copy.deepcopy(self.m)
        for r in [m['records'][0],m['records'][4]]:
            r['input']['rgb']={'reference':'toy-placeholder-not-rendered.png','decoded_rgb_sha256':'a'*64}
            r['input_available']=protocol.available(r)
        with self.assertRaisesRegex(ValueError,'component'):protocol.validate_manifest(m)
    def test_v1_group_identity_retained(self):
        m=copy.deepcopy(self.m)
        m['source_registry']['prior_v1']={'origin':'v1','parents':[],'v1_source_group':'toy-v1-source-group'}
        m['source_registry']['planned_A']['parents']=['prior_v1']
        self.assertEqual(protocol.source_components(m)[m['records'][0]['sample_id']],'toy-v1-source-group')
    def test_global_export_rejects_subset(self):
        m=copy.deepcopy(self.m);m['records']=m['records'][:4]
        with self.assertRaises(ValueError):protocol.export_manifest(m)
    def test_export_skeletons_keep_sixteen_records_and_no_rgb_baseline(self):
        out=protocol.export_manifest(self.m)
        self.assertEqual(len(out['records']),16)
        for r in out['records']:
            self.assertEqual(set(r['profiles']),set(protocol.PROFILES))
            self.assertNotIn('rgb',r['profiles']['I1-E-no-RGB'])
            self.assertEqual(r['profiles']['I1-E']['configuration'],r['profiles']['I1-E-no-RGB']['configuration'])
    def test_source_registry_cycle_rejected(self): self.rejects(lambda m:m['source_registry']['planned_A'].update(parents=['planned_A']))
    def test_S_and_B_definition_confirmation_not_invented(self): self.rejects(lambda m:m['definitions']['B'].update(business_confirmed=True))
    def test_B_does_not_auto_include_yards(self): self.assertFalse(self.m['definitions']['B']['auto_include_yards_or_walkways'])

if __name__=='__main__': unittest.main()
