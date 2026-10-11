import copy
import importlib
import json
import unittest
from pathlib import Path
from learning.contract import RELATIONS, fingerprint, validate_manifest


def synthetic_preparation():
    """Self-authored metadata only; no real frame/label is manufactured."""
    base=json.loads((Path(__file__).resolve().parents[2]/'learning/data/software_manifest.json').read_text())
    row=base['samples'][0]
    versions=row['versions']
    return {'schema_version':'private-annotation-preparation.v1','purpose':'private_annotation_preparation_only',
        'source':{'publisher_url':'https://example.invalid/software-only','source_license_metadata':'self-authored software metadata; no media','prior_source_access_date_utc':'2026-10-11'},
        'samples':[{'sample_id':'software:frame0','frame_index':0,'source_member':'software/0.jpg','source_file':{'sha256':'a'*64,'bytes':20,'ref':'private-original.jpg'},
        'grouping':{'dataset':'software','sequence':'sequence0','camera':'camera0','building_id':None,'layout_family_id':None,'physical_door_ids':None,'source_plan_sha256':None,'variant_family_id':None,'split':None},
        'versions':copy.deepcopy(versions),'input_draft':{'views':[{'local_id':'v0','asset_ref':'private-original.jpg','sha256':'a'*64}], 'allowed_context':'Self-authored software fixture, no real media','regions':[{'local_id':'region0','description':'software A'},{'local_id':'region1','description':'software B'}],
         'door_and_passage_candidates':[{'local_id':'door0','kind':'AI_visible_passage_candidate','xyxy_normalized':[.1,.1,.8,.9],'physical_door_id':None,'building_exterior_status':'unknown'}],
         'region_pairs':[{'id':'r0','door_id':'door0','region_a':'region0','region_b':'region1'}],'view_pairs':[]},
        'ai_preannotation':{'producer_model_revision':None,'risk':'high','status':'proposed_pending_review','labels':{'relations':{'r0':'unknown'},'matches':{},'sufficiency':{'r0':'insufficient'}},'versions':copy.deepcopy(versions)},
        'existing_person_model_output':{'model_id':'software-person-only','model_revision':'software-person-revision','prompt':'person.','box_threshold':.25,'text_threshold':.25,'nms_iou':.7,'source_sha256':'a'*64,'kept_count':0,'door_model_detection':False},
        'review':{'status':'pending','reviewer':None,'evidence_refs':[],'versions':None},'truth':None}]}


def reviewed_software(preparation):
    """Hypothetical reviewed metadata, explicitly software-only, never real GT."""
    p=copy.deepcopy(preparation)
    for s in p['samples']:
        s['grouping'].update(building_id='software:building0',layout_family_id='software:layout0',physical_door_ids=['software:physicaldoor0'],variant_family_id='software:variant0',split='train')
        s['review']={'status':'confirmed','reviewer':'software-test-reviewer','evidence_refs':['software-only-proof'],'versions':copy.deepcopy(s['versions'])}
        s['truth']=copy.deepcopy(s['ai_preannotation']['labels'])
    return p


class AdmissionTests(unittest.TestCase):
    def adapter(self):
        module=importlib.import_module('learning_adapter.admission')
        for name in ['adapt_preparation','admit_supervision','public_index']:
            self.assertTrue(callable(getattr(module,name,None)),name+' behavior missing')
        return module

    def ready(self):
        a=self.adapter();p=reviewed_software(synthetic_preparation());index=a.adapt_preparation(p)
        row=copy.deepcopy(index['samples'][0]['boundary_v1_template']);row['source']['permission_status']='verified'
        return a,index,{'schema_version':'boundary-v1','purpose':'real_research','samples':[row]}

    def test_pending_high_risk_cannot_supervise(self):
        a=self.adapter();index=a.adapt_preparation(synthetic_preparation());row=index['samples'][0]
        self.assertIsNone(row['boundary_v1_template']['truth'])
        self.assertFalse(row['supervision_ready'])
        with self.assertRaises(ValueError):a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[row['boundary_v1_template']]},index)

    def test_ai_default_adoption_does_not_make_truth(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['ai_preannotation']['status']='default_adopted_pending_review'
        self.assertIsNone(a.adapt_preparation(p)['samples'][0]['boundary_v1_template']['truth'])

    def test_unknown_geometry_revision_not_filled_with_person_model(self):
        a=self.adapter();r=a.adapt_preparation(synthetic_preparation())['samples'][0]
        self.assertIsNone(r['boundary_v1_template']['ai_preannotation'])
        self.assertEqual(r['provenance']['person_detector']['revision_status'],'known')
        self.assertEqual(r['provenance']['geometry_AI']['revision_status'],'unknown')
        self.assertEqual(r['provenance']['AI_labels']['relations']['r0'],'unknown')

    def test_known_geometry_revision_uses_existing_ai_schema(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['ai_preannotation']['producer_model_revision']='software-geometry-revision'
        r=a.adapt_preparation(p)['samples'][0]['boundary_v1_template']
        self.assertEqual(r['ai_preannotation']['model_revision'],'software-geometry-revision')
        self.assertEqual(r['ai_preannotation']['status'],'proposed')

    def test_reviewed_unknown_remains_third_class(self):
        a,index,m=self.ready();result=a.admit_supervision(m,index)
        self.assertEqual(result['label_indices']['software:frame0']['relations'],{'r0':RELATIONS.index('unknown')})
        self.assertNotEqual(result['label_indices']['software:frame0']['relations']['r0'],RELATIONS.index('not_connected'))

    def test_reviewed_known_label_keeps_existing_encoding(self):
        a,index,m=self.ready();m['samples'][0]['truth']['relations']['r0']='connected'
        self.assertEqual(a.admit_supervision(m,index)['label_indices']['software:frame0']['relations']['r0'],1)

    def test_normal_risk_does_not_bypass_pending_review(self):
        a,index,m=self.ready();index['samples'][0]['provenance']['risk']='normal';m['samples'][0]['review']={'status':'pending','reviewer':None,'evidence_refs':[],'versions':None};m['samples'][0]['truth']=None
        with self.assertRaisesRegex(ValueError,'human review'):a.admit_supervision(m,index)

    def test_stale_review_version_is_rejected(self):
        a,index,m=self.ready();m['samples'][0]['review']['versions']={'evidence':'stale','labels':'stale','rules':'stale'}
        with self.assertRaises(ValueError):a.admit_supervision(m,index)

    def test_changed_input_without_readaptation_is_rejected(self):
        a,index,m=self.ready();m['samples'][0]['input']['door_candidates'][0]['box']=[.2,.2,.7,.8]
        with self.assertRaisesRegex(ValueError,'input binding'):a.admit_supervision(m,index)

    def test_missing_groups_cannot_be_filled_with_placeholders(self):
        a,index,m=self.ready();index['samples'][0]['declared_identity']['building_id']=None
        with self.assertRaises(ValueError):a.admit_supervision(m,index)
        for value in ['unknown','unknown:camera0','pending','TBD']:
            with self.subTest(value=value):
                _,i,x=self.ready();i['samples'][0]['declared_identity']['building_id']=value;x['samples'][0]['identity']['building_id']=value
                with self.assertRaises(ValueError):a.admit_supervision(x,i)

    def test_same_sequence_frames_cannot_cross_split_even_renamed_groups(self):
        a=self.adapter();p=reviewed_software(synthetic_preparation());second=copy.deepcopy(p['samples'][0]);second['sample_id']='software:frame1';second['frame_index']=1;second['source_file']['sha256']='b'*64;second['source_file']['ref']='private-other.jpg';second['input_draft']['views'][0].update(asset_ref='private-other.jpg',sha256='b'*64)
        second['existing_person_model_output']['source_sha256']='b'*64
        second['grouping'].update(building_id='software:building1',layout_family_id='software:layout1',physical_door_ids=['software:physicaldoor1'],variant_family_id='software:variant1',split='test')
        p['samples'].append(second);index=a.adapt_preparation(p);rows=[copy.deepcopy(x['boundary_v1_template']) for x in index['samples']]
        for row in rows:row['source']['permission_status']='verified'
        m={'schema_version':'boundary-v1','purpose':'real_research','samples':rows}
        validate_manifest(m) # Original f23 cannot detect different media and renamed IDs by itself.
        with self.assertRaisesRegex(ValueError,'source sequence.*cross-split'):a.admit_supervision(m,index)

    def two_frame_software_index(self, second_split, same_sequence=True):
        a=self.adapter();p=reviewed_software(synthetic_preparation())
        second=copy.deepcopy(p['samples'][0]);second['sample_id']='software:frame1';second['frame_index']=1
        second['source_file'].update(sha256='b'*64,ref='private-other.jpg')
        second['input_draft']['views'][0].update(asset_ref='private-other.jpg',sha256='b'*64)
        second['existing_person_model_output']['source_sha256']='b'*64
        second['grouping'].update(building_id='software:building1',layout_family_id='software:layout1',
            physical_door_ids=['software:physicaldoor1'],variant_family_id='software:variant1',split=second_split)
        if not same_sequence:second['grouping']['sequence']='different-software-sequence'
        p['samples'].append(second);index=a.adapt_preparation(p)
        rows=[copy.deepcopy(x['boundary_v1_template']) for x in index['samples']]
        for row in rows:row['source']['permission_status']='verified'
        return a,index,rows

    def test_I1_conflicting_full_index_rejects_train_only_subset(self):
        a,index,rows=self.two_frame_software_index('test')
        with self.assertRaisesRegex(ValueError,'source sequence.*cross-split'):
            a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[rows[0]]},index)

    def test_I1_conflicting_full_index_rejects_test_only_subset(self):
        a,index,rows=self.two_frame_software_index('test')
        with self.assertRaisesRegex(ValueError,'source sequence.*cross-split'):
            a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[rows[1]]},index)

    def test_I1_same_source_unassigned_neighbor_blocks_selected_train(self):
        a,index,rows=self.two_frame_software_index(None)
        with self.assertRaisesRegex(ValueError,'source sequence.*unassigned'):
            a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[rows[0]]},index)

    def test_I1_consistent_full_index_allows_two_separate_subsets(self):
        a,index,rows=self.two_frame_software_index('train')
        for row in rows:
            with self.subTest(sample=row['id']):
                result=a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[row]},index)
                self.assertEqual(set(result['label_indices']),{row['id']})
                self.assertEqual(result['source_components'],1)

    def test_I1_unrelated_unassigned_component_is_not_supervised(self):
        a,index,rows=self.two_frame_software_index(None,same_sequence=False)
        manifest={'schema_version':'boundary-v1','purpose':'real_research','samples':[rows[0]]}
        self.assertEqual(set(a.admit_supervision(manifest,index)['label_indices']),{rows[0]['id']})
        manifest['samples']=[rows[1]]
        with self.assertRaises(ValueError):a.admit_supervision(manifest,index)

    def test_same_sequence_all_in_one_split_allowed_software_case(self):
        a,index,m=self.ready();second=copy.deepcopy(m['samples'][0]);second['id']='software:frame1';second['input']['views'][0]['sha256']='b'*64
        idx=copy.deepcopy(index['samples'][0]);idx['sample_id']='software:frame1';idx['boundary_v1_template']=copy.deepcopy(second);idx['input_fingerprint']=fingerprint(second['input']);index['samples'].append(idx);m['samples'].append(second)
        self.assertEqual(len(a.admit_supervision(m,index)['label_indices']),2)

    def test_unapproved_source_permission_rejected(self):
        a,index,m=self.ready();m['samples'][0]['source']['permission_status']='pending'
        with self.assertRaises(ValueError):a.admit_supervision(m,index)

    def test_broken_person_source_binding_rejected(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['existing_person_model_output']['source_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'person.*source'):a.adapt_preparation(p)

    def test_real_door_detection_cannot_be_invented(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['existing_person_model_output']['door_model_detection']=True
        with self.assertRaisesRegex(ValueError,'person.*door'):a.adapt_preparation(p)

    def test_public_projection_omits_private_coordinates_paths_boxes(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['existing_person_model_output']['model_boxes_after_nms']=[{'bbox':[1,2,3,4]}]
        public=a.public_index(a.adapt_preparation(p));text=json.dumps(public)
        self.assertNotIn('private-original.jpg',text);self.assertNotIn('xyxy',text);self.assertNotIn('model_boxes_after_nms',text)
        self.assertEqual(public['samples'][0]['source_sha256'],'a'*64)
        self.assertFalse(public['samples'][0]['supervision_ready'])

    def test_duplicate_samples_and_bad_coordinates_rejected(self):
        a=self.adapter();p=synthetic_preparation();p['samples'].append(copy.deepcopy(p['samples'][0]))
        with self.assertRaises(ValueError):a.adapt_preparation(p)
        p=synthetic_preparation();p['samples'][0]['input_draft']['door_and_passage_candidates'][0]['xyxy_normalized']=[-1,0,1,1]
        with self.assertRaises(ValueError):a.adapt_preparation(p)

    def test_ai_label_versions_or_classes_invalid_rejected(self):
        a=self.adapter();p=synthetic_preparation();p['samples'][0]['ai_preannotation']['labels']['relations']['r0']='safe'
        with self.assertRaises(ValueError):a.adapt_preparation(p)
        p=synthetic_preparation();p['samples'][0]['ai_preannotation']['versions']['rules']='stale'
        with self.assertRaises(ValueError):a.adapt_preparation(p)

    def test_preparation_is_not_silently_declared_formal_manifest(self):
        a=self.adapter();index=a.adapt_preparation(synthetic_preparation())
        self.assertFalse(index['formal_manifest'])
        with self.assertRaises(ValueError):validate_manifest(index)

    def test_review_split_must_match_declared_assignment(self):
        a,index,m=self.ready();m['samples'][0]['split']='test'
        with self.assertRaisesRegex(ValueError,'declared split'):a.admit_supervision(m,index)

    def test_identity_change_without_source_readaptation_rejected(self):
        a,index,m=self.ready();m['samples'][0]['identity']['building_id']='software:other-building'
        with self.assertRaisesRegex(ValueError,'identity binding'):a.admit_supervision(m,index)

    def test_human_labels_need_recorded_evidence(self):
        a,index,m=self.ready();m['samples'][0]['review']['evidence_refs']=[]
        with self.assertRaisesRegex(ValueError,'review evidence'):a.admit_supervision(m,index)

    def test_empty_supervision_is_rejected(self):
        a=self.adapter()
        with self.assertRaises(ValueError):a.admit_supervision({'schema_version':'boundary-v1','purpose':'real_research','samples':[]},a.adapt_preparation(synthetic_preparation()))

if __name__=='__main__':unittest.main()
