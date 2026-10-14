"""Synthetic review edits only; no actual reviewer or newly confirmed geometry."""
import copy,datetime,unittest,json,subprocess,sys
import validate_human_review as h
import audit_returns,check_private_return
from pathlib import Path
import tempfile
REPO=Path(__file__).resolve().parents[2]
class ReviewContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with tempfile.TemporaryDirectory() as t:
   _,records=check_private_return.inspect(REPO,Path(t),'cpu-matrix')
  cls.template={'schema_version':1,'human_GT':False,'AI_assisted_review_not_blind_GT':True,'records':records}
 def edited(self):
  x=copy.deepcopy(self.template);r=x['records'][0];r.update(review_status='partial_review',reviewer='SYNTHETIC TEST ONLY',reviewed_utc='2026-10-13T00:00:00Z',viewed_original_sha256=r['source_sha256'],seen_AI_candidates=True,note='Synthetic partial visibility, not research outcome',left_visibility='visible',bottom_visibility='not_visible');return x
 def check(self,x):return h.validate(self.template,x,datetime.datetime(2026,10,13,1,tzinfo=datetime.timezone.utc))
 def integrity_fixture(self):
  return {'run_name':'cpu-matrix','status':'private_bytes_and_original_validator_checks_passed','missing':[],'raw_and_processed_tensor_validation':'passed; normal padded -inf logits allowed','public_receipts_verified':True,'human_GT':False,'decision':'unknown','scientific_ROI_admission':False}
 def test_review_requires_private_integrity_report(self):
  with self.assertRaisesRegex(ValueError,'integrity report'):h.check_review_readiness(self.edited(),'cpu-matrix',None)
 def test_review_rejects_missing_raw_failed_or_running_return(self):
  for state in ['incomplete_private_return','failed_private_return']:
   r=self.integrity_fixture();r['status']=state
   with self.assertRaises(ValueError):h.check_review_readiness(self.edited(),'cpu-matrix',r)
  for field,value in [('missing',['original/C1-f00000000.jpg']),('raw_and_processed_tensor_validation','not_run_missing_bytes')]:
   r=self.integrity_fixture();r[field]=value
   with self.assertRaises(ValueError):h.check_review_readiness(self.edited(),'cpu-matrix',r)
 def test_review_rejects_other_matrix_or_GT_escalation(self):
  for field,value in [('run_name','gpu-matrix'),('human_GT',True),('scientific_ROI_admission',True),('decision','safe')]:
   r=self.integrity_fixture();r[field]=value
   with self.assertRaises(ValueError):h.check_review_readiness(self.edited(),'cpu-matrix',r)
 def test_synthetic_passed_integrity_allows_partial_contract_only(self):
  self.assertEqual(h.check_review_readiness(self.edited(),'cpu-matrix',self.integrity_fixture()),'report_checked_for_partial_review')
  r=self.check(self.edited());self.assertFalse(r['scientific_ROI_admission']);self.assertFalse(r['human_GT'])
 def test_unedited_template_needs_no_materials_and_stays_unreviewed(self):
  self.assertEqual(h.check_review_readiness(self.template,'cpu-matrix',None),'not_required_for_unreviewed_template')
 def test_CLI_rejects_filled_review_without_integrity_before_writing(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);template=root/'template.json';review=root/'review.json';out=root/'report.json'
   template.write_text(json.dumps(self.template));review.write_text(json.dumps(self.edited()))
   result=subprocess.run([sys.executable,str(Path(h.__file__)), '--repo',str(REPO),'--run-name','cpu-matrix','--template',str(template),'--review',str(review),'--out',str(out)],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertIn('require a private integrity report',result.stderr);self.assertFalse(out.exists())
 def test_bind_template_to_actual_pinned_receipts(self):
  h.bind_template(copy.deepcopy(self.template),REPO,'cpu-matrix')
  x=copy.deepcopy(self.template);x['records'][0]['AI_score']=.999
  with self.assertRaisesRegex(ValueError,'pinned'):h.bind_template(x,REPO,'cpu-matrix')
 def test_unchanged_template_remains_unreviewed(self):
  r=self.check(copy.deepcopy(self.template));self.assertEqual(r['review_status_counts']['unreviewed'],13);self.assertFalse(r['human_GT'])
 def test_synthetic_partial_visible_edge_allowed(self):
  x=self.edited();x['records'][0]['coordinates_added_by_human']={'left':[[10,20],[12,200]]};self.assertEqual(self.check(x)['review_status_counts']['partial_review'],1)
 def test_reject_unknown_bottom_coordinates(self):
  x=self.edited();x['records'][0]['coordinates_added_by_human']={'bottom':[[0,600],[800,600]]}
  with self.assertRaisesRegex(ValueError,'invisible'):self.check(x)
 def test_reject_full_ROI_and_safe_events(self):
  for field,value in [('full_opening_geometry_confirmed',True),('events','safe'),('threshold_location',[10,600])]:
   x=self.edited();x['records'][0][field]=value
   with self.assertRaises(ValueError):self.check(x)
 def test_reject_AI_and_source_substitution(self):
  for field,value in [('AI_score',.99),('source_sha256','0'*64),('AI_bbox_xyxy_original_unclipped',[0,0,800,600])]:
   x=self.edited();x['records'][0][field]=value
   with self.assertRaises(ValueError):self.check(x)
 def test_reject_missing_candidate(self):
  x=self.edited();x['records'].pop()
  with self.assertRaisesRegex(ValueError,'missing'):self.check(x)
 def test_reject_forged_blind_GT(self):
  for change in ['blind','GT']:
   x=self.edited()
   if change=='blind':x['records'][0]['seen_AI_candidates']=False
   else:x['human_GT']=True
   with self.assertRaises(ValueError):self.check(x)
 def test_reject_future_or_missing_reviewer(self):
  for field,value in [('reviewed_utc','2099-01-01T00:00:00Z'),('reviewer',None),('viewed_original_sha256','0'*64)]:
   x=self.edited();x['records'][0][field]=value
   with self.assertRaises(ValueError):self.check(x)
 def test_reject_nan_closed_and_outside_fragment(self):
  for pts in [[[10,0],[float('nan'),20]],[[0,0],[0,0]],[[-1,0],[10,20]]]:
   x=self.edited();x['records'][0]['coordinates_added_by_human']={'left':pts}
   with self.assertRaises(ValueError):self.check(x)
 def test_unreviewed_record_cannot_hide_changes(self):
  x=copy.deepcopy(self.template);x['records'][0]['left_visibility']='visible'
  with self.assertRaisesRegex(ValueError,'unchanged'):self.check(x)
if __name__=='__main__':unittest.main()
