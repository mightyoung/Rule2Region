"""Synthetic review edits only; no actual reviewer or newly confirmed geometry."""
import copy,datetime,unittest
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
