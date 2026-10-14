"""Synthetic partial/failed return states only; no server/model execution."""
import copy,json,tempfile,unittest
from pathlib import Path
import audit_returns as audit
import check_private_return as p
REPO=Path(__file__).resolve().parents[2]
class ReturnState(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  _,_,aggs=audit.audit(REPO);cls.aggregate=aggs['cpu-matrix']
  cls.hashes={f:audit.digest(audit.blob(REPO,cls.aggregate['git_commit'],audit.BASE+f)) for f in ['input_manifest.json','model_manifest.json','experiment.json','forward.py','handoff.py']}
 def fixture(self):
  return {'schema_version':1,'status':'failed','kind':'real_prompt_probe','device':'cpu','git_commit':self.aggregate['git_commit'],'decision':'unknown','human_GT':False,'manifest_hashes':self.hashes.copy(),'failure_type':'SYNTHETIC_CHILD_FAILURE','failure_reason':'Synthetic unit-test fixture; no experiment ran','trials':[{'source_id':'C1-f00000000','arm':'combined','decision':'unknown','status':'running','exit_code':1}]}
 def test_failed_state_preserved_when_other_files_missing(self):
  with tempfile.TemporaryDirectory() as t:
   Path(t,'RUN_PRIVATE.json').write_text(json.dumps(self.fixture()))
   report,review=p.inspect(REPO,Path(t),'cpu-matrix')
   self.assertEqual(report['status'],'failed_private_return');self.assertEqual(report['run_state']['status'],'failed');self.assertFalse(report['scientific_ROI_admission']);self.assertEqual(len(review),13)
   self.assertTrue(report['missing']);self.assertFalse(Path(t,'PUBLIC_AGGREGATE.json').exists())
 def test_wrong_version_rejected_even_in_partial_package(self):
  with tempfile.TemporaryDirectory() as t:
   r=self.fixture();r['schema_version']=2;Path(t,'RUN_PRIVATE.json').write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'schema'):p.inspect(REPO,Path(t),'cpu-matrix')
 def test_synthetic_wrong_hash_version_or_GT(self):
  for field,value in [('git_commit','0'*40),('human_GT',True),('decision','safe')]:
   r=self.fixture();r[field]=value
   with self.assertRaises(ValueError):p.validate_return_state(r,self.aggregate,self.hashes)
  r=self.fixture();r['manifest_hashes']['forward.py']='0'*64
  with self.assertRaisesRegex(ValueError,'hash'):p.validate_return_state(r,self.aggregate,self.hashes)
 def test_completed_state_cannot_be_partial(self):
  r=self.fixture();r['status']='completed'
  with self.assertRaisesRegex(ValueError,'complete'):p.validate_return_state(r,self.aggregate,self.hashes)
 def test_failure_cannot_hide_duplicate_or_safe_trial(self):
  r=self.fixture();r['trials'].append(copy.deepcopy(r['trials'][0]))
  with self.assertRaisesRegex(ValueError,'duplicate'):p.validate_return_state(r,self.aggregate,self.hashes)
  r=self.fixture();r['trials'][0]['decision']='safe'
  with self.assertRaises(ValueError):p.validate_return_state(r,self.aggregate,self.hashes)
 def test_running_remains_incomplete_not_success(self):
  r=self.fixture();r['status']='running';state=p.validate_return_state(r,self.aggregate,self.hashes)
  self.assertEqual(state['status'],'running');self.assertFalse(state['successful_matrix'])
if __name__=='__main__':unittest.main()
