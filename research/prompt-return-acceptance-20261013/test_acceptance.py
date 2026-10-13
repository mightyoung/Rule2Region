"""Offline contract tests; synthetic corruptions are not research truth."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import audit_returns as audit
import check_private_return as private
REPO=Path(__file__).resolve().parents[2]
class Acceptance(unittest.TestCase):
 def test_pinned_public_returns(self):
  report,receipts,_=audit.audit(REPO)
  self.assertEqual(report['verified_receipts'],19);self.assertEqual(len(receipts),19)
  self.assertFalse(report['private_original_package_verified']);self.assertFalse(report['human_GT'])
 def test_synthetic_receipt_byte_corruption(self):
  original=audit.blob
  def changed(repo,commit,path):
   b=original(repo,commit,path)
   return b+b' ' if path.endswith('cpu-matrix/C1-f00000000--combined.json') else b
  with patch.object(audit,'blob',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'byte/hash'):audit.audit(REPO)
 def test_synthetic_scope_escalation(self):
  original=audit.blob
  def changed(repo,commit,path):
   b=original(repo,commit,path)
   if path.endswith('cpu-operator-run-1/aggregate.json'):
    x=json.loads(b);x['decision']='safe';return json.dumps(x).encode()
   return b
  with patch.object(audit,'blob',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'scope drift'):audit.audit(REPO)
 def test_synthetic_duplicate_arm(self):
  original=audit.blob
  def changed(repo,commit,path):
   b=original(repo,commit,path)
   if path.endswith('sanitized-receipts/MANIFEST.json'):
    x=json.loads(b);x['runs']['cpu-matrix']['receipts'][1]=x['runs']['cpu-matrix']['receipts'][0];return json.dumps(x).encode()
   return b
  with patch.object(audit,'blob',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'duplicate'):audit.audit(REPO)
 def test_missing_private_package_is_incomplete(self):
  with tempfile.TemporaryDirectory() as t:
   report,review=private.inspect(REPO,Path(t)/'missing','cpu-matrix')
   self.assertEqual(report['status'],'incomplete_private_return');self.assertEqual(len(report['missing']),69)
   self.assertEqual(len(review),13);self.assertFalse(report['scientific_ROI_admission'])
   self.assertTrue(all(x['review_status']=='unreviewed' and x['threshold_location']=='unknown' for x in review))
 def test_synthetic_raw_tensor_hash_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'C1-f00000000--combined';p.mkdir();(p/'DOOR_RAW_OUTPUT.safetensors').write_bytes(b'SYNTHETIC_CORRUPTION_NOT_A_MODEL_TENSOR')
   with self.assertRaisesRegex(ValueError,'declared public hash'):private.inspect(REPO,Path(t),'cpu-matrix')
 def test_synthetic_private_receipt_substitution_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'C1-f00000000--combined';p.mkdir();(p/'DOOR_PROBE_RECEIPT_PRIVATE.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'pinned public bytes'):private.inspect(REPO,Path(t),'cpu-matrix')
if __name__=='__main__':unittest.main()
