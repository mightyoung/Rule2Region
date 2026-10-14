"""Tiny synthetic file/archive fixtures, never real media or model tensors."""
import hashlib,json,tarfile,tempfile,unittest
from pathlib import Path
import pack_private_return as pack
class PrivateArchive(unittest.TestCase):
 def fixture(self,root):
  spec=[]
  for n in [1,2,3]:
   sid=f'C{n}-f00000000';b=('SYNTHETIC ORIGINAL '+sid).encode();p=root/'data'/(sid+'.jpg');p.parent.mkdir(exist_ok=True);p.write_bytes(b);spec.append({'id':sid,'sha256':hashlib.sha256(b).hexdigest()})
  license_bytes=b'SYNTHETIC LICENSE TEST ONLY';(root/'data/SOURCE_LICENSE_ORIGINAL.html').write_bytes(license_bytes)
  for f in ['RUN_PRIVATE.json','MODEL_ENV_PRIVATE.json']:
   p=root/'cpu-matrix'/f;p.parent.mkdir(exist_ok=True);p.write_bytes(b'SYNTHETIC NON-VALIDATED STATE')
  for x in spec:
   for arm in ['combined','door_only','doorway_only']:
    d=root/'cpu-matrix'/(x['id']+'--'+arm);d.mkdir()
    for f in pack.ARTIFACTS:(d/f).write_bytes(b'SYNTHETIC NOT REAL '+f.encode())
  return spec,hashlib.sha256(license_bytes).hexdigest()
 def test_synthetic_archive_contains_all_three_originals(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);spec,license_sha=self.fixture(root);out=root/'new.tgz';m=pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha)
   with tarfile.open(out) as a:
    for x in spec:self.assertEqual(hashlib.sha256(a.extractfile('data/'+x['id']+'.jpg').read()).hexdigest(),x['sha256'])
    manifest=json.load(a.extractfile('RETURN_ARCHIVE_MANIFEST_PRIVATE.json'));self.assertTrue(manifest['complete_required_file_set']);self.assertFalse(manifest['artifact_contents_validated'])
   self.assertEqual(len(m['files']),69)
 def test_missing_each_original_fails_before_archive_creation(self):
  for n in [1,2,3]:
   with tempfile.TemporaryDirectory() as t:
    root=Path(t);spec,license_sha=self.fixture(root);(root/'data'/f'C{n}-f00000000.jpg').unlink();out=root/'new.tgz'
    with self.assertRaisesRegex(ValueError,'missing/nonregular original'):pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha)
    self.assertFalse(out.exists())
 def test_wrong_source_hash_and_existing_archive_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);spec,license_sha=self.fixture(root);out=root/'new.tgz';spec[0]['sha256']='0'*64
   with self.assertRaisesRegex(ValueError,'source hash'):pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha)
   out.write_bytes(b'OLD SYNTHETIC EVIDENCE')
   with self.assertRaisesRegex(ValueError,'archive exists'):pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha)
   self.assertEqual(out.read_bytes(),b'OLD SYNTHETIC EVIDENCE')
 def test_partial_failure_requires_explicit_flag_and_is_not_acceptance(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);spec,license_sha=self.fixture(root);(root/'cpu-matrix/MODEL_ENV_PRIVATE.json').unlink();out=root/'partial.tgz'
   with self.assertRaisesRegex(ValueError,'incomplete'):pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha)
   m=pack.pack(root,'cpu-matrix',out,source_spec=spec,license_sha=license_sha,allow_partial=True);self.assertFalse(m['complete_required_file_set']);self.assertFalse(m['artifact_contents_validated']);self.assertTrue(m['missing'])
if __name__=='__main__':unittest.main()
