"""Synthetic contract tests only: never event truth or recognition accuracy."""
import copy,json,tempfile,unittest
from pathlib import Path
import handoff as h
class Contracts(unittest.TestCase):
 def fixture(self):
  return {'human_GT':False,'accuracy_metrics':None,'S_image_relation':'unknown','B_image_relation':'unknown','event_decision':'unknown','trajectory':'unknown','direction':'unknown','box_threshold':.25,'text_threshold':.25,'classwise_nms_iou':.7,'kept_count':1,'model_candidates_after_nms':[{'bbox_xyxy_original_unclipped':[10,10,400,600],'score':.7}],'forward_seconds':1.0,'peak_cuda_bytes':None}
 def test_synthetic_clipping_is_not_geometry_truth(self):
  r=self.fixture();s=h.summarize_receipt(r);self.assertEqual(s['image_edge_touch_or_overrun_count'],1);self.assertIsNone(r['accuracy_metrics'])
 def test_reject_forged_safe_decision(self):
  r=self.fixture();r['event_decision']='safe'
  with self.assertRaises(ValueError):h.summarize_receipt(r)
 def test_reject_gt_or_accuracy(self):
  for key,value in [('human_GT',True),('accuracy_metrics',{'accuracy':1})]:
   r=self.fixture();r[key]=value
   with self.assertRaises(ValueError):h.summarize_receipt(r)
 def test_reject_nonfinite_and_degenerate(self):
  for box in [[0,0,float('nan'),10],[10,0,10,20]]:
   r=self.fixture();r['model_candidates_after_nms'][0]['bbox_xyxy_original_unclipped']=box
   with self.assertRaises(ValueError):h.summarize_receipt(r)
 def test_threshold_and_count_drift(self):
  for key,value in [('box_threshold',.1),('kept_count',2)]:
   r=self.fixture();r[key]=value
   with self.assertRaises(ValueError):h.summarize_receipt(r)
 def test_private_path_reject_checkout(self):
  with self.assertRaises(ValueError):h.outside_git(h.ROOT/'output')
 def test_failed_run_cannot_export(self):
  with tempfile.TemporaryDirectory() as t:
   h.write(Path(t)/'RUN_PRIVATE.json',{'status':'failed'})
   with self.assertRaises(ValueError):h.validate(type('A',(),{'out':t})())
 def test_source_hash_guard(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'fake.tar.xz';p.write_bytes(b'synthetic-not-real-data')
   with self.assertRaises(ValueError):h.prepare(type('A',(),{'archive':p,'license':p,'data':Path(t)/'data'})())
class TensorMasks(unittest.TestCase):
 def test_synthetic_masked_negative_inf_allowed_active_rejected(self):
  import torch
  logits=torch.full((1,900,256),float('-inf'));logits[:,:,:6]=0
  raw={'logits':logits,'pred_boxes':torch.zeros(1,900,4),'attention_mask':torch.ones(1,6,dtype=torch.int64),'input_ids':torch.ones(1,6,dtype=torch.int64)}
  inp={'pixel_values':torch.zeros(1,3,2,2),'input_ids':raw['input_ids'],'attention_mask':raw['attention_mask']}
  h.check_tensors(raw,inp)
  for v in [float('-inf'),float('inf'),float('nan')]:
   raw['logits'][0,0,0]=v
   with self.assertRaises(ValueError):h.check_tensors(raw,inp)
 def test_synthetic_padding_must_be_negative_inf(self):
  import torch
  raw={'logits':torch.zeros(1,900,256),'pred_boxes':torch.zeros(1,900,4),'attention_mask':torch.ones(1,6,dtype=torch.int64),'input_ids':torch.ones(1,6,dtype=torch.int64)}
  inp={'pixel_values':torch.zeros(1,3,2,2),'input_ids':raw['input_ids'],'attention_mask':raw['attention_mask']}
  with self.assertRaises(ValueError):h.check_tensors(raw,inp)
if __name__=='__main__':unittest.main()
