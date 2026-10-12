"""Non-neural pixel-supported open jamb fragments; no closed counting ROI."""
import argparse,datetime,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 a.output_dir.mkdir(parents=True,exist_ok=False)
 log={'operation':'non_neural_visible_fragment_refinement','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'actual_argv':sys.argv,'script_sha256':sha(Path(__file__)),'new_model_calls':0,'input_files':[]};start=time.monotonic();exit_code=1
 try:
  door=json.loads(a.door_receipt.read_text());person=json.loads(a.person_receipt.read_text());frames=[]
  for frame,expected in [(0,door['source_sha256']),(142,person['source_sha256'])]:
   p=a.frame_dir/f'{frame:08d}.jpg';assert sha(p)==expected
   log['input_files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':expected})
   with Image.open(p) as im:
    im=im.convert('RGB');assert im.size==(800,600);rgb=np.asarray(im,dtype=np.float32)
   gradient=np.linalg.norm(rgb[:,2:]-rgb[:,:-2],axis=2);sides=[]
   for name,lo,hi in [('left',30,140),('right',600,730)]:
    points=[]
    for y in [50,100,200,300,400,500]:
     scan=gradient[y-2:y+3,lo:hi].mean(axis=0);index=int(np.argmax(scan));points.append({'x':index+lo+1,'y':y,'rgb_gradient_norm':float(scan[index])})
    sides.append({'side':name,'search_window_x':[lo,hi],'open_polyline_points':points,'human_GT':False,'method':'AI-selected windows, fixed horizontal RGB [-1,0,1] finite difference, five-row mean, argmax; not learned segmentation'})
   frames.append({'frame_index':frame,'source_sha256':expected,'width':800,'height':600,'visible_fragment_candidates':sides,'closed_counting_roi':None,'top_lintel':None,'bottom_sill_or_crossing_line':None,'missing_edge_reason':'Not completely localized from these images. Crop/unmarked floor transition possible; no artificial closure at model bbox edge.','S_membership':'unknown','B_membership':'unknown','annotation_status':'AI_geometric_proposal_pending_human_review'})
  matches=sum(x['x']==y['x'] and x['y']==y['y'] for one,two in zip(frames[0]['visible_fragment_candidates'],frames[1]['visible_fragment_candidates']) for x,y in zip(one['open_polyline_points'],two['open_polyline_points']))
  result={'version':'visible-fragments-v1','frames':frames,'matching_selected_peak_positions':matches,'selected_peak_positions_per_frame':12,'interpretation':'Agreement of selected visible edge peaks supports local structural stability only, not whole-image registration, camera calibration, physical gate identity or counting membership','old_model_bbox_modified':False,'closed_roi_emitted':False,'new_model_calls':0,'human_GT':False}
  (a.output_dir/'VISIBLE_FRAGMENT_PROPOSALS_PRIVATE.json').write_text(json.dumps(result,indent=2)+'\n')
  os.environ['MPLCONFIGDIR']=str(a.output_dir/'mpl-cache')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle
  fig,axes=plt.subplots(1,2,figsize=(16,7),dpi=120)
  for ax,fr in zip(axes,frames):
   with Image.open(a.frame_dir/f"{fr['frame_index']:08d}.jpg") as im:ax.imshow(im)
   for row in door['model_candidates_after_nms']:
    x0,y0,x1,y1=row['bbox_xyxy_original_unclipped'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#ffaa22',linestyle='--',linewidth=1))
   for side in fr['visible_fragment_candidates']:
    points=side['open_polyline_points'];ax.plot([p['x'] for p in points],[p['y'] for p in points],'-o',color='#22e2ee',linewidth=2,markersize=4)
   ax.set_xlim(0,800);ax.set_ylim(600,0);ax.axis('off');ax.set_title(f"Frame {fr['frame_index']}: pixel-supported open jamb fragments\nAmber = frame0 model reference; cyan = new non-neural proposal",fontsize=10)
  fig.suptitle('AI proposals only. Top/bottom not localized: no closed counting polygon or crossing line.',fontsize=12);fig.tight_layout();fig.savefig(a.output_dir/'VISIBLE_FRAGMENTS_DIAGNOSTIC.png');plt.close(fig)
  for row in log['input_files']:assert sha(Path(row['path']))==row['sha256']
  log['status']='completed';exit_code=0;print(json.dumps({'status':'completed','matching_positions':matches,'closed_roi_emitted':False,'new_model_calls':0}))
 finally:
  log['exit_code']=exit_code;log['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();log['elapsed_seconds']=time.monotonic()-start
  log['output_files']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in a.output_dir.iterdir() if p.is_file() and p.name!='ACTUAL_REFINEMENT_RUN_PRIVATE.json']
  (a.output_dir/'ACTUAL_REFINEMENT_RUN_PRIVATE.json').write_text(json.dumps(log,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['frame-dir','door-receipt','person-receipt','output-dir']:p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
