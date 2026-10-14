"""Validate private AI-assisted partial reviews, never manufacture geometry GT."""
import argparse,datetime,json,math,sys,tempfile
from pathlib import Path
IMMUTABLE={'source_id','arm','prediction_id','receipt_sha256','source_sha256','AI_bbox_xyxy_original_unclipped','AI_score','AI_text_label'}
VISIBILITY={'visible','not_visible','occluded','unknown'}
STATUSES={'unreviewed','partial_review','needs_more_evidence'}
def require(c,m):
 if not c:raise ValueError(m)
def key(r):return (r['source_id'],r['arm'],r['prediction_id'])
def validate(template,review,now=None):
 now=now or datetime.datetime.now(datetime.timezone.utc)
 require(set(template)=={'schema_version','human_GT','AI_assisted_review_not_blind_GT','records'},'unexpected template fields')
 require(set(review)==set(template),'review envelope drift')
 for field in ['schema_version','human_GT','AI_assisted_review_not_blind_GT']:require(review[field]==template[field],'review envelope scope drift')
 require(review['schema_version']==1 and review['human_GT'] is False and review['AI_assisted_review_not_blind_GT'] is True,'not partial AI-assisted review contract')
 expected={key(x):x for x in template['records']};require(len(expected)==len(template['records']),'duplicate template candidate')
 require(len(review['records'])==len(expected) and {key(x) for x in review['records']}==set(expected),'missing/duplicate/substituted candidate')
 counts={x:0 for x in STATUSES}
 for r in review['records']:
  original=expected[key(r)];require(set(r)==set(original),'record schema drift')
  for field in IMMUTABLE:require(r[field]==original[field],'AI/source identity changed: '+field)
  status=r['review_status'];require(status in STATUSES,'unsupported review state');counts[status]+=1
  for field in ['S','B','events','trajectory','direction','threshold_location']:require(r[field]=='unknown','unadmitted physical/event claim: '+field)
  require(r['full_opening_geometry_confirmed'] is False,'complete opening GT is not admitted in this batch')
  for edge in ['top','left','right','bottom']:require(r[edge+'_visibility'] in VISIBILITY,'invalid visibility')
  if status=='unreviewed':
   require(r==original,'unreviewed record must remain unchanged');continue
  require(isinstance(r['reviewer'],str) and r['reviewer'].strip(),'missing real reviewer')
  require(isinstance(r['note'],str) and r['note'].strip(),'missing bounded review note')
  require(r['viewed_original_sha256']==r['source_sha256'],'reviewed image identity mismatch')
  require(r['seen_AI_candidates'] is True,'this template is AI-assisted; cannot be reported as blind GT')
  try:stamp=datetime.datetime.fromisoformat(r['reviewed_utc'].replace('Z','+00:00'))
  except (ValueError,TypeError,AttributeError):raise ValueError('missing/invalid actual review UTC')
  require(stamp.tzinfo is not None and stamp.utcoffset()==datetime.timedelta(0) and stamp<=now,'review UTC must be actual UTC, not future')
  require(r['physical_door_identity']=='unknown','cross-view physical door identity is not admitted')
  coordinates=r['coordinates_added_by_human']
  if coordinates is not None:
   require(isinstance(coordinates,dict) and bool(coordinates),'coordinates must be explicit visible edge fragments')
   require(set(coordinates)<= {'top','left','right','bottom'},'threshold/closed ROI coordinates not admitted')
   for edge,points in coordinates.items():
    require(r[edge+'_visibility']=='visible','coordinates added for invisible/occluded/unknown edge')
    require(isinstance(points,list) and len(points)>=2,'visible fragment needs two or more points')
    for point in points:
     require(isinstance(point,list) and len(point)==2,'fragment point must be [x,y]')
     require(all(type(v) in (int,float) and math.isfinite(v) for v in point),'nonfinite or invalid coordinate')
     require(0<=point[0]<=800 and 0<=point[1]<=600,'coordinate outside original image')
    require(points[0]!=points[-1],'closed polygons not admitted as visible fragments')
 # Aggregate is deliberately path/name/coordinate free. It does not change original annotations.
 return {'schema_version':1,'status':'partial_review_contract_valid','candidate_records':len(expected),'review_status_counts':counts,'human_GT':False,'scientific_ROI_admission':False,'decision':'unknown','accuracy_metrics':None,'limits':['review content truth not independently verified','invisible threshold remains unknown','no full ROI or entry-event admission','AI-assisted review, not blind GT']}
def check_review_readiness(review, run_name, repo=None, run=None, data=None, license_path=None, model=None, comparison=None):
 if all(x['review_status']=='unreviewed' for x in review['records']):return 'not_required_for_unreviewed_template'
 require(repo is not None and run is not None and data is not None and license_path is not None,'reviewed records require actual --run, --source-data and --license')
 from check_private_return import inspect
 # Only this live read-only inspection can establish readiness, never caller-supplied JSON.
 integrity,_=inspect(repo,run,run_name,data,license_path,model)
 require(isinstance(integrity,dict),'reviewed records require a private integrity report')
 require(integrity.get('run_name')==run_name,'integrity report matrix mismatch')
 require(integrity.get('status')=='private_bytes_and_original_validator_checks_passed' and integrity.get('missing')==[],'private bytes incomplete/failed; stop reviewed-record acceptance')
 require(integrity.get('raw_and_processed_tensor_validation')=='passed; normal padded -inf logits allowed','raw/processed tensors unverified; stop reviewed-record acceptance')
 require(integrity.get('public_receipts_verified') is True,'receipt generation binding not verified')
 require(integrity.get('human_GT') is False and integrity.get('scientific_ROI_admission') is False and integrity.get('decision')=='unknown','integrity scope escalation')
 require(integrity.get('schema_version')==1,'inspection schema mismatch')
 require(integrity.get('run_state',{}).get('status')=='completed' and integrity.get('run_state',{}).get('successful_matrix') is True,'run not completed during actual inspection')
 if comparison is not None:require(comparison==integrity,'supplied integrity report differs from actual inspection')
 return 'actual_readonly_inspection_checked_for_partial_review'
def bind_template(template, repo, run_name):
 from check_private_return import inspect
 require(run_name in {'cpu-matrix','gpu-matrix'},'unknown returned matrix')
 # Regenerate identities from pinned Git receipt bytes; supplied template cannot redefine AI.
 with tempfile.TemporaryDirectory(prefix='rule2region-review-contract-') as temp:
  _,records=inspect(repo,Path(temp)/'absent',run_name)
 expected={'schema_version':1,'human_GT':False,'AI_assisted_review_not_blind_GT':True,'records':records}
 require(template==expected,'template differs from pinned receipt-derived initial contract')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--run-name',choices=['cpu-matrix','gpu-matrix'],required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--run',type=Path);p.add_argument('--source-data',type=Path);p.add_argument('--license',type=Path);p.add_argument('--model',type=Path);p.add_argument('--integrity-report',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu-prompt-boundary-handoff-20261012'));import handoff
 handoff.outside_git(a.out)
 template=json.loads(a.template.read_text());bind_template(template,a.repo.resolve(),a.run_name)
 review=json.loads(a.review.read_text());result=validate(template,review)
 integrity=json.loads(a.integrity_report.read_text()) if a.integrity_report else None
 result['private_integrity_prerequisite']=check_review_readiness(review,a.run_name,repo=a.repo.resolve(),run=a.run,data=a.source_data,license_path=a.license,model=a.model,comparison=integrity)
 # New report only: never overwrite the template, user review, previous report or Site annotation.
 with a.out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'human_GT':False,'decision':'unknown'}))
