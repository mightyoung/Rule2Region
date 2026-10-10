"""Feature-view projection only. No geometry, motion, side or event inference."""
from copy import deepcopy
import math
ARMS=('G','G_DIR','G_DIST','G_DIR_DIST')

def direction_runs(segments,anchor):
 run=best=0
 for d in segments:
  a=d['anchors'].get(anchor,{})
  if a.get('bearing_degrees') is not None and a.get('norm_px',0)>0:
   run+=1;best=max(best,run)
  else:run=0
 return best

def usable_distance(d):
 if d['geometry_status']!='valid' or not d['anchors']:return False
 for a in d['anchors'].values():
  if not a.get('portals'):return False
  for p in a['portals']:
   if any(type(p.get(k)) not in (int,float) or not math.isfinite(p[k]) for k in ('set_distance_px','boundary_distance_px','vertex_mean_distance_px')):return False
 return True

def arm_view(payload,arm):
 if arm not in ARMS:raise ValueError('unknown ablation arm')
 target=payload['current']['id']
 current=next((d for d in payload['geometry_descriptors'] if d['endpoint_bindings'][0]['observation_id']==target),None)
 direction=payload['motion_descriptors'] if arm in ('G_DIR','G_DIR_DIST') else []
 runs={a:direction_runs(direction,a) for a in ('center','bottom_proxy')}
 sequence=[]
 if arm in ('G_DIST','G_DIR_DIST') and current is not None:
  ds=payload['geometry_descriptors']
  # No jumping over a missing/unknown geometry record and no reference mixing.
  refs=lambda d:{(d['geometry_source_sha256'],a,p['reference_id']) for a,v in d['anchors'].items() for p in v.get('portals',[])}
  if len(ds)>=2 and len(ds)==len(payload['accepted_history'])+1 and all(usable_distance(d) for d in ds) and all(refs(d)==refs(ds[0]) for d in ds):sequence=ds
 missing=list(payload['missing_inputs'])
 if arm in ('G_DIR','G_DIR_DIST'):
  if max(runs.values())<1:missing.append('qualified_nonzero_direction_segment')
  if max(runs.values())<2:missing.append('two_consecutive_nonzero_direction_segments_same_anchor')
 if arm in ('G_DIST','G_DIR_DIST'):
  if len(sequence)<2:missing.append('associated_historical_distance_sequence')
  if len(sequence)<3:missing.append('three_associated_distance_observations')
 return deepcopy({'arm':arm,'target_id':target,'query_time':payload['query_time'],'current_geometry':current,
  'direction_segments':direction,'historical_distance_sequence':sequence,
  'direction_segment_available':max(runs.values())>=1,'direction_continuity_available':max(runs.values())>=2,
  'consecutive_usable_direction_segments_by_anchor':runs,'three_point_distance_available':len(sequence)>=3,
  'side_evidence':'unknown','side_provenance':'unknown','event_status':'unknown','identity_verified':False,
  'same_source_correlation':'current distance, historical distance and motion derived from shared bbox source groups; no independent votes',
  'missing_inputs':sorted(set(missing)), 'inference_missing_inputs':sorted(set(missing)), 'evaluation_gaps':payload['evaluation_gaps'], 'model_calls':0,'no_device_action':True})
