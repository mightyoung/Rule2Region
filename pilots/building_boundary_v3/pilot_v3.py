"""Bounded finite-corridor and mandatory-source fixes; preserved counter reused."""
from copy import deepcopy
import json
from pathlib import Path
from baseline import v2,v1
ROOT=Path(__file__).resolve().parent
VERSION='building-boundary-admission-v3'
digest=v1.digest

def config_binding(config,topology,goals,evidence,reference):
    b=v2.config_binding(config,topology,goals,evidence,reference)
    b['admission_version']=VERSION
    return b

def fixture_config_review(config,topology,goals,evidence,reference):
    return {'receipt_version':'config-review-v3','scope':'synthetic_fixture_only',
        'reviewer':'fixture_author_simulation','binding':config_binding(config,topology,goals,evidence,reference)}

def fixture_session_review(session,config_review):
    return {'receipt_version':'session-review-v3','scope':'synthetic_fixture_only',
        'reviewer':'fixture_author_simulation','admission_version':VERSION,
        'session_sha256':digest(session),'config_review_sha256':digest(config_review)}

def finite_corridor_cut(entry,view):
    """Test finite wall-to-wall separator, using reviewed spatial evidence only."""
    c=view.get('passage_corridor') if isinstance(view,dict) else None
    if not isinstance(c,dict) or c.get('shape')!='axis_aligned_rectangle' or c.get('review_status')!='AUTHOR_REVIEWED_SYNTHETIC_CORRIDOR':
        return 'UNKNOWN_CORRIDOR_PARTITION_EVIDENCE'
    axis=c.get('passage_axis')
    if axis not in ('x','y'):
        return 'UNSUPPORTED_CORRIDOR_GEOMETRY'
    traversal=0 if axis=='x' else 1; wall=1-traversal
    axis_name='x' if traversal==0 else 'y';wall_name='x' if wall==0 else 'y'
    if c.get('barrier_faces')!=[wall_name+'_min',wall_name+'_max'] or {c.get('a_face'),c.get('b_face')}!={axis_name+'_min',axis_name+'_max'}:
        return 'UNKNOWN_CORRIDOR_PARTITION_EVIDENCE'
    rect=c.get('roi')
    if not v1.valid_geometry({'line':entry.get('line'),'roi':rect,'a_side':1}):
        return 'MALFORMED_CORRIDOR_GEOMETRY'
    if not v1.valid_geometry({'line':[view.get('a_anchor'),view.get('b_anchor')],
        'roi':rect,'a_side':1}):
        return 'MALFORMED_CORRIDOR_GEOMETRY'
    # Confirm region-face labels against reference anchors, not observed person tracks.
    for region in ('a','b'):
        anchor=view.get(region+'_anchor')
        if not isinstance(anchor,list) or len(anchor)!=2:
            return 'MALFORMED_CORRIDOR_GEOMETRY'
        face=c[region+'_face']
        at_face=(anchor[traversal]<rect[traversal] if face.endswith('_min') else anchor[traversal]>rect[traversal+2])
        if not at_face or not rect[wall]<anchor[wall]<rect[wall+2]:
            return 'CORRIDOR_REGION_FACE_CONFLICT'
    roi=entry['roi']
    if not (roi[0]<=rect[0] and roi[1]<=rect[1] and roi[2]>=rect[2] and roi[3]>=rect[3]):
        return 'ROI_MISSES_REVIEWED_CORRIDOR'
    p,q=entry['line'];delta=q[wall]-p[wall]
    if abs(delta)<1e-12:
        return 'FINITE_LINE_DOES_NOT_SEPARATE_CORRIDOR'
    for bound in (rect[wall],rect[wall+2]):
        t=(bound-p[wall])/delta
        crossing=p[traversal]+t*(q[traversal]-p[traversal])
        if not (0 <= t <= 1 and rect[traversal] < crossing < rect[traversal+2]):
            return 'FINITE_LINE_DOES_NOT_SEPARATE_CORRIDOR'
    return None

def validate_config(config,topology,goals,evidence,reference,review):
    # Legacy receipt check is translated; v3 receipt integrity is checked separately.
    legacy=deepcopy(review) if isinstance(review,dict) else None
    if isinstance(legacy,dict):
        legacy['receipt_version']='config-review-v2'
        if isinstance(legacy.get('binding'),dict): legacy['binding']['admission_version']=v2.VERSION
    reasons=v2.validate_config(config,topology,goals,evidence,reference,legacy)
    if (not isinstance(review,dict) or review.get('receipt_version')!='config-review-v3'
        or review.get('scope')!='synthetic_fixture_only'
        or review.get('binding')!=config_binding(config,topology,goals,evidence,reference)):
        reasons.append('MISSING_OR_STALE_V3_CONFIG_REVIEW')
    views=reference.get('views',{}) if isinstance(reference,dict) else {}
    for name,entries in config.get('targets',{}).items():
        for entry in entries:
            if not v1.valid_geometry(entry): continue
            error=finite_corridor_cut(entry,views.get(entry.get('owner_camera')))
            if error: reasons.append(name+'/'+str(entry.get('door'))+':'+error)
    return sorted(set(reasons))

def validate_session(session,source,config_review,review):
    # Reuse legacy input/source consistency checks; this generated adapter is not approval.
    legacy_review=v2.fixture_session_review(session,config_review)
    reasons=v2.validate_session(session,source,config_review,legacy_review)
    for value,label in ((source,'SOURCE'),(session,'SESSION_SOURCE')):
        id_key='source_id';version_key='revision' if label=='SOURCE' else 'source_revision'
        if (not isinstance(value,dict) or not isinstance(value.get(id_key),str)
            or not value[id_key].strip() or type(value.get(version_key)) is not int or value[version_key]<1):
            reasons.append(label+'_REQUIRED_ID_OR_REVISION')
    if (not isinstance(review,dict) or review.get('receipt_version')!='session-review-v3'
        or review.get('scope')!='synthetic_fixture_only' or review.get('admission_version')!=VERSION
        or review.get('session_sha256')!=digest(session) or review.get('config_review_sha256')!=digest(config_review)):
        reasons.append('MISSING_OR_STALE_V3_SESSION_QUALIFICATION')
    return sorted(set(reasons))

def execute(config,topology,goals,observations,evidence,reference,config_review,session,source,session_review):
    cr=validate_config(config,topology,goals,evidence,reference,config_review)
    sr=validate_session(session,source,config_review,session_review)
    if 'initial' in goals and (not isinstance(session,dict) or goals['initial']!=session.get('initial')):
        sr.append('LEGACY_INITIAL_SESSION_CONFLICT')
    result={'admission_version':VERSION,'config_status':'QUALIFIED' if not cr else 'REJECTED_OR_UNKNOWN',
        'session_status':'QUALIFIED' if not sr else 'REJECTED_OR_UNKNOWN',
        'config_reasons':cr,'session_reasons':sr,'metrics':None}
    if cr or sr:
        result['status']='REJECTED_OR_UNKNOWN';return result
    runtime=deepcopy(goals);runtime['initial']=session['initial']
    counted=v1.execute(config,topology,runtime,observations,v1.fixture_review(config),diagnostic=True)
    result.update(status='SYNTHETIC_V3_ADMITTED',series=counted['series'],events=counted['events'])
    return result

if __name__=='__main__':
    from run_revision import run_all
    result=run_all()
    (ROOT/'results_v3.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['status'] for k,v in result['strategies'].items()},indent=2))
