"""Layered admission. Scene evidence and session receipts never read scoring GT."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from reproduce_v1 import v1, reproduce
ROOT=Path(__file__).resolve().parent
VERSION='building-boundary-admission-v2'
read=lambda name: json.loads((ROOT/name).read_text())
digest=v1.digest


def side(line,point):
    p,q=line
    value=(q[0]-p[0])*(point[1]-p[1])-(q[1]-p[1])*(point[0]-p[0])
    return 1 if value>1e-9 else -1 if value < -1e-9 else 0


def intersects_interior(line,roi):
    """Clip finite segment against rectangle; positive length inside open ROI."""
    p,q=line
    low,high=0.,1.
    for axis in (0,1):
        delta=q[axis]-p[axis]
        lower,upper=roi[axis],roi[axis+2]
        if abs(delta)<1e-12:
            if not lower < p[axis] < upper:
                return False
        else:
            a,b=(lower-p[axis])/delta,(upper-p[axis])/delta
            low,high=max(low,min(a,b)),min(high,max(a,b))
    return high-low>1e-9


def goals_contract(goals):
    return {k:v for k,v in goals.items() if k!='initial'}


def config_binding(config,topology,goals,evidence,reference):
    return {'admission_version':VERSION,'config_sha256':digest(config),
            'topology_sha256':digest(topology),'goals_contract_sha256':digest(goals_contract(goals)),
            'geometry_evidence_sha256':digest(evidence),'reference_sha256':digest(reference)}


def fixture_config_review(config,topology,goals,evidence,reference):
    """Simulated review artifact, NOT a validator or real reviewer authentication."""
    return {'receipt_version':'config-review-v2','scope':'synthetic_fixture_only',
            'reviewer':'fixture_author_simulation',
            'binding':config_binding(config,topology,goals,evidence,reference)}


def fixture_session_review(session,config_review):
    return {'receipt_version':'session-review-v2','scope':'synthetic_fixture_only',
            'reviewer':'fixture_author_simulation','admission_version':VERSION,
            'session_sha256':digest(session),'config_review_sha256':digest(config_review)}


def validate_config(config,topology,goals,evidence,reference,review):
    reasons=[]
    if config.get('schema_version')!='building-boundary-v1':
        reasons.append('CONFIG_SCHEMA')
    if not topology.get('complete_channels_asserted_by'):
        reasons.append('UNKNOWN_CHANNEL_COVERAGE')
    if set(config.get('targets',{}))!=set(goals['goals']):
        reasons.append('TARGET_SET')
    views=reference.get('views',{}) if isinstance(reference,dict) else {}
    evidence_ok=(isinstance(evidence,dict) and isinstance(reference,dict) and evidence.get('schema_version')=='scene-evidence-v2'
        and evidence.get('scope')=='synthetic_fixture_only'
        and evidence.get('review_status')=='AUTHOR_REVIEWED_SYNTHETIC_REFERENCE'
        and evidence.get('reference_sha256')==digest(reference)
        and evidence.get('reference_kind')=='self_authored_coordinate_reference_not_image'
        and reference.get('kind')=='self_authored_coordinate_reference_not_image'
        and type(evidence.get('revision')) is int and evidence['revision']>0
        and type(reference.get('revision')) is int and reference['revision']>0
        and set(evidence.get('view_ids',[]))==set(views))
    if not evidence_ok:
        reasons.append('UNKNOWN_OR_STALE_SCENE_EVIDENCE')
    camera_ids=[]
    for door in topology['doors'].values():
        if door['a'] not in topology['regions'] or door['b'] not in topology['regions'] or door['a']==door['b']:
            reasons.append('INVALID_TOPOLOGY_ENDPOINT')
        camera_ids.extend(door['cameras'])
    if len(set(camera_ids))!=len(camera_ids):
        reasons.append('CAMERA_DOOR_AMBIGUOUS')
    for name,goal in goals['goals'].items():
        if not set(goal['regions'])<=set(topology['regions']):
            reasons.append(name+':UNKNOWN_REGION')
        expected=v1.cut(goal,topology)
        entries=config.get('targets',{}).get(name,[])
        actual=Counter(e.get('door') for e in entries)
        if set(actual)!=set(expected):
            reasons.append(name+':BOUNDARY_COVERAGE')
        if any(v!=1 for v in actual.values()):
            reasons.append(name+':MULTIPLE_RESPONSIBILITY')
        for e in entries:
            prefix=name+'/'+str(e.get('door'))+':'
            d=topology['doors'].get(e.get('door'))
            camera=e.get('owner_camera')
            if not d or camera not in d['cameras']:
                reasons.append(prefix+'INVALID_OWNER');continue
            if type(e.get('delta_ab')) is not int or e['delta_ab']!=expected.get(e['door']):
                reasons.append(prefix+'DIRECTION_OR_MEMBERSHIP')
            numeric=v1.valid_geometry(e) and type(e.get('a_side')) is int
            if not numeric:
                reasons.append(prefix+'NUMERIC_GEOMETRY_INVALID');continue
            if not intersects_interior(e['line'],e['roi']):
                reasons.append(prefix+'LINE_ROI_INCONSISTENT')
            if not evidence_ok:
                continue
            ev=views.get(camera)
            if not isinstance(ev,dict) or (ev.get('door'),ev.get('a_region'),ev.get('b_region'))!=(e['door'],d['a'],d['b']):
                reasons.append(prefix+'UNKNOWN_REGION_SIDE_EVIDENCE');continue
            if not v1.valid_geometry({'line':[ev.get('a_anchor'),ev.get('b_anchor')],
                'roi':ev.get('doorway_support_roi'),'a_side':1}):
                reasons.append(prefix+'MALFORMED_SCENE_GEOMETRY_EVIDENCE');continue
            # Independent scene coordinates, not original candidate geometry or scoring GT.
            if side(e['line'],ev['a_anchor'])!=e['a_side'] or side(e['line'],ev['b_anchor'])!=-e['a_side']:
                reasons.append(prefix+'DIRECTED_LINE_REGION_SIDE_CONFLICT')
            support=ev['doorway_support_roi']
            roi=e['roi']
            if not (roi[0]<=support[0] and roi[1]<=support[1] and roi[2]>=support[2] and roi[3]>=support[3]):
                reasons.append(prefix+'ROI_MISSES_REVIEWED_PORTAL_SUPPORT')
            if not intersects_interior(e['line'],support):
                reasons.append(prefix+'LINE_MISSES_REVIEWED_PORTAL_SUPPORT')
    if (not isinstance(review,dict) or review.get('receipt_version')!='config-review-v2'
        or review.get('scope')!='synthetic_fixture_only'
        or review.get('binding')!=config_binding(config,topology,goals,evidence,reference)):
        reasons.append('MISSING_OR_STALE_CONFIG_REVIEW')
    return sorted(set(reasons))


def validate_session(session,source,config_review,review):
    if not isinstance(session,dict):
        return ['UNKNOWN_INITIAL_SESSION']
    reasons=[]
    if (session.get('schema_version')!='initial-session-v2' or not session.get('session_id')
        or type(session.get('revision')) is not int or session['revision']<1):
        reasons.append('SESSION_SCHEMA_OR_REVISION')
    if type(session.get('t0')) is not int or session['t0']!=0 or session.get('time_basis')!='synthetic_step':
        reasons.append('UNSUPPORTED_OR_UNKNOWN_T0')
    initial=session.get('initial',{})
    if set(initial)!={'building','floor1','floor2'} or any(type(v) is not int or v<0 for v in initial.values()):
        reasons.append('UNKNOWN_INITIAL_OCCUPANCY')
    elif initial['building']!=initial['floor1']+initial['floor2']:
        reasons.append('INITIAL_CONSERVATION')
    if (not isinstance(source,dict) or source.get('schema_version')!='initial-census-v2'
        or source.get('scope')!='synthetic_fixture_only'
        or source.get('status')!='AUTHOR_RECORDED_NOT_REAL_FIELD_CENSUS'
        or session.get('source_sha256')!=digest(source)
        or session.get('source_id')!=source.get('source_id')
        or session.get('source_revision')!=source.get('revision')
        or session.get('t0')!=source.get('t0') or session.get('time_basis')!=source.get('time_basis')
        or initial!=source.get('initial')):
        reasons.append('UNKNOWN_OR_INCONSISTENT_INITIAL_SOURCE')
    if (not isinstance(review,dict) or review.get('receipt_version')!='session-review-v2'
        or review.get('scope')!='synthetic_fixture_only' or review.get('admission_version')!=VERSION
        or review.get('session_sha256')!=digest(session)
        or review.get('config_review_sha256')!=digest(config_review)):
        reasons.append('MISSING_OR_STALE_SESSION_QUALIFICATION')
    return sorted(set(reasons))


def execute(config,topology,goals,observations,evidence,reference,config_review,session,source,session_review):
    config_reasons=validate_config(config,topology,goals,evidence,reference,config_review)
    session_reasons=validate_session(session,source,config_review,session_review)
    if 'initial' in goals and (not isinstance(session,dict) or goals['initial']!=session.get('initial')):
        session_reasons.append('LEGACY_INITIAL_SESSION_CONFLICT')
    outcome={'admission_version':VERSION,'config_status':'QUALIFIED' if not config_reasons else 'REJECTED_OR_UNKNOWN',
        'session_status':'QUALIFIED' if not session_reasons else 'REJECTED_OR_UNKNOWN',
        'config_reasons':config_reasons,'session_reasons':session_reasons,'metrics':None}
    if config_reasons or session_reasons:
        outcome['status']='REJECTED_OR_UNKNOWN';return outcome
    runtime=deepcopy(goals);runtime['initial']=session['initial']
    # v1's numerical computation is retained exactly. Its old gate is explicitly bypassed;
    # only its series/events are consumed, after both v2 gates have passed.
    counted=v1.execute(config,topology,runtime,observations,v1.fixture_review(config),diagnostic=True)
    outcome.update(status='SYNTHETIC_V2_ADMITTED',series=counted['series'],events=counted['events'],
        config_sha256=digest(config),config_review_sha256=digest(config_review),
        session_sha256=digest(session),session_review_sha256=digest(session_review),
        initial_source_sha256=digest(source),t0=session['t0'])
    return outcome


def run_all():
    topology,goals=v1.read('topology.json'),v1.read('goals.json')
    obs=v1.read('observations.json')['observations']
    evidence,reference=read('geometry_evidence.json'),read('scene_reference.json')
    session,source=read('initial_session.json'),read('initial_census.json')
    # Scoring GT is only loaded by this orchestration function, never by admission.
    truth=v1.ground_truth(topology,goals,v1.read('states_gt.json')['states'],v1.read('events_gt.json')['events'])
    strategies={}
    for name in ('manual_truth','metadata_rule','door_only','model_candidate','model_plus_metadata_or_plan'):
        config=v1.read('manual_config.json') if name=='manual_truth' else v1.propose(topology,goals,name)
        if config is None:
            strategies[name]={'status':'NOT_RUN','metrics':None};continue
        cr=fixture_config_review(config,topology,goals,evidence,reference)
        sr=fixture_session_review(session,cr)
        run=execute(config,topology,goals,obs,evidence,reference,cr,session,source,sr)
        run['metrics']=v1.score(run,truth)
        strategies[name]=run
    base=v1.read('manual_config.json')
    base_cr=fixture_config_review(base,topology,goals,evidence,reference)
    base_sr=fixture_session_review(session,base_cr)
    failures={}
    for case in ('reverse_line_only','displaced_valid_roi','changed_initial_old_approval',
                 'missing_scene_evidence','modified_reference','line_roi_consistent_wrong_portal',
                 'roi_intersects_but_incomplete_portal','changed_t0','changed_source','changed_session_revision',
                 'stale_config_after_line_change','missing_session_review'):
        c,g,e,r,s,src=map(deepcopy,(base,goals,evidence,reference,session,source))
        entry=c['targets']['building'][0]
        cr,sr=base_cr,base_sr
        if case=='reverse_line_only': entry['line'].reverse()
        elif case=='displaced_valid_roi': entry['roi']=[.01,.01,.1,.1]
        elif case=='changed_initial_old_approval':
            s['initial']={'building':30,'floor1':20,'floor2':10};g['initial']=dict(s['initial'])
        elif case=='missing_scene_evidence': e=None
        elif case=='modified_reference': r['revision']+=1
        elif case=='line_roi_consistent_wrong_portal':
            entry['line']=[[.1,.2],[.1,.8]];entry['roi']=[.01,.1,.2,.9]
        elif case=='roi_intersects_but_incomplete_portal': entry['roi']=[.49,.4,.51,.6]
        elif case=='changed_t0': s['t0']=1
        elif case=='changed_source': s['source_sha256']='changed_source'
        elif case=='changed_session_revision': s['revision']+=1
        elif case=='stale_config_after_line_change':
            entry['line'].reverse();entry['a_side']*=-1
        elif case=='missing_session_review': sr=None
        if case in ('reverse_line_only','displaced_valid_roi','line_roi_consistent_wrong_portal','roi_intersects_but_incomplete_portal'):
            cr=fixture_config_review(c,topology,g,e,r);sr=fixture_session_review(s,cr)
        failures[case]=execute(c,topology,g,obs,e,r,cr,s,src,sr)
    positives={}
    for case in ('reverse_line_and_side','alternative_supported_line'):
        c=deepcopy(base);entry=c['targets']['building'][0]
        if case=='reverse_line_and_side': entry['line'].reverse();entry['a_side']*=-1
        else: entry['line']=[[.48,.15],[.48,.85]];entry['roi']=[.15,.1,.85,.9]
        cr=fixture_config_review(c,topology,goals,evidence,reference);sr=fixture_session_review(session,cr)
        run=execute(c,topology,goals,obs,evidence,reference,cr,session,source,sr)
        run['metrics']=v1.score(run,truth);positives[case]=run
    return {'admission_version':VERSION,'scope':'synthetic_software_only_no_model',
        'baseline_commit':'40f0240b6929c13c6ce9bad62b25a5111ec49e85',
        'fixed_v1_executor_sha256':hashlib.sha256((v1.ROOT/'pilot.py').read_bytes()).hexdigest(),
        'example_fixture_receipts':{'config_review':base_cr,'session_qualification':base_sr},
        'strategies':strategies,'counterexamples_v1':{k:{'status':v['status'],'metrics':v['metrics']} for k,v in reproduce().items()},
        'failures_v2':failures,'non_oracle_positive_controls':positives,
        'real_visual':{'status':'NOT_RUN','metrics':None}}

if __name__=='__main__':
    results=run_all()
    (ROOT/'results_v2.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'strategies':{k:v['status'] for k,v in results['strategies'].items()},
        'failures':{k:[v['config_status'],v['session_status']] for k,v in results['failures_v2'].items()},
        'positive_controls':{k:v['status'] for k,v in results['non_oracle_positive_controls'].items()}},indent=2))
