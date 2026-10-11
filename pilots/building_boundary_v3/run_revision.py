"""Retain v2 false admissions and score only admitted v3 software examples."""
from copy import deepcopy
import json
from pathlib import Path
from baseline import v2,v1
import pilot_v3 as v3
ROOT=Path(__file__).resolve().parent

def inputs():
    t,g=v1.read('topology.json'),v1.read('goals.json')
    e,r=v2.read('geometry_evidence.json'),v2.read('scene_reference.json')
    s,src=v2.read('initial_session.json'),v2.read('initial_census.json')
    for view in r['views'].values():
        view['passage_corridor']={'shape':'axis_aligned_rectangle','roi':list(view['doorway_support_roi']),
            'passage_axis':'x','barrier_faces':['y_min','y_max'],
            'a_face':'x_min' if view['a_anchor'][0]<view['b_anchor'][0] else 'x_max',
            'b_face':'x_max' if view['a_anchor'][0]<view['b_anchor'][0] else 'x_min',
            'review_status':'AUTHOR_REVIEWED_SYNTHETIC_CORRIDOR'}
    r['revision']+=1;e['revision']+=1;e['reference_sha256']=v1.digest(r)
    return t,g,e,r,s,src

def run(gate,c,t,g,obs,e,r,s,src):
    cr=gate.fixture_config_review(c,t,g,e,r);sr=gate.fixture_session_review(s,cr)
    return gate.execute(c,t,g,obs,e,r,cr,s,src,sr)

def run_all():
    t,g,e,r,s,src=inputs()
    obs=v1.read('observations.json')['observations']
    truth=v1.ground_truth(t,g,v1.read('states_gt.json')['states'],v1.read('events_gt.json')['events'])
    strategies={}
    for name in ('manual_truth','metadata_rule','door_only','model_candidate','model_plus_metadata_or_plan'):
        c=v1.read('manual_config.json') if name=='manual_truth' else v1.propose(t,g,name)
        if c is None: strategies[name]={'status':'NOT_RUN','metrics':None};continue
        result=run(v3,c,t,g,obs,e,r,s,src);result['metrics']=v1.score(result,truth)
        strategies[name]=result
    cases={}
    for case in ('short_finite_line','missing_source_id_and_revision'):
        c=v1.read('manual_config.json');session,source=deepcopy(s),deepcopy(src)
        if case=='short_finite_line': c['targets']['building'][0]['line']=[[.5,.26],[.5,.30]]
        else:
            for key in ('source_id','revision'): source.pop(key)
            for key in ('source_id','source_revision'): session.pop(key)
            session['source_sha256']=v1.digest(source)
        before=run(v2,c,t,g,obs,e,r,session,source);before['metrics']=v1.score(before,truth)
        after=run(v3,c,t,g,obs,e,r,session,source)
        cases[case]={'v2_false_admission':before,'v3':after,'candidate_config':c,
                     'session':session,'source':source}
    positives={}
    for name,line in {'wall_endpoint_cut':[[.49,.25],[.51,.75]],'tilted_full_cut':[[.46,.2],[.54,.8]]}.items():
        c=v1.read('manual_config.json');c['targets']['building'][0]['line']=line
        result=run(v3,c,t,g,obs,e,r,s,src);result['metrics']=v1.score(result,truth)
        positives[name]={'line':line,'run':result}
    return {'admission_version':v3.VERSION,'scope':'bounded_synthetic_revision_not_admission_completeness',
        'independent_review_of_850b4f3':{'critical':0,'important':1,'minor':1,'software_tests_independent_passed':38,
            'v2_browser_checks':'10_author_evidence_only'},
        'reference_v3':r,'geometry_evidence_v3':e,'strategies':strategies,'counterexamples':cases,
        'different_geometry_positive_controls':positives,'real_visual':{'status':'NOT_RUN','metrics':None},
        'new_head_independent_review':'PENDING'}

if __name__=='__main__':
    result=run_all();(ROOT/'results_v3.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{'old':v['v2_false_admission']['status'],'v3':v['v3']['status'],
        'old_building':v['v2_false_admission']['metrics']['building']} for k,v in result['counterexamples'].items()},indent=2))
