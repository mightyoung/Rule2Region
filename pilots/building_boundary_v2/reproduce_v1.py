"""Retain genuine v1 false admissions as evidence; never overwrite the v1 report."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'building_boundary'
spec=importlib.util.spec_from_file_location('boundary_v1',OLD/'pilot.py')
v1=importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

def reproduce():
    t,g=v1.read('topology.json'),v1.read('goals.json')
    cfg=v1.read('manual_config.json'); obs=v1.read('observations.json')['observations']
    truth=v1.ground_truth(t,g,v1.read('states_gt.json')['states'],v1.read('events_gt.json')['events'])
    result={}
    old_review=v1.fixture_review(cfg)
    for case in ('reverse_line_only','displaced_valid_roi','changed_initial_old_approval'):
        c,goals=deepcopy(cfg),deepcopy(g)
        if case=='reverse_line_only':
            c['targets']['building'][0]['line'].reverse()
        elif case=='displaced_valid_roi':
            c['targets']['building'][0]['roi']=[.01,.01,.1,.1]
        else:
            goals['initial']={'building':30,'floor1':20,'floor2':10}
        review=old_review if case=='changed_initial_old_approval' else v1.fixture_review(c)
        run=v1.execute(c,t,goals,obs,review)
        run['metrics']=v1.score(run,truth)
        run['candidate_config']=c
        run['input_initial']=goals['initial']
        result[case]=run
    return result

if __name__=='__main__':
    result=reproduce()
    (ROOT/'v1_counterexamples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{'v1_status':v['status'],'building':v['metrics']['building']} for k,v in result.items()},indent=2))
