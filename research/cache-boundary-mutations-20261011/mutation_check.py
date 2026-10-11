"""Compile isolated in-memory mutants; never edits frozen or development files."""
from pathlib import Path
import io
import json
import sys
import unittest
import hashlib
from test_boundaries import Boundaries
from development_cache import DevelopmentCache,FROZEN

HERE=Path(__file__).resolve().parent
ADAPTER=HERE.parent/'cache-diagnostics-dev-20261011/development_cache.py'


def checks():
    adapter=ADAPTER.read_text();kernel=(FROZEN/'continuous_cache.py').read_text()
    mutants=[
      ('unknown_as_accepted','adapter',"r['b']!=target or r['status']!='accepted_hypothesis'","r['b']!=target",'unknown cannot become accepted'),
      ('endpoint_cleanup_omitted','adapter','self.obs[oid],super().remove_obs','self.obs[oid],lambda item,reason: self.obs.pop(item)','deleted endpoints cannot leave dependent records'),
      ('sampling_adjacency_ignored','kernel',"if self.obs[b]['_frame_no']!=self.obs[a]['_frame_no']+1:","if False:",'dense samples cannot silently become continuous links'),
      ('read_renews_source_time','adapter','if payload is None:return None','if payload is None:return None\n        for observation in self.obs.values():\n            if observation["id"]!=request["target_id"]:observation["_t"]=clock(now)','query cannot extend source TTL'),
      ('query_maintenance_omitted','kernel',"self.advance(n,k);self.stats['retrievals']+=1","self.stats['retrievals']+=1",'valid query must actually delete expired internal evidence'),
      ('late_hypothesis_leaks','adapter',"or clock(r['available_at'])>q",'', 'query cutoff controls evidence availability'),
      ('candidate_diagnostic_accepted','adapter',"'unscored_candidate' if association=='unknown' else association","'accepted_hypothesis' if association=='unknown' else association",'deletion diagnostic must preserve association class'),
      ('capacity_reported_as_TTL','adapter',"'capacity' if reason.startswith('evicted_')","'source_ttl' if reason.startswith('evicted_')",'capacity and source TTL causes stay separate'),
      ('endpoint_export_alias','adapter',"'endpoint_bindings':deepcopy(r['endpoint_bindings'])","'endpoint_bindings':r['endpoint_bindings']",'exported evidence must not rebind internal sources')]
    def run(cls):
        # Each suite gets its own class, cache instances and finite synthetic data.
        case=type('MutationBoundaries',(Boundaries,),{'cache_class':cls})
        result=unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromTestCase(case))
        return result
    baseline=run(DevelopmentCache);assert baseline.wasSuccessful()
    rows=[]
    for name,layer,old,new,protects in mutants:
        source=adapter if layer=='adapter' else kernel
        assert source.count(old)==1,(name,source.count(old))
        changed=source.replace(old,new,1)
        env={'__file__':str(ADAPTER),'__name__':'_isolated_adapter_mutant'}
        if layer=='kernel':
            ke={'__name__':'_isolated_frozen_mutant'}
            exec(compile(changed,'<kernel mutant '+name+'>','exec'),ke)
            env['mutated_frozen']=ke['ContinuousCache']
            changed=adapter.replace('from continuous_cache import ContinuousCache','ContinuousCache=mutated_frozen',1)
        exec(compile(changed,'<adapter mutant '+name+'>','exec'),env)
        result=run(env['DevelopmentCache'])
        # Errors alone are not a kill: require a real failed boundary assertion.
        rows.append({'mutation':name,'layer':layer,'protects':protects,'executed_methods':result.testsRun,
                     'assertion_failure_tests':[t.id().split('.')[-1] for t,_ in result.failures],
                     'unexpected_errors':[t.id().split('.')[-1] for t,_ in result.errors],
                     'detected_by_assertion':bool(result.failures)})
    return {'scope':'finite hand-authored in-memory software mutants, not exhaustive mutation coverage or recognition validation',
            'source_commit':'2baf16615b61bd7c04193b7c8b06b81b056aa06d',
            'baseline_boundary_methods':baseline.testsRun,'baseline_pass':baseline.wasSuccessful(),
            'mutants':rows,'all_detected_by_assertion':all(r['detected_by_assertion'] for r in rows),
            'source_files_unchanged':ADAPTER.read_text()==adapter and (FROZEN/'continuous_cache.py').read_text()==kernel,
            'source_sha256':{'adapter':hashlib.sha256(adapter.encode()).hexdigest(),'frozen_kernel':hashlib.sha256(kernel.encode()).hexdigest()},
            'new_model_calls':0,'accuracy':None,'human_GT_available':False}

if __name__=='__main__':
    result=checks();p=HERE/'mutation_results.json'
    if sys.argv[1:]==['--generate']:p.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:raise SystemExit('Only optional --generate supported')
    elif result!=json.loads(p.read_text()):raise SystemExit('Mutation receipt differs')
    if not result['all_detected_by_assertion']:raise SystemExit('Undetected hand-authored mutation')
    if any(r['unexpected_errors'] for r in result['mutants']):raise SystemExit('Mutant produced errors instead of clean assertion failures')
    print(json.dumps({'baseline_pass':result['baseline_pass'],'mutants':len(result['mutants']),
                      'all_detected_by_assertion':result['all_detected_by_assertion'],
                      'unexpected_errors':sum(len(x['unexpected_errors']) for x in result['mutants'])},indent=2))
