"""Diagnostic replay of existing public cache; no model, new labels or pixels."""
from pathlib import Path
from collections import Counter
import json
import sys
from fractions import Fraction
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'cache-diagnostics-dev-20261011'))
from development_cache import DevelopmentCache
sys.path.insert(0,str(HERE.parent/'nine-frame-conditional-replay-20261011'))
import conditional_replay as frozen_replay


def calculate():
    stream=[];caches=[]
    def factory():
        c=DevelopmentCache(event_sink=stream.append);caches.append(c);return c
    original=frozen_replay.ContinuousCache
    frozen_replay.ContinuousCache=factory
    try:replay=frozen_replay.calculate(False)
    finally:frozen_replay.ContinuousCache=original
    expected=json.loads((HERE.parent/'nine-frame-conditional-replay-20261011/conditional_results.json').read_text())['without_requests']
    assert replay==expected,'instrumentation changed frozen replay output'
    c=caches[0]
    deletions=[(index,e) for index,e in enumerate(stream) if 'deletion' in e]
    counts=Counter((e['kind'],e['status'],e['deletion']['root_cause'],e['deletion']['association_class']) for _,e in deletions)
    accepted=[(i,e) for i,e in deletions if e['kind']=='relation' and e['deletion']['association_class']=='accepted_hypothesis']
    direct_capacity=[(i,e) for i,e in deletions if e['status']=='evicted_relation']
    assert len(accepted)==1
    index,event=accepted[0];d=event['deletion']
    assert d['reason']=='endpoint_removed_relation' and d['root_reason']=='evicted_observation'
    assert d['root_cause']=='capacity'
    assert d['counts_before']['relations']<c.pr
    assert all(index<i for i,_ in direct_capacity)
    assert all(e['deletion']['association_class']=='unscored_candidate' for _,e in direct_capacity)
    relation_deleted=sum(v for (kind,*_),v in counts.items() if kind=='relation')
    assert 702==relation_deleted+len(c.rels)
    assert 83==c.stats['evicted_observation']+c.stats['expired_observation']+len(c.obs)
    assert 83==c.stats['endpoint_removed_descriptor']+len(c.descriptors)
    legacy=replay['legacy_relation_at_ingest']
    assert legacy['motion_missing_inputs']==['nonadjacent_frames']
    previous=Fraction(legacy['endpoint_bindings'][0]['pts'])*Fraction(legacy['endpoint_bindings'][0]['time_base'])
    current=Fraction(legacy['endpoint_bindings'][1]['pts'])*Fraction(legacy['endpoint_bindings'][1]['time_base'])
    removed=Fraction(event['now'])
    root_events=[e for _,e in deletions if e['kind']=='observation' and e['item']==legacy['a']]
    assert len(root_events)==1
    root=root_events[0]['deletion']
    assert removed-previous==c.ttl and root['counts_before']['observations']==c.po+1<c.to
    assert len(c.keys)==1
    deletion_table=[{'kind':kind,'reason':reason,'root_cause':root,'association_class':association,'count':count}
                    for (kind,reason,root,association),count in sorted(counts.items(),key=lambda x:repr(x[0]))]
    # Export aggregates and the single public hypothesis's lifecycle without raw
    # observation identifiers, bbox values, pixels, HSV or a new association path.
    return {'scope':'one unchanged public nine-frame no-request replay with deletion sink; evidence accounting only',
      'source_commit':'87b4c11fbebf820e5368e0d9ac93b341db523959',
      'frozen_result_exact_match':True,'configuration_changed':False,
      'configuration':{'per_view_observations':c.po,'total_observations':c.to,'per_view_relations':c.pr,
                       'total_relations':c.tr,'per_view_descriptors':c.pd,'ttl_source_seconds':str(c.ttl)},
      'inputs':{'frames':9,'observations':83,'unscored_candidate_relations':701,'accepted_sparse_hypotheses':1},
      'accepted_hypothesis_lifecycle':{
        'label':'existing_public_sparse_endpoint_hypothesis','previous_source_seconds':str(previous),'current_source_seconds':str(current),
        'qualification_at_ingest':legacy['motion_status'],'qualification_missing_inputs':legacy['motion_missing_inputs'],
        'motion_computed':0,'removed_source_seconds':event['now'],'removed_source_pts':int(removed/Fraction('1/30')),'time_base':'1/30',
        'reason':d['reason'],'root_cause':d['root_cause'],'root_reason':d['root_reason'],
        'relation_counts_at_actual_deletion':{'before':d['counts_before']['relations'],'after':d['counts_after']['relations']},
        'relation_limit_not_reached_at_accepted_deletion':True,'deletion_precedes_all_direct_relation_capacity_deletions':True,
        'observation_counts_at_root_deletion':{'before':root['counts_before']['observations'],'after':root['counts_after']['observations']},
        'single_view_per_view_observation_limit_reached':True,
        'prior_endpoint_age_seconds_at_deletion':str(removed-previous),'at_inclusive_TTL_boundary_not_expired':True,
        'importance_assessed':False},
      'direct_relation_capacity_deletions':{'unscored_candidates':len(direct_capacity),'accepted_hypotheses':0,
                                          'source_seconds':sorted({e['now'] for _,e in direct_capacity})},
      'deletion_table':deletion_table,
      'conservation':{'relations_created':702,'relations_deleted':relation_deleted,'live_relations':len(c.rels),
                      'observations_created':83,'observation_capacity_deletions':c.stats['evicted_observation'],
                      'observation_TTL_deletions':c.stats['expired_observation'],'live_observations':len(c.obs),
                      'geometry_descriptors_created':83,'endpoint_cascade_descriptor_deletions':c.stats['endpoint_removed_descriptor'],
                      'live_descriptors':len(c.descriptors)},
      'interpretation':{
        'accepted_loss_due_to_701_candidates_directly_displacing_it':False,
        'actual_retention_failure_path':'observation capacity -> previous endpoint removal -> relation removal',
        'qualification_failure_path':'nonadjacent ingested frames already at relation ingest; independent of deletion',
        'accepted_relation_TTL_deletions_observed':0,
        'priority_or_relation_capacity_change_justified_for_this_loss':False,
        'observation_retention_design_review_supported':True,
        'recognition_gain_supported':False,'accuracy':None,'event_unknown':83},
      'limits':['One published bbox-only replay, not original media or human truth',
                '701 candidates is total enumeration, not simultaneous occupancy or verified associations',
                'Deletion causes are observed paths, not evidence of relation importance',
                'No capacity, priority, TTL, source or gate was altered; no counterfactual configuration was run',
                'No new trajectory links, model inference, pixel/HSV evidence or real negative labels',
                'Retention review is conditional on needing within-TTL availability; TTL currently gives an upper bound only']}

if __name__=='__main__':
    result=calculate();p=HERE/'attribution_results.json'
    if sys.argv[1:]==['--generate']:p.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:raise SystemExit('Only optional --generate supported')
    elif result!=json.loads(p.read_text()):raise SystemExit('Attribution receipt differs')
    print(json.dumps({'frozen_result_exact_match':result['frozen_result_exact_match'],
                      'accepted_loss':result['interpretation']['actual_retention_failure_path'],
                      'direct_relation_capacity_deletions':result['direct_relation_capacity_deletions'],
                      'conservation':result['conservation']},indent=2))
