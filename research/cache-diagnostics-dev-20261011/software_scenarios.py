"""Finite synthetic development receipts, never an image recognition experiment."""
from pathlib import Path
import json
import sys
from development_cache import DevelopmentCache
from test_continuous_cache import geometry,frame,request,relation,K


def scenarios():
    def make(**kwargs):
        c=DevelopmentCache(**kwargs);c.register(*K,geometry());return c
    sparse=make()
    sparse.ingest(frame(0,'a'),0);sparse.ingest(frame(1,None),1);sparse.ingest(frame(2,'b'),2)
    relation(sparse,now=2)
    payload=sparse.retrieve(request(2,'b'),2)
    samples={}
    for first_status in ('unknown','accepted_hypothesis'):
        events=[];c=make(per_view_rel=1,event_sink=events.append)
        c.ingest(frame(),0);c.ingest(frame(1,'b'),1);relation(c,status=first_status)
        c.ingest(frame(2,'c'),2);relation(c,'b','c',2,'unknown' if first_status=='accepted_hypothesis' else 'accepted_hypothesis')
        samples['relation_capacity_first_'+first_status]=[e for e in events if 'deletion' in e]
    events=[];ttl=make(event_sink=events.append)
    ttl.ingest(frame(),0);ttl.ingest(frame(1,None),1);ttl.ingest(frame(2,'b'),2);relation(ttl,now=2)
    ttl.retrieve(request(2,'b',query_pts=4,query_time_base='1/1'),4)
    samples['valid_query_TTL_maintenance']=[e for e in events if 'deletion' in e]
    return {'scope':'synthetic software development only; no nine-frame image or recognition replay',
            'retained_sparse_endpoint_hypotheses':payload['accepted_endpoint_hypotheses'],
            'qualified_continuous_history_count':len(payload['accepted_history']),
            'motion_descriptor_count':len(payload['motion_descriptors']),
            'event_status':payload['event_status'],'deletion_stream_examples':samples,
            'unchanged_default_capacities':{'observations_per_view':sparse.po,'relations_per_view':sparse.pr,
                                          'descriptors_per_view':sparse.pd,'ttl_source_seconds':str(sparse.ttl)},
            'limits':['small capacities are explicit synthetic pressure tests, not changed defaults',
                      'no assertion that a real important relation was evicted',
                      'stream ownership/backpressure/durability remain caller responsibilities',
                      'accepted endpoint hypotheses are not person identity or event GT'],
            'model_calls':0,'new_installations':0,'accuracy':None}


if __name__=='__main__':
    result=scenarios();p=Path(__file__).with_name('software_results.json')
    if sys.argv[1:]==['--generate']:p.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:raise SystemExit('Only optional --generate supported')
    elif result!=json.loads(p.read_text()):raise SystemExit('Software scenario receipt differs')
    print(json.dumps({'retained_sparse_hypotheses':len(result['retained_sparse_endpoint_hypotheses']),
                      'qualified_history':result['qualified_continuous_history_count'],
                      'motion_descriptors':result['motion_descriptor_count'],'event_status':result['event_status']},indent=2))
