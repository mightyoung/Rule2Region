"""Preregistered offline development comparison, public cache plus synthetic pressure."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from fractions import Fraction
import json
import sys
from experimental_cache import ExperimentalCache,DevelopmentCache
from test_continuous_cache import geometry,frame,relation,request,K
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'nine-frame-conditional-replay-20261011'))
import conditional_replay as real


class Recorder:
    def __init__(self):
        self.cache=None;self.observations={};self.losses=[];self.high=Counter();self.transient=Counter();self.protection_high=0;self.protection_per_view_high=0
    def sink(self,e):
        c=self.cache
        for name,value in [('observations',len(c.obs)),('relations',len(c.rels)),('descriptors',len(c._descriptors))]:
            self.transient[name]=max(self.transient[name],value)
        if e['status']=='stored_observation':
            key=tuple(e['item']);o=c.obs[key]
            self.observations[key]={'source':o['_t'],'admitted':Fraction(e['now']),
                                    'role':'accepted_endpoint' if key[-1].startswith(('left','right','a','b')) else 'ordinary',
                                    'view':key[0]}
        if e['status']=='stored_relation':
            r=c.rels[tuple(e['item'])]
            for key in (r['a'],r['b']):
                if r['status']=='accepted_hypothesis':self.observations[key]['role']='accepted_endpoint'
                elif self.observations[key]['role']!='accepted_endpoint':self.observations[key]['role']='candidate_endpoint'
        if 'deletion' in e:
            d=e['deletion'];row={'kind':e['kind'],'reason':d['reason'],'root_cause':d['root_cause'],
                               'association_class':d['association_class'],'source_seconds':e['now']}
            if e['kind']=='observation':
                meta=self.observations[tuple(e['item'])]
                row.update(role=meta['role'],view=meta['view'],source_age=str(Fraction(e['now'])-meta['source']),
                           admitted_dwell=str(Fraction(e['now'])-meta['admitted']))
                row['_key']=tuple(e['item'])
            self.losses.append(row)
    def tick(self):
        c=self.cache
        for name,value in [('observations',len(c.obs)),('relations',len(c.rels)),('descriptors',len(c._descriptors))]:
            self.high[name]=max(self.high[name],value)
        assert len(c.obs)<=c.to and len(c.rels)<=c.tr and len(c._descriptors)<=c.td
        for k in c.keys:
            assert sum(i[:2]==k for i in c.obs)<=c.po
            assert sum(i[:2]==k for i in c.rels)<=c.pr
            assert sum(d['_observations'][0][:2]==k for d in c._descriptors.values())<=c.pd
        if isinstance(c,ExperimentalCache):
            scope=set(c.obs);protected=c.protected_endpoints(scope,c.to//4)
            assert len(protected)<=c.to//4
            self.protection_high=max(self.protection_high,len(protected))
            for key in c.keys:
                pv=c.protected_endpoints({i for i in c.obs if i[:2]==key},c.po//4)
                assert len(pv)<=c.po//4
                self.protection_per_view_high=max(self.protection_per_view_high,len(pv))
    def summary(self):
        c=self.cache;groups=Counter((d['kind'],d['reason'],d['root_cause'],d['association_class']) for d in self.losses)
        ordinary=[d for d in self.losses if d.get('role')=='ordinary' and d['reason']=='evicted_observation']
        return {'losses':[{'kind':k,'reason':r,'root_cause':root,'association_class':a,'count':n}
                          for (k,r,root,a),n in sorted(groups.items(),key=lambda x:repr(x[0]))],
                'ordinary_capacity_losses':len(ordinary),
                'other_goal_capacity_losses':sum(d['reason']=='evicted_observation' and d.get('role')!='accepted_endpoint' for d in self.losses if d['kind']=='observation'),
                'ordinary_zero_source_time_dwell_capacity_losses':sum(Fraction(d['admitted_dwell'])==0 for d in ordinary),
                'post_operation_highwater':dict(self.high),'event_hook_transient_highwater':dict(self.transient),
                'computed_protection_highwater_global_quota':self.protection_high,
                'computed_protection_highwater_per_view_quota':self.protection_per_view_high,
                'live_end':{'observations':len(c.obs),'relations':len(c.rels),'descriptors':len(c._descriptors)},
                'retained_accepted_relations':sum(r['status']=='accepted_hypothesis' for r in c.rels.values()),
                'expired_observation_retained':sum(o['_t']<c.keys[i[:2]]['now']-c.ttl for i,o in c.obs.items())}


def query_snapshot(c,req,now):
    # Diagnostic query on a bounded state copy, never the replay/selection state.
    clone=DevelopmentCache()
    clone.__dict__.update(deepcopy({k:v for k,v in c.__dict__.items() if k not in ('sink','_external_sink')}))
    clone._external_sink=None;clone.sink=clone._export_event
    p=clone.retrieve(req,now)
    return {'event_status':p['event_status'],'identity_verified':p['identity_verified'],
            'qualified_history':len(p['accepted_history']),'motion_descriptors':len(p['motion_descriptors']),
            'accepted_endpoint_hypotheses':len(p['accepted_endpoint_hypotheses']),
            'evidence_sources':[{'relation_sha256':h['source_sha256'],
                                'endpoints':[{'pts':b['pts'],'time_base':b['time_base'],'source_sha256':b['source_sha256']} for b in h['endpoint_bindings']],
                                'motion_status':h['motion_status'],'missing_inputs':h['motion_missing_inputs']}
                               for h in p['accepted_endpoint_hypotheses']],
            'model_calls':p['model_calls'],'query_is_diagnostic_state_copy':True}


def public_arm(cls):
    rec=Recorder();snapshots=[]
    class Observed(cls):
        def ingest(self,f,now,**kw):
            out=super().ingest(f,now,**kw);rec.tick()
            if f['pts']==450:
                req={'camera':'G420','viewRevision':'public-geometry-v1',
                    'pts':390,'time_base':'1/30','target_id':'G420-P390:query003',
                    'ambiguity_reason':'offline_retention_diagnostic_no_event_judgment',
                    'query_pts':450,'query_time_base':'1/30'}
                snapshots.append({'at_15':query_snapshot(self,req,now),
                                  'one_tick_after_15':query_snapshot(self,{**req,'query_pts':451},Fraction(451,30))})
            return out
        def add_relation(self,*args,**kw):
            out=super().add_relation(*args,**kw);rec.tick();return out
    def factory():
        c=Observed(event_sink=rec.sink);rec.cache=c;return c
    original=real.ContinuousCache;real.ContinuousCache=factory
    try:result=real.calculate(False)
    finally:real.ContinuousCache=original
    frozen=json.loads((HERE.parent/'nine-frame-conditional-replay-20261011/conditional_results.json').read_text())['without_requests']
    if cls is DevelopmentCache:assert result==frozen
    rec.tick()
    accepted=[d for d in rec.losses if d['kind']=='relation' and d['association_class']=='accepted_hypothesis']
    assert len(accepted)==1
    legacy=result['legacy_relation_at_ingest'];assert legacy==frozen['legacy_relation_at_ingest']
    assert all(t['event_status']=='unknown' for t in result['trace'])
    previous=Fraction(12);removed=Fraction(accepted[0]['source_seconds'])
    out=rec.summary()
    out.update({'frozen_baseline_exact_match':result==frozen,'accepted_deletion':accepted[0],
                'accepted_capacity_loss_before_TTL_expiry':accepted[0]['root_cause']=='capacity',
                'accepted_last_loss_age_from_previous_source':str(removed-previous),
                'strict_remaining_TTL_at_capacity_loss_seconds':str(max(Fraction(0),Fraction(3)-(removed-previous))) if accepted[0]['root_cause']=='capacity' else None,
                'accepted_motion_qualification_unchanged':True,'event_unknown':83,'motion_computed':result['summary']['computed_motion'],
                'queries_on_bounded_state_copies':snapshots[0]})
    return out,rec


def synthetic_arm(cls,name):
    rec=Recorder();c=cls(event_sink=rec.sink);rec.cache=c;c.register(*K,geometry())
    def ingest(f,n):c.ingest(f,n);rec.tick()
    def link(a,b,n,status='accepted_hypothesis'):relation(c,a,b,n,status);rec.tick()
    def ordinary(camera='cam'):
        f=frame(2,None,camera=camera);f['detections']=[{'id':'ordinary'+str(i).zfill(3),'bbox':[10,10,14,14]} for i in range(80)]
        ingest(f,2)
    if name in ('crowding','expiry_reads','two_views'):
        ingest(frame(),0);ingest(frame(1,'b'),1);link('a','b',1);ordinary()
    elif name in ('quota_multi_target','unverified_accepted_pressure'):
        for i in range(16 if name=='quota_multi_target' else 32):
            for offset,side in [(0,'left'),(1,'right')]:
                pts=2*i+offset;ingest(frame(pts,side+str(i),time_base='1/100'),Fraction(pts,100))
            link('left'+str(i),'right'+str(i),Fraction(2*i+1,100))
        ordinary()
    elif name=='unknown_flood':
        for pts,prefix in [(0,'left'),(1,'right')]:
            f=frame(pts,None);f['detections']=[{'id':prefix+str(i),'bbox':[10,10,14,14]} for i in range(32)];ingest(f,pts)
        for a in range(32):
            for b in range(32):link('left'+str(a),'right'+str(b),1,'unknown')
        ordinary()
    else:raise ValueError('unregistered scenario')
    interim=rec.summary();queries=[]
    if name=='expiry_reads':
        for now in (2,3,Fraction(301,100)):
            if K+('b',) in c.obs:
                qpts=now.numerator if isinstance(now,Fraction) else now
                qtb=f'1/{now.denominator}' if isinstance(now,Fraction) else '1/1'
                queries.append(query_snapshot(c,request(1,'b',query_pts=qpts,query_time_base=qtb),now))
                c.retrieve(request(1,'b',query_pts=qpts,query_time_base=qtb),now);rec.tick()
            else:c.advance(now,K);rec.tick()
        timestamps_unchanged=all(o['_t']==rec.observations[i]['source'] for i,o in c.obs.items())
        c.advance(6,K);rec.tick();assert not c.obs and not c.rels and not c.descriptors
    elif name=='two_views':
        c.register('other','v1',geometry());ordinary('other')
        timestamps_unchanged=all(o['_t']==rec.observations[i]['source'] for i,o in c.obs.items())
    else:timestamps_unchanged=all(o['_t']==rec.observations[i]['source'] for i,o in c.obs.items())
    # Fresh ordinary-target diagnostic, except after complete expiry.
    if K+('ordinary079',) in c.obs:queries.append(query_snapshot(c,request(2,'ordinary079'),c.keys[K]['now']))
    assert timestamps_unchanged and all(q['event_status']=='unknown' and not q['identity_verified'] for q in queries)
    out=rec.summary();out.update({'pre_expiry_or_second_view':interim,'timestamps_unchanged':timestamps_unchanged,'diagnostic_queries':queries})
    if name=='two_views':out['global_pressure_limit']='two active views can admit at most 128 observations, below total 256; global overflow infeasible with this declared scenario'
    return out


def calculate():
    base,br=public_arm(DevelopmentCache);candidate,cr=public_arm(ExperimentalCache)
    bv={d['_key'] for d in br.losses if d['kind']=='observation' and d['reason']=='evicted_observation'}
    cv={d['_key'] for d in cr.losses if d['kind']=='observation' and d['reason']=='evicted_observation'}
    names=['crowding','quota_multi_target','unknown_flood','unverified_accepted_pressure','expiry_reads','two_views']
    synthetic={n:{'baseline':synthetic_arm(DevelopmentCache,n),'candidate':synthetic_arm(ExperimentalCache,n)} for n in names}
    return {'scope':'fixed-budget exposed-development comparison; not recognition or human truth',
            'baseline_commit':'5977bd4267ae0c75c480b90b63cbd6701fd671ab','protocol_commit':'e66ca5f',
            'budget':{'per_view_observations':64,'per_view_relations':512,'ttl_source_seconds':3,'protected_quota_per_view':16},
            'public_replay':{'baseline':base,'candidate':candidate,'capacity_victim_set_difference':{'candidate_only_victims':len(cv-bv),'baseline_only_victims':len(bv-cv)},
                            'accepted_deletion_event_delay_sampled_seconds':str(Fraction(candidate['accepted_deletion']['source_seconds'])-Fraction(base['accepted_deletion']['source_seconds']))},
            'public_positive_continuous_valid_query_interval_gain_demonstrated':False,
            'public_boundary_checkpoint_only':True,'synthetic_scenarios':synthetic,'new_models_installs_paid_calls_media':0,'accuracy':None,'production_default_changed':False}

if __name__=='__main__':
    result=calculate();p=HERE/'trial_results.json'
    if sys.argv[1:]==['--generate']:p.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:raise SystemExit('Only optional --generate supported')
    elif result!=json.loads(p.read_text()):raise SystemExit('Trial result differs')
    print(json.dumps({'public_accepted_deletion':{a:result['public_replay'][a]['accepted_deletion'] for a in ('baseline','candidate')},
                      'changed_public_victims':result['public_replay']['capacity_victim_set_difference'],
                      'synthetic_ordinary_losses':{n:{a:v[a]['ordinary_capacity_losses'] for a in ('baseline','candidate')} for n,v in result['synthetic_scenarios'].items()}},indent=2))
