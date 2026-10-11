"""Versioned software adapter; frozen cache, fixtures and outputs stay unchanged."""
from pathlib import Path
from copy import deepcopy
import sys
FROZEN=Path(__file__).resolve().parents[1]/'continuous-cache-20261010/cache'
sys.path.insert(0,str(FROZEN))
from continuous_cache import ContinuousCache
from vendor.base_cache import clock, time_of


class DevelopmentCache(ContinuousCache):
    """Adds endpoint evidence and deletion telemetry; never relaxes motion gates."""
    def __init__(self, **kwargs):
        self._deletion_context=[]
        self._external_sink=kwargs.get('event_sink')
        kwargs['event_sink']=self._export_event
        super().__init__(**kwargs)

    def _counts(self):
        return {'observations':len(self.obs),'relations':len(self.rels),
                'descriptors':len(self._descriptors)}

    def _export_event(self, event):
        # Existing bounded audit ring is unchanged. Enrichment is streamed only;
        # this adapter retains no deletion history or deleted observations.
        if self._external_sink is None:
            return
        if self._deletion_context:
            context=self._deletion_context[-1]
            item=list(context['item']) if isinstance(context['item'],tuple) else context['item']
            if (event['kind'],event['item'],event['status'])==(context['kind'],item,context['reason']):
                event['deletion']=deepcopy({k:v for k,v in context.items() if k!='item'})
                event['deletion']['counts_after']=self._counts()
        self._external_sink(event)

    def _remove_with_diagnostic(self, kind, item, reason, record, remove):
        if self._deletion_context:
            root=self._deletion_context[0]
            cause,root_reason=root['root_cause'],root['root_reason']
        else:
            cause=('capacity' if reason.startswith('evicted_') else
                   'source_ttl' if reason.startswith('expired_') else
                   'view_change' if reason.startswith('view_changed_') else 'explicit_removal')
            root_reason=reason
        association=None
        descriptor_kind=None
        if kind=='relation':
            association=record['status']
        elif kind=='descriptor':
            descriptor_kind=record['kind']
            relation=self.rels.get(record['_relation'])
            if relation is not None:association=relation['status']
        context={'schema_version':1,'kind':kind,'item':item,'reason':reason,
                 'root_cause':cause,'root_reason':root_reason,
                 'association_class':'unscored_candidate' if association=='unknown' else association,
                 'descriptor_kind':descriptor_kind,'counts_before':self._counts(),
                 'event_status':'unknown','importance_assessed':False}
        self._deletion_context.append(context)
        try:
            remove(item,reason)
        finally:
            self._deletion_context.pop()

    def remove_obs(self, oid, status):
        self._remove_with_diagnostic('observation',oid,status,self.obs[oid],super().remove_obs)

    def remove_rel(self, rid, status):
        self._remove_with_diagnostic('relation',rid,status,self.rels[rid],super().remove_rel)

    def remove_descriptor(self, did, reason):
        self._remove_with_diagnostic('descriptor',did,reason,self._descriptors[did],super().remove_descriptor)

    def retrieve(self, request, now):
        payload=super().retrieve(request,now)
        if payload is None:return None
        # The base validates before maintenance, then advances source-time TTL.
        # Export only hypotheses into this target, within the same causal window.
        k=(request['camera'],request['viewRevision']);target=k+(request['target_id'],)
        t=time_of(request['pts'],request['time_base']);n=clock(now);q=clock(payload['query_time'])
        live={i:o for i,o in self.obs.items() if i[:2]==k and max(n-self.ttl,t-self.ttl)<=o['_t']<=t
              and clock(o['available_at'])<=q and clock(o['received_at'])<=q}
        hypotheses=[]
        for rid,r in self.rels.items():
            if r['b']!=target or r['status']!='accepted_hypothesis':continue
            if r['a'] not in live or r['b'] not in live or clock(r['available_at'])>q:continue
            a,b=live[r['a']],live[r['b']]
            descriptor_available=any(d['_relation']==rid and all(i in live for i in d['_observations'])
                                     and clock(d['available_at'])<=q and clock(d['computed_at'])<=q
                                     for d in self._descriptors.values() if d['kind']=='motion')
            hypotheses.append({'previous_id':a['id'],'current_id':b['id'],
                'source_sha256':r['source_sha256'],'endpoint_bindings':deepcopy(r['endpoint_bindings']),
                'received_at':r['received_at'],'available_at':r['available_at'],
                'upstream_available_at':r['upstream_available_at'],
                'span_seconds':str(b['_t']-a['_t']),
                'intervening_ingested_frames':b['_frame_no']-a['_frame_no']-1,
                'intermediate_association_status':'not_established',
                'motion_status':r['motion_status'],'motion_missing_inputs':list(r['motion_missing_inputs']),
                'motion_descriptor_available':descriptor_available,
                'direction_continuity':'unknown','identity_verified':False,'event_status':'unknown'})
        payload['accepted_endpoint_hypotheses']=hypotheses
        payload['accepted_endpoint_hypotheses_semantics']='retained incoming endpoint hypotheses, not continuous history, identity GT or event proof'
        return payload
