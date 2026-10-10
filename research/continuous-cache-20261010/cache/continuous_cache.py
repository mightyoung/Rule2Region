"""Continuously derived, bounded causal evidence cache. Standard library only.

No detector/model client, image loader, network client or event classifier.
The vendored legacy cache is unchanged. Its atomic admission/bounds are reused.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from vendor.base_cache import Cache, token, sha, time_of, clock
from vendor import roi_core as R


def content_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def point_segment(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)
    u = min(1, max(0, t))
    return math.hypot(point[0]-a[0]-u*dx, point[1]-a[1]-u*dy), t


def portal_distances(point, portal):
    p = portal['points']
    rational = R._rational(p)
    q = tuple(Fraction(x) for x in point)
    inside = any(R._on_segment(q, rational[i-1], rational[i]) for i in range(len(p)))
    if not inside:
        # Exact rational horizontal ray crossing supports every validated simple ring.
        parity = False
        for i, b in enumerate(rational):
            a = rational[i-1]
            if (a[1] > q[1]) != (b[1] > q[1]):
                crossing_x = a[0] + (q[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if q[0] < crossing_x: parity = not parity
        inside = parity
    boundary = min(point_segment(point, p[i-1], p[i])[0] for i in range(len(p)))
    mean = [sum(v[j] for v in p)/len(p) for j in (0, 1)]
    return {'reference_id': portal['reference_id'], 'door_id': portal.get('door_id', portal['reference_id']),
            'set_distance_px': 0.0 if inside else boundary, 'boundary_distance_px': boundary,
            'vertex_mean_distance_px': math.dist(point, mean), 'inside_closed_polygon': inside}


def anchors(box):
    return {'center': [(box[0]+box[2])/2, (box[1]+box[3])/2],
            'bottom_proxy': [(box[0]+box[2])/2, box[3]]}


class ContinuousCache(Cache):
    def __init__(self, per_view_descriptors=128, total_descriptors=512, max_motion_gap=2, **kwargs):
        for n in (per_view_descriptors, total_descriptors, max_motion_gap):
            if type(n) is not int or n <= 0:
                raise ValueError('positive integer derived capacities/gap required')
        super().__init__(**kwargs)
        self.pd, self.td, self.max_gap = per_view_descriptors, total_descriptors, Fraction(max_motion_gap)
        self._descriptors = {}
        self._geometry, self._active, self._frame_no = {}, {}, {}
        self._ingest_meta = None
        self._dseq = 0

    @property
    def descriptors(self):
        """Read-only-by-copy inspection; retrieval remains explicitly ambiguity-gated."""
        return deepcopy(self._descriptors)

    @staticmethod
    def validate_geometry(geometry):
        g = deepcopy(geometry)
        if not isinstance(g, dict) or set(g) != {'width','height','portal_source_sha256','available_at','portals','sill'}:
            raise ValueError('geometry schema')
        for x in ('width', 'height'):
            if type(g[x]) is not int or not 0 < g[x] <= 100000:
                raise ValueError('geometry dimensions')
        sha(g['portal_source_sha256']); clock(g['available_at'])
        if not isinstance(g['portals'], list) or not 1 <= len(g['portals']) <= 8:
            raise ValueError('portal capacity')
        seen = set()
        for p in g['portals']:
            if not isinstance(p, dict) or not {'reference_id','points'} <= set(p) or set(p)-{'reference_id','points','door_id'}:
                raise ValueError('portal schema')
            token(p['reference_id']); token(p.get('door_id', p['reference_id']))
            if p['reference_id'] in seen: raise ValueError('duplicate portal reference')
            seen.add(p['reference_id'])
            if not isinstance(p['points'], list) or not 3 <= len(p['points']) <= 32:
                raise ValueError('portal vertex capacity')
            valid = R.polygon_validation(p['points'], g['width'], g['height'])
            if not valid['valid']: raise ValueError('invalid portal: '+str(valid['reason']))
            # The legacy validator accepts an explicit closing vertex; normalize it
            # before edge-distance iteration so there is no zero-length final edge.
            if p['points'][0] == p['points'][-1]: p['points'] = p['points'][:-1]
            for i, b in enumerate(p['points']):
                a = p['points'][i-1]
                if (b[0]-a[0])**2+(b[1]-a[1])**2 == 0: raise ValueError('portal edge not numerically representable')
        s = g['sill']
        if s is not None:
            required = {'reference_id','points','source_sha256','source_frame_sha256','source_pts','time_base','available_at','provenance','coordinate_tolerance_px'}
            if not isinstance(s, dict) or not required <= set(s) or set(s)-required-{'door_id'}:
                raise ValueError('sill schema')
            token(s['reference_id']); token(s.get('door_id', s['reference_id']));token(s['provenance'])
            sha(s['source_sha256']);sha(s['source_frame_sha256'])
            t = time_of(s['source_pts'], s['time_base'])
            if clock(s['available_at']) < t: raise ValueError('sill availability before source')
            if not isinstance(s['points'],list) or len(s['points']) != 2: raise ValueError('sill points')
            for p in s['points']:
                if not isinstance(p,list) or len(p)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p):raise ValueError('sill coordinate')
                if not (0 <= p[0] <= g['width'] and 0 <= p[1] <= g['height']):raise ValueError('sill out of image')
            if s['points'][0] == s['points'][1]: raise ValueError('zero sill')
            a,b=s['points']
            if (b[0]-a[0])**2+(b[1]-a[1])**2 == 0:raise ValueError('sill edge not numerically representable')
            if type(s['coordinate_tolerance_px']) not in (int,float) or not math.isfinite(s['coordinate_tolerance_px']) or not 0 <= s['coordinate_tolerance_px'] <= 100000:raise ValueError('sill tolerance')
        return g

    def register(self, camera, viewRevision, geometry):
        k = (token(camera), token(viewRevision)); g = self.validate_geometry(geometry)
        if k in self.keys:
            if self._active.get(camera) != viewRevision: raise ValueError('retired viewRevision')
            if self._geometry[k] != g: raise ValueError('geometry changed within viewRevision')
            return
        # Registration may fail on capacity; do not invalidate old view until it succeeds.
        super().register(camera, viewRevision)
        old = self._active.get(camera)
        if old is not None:
            for oid in list(self.obs):
                if oid[:2] == (camera, old): self.remove_obs(oid, 'view_changed_observation')
        self._active[camera] = viewRevision;self._geometry[k] = g;self._frame_no[k] = 0

    def key(self, camera, viewRevision):
        k = super().key(camera, viewRevision)
        if self._active.get(camera) != viewRevision: raise ValueError('inactive viewRevision')
        return k

    @staticmethod
    def binding(o):
        return {'observation_id':o['id'], 'camera':o['camera'], 'viewRevision':o['viewRevision'],
                'pts':o['pts'], 'time_base':o['time_base'], 'source_sha256':o['source_sha256'],
                'received_at':o['received_at'], 'available_at':o['available_at']}

    def base_descriptor(self, kind, oids, now):
        endpoints = [self.binding(self.obs[i]) for i in oids]
        return {'kind':kind, 'endpoint_bindings':endpoints, 'computed_at':str(now), 'available_at':str(now),
                'source_groups':[content_id(e) for e in endpoints], 'identity_verified':False,
                'physical_side':'unknown', 'footcontact':'unknown', 'event_status':'unknown'}

    def store_descriptor(self, value, oids, relation=None):
        value = deepcopy(value); did = content_id(value);self._dseq += 1
        value.update(descriptor_id=did, _observations=tuple(oids), _relation=relation, _seq=self._dseq)
        self._descriptors[did] = value;self.event('computed_'+value['kind'], 'descriptor', did)
        k = oids[0][:2]
        for same_key, limit in ((True,self.pd),(False,self.td)):
            while True:
                items = [(d,v) for d,v in self._descriptors.items() if not same_key or v['_observations'][0][:2]==k]
                if len(items) <= limit: break
                self.remove_descriptor(min(items,key=lambda x:x[1]['_seq'])[0], 'evicted_descriptor')

    def remove_descriptor(self, did, reason):
        del self._descriptors[did]; self.event(reason, 'descriptor', did)

    def remove_obs(self, oid, status):
        for did,d in list(self._descriptors.items()):
            if oid in d['_observations']:self.remove_descriptor(did,'endpoint_removed_descriptor')
        super().remove_obs(oid,status)

    def remove_rel(self, rid, status):
        for did,d in list(self._descriptors.items()):
            if rid == d['_relation']:self.remove_descriptor(did,'relation_removed_descriptor')
        super().remove_rel(rid,status)

    def event(self, status, kind='', item=None):
        super().event(status,kind,item)
        if status == 'stored_observation':
            o = self.obs[item]; meta = self._ingest_meta
            o.update(received_at=str(meta['now']), available_at=str(meta['available']), _frame_no=self._frame_no[item[:2]]+1)
            self.compute_geometry(item,meta['now'])

    def compute_geometry(self, oid, now):
        o = self.obs[oid]; g = self._geometry[oid[:2]]
        d = self.base_descriptor('geometry', [oid], now)
        d.update(geometry_source_sha256=g['portal_source_sha256'], anchors={}, geometry_status='valid',
                 center_in_any_portal=False, bbox_hits=[], candidate_door_ids=[], anchor_semantics={'center':'bbox center','bottom_proxy':'bbox bottom-center; not foot/contact'})
        if clock(g['available_at']) > now:
            d.update(geometry_status='unknown_unavailable_reference', center_in_any_portal=None)
        else:
            for name, point in anchors(o['bbox']).items():
                distances = [portal_distances(point,p) for p in g['portals']]
                sill = None; s = g['sill']
                if s and clock(s['available_at']) <= now:
                    a,b = s['points']; finite,projection = point_segment(point,a,b)
                    dx,dy = b[0]-a[0],b[1]-a[1]
                    normal = (dx*(point[1]-a[1])-dy*(point[0]-a[0]))/math.hypot(dx,dy)
                    sill = {**deepcopy(s), 'signed_normal_px':normal, 'absolute_normal_px':abs(normal), 'finite_segment_distance_px':finite,
                            'projection_t':projection, 'within_observed_fragment':0 <= projection <= 1,
                            'projection_status':'within_fragment' if 0 <= projection <= 1 else 'extrapolated_supporting_line_only',
                            'door_id':s.get('door_id',s['reference_id']), 'physical_side':'unknown',
                            'applicability':'fixed image reference only; target door unresolved, never physical side or same-door event proof'}
                d['anchors'][name] = {'xy':point,'portals':distances,'sill':sill}
            d['center_in_any_portal'] = any(p['inside_closed_polygon'] for p in d['anchors']['center']['portals'])
            for p in g['portals']:
                routed = R.route_box(o['bbox'],p['points'],g['width'],g['height'])
                hit = None if routed['decision']=='review_unknown' else routed['intersection_area_px2'] > 0 or routed['boundary_touch']
                if hit is None:d['geometry_status']='partial_unknown_route'
                d['bbox_hits'].append({'reference_id':p['reference_id'],'door_id':p.get('door_id',p['reference_id']), 'hit':hit, 'geometry':routed})
                if hit:d['candidate_door_ids'].append(p.get('door_id',p['reference_id']))
        self.store_descriptor(d,[oid])

    def ingest(self, frame, now, available_at=None):
        if not isinstance(frame,dict):raise ValueError('frame schema')
        k = self.key(frame.get('camera'),frame.get('viewRevision'))
        t = time_of(frame.get('pts'),frame.get('time_base'));n = clock(now)
        available = n if available_at is None else clock(available_at)
        if not t <= available <= n:raise ValueError('invalid source availability')
        g=self._geometry[k]
        for det in frame.get('detections',[]):
            box=det.get('bbox') if isinstance(det,dict) else None
            if not isinstance(box,list) or len(box)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in box):raise ValueError('bbox schema')
            if not (0 <= box[0] < box[2] <= g['width'] and 0 <= box[1] < box[3] <= g['height']):raise ValueError('bbox outside registered image')
            if (box[2]-box[0])*(box[3]-box[1]) == 0:raise ValueError('bbox area not numerically representable')
        self._ingest_meta={'now':n,'available':available}
        try:result=super().ingest(frame,n)
        finally:self._ingest_meta=None
        if result=='stored':self._frame_no[k]+=1
        return result

    def add_relation(self,camera,viewRevision,previous_id,current_id,status,source_sha256,now,*,endpoints,available_at=None):
        k=self.key(camera,viewRevision);a=k+(token(previous_id),);b=k+(token(current_id),);n=clock(now)
        available=n if available_at is None else clock(available_at)
        if b not in self.obs or a not in self.obs:raise ValueError('relation missing endpoints')
        if not max(self.obs[i]['_t'] for i in (a,b)) <= available <= n:raise ValueError('invalid relation availability')
        if available < max(clock(self.obs[i]['available_at']) for i in (a,b)):raise ValueError('relation unavailable endpoints')
        result=super().add_relation(camera,viewRevision,previous_id,current_id,status,source_sha256,n,endpoints=endpoints)
        rid=k+(previous_id,current_id)
        if result!='stored' or rid not in self.rels:return result
        r=self.rels[rid];r.update(received_at=str(n),available_at=str(n), upstream_available_at=str(available),endpoint_bindings=[self.binding(self.obs[i]) for i in (a,b)])
        reasons=[]
        if status!='accepted_hypothesis':reasons.append('unknown_association')
        if self.obs[b]['_t']-self.obs[a]['_t']>self.max_gap:reasons.append('max_time_gap_exceeded')
        if self.obs[b]['_frame_no']!=self.obs[a]['_frame_no']+1:reasons.append('nonadjacent_frames')
        r['motion_status']='unknown_motion' if reasons else 'qualified_hypothesis';r['motion_missing_inputs']=reasons
        if not reasons:
            d=self.base_descriptor('motion',[a,b],n);dt=self.obs[b]['_t']-self.obs[a]['_t']
            d.update(relation_source_sha256=source_sha256,association_status=status, dt_seconds=str(dt),anchors={},motion_semantics='sparse accepted-hypothesis chord, not verified person trajectory',direction_continuity='unknown_single_chord')
            aa,bb=anchors(self.obs[a]['bbox']),anchors(self.obs[b]['bbox'])
            for name in aa:
                dx,dy=bb[name][0]-aa[name][0],bb[name][1]-aa[name][1];norm=math.hypot(dx,dy)
                d['anchors'][name]={'dx_px':dx,'dy_px':dy,'norm_px':norm,'bearing_degrees':math.degrees(math.atan2(dy,dx)) if norm else None,'unit_vector':[dx/norm,dy/norm] if norm else None}
            self.store_descriptor(d,[a,b],rid)
        return result

    def retrieve(self,request,now):
        if request is None:return None
        required={'camera','viewRevision','pts','time_base','target_id','ambiguity_reason'};optional={'query_pts','query_time_base'}
        if not isinstance(request,dict) or not required <= set(request) or set(request)-required-optional or bool('query_pts' in request)!=bool('query_time_base' in request):raise ValueError('request schema fields')
        k=self.key(request['camera'],request['viewRevision']);target=k+(token(request['target_id']),);reason=token(request['ambiguity_reason'])
        t=time_of(request['pts'],request['time_base']);n=clock(now)
        q=time_of(request.get('query_pts',request['pts']),request.get('query_time_base',request['time_base']))
        if not t <= q <= n:raise ValueError('noncausal query')
        if target not in self.obs or self.obs[target]['_t']!=t or t<n-self.ttl:raise ValueError('current target/time missing or expired')
        if clock(self.obs[target]['available_at'])>q or clock(self.obs[target]['received_at'])>q:raise ValueError('target not available at query')
        self.advance(n,k);self.stats['retrievals']+=1
        live={i:o for i,o in self.obs.items() if i[:2]==k and max(n-self.ttl,t-self.ttl)<=o['_t']<=t and clock(o['available_at'])<=q and clock(o['received_at'])<=q}
        eligible={did:d for did,d in self._descriptors.items() if all(i in live for i in d['_observations']) and clock(d['available_at'])<=q and clock(d['computed_at'])<=q}
        motions={d['_relation']:d for d in eligible.values() if d['kind']=='motion'}
        relations={rid:r for rid,r in self.rels.items() if r['a'] in live and r['b'] in live and clock(r['available_at'])<=q}
        chain=[];motion=[];missing=[];cursor=target
        while True:
            prior=[(rid,r) for rid,r in relations.items() if r['b']==cursor and r['status']=='accepted_hypothesis']
            if not prior:break
            rid,r=prior[0]
            if rid not in motions:
                missing.extend(r['motion_missing_inputs'] or ['motion_descriptor_unavailable']);break
            motion.append(self.public(motions[rid]));cursor=r['a'];chain.append(self.public(live[cursor]))
        history=list(reversed(chain));wanted={target}|{k+(o['id'],) for o in history}
        geometric=[self.public(d) for d in eligible.values() if d['kind']=='geometry' and d['_observations'][0] in wanted]
        geometric.sort(key=lambda d:time_of(d['endpoint_bindings'][0]['pts'],d['endpoint_bindings'][0]['time_base']))
        unknown=[{'previous':self.public(live[r['a']]),'current_id':target[-1], 'status':'unknown','motion_status':'unknown_motion','source_sha256':r['source_sha256'],'endpoint_bindings':deepcopy(r['endpoint_bindings']),'identity_verified':False} for r in relations.values() if r['status']=='unknown' and r['b']==target]
        if not history:missing.append('qualified_accepted_history')
        if not any(d['endpoint_bindings'][0]['observation_id']==target[-1] for d in geometric):missing.append('current_geometry_descriptor')
        missing.append('automatic_side_evidence')
        return {'current':self.public(live[target]), 'accepted_history':history,'accepted_history_semantics':'qualified upstream association hypotheses only, not person identity GT',
                'unknown_candidates':unknown,'geometry_descriptors':geometric,'motion_descriptors':list(reversed(motion)),
                'ambiguity_reason':reason,'query_time':str(q),'missing_inputs':sorted(set(missing)),
                'inference_missing_inputs':sorted(set(missing)), 'evaluation_gaps':['independent_event_reference','real_negative_events'],
                'event_status':'unknown','entry':'unknown','safety':'unknown','footcontact':'unknown','hidden_crossing':'unknown','predicate_A':'unknown','predicate_B':'unknown',
                'identity_verified':False,'independent_feature_votes':0,'model_calls':0,'no_device_action':True}
