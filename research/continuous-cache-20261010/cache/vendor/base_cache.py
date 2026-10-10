"""Bounded reference cache. No model, images, identity or region inference."""
from fractions import Fraction
from collections import Counter, deque
from copy import deepcopy
import hashlib,json,math,re

def token(x):
 if not isinstance(x,str) or not x.strip() or len(x)>160 or x.lower() in ('unknown','null','none'):raise ValueError('invalid identifier')
 return x

def sha(x):
 if not isinstance(x,str) or not re.fullmatch('[0-9a-f]{64}',x):raise ValueError('invalid source hash')
 return x

def time_of(pts,tb):
 if type(pts) is not int or pts<0 or not isinstance(tb,str) or not re.fullmatch(r'[1-9][0-9]*/[1-9][0-9]*',tb):raise ValueError('invalid PTS/timebase')
 if len(tb)>40 or pts>2**63-1:raise ValueError('time bound')
 return pts*Fraction(tb)

def clock(x):
 if type(x) not in (int,str,Fraction):raise ValueError('now must be exact int/rational string/Fraction')
 try:y=Fraction(x)
 except (ValueError,ZeroDivisionError):raise ValueError('invalid now')
 if y<0:raise ValueError('negative now')
 return y

class Cache:
 def __init__(self,ttl=3,per_view_obs=64,per_view_rel=512,total_obs=256,total_rel=2048,max_cameras=4,max_keys=8,audit_size=256,event_sink=None,allowed_rgb_pointers=()):
  for v in (ttl,per_view_obs,per_view_rel,total_obs,total_rel,max_cameras,max_keys,audit_size):
   if type(v) is not int or v<=0:raise ValueError('positive integer capacities required')
  self.ttl=Fraction(ttl);self.po=per_view_obs;self.pr=per_view_rel;self.to=total_obs;self.tr=total_rel;self.mc=max_cameras;self.mk=max_keys
  self.seq=0;self.keys={};self.obs={};self.rels={};self.now=Fraction(0);self.stats=Counter();self.events=deque(maxlen=audit_size);self.sink=event_sink
  self.pointers=frozenset(allowed_rgb_pointers)
  if len(self.pointers)>64 or any(not isinstance(p,str) or len(p)>512 for p in self.pointers):raise ValueError('pointer allowlist bound')
 def event(self,status,kind='',item=None):
  self.stats[status]+=1
  e={'status':status,'kind':kind,'item':list(item) if isinstance(item,tuple) else item,'now':str(self.now)}
  if len(self.events)==self.events.maxlen:self.stats['audit_ring_dropped']+=1
  self.events.append(e)
  if self.sink:self.sink(deepcopy(e))
 def register(self,camera,viewRevision):
  k=(token(camera),token(viewRevision))
  if k in self.keys:return
  if len(self.keys)>=self.mk or (camera not in {x[0] for x in self.keys} and len({x[0] for x in self.keys})>=self.mc):raise ValueError('registration capacity; explicit new session required')
  self.keys[k]={'watermark':None,'fingerprint':None,'now':Fraction(0)}
 def key(self,camera,viewRevision):
  k=(token(camera),token(viewRevision))
  if k not in self.keys:raise ValueError('unknown camera/viewRevision')
  return k
 def remove_rel(self,rid,status):
  del self.rels[rid];self.event(status,'relation',rid)
 def remove_obs(self,oid,status):
  del self.obs[oid];self.event(status,'observation',oid)
  for rid in list(self.rels):
   r=self.rels[rid]
   if oid in (r['a'],r['b']):
    self.stats['expired_endpoint_relation' if status=='expired_observation' else 'evicted_endpoint_relation']+=1
    self.remove_rel(rid,'endpoint_removed_relation')
 def advance(self,now,key=None):
  if key is None:
   if len(self.keys)!=1:raise ValueError('explicit view key required for maintenance')
   key=next(iter(self.keys))
  if key not in self.keys:raise ValueError('unregistered maintenance key')
  now=clock(now)
  if now<self.keys[key]['now']:raise ValueError('clock rewind within view')
  self.keys[key]['now']=now;self.now=now
  for oid,o in list(self.obs.items()):
   if oid[:2]==key and o['_t']<now-self.ttl:self.remove_obs(oid,'expired_observation')
 def bounds(self,k):
  while len([i for i in self.obs if i[:2]==k])>self.po:
   self.remove_obs(min((i for i in self.obs if i[:2]==k),key=lambda i:(self.obs[i]['_t'],i)),'evicted_observation')
  while len(self.obs)>self.to:self.remove_obs(min(self.obs,key=lambda i:(self.obs[i]['_seq'],i)),'evicted_observation')
  while len([i for i in self.rels if i[:2]==k])>self.pr:
   self.remove_rel(min((i for i in self.rels if i[:2]==k),key=lambda i:(self.rels[i]['_t'],i)),'evicted_relation')
  while len(self.rels)>self.tr:self.remove_rel(min(self.rels,key=lambda i:(self.rels[i]['_seq'],i)),'evicted_relation')
 def ingest(self,f,now):
  if not isinstance(f,dict) or set(f)!={'camera','viewRevision','pts','time_base','source_sha256','model','detections','rgb_pointer'}:raise ValueError('frame schema fields')
  k=self.key(f['camera'],f['viewRevision']);t=time_of(f['pts'],f['time_base']);n=clock(now)
  if t>n:raise ValueError('future frame')
  source=sha(f['source_sha256']);model=f['model']
  if not isinstance(model,dict) or not model or len(json.dumps(model,allow_nan=False))>1024:raise ValueError('model metadata invalid')
  if any(not isinstance(a,str) or not isinstance(b,str) for a,b in model.items()):raise ValueError('model metadata must be flat strings')
  rgb=f.get('rgb_pointer')
  if rgb is not None and rgb not in self.pointers:raise ValueError('RGB pointer not registered existing reference')
  ds=f['detections']
  if not isinstance(ds,list) or len(ds)>4096:raise ValueError('frame detection bound')
  normalized=[];ids=set()
  for d in ds:
   if not isinstance(d,dict) or set(d)!={'id','bbox'}:raise ValueError('detection schema fields')
   i=token(d['id']);b=d['bbox']
   if i in ids:raise ValueError('duplicate id in frame')
   ids.add(i)
   if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or abs(x)>1e9 for x in b) or b[0]>=b[2] or b[1]>=b[3]:raise ValueError('invalid bbox')
   normalized.append({'id':i,'bbox':b})
  canonical={'t':str(t),'source':source,'model':model,'rgb':rgb,'detections':sorted(normalized,key=lambda x:x['id'])}
  fp=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
  state=self.keys[k]
  if n<state['now']:raise ValueError('clock rewind within view')
  if (state['watermark'] is None or t>state['watermark']) and any(k+(d['id'],) in self.obs and self.obs[k+(d['id'],)]['_t']>=n-self.ttl for d in normalized):raise ValueError('live observation id conflict')
  self.advance(n,k)
  if state['watermark'] is not None and t<=state['watermark']:
   status=('duplicate' if fp==state['fingerprint'] else 'replay_conflict') if t==state['watermark'] else 'out_of_order'
   self.event(status);return status
  state.update(watermark=t,fingerprint=fp)
  if t<n-self.ttl:
   for d in normalized:self.event('rejected_expired_observation','observation',k+(d['id'],))
   return 'expired_frame'
  for d in normalized:
   oid=k+(d['id'],);self.seq+=1
   self.obs[oid]={'camera':k[0],'viewRevision':k[1],'id':d['id'],'bbox':deepcopy(d['bbox']),'pts':f['pts'],'time_base':f['time_base'],'time':str(t),'source_sha256':source,'model':deepcopy(model),'rgb_pointer':rgb,'_t':t,'_seq':self.seq}
   self.event('stored_observation','observation',oid)
   self.bounds(k)
  self.event('stored_frame');return 'stored'
 def add_relation(self,camera,viewRevision,previous_id,current_id,status,source_sha256,now,*,endpoints):
  k=self.key(camera,viewRevision);a=k+(token(previous_id),);b=k+(token(current_id),);sha(source_sha256)
  if status not in ('unknown','accepted_hypothesis'):raise ValueError('relation status')
  n=clock(now)
  if n<self.keys[k]['now']:raise ValueError('clock rewind within view')
  if a not in self.obs or b not in self.obs or any(self.obs[i]['_t']<n-self.ttl for i in (a,b)):raise ValueError('relation missing live endpoint')
  if not isinstance(endpoints,list) or len(endpoints)!=2:raise ValueError('two exact event references required')
  for oid,ref in zip((a,b),endpoints):
   if not isinstance(ref,dict) or set(ref)!={'time','source_sha256'}:raise ValueError('event reference fields')
   if clock(ref['time'])!=self.obs[oid]['_t'] or sha(ref['source_sha256'])!=self.obs[oid]['source_sha256']:raise ValueError('stale or mismatched event reference')
  if self.obs[a]['_t']>=self.obs[b]['_t']:raise ValueError('noncausal relation')
  rid=k+(previous_id,current_id)
  if rid in self.rels:
   r=self.rels[rid]
   outcome='duplicate_relation' if r['status']==status and r['source_sha256']==source_sha256 else 'relation_conflict'
   if outcome=='duplicate_relation':self.advance(n,k)
   self.event(outcome);return outcome
  # A single accepted predecessor/successor avoids branching identity assumptions.
  if status=='accepted_hypothesis' and any(r['status']==status and (r['a']==a or r['b']==b) for r in self.rels.values()):raise ValueError('ambiguous accepted branch')
  self.advance(n,k);self.seq+=1
  self.rels[rid]={'a':a,'b':b,'status':status,'source_sha256':source_sha256,'_t':self.obs[b]['_t'],'_seq':self.seq,'endpoints':deepcopy(endpoints)};self.event('stored_relation','relation',rid);self.bounds(k);return 'stored'
 @staticmethod
 def public(o):return deepcopy({k:v for k,v in o.items() if not k.startswith('_')})
 def retrieve(self,request,now):
  if request is None:return None
  if not isinstance(request,dict) or set(request)!={'camera','viewRevision','pts','time_base','target_id','ambiguity_reason'}:raise ValueError('request schema fields')
  k=self.key(request['camera'],request['viewRevision']);target=k+(token(request['target_id']),);reason=token(request['ambiguity_reason']);t=time_of(request['pts'],request['time_base']);n=clock(now)
  if t>n:raise ValueError('future query')
  if target not in self.obs or self.obs[target]['_t']!=t or t<n-self.ttl:raise ValueError('current target/time missing or expired')
  self.advance(n,k)
  self.stats['retrievals']+=1
  live={i:o for i,o in self.obs.items() if i[:2]==k and max(n-self.ttl,t-self.ttl)<=o['_t']<=t}
  if target not in live:raise ValueError('current target outside window')
  chain=[];cursor=target
  while True:
   predecessors=[r['a'] for r in self.rels.values() if r['status']=='accepted_hypothesis' and r['b']==cursor and r['a'] in live and r['b'] in live]
   if not predecessors:break
   cursor=predecessors[0];chain.append(self.public(live[cursor]))
  unknown=[]
  for r in self.rels.values():
   if r['status']=='unknown' and r['b']==target and r['a'] in live:
    unknown.append({'previous':self.public(live[r['a']]),'current_id':target[-1],'status':'unknown','source_sha256':r['source_sha256'],'identity_verified':False})
  return {'current':self.public(live[target]),'accepted_history':list(reversed(chain)),'accepted_history_semantics':'upstream automatic association hypothesis, not identity GT','unknown_candidates':unknown,'ambiguity_reason':reason,'identity_verified':False,'entry':'unknown','safety':'unknown','footcontact':'unknown','hidden_crossing':'unknown','predicate_A':'unknown','predicate_B':'unknown','model_calls':0}
