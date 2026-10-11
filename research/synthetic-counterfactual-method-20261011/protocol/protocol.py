"""Counterfactual development-only protocol. Standard library; no media/model I/O."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

VERSION='r2r-counterfactual-empty-dev-v2'
PROFILES=('I0','I1-R','I1-E','I1-E-no-RGB')
TASK='Assess the specified doorway relation for the selected counting boundary; abstain if required evidence or boundary configuration is unavailable.'
UNKNOWN='unknown'

def require(ok,message):
    if not ok: raise ValueError(message)

def keys(obj,expected,where):
    require(isinstance(obj,dict) and set(obj)==set(expected),f'{where}: unexpected/missing fields')

def nonempty(x): return isinstance(x,str) and bool(x.strip())

def available(r):
    base=r['input'];c=r['context'];rgb=nonempty(base['rgb']['reference']) and nonempty(base['rgb']['decoded_rgb_sha256'])
    er=c['I1-E'];e=er is not None and nonempty(er.get('B_version')) and all(x in ('inside','outside') for x in er.get('endpoint_B_membership',[])) and len(er.get('endpoint_B_membership',[]))==2
    return {'I0':{'rgb':rgb,'generic_task':base['generic_task']==TASK,'neutral_door_roi':base['neutral_door_roi'] is not None},
            'I1-R':{k:c['I1-R'][k] is not None for k in ('map','B_config','camera_registration')},
            'I1-E':{'configuration':er is not None,'endpoint_membership_known':e},
            'I1-E-no-RGB':{'configuration':er is not None,'endpoint_membership_known':e,'rgb':False}}

def geometry_empty():
    return {'S_version':None,'endpoint_S_membership':[UNKNOWN,UNKNOWN],'physical_adjacency':None,'enclosure_features':None,'review_status':'pending'}

def policy_empty():
    return {'B_version':None,'endpoint_B_membership':[UNKNOWN,UNKNOWN],'count_boundary_crossing':UNKNOWN,'review_status':'pending'}

def definitions():
    return {'S':{'meaning':'physical building-envelope volume; an open door does not erase its enclosed interior volume','business_confirmed':False},
            'B':{'meaning':'user-selected union of regions for population accounting; not inferred from S','business_confirmed':False,'selected_region_union':None,'auto_include_yards_or_walkways':False}}

def make_template():
    m={'schema_version':VERSION,'stage':'empty_development_templates','definitions':definitions(),
       'independent_sample_count':None,'training_admitted_ids':[],'effect_rows':[],
       'planned_record_count':16,'planned_unique_rgb_upper_bound':14,
       'pair_contracts':{'A':'narrow_roof_shortcut_test_not_forced_observability','B':'narrow_sky_shortcut_test_no_forced_S_opposition','C':'invariance_only_if_key_evidence_equally_sufficient','D':'policy_response_S_unchanged_same_RGB_I0_nonidentifiable'},
       'source_registry':{f'planned_{f}':{'origin':'planned','parents':[],'v1_source_group':None} for f in 'ABCD'},'records':[]}
    for f,c,side in itertools.product('ABCD',(0,1),(0,1)):
        sid=hashlib.sha256(f'counterfactual-empty-v2|{f}|{c}|{side}'.encode()).hexdigest()[:20]
        r={'sample_id':sid,'pair':{'family':f,'condition':c,'side':side},
           'source_links':{'layout_lineage_id':f'planned_{f}','topology_family_id':f'planned_{f}','physical_door_lineage_id':f'planned_{f}_door','rgb_reuse_id':f'planned_{f}_{0 if f=="D" else c}_{side}'},
           'component_group':None,'split':None,'usage':'development_only',
           'input':{'rgb':{'reference':None,'decoded_rgb_sha256':None},'generic_task':TASK,'neutral_door_roi':None},
           'context':{'I1-R':{'map':None,'B_config':None,'camera_registration':None},'I1-E':None},
           'source_geometry_truth':geometry_empty(),'source_policy_truth':policy_empty(),
           'image_evidence':{'status':'unreviewed','reviewer':None,'key_evidence_sufficiency':None,'observations':[]},
           'input_available':{},'decision':{p:None for p in PROFILES},
           'temporal_evidence':{'trajectory':None,'direction':None,'status':'unavailable_static_plan'}}
        r['input_available']=available(r);m['records'].append(r)
    groups=source_components(m)
    for r in m['records']:r['component_group']=groups[r['sample_id']]
    return m

def source_components(m):
    """Global ancestry/reuse union before selecting any records; not an authenticity check."""
    registry=m['source_registry'];require(isinstance(registry,dict) and bool(registry),'source registry required')
    visiting=set();done=set()
    def check(node):
        require(node in registry,'undeclared layout ancestor')
        if node in done:return
        require(node not in visiting,'source ancestry cycle');visiting.add(node)
        x=registry[node];keys(x,('origin','parents','v1_source_group'),'source registry node')
        require(x['origin'] in ('planned','v1'),'undeclared source origin')
        require(isinstance(x['parents'],list) and all(nonempty(p) for p in x['parents']),'parents must be explicit IDs')
        if x['origin']=='v1':require(nonempty(x['v1_source_group']),'v1 borrowing requires original source group')
        else:require(x['v1_source_group'] is None,'planned origin cannot hide a v1 group')
        for p in x['parents']:check(p)
        visiting.remove(node);done.add(node)
    for node in registry:check(node)
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):parent[find(a)]=find(b)
    for node,x in registry.items():
        find(('layout',node))
        for ancestor in x['parents']:union(('layout',node),('layout',ancestor))
        if x['origin']=='v1':union(('layout',node),('v1',x['v1_source_group']))
    ids=[]
    for r in m['records']:
        sid=r['sample_id'];require(nonempty(sid) and sid not in ids,'duplicate/missing sample ID');ids.append(sid)
        token=('sample',sid);find(token);link=r['source_links'];node=link['layout_lineage_id'];require(node in registry,'undeclared layout lineage')
        union(token,('layout',node))
        for field in ('topology_family_id','physical_door_lineage_id','rgb_reuse_id'):
            require(nonempty(link[field]),'missing source reuse/lineage anchor');union(token,(field,link[field]))
        rgb_hash=r['input']['rgb']['decoded_rgb_sha256']
        if rgb_hash is not None:union(token,('decoded_rgb',rgb_hash))
    members={};origins={}
    for sid in ids:members.setdefault(find(('sample',sid)),[]).append(sid)
    for node,x in registry.items():
        if x['origin']=='v1':origins.setdefault(find(('layout',node)),set()).add(x['v1_source_group'])
    result={}
    for root,ss in members.items():
        v1=sorted(origins.get(root,set()))
        # Retain the original v1 group identity when one origin is present.
        gid=v1[0] if len(v1)==1 else 'component-'+hashlib.sha256(json.dumps({'samples':sorted(ss),'v1_origins':v1},sort_keys=True).encode()).hexdigest()
        for sid in ss:result[sid]=gid
    return result

def metadata(x,where):
    if x is None:return
    keys(x,('reference','version','provenance','review_status','contains_target_endpoint_membership'),where)
    require(nonempty(x['reference']) and nonempty(x['version']),where+': unresolved reference/version')
    require(x['provenance']=='runtime_available' and x['review_status']=='confirmed',where+': operational availability must be reviewed')
    require(x['contains_target_endpoint_membership'] is False,where+': target endpoint answer belongs to I1-E, not I1-R')

def validate_manifest(m):
    expected=make_template_root_keys()
    keys(m,expected,'manifest');require(m['schema_version']==VERSION and m['stage']=='empty_development_templates','wrong stage/version')
    require(m['definitions']==definitions(),'S/B definitions pending business confirmation; no inferred policy')
    require(m['independent_sample_count'] is None,'16 records are not 16 independent samples')
    require(m['training_admitted_ids']==[] and m['effect_rows']==[],'unreviewed development records cannot enter training/effect tables')
    require(m['planned_record_count']==16 and m['planned_unique_rgb_upper_bound']==14,'plan counts are not measured results')
    require(m['pair_contracts']=={'A':'narrow_roof_shortcut_test_not_forced_observability','B':'narrow_sky_shortcut_test_no_forced_S_opposition','C':'invariance_only_if_key_evidence_equally_sufficient','D':'policy_response_S_unchanged_same_RGB_I0_nonidentifiable'},'pair conditions must not force labels')
    require(isinstance(m['records'],list) and len(m['records'])==16,'requires full 16-record index')
    coverage=[]
    record_keys=('sample_id','pair','source_links','component_group','split','usage','input','context','source_geometry_truth','source_policy_truth','image_evidence','input_available','decision','temporal_evidence')
    for r in m['records']:
        keys(r,record_keys,'record');keys(r['pair'],('family','condition','side'),'pair')
        pair=r['pair'];require(pair['family'] in 'ABCD' and len(pair['family'])==1,'invalid family')
        require(type(pair['condition']) is int and pair['condition'] in (0,1) and type(pair['side']) is int and pair['side'] in (0,1),'invalid condition/side')
        coverage.append((pair['family'],pair['condition'],pair['side']))
        keys(r['source_links'],('layout_lineage_id','topology_family_id','physical_door_lineage_id','rgb_reuse_id'),'source links')
        require(r['split'] is None and r['usage']=='development_only','only unsplit development records admitted')
        keys(r['input'],('rgb','generic_task','neutral_door_roi'),'I0 input');keys(r['input']['rgb'],('reference','decoded_rgb_sha256'),'RGB reference')
        rgb=r['input']['rgb'];require((rgb['reference'] is None)==(rgb['decoded_rgb_sha256'] is None),'RGB reference/hash must be paired')
        if rgb['reference'] is not None:
            require(nonempty(rgb['reference']) and isinstance(rgb['decoded_rgb_sha256'],str) and len(rgb['decoded_rgb_sha256'])==64 and all(c in '0123456789abcdef' for c in rgb['decoded_rgb_sha256']),'invalid RGB reference/hash')
        require(r['input']['generic_task']==TASK,'generic task cannot include target answer')
        roi=r['input']['neutral_door_roi']
        if roi is not None:require(isinstance(roi,list) and len(roi)==4 and all(type(x) in (int,float) and 0<=x<=1 for x in roi) and roi[0]<roi[2] and roi[1]<roi[3],'neutral ROI must be normalized rectangle only')
        keys(r['context'],('I1-R','I1-E'),'contexts');keys(r['context']['I1-R'],('map','B_config','camera_registration'),'I1-R')
        for k,v in r['context']['I1-R'].items():metadata(v,'I1-R '+k)
        e=r['context']['I1-E']
        if e is not None:
            keys(e,('B_version','endpoint_B_membership'),'I1-E configuration')
            require(nonempty(e['B_version']) and isinstance(e['endpoint_B_membership'],list) and len(e['endpoint_B_membership'])==2 and all(x in ('inside','outside',UNKNOWN) for x in e['endpoint_B_membership']),'invalid execution configuration')
        require(r['source_geometry_truth']==geometry_empty(),'do not fabricate source geometry labels in empty templates')
        require(r['source_policy_truth']==policy_empty(),'do not fabricate source policy labels in empty templates')
        require(r['image_evidence']=={'status':'unreviewed','reviewer':None,'key_evidence_sufficiency':None,'observations':[]},'unseen image evidence must remain unreviewed')
        require(r['decision']=={p:None for p in PROFILES},'no decisions in unreviewed templates')
        require(r['input_available']==available(r),'availability must reflect declared inputs, not source truth')
        require(r['temporal_evidence']=={'trajectory':None,'direction':None,'status':'unavailable_static_plan'},'no invented static trajectories/direction')
    require(set(coverage)==set(itertools.product('ABCD',(0,1),(0,1))) and len(set(coverage))==16,'coverage duplicate/missing condition')
    groups=source_components(m)
    for r in m['records']:require(r['component_group']==groups[r['sample_id']],'forged or incomplete component grouping')
    # All variants/sides in a pair family must inherit a single source component.
    for f in 'ABCD':require(len({groups[r['sample_id']] for r in m['records'] if r['pair']['family']==f})==1,'pair family split across source components')
    for side in (0,1):
        rs=[r for r in m['records'] if r['pair']['family']=='D' and r['pair']['side']==side]
        require(rs[0]['source_links']['rgb_reuse_id']==rs[1]['source_links']['rgb_reuse_id'],'D pair must declare RGB reuse')
        require(export_inputs(rs[0],'I0')==export_inputs(rs[1],'I0'),'D same RGB must expose identical I0 inputs')
    return {'status':'pass','records':16,'declared_source_components':len(set(groups.values())),'independent_sample_count':None,'reviewed_images':0,'decisions':0,'training_admitted':0,'effect_rows':0,'declared_unique_rgb_hash_count':len({r['input']['rgb']['decoded_rgb_sha256'] for r in m['records'] if r['input']['rgb']['decoded_rgb_sha256'] is not None}),'classification':'software protocol checks only, not visual performance'}

def make_template_root_keys():
    return ('schema_version','stage','definitions','independent_sample_count','training_admitted_ids','effect_rows','planned_record_count','planned_unique_rgb_upper_bound','pair_contracts','source_registry','records')

def export_inputs(r,profile):
    require(profile in PROFILES,'unknown input profile')
    keys(r['input'],('rgb','generic_task','neutral_door_roi'),'export input')
    base={k:copy.deepcopy(r['input'][k]) for k in ('rgb','generic_task','neutral_door_roi')}
    if profile=='I0':return base
    if profile=='I1-R':
        keys(r['context']['I1-R'],('map','B_config','camera_registration'),'export I1-R')
        for k,v in r['context']['I1-R'].items():metadata(v,'export I1-R '+k)
        base['operational_context']=copy.deepcopy(r['context']['I1-R']);return base
    out={'generic_task':r['input']['generic_task'],'configuration':copy.deepcopy(r['context']['I1-E'])}
    if profile=='I1-E':out.update(rgb=copy.deepcopy(r['input']['rgb']),neutral_door_roi=copy.deepcopy(r['input']['neutral_door_roi']))
    return out

def export_manifest(m):
    """Validate full global index before producing input-only profile skeletons."""
    validate_manifest(m)
    return {'schema_version':VERSION+'-input-profiles','records':[{'sample_id':r['sample_id'],'profiles':{p:export_inputs(r,p) for p in PROFILES}} for r in m['records']]}

def execute_configuration(configuration):
    """No RGB argument: deterministic configured XOR, never a visual discovery."""
    if configuration is None:return UNKNOWN
    keys(configuration,('B_version','endpoint_B_membership'),'execution config')
    values=configuration['endpoint_B_membership']
    require(isinstance(values,list) and len(values)==2 and all(x in ('inside','outside',UNKNOWN) for x in values),'invalid endpoint membership')
    if not nonempty(configuration['B_version']) or UNKNOWN in values:return UNKNOWN
    return (values[0]=='inside') != (values[1]=='inside')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--write-template',type=Path);g.add_argument('--validate',type=Path);args=p.parse_args()
    if args.write_template:
        m=make_template();validate_manifest(m);args.write_template.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');print('Wrote 16 empty development records; no images/labels/inference produced.')
    else:print(json.dumps(validate_manifest(json.loads(args.validate.read_text())),indent=2))
