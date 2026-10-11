"""Continuous recording and conditional reads of nine public actual bbox frames."""
from pathlib import Path
from fractions import Fraction
from copy import deepcopy
from itertools import product
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent/'continuous-cache-20261010'
sys.path.insert(0,str(PUBLIC/'cache'))
from continuous_cache import ContinuousCache
from ablation import ARMS,arm_view


def calculate(do_queries=True):
    def read(path):return json.loads(path.read_text())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    inventory=read(PUBLIC/'detection/nine_frame_bbox_inventory.json')['frames']
    old_frames=read(PUBLIC/'cache/bbox_fixture.json')
    old={d['id']:(f,d) for f in old_frames for d in f['detections']}
    bbox_hash=digest(PUBLIC/'cache/bbox_fixture.json')
    relation_hash=digest(PUBLIC/'cache/relation_fixture.json')
    inventory_hash=digest(PUBLIC/'detection/nine_frame_bbox_inventory.json')
    geometry_hash=digest(PUBLIC/'cache/geometry_fixture.json')
    geometry=read(PUBLIC/'cache/geometry_fixture.json')
    geometry['portal_source_sha256']=geometry_hash
    geometry['sill'].update(source_sha256=geometry_hash,source_frame_sha256=bbox_hash)
    frames=[]
    endpoints={}
    for f in inventory:
        sid=f['sample_id']
        source=bbox_hash if f['prediction_provenance']=='existing_frozen_prediction' else digest(PUBLIC/f'detection/grounding-dino-tiny__{sid}.json')
        for d in f['detections']:
            if f['prediction_provenance']=='existing_frozen_prediction':
                of,od=old[d['prediction_id']]
                assert (of['pts'],of['time_base'],od['bbox'])==(f['pts'],f['time_base'],d['bbox_xyxy'])
            endpoints[d['prediction_id']]={'time':str(Fraction(f['pts'])*Fraction(f['time_base'])),'source_sha256':source}
        frames.append({'camera':'G420','viewRevision':'public-geometry-v1','pts':f['pts'],
                       'time_base':f['time_base'],'source_sha256':source,
                       'model':{'name':'frozen_public_bbox_record','provenance_kind':'public_JSON_not_original_pixels'},
                       'rgb_pointer':None,'detections':[{'id':d['prediction_id'],'bbox':d['bbox_xyxy']} for d in f['detections']]})
    accepted=[tuple(p) for p in read(PUBLIC/'cache/relation_fixture.json')['accepted_hypothesis_pairs']]
    c=ContinuousCache()
    trace=[];reads=[];checkpoints=[];legacy=None
    previous=None
    for f in frames:
        now=Fraction(f['pts'])*Fraction(f['time_base']);k=(f['camera'],f['viewRevision'])
        c.register(*k,geometry);assert c.ingest(f,now)=='stored'
        if previous:
            for a,b in product(previous['detections'],f['detections']):
                c.add_relation(*k,a['id'],b['id'],'unknown',inventory_hash,now,
                               endpoints=[endpoints[a['id']],endpoints[b['id']]])
        for a,b in accepted:
            if b in {d['id'] for d in f['detections']} and a in endpoints:
                c.add_relation(*k,a,b,'accepted_hypothesis',relation_hash,now,endpoints=[endpoints[a],endpoints[b]])
                legacy=c.public(c.rels[k+(a,b)])
                # Public result uses JSON-native endpoint keys, not cache tuples.
                legacy['a']=list(legacy['a']);legacy['b']=list(legacy['b'])
        current={d['endpoint_bindings'][0]['observation_id']:d for d in c.descriptors.values()
                 if d['kind']=='geometry' and d['endpoint_bindings'][0]['pts']==f['pts']}
        for observation in f['detections']:
            oid=observation['id'];descriptor=current[oid]
            center={p['reference_id']:p['inside_closed_polygon'] for p in descriptor['anchors']['center']['portals']}
            flagged=any(hit['hit'] is True and not center[hit['reference_id']] for hit in descriptor['bbox_hits'])
            row={'observation_id':oid,'pts':f['pts'],'time_base':f['time_base'],
                 'geometry_computed':True,'review_flag':flagged,
                 'review_flag_semantics':'caller geometry-disagreement proxy, not human/event-ambiguity GT',
                 'trajectory_status':'unknown_no_qualified_motion_history',
                 'direction_status':'unknown','identity_verified':False,'side_status':'unknown','event_status':'unknown'}
            trace.append(row)
            before=deepcopy((c.obs,c.rels,c.descriptors,c.keys));stats_before=dict(c.stats)
            assert c.retrieve(None,now) is None and dict(c.stats)==stats_before
            if do_queries and flagged:
                request={'camera':k[0],'viewRevision':k[1],'pts':f['pts'],'time_base':f['time_base'],
                         'target_id':oid,'ambiguity_reason':'bbox_hit_but_center_outside_same_portal'}
                payload=c.retrieve(request,now)
                views={arm:arm_view(payload,arm) for arm in ARMS}
                reads.append({'observation_id':oid,'pts':f['pts'],'review_flag':flagged,
                              'identity_verified':False,'side_status':'unknown',
                              'arms':{arm:{key:v[key] for key in ['event_status','direction_segment_available',
                                      'direction_continuity_available','three_point_distance_available','missing_inputs']}
                                      for arm,v in views.items()}})
            assert before==(c.obs,c.rels,c.descriptors,c.keys)
        checkpoints.append({'pts':f['pts'],'live_observations':len(c.obs),'live_relations':len(c.rels),
                            'live_descriptors':len(c.descriptors),'geometry_computed_total':c.stats['computed_geometry'],
                            'motion_computed_total':c.stats['computed_motion'],
                            'capacity_evictions_total':c.stats['evicted_observation'],
                            'expired_observations_total':c.stats['expired_observation']})
        previous=f
    return {'scope':'public_actual_bbox_nine_frame_continuous_geometry_and_conditional_feature_retrieval',
            'summary':{'frames':len(frames),'observations':len(trace),'computed_geometry':c.stats['computed_geometry'],
                       'computed_motion':c.stats['computed_motion'],'retrievals':c.stats['retrievals'],
                       'geometry_disagreement_flags':sum(r['review_flag'] for r in trace),
                       'adjacent_unknown_candidate_pairs':sum(len(a['detections'])*len(b['detections']) for a,b in zip(frames,frames[1:])),
                       'legacy_accepted_hypotheses_ingested':1 if legacy else 0,
                       'observation_capacity_evictions':c.stats['evicted_observation'],
                       'observation_expirations':c.stats['expired_observation'],
                       'live_observations_at_end':len(c.obs),'no_read_mutation':True,'event_unknown':len(trace),
                       'new_model_calls':0,'judgment_engine_calls':0,'accuracy':None},
            'cache_contract':{'ttl_source_seconds':3,'per_view_observations':64,'per_view_relations':512,
                              'per_view_descriptors':128,'source_clock_simulation_not_wall_clock':True},
            'legacy_relation_at_ingest':legacy,'trace':trace,'reads':reads,'checkpoints':checkpoints,
            'limits':['No pixels/HSV/original provenance mapping recovered',
                      '701 adjacent pairs are unscored unknown candidates, not new associations',
                      'Old accepted endpoint hypothesis cannot bypass intervening unassociated frames',
                      'Four views project cached feature availability; no event judgment engine is instantiated',
                      'Review flag is an exposed-development geometry proxy, not verified business ambiguity',
                      'All event statuses unknown; unknown not safe; no recognition gain']}


if __name__=='__main__':
    absent=calculate(False);conditional=calculate(True)
    assert absent['trace']==conditional['trace'] and absent['checkpoints']==conditional['checkpoints']
    result={'without_requests':absent,'with_geometry_flagged_reads':conditional}
    path=HERE/'conditional_results.json'
    if sys.argv[1:]==['--generate']:path.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:raise SystemExit('Only optional --generate supported')
    elif result!=json.loads(path.read_text()):raise SystemExit('Conditional replay differs')
    print(json.dumps({k:v['summary'] for k,v in result.items()},indent=2))
