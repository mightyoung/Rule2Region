"""Centre-vs-rectangle overlap only; no model, tracking or event classifier."""
import argparse
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'continuous-cache-20261010/cache'))
from continuous_cache import anchors, portal_distances
from vendor import roi_core


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['attempt','geometry-audit','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol_path=Path(__file__).with_name('PROTOCOL.json')
    protocol=json.loads(protocol_path.read_text())
    before={p.name:digest(p) for p in args.attempt.iterdir() if p.is_file()}
    variants=json.loads((args.geometry_audit/'ROI_VARIANTS_PRIVATE.json').read_text())['variants']
    prior=json.loads((args.geometry_audit/'ADAPTED_MODEL_GEOMETRY_PRIVATE.json').read_text())
    receipts=json.loads((args.attempt/'ALL_FRAME_RECEIPTS.json').read_text())
    assert [r['frame_index'] for r in receipts]==[0,142,262,312]
    dump(args.output/'FROZEN_PROTOCOL_COPY.json',dict(protocol,protocol_sha256=digest(protocol_path)))
    observed=[]
    for frame in receipts:
        assert frame['status']=='completed'
        for box in frame['model_boxes_after_nms']:
            verified=[r for r in prior if r['frame_index']==frame['frame_index'] and r['model_query_index_frame_local']==box['query_index']]
            assert len(verified)==1
            assert verified[0]['unchanged_model_box']==box['bbox_xyxy_original_unclipped']
            assert verified[0]['model_score']==box['score']
            assert verified[0]['source_sha256']==frame['source_sha256']
            b=box['bbox_xyxy_original_unclipped'];centre=anchors(b)['center'];area=(b[2]-b[0])*(b[3]-b[1])
            item={'frame_index':frame['frame_index'],'model_query_index_frame_local':box['query_index'],
                  'unchanged_model_box':b,'score':box['score'],'source_sha256':frame['source_sha256'],
                  'raw_output_sha256':digest(args.attempt/f"{frame['frame_index']:08d}-raw.safetensors"),
                  'centre':centre,'identity_or_track_verified':False,
                  'physical_event':'unknown_unassessed','regions':{}}
            for family,candidates in variants.items():
                assert [c['horizontal_inset_px'] for c in candidates]==protocol['horizontal_inset_px']
                outcomes=[]
                for candidate in candidates:
                    ring=candidate['points'];v=roi_core.polygon_validation(ring,800,600)
                    if not v['valid']:
                        outcomes.append({'inset_px':candidate['horizontal_inset_px'],'centre':None,'overlap_class':'unknown','overlap_fraction':None,'reason':v['reason']});continue
                    center_result=portal_distances(centre,{'reference_id':'illustrative_projection','points':ring})
                    overlap=roi_core.route_box(b,ring,800,600)
                    if overlap['decision']=='review_unknown':kind='unknown'
                    elif overlap['intersection_area_px2']>0:kind='positive_partial' if overlap['boundary_crossing'] else 'positive_full'
                    elif overlap['boundary_touch']:kind='zero_area_boundary_touch'
                    else:kind='outside'
                    outcomes.append({'inset_px':candidate['horizontal_inset_px'],
                                     'centre_inside':center_result['inside_closed_polygon'],
                                     'centre': 'inside' if center_result['inside_closed_polygon'] else 'outside',
                                     'overlap_class':kind,'exact_geometry_result':overlap,
                                     'overlap_fraction':None if kind=='unknown' else overlap['intersection_area_px2']/area})
                centers={r['centre'] for r in outcomes};classes={r['overlap_class'] for r in outcomes}
                def status(values):return 'unknown' if None in values or 'unknown' in values else ('stable' if len(values)==1 else 'sensitive')
                item['regions'][family]={'outcomes':outcomes,'centre_stability':status(centers),
                    'overlap_class_stability':status(classes),
                    'positive_overlap_stability':status({None if r['overlap_class']=='unknown' else r['overlap_class'].startswith('positive_') for r in outcomes}),
                    'finite_variants_only':True,'ROI_semantics_and_physical_events':'unknown'}
            observed.append(item)
    dump(args.output/'CENTRE_VS_OVERLAP_PRIVATE.json',observed)
    summary={'baseline':protocol['baseline'],'protocol_sha256':digest(protocol_path),
        'target_selection':protocol['target_choice'],'target_is_production_truth':False,
        'source_portal_label_supported':True,'physical_pixel_boundary_verified':False,
        'input_frames':4,'input_model_boxes':len(observed),'new_model_calls':0,'modified_model_boxes':0,
        'evaluations_per_predicate':len(observed)*2*5,'groups':{},
        'accuracy_recall_or_error_rate':None,'continuous_trajectory_constructed':False,
        'event_classifier_executed':False,'DS_implemented':False,'Library_retry':False,
        'future_evidence_interfaces':'source span/completeness, association, body/foot pose, optional physical threshold; declarations only',
        'semantic_and_physical_unknown_pairs':len(observed)*2}
    for family in variants:
        group={}
        for field in ['centre_stability','overlap_class_stability','positive_overlap_stability']:
            group[field]={state:sum(r['regions'][family][field]==state for r in observed) for state in ['stable','sensitive','unknown']}
        nominal=[next(o for o in r['regions'][family]['outcomes'] if o['inset_px']==0) for r in observed]
        group['nominal_center_inside']=sum(o['centre']=='inside' for o in nominal)
        group['nominal_overlap_class']={kind:sum(o['overlap_class']==kind for o in nominal) for kind in ['positive_full','positive_partial','zero_area_boundary_touch','outside','unknown']}
        group['nominal_centre_outside_but_positive_box_overlap']=sum(o['centre']=='outside' and o['overlap_class'].startswith('positive_') for o in nominal)
        group['nominal_centre_inside_but_box_not_fully_inside']=sum(o['centre']=='inside' and o['overlap_class']=='positive_partial' for o in nominal)
        values=[o['overlap_fraction'] for r in observed for o in r['regions'][family]['outcomes'] if o['overlap_fraction'] is not None]
        group['finite_sweep_descriptive_overlap_fraction_range']=[min(values),max(values)]
        summary['groups'][family]=group
    assert before=={p.name:digest(p) for p in args.attempt.iterdir() if p.is_file()}
    summary['original_model_output_files_unchanged']=True
    dump(args.output/'PUBLIC_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
