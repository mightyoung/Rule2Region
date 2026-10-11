"""Read immutable real detections; private AI comparison and ROI sensitivity.

No inference, fusion, trajectory construction or human-GT evaluation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'continuous-cache-20261010/cache'))
from continuous_cache import anchors, portal_distances
from vendor import roi_core

DELTAS = (-16, -8, 0, 8, 16)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def variants(points):
    # Both frozen rings have left/right jamb vertices 0,3 and 1,2.
    # This fixed +/-1% and +/-2% image-width sweep is NOT measured calibration.
    return [{'horizontal_inset_px':delta,
             'points':[[x+(delta if j in (0,3) else -delta),y]
                       for j,(x,y) in enumerate(points)]}
            for delta in DELTAS]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--illustrations',type=Path,required=True)
    p.add_argument('--originals',type=Path,required=True)
    p.add_argument('--axis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    import torch
    from safetensors.torch import load_file
    before={p.name:digest(p) for p in args.attempt.iterdir() if p.is_file()}
    originals={r['frame_index']:r for r in json.loads(args.illustrations.read_text())}
    rows=json.loads((args.attempt/'ALL_FRAME_RECEIPTS.json').read_text())
    axis={r['frame_index'] for r in json.loads(args.axis.read_text())['members']}
    # Freeze variant policy/provenance before checking any target membership.
    proposal=originals[0]
    roi_families={key:variants(proposal[key]) for key in
                  ('candidate_far_door_polygon','candidate_near_frame_polygon')}
    for candidates in roi_families.values():
        for candidate in candidates:
            assert roi_core.polygon_validation(candidate['points'],800,600)['valid']
    dump(args.output/'ROI_VARIANTS_PRIVATE.json',{
        'policy':'horizontal jamb inset/outset +/-8,+/-16 pixels, y fixed; 1%/2% image width, not measured geometric error',
        'source':'earlier AI image illustration; not detector ROI or human boundary truth',
        'physical_sill':None,'human_reasonableness_review':'pending',
        'variants':roi_families})
    records=[];comparisons=[];review=[]
    for row in rows:
        idx=row['frame_index'];assert row['status']=='completed'
        prior=originals[idx];path=args.originals/row['original_filename']
        assert digest(path)==row['source_sha256']==prior['original_sha256']
        with Image.open(path) as image: assert image.size==(800,600)
        raw=load_file(str(args.attempt/f'{idx:08d}-raw.safetensors'))
        logits=raw['logits'][0];pred=raw['pred_boxes'][0]
        cx,cy,w,h=pred.unbind(-1)
        corners=torch.stack((cx-w/2,cy-h/2,cx+w/2,cy+h/2),dim=-1)*torch.tensor([800,600,800,600])
        scores=logits.sigmoid().max(dim=-1).values
        model_boxes=row['model_boxes_after_nms'];ai_boxes=prior['candidate_person_boxes']
        for box in row['model_boxes_before_nms']:
            q=box['query_index'];assert float(scores[q])==box['score']
            torch.testing.assert_close(corners[q],torch.tensor(box['bbox_xyxy_original_unclipped']),rtol=0,atol=0)
        # Frame-local maximum-IoU correspondence is illustration alignment only.
        # Match cutoff 0.5 is fixed; this is neither GT assignment nor tracking.
        n=min(len(ai_boxes),len(model_boxes));best=[];best_total=-1
        for ai_ids in itertools.combinations(range(len(ai_boxes)),n):
            for model_ids in itertools.permutations(range(len(model_boxes)),n):
                pairs=list(zip(ai_ids,model_ids));total=sum(iou(ai_boxes[a],model_boxes[b]['bbox_xyxy_original_unclipped']) for a,b in pairs)
                if total>best_total:best_total=total;best=pairs
        matched=[]
        for a,b in best:
            original=model_boxes[b]['bbox_xyxy_original_unclipped'];overlap=iou(ai_boxes[a],original)
            if overlap>=.5:
                matched.append({'AI_box_index':a,'model_query_index':model_boxes[b]['query_index'],
                    'AI_box':ai_boxes[a],'unchanged_model_box':original,'model_score':model_boxes[b]['score'],
                    'IoU_illustration_alignment_not_accuracy':overlap,
                    'model_minus_AI_edge_delta_px':[v-u for u,v in zip(ai_boxes[a],original)]})
        comparisons.append({'frame_index':idx,'AI_provenance':'AI illustrative discussion, not detector or human GT',
            'matched':matched,'unmatched_AI_illustrations':[a for a in range(len(ai_boxes)) if a not in {m['AI_box_index'] for m in matched}],
            'unmatched_model_queries':[b['query_index'] for b in model_boxes if b['query_index'] not in {m['model_query_index'] for m in matched}],
            'miss_or_false_positive_count':None})
        for box in model_boxes:
            item={'frame_index':idx,'source_sha256':row['source_sha256'],
                  'model_query_index_frame_local':box['query_index'],'model_score':box['score'],
                  'unchanged_model_box':box['bbox_xyxy_original_unclipped'],
                  'anchors':anchors(box['bbox_xyxy_original_unclipped']),
                  'identity_verified':False,'source_native_pts':None,
                  'trajectory_direction':'unknown_unassessed','event_decision':'unknown_unassessed',
                  'ROI_results':{}}
            for key,candidates in roi_families.items():
                outcomes=[]
                for candidate in candidates:
                    result=portal_distances(item['anchors']['center'],{'reference_id':'AI_projection','points':candidate['points']})
                    outcomes.append(dict(result,horizontal_inset_px=candidate['horizontal_inset_px']))
                membership={o['inside_closed_polygon'] for o in outcomes}
                item['ROI_results'][key]={'outcomes':outcomes,
                    'finite_sweep_membership':'sensitive_unknown' if len(membership)>1 else ('stable_inside' if True in membership else 'stable_outside'),
                    'semantic_and_physical_event_truth':'unknown'}
            records.append(item)
        overlaps=[{'model_query_a':a['query_index'],'model_query_b':b['query_index'],
                   'positive_bbox_overlap':iou(a['bbox_xyxy_original_unclipped'],b['bbox_xyxy_original_unclipped'])>0,
                   'bbox_iou':iou(a['bbox_xyxy_original_unclipped'],b['bbox_xyxy_original_unclipped'])}
                  for a,b in itertools.combinations(model_boxes,2)]
        review.append({'frame_index':idx,'human_GT':'missing','frame_local_raw_agreement':True,
                      'near_image_bottom_queries':[b['query_index'] for b in model_boxes if b['bbox_xyxy_original_unclipped'][3]>=598],
                      'near_bottom_means':'proxy image-boundary proximity, not verified feet/threshold contact',
                      'overlapping_model_boxes':overlaps})
    gaps=[]
    indices=sorted(originals)
    for a,b in zip(indices,indices[1:]):
        absent=sum(i not in axis for i in range(a+1,b))
        gaps.append({'from_index':a,'to_index':b,'source_index_distance':b-a,
                     'absent_archive_members':absent,'unselected_existing_intermediate_members':b-a-1-absent,
                     'continuous_trajectory_eligible':False,'native_pts_recovered':False,
                     'identity_association_supplied':False})
    dump(args.output/'ADAPTED_MODEL_GEOMETRY_PRIVATE.json',records)
    dump(args.output/'MODEL_VS_AI_DIFFERENCES_PRIVATE.json',comparisons)
    dump(args.output/'REVIEW_AND_FRAME_GAPS_PRIVATE.json',{'frames':review,'gaps':gaps})
    summary={'source':'same four admitted originals and immutable DINO attempt01; no inference rerun',
             'input_frames':len(rows),'model_boxes':len(records),'verified_model_query_scores_and_boxes':sum(len(r['model_boxes_before_nms']) for r in rows),
             'model_boxes_modified':0,'model_calls':0,'confidence_threshold_changed':False,
             'AI_comparison_pairs':sum(len(c['matched']) for c in comparisons),
             'unmatched_AI_illustrations':sum(len(c['unmatched_AI_illustrations']) for c in comparisons),
             'unmatched_model_queries':sum(len(c['unmatched_model_queries']) for c in comparisons),
             'illustration_match_is_accuracy_or_identity':False,'accuracy_and_miss_rate':None,
             'ROI_variants_per_family':len(DELTAS),'horizontal_deltas_px':list(DELTAS),
             'ROI_membership_evaluations':len(records)*len(DELTAS)*2,
             'ROI_sources':'separate AI illustration and uncalibrated fixed sensitivity envelope; human review pending',
             'ROI_families':{key:{label:sum(r['ROI_results'][key]['finite_sweep_membership']==label for r in records)
                            for label in ('stable_inside','stable_outside','sensitive_unknown')} for key in roi_families},
             'truncated_or_image_bottom_proximity_queries':sum(len(r['near_image_bottom_queries']) for r in review),
             'frames_with_positive_pair_bbox_overlap':sum(any(p['positive_bbox_overlap'] for p in r['overlapping_model_boxes']) for r in review),
             'frame_gaps':gaps,'continuous_trajectory_eligible_count':0,
             'event_classifier_executed':False,'physical_entry_or_exit':'unknown_unassessed',
             'evidence_correlation':'person boxes/scores/centers and AI ROI originate from same RGB camera; derived features not independent sources',
             'DS_implemented':False,'Library_retry':False,'public_media':False}
    assert before=={p.name:digest(p) for p in args.attempt.iterdir() if p.is_file()}
    summary['original_model_output_files_unchanged']=True
    dump(args.output/'PUBLIC_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
