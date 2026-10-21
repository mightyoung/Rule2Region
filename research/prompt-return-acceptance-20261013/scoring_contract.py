"""Exact reference scorer for explicitly SYNTHETIC software fixtures only.

User-approved person ROI retention contract v1. This module does not establish
real-source admission, validate human truth, load a model, or change legacy gates.
"""
import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess

VERSION = 'person-roi-retention.v1'
NAMES = ('in_region_recall', 'in_region_recall_loss',
         'boundary_retention', 'outside_suppression')
ADMISSION = {'source_authorized', 'bundle_integrity', 'independent_full_frame_labels',
             'roi_reviewed', 'camera_registered', 'policy_bound',
             'observability_reviewed', 'stable_ids_frozen', 'preregistered_before_results'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fields(value, required, optional=()):
    require(isinstance(value, dict) and set(value) >= set(required)
            and set(value) <= set(required) | set(optional), 'schema_mismatch')


def identity(value):
    require(isinstance(value, str) and bool(value.strip())
            and value not in {'unknown', 'unreviewed', 'pending'}, 'identity_unknown')
    return value


def loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    def reject(_):
        raise ValueError('nonfinite_json')
    return json.loads(text, parse_float=Decimal, parse_constant=reject,
                      object_pairs_hook=unique)


def number(value):
    require(type(value) is int or isinstance(value, Decimal), 'exact_decimal_required')
    require(not isinstance(value, Decimal) or value.is_finite(), 'nonfinite_coordinate')
    return Fraction(value)


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    # Public helper also supports already normalized rational boxes.
    a, b = [Fraction(x) for x in a], [Fraction(x) for x in b]
    require(len(a) == len(b) == 4 and area(a) > 0 and area(b) > 0
            and a[2] > a[0] and a[3] > a[1]
            and b[2] > b[0] and b[3] > b[1], 'invalid_box')
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area(a) + area(b) - intersection)


def maximum_matching(edges):
    """Min-cost max-flow: cardinality, exact IoU sum, then sorted pair IDs.

    For fixed cardinality, maximizing the binary incidence vector in sorted
    edge order selects the lexicographically smallest sorted pair sequence.
    Costs are (negative rational IoU, negative binary priority); tuple ordering
    keeps this tie-break strictly subordinate to total IoU, without epsilon.
    Bellman-Ford supports reverse residual edges and exact tuple costs.
    """
    require(isinstance(edges, dict), 'invalid_edges')
    for pair, weight in edges.items():
        require(isinstance(pair, tuple) and len(pair) == 2, 'invalid_edge_identity')
        identity(pair[0]); identity(pair[1])
        require(isinstance(weight, Fraction) and Fraction(1,2) <= weight <= 1,
                'invalid_edge_weight')
    pairs = sorted(edges)
    gt = sorted({g for g, _ in pairs}); pred = sorted({p for _, p in pairs})
    gnodes = {g: i+1 for i,g in enumerate(gt)}
    pnodes = {p: i+1+len(gt) for i,p in enumerate(pred)}
    sink = 1 + len(gt) + len(pred); graph = [[] for _ in range(sink+1)]
    zero = (Fraction(0), 0)
    def add(a, b, cost):
        forward = [b, len(graph[b]), 1, cost]
        reverse = [a, len(graph[a]), 0, (-cost[0], -cost[1])]
        graph[a].append(forward); graph[b].append(reverse)
        return forward
    for g in gt: add(0, gnodes[g], zero)
    for p in pred: add(pnodes[p], sink, zero)
    selected = {}
    for i,pair in enumerate(pairs):
        selected[pair] = add(gnodes[pair[0]], pnodes[pair[1]],
                             (-edges[pair], -(1 << (len(pairs)-1-i))))
    while True:
        distance = [None] * len(graph); previous = [None] * len(graph)
        distance[0] = zero
        for _ in range(len(graph)-1):
            changed = False
            for node, neighbors in enumerate(graph):
                if distance[node] is None: continue
                for index, edge in enumerate(neighbors):
                    to, _, capacity, cost = edge
                    if not capacity: continue
                    candidate = (distance[node][0]+cost[0], distance[node][1]+cost[1])
                    if distance[to] is None or candidate < distance[to]:
                        distance[to] = candidate; previous[to] = (node,index); changed = True
            if not changed: break
        if distance[sink] is None: break
        node = sink
        while node:
            before, index = previous[node]; edge = graph[before][index]
            edge[2] = 0; graph[node][edge[1]][2] = 1; node = before
    return [pair for pair in pairs if selected[pair][2] == 0]


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, p):
    return cross(a,b,p) == 0 and min(a[0],b[0]) <= p[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= p[1] <= max(a[1],b[1])


def intersects(a, b, c, d):
    x,y,z,w = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    return (x*y < 0 and z*w < 0) or any([
        x == 0 and on_segment(a,b,c), y == 0 and on_segment(a,b,d),
        z == 0 and on_segment(c,d,a), w == 0 and on_segment(c,d,b)])


def polygon_area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in
                   zip(points, points[1:]+points[:1]))) / 2


def roi_polygon(roi, width, height):
    fields(roi, {'version','points'}); identity(roi['version'])
    points = roi['points']
    require(isinstance(points,list) and len(points) >= 4, 'invalid_roi')
    require(all(isinstance(p,list) and len(p)==2 for p in points), 'invalid_roi')
    points = [tuple(number(x) for x in p) for p in points]
    require(points[0] == points[-1], 'roi_not_closed'); points = points[:-1]
    require(len(points) == len(set(points)) and polygon_area(points) > 0, 'invalid_roi')
    require(all(0<=x<=width and 0<=y<=height for x,y in points), 'roi_outside_image')
    n = len(points)
    for i in range(n):
        a,b,c = points[i-1],points[i],points[(i+1)%n]
        # Adjacent backtracking is an overlapping edge, not a simple polygon.
        require(not (cross(a,b,c)==0 and (on_segment(a,b,c) or on_segment(b,c,a))), 'roi_overlapping_edge')
        for j in range(i+1,n):
            if j == i+1 or (i==0 and j==n-1): continue
            require(not intersects(points[i],points[(i+1)%n],points[j],points[(j+1)%n]), 'roi_self_intersection')
    return points


def relation(box, polygon):
    # Clip a simple (possibly concave) polygon to a convex rectangle. Bridges
    # across disconnected pieces cancel in the shoelace sum; no float geometry.
    clipped = polygon[:]
    for axis, bound, sign in [(0,box[0],1),(0,box[2],-1),(1,box[1],1),(1,box[3],-1)]:
        output = []
        for a,b in zip(clipped[-1:]+clipped[:-1],clipped):
            ain,bin = sign*(a[axis]-bound)>=0, sign*(b[axis]-bound)>=0
            if ain != bin:
                t = (bound-a[axis])/(b[axis]-a[axis])
                output.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
            if bin: output.append(b)
        clipped = output
    intersection = polygon_area(clipped) if clipped else Fraction(0)
    if intersection > 0: return 'boundary' if intersection < area(box) else 'in'
    corners = [(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]
    touch = any(intersects(a,b,c,d) for a,b in zip(polygon,polygon[1:]+polygon[:1])
                for c,d in zip(corners,corners[1:]+corners[:1]))
    return 'touch' if touch else 'out'


def boxes(rows, width, height, kind):
    require(isinstance(rows,list), 'invalid_objects')
    result = {}; audit = {}
    for row in rows:
        required = {'id','category','bbox','truncated'} | ({'route'} if kind=='pred' else {'reason'} if kind=='ignore' else set())
        fields(row,required); key=identity(row['id'])
        require(key not in result, 'duplicate_object_id')
        require(row['category']=='person' and type(row['truncated']) is bool, 'invalid_category_or_truncation')
        require(isinstance(row['bbox'],list) and len(row['bbox'])==4, 'invalid_box')
        original=[number(v) for v in row['bbox']]
        require(original[2]>original[0] and original[3]>original[1], 'invalid_box')
        clipped=[max(0,original[0]),max(0,original[1]),min(width,original[2]),min(height,original[3])]
        require(clipped[2]>clipped[0] and clipped[3]>clipped[1], 'empty_clipped_box')
        if kind!='pred' and clipped!=original: require(row['truncated'] is True, 'truncation_unreviewed')
        if kind=='pred': require(row['route'] in ('keep','suppress'), 'route_unknown')
        if kind=='ignore': identity(row['reason'])
        result[key]=(clipped,row)
        audit[key]={'original':[str(v) for v in original],'clipped':[str(v) for v in clipped]}
    return result,dict(sorted(audit.items()))


def rate(n, d):
    return {'numerator':n, 'denominator':d, 'value':n/d if d else None,
            'reason':None if d else 'no_support'}


def blocked_metrics():
    return {name:{'value':None,'numerator':None,'denominator':None,'reason':'blocked'} for name in NAMES}


def reference_signature(unit):
    """Compare actual frozen content, not only caller-supplied version names."""
    polygon=roi_polygon(unit['roi'],unit['width'],unit['height'])
    rotations=[]
    for direction in [polygon,list(reversed(polygon))]:
        rotations.extend(tuple(direction[i:]+direction[:i]) for i in range(len(direction)))
    def label_rows(rows):
        return tuple(sorted((r['id'],r['category'],tuple(number(v) for v in r['bbox']),
            r['truncated'],r.get('reason')) for r in rows))
    return (unit['width'],unit['height'],min(rotations),
            label_rows(unit['labels']['objects']),label_rows(unit['labels']['ignore']))


def unit_score(unit):
    fields(unit, {'unit_id','source_sha256','camera_id','source_version','arm','width','height',
        'split','lineage_ids','input_layer','task','counterfactual_family','target_answers_supplied',
        'source_geometry_substitutes_observability','fixture_admission','roi','policy_version','labels','predictions'},
        {'equivalent_evidence_reviewed'})
    for k in ['unit_id','camera_id','source_version','arm','policy_version']: identity(unit[k])
    require(isinstance(unit['source_sha256'],str) and re.fullmatch('[0-9a-f]{64}',unit['source_sha256']), 'invalid_source_hash')
    require(unit['split'] in ('development','calibration','holdout'), 'split_unknown')
    require(isinstance(unit['lineage_ids'],list) and bool(unit['lineage_ids']), 'lineage_unknown')
    for item in unit['lineage_ids']: identity(item)
    require(len(unit['lineage_ids'])==len(set(unit['lineage_ids'])), 'duplicate_lineage_id')
    require(unit['task']=='person_roi_retention', 'task_out_of_scope')
    require(unit['input_layer'] in ('I0','I1-R','I1-R-no-RGB','I1-E'), 'input_layer_unknown')
    require(unit['target_answers_supplied'] is False or
            (unit['input_layer']=='I1-E' and unit['target_answers_supplied'] is True), 'target_answer_leakage')
    require(unit['source_geometry_substitutes_observability'] is False, 'observability_substitution')
    require(unit['counterfactual_family'] in ('none','A','B','C','D'), 'counterfactual_family_unknown')
    if unit['counterfactual_family']=='C': require(unit.get('equivalent_evidence_reviewed') is True, 'equivalent_evidence_unreviewed')
    fields(unit['fixture_admission'],ADMISSION)
    require(all(v is True for v in unit['fixture_admission'].values()), 'fixture_admission_incomplete')
    width,height=unit['width'],unit['height']
    require(type(width) is int and type(height) is int and width>0 and height>0, 'invalid_dimensions')
    polygon=roi_polygon(unit['roi'],width,height)
    labels=unit['labels'];fields(labels,{'status','annotation_complete','scope','fixture_reviewed','binding','objects','ignore'})
    require(labels['status']=='complete' and labels['annotation_complete'] is True
            and labels['fixture_reviewed'] is True and labels['scope']=='full_frame_visible_person', 'labels_incomplete')
    binding={k:unit[k] for k in ['source_sha256','camera_id','source_version','policy_version']}
    binding['roi_version']=unit['roi']['version']
    require(labels['binding']==binding, 'label_binding_mismatch')
    gt,ga=boxes(labels['objects'],width,height,'gt')
    ignore,ia=boxes(labels['ignore'],width,height,'ignore')
    pred,pa=boxes(unit['predictions'],width,height,'pred')
    require(not (set(gt)&set(ignore)), 'duplicate_label_id')
    edges={(g,p):iou(gb,pb) for g,(gb,_) in gt.items() for p,(pb,_) in pred.items() if iou(gb,pb)>=Fraction(1,2)}
    pairs=maximum_matching(edges);matched=dict(pairs);matched_pred=set(matched.values())
    ignored=sorted(p for p,(pb,_) in pred.items() if p not in matched_pred
                   and any(iou(pb,ib)>=Fraction(1,2) for ib,_ in ignore.values()))
    # Freeze M above. Routing only looks up its original pairs; never re-match.
    gt_relation={g:relation(b,polygon) for g,(b,_) in gt.items()}
    pred_relation={p:relation(b,polygon) for p,(b,_) in pred.items()}
    inside={g for g,r in gt_relation.items() if r in ('in','boundary')}
    boundary={g for g,r in gt_relation.items() if r=='boundary'}
    outside={p for p,r in pred_relation.items() if r=='out' and p not in ignored}
    retained=lambda g:g in matched and pred[matched[g]][1]['route']=='keep'
    suppressed=lambda g:g in matched and pred[matched[g]][1]['route']=='suppress'
    metrics=dict(zip(NAMES,[rate(sum(retained(g) for g in inside),len(inside)),
        rate(sum(suppressed(g) for g in inside),len(inside)),
        rate(sum(retained(g) for g in boundary),sum(g in matched for g in boundary)),
        rate(sum(pred[p][1]['route']=='suppress' for p in outside),len(outside))]))
    return {'unit_id':unit['unit_id'],'arm':unit['arm'],'camera_id':unit['camera_id'],
        'status':'synthetic_unit_scored','metrics':metrics,'pairs':[list(pair) for pair in pairs],
        'pair_iou':{g:str(edges[(g,p)]) for g,p in pairs},'ignored_prediction_ids':ignored,
        'unmatched_prediction_ids':sorted(set(pred)-matched_pred-set(ignored)),
        'gt_relation':dict(sorted(gt_relation.items())), 'prediction_relation':dict(sorted(pred_relation.items())),
        'boxes':{'gt':ga,'ignore':ia,'predictions':pa},
        'inventory':{'boundary_gt':len(boundary),'boundary_baseline_missed':sum(g not in matched for g in boundary),
            'gt_touch':sum(r=='touch' for r in gt_relation.values()),
            'prediction_touch':sum(r=='touch' for r in pred_relation.values()),
            'unmatched_predictions':len(set(pred)-matched_pred-set(ignored)),
            'duplicate_prediction_boxes':len(pred)-len({tuple(b) for b,_ in pred.values()}),
            'unknown_routes':0},
        'interpretation':'configuration_execution_only' if unit['input_layer']=='I1-E' else 'observed_detection_retention_only',
        'hidden_relation_decision':'unknown'}


def envelope():
    return {'contract_version':VERSION,'kind':'synthetic_software_validation',
        'status':'blocked','human_GT':False,'scientific_ROI_admission':False,
        'accuracy_metrics':None,'decision':'unknown','software_metrics_by_arm':{},
        'camera_support_by_arm':{},'locally_valid_subset_camera_support_by_arm':{},
        'arm_scope':{},'cross_arm_comparison':'not_evaluated', 'units':[], 'blockers':[],
        'coverage':{'preregistered_units':0,'admitted_units':0,'locally_valid_units':0,'value':None},
        'limits':['synthetic software counts only; real-source admission not implemented',
                  'fixture evidence flags do not verify truth or real provenance']}


def evaluate(payload):
    report=envelope()
    try:
        require(isinstance(payload,dict) and payload.get('kind')=='synthetic_software_fixture', 'real_source_admission_not_implemented')
        fields(payload,{'contract_version','kind','preregistered_units','units'})
        require(payload['contract_version']==VERSION, 'contract_version_mismatch')
        expected=payload['preregistered_units'];units=payload['units']
        require(isinstance(expected,list) and bool(expected), 'preregistration_missing')
        for key in expected: identity(key)
        require(len(expected)==len(set(expected)), 'duplicate_preregistered_unit')
        report['coverage']['preregistered_units']=len(expected)
        require(isinstance(units,list), 'invalid_units')
        ids=[u.get('unit_id') if isinstance(u,dict) else None for u in units]
        require(all(isinstance(key,str) for key in ids) and len(ids)==len(set(ids))
                and set(ids)==set(expected), 'preregistered_population_mismatch')
        for unit in sorted(units,key=lambda u:u['unit_id']):
            try:
                result=unit_score(unit)
            except ValueError as exc:
                result={'unit_id':unit['unit_id'],'arm':unit.get('arm'),
                        'status':'blocked','reason':str(exc),'metrics':blocked_metrics()}
            report['units'].append(result)
        # Enforce source-connected split isolation and per-arm image uniqueness.
        parents=list(range(len(units)))
        def find(i):
            while parents[i]!=i: parents[i]=parents[parents[i]];i=parents[i]
            return i
        tokens={};seen=set()
        for i,unit in enumerate(units):
            # Invalid units already block the population; never trust malformed lineage.
            if not any(r['unit_id']==unit['unit_id'] and r['status']=='synthetic_unit_scored' for r in report['units']):continue
            key=tuple(unit[k] for k in ['source_sha256','camera_id','source_version','arm'])
            require(key not in seen,'duplicate_group_arm');seen.add(key)
            for token in [('image',unit['source_sha256'])]+[('lineage',v) for v in unit['lineage_ids']]:
                if token in tokens: parents[find(i)]=find(tokens[token])
                else: tokens[token]=i
        components={}
        for i,unit in enumerate(units):
            if any(r['unit_id']==unit['unit_id'] and r['status']=='synthetic_unit_scored' for r in report['units']):
                components.setdefault(find(i),set()).add(unit['split'])
        require(all(len(splits)==1 for splits in components.values()), 'source_component_split_leakage')
        arm_groups={}
        for unit in units:
            if not any(r['unit_id']==unit['unit_id'] and r['status']=='synthetic_unit_scored' for r in report['units']):continue
            arm_groups.setdefault(unit['arm'],[]).append(unit)
        populations=[]; references={}
        for arm, rows in sorted(arm_groups.items()):
            require(len({u['input_layer'] for u in rows})==1, 'arm_mixes_input_layers')
            require(len({u['split'] for u in rows})==1, 'arm_mixes_splits')
            layer=rows[0]['input_layer']
            report['arm_scope'][arm]={'input_layer':layer,'split':rows[0]['split'],
                'interpretation':'configuration_execution_only' if layer=='I1-E' else 'observed_detection_retention_only'}
            populations.append({tuple(u[k] for k in ['source_sha256','camera_id','source_version','policy_version'])+(u['roi']['version'],) for u in rows})
            for unit in rows:
                key=tuple(unit[k] for k in ['source_sha256','camera_id','source_version','policy_version'])+(unit['roi']['version'],)
                signature=reference_signature(unit)
                require(key not in references or references[key]==signature,'shared_reference_content_mismatch')
                references[key]=signature
        require(not populations or all(pop==populations[0] for pop in populations), 'arm_population_mismatch')
        report['cross_arm_comparison']=('not_comparable_across_input_layers'
            if len({v['input_layer'] for v in report['arm_scope'].values()})>1
            else 'same_population_same_input_layer_counts_only')
    except ValueError as exc:
        report['blockers'].append({'reason':str(exc)})
    global_blocked=bool(report['blockers'])
    report['blockers'] += [{'unit_id':r['unit_id'],'reason':r['reason']} for r in report['units'] if r['status']=='blocked']
    admitted=[r for r in report['units'] if r['status']=='synthetic_unit_scored']
    n=report['coverage']['preregistered_units']
    admitted_count=0 if global_blocked else len(admitted)
    report['coverage'].update(admitted_units=admitted_count,locally_valid_units=len(admitted),value=admitted_count/n if n else None)
    arms=sorted({r['arm'] for r in report['units'] if isinstance(r.get('arm'),str)})
    if not report['blockers'] and n and len(admitted)==n: report['status']='synthetic_scored'
    for arm in arms:
        rows=[r for r in admitted if r['arm']==arm]
        report['software_metrics_by_arm'][arm]=({name:rate(sum(r['metrics'][name]['numerator'] for r in rows),
            sum(r['metrics'][name]['denominator'] for r in rows)) for name in NAMES}
            if report['status']=='synthetic_scored' else blocked_metrics())
        supports={camera:{name:{'numerator':sum(r['metrics'][name]['numerator'] for r in rows if r['camera_id']==camera),
            'denominator':sum(r['metrics'][name]['denominator'] for r in rows if r['camera_id']==camera)} for name in NAMES}
            for camera in sorted({r['camera_id'] for r in rows})}
        report['locally_valid_subset_camera_support_by_arm'][arm]=supports
        if report['status']=='synthetic_scored':report['camera_support_by_arm'][arm]=supports
    report['blocked_reason_counts']=dict(sorted(Counter(b['reason'] for b in report['blockers']).items()))
    report['unknown_routes']=sum(1 for u in payload.get('units',[]) if isinstance(u,dict)
        for p in (u.get('predictions') if isinstance(u.get('predictions'),list) else [])
        if isinstance(p,dict) and p.get('route') not in ('keep','suppress')) if isinstance(payload,dict) and isinstance(payload.get('units',[]),list) else 0
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    # Candidate geometry stays outside every checkout, even for blocked input.
    for parent in args.out.resolve().parents:
        if (parent/'.git').exists():
            checkout=subprocess.run(['git','-C',str(parent),'rev-parse','--show-toplevel'],capture_output=True,text=True)
            require(checkout.returncode!=0, 'output_must_be_outside_git')
    try:
        report=evaluate(loads(args.input.read_text()))
    except ValueError as exc:
        report=envelope();report['blockers']=[{'reason':str(exc)}]
    with args.out.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'kind':report['kind'], 'accuracy_metrics':None}))
    return 0 if report['status']=='synthetic_scored' else 2


if __name__=='__main__':
    raise SystemExit(main())
