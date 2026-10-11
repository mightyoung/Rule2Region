"""Companion mapping for unchanged f23 boundary-v1; never reads media or torch.

The preparation index is not a second training schema. Its boundary_v1_template
contains unresolved fields until evidence is actually reviewed. The original
learning.contract remains authoritative for completed manifests. Declared source
bindings add a same-sequence split guard; hashes do not authenticate human truth.
"""
import copy
import math
from learning.contract import (
    MATCHES, RELATIONS, SUFFICIENCY, fingerprint, nonempty, require, sha,
    validate_labels, validate_manifest,
)

BASE_COMMIT = 'f23d587de0e8d82b4a823173359d3f34b0990d77'
IDENTITY_KEYS = ('building_id', 'layout_family_id', 'physical_door_ids',
                 'source_plan_sha256', 'variant_family_id')


def _known_identity(value):
    if not nonempty(value):
        return False
    return value.strip().lower().split(':', 1)[0] not in {
        'unknown', 'pending', 'unverified', 'tbd', 'null', 'none',
    }


def _identity_complete(identity):
    return (all(_known_identity(identity.get(k)) for k in
                ('building_id', 'layout_family_id', 'variant_family_id'))
            and isinstance(identity.get('physical_door_ids'), list)
            and bool(identity['physical_door_ids'])
            and all(_known_identity(x) for x in identity['physical_door_ids'])
            and (identity.get('source_plan_sha256') is None
                 or sha(identity['source_plan_sha256'])))


def _input(sample):
    draft = sample['input_draft']
    views = [{'id': v['local_id'], 'asset_ref': v['asset_ref'], 'sha256': v['sha256']}
             for v in draft['views']]
    require(len(views) == 1 and views[0]['sha256'] == sample['source_file']['sha256'],
            'single original view/source binding required')
    require(nonempty(views[0]['asset_ref']) and nonempty(views[0]['id']), 'view reference')
    doors = []
    for d in draft['door_and_passage_candidates']:
        box = d['xyxy_normalized']
        require(isinstance(box, list) and len(box) == 4
                and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in box)
                and box[0] < box[2] and box[1] < box[3], 'normalized candidate box')
        doors.append({'id': d['local_id'], 'view_id': views[0]['id'], 'box': copy.deepcopy(box)})
    regions = [{'id': r['local_id'], 'description': r['description']} for r in draft['regions']]
    for rows in (doors, regions):
        ids = [r['id'] for r in rows]
        require(all(nonempty(i) for i in ids) and len(ids) == len(set(ids)), 'candidate ids')
    inp = {'views': views, 'allowed_context': draft['allowed_context'], 'regions': regions,
           'door_candidates': doors, 'region_pairs': copy.deepcopy(draft['region_pairs']),
           'view_pairs': copy.deepcopy(draft['view_pairs'])}
    door_ids, region_ids = {r['id'] for r in doors}, {r['id'] for r in regions}
    for pair in inp['region_pairs']:
        require(pair['door_id'] in door_ids and pair['region_a'] in region_ids
                and pair['region_b'] in region_ids and pair['region_a'] != pair['region_b'],
                'region pair reference')
    for pair in inp['view_pairs']:
        require(pair['door_a'] in door_ids and pair['door_b'] in door_ids
                and pair['door_a'] != pair['door_b'], 'view pair reference')
    for key in ('region_pairs', 'view_pairs'):
        ids = [p['id'] for p in inp[key]]
        require(all(nonempty(i) for i in ids) and len(ids) == len(set(ids)), 'pair ids')
    return inp


def adapt_preparation(preparation):
    """Map actual preparation metadata; unknowns stay unresolved, not negatives."""
    require(preparation['schema_version'] == 'private-annotation-preparation.v1'
            and preparation['purpose'] == 'private_annotation_preparation_only', 'preparation format')
    source = preparation['source']
    samples, seen = [], set()
    for s in preparation['samples']:
        sid = s['sample_id']
        require(nonempty(sid) and sid not in seen, 'duplicate/empty sample')
        seen.add(sid)
        require(sha(s['source_file']['sha256']), 'original SHA')
        groups = s['grouping']
        require(all(nonempty(groups.get(k)) for k in ('dataset', 'sequence', 'camera')), 'source grouping')
        identity = {k: copy.deepcopy(groups.get(k)) for k in IDENTITY_KEYS}
        inp = _input(s)
        ai = s['ai_preannotation']
        require(ai['versions'] == s['versions'], 'stale AI versions')
        require(ai['risk'] in ('normal', 'high'), 'AI risk')
        status = {'proposed_pending_review': 'proposed', 'proposed': 'proposed',
                  'default_adopted_pending_review': 'default_adopted_pending_review'}.get(ai['status'])
        require(status is not None, 'AI status')
        validate_labels(ai['labels'], inp)
        geometry_revision = ai['producer_model_revision']
        require(geometry_revision is None or nonempty(geometry_revision), 'geometry revision')
        boundary_ai = None if geometry_revision is None else {
            'model_revision': geometry_revision, 'status': status, 'risk': ai['risk'],
            'labels': copy.deepcopy(ai['labels']), 'versions': copy.deepcopy(s['versions']),
        }
        model = s['existing_person_model_output']
        require(model['source_sha256'] == s['source_file']['sha256'], 'person/source binding')
        require(model['door_model_detection'] is False, 'person output is not a door detector')
        require(type(model['kept_count']) is int and model['kept_count'] >= 0, 'person kept count')
        person_revision = model.get('model_revision')
        review = {k: copy.deepcopy(s['review'][k]) for k in ('status', 'reviewer', 'evidence_refs', 'versions')}
        truth = copy.deepcopy(s['truth'])
        if review['status'] == 'pending':
            require(truth is None and review['reviewer'] is None and review['versions'] is None,
                    'pending AI is not human truth')
        row = {'id': sid, 'kind': 'real', 'split': groups.get('split'), 'identity': identity,
               'source': {'url': source['publisher_url'], 'license': source['source_license_metadata'],
                          'access_date': source.get('prior_source_access_date_utc', ''),
                          'permission_status': source.get('permission_status', 'pending')},
               'versions': copy.deepcopy(s['versions']), 'input': inp, 'ai_preannotation': boundary_ai,
               'review': review, 'truth': truth}
        blockers = []
        if not _identity_complete(identity): blockers.append('missing_verified_group_identity')
        if row['split'] not in ('train', 'calibration', 'test'): blockers.append('split_not_assigned')
        if review['status'] not in ('confirmed', 'corrected') or truth is None: blockers.append('human_review_pending')
        if row['source']['permission_status'] != 'verified': blockers.append('formal_source_permission_not_recorded')
        if geometry_revision is None: blockers.append('AI_revision_unknown_labels_kept_in_sidecar')
        if truth is not None: validate_labels(truth, inp)
        samples.append({'sample_id': sid, 'frame_index': s['frame_index'],
            'boundary_v1_template': row, 'declared_identity': copy.deepcopy(identity),
            'source_component': {'dataset': groups['dataset'], 'sequence': groups['sequence']},
            'camera': groups['camera'], 'input_fingerprint': fingerprint(inp),
            'source_sha256': s['source_file']['sha256'],
            'provenance': {'person_detector': {'model_id': model['model_id'],
                'model_revision': person_revision, 'revision_status': 'known' if nonempty(person_revision) else 'unknown',
                'prompt': model['prompt'], 'box_threshold': model['box_threshold'],
                'text_threshold': model['text_threshold'], 'nms_iou': model['nms_iou'],
                'kept_count': model['kept_count'], 'is_GT': False, 'door_detection': False},
                'geometry_AI': {'model_revision': geometry_revision, 'revision_status': 'known' if geometry_revision else 'unknown',
                    'origin': 'AI visual candidate, not person detector door output'},
                'AI_labels': copy.deepcopy(ai['labels']), 'risk': ai['risk'],
                'candidate_kinds': {d['local_id']: d['kind'] for d in s['input_draft']['door_and_passage_candidates']},
                'building_exterior_status': 'unknown', 'human_confirmed': review['status'] in ('confirmed', 'corrected')},
            'blockers': blockers, 'supervision_ready': False})
    return {'kind': 'boundary-v1-preparation-index', 'formal_manifest': False,
            'learning_base_commit': BASE_COMMIT, 'samples': samples,
            'note': 'No media/features/GT generated. Templates are unresolved; source hashes are not access or human proof.'}


def admit_supervision(manifest, index):
    """Fail closed before supervision; no dataset/optimizer/feature generation."""
    require(index.get('formal_manifest') is False and index.get('learning_base_commit') == BASE_COMMIT,
            'preparation index/base binding')
    require(manifest.get('samples'), 'empty supervision')
    prepared = {s['sample_id']: s for s in index['samples']}
    require(len(prepared) == len(index['samples']), 'duplicate index sample')
    # Validate the entire declared index before admitting any selected subset.
    # Otherwise train/test calls made separately cannot see omitted neighbors.
    declared_source_splits, unassigned_sources = {}, set()
    for prepared_sample in index['samples']:
        component = prepared_sample['source_component']
        require(all(nonempty(component.get(k)) for k in ('dataset', 'sequence')), 'source sequence binding')
        token = (component['dataset'], component['sequence'])
        split = prepared_sample['boundary_v1_template']['split']
        require(split is None or split in ('train', 'calibration', 'test'), 'source sequence declared split invalid')
        if split is None:
            unassigned_sources.add(token)
            continue
        require(token not in declared_source_splits or declared_source_splits[token] == split,
                'source sequence cross-split leakage in full index')
        declared_source_splits[token] = split
    source_owners = {}
    for row in manifest['samples']:
        require(row['id'] in prepared, 'sample not in preparation index')
        p = prepared[row['id']]
        require(row['review']['status'] in ('confirmed', 'corrected') and row['truth'] is not None,
                'human review required before supervision')
        require(_identity_complete(p['declared_identity']) and row['identity'] == p['declared_identity'],
                'verified source identity binding required; no placeholders')
        require(row['split'] == p['boundary_v1_template']['split'], 'declared split binding changed; re-adapt source assignment')
        require(fingerprint(row['input']) == p['input_fingerprint'], 'input binding changed; re-adapt new version')
        require(row['versions'] == p['boundary_v1_template']['versions'], 'preparation version binding')
        for key in ('url', 'license', 'access_date'):
            require(row['source'][key] == p['boundary_v1_template']['source'][key], 'source provenance binding')
        component = p['source_component']
        require(all(nonempty(component.get(k)) for k in ('dataset', 'sequence')), 'source sequence binding')
        token = (component['dataset'], component['sequence'])
        require(token not in unassigned_sources, 'source sequence has unassigned index members; do not supervise subset')
        require(declared_source_splits[token] == row['split'], 'source sequence declared split binding')
        require(token not in source_owners or source_owners[token] == row['split'], 'source sequence cross-split leakage')
        source_owners[token] = row['split']
    validate_manifest(manifest) # Original unchanged f23 contract, including evidence/versions/permission/labels.
    classes = {'relations': RELATIONS, 'matches': MATCHES, 'sufficiency': SUFFICIENCY}
    labels = {r['id']: {task: {pid: classes[task].index(value) for pid, value in mapping.items()}
                       for task, mapping in r['truth'].items()} for r in manifest['samples']}
    return {'label_indices': labels, 'source_components': len(source_owners),
            'scope': 'declared metadata admission only; no training or authenticity certification'}


def public_index(index):
    """Identity-only projection: omit media refs, boxes, geometry and raw outputs."""
    return {'kind': 'public-preparation-index', 'formal_manifest': False,
            'learning_base_commit': index['learning_base_commit'],
            'samples': [{'sample_id': s['sample_id'], 'frame_index': s['frame_index'],
                'source_sha256': s['source_sha256'], 'source_component': s['source_component'], 'camera': s['camera'],
                'versions': s['boundary_v1_template']['versions'], 'split': s['boundary_v1_template']['split'],
                'door_candidate_ids': list(s['provenance']['candidate_kinds']),
                'candidate_kinds': s['provenance']['candidate_kinds'],
                'AI_labels': s['provenance']['AI_labels'], 'AI_revision': s['provenance']['geometry_AI'],
                'person_detector': s['provenance']['person_detector'],
                'building_exterior_status': 'unknown', 'human_review_status': s['boundary_v1_template']['review']['status'],
                'truth_available': s['boundary_v1_template']['truth'] is not None,
                'blockers': s['blockers'], 'supervision_ready': False} for s in index['samples']],
            'note': 'Metadata-only index is not training input or human truth; original media stays private.'}
