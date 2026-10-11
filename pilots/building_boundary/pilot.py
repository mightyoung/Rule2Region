"""Stdlib-only boundary compiler, fixed segment counter, independent GT scorer.
No model imports, credentials, image generation, training or dataset downloads.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parent
VERSION = 'building-boundary-v1'


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cut(goal, topology):
    members = set(goal['regions'])
    result = {}
    for name, d in topology['doors'].items():
        if goal['kind'] == 'external_net' and name != goal['door']:
            continue
        delta = int(d['b'] in members) - int(d['a'] in members)
        if delta:
            result[name] = delta
    return result


def propose(topology, goals, strategy):
    """Only topology+goals are accepted; no GT or observations parameter exists."""
    if strategy in ('model_candidate', 'model_plus_metadata_or_plan'):
        return None
    if strategy not in ('metadata_rule', 'door_only'):
        raise ValueError('Unknown proposer')
    targets = {}
    for name, goal in goals['goals'].items():
        boundary = cut(goal, topology) if strategy == 'metadata_rule' else {
            door: 1 for door in topology['doors']
            if goal['kind'] != 'external_net' or door == goal['door']}
        targets[name] = []
        for door, delta in boundary.items():
            cameras = topology['doors'][door]['cameras']
            owner = next(iter(cameras))  # Stable author-provided preferred view, no GT optimization.
            targets[name].append({'door':door, 'owner_camera':owner, 'delta_ab':delta,
                                  **deepcopy(cameras[owner])})
    return {'schema_version':VERSION, 'strategy':strategy, 'targets':targets}


def valid_geometry(entry):
    try:
        p, q = entry['line']
        roi = entry['roi']
        return (len(p) == len(q) == 2 and len(roi) == 4 and
                all(isinstance(v, (int,float)) and math.isfinite(v) and 0 <= v <= 1
                    for v in [*p,*q,*roi]) and p != q and
                roi[0] < roi[2] and roi[1] < roi[3] and
                entry['a_side'] in (-1,1))
    except (KeyError, TypeError, ValueError):
        return False


def validate(config, topology, goals):
    reasons = []
    if config.get('schema_version') != VERSION:
        reasons.append('SCHEMA_VERSION')
    if topology.get('complete_channels_asserted_by') is None:
        reasons.append('UNKNOWN_CHANNEL_COVERAGE')
    init = goals.get('initial', {})
    occupancy = [n for n,g in goals['goals'].items() if g['kind'] == 'occupancy']
    if any(type(init.get(n)) is not int or init[n] < 0 for n in occupancy):
        reasons.append('UNKNOWN_INITIAL_OCCUPANCY')
    elif init['building'] != init['floor1'] + init['floor2']:
        reasons.append('INITIAL_CONSERVATION')
    if set(config.get('targets', {})) != set(goals['goals']):
        reasons.append('TARGET_SET')
    cameras_seen = set()
    for d in topology['doors'].values():
        if d['a'] not in topology['regions'] or d['b'] not in topology['regions'] or d['a'] == d['b']:
            reasons.append('INVALID_TOPOLOGY_ENDPOINT')
        for cam in d['cameras']:
            if cam in cameras_seen:
                reasons.append('CAMERA_DOOR_AMBIGUOUS')
            cameras_seen.add(cam)
    for name, goal in goals['goals'].items():
        if not set(goal['regions']) <= set(topology['regions']):
            reasons.append(name + ':UNKNOWN_REGION')
        expected = cut(goal, topology)
        entries = config.get('targets', {}).get(name, [])
        actual = Counter(e.get('door') for e in entries)
        if set(actual) != set(expected):
            reasons.append(name + ':BOUNDARY_COVERAGE')
        if any(v != 1 for v in actual.values()):
            reasons.append(name + ':MULTIPLE_RESPONSIBILITY')
        for e in entries:
            door = topology['doors'].get(e.get('door'))
            if not door or e.get('owner_camera') not in door['cameras']:
                reasons.append(name + ':INVALID_OWNER')
                continue
            if e.get('delta_ab') != expected.get(e['door']):
                reasons.append(name + ':DIRECTION_OR_MEMBERSHIP')
            if not valid_geometry(e):
                reasons.append(name + ':INVALID_GEOMETRY')
            elif e['a_side'] != door['cameras'][e['owner_camera']]['a_side']:
                reasons.append(name + ':DIRECTION_CALIBRATION')
    return sorted(set(reasons))


def fixture_review(config):
    return {'scope':'synthetic_fixture_only', 'approved_config_sha256':digest(config),
            'reviewer':'test_fixture_author_simulation_not_user_or_production_approval'}


def crossing(obs, entry):
    """Intersect two finite segments; count strict side changes at ROI intersection."""
    p, q = entry['line']
    u, v = obs['prev'], obs['next']
    def side(x):
        return (q[0]-p[0])*(x[1]-p[1]) - (q[1]-p[1])*(x[0]-p[0])
    su, sv = side(u), side(v)
    if su * sv >= 0:
        return None
    t = su / (su - sv)
    xy = [u[i] + t*(v[i]-u[i]) for i in (0,1)]
    length2 = sum((q[i]-p[i])**2 for i in (0,1))
    s = sum((xy[i]-p[i])*(q[i]-p[i]) for i in (0,1)) / length2
    x0,y0,x1,y1 = entry['roi']
    if not (0 <= s <= 1 and x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
        return None
    return 1 if (1 if su > 0 else -1) == entry['a_side'] else -1


def execute(config, topology, goals, observations, review=None, diagnostic=False, steps=12):
    """Same executor in every arm. Receives no GT state/event ledger."""
    reasons = validate(config, topology, goals)
    if review is None or review.get('scope') != 'synthetic_fixture_only' or review.get('approved_config_sha256') != digest(config):
        reasons.append('MISSING_OR_STALE_REVIEW')
    if reasons and not diagnostic:
        return {'status':'REJECTED_OR_UNKNOWN', 'reasons':reasons, 'metrics':None}
    values = {n:goals['initial'][n] if g['kind']=='occupancy' else 0
              for n,g in goals['goals'].items()}
    series = [{'step':0, 'values':dict(values), 'inflow':{}, 'outflow':{}}]
    observed = {n:[] for n in values}
    for step in range(1,steps):
        inflow, outflow = Counter(), Counter()
        for name, entries in config['targets'].items():
            seen = set()
            for e in entries:
                for obs in observations:
                    if obs['step'] != step or obs['camera'] != e['owner_camera']:
                        continue
                    # Stream retransmission de-dup by local track/camera/step, not global GT identity.
                    key = (obs['camera'],obs['track_id'],step,e['door'])
                    if key in seen:
                        continue
                    direction = crossing(obs,e)
                    if direction is None:
                        continue
                    seen.add(key)
                    delta = direction * e['delta_ab']
                    values[name] += delta
                    (inflow if delta > 0 else outflow)[name] += 1
                    observed[name].append([step,e['door'],direction])
        series.append({'step':step, 'values':dict(values), 'inflow':dict(inflow), 'outflow':dict(outflow)})
    return {'status':'DIAGNOSTIC_INVALID_CONFIG' if reasons else 'SYNTHETIC_ADMITTED',
            'reasons':reasons, 'config_sha256':digest(config), 'series':series,
            'events':observed}


def ground_truth(topology, goals, states, events):
    """Occupancy from manual person-location snapshots, net flow from manual ledger."""
    series, expected = [], {n:[] for n in goals['goals']}
    for state in states:
        row = {}
        for name, goal in goals['goals'].items():
            if goal['kind'] == 'occupancy':
                row[name] = sum(r in goal['regions'] for r in state['people'].values())
            else:
                row[name] = sum((int(e['destination'] in goal['regions']) - int(e['source'] in goal['regions']))
                                for e in events if e['step'] <= state['step'] and e['door'] == goal['door'])
        series.append({'step':state['step'], 'values':row})
    for name, goal in goals['goals'].items():
        for e in events:
            if e['door'] in cut(goal, topology):
                d = topology['doors'][e['door']]
                expected[name].append([e['step'],e['door'],1 if e['source']==d['a'] else -1])
    return {'series':series,'events':expected}


def score(run, truth):
    if 'series' not in run:
        return None
    if [r['step'] for r in run['series']] != [r['step'] for r in truth['series']]:
        raise ValueError('Scoring requires identical checkpoint steps')
    metrics = {}
    for name in truth['events']:
        errors = [r['values'][name] - t['values'][name]
                  for r,t in zip(run['series'],truth['series'],strict=True)]
        pred, gt = Counter(map(tuple,run['events'][name])), Counter(map(tuple,truth['events'][name]))
        metrics[name] = {'mae':sum(abs(e) for e in errors)/len(errors),
                         'max_abs_error':max(map(abs,errors)), 'final_error':errors[-1],
                         'event_tp':sum((pred & gt).values()),
                         'event_fp':sum((pred-gt).values()), 'event_fn':sum((gt-pred).values())}
    return metrics


def verify_ledgers(topology, states, events):
    assert [s['step'] for s in states] == list(range(len(states))), 'Checkpoint steps'
    assert all(0 < e['step'] < len(states) for e in events), 'Event outside checkpoint range'
    inventory = set(states[0]['people'])
    assert all(set(s['people']) == inventory for s in states), 'Person inventory changed'
    assert all(r in topology['regions'] for s in states for r in s['people'].values()), 'Unknown GT region'
    for before, after in zip(states,states[1:]):
        observed = {p:(before['people'][p],r) for p,r in after['people'].items() if before['people'][p]!=r}
        ledger = {}
        for e in events:
            if e['step'] != after['step']:
                continue
            assert e['person'] not in ledger, 'Multiple moves of same person in one step'
            d = topology['doors'][e['door']]
            assert {e['source'],e['destination']} == {d['a'],d['b']}, 'Invalid crossing endpoints'
            ledger[e['person']] = (e['source'],e['destination'])
        assert observed == ledger, f'Independent ledger mismatch at {after["step"]}'


def run_all():
    topology, goals = read('topology.json'), read('goals.json')
    states, events, obs = read('states_gt.json')['states'], read('events_gt.json')['events'], read('observations.json')['observations']
    verify_ledgers(topology,states,events)
    truth = ground_truth(topology,goals,states,events)
    assert all(truth['series'][0]['values'][name] == value
               for name,value in goals['initial'].items()), 'Initial occupancy differs from independent GT'
    results = {'schema_version':VERSION, 'experiment':'synthetic_software_only',
               'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
               'fixed_executor_sha256':hashlib.sha256((ROOT/'pilot.py').read_bytes()).hexdigest(),
               'input_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                              ['topology.json','goals.json','states_gt.json','events_gt.json','observations.json','manual_config.json']},
               'truth':truth, 'strategies':{}, 'failures':{}}
    for strategy in ('manual_truth','door_only','metadata_rule','model_candidate','model_plus_metadata_or_plan'):
        config = read('manual_config.json') if strategy=='manual_truth' else propose(topology,goals,strategy)
        if config is None:
            results['strategies'][strategy] = {'status':'NOT_RUN','metrics':None,'reason':'No model invocation authorized in this phase'}
            continue
        review = fixture_review(config)
        run = execute(config,topology,goals,obs,review)
        run['metrics'] = score(run,truth)
        run['config'] = config
        run['review'] = review
        if run['metrics'] is None:
            diag = execute(config,topology,goals,obs,review,diagnostic=True)
            diag['diagnostic_metrics'] = score(diag,truth)
            run['diagnostic'] = diag
        results['strategies'][strategy] = run
    base = read('manual_config.json')
    for case in ('missing_service','missing_vertical','double_owner','reversed_direction','unknown_initial','missing_roi','unreviewed','stale_review','unknown_channels'):
        config, g, t = deepcopy(base), deepcopy(goals), deepcopy(topology)
        review = None
        if case == 'missing_service':
            config['targets']['building'] = [e for e in config['targets']['building'] if e['door']!='service']
        elif case == 'missing_vertical':
            config['targets']['floor1'] = [e for e in config['targets']['floor1'] if e['door'] not in ('stairs','lift')]
        elif case == 'double_owner':
            other = deepcopy(config['targets']['building'][0]); other['owner_camera']='east_side'
            other.update(t['doors']['east']['cameras']['east_side'])
            config['targets']['building'].append(other)
        elif case == 'reversed_direction':
            config['targets']['building'][0]['a_side'] *= -1
        elif case == 'unknown_initial':
            del g['initial']['building']
        elif case == 'missing_roi':
            del config['targets']['building'][0]['roi']
        elif case == 'unknown_channels':
            t['complete_channels_asserted_by'] = None
        if case == 'stale_review':
            review = fixture_review(base)
            config['strategy'] = 'changed_after_review'
        elif case != 'unreviewed':
            review = fixture_review(config)
        run = execute(config,t,g,obs,review)
        if case in ('missing_service','missing_vertical','double_owner','reversed_direction'):
            diag = execute(config,t,g,obs,review,diagnostic=True)
            run['diagnostic_metrics'] = score(diag,truth)
        results['failures'][case] = run
    results['real_public_subtasks'] = {'status':'OFFICIAL_DOCUMENT_ADMISSION_ONLY',
        'visual_calls':'NOT_RUN', 'occupancy_metrics':None, 'event_metrics':None,
        'gt_ledger':'real_gt_ledger.template.json', 'media_acquired':False}
    return results


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    results = run_all()
    (ROOT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({n:{'status':r['status'],'metrics':r['metrics']} for n,r in results['strategies'].items()},indent=2))

if __name__ == '__main__':
    main()
