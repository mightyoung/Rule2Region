"""Author small deterministic software fixtures. No rendered/video/model input."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def write(name, obj):
    (ROOT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    regions = {'O': None, 'A': 1, 'B': 1, 'C': 2, 'D': 2}
    doors = [
        ('east', 'O', 'A', 'external', ['east_front', 'east_side'], -1),
        ('upper', 'O', 'C', 'external', ['upper_front', 'upper_side'], 1),
        ('service', 'O', 'B', 'external', ['service_cam'], -1),
        ('room1', 'A', 'B', 'internal', ['room1_cam'], -1),
        ('room2', 'C', 'D', 'internal', ['room2_cam'], -1),
        ('stairs', 'A', 'C', 'stairs', ['stairs_cam'], -1),
        ('lift', 'B', 'D', 'elevator', ['lift_cam'], 1),
    ]
    topology = {'schema_version': 'building-boundary-v1', 'source': 'self_authored_no_visual_model',
                'complete_channels_asserted_by': 'synthetic_fixture_author',
                'regions': regions, 'doors': {}}
    for name, a, b, kind, cams, side in doors:
        topology['doors'][name] = {'a': a, 'b': b, 'kind': kind, 'cameras': {}}
        for i, cam in enumerate(cams):
            topology['doors'][name]['cameras'][cam] = {
                'line': [[0.5, 0.2], [0.5, 0.8]], 'roi': [0.2, 0.1, 0.8, 0.9],
                'a_side': side if i == 0 else -side}
    write('topology.json', topology)
    write('goals.json', {'schema_version': 'building-boundary-v1',
        'business_rules': '统计整栋及两层占用；逐外门净流量；同层换房不改变层/栋总数；全通道覆盖。',
        'initial': {'building': 3, 'floor1': 2, 'floor2': 1},
        'goals': {'building': {'kind': 'occupancy', 'regions': ['A', 'B', 'C', 'D']},
                  'floor1': {'kind': 'occupancy', 'regions': ['A', 'B']},
                  'floor2': {'kind': 'occupancy', 'regions': ['C', 'D']},
                  'net_east': {'kind': 'external_net', 'door': 'east', 'regions': ['A', 'B', 'C', 'D']},
                  'net_upper': {'kind': 'external_net', 'door': 'upper', 'regions': ['A', 'B', 'C', 'D']},
                  'net_service': {'kind': 'external_net', 'door': 'service', 'regions': ['A', 'B', 'C', 'D']}}})
    # Explicit manually authored checkpoints. GT never imports executor/planner/observations.
    positions = [
        ['A','B','C','O','O'], ['A','B','C','A','O'],
        ['B','B','C','A','O'], ['B','D','C','A','O'],
        ['B','D','A','A','O'], ['B','D','A','O','O'],
        ['B','D','A','O','C'], ['O','D','A','O','C'],
        ['O','D','A','O','D'], ['O','B','A','O','D'],
        ['O','B','O','C','D'], ['O','B','O','A','D']]
    write('states_gt.json', {'provenance': 'manual_person_location_checkpoints_before_execution',
        'states': [{'step': t, 'people': {f'p{i+1}': r for i, r in enumerate(row)}}
                   for t, row in enumerate(positions)]})
    # Separate manual physical crossing ledger; includes repeated passage, simultaneous moves.
    events = [(1,'p4','east','O','A'), (2,'p1','room1','A','B'),
              (3,'p2','lift','B','D'), (4,'p3','stairs','C','A'),
              (5,'p4','east','A','O'), (6,'p5','upper','O','C'),
              (7,'p1','service','B','O'), (8,'p5','room2','C','D'),
              (9,'p2','lift','D','B'), (10,'p3','east','A','O'),
              (10,'p4','upper','O','C'), (11,'p4','stairs','C','A')]
    write('events_gt.json', {'provenance': 'manual_physical_crossing_ledger_not_counter_output',
        'events': [dict(step=t, person=p, door=d, source=a, destination=b)
                   for t,p,d,a,b in events]})
    # Observation schedule is separately authored. No GT person IDs/global event IDs in streams.
    schedule = [(1,'east',1),(2,'room1',1),(3,'lift',1),(4,'stairs',-1),
                (5,'east',-1),(6,'upper',1),(7,'service',-1),(8,'room2',1),
                (9,'lift',-1),(10,'east',-1),(10,'upper',1),(11,'stairs',-1)]
    obs = []
    for t, name, direction in schedule:
        for cam, geometry in topology['doors'][name]['cameras'].items():
            # Signed line side: x < .5 is positive for this vertical oriented line.
            start_side = geometry['a_side'] * direction
            start_x = 0.3 if start_side == 1 else 0.7
            obs.append({'step': t, 'camera': cam, 'track_id': f'{cam}_local_{t}',
                        'prev': [start_x, 0.5], 'next': [1-start_x, 0.5]})
    # A pass alongside a door that never crosses and a crossing outside the line/ROI.
    obs.extend([{'step': 2,'camera':'east_front','track_id':'pass_by',
                 'prev':[0.3,0.5],'next':[0.35,0.6]},
                {'step': 6,'camera':'east_front','track_id':'outside_roi',
                 'prev':[0.3,0.95],'next':[0.7,0.95]}])
    write('observations.json', {'provenance': 'hand_authored_per_camera_coordinate_segments_no_pixels',
                              'observations': obs})
    # Explicit manual membership/cuts, independent of automatic boundary selection.
    cuts = {'building': {'east':1,'upper':1,'service':1},
            'floor1': {'east':1,'service':1,'stairs':-1,'lift':-1},
            'floor2': {'upper':1,'stairs':1,'lift':1},
            'net_east': {'east':1},'net_upper': {'upper':1},'net_service': {'service':1}}
    targets = {}
    for goal, cut in cuts.items():
        targets[goal] = []
        for d, delta in cut.items():
            door = topology['doors'][d]
            cam = next(iter(door['cameras']))
            targets[goal].append({'door':d, 'owner_camera':cam, 'delta_ab':delta,
                                 **door['cameras'][cam]})
    write('manual_config.json', {'schema_version':'building-boundary-v1',
                                'strategy':'manual_truth', 'targets':targets})

if __name__ == '__main__':
    main()
