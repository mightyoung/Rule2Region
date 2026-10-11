"""Author synthetic scene calibration/census inputs; no GT ledger is read."""
import json
from pathlib import Path
from reproduce_v1 import v1
ROOT=Path(__file__).resolve().parent

def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    topology=v1.read('topology.json')
    reference={'kind':'self_authored_coordinate_reference_not_image', 'revision':1,'views':{}}
    # Independently documented reference labels. No person positions, events or count outputs.
    left_a={'east_side','upper_front','lift_cam'}
    for door,d in topology['doors'].items():
        for camera in d['cameras']:
            a=[.25,.5] if camera in left_a else [.75,.5]
            b=[1-a[0],.5]
            reference['views'][camera]={'door':door,'a_region':d['a'],'b_region':d['b'],
                'a_anchor':a,'b_anchor':b,'doorway_support_roi':[.45,.25,.55,.75]}
    write('scene_reference.json',reference)
    evidence={'schema_version':'scene-evidence-v2','revision':1,
        'scope':'synthetic_fixture_only','review_status':'AUTHOR_REVIEWED_SYNTHETIC_REFERENCE',
        'reviewer':'fixture_author_simulation_not_field_inspection',
        'reference_sha256':v1.digest(reference),'reference_kind':reference['kind'],
        'view_ids':sorted(reference['views'])}
    write('geometry_evidence.json',evidence)
    census={'schema_version':'initial-census-v2','revision':1,'scope':'synthetic_fixture_only',
        't0':0,'time_basis':'synthetic_step','source_id':'fixture_author_independent_census',
        'status':'AUTHOR_RECORDED_NOT_REAL_FIELD_CENSUS',
        'initial':{'building':3,'floor1':2,'floor2':1}}
    write('initial_census.json',census)
    session={'schema_version':'initial-session-v2','session_id':'synthetic-session-001','revision':1,
        't0':0,'time_basis':'synthetic_step','initial':dict(census['initial']),
        'source_id':census['source_id'],'source_revision':census['revision'],
        'source_sha256':v1.digest(census)}
    write('initial_session.json',session)

if __name__=='__main__':
    main()
