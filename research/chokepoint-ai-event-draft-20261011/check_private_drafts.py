"""Check AI draft evidence binding only; no inference or truth creation."""
from pathlib import Path
import argparse,json,hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draft',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    args=p.parse_args();data=json.loads(args.draft.read_text())
    rows=json.loads((args.source/'ORIGINAL_FRAME_MANIFEST_PRIVATE.json').read_text())
    by_index={r['original_frame_index']:r for r in rows}
    assert sorted(by_index)==list(range(112,333))
    bound=set()
    for candidate in data['candidates']:
        assert candidate['status']=='AI_VISUAL_DRAFT_NOT_HUMAN_GT'
        assert candidate['physical_event_type']=='unknown' and candidate['complete_physical_event'] is None
        assert candidate['physical_inside_before']==candidate['physical_outside_after']=='unknown'
        for field in ['candidate_start_index_interval','candidate_end_index_interval']:
            a,b=candidate[field];assert 112<=a<=b<=332
        for row in candidate['evidence_frames']:
            index=row['original_frame_index'];expected=by_index[index]
            assert row['source_sha256']==expected['sha256'] and row['original_filename']==expected['filename']
            path=args.source/'original-jpegs'/expected['filename']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']
            bound.add(index)
    result={'candidate_AI_drafts':len(data['candidates']),'unique_evidence_frames_verified':len(bound),
            'source_span':[112,332],'source_member_axis_complete':True,
            'physical_inside_outside_changes_confirmed':0,'confirmed_complete_physical_event_count':None,
            'human_GT_created':False,'seconds_generated':False,'model_calls':0,'Library_retry':False,
            'AI_draft_does_not_validate_human_truth':True}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
