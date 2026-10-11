"""Recover one original-index span from an authorised local C1 archive.

No downloads, inference, temporal interpolation, renumbering or tar extraction.
"""
import argparse
from pathlib import Path
import hashlib
import json
import re
import shutil
import tarfile
import time

EXPECTED_C1_SHA='23bb285929056603e387fc6898d56c0a5cdfa64ebd717e8060682d13afd36201'
SPAN=(112,332)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['archive','axis','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from PIL import Image
    t=time.perf_counter();actual=digest(args.archive);assert actual==EXPECTED_C1_SHA
    members=json.loads(args.axis.read_text())['members'];inventory={r['frame_index']:r for r in members}
    selected=set(range(SPAN[0],SPAN[1]+1));assert selected<=set(inventory)
    assert shutil.disk_usage(args.output).free>sum(inventory[i]['bytes'] for i in selected)+512*1024**2
    dump(args.output/'FROZEN_SELECTION.json',{
        'source_archive_sha256':actual,'selected_original_indices':list(SPAN),
        'selection_basis':'already seen original112 background; original142/262/312 approach/passage/occlusion candidates; new original332 background sample',
        'complete_visible_cycle_is_candidate_not_physical_event_truth':True,
        'time_axis':'original filename frame indices only','seconds':None,
        'metadata_nominal_publisher_fps':30,'actual_native_capture_fps_verified':False,
        'native_pts_time_base_available':False,'no_model_calls':True})
    images=args.output/'original-jpegs';images.mkdir();rows=[];seen=set()
    with tarfile.open(args.archive,'r:xz') as archive:
        for m in archive:
            match=re.fullmatch(r'P1E_S1_C1/(\d{8})\.jpg',m.name)
            if not match:continue
            index=int(match[1])
            if index not in selected:continue
            assert m.isfile() and index not in seen and m.size==inventory[index]['bytes']
            stream=archive.extractfile(m);assert stream is not None
            path=images/f'{index:08d}.jpg'
            with path.open('xb') as output:shutil.copyfileobj(stream,output)
            with Image.open(path) as image:
                assert image.format=='JPEG' and image.size==(800,600)
                image.load()
            assert path.stat().st_size==m.size
            rows.append({'original_member':m.name,'original_frame_index':index,
                         'filename':path.name,'bytes':m.size,'sha256':digest(path),'dimensions':[800,600]})
            seen.add(index)
            if seen==selected:break
    assert seen==selected
    rows.sort(key=lambda r:r['original_frame_index'])
    dump(args.output/'ORIGINAL_FRAME_MANIFEST_PRIVATE.json',rows)
    result={'source_archive_sha256':actual,'source_axis_sha256':digest(args.axis),
            'span_first_index':SPAN[0],'span_last_index':SPAN[1],
            'original_JPEG_count':len(rows),'decoded_dimensions':[800,600],
            'original_filename_axis_contiguous':True,'missing_indices_within_span':[],
            'missing_before_span':[100,111],'next_original_gap_after_parent_run':[347,455],
            'parent_contiguous_original_member_run':[112,346],
            'original_image_bytes':sum(r['bytes'] for r in rows),'images_transformed':False,
            'seconds_or_duration_reconstructed':None,'native_pts_available':False,
            'time_base_available':False,'actual_source_fps_verified':False,
            'publisher_nominal_fps':30,'frame_axis_is_native_time':False,
            'actual_acquisition_continuity_verified':False,
            'original_frame_manifest_sha256':digest(args.output/'ORIGINAL_FRAME_MANIFEST_PRIVATE.json'),
            'model_calls':0,'model_outputs_modified':False,'GT_created':False,
            'physical_event_label':None,'extraction_elapsed_seconds':time.perf_counter()-t}
    dump(args.output/'RECOVERY_RECEIPT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
