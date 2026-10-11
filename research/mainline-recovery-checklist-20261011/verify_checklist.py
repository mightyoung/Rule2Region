"""Verify public checklist against pinned repository evidence, without reading media."""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def verify():
    checklist = json.loads((HERE / 'SAMPLE_RECOVERY_CHECKLIST.json').read_text())
    for source in checklist['public_evidence_bindings']:
        original = subprocess.check_output(['git', 'show', source['commit'] + ':' + source['path']], cwd=REPO)
        assert len(original) == source['bytes']
        assert hashlib.sha256(original).hexdigest() == source['sha256']
        assert original == (REPO / source['path']).read_bytes(), 'historical source changed'
    index = json.loads((REPO / 'docs/research-handoff/evidence/boundary-16-20261010.json').read_text())
    partial = json.loads((REPO / 'docs/research-handoff/evidence/partial-human-review-five-20261010.json').read_text())
    rows = checklist['samples']
    assert len(rows) == len(index['samples']) == 16
    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in index['samples']]
    for r, s in zip(rows, index['samples']):
        assert r['original_image']['expected_file_sha256'] == s['image_sha256']
        assert r['original_image']['expected_decoded_RGB_sha256'] == s['decoded_RGB_sha256']
        assert r['fixed_full_frame_F']['candidate_count'] == s['F_candidates']
        assert r['fixed_full_frame_F']['cache_mode'] == s['cache_mode']
        assert r['human_ROI_H']['historically_available'] == s['H_available']
        assert r['candidate_ROI_M_M0']['historically_available'] == s['M_available']
        assert r['human_reference']['complete'] == s['human_person_annotation_complete'] == False
        assert not r['original_image']['local_exact_file_received']
        assert not r['fixed_full_frame_F']['local_prediction_bytes_verified']
    artifacts = json.loads((HERE / 'ARTIFACT_RECOVERY_CHECKLIST.json').read_text())
    for r, s in zip(artifacts['frozen_artifacts'], index['artifacts']):
        assert r['artifact_id'] == s['artifact_id'] and r['sha256'] == s['sha256'] and r['bytes'] == s['bytes']
    assert len(artifacts['frozen_artifacts']) == len(index['artifacts']) == 14
    assert len(artifacts['partial_human_receipts']) == len(partial['receipt_artifacts']) == 6
    for r, s in zip(artifacts['partial_human_receipts'], partial['receipt_artifacts']):
        assert r['artifact'] == s['artifact'] and r['sha256'] == s['sha256']
    assert sum(r['fixed_full_frame_F']['candidate_count'] for r in rows) == 58
    assert sum(r['human_ROI_H']['historically_available'] for r in rows) == 4
    assert sum(r['candidate_ROI_M_M0']['historically_available'] for r in rows) == 14
    assert sum(r['fixed_full_frame_F']['cache_mode'] == 'reused_identical_decoded_RGB' for r in rows) == 2
    receipt = json.loads((HERE / 'EXECUTION_RECEIPT.json').read_text())
    assert receipt['scan_tool_sha256'] == hashlib.sha256((HERE / 'check_available_bytes.py').read_bytes()).hexdigest()
    for r in json.loads((HERE / 'PUBLIC_MANIFEST.json').read_text())['files']:
        b = (HERE / r['path']).read_bytes()
        assert len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256']
    print('PASS: 16 sample identities, 14 frozen artifact identities, 6 partial review identities, seven pinned source files, fixed coverage/counts and public manifest. Missing bytes and GT remain missing.')


if __name__ == '__main__':
    verify()
