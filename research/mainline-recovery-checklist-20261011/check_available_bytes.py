"""Read-only exact-byte availability audit; no download, reconstruction or inference.

Run from repository root. Scan only authorized local roots. Ignore symlinks,
credentials/config dot directories, installed runtimes and synthetic pytest outputs.
Media/path-level matches stay in the private output, not the public aggregate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SKIP = {'.git', '.aws', '.codex', '.cache', '.config', '.ssh', '.gnupg',
        'site-packages', '__pycache__', 'node_modules', 'pytest-of-agent', 'pytest-of-root'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(repo, roots):
    evidence = json.loads((repo / 'docs/research-handoff/evidence/boundary-16-20261010.json').read_text())
    partial = json.loads((repo / 'docs/research-handoff/evidence/partial-human-review-five-20261010.json').read_text())
    expected = [{'kind': 'artifact', 'id': r['artifact_id'], 'sha256': r['sha256'], 'bytes': r['bytes']} for r in evidence['artifacts']]
    expected += [{'kind': 'partial_review', 'id': r['artifact'], 'sha256': r['sha256'], 'bytes': None} for r in partial['receipt_artifacts']]
    expected += [{'kind': 'image', 'id': r['sample_id'], 'sha256': r['image_sha256'], 'bytes': None} for r in evidence['samples']]
    try:
        from PIL import Image
    except ImportError:
        Image = None
    rgb_targets = {r['decoded_RGB_sha256']: r['sample_id'] for r in evidence['samples']}
    rgb_matches = {sid: [] for sid in rgb_targets.values()}
    rgb_checked, rgb_errors = 0, 0
    targets = {r['sha256'] for r in expected}
    sizes = {r['bytes'] for r in expected if r['bytes'] is not None}
    matches = {s: [] for s in targets}
    errors, file_count, candidate_count = [], 0, 0
    seen = set()
    for root in roots:
        def onerror(error):
            errors.append({'path': error.filename, 'type': type(error).__name__})
        for base, dirs, filenames in os.walk(root, followlinks=False, onerror=onerror):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(base, d).is_symlink())
            for name in sorted(filenames):
                p = Path(base, name)
                if name.startswith('.env') or p.is_symlink():
                    continue
                try:
                    st = p.stat()
                    key = (st.st_dev, st.st_ino)
                    if key in seen or not p.is_file():
                        continue
                    seen.add(key)
                    file_count += 1
                    is_image = p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                    # Review JSON has no published length; all JSON is eligible.
                    if st.st_size not in sizes and not is_image and p.suffix.lower() != '.json':
                        continue
                    candidate_count += 1
                    s = digest(p)
                    if s in matches:
                        matches[s].append({'path': str(p), 'bytes': st.st_size})
                    if is_image and Image is not None:
                        try:
                            with Image.open(p) as image:
                                rgb = image.convert('RGB')
                                rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                                rgb_checked += 1
                                if rgb_hash in rgb_targets:
                                    rgb_matches[rgb_targets[rgb_hash]].append({'path': str(p), 'decoded_size': list(rgb.size), 'original_file_sha256': s})
                        except (OSError, ValueError):
                            rgb_errors += 1
                except (OSError, ValueError) as error:
                    errors.append({'path': str(p), 'type': type(error).__name__})
    for r in expected:
        r['matches'] = [m for m in matches[r['sha256']] if r['bytes'] is None or m['bytes'] == r['bytes']]
        r['status'] = 'EXACT_BYTES_VERIFIED' if r['matches'] else 'NOT_FOUND_IN_SCANNED_LOCAL_FILES'
    return {'scope': 'standalone regular files only; archive contents and inaccessible/excluded roots not exhaustively searched',
            'roots': [str(r) for r in roots], 'excluded_directory_names': sorted(SKIP),
            'regular_files_examined': file_count, 'eligible_files_hashed': candidate_count,
            'errors': errors, 'expected': expected,
            'decoded_RGB_check': {'method': 'Pillow convert RGB then row-major tobytes SHA256; published full decoding convention unavailable', 'Pillow_available': Image is not None, 'images_decoded': rgb_checked, 'decode_errors': rgb_errors, 'matches': rgb_matches},
            'note': 'A missing local match is not proof of absence everywhere. Decoded-RGB equality is not original-file byte equality. No model or GT regenerated.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve(), [p.resolve() for p in args.root])
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'regular_files_examined': result['regular_files_examined'],
                      'eligible_files_hashed': result['eligible_files_hashed'],
                      'scan_errors': len(result['errors']),
                      'exact_matches': sum(bool(r['matches']) for r in result['expected']),
                      'expected_records': len(result['expected'])}))


if __name__ == '__main__':
    main()
