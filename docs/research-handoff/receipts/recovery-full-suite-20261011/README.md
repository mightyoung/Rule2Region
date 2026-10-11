# Recovery validation closure — 2026-10-11

This append-only receipt supplements the earlier recovery at `95242a7529f1d1581b760b370216191b711d12d2`. The earlier missing-pytest failure remains in its original receipt. No source behavior, tests, assertions, research gates, frozen evidence or main branch were changed.

## Actual coverage and failures

An isolated Python 3.12.14 venv successfully installed the repository package/base requirements and pytest from PyPI. Collection then required the declared OpenCV demo dependency and existing code's scikit-learn import. Two remaining tests required CPU PyTorch; its 2.8.0 version matches the existing public detection configuration and was installed from the official CPU wheel index. Package versions are in `environment-packages.txt`. No previously installed base dependency was upgraded during supplementation; `pip check` passed. These are runtime packages, not model weights. No transformers package, weights, real model inference, paid call or data upload occurred. The teacher test already injects a stub loader and tensor tests use synthetic data.

| Scope | Executed cases | Passed | Failed | Collection errors |
|---|---:|---:|---:|---:|
| Repository `testpaths=tests` | 461 | 409 | 52 | 2 modules |
| Additional statistical/compatibility tools | 22 | 22 | 0 | 0 |
| Existing cache unittest | 64 | 64 | 0 | 0 |
| New qualification unittest | 13 | 13 | 0 | 0 |
| Total | 560 | 508 | 52 | 2 modules |

No test was skipped. All 125 existing test files were attempted: 117 under default testpaths, two additional tool files and six cache unittest files. Two default modules could not be collected, so their six test functions did not run. JUnit's 463 default entries include two collection-error entries; this is **461 executed test cases**, not 463.

Final root command exited 1. `--continue-on-collection-errors` permits the collectable tests to execute while retaining errors in the result; no files were ignored and no failure became a pass. Earlier attempts exited 2 for four collection errors, then 1 for 54 failures/two collection errors before CPU torch supplied the two runtime-only tests.

The suite remains red:

- 51 failures are `FileNotFoundError` for historical `.omx`, artifacts, remote launchers or protocol files absent from the published snapshot.
- `tests/test_evidencegate_runner.py::test_cli_consumes_real_g0_scope_manifest` raises the existing `scope_path_required_without_fixture` gate because its G0 scope file is absent. The gate was preserved.
- `tests/test_build_codec_derivatives.py` and `tests/test_build_development_video_receipt.py` fail collection because `research_tools.remote` is unpublished. Their five plus one test functions are unexecuted. No replacement modules were invented.

Every default case and its outcome, including all exact failure names and messages, is in [repository-test-results.json](repository-test-results.json). Additional cases are in [additional-test-results.json](additional-test-results.json). [VALIDATION.json](VALIDATION.json) records attempts, install commands, exits, coverage, output hashes and limits. Missing historical files cannot be recreated as research truth or made available by installing software packages. These failures do not establish runtime defects in the recovered cache; neither does this receipt claim that the whole repository passes.

## Reproduction

Use a fresh local venv and the versions in `environment-packages.txt`; install CPU torch from the official index. From the repository root, with `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`:

```sh
python -m pytest --continue-on-collection-errors --junitxml=local-root-results.xml
python -m pytest -q research_tools/statistical_gates/test_engine.py research_tools/model_compatibility/test_harness.py
(cd research/continuous-cache-20261010/cache && python -m unittest discover -v && python replay_public.py)
python -m unittest discover -s research/cache-qualification-20261011 -v
python research/cache-qualification-20261011/qualification.py
```

Root exit 1 with the recorded missing-file and collection failures is expected on this publication snapshot, and must not be relabelled as a pass. The two public replays still pass; the four original arms retain 62 unknown event decisions each.

## Independent count and remote audit

A separate dictionary join, without importing `qualification.py`, recomputed camera, original PTS/time-base representation and exact bbox equality. It confirmed 83 inventory observations, 36 matched old geometry endpoints, 47 absent old endpoints, 531 old candidate pairs, 243 pairs with both endpoints in this population, one original accepted hypothesis, and zero accepted adjacent links on the denser time grid. All 83 observation events remain unknown. The one-second accepted chord cannot bridge P375 into a verified continuous track.

The remote task branch was fetched at `95242a7529f1d1581b760b370216191b711d12d2`; all six recovery files were compared directly to the fetched git blobs and matched byte-for-byte. File hashes and independent counts are retained in [independent-verification.json](independent-verification.json). The original 32-entry public manifest remains valid. No original-pixel provenance, HSV, identity, GT or accuracy was inferred from these matches.

## Minimal next inputs, without widening the experiment

Keep the same scene, frozen detector, thresholds, candidate set and nine sampled times. To move from public arithmetic to pixel-backed verification, the minimum needed is:

1. **Existing original pixels and their binding:** the nine full frames at P360, P375, P390, P405, P420, P435, P450, P465 and P480, with each original file's byte hash, exact video PTS/time-base, image dimensions/coordinate convention and fixed-view version. Originals must be supplied through an authorized private access route; public fixture hashes cannot substitute. If continuous crossing time is assessed, the existing short source segment covering before/at/after the event is needed, rather than interpolated frames.
2. **Independent human reference on those same inputs:** full-frame person/negative annotations, explicit uncertain/occluded cases, association labels, physically reviewed door/sill and side labels, and event decision with uncertainty. The annotator should first work without AI prelabels, automatic associations or system event decisions, then record adjudication and exposure. Default adoption or a second AI review does not supply independent human truth.
3. **For the still-open negative-event gate only:** an already-existing, authorized complete parallel-pass example and an already-existing pre-crossing turn-back example, human-confirmed and within the frozen scene/scope. Turning back after exiting cannot replace them. If such existing examples are unavailable, record the missing negative gate and keep precision/false-exclusion claims unavailable; do not expand collection or invent negative labels.

Raw HSV is not necessary to reproduce this public join. A renewed appearance-based association would require the existing pixel/HSV bindings and separate access/publication authorization, and remains outside this completed validation. Permission to use inputs privately must stay separate from permission to publish them. All results remain development-exposed; unknown is not safe, direction change is one evidence item, and accuracy/generalization gains remain unclaimed.
