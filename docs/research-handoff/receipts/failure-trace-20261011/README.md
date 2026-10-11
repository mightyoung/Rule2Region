# Failure dependency trace and further public validation — 2026-10-11

This continues the same recovery at `6b12baa67e72a544a1c445c546ebc73da1ddfdb4`. No historical fixtures, protocol text, skill bodies or launchers were fabricated. No existing test assertion or research gate was changed. No model or paid API was called, and no unapproved private material was uploaded.

## Missing-file dependency groups

The 52 failures collapse to 24 **first blocking file paths**, not 52 independent defects:

| Dependency class | Failed tests |
|---|---:|
| Historical JSON, manifests and frozen execution scope | 42 |
| Historical preregistration/protocol/claims documents | 5 |
| Required execution-policy skill body | 2 |
| Remote launch scripts | 3 |

Fifteen tests stop at the same camera decision catalog. Nine stop at the same G0 execution scope, including the existing CLI refusal `scope_path_required_without_fixture`. These two roots account for 24 failures. Recovering a root could reveal further dependencies, so these are not guaranteed pass counts.

The two collection errors are separate: `research_tools/remote/build_codec_derivatives.py` and `research_tools/remote/build_development_video_receipt.py` are absent. Their five plus one test functions never execute. `file_browse/SKILL.md` blocks two execution-context tests; the registry also requires the absent `count_schedule/SKILL.md`, which those tests have not yet reached. The registry's requirement was preserved.

Every exact path, affected test, category and history-search outcome is listed in [history_probe_results.json](history_probe_results.json). These missing dependencies mostly belong to earlier EvidenceGate/VIRAT/Qwen experiments; they are distinct from the current nine-frame Rule2Region pixel/GT gaps. Installing dependencies already resolved cv2, sklearn and torch imports. The remaining red suite cannot be repaired by installing more packages.

## History and version intent

The actual remote branch query returned only main at `cf22ddd8e46ee83ddd9565b058e1df4e362377ea` and the task branch at the source commit above. The tag query returned no tags. Fetching both branches yielded nine reachable commits; the repository is not shallow. The probe examines the tree of **every one of those nine commits**, checking exact paths and all same-basename alternatives. No primary missing file or secondary source dependency exists in those trees, and no same-basename candidate exists either.

The source publication root has no imported `human-count` history. `docs/publication-checks.md` records that the snapshot excludes remote launchers and original research evidence, and warns that missing assets must fail rather than be replaced with placeholders. `SOURCE_MANIFEST.json` identifies published files; a source hash is not access to missing bytes. These records establish why reconstructing scripts or frozen JSON from expected test strings would change version/intent rather than restore it. There is **no provenance-preserving historical restoration available in this repository history**. Restored historical files = 0.

The pinned probe is reproducible from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/research-handoff/receipts/failure-trace-20261011/probe_history.py
```

It recomputes the committed tree inventory and compares the recorded JSON. The source commit is intentionally pinned; this newly appended report does not retroactively become evidence that missing bytes existed in earlier history.

## Additional useful work on available public bytes

[Public coverage audit](../../../../research/public-coverage-audit-20261011/README.md) adds a separate standard-library cross-check and twelve controlled counterexample tests. It reconstructs greedy classwise NMS from the five already-public raw-above-threshold records, preserving confidence-descending/ID order and strict IoU > 0.7 suppression:

- 50 supplied threshold candidates → 47 retained, 3 suppressed; every retained bbox/confidence and suppression endpoint/IoU matches the published record.
- All 47 retained observations match the five corresponding inventory frames exactly, including original PTS/time-base representation, bbox and confidence.
- All 36 observations in the four old frames match the older bbox fixture by sample, ID, PTS/time-base and bbox.
- Zero unmatched inventory observations; this is fixture coverage, **not person recall**. All 83 event statuses remain unknown; no identity, crossing, side, body direction or new link is inferred.

The old fixture has no confidences, so its 36 inventory confidences cannot be independently checked against it. Original text-token scores are absent; text-threshold filtering and model forwards cannot be replayed. Pixel/HSV/physical side/independent GT gaps remain. Controlled corruptions are software counterexamples and must not be counted as real negative events.

## Fresh verification

The root suite reran with unchanged tests and `--continue-on-collection-errors`: 409 passed, 52 failed, two collection errors, zero skipped; exit 1. The exact failures, collection errors and case identities match the previous closure receipt. The additional tools passed 22 cases; the old cache passed 64, qualification passed 13, and new coverage passed 12. Both old public replays and the new coverage comparison passed.

Total actual executed test cases now = 572: 520 passed and 52 failed. The six functions in the two uncollectable modules remain unexecuted. All 126 present test files were attempted across the configured root scope and explicit additional scopes. This is not a whole-repository pass. [VALIDATION.json](VALIDATION.json) retains commands, exits, output hashes, exact failure-set comparison and limits. Existing public manifests were reverified and unchanged.

## Minimal inputs that can actually be reviewed

For historical software restoration, the smallest source-only set addressing the three launcher failures, two context failures and two module-import blockers is the **seven exact-version source files** listed under `source_only_restore_candidates` in [input-requirements.json](input-requirements.json). Require actual bytes, original revision, byte hash, source intent and applicable publication rights. That set addresses first blockers for five failed tests and six unexecuted functions; it does not guarantee those tests pass. Do not recreate a former preregistration or a result JSON to satisfy an assertion. Full historical replay additionally requires the 24 primary paths and their original transitive manifests/files; supplying one catalog is not proof of a complete artifact pack.

For the active Rule2Region experiment, [input-requirements.json](input-requirements.json) instead gives the nine existing sample IDs/PTS and explicit pixel/annotation requirements. Supply those originals privately with byte hashes, dimensions, exact time-base and fixed-view mapping; obtain a first independent human pass on full-frame people/negatives, associations, reviewed physical door geometry, sides and events without showing system decisions. Preserve uncertain/occluded cases and all exposure. Assessing continuous crossing time additionally needs the existing short source segment spanning before/at/after crossing. None of these inputs is currently verified or auto-labelled as truth.

The two missing complete negative types are an existing parallel pass and an existing pre-crossing turn-back. If unavailable, retain the missing gate and null precision/false-exclusion claims; turning back after exiting does not qualify. No new scene, media acquisition or experiment expansion is authorized by this checklist. Private input access and public redistribution are separate; HSV and the private second batch remain outside publication scope. Unknown is not safe; direction change is one evidence item; no recognition gain is claimed.
