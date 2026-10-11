# Public inventory and NMS coverage audit

This is an additional verification of the existing Rule2Region public package. It uses the nine-frame inventory, old bbox fixture, five detector record JSON files and published detector configuration at source commit `6b12baa67e72a544a1c445c546ebc73da1ddfdb4`. It imports only the Python standard library and never loads AI visual correspondence, media, HSV or a model.

The audit independently reconstructs confidence-descending/ID-tiebroken classwise greedy NMS with strict IoU > the recorded 0.7 threshold. All 50 supplied raw-above-threshold candidates yield exactly the published 47 kept and 3 suppressed observations. Suppression pairs and IoU values match within 1e-12; kept IDs/classes/bboxes/confidences match exactly. Each of those 47 observations also matches the nine-frame inventory's appropriate frame. The remaining 36 old observations match the old fixture's exact ID, bbox and original PTS/time-base representation. All 83 inventory observations are accounted for.

This establishes public fixture consistency, not actual-person recall, original-pixel provenance, model inference or event accuracy. The 36 old-frame confidences lack a second source in the old fixture and are explicitly unchecked. Original text-token scores are not published, so the text-threshold filtering cannot be replayed. The historical five model calls are not rerun. This audit computes no associations and leaves trajectory, direction continuity, physical side and events unknown. Direction change remains one evidence item, not an event condition. No ambiguous business target was supplied; composite query count = 0.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/public-coverage-audit-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/public-coverage-audit-20261011/coverage_audit.py
```

Twelve tests cover changed bbox/confidence/time binding, duplicate IDs, suppressed boxes wrongly inserted into inventory, false suppressors, missing forward records, changed source declaration, changed NMS threshold and input immutability. The tests perturb deep copies only; thresholds and all actual frozen fixtures are unchanged. These are controlled software counterexamples, **not** human-verified parallel-pass/turn-back negatives or recognition evidence. The initial missing-module and missing-implementation test failures are recorded in the linked validation receipt.

`coverage_results.json` holds summary counts and per-frame coverage, without republishing raw bboxes. The command recomputes and compares the JSON; only explicit `--generate` writes it. All 83 event decisions stay unknown, accuracy is null, confirmed real negative events remain zero, and unknown does not mean safe. Limits and history blockers are recorded in [failure trace](../../docs/research-handoff/receipts/failure-trace-20261011/README.md).
