# Synthetic boundary and mutation validation

Baseline: `2baf16615b61bd7c04193b7c8b06b81b056aa06d`. This is an independent pure-Python check of the versioned endpoint-evidence/deletion adapter. **No production change or real bug was found in the exercised boundaries.** The adapter, frozen cache/kernel, previous test assertions and all real replay outputs remain byte-identical. This package contains synthetic software tests and process receipts only, not recovered material, a recognition experiment or human truth.

## What the checks protect

| Research boundary | Actual check and deliberately broken variant |
|---|---|
| unknown cannot impersonate accepted | Unknown relation queries and a conflicting accepted resubmission stay unknown; removing the accepted-status filter fails an assertion. |
| endpoint evidence has a bounded lifetime | Removing either endpoint leaves no dependent relation/descriptor; omitting cascade cleanup fails. |
| denser sampling cannot invent identity continuity | Zero, one, two and five intervening synthetic frames retain the same endpoint span; only the zero-gap chord can qualify, while identity and event remain unknown. Disabling adjacency qualification fails. |
| reading does not renew evidence | Reads at source times 1, 2 and the exact inclusive TTL boundary 3 preserve the original event timestamp. At 3+1/30 the old endpoint and dependent evidence actually disappear. Renewing its source timestamp fails. |
| query filtering is not actual deletion | Valid delayed queries must remove expired internal records, not merely omit them from output. Removing maintenance fails. |
| late evidence respects the query cutoff | Receipt time 2 cannot expose a relation in a historical query at 1. Removing the availability guard fails. |
| diagnostic labels preserve association class | Capacity cascades separately label accepted and unknown/unscored candidates. Labeling a candidate accepted fails. |
| capacity and TTL causes stay distinct | A capacity cascade followed by independent TTL maintenance must not carry the earlier cause forward. Mislabeling capacity as TTL fails. |
| returned evidence cannot rewrite provenance | Mutating an exported source binding leaves internal relations unchanged. Returning the internal list by alias fails. |

Global observation pressure across two registered views also verifies dependent cleanup and query isolation. The mutations cover these specific assertions, not every possible implementation defect. Passing software boundaries supplies no person association, physical-side evidence, event negative or recognition gain; unknown still does not mean safe. The sampling tests concern ingested frames, not the completeness of a real video source.

## Executed procedure

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/cache-boundary-mutations-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/cache-boundary-mutations-20261011/mutation_check.py
```

Eight boundary methods pass, including their explicit subcases. Nine hand-authored variants are compiled in isolated memory namespaces; each receives fresh synthetic caches. Source files are never patched on disk, no mutant is used in a real replay, and no compiled mutant is written into production. All nine are detected by **failed assertions**, with zero unexpected errors; the saved receipt repeats exactly. This is not an exhaustive mutation score or nine discovered vulnerabilities.

The initial timestamp-renewal variant also produced two secondary query exceptions by changing the current target's timestamp. Those exceptions were not counted as successful detection. It was narrowed to renew only the preceding endpoint, leaving the target binding valid; the final nine variants produce assertion failures without exceptions. The first receipt/log hashes are retained in validation evidence.

The repository command was executed in the existing isolated environment: 409 passed, 52 failed, two collection errors. Every module/test/status equals the preceding frozen full-suite receipt. Additional tools 22, old cache 64, qualification 13, coverage 12, nine-frame 7 and development-adapter 17 pass; five existing real/software replays match. Including the eight new methods, ordinary validation executes 604 methods/cases: 552 pass and 52 fail, plus the same two collection errors leaving six functions unexecuted. Intentional mutant failures are reported separately and are not added to ordinary passing-test totals. See `VALIDATION.json` and the prior complete failure identities.

No installation, model, paid API, media acquisition, private nine-frame box/HSV/identifier publication, capacity increase, main change or deployment. Code and summaries may be published on the existing task branch. A separate outside-git handoff manifest gives the pinned commit, sizes/hashes and log archive; it does not overwrite the ongoing Mac snapshot task's manifest. Mac transfer is not claimed here.
