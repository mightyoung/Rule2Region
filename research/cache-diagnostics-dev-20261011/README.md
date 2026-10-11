# Versioned pure-software cache development

Source baseline: `26a58781decff0c18fe735cdfb10113cd95260d3`. This adapter subclasses the frozen `ContinuousCache` without editing its code, vendored kernel, fixtures, previous experiments or outputs. All new executed inputs here are small synthetic boxes on a 100×100 test canvas, with synthetic identifiers/hashes. This is software development validation, not an 83-box recognition experiment, original-image recovery or human truth.

## Endpoint evidence remains separate from motion qualification

`retrieve` adds `accepted_endpoint_hypotheses`: **retained accepted relations directly into the requested target**, with relation source hash, exact endpoint bindings, availability/receipt times, source-time span, count of intervening ingested frames, motion qualification/missing inputs and descriptor availability. It exports copies. It neither walks an extra continuous chain nor adds observations to `accepted_history` or `motion_descriptors`.

The synthetic endpoint a@0 → b@2 remains visible after an intervening empty frame at 1. Its gap is one ingested frame; intermediate association is not established. It has `nonadjacent_frames`, no motion descriptor and no qualified continuous history. This directly exercises the insertion issue observed in the separate public nine-frame replay without importing private frame boxes or identifiers. An ingested-frame count is not a count of all source-camera frames, unidentified people or missing physical movement. No middle observation, identity link, motion vector or side evidence is manufactured.

Only live endpoints/relations inside the same view, source-time window and historical query availability are exported. Legitimate TTL or capacity deletion can make a hypothesis unavailable; this adapter does not persist evidence beyond the existing retention contract. Unknown candidates never enter the accepted field. Adjacent accepted pairs still qualify only under the frozen motion rules; one chord still does not establish direction continuity. Descriptor eviction can leave an accepted relation visible while its motion feature remains unavailable. Side, event, safety and identity judgments stay unknown/unverified.

## Deletion diagnostics use the existing sink

The existing `event_sink` now receives an optional `deletion` object **at the actual deletion event**. It distinguishes observation/relation/descriptor; accepted hypothesis versus an unknown unscored candidate; geometry/motion descriptor; direct reason and propagated root cause (`capacity`, `source_ttl`, `view_change`, or explicit removal); and entity counts before/after. No deleted bbox or full record is copied into it. `importance_assessed` is false. The root cause propagates through endpoint/relation descriptor cascades instead of mislabeling every dependent removal as a fresh capacity or TTL decision.

Counts are snapshots at each entity's event hook, not a claim that all dependent removals have already finished. Capacity reason does not assert which per-view/global bound selected the item. Unknown relation schema contains no score; `unscored_candidate` is the diagnostic label for that status, not a hidden new score/classification. Synthetic one-relation tests independently demonstrate deletion of each class; they do not claim that an important accepted relation was lost in the real nine-frame run. Selection priority and all default capacities are unchanged: per view 64 observations, 512 relations, 128 descriptors; TTL three source seconds.

The original audit ring stays bounded and unchanged. Enriched events are streamed to the caller's sink; there is no extra retained deletion-history collection, observation archive or background I/O. The transient context is cleared after each deletion/cascade. Caller owns sink durability, backpressure and retention; a sink is not an automatic durable file writer. Sink exceptions follow the existing callback behavior and do not introduce rollback guarantees.

## Query maintenance versus read lifetime

A valid query runs the original source-time maintenance, so old unrelated/endpoint records may expire and produce diagnostics. A read within the existing window does not recompute descriptors or renew observation, relation or descriptor lifetime. The source-time TTL boundary remains inclusive. Null queries do not maintain; malformed/noncausal/rejected queries keep the original validation-before-maintenance behavior. No application ambiguity classifier or event judgment engine is added.

## Executed verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/cache-diagnostics-dev-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/cache-diagnostics-dev-20261011/software_scenarios.py
```

Fourteen initial tests were run against the frozen class alias: 13 assertion failures exposed absent endpoint/diagnostic exports, and the existing rejected/null-query behavior passed. Minimal implementation made all 14 pass. Three additional edge checks cover target scope, independently evicted motion descriptors and deletion payload/history bounds. All **17** pass. The finite synthetic scenario receipt recalculates and matches exactly. The scenario runner deliberately uses existing synthetic test helpers, not production images or external models.

Full repository validation still has **409 passed, 52 failed, 2 collection errors**; the exact module/test/status set matches the preceding receipt. Additional tool tests 22, original cache 64, qualification 13, coverage 12 and nine-frame tests 7 all pass; four frozen public replays match. Including the 17 new tests, 596 cases were actually executed: **544 passed, 52 failed**. The two collection errors leave the same six functions unexecuted. Failures were not repaired with fabricated fixtures, skips or changed assertions. See `VALIDATION.json`, including failure identities and log hashes.

No model, paid API, media download, network material access, new installation, capacity increase, private box/HSV publication, main change or deployment occurs in this step. Git synchronization publishes code, synthetic tests and software summaries only. Accuracy stays null; no recognition gain is asserted, AI is not human GT and unknown does not mean safe. Important public outputs and private log hashes are added to the outside-git persistence manifest; Mac is offline and saving there remains pending, not reported complete.
