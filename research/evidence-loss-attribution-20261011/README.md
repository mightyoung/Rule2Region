# Evidence-loss attribution on the existing public replay

Baseline `87b4c11fbebf820e5368e0d9ac93b341db523959`. This version instruments the existing nine-frame **no-request** replay with the already implemented deletion `event_sink`. It uses the same published 83 observations, 701 unscored candidate relations, one old accepted endpoint hypothesis and all default capacities/TTL. No original kernel, development adapter, input, configuration, previous result or gate is changed. The instrumented replay output matches the frozen branch exactly.

## Observed cause, not an inferred priority problem

The accepted hypothesis connects public source times 12 → 13 s. It is already `unknown_motion` at ingestion at 13 s because an intervening frame makes its endpoints nonadjacent. That qualification failure precedes retention loss and is independent of capacity/TTL. An accepted endpoint hypothesis remains association evidence, not a verified continuous person trajectory.

At source time **15 s (PTS 450, time base 1/30)**, its previous observation is deleted by the **64-observation per-view limit**. The sink reports 65 → 64 observations in a single view; the total observation limit 256 is not reached. The resulting accepted relation deletion is `endpoint_removed_relation`, root `capacity / evicted_observation`.

At that relation deletion, live relation count is **454 → 453**, below the 512 relation limit. It precedes every direct relation-capacity deletion in the replay. The previous endpoint's age is exactly three seconds, the inclusive TTL boundary; it is **not expired by TTL at this deletion**. The relation did not generate a motion descriptor before its deletion, because qualification had already failed at ingestion.

Thus the **701 candidates did not directly evict this accepted relation through the relation-capacity policy**. They are a total enumeration count, not simultaneous live occupancy or 701 verified associations. Direct relation-capacity pressure does occur subsequently: 13 relations are evicted at the same source time 15 s, all unscored candidates. This is not evidence that candidates are harmless in other workloads, or that the accepted hypothesis is important or true.

## Complete loss accounting

| Deleted entity / cause | Count |
|---|---:|
| Observations, capacity | 6 |
| Observations, source TTL | 18 |
| Accepted relation, observation-capacity cascade | 1 |
| Unscored candidates, observation-capacity cascade | 74 |
| Unscored candidates, direct relation capacity | 13 |
| Unscored candidates, endpoint TTL cascade | 179 |
| Geometry descriptors, endpoint capacity cascade | 6 |
| Geometry descriptors, endpoint TTL cascade | 18 |

Relations: 702 created = 267 deleted + 435 live. Observations: 83 = 6 capacity + 18 TTL + 59 live. Geometry descriptors: 83 = 24 endpoint removals + 59 live. No descriptor-capacity eviction or qualified motion descriptor occurs. Accepted-relation TTL deletions observed: zero, because the single accepted relation was already removed by observation capacity. No counterfactual larger-capacity or priority run was performed.

## Design implications supported by this replay

**The retention contract needs to state capacity preemption explicitly.** TTL currently supplies an upper lifetime bound, not a promise of availability throughout those three seconds. A consumer cannot infer that a within-TTL hypothesis remains queryable. If the research requires that availability, the demonstrated review target is observation-level retention/budget or a separately bounded hypothesis-summary contract with explicit missing-endpoint status. This run does not select a replacement policy, enlarge capacity, retain deleted endpoints or establish an importance score. Any change must preserve boundedness and distinguish retained summary from qualified continuous evidence.

**No evidence here justifies prioritizing accepted relations over candidates or increasing the relation limit to fix this loss.** The accepted relation disappeared while relation capacity still had room. Those changes would address a different mechanism. Candidate enumeration pressure is measurable, but it is not the cause shown for the accepted loss.

**Do not relax adjacency qualification to compensate.** The missing intermediate association remains a real qualification gap. Retaining the endpoint hypothesis longer would not create a middle identity link, continuous direction, physical-side answer or a safe/negative event. All 83 event states remain unknown; this is neither an accuracy result nor an innovation claim, and AI hypotheses are not human truth.

## Executed verification and boundaries

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research/evidence-loss-attribution-20261011/attribute_loss.py
```

The finite runner recomputes the diagnostics, asserts the frozen result's exact equality, causal event order, capacity/TTL distinction and entity conservation, then compares `attribution_results.json` exactly. It exports aggregates and a source-time lifecycle without raw observation identifiers, boxes, HSV or new identity paths. The raw stream is collected only in memory for this finite offline attribution; production sink durability/retention remains caller-owned.

Prior full-suite failures remain unchanged and referenced: 52 failures and two collection errors in the preceding receipt. They are neither repaired nor suppressed. This step does not rerun the full repository suite merely to count tests; it verifies frozen manifest entries and executes the actual diagnostic replay. No model, new media, private-material publication, install, paid API, configuration change, main modification or deployment. Mac persistence is handled separately and is not claimed by this task.
