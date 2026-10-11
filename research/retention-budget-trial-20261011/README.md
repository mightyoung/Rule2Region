# Same-budget observation retention: preregistered development trial

The protocol was committed **before candidate implementation or execution** at `e66ca5f0f463d3d1a2f79d4ecde52392faaf1c2b`, on baseline `5977bd4267ae0c75c480b90b63cbd6701fd671ab`. See [PROTOCOL.md](PROTOCOL.md). One opt-in candidate, no tuning search and no production-default change. The original code, data, attribution and experiment outputs remain immutable.

The candidate protects whole live source-bound accepted endpoint pairs, oldest admitted relation first, within one quarter of the applicable observation budget. Default per-view protection is at most 16 of 64 observations; all remaining victims follow the baseline ordering. Unknowns and mismatched bindings get no protection. Existing relationship/descriptor limits, admission, TTL, query causality and qualification rules are reused unchanged. This is bounded priority sharing, not verified identity, importance or per-person fairness.

## Actual public-cache result: boundary availability only

| Measure | Baseline | Candidate |
|---|---:|---:|
| Accepted hypothesis deletion source time | 15 s | 15.5 s |
| Deletion root | Observation capacity | Endpoint TTL |
| Accepted hypothesis available in diagnostic copy at 15 s | No | Yes |
| Available one source tick later, 15+1/30 s | No | No |
| Other-goal observation capacity victims | 5 | 6 |
| Total observation capacity victims | 6 | 6 |
| Qualified motion records | 0 | 0 |
| Event unknown observations | 83 | 83 |

**The half-second deletion-event delay is not a half-second valid query-window gain.** The previous endpoint is at source time 12, and its three-second TTL reaches the inclusive boundary at 15. Baseline capacity deletion already occurred with zero strictly remaining TTL. Candidate protects that exact checkpoint, but a query at 15+1/30 triggers TTL maintenance and loses it. Deletion is otherwise reported at the next ingested frame, 15.5. No positive continuous valid-query interval gain is demonstrated on this real public sequence.

One different observation takes the capacity loss instead; the final victim sets differ by one in each direction. These are observation-level targets, not independently identified people. The sparse endpoint hypothesis still has `nonadjacent_frames`, zero qualified history/motion and unknown event/identity; keeping it cannot fill the intervening link. Query source bindings reference the same published JSON hashes, not recovered original pixels or human truth.

Primary public replay has no requests and unchanged inputs/budgets. Baseline output equals the frozen no-request JSON exactly. Diagnostic queries run on bounded **state copies** at checkpoints, so they cannot renew or otherwise affect either selection arm. Candidate output is separately versioned; original results are not overwritten.

## Negative costs under declared synthetic pressure

All these records use the existing synthetic 100×100 test geometry, synthetic ids and placeholder hash bindings, never nine-frame private boxes/HSV or person GT. An 80-record ordinary burst at source time 2 deliberately exposes zero-source-time dwell; that is a source-clock batch effect, not wall-clock latency or independent person starvation measurement.

| Scenario | Accepted relations retained, baseline → candidate | Ordinary capacity losses, baseline → candidate |
|---|---:|---:|
| One accepted pair + overcrowding | 0 → 1 | 16 → 18 |
| 16 disjoint accepted pairs + crowding | 0 → 8 | 16 → 32 |
| 1024 unknown candidate edges + crowding | 0 → 0 | 16 → 16 |
| 32 unverified accepted pairs + crowding | 0 → 8 | 16 → 32 |
| Two views, first with one accepted pair | 0 → 1 | 32 → 34 |

Every ordinary capacity loss in these burst scenarios has zero source-time dwell. Candidate preservation shifts losses onto other goals; the quota cannot guarantee all goals retain evidence throughout TTL. Eight of sixteen accepted pairs still fail under quota saturation. The 32-pair pressure also shows that source/binding admission alone cannot tell honest/important accepted evidence from an adversarial or incorrect upstream stream. Such inputs can occupy the protected slice. Protection is not a truth detector or an optimal fair scheduler.

In the expiry/read scenario, candidate evidence remains available at the inclusive TTL boundary and disappears one tick later. Source timestamps never change; after final maintenance at time 6, both arms have zero observations, relations and descriptors. No expired endpoint is retained by protection or reads. Unknown flood creates no protection and preserves baseline eviction behavior.

Both arms stay within **post-operation** limits: per-view 64 observations, 512 relations and 128 descriptors; totals 256/2048/512. Existing insertion-before-bounds behavior permits transient event-hook counts of 65 observations per overflowing view and 513 relations during the unknown-edge flood. This transient behavior also exists in baseline; it is not a strict byte-memory or zero-transient-overflow guarantee. Descriptor highwater remains within its independent limit. Protection is recomputed from bounded existing state and does not archive extra observation records. The finite offline measurement ledger/state copies are experiment instrumentation, not production memory accounting.

The declared two-view scenario cannot trigger the global observation limit: two views admit at most 128, below 256. It verifies per-view isolation/bounds and reports that infeasibility rather than lowering the budget. Global-overflow victim fairness is therefore **not independently exercised by this trial**. Per-view/global computed protection-size counters are quota diagnostics, not proof that those sets were all simultaneously pinned in a selection operation.

## Decision and unsupported conclusions

Keep this candidate as development code; **the public sequence does not justify changing the default strategy**. It demonstrates preservation of one valid boundary checkpoint at a measured cost to another observation, not an added positive-duration valid interval or recognition gain. Synthetic cases verify the intended retention mechanism where capacity arrives earlier in TTL, but also expose displaced ordinary goals, quota-excluded accepted goals and source-valid upstream abuse. Choosing a policy needs an explicit utility/fairness contract beyond accepted status. Under fixed capacity, all within-TTL availability cannot be guaranteed when offered observations exceed capacity.

No new identity chain, physical-side evidence, safe/negative event, accuracy, model result, human reference, innovation, optimal fairness or deployment validation follows. Unknown remains unknown. Quota, capacity, TTL, thresholds and qualification were not tuned after observing these costs.

## Reproduce and actual validation

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/retention-budget-trial-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/retention-budget-trial-20261011/run_trial.py
```

Five synthetic guardrails pass. Test-first run against the baseline alias had three assertion failures and two passing existing guards. The initial query-lifetime test first raised because its target had already been evicted; an explicit target-presence assertion made the missing behavior a failed assertion before implementation. No policy quota/arm was changed. Initial result wording counted sampled deletion delay as availability; it was corrected by adding the next-tick diagnostic query, not by changing strategy or dropping the negative result. Role/protection counters were made explicit, with initial receipt hashes retained. Final full replay repeats exactly.

Full repository execution: **409 pass, 52 fail, two collection errors**, with identical module/test/status identities to the prior frozen receipt. Additional tools 22 and existing unittest groups 64/13/12/7/17/8 pass. Including five new guards, 609 ordinary methods/cases were executed: 557 pass, 52 fail; the same two collection errors leave six functions unexecuted. No skips, assertion weakening or fabricated fixture repair. See `VALIDATION.json`.

Pure software cloud computation only: no model, pixels, paid API, download, install, private publication, main modification or deployment. Public code/methods/aggregates are committed to the research branch. Mac persistence is pending and handled separately; this trial makes no Mac save claim.
