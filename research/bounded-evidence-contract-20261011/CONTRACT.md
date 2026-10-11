# Preregistered bounded evidence contract — one missing-aware fragment candidate

Status: **DOCUMENT ONLY / NOT READY FOR IMPLEMENTATION OR VALIDATION**. Frozen against `ed6485e1dc6393b99803f64f617423e6562515e8`. Stop accepted-endpoint protection tuning, additional toy tests and new experiment runs. This contract names exactly one future candidate. It does not change any default, implement compression, extend a cache or claim a new result. Failure ends this candidate; it is not permission to search for another design.

## 1. Verified sources and limits of transfer

**Dataflow — source facts.** Section 1.3 separates event time from processing time; watermarks can be heuristic progress estimates, not certainty that all input has arrived. Sections 2.2–2.3 discuss windows and triggers. [The Dataflow Model, Akidau et al., PVLDB 2015, pp. 1794–1797](https://web.stanford.edu/class/cs245/readings/dataflow-model.pdf).

**Rule2Region inference, not a Dataflow theorem.** Source PTS/time-base, receipt/availability and query time need separate fields. A progress marker does not restore discarded observations, supply missing associations or turn retention loss into completed negative evidence. The existing cache's per-view event/source-time cursor is not a proven completeness watermark. No Dataflow windowing/engine is being adopted.

**ABQS — source facts.** Definition 3.3 measures maximum sampled-point distance to a segment; Definition 3.6 defines error-bounded compressed trajectories. Section 5/PBQS propagates prior compression uncertainty; ABQS manages limited storage with age-dependent precision. This is not a uniform fixed-error guarantee for indefinitely old data. [A Novel Framework for Online Amnesic Trajectory Compression in Resource-constrained Environments, Liu et al., arXiv:1605.02337v1, §§3.3, 5](https://arxiv.org/pdf/1605.02337).

**Rule2Region inference, not a result of ABQS.** Geometric deviation bounds do not by themselves preserve a doorway crossing, continuous dwell duration or event order, or infer motion during unsampled/identity-ambiguous intervals. Time-synchronized error and perpendicular deviation are different properties. This candidate does not implement ABQS or claim its error bounds; it keeps explicit omitted/missing coverage and makes no interpolation claim.

**CacheFlow — source facts.** Sections 2–3 represent rule dependencies and require dependency-aware selection; cover-set splicing redirects affected traffic to software. Section 4.1 states that software switches store the entire policy for cache misses. [CacheFlow: Dependency-Aware Rule-Caching for Software-Defined Networks, Katta et al., SOSR 2016, §§2, 3.2–3.3, 4.1](https://www.cs.princeton.edu/~dpw/papers/cacheflow16.pdf).

**Rule2Region inference, not CacheFlow's guarantee.** Track the closure needed by each claim, but do not borrow equivalence guarantees: Rule2Region has no complete online slow path containing evicted observations, identity truth and physical-side information. Offline reference video is an evaluation input, not a hidden online fallback. Dependency loss must downgrade a claim or return unknown; it cannot be repaired by a cache-splicing analogy.

Access receipt: Dataflow and ABQS were read as their original PDFs through browsing. Princeton browsing returned HTTP 403 on follow-up; the same original Princeton URL was successfully fetched and locally text-extracted, verifying the entire-policy slow-path statement. Source metadata is in `SOURCES.json`. No PDF/text copy is republished here; the source facts are paraphrased and separate from the proposed contract.

## 2. Time and claim semantics

Each admitted source item binds camera/view generation, exact PTS and time-base, source artifact hash, model/method version, receipt time and availability time. Processing/query times are separate; none overwrite original source time. Late input can be unavailable at a historical query even while stored now. No new association or human truth follows from metadata validity.

TTL is **a longest validity bound**, not a minimum three-second retention promise. For an entire fragment, set `expires_at = min(original_source_time_of_each_retained_support + 3 seconds)`; keep the existing inclusive boundary and reject evidence after it. Its validity is also bounded by every dependency's expiry/revision. Creating, serializing, merging, copying, reading or partially dropping a fragment must never recompute expiry from the current time or a later endpoint. No rolling copy can renew old support. Different clocks must not be conflated to report delayed deletion as useful query lifetime.

An event/source-time progress cursor does not imply complete camera sampling. Without declared upper bounds on input rate, burst size, simultaneous targets, lateness and query concurrency, promise **no shortest retention or response latency**. Fixed memory can reject/drop work and return unknown; it cannot retain all offered evidence throughout TTL.

Geometry and trajectory/direction status remain continuously recorded where admitted source supports them. Composite event judgment remains limited to independently defined ambiguous targets; this proposal adds no ambiguity classifier. A direction change is one evidence item, never a standalone safe/non-entry rule. Missing storage, missing association, missing original pixels and missing human labels are different gaps; unknown is never safe.

## 3. The single candidate: missing-aware trajectory fragment record

One candidate is a fixed-size, self-contained **fragment record with up to three exact retained source samples and explicit missing information**. It is neither a complete trajectory nor a synthetic identity chain.

A record contains bounded camera/view/generation tokens; original source/availability/receipt times; up to three source bindings and copied anchor/feature values with method/geometry revisions; at most seven dependency digests; a coverage/missing mask; bounded first/last affected time; reason codes and saturated counters; original immutable expiry; and `identity_verified=false`, side/event unknown until separately warranted. Upstream association status stays distinct from identity truth. No raw image/HSV or model output is newly inferred. Exact copied values are not a claim of physical anchor accuracy.

Selection is deterministic: retain first and last admitted source samples, plus the earliest interior admitted sample if one exists. Never manufacture a third sample. All other samples are explicitly omitted; omitted-known input is different from unobserved input. Missing/cross-fragment identity links remain missing. The record's error-certificate field is `NOT_ESTABLISHED`: there is no geometric interpolation/epsilon guarantee in this candidate. An endpoint line is not a recovered path. The three-point limit is frozen, not a knob to tune on previous failures.

Capacity eviction may remove original cache records while this charged self-contained fragment remains available as a **historical copied fragment**, with `original_endpoint_resident=false` and a retention-loss reason. That does not make an endpoint live, restore qualified history in the old API, renew TTL or qualify a motion chain. It cannot be exposed in `accepted_history` or current `motion_descriptors` merely because a summary exists.

Retraction, stale/mismatched binding, retired camera/view, geometry revision or dependency expiry invalidates derived claims and any dependent fragment. Independent copied source facts may remain only within their original expiry and with unavailable/unverified dependency flags; they cannot satisfy a claim requiring the lost dependency. There is no silent splice over a missing identity/frame interval. If an invalidation index or missing-detail record is lost, invalidate affected fragments conservatively (bounded full-view scan/flush) and report coverage unknown; do not treat absent diagnostics as proof of complete coverage.

## 4. Budget ledger: no free metadata or external slow path

Existing count ceilings remain 64 observations/512 relations/128 descriptors per active view, and total 256/2048/512, with at most four active cameras and eight registered keys. This is a proposed allocation contract, **not an implemented or measured physical-byte guarantee**. Byte parity with baseline is an entry gate, not an assumption.

All candidate storage is debited to the existing descriptor pool; no additional archive is allowed. Reserve at most 26 descriptor slots per active view, leaving at least 102 for ordinary descriptors. The candidate must report any ordinary descriptor loss from this reservation; it is not a free saving.

| Charged component | Max records/view | Packed payload cap/record | Slot debit |
|---|---:|---:|---:|
| Three-sample fragment, including bindings and missing mask | 8 | 1024 bytes | 8 descriptors |
| Dependency/reverse-index record, including bounded entries | 8 | 128 bytes | 8 descriptors |
| Missing-detail/tombstone record, including expiry/reason | 8 | 128 bytes | 8 descriptors |
| Ingest/query/serialization scratch buffers | 2 | 1024 bytes | 2 descriptors |
| Fixed allocator/record header | 26 | 64 bytes | Included with each slot, not free |

The planned packed subtotal is 13,952 bytes/view; four active views total 55,808 bytes. A fixed 256-byte registered-key control/generation/coverage entry, at most eight, adds 2,048 charged bytes. Planned candidate-specific packed overhead is thus at most 57,856 bytes globally. Control storage is charged to the shared total byte envelope even though it is not a source observation. Indices, hashes, expiry queues, clocks, counters, query responses, invalidation latches and allocation padding must fit these caps or be explicitly charged to common baseline storage. No duplicate hash table or Python-object overhead may be omitted. Any packet exceeding its cap is rejected/dropped as unknown, not spilled to an uncounted list.

Before any future implementation/run, establish a **single absolute physical-byte envelope `B` from an independently supplied deployment budget**, plus a fixed accounting/allocator method and bounded input/message size and request concurrency. Freeze these before seeing outcomes; run baseline and candidate under identical `B`. Count common cache state, container overhead, fragmentation, temporary copies, frame normalization/input staging and external-process memory serving the online feature. The current object ceilings do not specify `B`; therefore memory feasibility and same-byte-budget equality are **unresolved blockers**. The packed table alone does not certify RSS or transient memory bounds. Do not enlarge `B` because the candidate does not fit.

One query/ingest operation may occupy the two charged scratch buffers; additional concurrency or oversized frames are rejected without an internal unbounded queue. There is no positive retention guarantee derived from this rejection policy. Offline video/GT and durable research receipts may be separate evaluation artifacts, but cannot supply an uncounted production cache or recovery path. Diagnostic export is a bounded message; retaining streamed messages consumes caller memory and must be included if it is part of the online system.

## 5. Allocation, fairness and overload

At most eight fragments per active view and one fragment per upstream dependency-group/generation. A group is not a verified person. Admit all groups under the same FIFO source-time ordering, break ties by stable binding digest, and evict the oldest complete packet when its pool is full. No accepted-status priority, read-driven LRU, hidden pin or protection sweep. Do not reuse the previous protection candidate as another arm.

Unknown/unverified groups cannot bypass per-group/record caps. Source-valid adversarial groups may still flood admission; this contract provides bounded sharing, **not a minimum fair dwell time or Sybil-proof person allocation**. On any overload, record loss/coverage status within the charged bounded ledger. If ledger detail itself overflows, preserve a fixed generation-level `coverage_unknown` flag and invalidate ambiguous affected claims; no per-target loss history is guaranteed forever. Rewriting that flag for newly lost input does not renew any old sample/fragment expiry.

Future reports must include ordinary-goal loss, zero-dwell drops, refused inputs/queries, per-group and per-view losses, missing-ledger/index overflows, byte/count highwater and candidate usefulness after expiry/qualification filtering. Average cache hits or deletion-event delay cannot substitute for these measurements. No trajectory/identity rate may be inferred from anonymous observation ids.

## 6. Unknown reasons and invalidation closure

Reason codes are combinable, fixed-width and versioned. Details may be bounded/coarsened; losing details must increase uncertainty.

| Code | Required meaning/action |
|---|---|
| `INPUT_UNOBSERVED` / `INPUT_DROPPED_OVERLOAD` | Sampling/arrival gap; no claim of negative event or complete interval. |
| `OBS_CAPACITY_REMOVED` / `REL_CAPACITY_REMOVED` / `DESC_CAPACITY_REMOVED` | Separate resource-loss causes; copied fragment is not a live dependency. |
| `SOURCE_TTL_EXPIRED` | Original support expired; no summary may revive it. |
| `ASSOCIATION_UNKNOWN` / `NONADJACENT_SAMPLES` | No reconstructed identity link or continuous motion. |
| `SAMPLES_OMITTED` / `ERROR_CERTIFICATE_MISSING` | No line/path or predicate fidelity guarantee across omitted material. |
| `SOURCE_BINDING_INVALID` / `SOURCE_RETRACTED` | Invalidate source-derived claims and their dependency closure. |
| `VIEW_RETIRED` / `GEOMETRY_REVISION_CHANGED` | Invalidate affected references; no cross-view/reference mixing. |
| `QUERY_NOT_YET_AVAILABLE` | Historical query cannot observe later-received evidence. |
| `INDEX_LOST` / `MISSING_DETAIL_LOST` / `COVERAGE_UNKNOWN` | Missing diagnostics are not complete evidence; conservative flush/downgrade. |
| `IDENTITY_UNVERIFIED` / `PHYSICAL_SIDE_UNKNOWN` | Source/geometry is not identity or physical doorway truth. |
| `PREDICATE_NOT_CERTIFIED` / `REFERENCE_GT_MISSING` | No verified event/negative/safety decision. |

A dependency closure is claim-specific: source samples, association bindings, method version, geometry/view revisions, coverage certificate and query availability. Retraction/expiry propagates before dependent output is exposed. Bounded full-view scans are allowed within scratch/accounting caps; an incomplete index cannot skip propagation. Copies and summaries do not remove a required dependency by renaming it. No online complete reference/slow path is assumed.

## 7. Predicate preservation and one future execution gate

The task predicates of interest are doorway entry/crossing, continuous dwell duration and temporal event order. Their physical/event definitions, doorway/region revision, any dwell threshold, tolerated uncertainty and abstention rules must be independently registered from application requirements **before execution**. Do not choose them from the already-exposed failures. Sample-center transitions are a separate geometry predicate and cannot be scored as physical entry.

For a registered predicate `P`, let `C(S)` be all underlying identity/trajectory/event completions consistent with a stored fragment `S`, its omissions, coverage and defensible bounds. A definitive positive/negative output requires `P` to have the same value on **every** completion in `C(S)`, with source/identity/side validity also established. Otherwise return unknown with reasons. A geometric deviation certificate alone is insufficient; an unbounded identity/temporal gap generally leaves contradictory completions. No proof is supplied here. Returning unknown for everything is safe abstention but cannot pass the usefulness gate.

The next phase is one fixed candidate-versus-baseline trial, only after all prerequisites below are obtained and independently hashed/frozen. It must distinguish (a) mathematical/semantic predicate preservation, (b) storage mechanism behavior, and (c) actual event/GT agreement. Success in one does not substitute for another. Existing public cache results and all prior synthetic/protection failures are exposed development data, not an independent validation set. They can be replayed for reproducibility after a later authorization, never relabeled held out.

### Required inputs before any implementation or experiment

- Existing original G420 video/full frames and original-byte SHA-256 / camera / PTS / time-base mapping; no recreated image or JSON hash substituted for original provenance.
- Source clock/receipt/availability trace, sampling/drop/late-arrival coverage and independently justified input-rate, burst, target/concurrency and lateness limits if a positive service/retention guarantee is contemplated. Without these, keep only an upper TTL and abstention contract.
- Independent human association/event/doorway/side annotations with disagreement/uncertainty and genuine negatives; AI correspondence is not manual truth. Enough continuous evidence to determine dwell/order, not merely nine sparse frames.
- A disjoint, never-used-for-design validation sequence and signed/frozen predicate registry and GT; current exposed failure cases cannot fill this role.
- A deployment byte envelope `B`, allocator accounting, message/concurrency limits and explicit utility/fairness loss budget from application requirements. An unspecified or retrospectively selected budget blocks the run.
- A predicate-preservation argument for what this three-sample/missing representation can answer, including omissions, dependency invalidation and overflow. If no nontrivial task predicate can be certified, **stop before implementation**, rather than add another summary family.

These materials are currently missing/unverified. The previously observed Library image-bearing materialization failure is not repaired by this document. Permissions for private boxes/HSV/GT publication remain separate. No user credential, model, install or media acquisition is authorized by this protocol document.

### Frozen pass/fail/stop rules

**Pass to evaluation, not deployment:** all admission budgets and prerequisites frozen; complete physical-byte ledger within identical `B`; source/TTL/dependency/unknown invariants demonstrated; a nontrivial registered predicate-preservation argument established; then one disjoint validation run reports task-relevant useful valid evidence, other-goal harms and independent GT/uncertainty. Claim no deployment or recognition gain until that evidence exists and its uncertainty permits the specific claim.

**Immediate fail:** any source-time renewal, expired/mismatched evidence presented as valid, omitted gap converted into identity/history, false definitive predicate/safe-negative from a missing dependency, uncharged/index/scratch/queue memory, or budget breach. Abort, preserve failure receipt and stop this candidate; do not repair the failed result by weakening a predicate or enlarging a pool.

**Stop without promotion:** cannot prove predicate preservation; all relevant outputs remain unknown; only loss is shifted to other goals; apparent benefit is delayed deletion rather than additional *valid, useful* query availability; fairness/byte/coverage preconditions cannot be supplied; or exposed old examples are the only available evaluation data. If useful gain and harm are both observed, judge only under the independently frozen utility/loss budget, never hide harmed targets behind an aggregate. No numeric acceptance threshold is fabricated while that application budget is missing.

There is no iteration/tuning exception here. A failing or blocked result closes the candidate and records why. Further research design would require a new explicit task, not automatic expansion of this protocol. The current 52 repository failures and two collection errors remain separately documented; no assertion/fixture/GT replacement is implied.

## 8. This document's actual completion evidence

Original source passages were read and applicability/inference separated. Local document/manifest checks and branch synchronization only; **zero new experiment/test runs, implementations, model or paid API calls or installs**. Existing experiment files and default policies are unchanged. Preregister this single contract on the research branch and retain hashes/source-access receipt. Mac storage is pending through the separate persistence task; this document does not claim host access or saving there.
