# Nine actual bbox frames: continuous state and conditional reads

This is an executed next step on the existing Rule2Region research, using only the already-public nine-frame inventory at source commit `1e434a82ed163d1c6579efe6eed3c29cdbc67d58`. Unlike the preceding join/coverage audits, it actually ingests all 83 observations through the existing `ContinuousCache`, continuously computes geometry, enumerates adjacent unknown candidates and conditionally retrieves four cached feature views. No frozen source data, original cache implementation, threshold or gate is changed.

## Actual result

| Measure | No requests | Geometry-flagged requests |
|---|---:|---:|
| Nine-frame observations | 83 | 83 |
| Continuously computed geometry records | 83 | 83 |
| Qualified motion records | 0 | 0 |
| Feature retrievals | 0 | 3 |
| Observation capacity evictions | 6 | 6 |
| Observation expirations | 18 | 18 |
| Live observations at source time 16 s | 59 | 59 |
| Event status unknown | 83 | 83 |
| New model / judgment-engine calls | 0 / 0 | 0 / 0 |

The 47 observations from the five additional published detector records now receive actual cached geometry, rather than just an eligibility check. The remaining 36 observations bind the exact older bbox fixture. Every observation receives a trace/direction status, even without a request; unsupported trajectory/direction remains unknown.

The conditional caller rule is precisely: **bbox intersects a particular door polygon while its center is outside that same polygon**. It flags three observations: P360:query001, P360:query002 and P375:query001. These are caller-authored geometry disagreements, not human-confirmed event ambiguity, an exhaustive uncertainty detector or tuned crossing threshold. Targets without this flag still remain event unknown; they are not declared safe or nonambiguous in a business sense. The flag decides only whether to retrieve existing features.

For each flagged observation, G, G_DIR, G_DIST and G_DIR_DIST retain unknown. Automatic physical-side evidence and qualified history are absent; there is no direction continuity or three-associated-point distance shape. These views expose existing cache inputs and gaps. **No event judgment engine is instantiated, and this trial does not validate a composite classifier.** Direction change remains one evidence item; it cannot be a standalone exclusion or safety condition.

## A concrete sampling effect

The original public accepted endpoint hypothesis P360:query003 → P390:query003 is ingested unchanged. In the denser frame sequence P375 intervenes, so the existing cache records `motion_status=unknown_motion`, `motion_missing_inputs=[nonadjacent_frames]`. It emits no motion descriptor. The old six-frame replay still produces its original one sparse chord; it is preserved as a separate version. The new zero is a sampling/qualification result, not recognition improvement or proof the person did not move.

Seven hundred and one adjacent cross-product candidate pairs are recorded as **unscored unknown**; they are not seven hundred and one associations. No AI visual identity path, HSV, side answer or historical private nine-frame association summary is imported. An accepted old endpoint hypothesis cannot silently fill the new half-second gaps.

## Bounds actually reached

Default capacities remain 64 observations, 512 relations and 128 descriptors per view, with a three-second source-time TTL. At P450 the 64-observation and 512-relation bounds are reached. Four observations are evicted then, another two at P465; six expire at P465 and another twelve at P480. Six evictions + eighteen expirations + fifty-nine live observations account for all 83 inputs. Eviction also removes dependent records. The denser inventory therefore demonstrates a real retention limit under unchanged defaults; no capacity was increased to preserve a desired chain.

The no-request and conditional branches have identical traces and checkpoints. Each read is checked against cache observations, relations, descriptors and view clocks before/after, so it neither recomputes nor renews expiry. Source-time delivery is simulated at each PTS; this is not a wall-clock/live-camera benchmark. Exported trace is a bounded nine-frame offline receipt, not an unbounded production history store.

## Reproduce

From the repository root, Python standard library only:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/nine-frame-conditional-replay-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/nine-frame-conditional-replay-20261011/conditional_replay.py
```

Seven tests check actual population/geometry computation, only flagged retrievals, read immutability, dense-grid motion qualification, genuine capacity/expiry behavior, unknown evidence separation and JSON roundtrip stability. The replay recalculates both branches and compares `conditional_results.json`; only explicit `--generate` writes it.

A real development failure is retained: the first repeat replay differed because legacy relation endpoint keys were tuples in memory and arrays after JSON loading. Recursive comparison found only those four type differences across the two branches, and JSON-normalized values matched exactly. A failing roundtrip test was added; converting only public result endpoint keys to lists fixed it. No frozen values or old assertions were changed.

## Material recovery pursued in parallel

Authorized text records could be read again from the user's materials, confirming the existing nine-frame and consumed five-forward scope. The image-bearing report candidate was identified, but its image bytes were not obtained: large HTML reads/find failed, and the official file transfer route returned an authorization error. No credentials were changed or workaround attempted. Exact historical-source title searches found no matches for the five queried source/protocol names. The 24 historical first-blocker paths are still separate from the nine-frame pixel/GT requirements.

No report's AI correspondence or side labels enter this replay. No original PNG, HSV, original-source hash mapping or independent human GT has been restored. Precise private material identifiers for coordination are retained outside git; they are not public links or upload permission. The minimum remaining input is the **existing nine original full frames and their original-byte hash/PTS/time-base index** for G420, not another model run or recreated image. No new scene or acquisition is proposed.

New model API calls, paid calls, installation, weight/media downloads and private bbox/HSV publication are zero for this step. Existing software validation uses the already-created isolated environment. All event decisions remain unknown, accuracy stays null, no real complete negative is confirmed and unknown does not mean safe. Full repository failures remain retained; see `VALIDATION.json` for actual coverage and unchanged failure sets.

## Persistence and environment

Computation remains in the cloud workspace (Python 3.12.14; the previously authorized isolated validation environment). Public source, summaries and receipts are saved to the existing task branch and identified by git SHA plus per-file SHA-256/size manifest. Main is unchanged. Saving does not trigger extra experiments.

A separate private transfer manifest outside this public repository records the readable text snapshots and image-report recovery candidate. Those snapshots preserve read text and native source/version identifiers; they are not a materialization of the original file bytes. No image-bearing file was recovered. Mac storage is requested but **not yet verified or completed by this cloud task**. The cloud has no direct access to the Mac. Public git transfer and private authorized Library retrieval must be completed there into a separate persistence directory after checking actual repository paths and worktree state, without overwriting tracked data or local user changes. No private material identifier or Mac host path is published here.
