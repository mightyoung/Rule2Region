# C1 discussion images and source-gap admission

Baseline `5b79179660422c6976768f3e190482e90c73554d`, same Rule2Region research trajectory. The existing environment and already admitted P1E_S1_C1 source are reused. No software/model install, new download, detector/model call, face matching or event classifier was added. Source/license/version and original-frame hashes remain in the [source admission](../chokepoint-source-admission-20261011/README.md). No raw image, thumbnail, portrait or AI person-box data is published here.

Four actual 800×600 originals (indices 0, 142, 262, 312) were placed without resizing/cropping on a Chinese private discussion sheet. Original JPEG bytes remain unchanged. Separate projected near-camera framing and rear interior-door candidates, illustrative person boxes and missing-threshold markings are added derivative overlays. Every proposal is AI visual illustration, not detector output or independent human GT. Two visible people are retained in frame 262; no identity is assigned. The near-camera lower threshold is partly outside the image; the rear door cannot silently substitute for Portal1. Neither a projected polygon nor a bbox bottom proxy proves physical entry/exit.

The figure was actually saved to private Library; private receipts retain its returned identifiers. The figure, original licence HTML, full attribution, private proposal JSON and source-gap receipt remain together in the private handoff. This public record grants no portrait publication rights and claims no Mac delivery. Mac persistence remains pending.

## Executed checks and outcome

The unchanged frozen cache passed all 64 tests again. `replay_public.py` passed with 62 observations, 62 geometry records, 531 relations and 1 engineering-hypothesis motion record, with no new model calls. Its four arms retain unknown decisions and null recognition metrics. This is public-fixture arithmetic reproduction; it does not recover historical pixel provenance or event truth.

`check_private_inputs.py` reuses `roi_core.polygon_validation` and `ContinuousCache.validate_geometry` against the private AI projection proposals: **8/8 geometry/schema checks pass**. The clock `0` is only a schema probe, not a recovered native timestamp; no observation/association is ingested into the cache. `sill=None` preserves the missing physical threshold. Successful polygon validation establishes neither semantic scene calibration nor GT.

Five original-axis pairs are checked in `RECEIPT.json`. For 99→112, original members 100–111 are absent. Pairs 112→122, 142→262 and 262→312 also cannot be treated as adjacent source-frame observations, even where intermediate source members exist. The 98→99 pair is source-adjacent, but native timestamp and verified identity evidence are still missing. All five remain ineligible for continuous motion and event decisions remain unknown. Sampling omissions and actual absent archive members are distinct; no renumbering, interpolated frames or fabricated PTS is used.

The frozen cache's `_frame_no` checks adjacency of **ingested frames**, not completeness of the original camera axis. Therefore the source-axis gap check is a prerequisite before offering any continuous-history interpretation; dense-looking ingestion does not repair missing source frames. This script is a development admission receipt, not a production continuity guard implementation. Tracking/direction remains unknown here because no qualified association is supplied; any future direction change is only one piece of evidence. Composite event assessment is ambiguity-gated; unknown never means safe.

## Evidence channels and current direction

A complete physical threshold is **not a necessary condition for the whole research**. Only predicates that specifically depend on that threshold receive a missing-threshold result. Other evidence can contribute to a future ambiguity-gated composite assessment, subject to its own provenance and limitations:

| Evidence | Available in this development batch | Limit |
| --- | --- | --- |
| Projected doorway/frame location | Two AI candidate projections; schema/geometry valid | Intended portal and semantic calibration unconfirmed |
| Trajectory direction / approach / retreat | Preserve original member axis and selected endpoints | No qualified association; gaps/sparse selection cannot become continuous movement |
| Body posture / foot-point visibility | Full-frame body silhouettes and some visible feet available for discussion | No pose estimator, verified foot annotations or physical-side calibration; truncation/occlusion varies by frame |
| Scene history | Already extracted 0–99 background candidates and later original samples | Only selected images visually reviewed; publisher background claim is not independent event-negative GT |
| Physical threshold | Near-camera lower boundary partly outside the image | Only threshold-dependent predicates unknown; do not veto unrelated evidence |

Same-RGB geometry, pose and direction features share source dependencies; they must not automatically receive independent-source weights. D-S remains a **possible ambiguity-reassessment ablation**, being separately investigated by the coordinating thread; no fusion implementation, mass assignments, effectiveness claim, dependency installation or mainline replacement occurs here. Related research can share provenance/missingness/dependency governance structures, but cannot share experimental truth or silently combine evaluation sets.

Next independent work: define the intended portal and the specific predicates, identify which evidence each predicate requires, retain missingness per predicate, and obtain independent event annotations before formal effectiveness measurement. Recover native time/association evidence for predicates requiring continuous motion. Threshold review applies to threshold-dependent predicates; it is not a blanket research admission requirement. These four exposed discussion frames cannot become a held-out test or human GT by default adoption. Existing second-batch/HSV publication approvals remain pending; no such material is added.
