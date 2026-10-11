# Portal1-source candidate screening: centre versus rectangle overlap

Baseline `bb57669fb2c9df4e276e94091609b8af252baee1`. The development target is explicitly the **near-camera visible framing in P1E_S1_C1**, named `Portal1-source near-frame development candidate`; the rear interior door remains a separate control region. This is an experiment choice, not a user-confirmed production ROI or official pixel annotation. The retained [official source page](https://arma.sourceforge.net/chokepoint/) defines P/S/C as portal/sequence/camera, supporting the Portal1 source label. The actual original images show near framing, supporting a projected candidate. Neither supplies an official exact pixel boundary or complete physical threshold; those remain unverified rather than being silently asserted. The previously admitted raw source/model/AI provenance is unchanged.

## Meaning and predeclared bounded comparison

**Centre membership only triggers a candidate that may have entry/exit conditions. It is not event adjudication, entry recognition truth or a physical crossing.** Likewise a bbox touching/overlapping a projected ROI is an alternate candidate condition, not behavioral evidence by itself. Sensitive centre membership does not invalidate the composite-evidence research mainline; stable membership does not prove entry. `SEMANTIC_CLARIFICATION.json` preserves the user's clarification separately from the preexecution protocol and numeric outputs.

`PROTOCOL.json` was written before this overlap execution. The same already exposed four frames and immutable actual DINO boxes are used; this is not blinded preregistration. Keep prior AI ROI polygons and the previously declared horizontal jamb insets `[-16,-8,0,8,16]` px, y unchanged. These 1%/2% image-width probes are uncalibrated development perturbations, not measured error bars or a human-confirmed reasonable physical ROI set. No outcome-based boundary/threshold is selected.

For each box and each of ten polygon variants, compare closed-polygon centre membership with unchanged exact-rational `roi_core.route_box`. Overlap has four distinct finite categories: positive full, positive partial, zero-area boundary touch and outside; invalid/uncomputable inputs receive unknown. Positive overlap uses the core's strict geometric area predicate, with no tuned overlap-fraction cutoff. Descriptive overlap fractions do not classify events. The core's legacy `boundary_crossing` field means a **static box straddles an image polygon**; it never means a person moved across a physical doorway.

Forty comparisons per predicate (4 boxes × 2 regions × 5 variants) completed. All original model files remain byte-identical; no model calls, downloads, added frames, corrected boxes, inference thresholds or new packages. Earlier raw-to-score/coordinate verification is reused and corresponding boxes/score/source bindings match exactly. Detailed boxes and geometry remain private.

## Actual candidate-screening result

| Region | Centre condition across five variants | Positive-overlap condition across five variants | Original-boundary rectangle extent |
| --- | --- | --- | --- |
| Near-frame development target | 4 stable candidate triggers; 0 sensitive | 4 stable candidate triggers; 0 sensitive | 3 fully overlapping, 1 partially overlapping |
| Rear-door independent control | 3 stable outside; 1 sensitive | 3 stable positive; 1 sensitive | 4 partially overlapping |

Arithmetic unknowns are zero for these valid finite inputs; **all eight box/region semantic or physical-event interpretations remain unknown/unassessed**. The empty frame produces no detected box and no person candidate; this is not a human-confirmed negative or a safe outcome.

For the near target, frame 312's actual box centre remains inside while its rectangle straddles the left projected jamb in every variant. Its descriptive overlap fraction ranges about 0.8404–0.9676, with 0.9116 at the original boundary. Thus both centre and positive-overlap conditions trigger stably, but centre membership does not describe the whole box's extent. No footprint/body mask or crossing is inferred from bbox background coverage.

For the rear control, the sensitive **centre** item is frame 142, while the sensitive **positive-overlap** item is the foreground box in frame 262. At the original boundary, three centre-outside boxes still have positive partial intersection. This illustrates different geometric candidate coverage, **not verified recovered people, improved recall, detector failure or improved entry recognition**. No business target switches to the rear door to select favorable results.

## Composite evidence remains the next step

The threshold is optional evidence; absence affects only predicates that depend on it. Four sparse frames are not stitched into trajectories. Qualified motion/direction and inside/outside position changes, plus body/foot and history evidence where available, are the future ambiguity-reassessment inputs. Missing required evidence means unknown, not safe/excluded. Same-RGB centre, bbox, model score and AI ROI features are correlated; they are not automatically independent sources or D-S masses. D-S is not added to the mainline.

`FUTURE_EVIDENCE_REQUIREMENTS.json` is a short declaration of missing existing-input fields and per-predicate gates, **not a new framework, tracker, pose estimator or evidence-cache implementation**. The downloaded original C1 archive contains possible continuous spans, but only the fixed four exposed observations are processed here. Entering continuous event validation requires:

1. A declared contiguous original-member segment and complete gap accounting; qualified multi-person association rather than frame-local query IDs. Additional extraction/inference scope is not silently expanded here.
2. An explicit source-axis/time interpretation. Native PTS is not recovered. Publisher nominal 30fps may support a clearly declared filename-derived nominal clock; it must not be renamed recovered camera PTS. Time-dependent predicates stay unknown when their clock requirement is unmet.
3. Defined entry/exit predicates and a physically meaningful inside/outside reference with observable position changes. Threshold/foot/pose evidence is required only where a predicate depends on it; centre-only screening is insufficient for event truth.
4. Independent human annotations for the requested people/associations/physical events, including occlusions and negatives. The author's E filename, face annotations or AI agreement cannot substitute for event GT.
5. A disjoint confirmation split and declared usefulness/error/unknown metrics before claims of recognition gain; these four already exposed frames cannot become held-out truth.

Software/data preparation that these four real outputs support is complete. No further synthetic effectiveness tests or threshold search is justified. The experiment choice lets candidate geometry proceed without requiring the user to guess an undelivered image; formal event validation still waits for the listed evidence. Private visualization labels centre as **candidate trigger only**, with model and ROI provenance separated. Original source licence/attribution accompanies private derivatives. Library HTTP401 is not retried or bypassed; no public portrait/raw output and no Mac-delivery claim.
