# Independent review plan for original C1 indices 112–332

This is a plan, **not annotations or event GT**. The near-camera Portal1-source projected candidate remains the explicit development target; the rear interior door is a separately named scene landmark/control. Neither earlier AI polygons nor detector boxes become physical ROI truth. Centre membership triggers possible entry/exit conditions only. Natural-language/small-model feasibility work does not change the physical-event GT definition here.

## Review material and independence

Provide the 221 original 800×600 JPEGs in original filename order, original-file SHA-256 manifest, original ChokePoint licence/attribution and a plain pending template. Do not show the prior AI boxes/ROI draft, DINO boxes/scores, geometric sensitivity results or suggested event labels during the initial annotation pass. Keep the assistant's selection notes separate. Source E/L filename metadata is not an event label. Source selection/development exposure is disclosed; independent annotations would not turn this exposed clip into a held-out test.

Two authorised human reviewers independently examine **all 221 originals**, seal first-pass records, then reconcile disagreements. This has not been assigned or executed; no reviewer, receipt, agreement or human label is invented. No messages are sent to reviewers. If only one reviewer is available, preserve single-review provenance and do not call it independently double reviewed.

## First pass: visibility and within-clip association

- Review every frame, including apparently empty ones. In the pending template, `visible_objects=null` means unreviewed; it must not be read as zero people or a negative.
- Record visible human bodies and occlusion/truncation, with frame-local annotation IDs and, where support permits, anonymous within-clip track references. Do not record real identities, perform face matching or connect these local references to another dataset/run. Pose/clothing/spatial continuity may support visual association; unresolved alternatives remain explicit.
- Mark partial/full occlusion and appearance/disappearance at the frame boundary. If association is ambiguous, record candidate alternatives and the ambiguity interval or split the tentative track. Never borrow an identity across missing source indices or treat a model's `q000` as a track.
- Record which support for body/foot location and scene side is actually visible, not inferred from bbox bottom or confidence. Geometry, source-frame order and time availability are separate fields.

The first run has two visible bodies in selected frames (including previously admitted frame262) and overlap/edge truncation. Sample viewing makes association a review problem; it does **not** prove that all local association is impossible. Dense neighbouring frames may resolve some alternatives, which requires actual human review. No automatic/dense tracker runs in this step.

## Second pass: physical event, with unknown retained

Reviewers first document the intended near-camera physical portal/side interpretation from observable scene evidence, with uncertainty. Use the existing physical entry/exit definition: crossing from one physical side to the other, where supported; candidate centre/box overlap is not that event. Optional threshold, body/foot, scene-side changes, direction and qualified trajectory evidence may contribute. Missing threshold alone does not veto unrelated evidence; missing a predicate's required support makes that predicate unknown.

For each supported local candidate record frame-index bounds for approaching the scene landmark, visible near-portal passage, and subsequent departure/disappearance. These are **review fields, not supplied phase annotations**. Distinguish visible disappearance from physically observed retreat: the camera cannot see behind itself. Record physical inside/outside-before/after only where observable; if sides/crossing cannot be established, use unknown rather than interpreting visibility loss as entry. A no-event label requires completed review and positive support for no relevant crossing within its declared window; it is not assigned from zero detections.

Use frame-index intervals (including uncertain bounds), not seconds. Native PTS/time base and actual capture FPS are unavailable; nominal publisher 30fps remains metadata only. Event types available to reviewers are entry, exit, no supported event and unknown, under the existing definition. Keep unknown reasons for side calibration, occlusion/association, off-image threshold, incomplete away-phase visibility or time-dependent requirements. No score-derived D-S belief or small-model verdict establishes these labels.

## Adjudication and later validation gates

Retain both original reviews and every disagreement; adjudication is a third record, never an overwrite. Labels remain pending until review receipts identify actual reviewers/method and scope. GT-dependent metrics stay null. Timed events require separately justified clock evidence; ordinal events can remain frame-index based without fabricating FPS.

This clip can support an exposed development association/observability pilot after annotations, subject to visibility. Formal event-effectiveness claims additionally need a disjoint confirmation set, fixed event/composite predicates and declared unknown/error/usefulness metrics. Source restoration alone grants neither more detector calls nor formal event validation. Current delivery is pending (existing Library HTTP401 is not retried); the independent review plan can be prepared now without asking the user to guess an unavailable image.

Future predictions preserve the user-defined routing boundary: direct rules for sufficient consistent evidence, optional on-demand model reassessment for conflict/ambiguity, and unknown where observation support is missing. This does not delay the current independent review or change physical-event GT. Reviewers' labels are not replaced by reassessment predictions; no new model is invoked or trained in preparing this plan.
