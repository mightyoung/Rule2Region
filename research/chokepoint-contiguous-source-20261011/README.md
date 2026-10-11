# Restore a bounded original-index segment; event labels still pending

Baseline `3068fe1e41903d7d73e9e5effd0bbad6ae9b426d`. This continues the same research, using the already acquired/licensed C1 archive. No data/model download, new model forward, training, tracking, paid call, new dependency, Library retry or small-model/fusion substitution occurs. All four prior DINO originals/raw outputs remain immutable.

## Actual recovery

The nested original archive SHA-256 is reverified as `23bb285929056603e387fc6898d56c0a5cdfa64ebd717e8060682d13afd36201`. Original member inventory gives 26 contiguous filename runs; the first relevant run is 112–346. Gaps 100–111 precede it and 347–455 follow it. Neither gap is bridged; background frame0/99 is not spliced onto the new segment.

Recover **original indices 112–332 inclusive: 221 byte-preserved JPEGs**, each 800×600, all decoded and individually hashed. This is a bounded interval anchored by the already viewed background frame112 and newly viewed post-disappearance background frame332, enclosing the previously exposed 142/262/312 candidates. It is the smallest interval for these two selected review anchors, not a GT-derived optimum or a complete source-video recovery. The model receives no new frames. `recover_original_span.py` uses exact member-name whitelisting and writes regular file bytes, never archive path extraction or interpolation. Private per-file manifest and execution receipts retain actual source names/hashes.

## What the selected source supports

Sampled original viewing supports selecting a **visible-cycle candidate**: background, a body appearing near the rear opening, nearer body visibility, near-frame passage/disappearance, then background. This is a clip-selection rationale, not verified association, directional event output or GT. At selected intermediate frames there are two visible bodies and overlapping/truncated views. Independent dense review may resolve some association; none is assumed from matching query IDs or faces.

The near-camera framing remains the Portal1-source development target, with uncalibrated pixel projection and physical sides unverified. The rear opening remains separate. The visible cycle has a terminal **disappearance from view**, not observed physical retreat behind the camera. Accordingly a fully physical approach–cross–retreat event is **not established** by this camera/source; physical event truth remains null/unknown. Do not advertise the restored segment as verified entry or infer completion solely from the publisher E name. This is an evidence ceiling to document, not a reason to abandon candidate-plus-composite research.

## File order versus source time

Known: 221 consecutive original filename indices and decoded originals, with no absent member index inside this interval; source publisher describes nominal 30fps and 800×600 imagery. Unknown: native PTS, native time base, exact capture timestamps/actual per-frame FPS and whether capture originally dropped frames before packaging. Thus **filename-axis completeness is verified; acquisition/time continuity is not**. Only frame indices are used; per-frame seconds/clip duration remain null. The archive-extraction wall-clock timing in the receipt is tool runtime, not footage duration. No 30fps playback/video encoding or inferred seconds is created.

## Prepared next review, no invented truth

[ANNOTATION_PLAN.md](ANNOTATION_PLAN.md) defines independent full-span visibility/anonymous within-clip association and physical-event review, with first-pass predictions/AI results hidden. A private JSONL template contains 221 **pending** records; visible objects, tracks and event labels are null, never fabricated empty negatives. Off-image threshold, optional pose/foot/side evidence, occlusion and disappearance ambiguity remain explicit. Centre/overlap membership only triggers possible entry/exit-condition candidates; physical events need qualified trajectory/direction/inside-outside changes or other required support. The threshold is optional per predicate. Time-dependent predicates wait for justified clock evidence; ordinal review uses source index order.

A separate first-pass private bundle contains only all 221 originals, their manifest, pending template, neutral reviewer instructions and original licence/attribution. It excludes the AI selection notes, model outputs and illustrative ROI boundaries; the wider private handoff keeps those contextual artifacts separate. Two additional original samples (342/346, outside the selected interval) were inspected only to check post-visibility context; they do not extend the recovered review span or bridge the next gap.

Source/images and the neutral frame-index contact sheet remain private with the original [ChokePoint licence/attribution](../chokepoint-source-admission-20261011/README.md). Public outputs are code, source integrity metadata, recovery summary and annotation plan, not images/portraits/person tracks. Existing Library HTTP401 is not retried or bypassed; no new delivery ID or Mac receipt is claimed. User review, human GT and a disjoint formal confirmation set remain pending; no more synthetic effectiveness trials or model changes are justified by restoration alone.

`REASSESSMENT_GATE.json` records the user's future decision boundary: sufficient consistent observations take direct rule evaluation; only conflict/ambiguity escalates to an optional reassessment model. Missing observations may remain unknown and cannot be manufactured by that model. Current source/event work proceeds without waiting for a new model; no all-frame default invocation, training, GT-definition change or model selection is introduced.
