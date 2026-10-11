# Complete reference annotation: first two-frame AI draft

Status (2026-10-10): 2 of the original 16 frames have a complete full-raster AI reference draft and an additional AI visual check. This is not completion of the 16-frame annotation gate, nor human-certified exhaustive ground truth. The main Stage A goal remains correct rule regions and reduced irrelevant outside-region targets, with in-region recall loss reported alongside suppression.

## What was actually produced

D06-S05 and D06-S09 were inspected as complete original 1280-by-720 images, including wall prints, desk and occlusion zones, the full window band, cabinet and foreground. The draft contains two visible live-person instances in each frame and six relevant nonlive photo/print-carrier regions in each frame. Carrier regions are not person boxes and do not count the people depicted inside photographs. A wall collage absent from the frozen detector candidates was added to the negative-carrier inventory; this is not a missed live-person detection.

All 18 existing detector candidates across the two frames are linked to a live-person draft or nonlive carrier, but the full-frame scan was not limited to detector suggestions. No additional definite live person was found by the second AI visual check. This is an AI completeness judgment, not independent human-certified absence.

Four visible-body box proposals and two box-edge review windows were authored from actual pixels. The seated-person proposal in S09 excludes the foreground standing person's hand, which expanded the old detector box. The foreground person's clothing/chair boundary in S05 remains uncertain. Hidden bodies were not completed by imagination.

## Review split and evidence scope

- 14 ordinary items: two clearer live-person boxes and twelve nonlive carrier regions, operationally default-adopted as AI drafts unless corrected.
- Two mandatory decisions: one occluded visible-body box per frame. These are field-specific geometry questions; already-confirmed person classes need not be asked again.
- Prior candidate-level class and scope decisions retain their exact scope. The S09 nonlive candidates are not silently inherited as human confirmations from S05.
- All newly authored exact boxes remain human-unconfirmed. Default adoption and a second AI check do not become human GT or blind GT.

The local deliverable includes per-frame structured labels, separate ordinary/mandatory queues, full-image overlays, original-pixel detail views, and a self-contained review document. Original media, coordinate-level private annotation files and conversation records are not published here.

## Current gate and next work

The other 14 frames still require the same full-frame draft process. M0 geometry and exhaustive human/reference adjudication remain open. The next batch prioritizes the C office view's interior/exterior cases; the existing ROI, detector thresholds, predictions and earlier receipts are preserved. No new model inference or acquisition was performed for this annotation batch.

Formal complete-GT metrics remain null. Any future descriptive comparison against ordinary default-adopted AI reference labels must identify that source explicitly and cannot be presented as human-reviewed accuracy. The memory subtopic's rule expansion is paused while this mainline annotation gap is addressed.

## Retained history

- [Five-image partial review](partial-human-review-five-20261010.md)
- [Memory smoke review and admission gaps](view-memory-rule-smoke-review-20261010.md)
- [Frozen detector batch](progress-20261010-boundary16.md)
