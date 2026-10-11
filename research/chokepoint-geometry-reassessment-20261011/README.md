# Bounded reassessment of immutable actual DINO outputs

Baseline `ebfa9c0f5048c32ec2b8315999c74faa568dc11d`. This reads the [four real frozen-model results](../chokepoint-actual-inference-20261011/README.md) and separate earlier AI illustrations. No model forward, download, expansion, threshold tuning, Library retry or D-S fusion is performed. Original model files and original image bytes remain unchanged. Person boxes/scores, AI boxes, ROI coordinates and all images stay private; this package publishes only the adapter and non-identifying summary/provenance.

## Actual bounded verification

Original JPEG SHA-256 and dimensions match both the actual model receipt and the separate AI discussion record. For every thresholded query, sigmoid-max confidence and original-image xyxy coordinates reconstructed from the **raw model safetensors** match the model receipt exactly. This confirms output-to-source binding and raw-to-box arithmetic; it does not rerun inference or establish human correctness.

Four real boxes were compared with four earlier AI illustrative boxes by frame-local maximum-IoU assignment, with a fixed 0.5 illustration-alignment cutoff. All four pair and no illustration/model item is unmatched; this is **not** proof of no detector miss/false positive, an accuracy score, a verified identity or tracking. The detailed edge-coordinate differences and local illustration IoUs remain in the private difference file. No AI box replaces a model box. Frame 262 has two positively overlapping boxes; frame 262's foreground box and frame 312's box have bottom proxies near the image boundary. These are review candidates for overlap/truncation, not verified foot points or detector errors. Miss/false-positive rates remain null without independent human GT.

## ROI sensitivity, with semantic uncertainty retained

Reuse the existing exact point-in-polygon/geometry implementation. Two separate candidate ROI families originate from the earlier AI illustration: rear interior doorway and near-camera visible framing. Both are **image projections**, with physical sill missing. The closed lower edge of the near-camera polygon is a mathematical closure, not a observed threshold. They cannot substitute for each other as the business target portal.

A fixed horizontal jamb sweep `[-16,-8,0,8,16]` pixels (outset to inset; y unchanged) is frozen before computing membership. These are transparent development sensitivity probes at 1%/2% of image width; they are **not a measured calibration tolerance, human-verified reasonable physical ROI set, or a statistical confidence interval**. Human plausibility review is pending. No boundary is selected to improve a detector outcome. All ten polygon variants pass the unchanged geometry validator. Four boxes × two families × five variants = 40 centre-membership evaluations.

| Candidate projection | Stable inside across these five variants | Stable outside | Sensitive / unknown |
| --- | ---: | ---: | ---: |
| Rear interior doorway | 0 | 3 | 1 |
| Near-camera framing | 4 | 0 | 0 |

The sensitive item is **frame 142**: the model box centre is inside the original rear-door candidate, but outside when both horizontal sides are inset by 16 pixels. Its membership is therefore sensitive/unknown within this declared sweep. Four stable near-frame memberships do not prove that four bodies physically passed that near doorway: distant bodies can project inside the same image region. Seven stable box/family memberships are only finite-sweep geometric facts; all eight box/family semantic/physical-event interpretations remain unknown/unassessed. No event classifier is called.

The threshold is one evidence channel. Missing threshold evidence affects threshold-dependent predicates, not every possible composite assessment. However this batch lacks qualified trajectory associations, calibrated physical-side/foot evidence and human event truth, so neither stable centres nor high detector confidence support a physical entry/exit conclusion.

## Source gaps and shared dependencies

| Selected pair | Absent original archive indices between endpoints | Existing intermediate members not selected |
| --- | ---: | ---: |
| 0→142 | 12 | 129 |
| 142→262 | 0 | 119 |
| 262→312 | 0 | 49 |

All selected endpoint pairs are nonadjacent on the original axis. Distinguish truly absent original members from sampling omissions. Native PTS is unrecovered; frame-local query IDs are not identities. No continuous direction, approach/retreat, three-point trajectory or entry/exit is reconstructed from these four images. `PROVENANCE_GROUPS.json` separates original source, actual detector, independent-method AI ROI, frame-axis inventory and missing GT. The first three share the same C1 RGB camera dependency; separate feature extraction methods do not imply independent evidence. Detector confidence is not calibrated event probability or a D-S mass assignment. D-S remains a separately investigated ablation candidate and is not added to the mainline.

## One concrete review item and stopping point

**Review frame 142: does the business rule refer to the rear interior door or the near-camera Portal1?** A private side-by-side image keeps the earlier purple AI illustration separate from the green real model box, and shows the rear-door boundary sensitivity. Confirming the intended portal is a scope/ROI review, not a person/event GT label. Library delivery remains blocked by the previously observed HTTP 401; it is not retried, bypassed or publicly shared. The private handoff includes original licence/attribution and hashes; Mac receipt is not claimed.

After this bounded result, wait for the intended-door/ROI review and independent physical-side/event annotations before assessing rule correctness. Additional continuous media preparation needs its own defined scope. No further synthetic effectiveness experiments or unsupported recognition-gain claims are justified by this batch.

The adapter reads a local actual-model attempt, original images, separate AI proposals and the original archive axis, and writes to a **new** private output directory. It preserves all raw inputs and emits a reproducible aggregate `PUBLIC_SUMMARY.json`; private geometry detail is not automatically publishable. No production input guard or default policy is changed.
