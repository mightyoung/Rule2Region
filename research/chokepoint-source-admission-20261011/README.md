# ChokePoint P1E_S1_C1: original-source admission only

Baseline `7b7a6a148980fb8e78c295109915a1f3bd1e27cb`. This is an **independent development source**, not G420 recovery or independent event validation. No candidate evidence cache is implemented/expanded and no recognition experiment is run.

## Access and licence gate actually passed

The [official ChokePoint page](https://arma.sourceforge.net/chokepoint/) links the author-published [Zenodo release 815657](https://zenodo.org/records/815657), DOI `10.5281/zenodo.815657`, displayed release v1, published 2017-06-22 and modified 2020-01-24. The API version field is null; the record/file/checksum, rather than an invented API version value, pins acquisition.

Public metadata, selected-file download and original licence-page retention all returned HTTP 200. No login, explicit consent/terms-acceptance interface or permission rejection was encountered. API access is `open` and its licence identifier is `cc-by-nc-4.0`. The official page additionally requires noncommercial use, NICTA/publication attribution, retention of the original licence notice and clear notices on altered material/derivatives. Use here is limited to noncommercial research. Original official licence HTML is retained alongside the private source media; it is not replaced by this paraphrase. This does not grant commercial use or silently waive either source's conditions.

Required attribution: NICTA; Y. Wong, S. Chen, S. Mau, C. Sanderson and B. C. Lovell, *Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved Video-based Face Recognition*, IEEE CVPR Workshops, pp. 81–88, 2011, DOI [10.1109/CVPRW.2011.5981881](https://doi.org/10.1109/CVPRW.2011.5981881). No face-identification/matching task or authors' verification protocol is adopted.

## Exactly what was acquired

Only **original `P1E_S1.tar.xz`**, 385,315,324 bytes, was downloaded. Publisher MD5 `195191772856c1a62650a1d47e897b2a` matches. Local SHA-256 is `3e310249afa86309175a13a04eb3e5cbea60f38e4018f9e4b8bd3fd80e39fcba`. Workspace disk precheck had 32,016,711,680 free bytes, exceeding package size plus a 1 GiB working allowance. The existing environment/tools were used; no installs or model/paid calls.

The publisher packages three nested camera archives together; that unavoidable original package was not substituted by a cropped-face file. Only nested C1 was extracted (125,805,716 bytes, SHA-256 `23bb285929056603e387fc6898d56c0a5cdfa64ebd717e8060682d13afd36201`). C2/C3 payloads remain only inside the original package, not separately extracted. No leaving-direction, P2/crowded, cropped-face or annotation/GT archive was downloaded. Metadata and licence HTML are small supporting source records, not additional media packages.

C1 contains 2,292 JPEG members on original filename indices 0–4409, with 25 gaps / 2,118 absent indices; the first gap is 100–111. It is **not contiguous video**. Preserve member names; never renumber retained files to fabricate continuity. Publisher nominal 30 fps is not recovered native PTS/time-base metadata. A filename-derived nominal axis would be an explicitly declared interpretation, not original timestamp evidence.

Extracted byte-identical originals: 100 background-candidate images with indices exactly 0–99, plus seven unlabelled admission samples 112, 122, 142, 172, 212, 262 and 312. All 107 decoded as JPEG 800×600 in the existing Pillow environment and match their stored byte hashes. This is raw fixed-camera imagery, not the separate cropped-face package. No pixels were edited, enhanced, resized, cropped or converted. Original archive, nested C1, selected JPEGs and licence are stored outside git; this public package contains no image, thumbnail or archive.

## What direct viewing supports and does not support

The assistant directly viewed full-frame originals 0, 50, 99, 112, 142, 172, 212, 262 and 312; no detector, face matcher or biometric identity inference was used. The three viewed background frames have no visible people. The publisher describes the first 100 as background frames; **the whole block has not been independently manually annotated as negative event truth**.

Visible fixed framing/corridor geometry and a rear interior door opening with human passage support annotation feasibility for projected doorway/corridor geometry. But the foreground portal's lower threshold is partly outside the image, the designated portal versus the rear interior door must be defined explicitly, and near-camera truncation/occlusion prevents direct complete physical-side/threshold truth. No polygon, bbox, person-identity, crossing timestamp or entry label is created in this gate.

Frame 262 visibly contains two people. Therefore the publisher's single-subject overview is not a per-frame occupancy guarantee. This observation is a scene-admission warning, not an identity assignment or a formal counting benchmark. Do not hide the second person by face cropping or substitute a different scene to obtain a convenient negative.

Admission outcome: **original-source/noncommercial-development gate PASS; projected-geometry use CONDITIONAL; formal physical-entry/event/identity validation NOT ADMITTED**. G420 source/provenance and independent human GT remain unrecovered. The ChokePoint publication's face/eye annotations (not acquired here) cannot automatically become Rule2Region doorway-event truth. This exposed admission sample is development material, not a held-out confirmation set.

## Receipts, immutability and persistence

`SOURCE.json` pins the original release, licence/access evidence, exact selected asset size/checksums and the aggregate visual outcome. `ASSET_DIGESTS.json` gives original selected-frame members/hashes without pixels, crop data or person identifiers. `RECEIPT.json` records the actual scope and old-file integrity. No default strategy, previous experiment output or original 52 repository failures/two collection errors is changed; no new software test/recognition experiment was run for this acquisition gate.

The private transfer manifest includes the actual cloud media paths, source/version and sizes/hashes, original licence and private frame-axis inventory for authorised Mac persistence. It does not overwrite the ongoing persistence task's manifest or claim that a cloud path exists on Mac. **Mac transfer remains pending.** Source media and images are never staged/pushed to GitHub. Retain the original licence with any authorised local copies; this task provides no further publication permission for images/private labels.
