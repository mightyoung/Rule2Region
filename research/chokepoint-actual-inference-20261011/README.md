# Actual four-frame Grounding DINO inference

Baseline `6f21b83f719712d855c36227a1044c7eeadeeaf7`. This completes the previously blocked [inference preflight](../chokepoint-inference-preflight-20261011/README.md) without replacing that failure record. Four already admitted ChokePoint C1 originals were processed by the existing research-specific **Grounding DINO tiny person detector**, not by AI illustrative boxes or another repository's YOLO experiment. This is exposed development material, not a formal accuracy/entry-event evaluation.

## Acquisition and frozen configuration

Only the [author's fixed revision](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71) `a2bb814dd30d776dcf7e30523b00659f4f141c71` was acquired: safetensors and eight small README/config/processor/tokenizer files, 690,308,125 bytes total. The author's card identifies Apache-2.0; no new explicit agreement/login/authentication was required. Actual safetensors SHA-256 matches publisher `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`. The duplicate pickle `.bin` was not downloaded. The model uses local-only/offline loading, `trust_remote_code=False` and `use_safetensors=True`.

A new isolated environment reuses the already installed Torch 2.8.0+cpu and existing NumPy/Pillow. Thirteen necessary official PyPI binary wheels, pinned by version and SHA-256 in `requirements-additions.lock`, total 22,896,995 bytes. The planning dry run also downloaded wheels; totals here describe selected install payloads, not all cumulative network traffic. No Torch/CUDA package was downloaded again. All dependencies pass `pip check`. Optional torchvision was unnecessary for the successfully imported slow PIL processor and explicit standalone NMS; it was not installed. Transformers 4.57.1, tokenizers 0.22.2, safetensors 0.6.2 and huggingface-hub 0.35.3 match frozen core versions. Existing NumPy 2.5.4 and Pillow 12.3.0 differ from the historical versions. This is a new actual execution with recorded environment differences, **not byte-exact reproduction of the old detector environment**.

Keep the frozen `person.` prompt, box/text thresholds 0.25, NMS IoU 0.7 (suppress strictly above threshold; score descending then query ID), CPU float32, batch 1, four Torch threads/one interop thread, seed 0, no warmups, no ROI masks, no threshold tuning. Original inputs remain 800×600; observed processor tensor is `[1,3,800,1066]` for all four. No credentials, paid calls or third-party inference services are used. Original media rights are governed separately by the [ChokePoint source admission](../chokepoint-source-admission-20261011/README.md).

## Actual results

| Original frame index | Thresholded person boxes | Boxes retained after NMS | Forward seconds |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 20.3345 |
| 142 | 1 | 1 | 8.1787 |
| 262 | 2 | 2 | 13.3461 |
| 312 | 1 | 1 | 12.8636 |

Four forwards completed, zero runtime errors, zero corrected/added boxes. Import time is 7.6908 seconds; processor/model load is separately 1.1172 seconds. First forward has no warmup; the subsequent three are recorded individually, not claimed as a repeat-based steady-state benchmark. Forward times exclude preprocessing, raw serialization, overlay creation and model loading; complete per-stage timings are in `RECEIPT.json` and private frame receipts. Hardware: Intel Xeon Platinum 8573C, five visible logical CPUs, no CUDA.

Every image's 900-query logits and normalized predicted boxes are retained before thresholding in private JSON and exact float32 safetensors, together with input IDs/masks, thresholded boxes/scores, deterministic NMS decisions and final boxes. JSON encodes masked nonfinite logits as explicit strings; verification reconstructs the exact safetensors tensors. All four original hashes still match prior frozen admission digests. No frame/output was omitted. Raw outputs, individual scores/coordinates and model-only overlays remain private; this public package contains code/configuration/aggregates only.

Green image overlays are the actual model output after frozen thresholding/NMS. No doorway illustration or AI box correction is mixed into them. The earlier AI discussion sheet and its proposed door/person geometry remain separate. Frame-local query IDs such as `q000` are **not identities or tracks**. The person prompt provides no model door output; it does not execute door recognition, pose estimation, physical-side classification or entry/exit inference.

Limited separate AI visual diagnostics retain occlusion/truncation and do not establish a verified miss/false-positive count. Independent human GT, accuracy, recall and precision remain null. Matching a prior AI visual count is not validation. Tracking/direction is unassessed: frame gaps are not repaired and no continuous association is fabricated. Event `unknown` is an explicit unassessed placeholder, not the result of a newly executed event classifier and never a safe decision. Threshold visibility is one evidence channel; only threshold-dependent predicates are blocked by missing threshold evidence. No D-S fusion or evidence-cache strategy is added in this execution.

## Private delivery and repeatability

The Chinese four-frame model-only contact sheet and private results ZIP contain the complete raw outputs, execution receipts/log, model metadata, dependency freeze, separate AI diagnostic and original ChokePoint licence/attribution. Model weights are omitted from the results ZIP but retained locally with their pinned acquisition receipt. Delivery status is recorded factually in `RECEIPT.json`; private manifests retain returned Library identities and file hashes. Mac transfer is still pending. Portraits, source JPEGs, raw outputs and unapproved second-batch/HSV materials are never staged or pushed.

With an authorised local pinned snapshot and originals, run:

```sh
python run_private_inference.py --snapshot LOCAL_MODEL_DIRECTORY \
  --frames ORIGINAL_JPEG_DIRECTORY --output NEW_PRIVATE_OUTPUT_DIRECTORY \
  --configuration ../continuous-cache-20261010/detection/configuration.json
```

Output directories must be new, so previous failures/results cannot be overwritten. Historical 52 repository test failures/two collection errors remain unchanged; this task verifies actual inference and output integrity, not the complete legacy suite.
