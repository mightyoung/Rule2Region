# Result schema v1 and private return contract

`handoff.py validate` is the executable strict validator; unsuccessful runs never become successful public exports. `RUN_PRIVATE.json` contains schema_version=1, kind (`real_prompt_probe` or `real_cpu_smoke`), git_commit, device, manifest_hashes, environment, status, decision=unknown, human_GT=false, UTC times, parent PID, trials. Each trial has source_id, arm, actual argv, child PID, start/end UTC and monotonic, exit_code, status, artifact-name -> SHA256 map. Failed runs keep failure_type/reason and partial artifacts; no auto retry/resume or silent CPU fallback. Ctrl-C leaves failed/incomplete status, never success.

Each successful trial directory `<source_id>--<arm>/` contains:

- `SOURCE_PRIVATE.json`: exact source_sha256, not invented GT.
- `DOOR_RAW_OUTPUT.safetensors`: logits float32 [1,900,256], pred_boxes float32 [1,900,4], input_ids/attention_mask int64. Raw normalized cxcywh boxes unchanged.
- `PROCESSED_INPUT.safetensors`: actual pixel_values, pixel_mask, input_ids, attention_mask. Original RGB full frame, slow PIL processor shortest800/longest1333, no ROI masks.
- `DOOR_PROBE_RECEIPT_PRIVATE.json`: fixed prompt/model/source/thresholds, pre-NMS and post-NMS query IDs/scores/text/continuous original-image xyxy boxes, NMS decisions; AI pending human, S/B/events/trajectory/direction unknown, human_GT=false, accuracy_metrics=null.
- `AI_PENDING_HUMAN.png`: red raw candidate boxes, explicitly AI-only; not a manual annotation.
- `stdout.log`, `stderr.log`: actual child output; do not reconstruct after a failed call.

Validation verifies all artifact hashes, input/model/arm/device metadata, fixed thresholds, 900-query tensor shapes/finiteness (normal negative-infinity logits permitted only on padded text columns), raw thresholded query IDs against candidate rows, complete 9-trial matrix (1 for smoke), no duplicate/missing arms, no GT/decision escalation. It verifies internal consistency, not a cryptographic proof of honest execution or anatomical/physical truth. No event calibration, ROI IoU/AP or recognition accuracy is computed.

`PUBLIC_AGGREGATE.json` contains only schema_version, kind, device, git_commit, decision, human_GT, accuracy_metrics, source_manifest_sha256, model_revision, completed_forward_calls, limitations and trials. Each trial contains source_id, arm, kept_count, image_edge_touch_or_overrun_count, forward_seconds, peak_cuda_bytes. Edge audit uses raw bbox touching/overrunning x0/y0/x800/y600; it does not establish a physical boundary. GPU times are synchronized, no warmup, peak allocated bytes measured (not reserved or total card usage).

Raw/private result directory must stay outside any actual git checkout. Return its intact archive using the already-authorized private Site upload or direct private attachment, preserving original source licence and attribution; do not place private link/token/URI in public JSON. Do not expose media via GitHub/LFS/releases/public issues. If private upload cannot accept archive, split by camera or individual files, retain a size/hash manifest. Never change site access or bypass an authorization denial.

Only the validator-generated `PUBLIC_AGGREGATE.json` can be copied to a new return branch for public git review. Failure messages/argv may contain local paths: send those privately, and publish only a manually reviewed failure-type/status summary with decision unknown. Do not force-push or update main.
