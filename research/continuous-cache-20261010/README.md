# Continuous cache and actual detection data

This public package contains actual bounding-box and geometry data, portable research code and synthetic tests. It is an exposed-development offline reference implementation, not production deployment or accuracy evidence.

## Verify

From `cache/`, run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v` and `PYTHONDONTWRITEBYTECODE=1 python3 replay_public.py`. Python standard library only, no models, installation or network.

The replay hashes the bundled public bbox, geometry and relation JSON files themselves. These hashes identify public fixtures, **not original images or private source artifacts**. It regenerates outputs from these public fixtures and compares numerical values and feature availability. Original-pixel provenance mapping and original-source audit receipts are not published, so this is not a full reproduction of model inference or historical source verification. New public results are not claimed byte-identical to historical outputs containing private provenance.

## Contents and limits

- `cache/bbox_fixture.json`: six frames, 62 actual observations
- `cache/relation_fixture.json`: 531 candidate pairs, 1 accepted engineering hypothesis, 530 unknown; not verified person identities
- `cache/geometry_fixture.json`: two door polygons and an AI-visible sill fragment; not human-confirmed physical boundaries
- `cache/public_descriptors.json`: 62 geometry records and 1 motion record regenerated from public fixtures
- `cache/public_four_arm_results.json`: feature availability and unknown decisions for every observation
- `detection/`: five actual detector results with threshold boxes, NMS decisions and scores; 47 kept observations. Nine-frame inventory has 83 observations, not 83 people
- `ai-visual-diagnostic/`: separately labelled posthoc AI correspondence; never fed into automatic replay

All 62 automatic event decisions remain unknown in each of four arms. One motion segment is available where an engineering association was accepted. Direction continuity, three-point distance shape and automatic side evidence are insufficient. Ground-truth-dependent accuracy remains null; unknown is neither safe nor correct exclusion. Human business-scope decisions do not establish person/box/geometry/event GT for this package. No human GT is inferred from AI reference, default adoption or software tests.

Earliest sampled center trigger for the visually associated candidate is 12.5 seconds; the first continuous-time trigger is unknown. Adjacent half-second bbox-center directions vary about -0.7 then -61.9 degrees, so these are not instantaneous body direction or sufficient exit conditions. Thresholds were not tuned on this case.

The 64 synthetic unit tests validate implementation contracts. Public replay validates actual fixture arithmetic and feature availability, not model accuracy or identity. Dataset source and attribution are in `DATA_SOURCE.json` and `MEVA-data-license.txt`. No project-code license is introduced. Public-file byte hashes are in `PUBLIC_MANIFEST.json`; no original private-source hashes or internal source-artifact mapping are included.
