# Research handoff v1

This directory exchanges reviewed text and artifact identities across research environments. It does not grant data access, publication rights, network exceptions, or execution authority.

- Pin source repository and commit; independently verify artifact SHA-256 and byte length before use. A metadata entry is not proof that bytes arrived.
- One named task owner writes each task record. Changes use reviewable commits/PRs; immutable execution receipts belong under receipts/<attempt_id>.json. Never overwrite earlier failures or frozen research evidence.
- Separate claimed, executed and independently verified states. Null command/exit/output means no execution receipt is supplied here, not successful execution.
- Record schema/coordinate/time conventions in artifact provenance or linked public contracts. Preserve split, origin, transformations and exposure. AI-assisted human review is not independent blind ground truth.
- Public records must omit private conversation IDs, private asset links, host paths, credentials, raw media and unapproved datasets. A private artifact hash does not publish its contents or authorize fetching it.
- Do not use repository transfers to evade denied network access. LFS and Release availability must be verified separately; neither is required by this protocol.
- No project or third-party license is created or changed here. Unknown redistribution rights remain unknown.
- This initial batch contains status snapshots and metadata only, except the explicitly identified synthetic-smoke text inputs in PCG. Publication approval is not execution approval.

## Current research snapshot (2026-10-10)

[Continuous cache and actual detection data](../../research/continuous-cache-20261010/README.md): portable code, 64 synthetic tests, actual boxes and numerical public-fixture replay. This supplements historical metadata-only snapshots. Hashes bind public fixture files, not original images; original-pixel provenance mapping is not distributed. AI correspondence and human review scope remain separate; no accuracy claim.

Earlier entries below are historical snapshots:

## Latest progress snapshots

- Current mainline: [Complete reference draft, first 2 of 16 frames](reports/full-reference-draft-batch01-20261010.md). Full-image AI drafts and a second AI visual check are available for two frames; 14 ordinary items and two mandatory box decisions are separated. Remaining frames and M0 geometry are open. Default-adopted AI labels are not human GT.
- Earlier review: [Five-image partial class/scope decisions](reports/partial-human-review-five-20261010.md); [fixed-view memory smoke checks and unresolved admission gates](reports/view-memory-rule-smoke-review-20261010.md). Memory rule expansion is paused while mainline annotation proceeds.

- 2026-10-10: [Boundary-16 progress report](reports/progress-20261010-boundary16.md), [schema-valid task status](tasks/status-20261010-boundary16.json), [sanitized evidence index](evidence/boundary-16-20261010.json), and [documentation validation receipt](receipts/docs-progress-20261010.json). AI prelabels executed; human GT/geometry gates remain open.
- 2026-10-09: [Initial reviewed status](tasks/status-20261009.json), retained unchanged.

Progress records on this review branch do not imply a merge to main or permission to distribute the underlying restricted artifacts.

