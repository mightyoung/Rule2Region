# Public cache qualification recovery — 2026-10-11

This continues the existing continuous-cache research at baseline `74f6ad2b40afe8dda44f327037253957daaea8a0`. It uses only bytes already published in `../continuous-cache-20261010/`. No new media, HSV data, private second-batch material, AI visual correspondences, model execution, installation or paid API calls are used.

## Executed recovery

Python 3.12.14, standard library only. The existing 64 tests pass. `replay_public.py` matches all three existing derived public JSON objects exactly: 62 geometry descriptors, one motion descriptor, 531 candidate relations and four arms with 62 unknown decisions each. This verifies the public numerical replay; historical private provenance and model inference cannot be reproduced without original pixels/mappings. The historical claim of 2,432 original-versus-public scalar comparisons is not newly verified here.

The workspace initially contained the source snapshot branch, rather than the research branch. The task branch was fetched, checked out and its baseline matched to the remote. An initial switch to a remote tracking ref failed because the fetched branch had no tracking ref; switching from verified FETCH_HEAD recovered it without modifying main. Local `.agents` is empty and no repository AGENTS.md or skill files are present. The old private research directory and raw images/HSV have not been recovered.

## Next discriminating validation

`qualification.py` evaluates exact endpoint consistency and offline availability for the already-public nine-frame inventory. IDs alone are insufficient: original PTS/time-base representation, sample/camera naming, exact bbox and explicit frozen-prediction source declaration must also match. Descriptors must bind those endpoints, the public bbox JSON hash, geometry revision and geometry/relation source hash. Every candidate relation remains unknown unless it was already an accepted engineering hypothesis. Duplicate observation IDs fail closed.

Results (`qualification_results.json`):

| Check | Count |
|---|---:|
| Nine-frame observations (not people) | 83 |
| Old geometry eligible at observation time | 36 |
| Observations lacking old endpoints | 47 |
| Old candidate relations inspected | 531 |
| Both endpoints qualify within new population | 243 |
| Candidate relations outside this population | 288 |
| Existing accepted endpoint hypothesis | 1 |
| Accepted adjacent links in half-second grid | 0 |
| Event status unknown | 83 |

The retained hypothesis is P360:query003 → P390:query003. Its one-second bbox-center chord is −83.120979° and bottom-center proxy chord −84.549579°. P375 is an intervening sampled frame with no accepted link. These are endpoint directions, not instantaneous body direction, verified identity, or proof of crossing. They cannot be split into half-second links or used to claim direction continuity. No AI visual path is loaded or substituted. The 243 candidate pairs are not 243 accepted links; 242 remain unknown.

The observation matrix records trajectory/direction status unconditionally, and the accepted motion record retains both anchor vectors. No ambiguous business target was supplied, so there are zero composite queries. Geometry reuse describes cached fixture arithmetic; it does not establish human physical boundaries or original-pixel provenance. The standalone time check rejects future evidence; it is an offline eligibility audit, not a live cache TTL/capacity simulation or retrieval endpoint.

## Reproduce

From the repository root:

```sh
(cd research/continuous-cache-20261010/cache && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -v && PYTHONDONTWRITEBYTECODE=1 python3 replay_public.py)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/cache-qualification-20261011 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/cache-qualification-20261011/qualification.py
```

The last command recomputes and compares the checked-in qualification JSON; only explicit `--generate` writes that output. Thirteen tests cover source/time/bbox rebinding, duplicate IDs, future availability, unknown associations and sparse-chord versus dense-continuity separation. The relation-source counterexample failed during development, then passed after adding the missing source binding. No existing cache behavior or frozen data was changed.

`VALIDATION.json` records executed commands, exits and output hashes, including repository-wide `python3 -m pytest -q` failing to start with `No module named pytest`. No claim is made that the repository-wide suite passed. `MANIFEST.json` binds the new files and input public bytes; the old public package and manifest remain unchanged.

## Continuing evidence gaps

Original pixels, source mapping, HSV, automatic side evidence, human person/geometry/event GT and confirmed complete negative events remain missing. No real parallel pass or pre-crossing return negative has been confirmed; turning back after exiting cannot substitute. AI prelabels and default adoption remain distinct from human GT. All event decisions stay unknown, unknown does not mean safe, and accuracy remains null. Direction change is one piece of evidence. No recognition gain, independent generalization or new model-inference reproduction is claimed.

Publication authorization for the separate private second batch and HSV remains pending. The inventory of 83 public observations was already published at the verified baseline; this audit derives only from that existing public file and does not republish any unapproved private material.
