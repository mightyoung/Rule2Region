# Approved scoring implementation plan

Goal: implement the five user-approved recommendations from
`research/prompt-return-acceptance-20261013/SCORING_DECISION_RECOMMENDATIONS_20261021.md`.
User has authorized proceeding; execute in this session, then independently review.

Architecture: isolated stdlib reference scorer with exact rational geometry and
min-cost maximum bipartite flow; frozen baseline pairs feed four count ledgers.
Only explicitly synthetic fixtures may execute. Real admission stays blocked.
Frozen legacy protocols and target_metrics are unchanged.

- [x] Write and observe RED tests for exact IoU, polygon validity, one-to-one
  objective, immutable pre-route matching, four denominators, negatives, ignore,
  invalid input, source split isolation, input layers and CLI failure closure.
- [x] Implement `scoring_contract.py`: strict decimal JSON, exact geometry,
  `maximum_matching(edges)`, `evaluate(payload)`, exclusive private-output CLI.
- [x] Independently compare matching against exhaustive small-graph oracle.
- [x] Record approval provenance without inventing source timestamp or GT;
  freeze versioned input/output semantics and provide synthetic usage fixture.
- [x] Re-run acceptance and scoring suites, attempt full repository tests,
  record exact commands/outputs/failures; independent review and resolve feedback.
- [ ] Commit and push existing task branch, verify remote SHA; no merge/deploy.

Review focus: duplicate IDs versus duplicate boxes; overlapping ignore and real
GT priority; nonconvex ROI/touch-only geometry; transitive lineage split leakage;
blocked units must never silently disappear from aggregate denominators.
