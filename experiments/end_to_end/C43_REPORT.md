# C43: C42 review fixes

Implemented the four fixes in `reviews/brainstorm_2026-10-12/tasks/c43_brief.md`.

- Hierarchy rescue kills use **positive-query rescue**, with aggregate and positive
  rates both reported. Missing simulated-100 rescue is unresolved unless the
  beam loses more than two accuracy points against flat, or the real-12 accuracy
  rule already killed hierarchy.
- False-fire certification uses the exact one-sided 95% upper bound at `1e-3`.
  Uncertified arms report `unresolved (power)`; point estimates remain available.
  VAL still tunes empirical cutoffs, and TEST remains locked.
- C42 dumps default to **10,000 general negatives per split**. When windows are
  scarce, each split gets the maximum equal allocation without resampling.
  Requested/actual counts and corpus limitation are saved in the cache, dump
  output, and routing analysis. Other branch-evaluation modes retain 3,000.
- Simulated-100 scaling is explicitly diagnostic and cannot change LLR or
  hierarchy verdicts. Margin-gate predictions use the mutually exclusive union
  sum; evidence-only predictions use the independence product, including the
  real-12 extrapolation.
- Default LLR parents retain beam expansion and report
  `parent_gate: "none (beam top-k)"`. `--c42-parent-gate` enables strict `LLR > 0`
  pre-gating in both the CPU selector CLI and composed GEN, with exact visited
  child/rescue cost accounting.

Validation: full end-to-end suite **1,039 passed, one skipped** (165.83 seconds;
one existing warning), including all 29 C42 tests. Python compilation and
`git diff --check` passed. Tests cover confidence-bound boundaries, null-heavy rescue rates,
missing rescue, diagnostic-only scaling, parent compensation/costs, and disjoint
window fallback. Tiny-model integrations use synthetic residuals; no production
model dump or quality experiment was run. No commit was made.
