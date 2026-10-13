# C32 — design-review fixes, part 2

Implemented in `branch_eval.py`, `biology_addressing.py`, `two_hop.py`, and
`exploration.py`, with shared safeguards in `review_eval.py`. `e2e_fsa.py` was
not edited. No `git commit` command was run. Existing working-tree changes were preserved.

- Branch preservation uses batch size **1** for both base and patched NLL.
  Benchmarks route each question through the arm's locked selector, holding that
  route for all answer choices. All-bank results are separately labeled
  **resident-bank stress test** and do not enter routed damage gates.
- C/I/DG capture the same zero-based **full-block inputs** and print the
  MoE-to-block mapping. Layers and L2/threshold settings are selected on VAL;
  only locked selections reach TEST. I reports single-bank and multibank modes
  separately, with separate multibank composition validation wording. Defaults
  search the configured I/DG layer plus available blocks 4, 26, and 28.
  A fixed attribute is excluded from fitting and selection; shared-attribute and
  held-out-attribute TEST routing accuracy are reported separately.
- G, CONJ, AC composition, and exploration use explicit person/value binding
  scores. Swaps, contradictions, and negated assertions fail. Presence scores
  remain diagnostics. G also runs frozen-base supplied-evidence composition;
  AC includes cross-person G items and reports absolute oracle success, oracle
  gap, available headroom, and per-task decisions. A 95%/100% comparison reports
  insufficient headroom; baselines within two points of the oracle report
  **ceiling reached**.
- DG compares against dense independent logistic regression with identical
  examples, same-attribute hard-negative weighting, BCE, LBFGS steps, L2 search,
  and activation budget. Both selectors are selected on VAL and report complete
  tensor/owner address bytes. Its comparison uses the locked dense control.
- I/INH use a separate, deduplicated sample of nonoverlapping general-text
  queries, disjoint from TRAIN anchor and reserved validation queries. Defaults
  request **3,000**; no resampling pads short datasets. Exact one-sided 95%
  Clopper–Pearson bounds are reported; zero events in 3,000 queries gives an
  upper bound about **0.000998**. Insufficient samples suppress rate conclusions.
  INH's tenfold reduction must be supported by the uncertainty bounds.
- CONJ requires retained recall (baseline minus at most one point), nonzero
  recall, binding-aware composition, and clean damage on either success branch.
  SLOT's dense candidate logistic resolver receives identical first/last,
  role-cue, and named-irrelevant TRAIN examples and exactly the same number of
  trainable bytes. Each selects its layer/L2 on VAL. Named irrelevant conversations
  receive real routing and NLL checks, including conversations in ordinary panels.
- Two-hop reports pointer/oracle-bank, **no-pointer + oracle-bank**, and explicit
  resolved-name controls; held-out attributes and source/attribute combinations;
  and raw two-read beam versus cleaned beam. Theta/margin are selected on VAL.
  Adapter buffers/parameters, canonical codes, selector state, bank residency,
  and implementation bytes are reported. Active pointer/raw-beam/cleanup/cleaned-beam
  damage uses matched batches; hooks and patches must revoke to exact base logits.
  Failed resolved-name preflight stops cascade training.
- All requested arms expose `--positive-control` (biological arms through
  `branch_eval.py`). Runtime solo-bank/empty-bank, supplied-evidence, and
  individually recalled fact-pair controls additionally guard applicable arms.
  Failed controls print **control-failed** and suppress verdicts. Exploration's
  actual zero-patch test exposed cancellation of base logits in sign contrast:
  decoding now retains the base logits and charges the extra reference forward.

CPU commands (cached Python environment with torch/numpy/pytest):

```bash
C32_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C32_PYTHON" -m pytest -q experiments/end_to_end/tests
OMP_NUM_THREADS=1 "$C32_PYTHON" experiments/end_to_end/branch_eval.py \
  --arms B,C,D,E,F,G,I,AC,DG,INH,CONJ,SLOT --positive-control
OMP_NUM_THREADS=1 "$C32_PYTHON" experiments/end_to_end/two_hop.py --positive-control
OMP_NUM_THREADS=1 "$C32_PYTHON" experiments/end_to_end/exploration.py --arm all --positive-control
git diff --check
```

Validation: **787 passed, 1 skipped** in the full end-to-end CPU suite. All standalone controls passed.
The suite exercises branch serving, validation locking, routing, exact bounds,
binding negatives, matched resolvers, raw beams, zero writes, and revocation.

Full-model experiments were not run. They require the external checkpoint and
trained banks. `--negative-text additional_tokens.json` can supply a larger
independent general-text corpus when preservation files are too short; a single
source needs at least 213,472 tokens before deduplication to provide the reserved
validation and 3,000 test windows. Window-level binomial uncertainty assumes
query independence; document-level correlation is not modeled. Binding scoring
is conservative and can undercount implicit or abbreviated answers. SLOT's
ambiguity criterion remains unmeasured. These are pilot recipes, not multi-seed
scientific conclusions.
