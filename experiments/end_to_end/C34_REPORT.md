# C34 — decisive evaluation suite

Implemented opt-in `--decisive-eval` in `e2e_fsa.py` and `branch_eval.py`, branch
arms `C34` and `PK`, and shared measurements in `decisive_eval.py`. Existing
training/evaluation defaults and existing working-tree edits were preserved.
No commit was run.

- C36 supersedes the original scorer and interpretation claims; see
  [C36_REPORT.md](C36_REPORT.md) for the review-item mapping and validation.
- Original two-fact prompts are evaluated at 40 and 64 new tokens. Primary
  correctness requires two terse values in question order. `presence_stop`
  and `presence` remain explicitly labeled diagnostics; C34 does not use a
  name-dependent binding scorer for its terse conjunction items.
- `CEILINGS {json}` reports `correct`, `presence_stop`, `presence`, cluster
  counts, and supported/unsupported splits. Panels include reverse lookup,
  actual sister chains, within/cross-person conjunctions, single-mention
  pronouns, and cued earlier/later antecedent retrieval with competitors.
- Legacy worlds contain sisters' names but no facts about those sisters. They
  report the chain ceiling as **unavailable**, rather than relabeling an indirect
  question. Opt-in `--relation-world` replaces one fact per person with a sister
  relation to another named world person, preserving counts. The world option is
  saved in metadata, validated on reload, and included in cache identity.
- Evidence controls cover needed-line/sheet order, header presence, newline vs
  sentence style, distractors before/after, and renaming the same 20 distractors
  to unrelated people. Distractors use the original oracle-evidence selection
  recipe. Direct TEST questions are included alongside the four task types.
  Reports retain paired samples, rescues/losses, and rescues where needed-only
  already emitted both values. These distinguish completion/stop changes from
  value changes; they cannot establish a causal explanation of the historical
  73%/88.7% inversion without running the external model.
- Decomposition runs solo, per-layer pinned independent outputs, and merged
  weights separately for each supplied block/unit family, using the same files
  and items. Cross-person conjunctions provide a multi-bank subset even in
  legacy worlds. Pins erase propagation through later hooked experts; solo
  forwards also have their own trajectories and router weights. Thus gaps to
  merged weights cannot isolate pure merge nonlinearity. Alignment uses actual
  dispatch token positions; unmatched rows are counted. Missing dispatch,
  bypassed dispatched Linears, and short output rows raise. Correlated panels
  report counts/deltas without iid SE or within-noise verdicts.
- Negative queries have at least sixteen unique prompts per feasible category
  and bank, including absent attributes, rotated foreign facts and seeded name
  mutations absent from both names and values. Rates always include n. Base
  and forced-bank conditions use identical prompts. Relation values and query
  name echoes are excluded. Emitted-value confidence reports joint probability
  and geometric mean; the latter sets the 0.8 threshold. `asserted_answer`
  separately measures non-refusal answers, including invented unstored values.
  The panel is labeled forced-bank leakage, bypassing selector abstention.
- Recall splits use actual top-k dispatch at answer prediction positions in the
  generated continuation, plus a gold-prefix routing control. Misses without a
  gold value span use a labeled expected-answer-length window. Each item records
  bank selection, nonzero-code experts, token hits, and its recall result.
- PK learns a residual projection into two ternary codebooks of ceil(sqrt(N))
  keys; each bank owns an (i,j) pair. PK/I/DG receive identical TRAIN features,
  held-out attributes, VAL questions/nulls, and activation budgets. Layers,
  regularization, and thresholds are locked before TEST. Exact allocated address
  bytes match through explicitly allocated uint8 reserves; useful bytes and
  reserves are reported separately. This measures equal residency budgets, **not
  equal effective capacity**. Patch bytes and key comparison counts are separate.
- Selected banks retain their stored strength; C34 rejects nonfinite scales and
  scales below 0.5α. Damage and optional benchmarks use each locked serving
  policy, matched batch size 1, and fixed routing across answer choices. The
  all-bank policy is labeled resident. Failed bank/zero controls suppress branch
  verdicts; final revocation must recover bit-exact base logits.

Frozen glossary (also emitted in JSON):

- **two-hop** = question requiring a relation hop through a stored relation fact
  (not the “indirect” probe).
- **bank** = a separately trained, separately revocable patch for one scope.
- **within noise** = not estimated in C34; correlated panels report counts/deltas only.

CPU validation: 22 C34 tests exercise actual tiny-model training/reloading, nine
full/block/unit banks, both entry points, nonlinear residual replay, format
samples, negative confidence, dispatch cleanup, exact revocation, equal allocated
bytes, and TEST-poisoning without changing validation-selected settings.
Standalone C34/PK controls pass. Full CPU suite: **822 passed, 1 skipped**;
`git diff --check` is clean.

```bash
C34_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C34_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests
"$C34_PYTHON" experiments/end_to_end/branch_eval.py --arms C34,PK --positive-control
"$C34_PYTHON" experiments/end_to_end/e2e_fsa.py --method fsa_forced \
  --positive-control --out /private/tmp/c34_positive_control.json
git diff --check
```

Full-model evaluation recipes below were **not run**: this workspace lacks the
external checkpoint, `/workspace/malleable/mlf` loader, and GPU runtime. Use
matching world/export/scales and recorded training budgets for all files. Old
banks without saved training metadata cannot enter C34/PK comparisons. New chain-capable banks must be trained
with `--relation-world`; that flag cannot upgrade existing learned facts.

```bash
# Existing banks: four-way decomposition plus locked I/DG/PK comparisons.
# Full, block and unit directories each contain p0/p1/p2/final.pt.
"$C34_PYTHON" experiments/end_to_end/branch_eval.py \
  --arms C34,PK,I,DG --decisive-eval \
  --patches results/c34/full/p{0,1,2}/final.pt \
  --partition-patches results/c34/block/p{0,1,2}/final.pt \
  --unit-patches results/c34/unit/p{0,1,2}/final.pt \
  --anchor /path/to/anchor_tokens.json --eval-set /path/to/wikitext_tokens.json \
  --fineweb /path/to/fineweb_tokens.json --code /path/to/code_tokens.json \
  --negative-text /path/to/independent_general_tokens.json \
  --out results/c34/evaluation.json

# Add --relation-world to each new bank's normal training command. For example:
"$C34_PYTHON" experiments/end_to_end/e2e_fsa.py \
  --method fsa_forced --fmt ternary --layers 12,14,17,20 --seed 0 \
  --people 3 --facts-per-person 16 --shared-attrs 16 --values disjoint \
  --relation-world --train-person 0 --sparse-state --slot-frac .5 \
  --patch-scale row --budget 20000 --optim bop --bop-tau .7 --steps 300 \
  --answer-only --aug --aug-all --local-lambda 10 \
  --anchor /path/to/anchor_tokens.json --eval-set /path/to/wikitext_tokens.json \
  --patch-dir results/c34/full/p0 --out results/c34/full/p0.json
# Repeat for persons 1/2. Block banks add --slot-partition 3 --slot-part PERSON;
# whole-unit banks additionally add --partition-unit.
# Reload with the same training configuration, --steps 0 --load-patch FILE
# and --decisive-eval to run the single-bank ceilings/negative/routing suite.
```
