# C29: indexed composition and hypothesis preservation

Implemented `indexed_composition.py`, available as `branch_eval.py --arms C29`
and `two_hop.py --c29`. No commit was run; supplied bank files remain read-only.

The first round retrieves IDs through `(relation, value)` inverse postings or a
TRAIN-fitted mention resolver, retaining correlated `(referent, target)` paths.
The second round cleans up to canonical entity codes and reads the selected
value bank. Intersections explicitly return the entire posting-set intersection,
including empty sets; entity names come from the address dictionary. Their
results are scored separately from two-value bank answers. Inverse postings retain
fact provenance and rebuild deterministically after fact removal.

Each retrieval experiment reports indexed evidence plus **sum, log-product,
minimum, learned joint keys, and a learned joint scorer**, crossed with argmax,
real posterior, joint beams of 2/4/8, amplitude beams of 2/4/8, oracle IDs, and
oracle values. Posterior answer selection uses `logsumexp(log p(ID) + bank
sequence log-probability)`. Amplitude arms apply `sqrt(p)` to saved sparse bank
projection contributions in one forward, without requantizing the mixture.
Generated bank continuations define the answer pool; gold answers enter only
TRAIN fitting, metrics, and the explicit oracles. Confidence is normalized over
this candidate pool, not the full LM answer space.

TRAIN/TEST query templates are disjoint. Truly ambiguous pronouns have two
same-gender candidates, identical prompts with balanced hidden labels, and both
direct and sister-path questions. Deterministic subsampling retains these pairs.
`HYPOTHESIS-PASS` prints the ≥5-point gain requirement against the strongest
single-hypothesis system on pronoun/multi-hop QA, with no increase in confident
errors (`p_max >= .8`) on the ambiguous subset. Tied best singles use the smallest
confident-error count. It reports unmeasured when no ambiguous items exist.

**Equal resources are enforced by a common-work suite.** Every readout retains
all learned selectors, bank files, keys, reverse postings, relation edges,
canonical codes, address metadata, posterior tables, and eight-slot scratchpads.
Every query computes every selector and all single-bank and amplitude likelihood
tables, including work an argmax readout ignores. With three banks plus null:
four candidate generations, 16 bank-answer forwards, and 72 amplitude-answer
forwards per query. Each report receives the same full executed budget. Repeated
answer slots consume work but are deduplicated before readout. Logical tensor
bytes and canonical serialized payloads are counted, including reverse postings;
Python/allocator overhead and the common frozen model/parametrization masters are
excluded. A common upper bound for transient mixed projection buffers is charged
as well. This measures inference quality under equal work, **not serving speed**.

Limitations: familiar facts/entities; lexical query fields are supplied by the
deterministic compiler; sister edges are the explicit C23 cyclic scaffold.
Four/eight-hypothesis beams can saturate below capacity in this small world;
actual counts are reported. Worlds with only two attributes use named/pronoun
two-value questions; larger worlds use a third inverse locator fact. Canonical
cleanup requires value-bank layers at or after the cleanup layer. The C29 indexed
round supersedes the learned pointer for these items; the existing C27 pointer
arms still run separately. Branch preservation panels abstain when there are no
mention slots or inverse query fields. No real-model accuracy gain or empirical
pass-bar result is claimed.

CPU validation:

```bash
OMP_NUM_THREADS=1 experiments/oscillation/.venv/bin/python -m pytest -q --tb=short --show-capture=no experiments/end_to_end/tests
git diff --check
```

Result: **695 passed, 1 skipped** (one pre-existing tensor-conversion warning).
Eleven C29 tests cover postings and provenance revocation, disjoint templates,
balanced ambiguity, joint-path legality, posterior math, TRAIN-only fitting,
matched resources, candidate deduplication, pass-bar boundaries, gold-independent
inference, failure cleanup, amplitude projection weighting, and CLI integration
with read-only patch checks and exact logit revocation.

Model-host commands using existing C26-compatible three-person artifacts
(Bash; the chosen full-block cleanup layer must precede all value-bank layers):

```bash
: "${PATCH_ROOT:?Existing three-person patch directory}"
: "${MODEL_EXPORT:?Saved export path}"
: "${MODEL_STEP:?Saved export step}"
: "${ANCHOR:?TRAIN anchor token JSON}"
: "${WIKITEXT:?Preservation WikiText token JSON}"
: "${FINEWEB:?Preservation FineWeb token JSON}"
: "${CODE:?Preservation code token JSON}"
full=(--patches
  "$PATCH_ROOT/R6_v4_full_p0/final.pt"
  "$PATCH_ROOT/R6_v4_full_p1/final.pt"
  "$PATCH_ROOT/R6_v4_full_p2/final.pt")
python experiments/end_to_end/branch_eval.py "${full[@]}" \
  --export "$MODEL_EXPORT" --export-step "$MODEL_STEP" --anchor "$ANCHOR" \
  --eval-set "$WIKITEXT" --fineweb "$FINEWEB" --code "$CODE" \
  --arms C29 --slot-early-layer 8 --slot-late-layers 12 --bio-steps 100 \
  --c29-max-per-kind 24 --gen-max-new 32 --gen-bs 1 --out out/c29_branch.json
python experiments/end_to_end/two_hop.py "${full[@]}" \
  --export "$MODEL_EXPORT" --export-step "$MODEL_STEP" --anchor "$ANCHOR" \
  --hop1-layer 8 --hop2-layer 12 --steps 100 --rank 8 --c29 \
  --c29-max-per-kind 24 --gen-max-new 32 --out out/c29_pointer.json
```

Read `arms["C29:full"].subarms` or `c29.subarms`; each has `by_kind`,
`hypothesis_questions`, `ambiguous`, `resources`, and predictions. Partition
patches can also be supplied to the branch command for a separate family report.
Amplitude mixing supports row-scaled ternary patches without occupied flips.
