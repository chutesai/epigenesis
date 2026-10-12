# C27: biology-inspired addressing prototypes

Implemented in `biology_addressing.py`, exposed through `branch_eval.py --arms
INH,CONJ,SLOT`, and extended `two_hop.py`. Existing defaults and C23/C26 entry
points remain available. Fact patches are read-only; only CPU selectors and the
existing temporary pointer adapter are fitted. No git commit was run.

Each requested subarm reports **address correctness**, **conditional value
correctness** (clean generation given the exact correct bank set, JSON null for
an empty denominator), **composition**, and **end-to-end** results separately.
Branch arms also report all three preservation panels, false activation events,
active patch bytes, address/scratch storage, selector time, and evaluation time.
Selector timing distinguishes scoring from shared residual extraction; it is
prototype wall time, not an optimized serving benchmark.

| Arm | Implemented controls and mechanism | Decision printed |
| --- | --- | --- |
| INH | I-centered positive keys; int8 ternary logistic exclusions fitted to nearest other-person TRAIN confusions and same-attribute impostors; matched-byte positive-only ternary ensemble; default ten seeded random no-patch distractors; bounded activation budget | `INH-KILL`: ≥10× fewer held-out unrelated-query false activations, with ≤1 point single-fact end-to-end recall loss |
| Pointer | Raw cascade, canonical cleanup, cleanup + two-item beam, oracle canonical pointer | `POINTER-CLEANUP-KILL`: ≥10 points held-out two-hop gain or half a positive oracle gap |
| SLOT | Up to four recent name-mention residual slots at L4; rank-at-most-eight late projection fitted by logistic loss on TRAIN same-gender pronouns; L26/L28 I, last-mentioned, and oracle controls | `SLOT-KILL`: ≥10 points pronoun QA over the strongest non-oracle baseline; ambiguous-error criterion explicitly unmeasured |
| CONJ | Entity logistic gate × OR of attribute-specific relation gates per person bank; additive control uses identical learned parameters and state bytes | `CONJ-KILL`: ≥5 points cross-person composed recall, or ≥2× fewer cross-person substitutions |

INH calibrates bank-specific offsets from TRAIN own/general score midpoints, then
uses the maximum over **all** candidates on TRAIN general queries to choose the
absolute null shift. Selection is strictly above the threshold; the empirical
general TRAIN query FPR is ≤1%. Both controls are calibrated independently on
the same data. Distractors participate in scoring, calibration, budgets and
false-activation counts, but carry no value patch. Zero baseline false events
cannot establish a tenfold reduction. Address bytes count the center, positive
keys, ternary features, bank offsets and null shift for each deployed variant.

Pointer cleanup uses anchor-centered canonical-code cosine, an absolute
threshold and runner-up margin. Confident cleanup replaces the question-token
residual at hop 2; uncertain cleanup abstains. The beam reads both canonical
banks separately and chooses the greater mean token log-probability of each
bank's **own generated continuation**, without scoring the gold answer. Oracle
pointer uses the same residual replacement with the gold target identity.
Cleanup may erase attribute information; oracle rescue and conditional answer
success are therefore reported separately. The pointer prototype still has no
unrelated-text damage evaluation (reported as unavailable).

SLOT uses full-prompt tokenizer offsets for mention positions, causal early
captures, and final-question-token late captures. TRAIN/test conversation
wording is disjoint; first/last targets and doctor/painter role targets are
balanced. QA and address results separate role and recency cues. Exact name
recognition supplies candidate identities; this is an explicit host scratchpad,
not EDA recurrent state. The pilot has two same-gender mentions, familiar
entities, no quotations, long distractors or ambiguous cases. Its kill result
is provisional. With no mention slots, the SLOT preservation control abstains
by construction. CONJ controls whole person patches containing all attributes;
it does not create attribute-specific value compartments.

CPU validation from the repository root:

```bash
OMP_NUM_THREADS=1 experiments/oscillation/.venv/bin/python -m pytest -q --tb=short --show-capture=no experiments/end_to_end/tests
git diff --check
```

Result: **684 passed, 1 skipped**; ten new C27 tests cover query-event FPR and
precision, distractor/budget accounting, deterministic TRAIN-only fitting,
matched state, opposing gate decisions, mention offsets/slot bounds, learned
resolution, cleanup uncertainty, generated-answer beam scoring, failure cleanup,
kill-rule boundaries, read-only patches, and exact branch revocation. No CUDA
or real-model evaluation was performed; empirical kill-rule outcomes remain
unmeasured.

Model-host commands (Bash; use existing C26-compatible artifacts and disjoint
training/preservation token JSONs, anchor ≥6,080 tokens):

```bash
: "${PATCH_ROOT:?Existing per-person patch directory}"
: "${MODEL_EXPORT:?Saved export path}"
: "${MODEL_STEP:?Saved export step}"
: "${ANCHOR:?TRAIN anchor JSON}"
: "${WIKITEXT:?Preservation WikiText JSON}"
: "${FINEWEB:?Preservation FineWeb JSON}"
: "${CODE:?Preservation code JSON}"
full=(--patches
  "$PATCH_ROOT/R6_v4_full_p0/final.pt"
  "$PATCH_ROOT/R6_v4_full_p1/final.pt"
  "$PATCH_ROOT/R6_v4_full_p2/final.pt")
partition=(--partition-patches
  "$PATCH_ROOT/R6_v4_part_p0/final.pt"
  "$PATCH_ROOT/R6_v4_part_p1/final.pt"
  "$PATCH_ROOT/R6_v4_part_p2/final.pt")
common=(--export "$MODEL_EXPORT" --export-step "$MODEL_STEP"
  --anchor "$ANCHOR" --eval-set "$WIKITEXT" --fineweb "$FINEWEB" --code "$CODE"
  --merge sum_clip --fast-moe exact --gen-bs 1)
python experiments/end_to_end/branch_eval.py "${full[@]}" "${partition[@]}" "${common[@]}" \
  --arms INH,CONJ,SLOT --bio-layer 26 --bio-steps 100 --bio-budget 3 \
  --inh-exclusions 2 --inh-distractors 10 --inh-lambda 1 \
  --slot-early-layer 4 --slot-late-layers 26,28 --slot-max 4 --out out/c27_biology.json
python experiments/end_to_end/two_hop.py "${full[@]}" \
  --export "$MODEL_EXPORT" --export-step "$MODEL_STEP" --anchor "$ANCHOR" \
  --hop1-layer 8 --hop2-layer 12 --steps 100 --rank 8 \
  --cleanup-theta .7 --cleanup-margin .1 --out out/c27_pointer.json
```

The pointer command requires value-bank full blocks at or after hop 2, as
validated by the existing prototype. In branch JSON, read
`arms["INH:full"].subarms`, `arms["CONJ:full"].subarms` and
`arms["SLOT:full"].subarms` (also `:partition` when supplied); in pointer JSON,
read `cleanup.subarms`. Keep both patch-family results separate.
