# C26 / harness v44

Implemented `branch_eval.py --arms AC` with fixed patch values: all banks applied,
dense cosine top-1 routing at full-block inputs 26/28, gold entity routing, and
gold entity routing with the C13 two-hop question rewritten to name the person.
Both full and partitioned patch families are evaluated when supplied. All arms
report bank selection accuracy, C21 pronoun QA, strict/lenient two-fact scoring,
single-fact GEN, and two-hop recall/conditional success. Routing reads the final
question token, before the answer cue. Oracle rewriting preserves that cue.

Conditional success uses a separate first-hop entity-generation question with
the same selected banks. Its denominator is the number of correct, clean entity
answers; zero gives JSON `null`. This observable probe does not measure hidden
reasoning inside the original generation. The named-entity intervention supplies
the first hop. `ADDR-CEILING` includes counts and oracle gains; the loud
`VALUE-USE-BOTTLENECK` line appears when neither oracle arm improves two-fact
lenient accuracy or unconditional two-hop recall by at least .10. Empty panels
do not trigger that decision.

`--arms DG` uses a seeded int8 ternary projection, default expansion 8, density
1/3, and the largest `round(.015*m)` coordinates (at least one) as binary codes.
The center is the training anchor mean. Each bank has an independent logistic
key fitted on `TRAIN_T` prompts, all other-person negatives, twice-weighted
same-attribute impostors, and general-text anchor positions. Positive/negative
mass is balanced per bank; selection requires score **strictly > 0**. Neither
held-out QA nor damage panels enter fitting. Model and patch values stay fixed.

DG reports routed composed recall, selection accuracy, false activations on
unrelated prompts and three damage panels, and routed ΔNLL. Dense cosine top-1
and threshold controls use the same layer/features (`DG:dense_*`); the existing
address-unit individual/composed diagnostics are included when patch ownership
metadata is available. Every sparse bank resides in host memory; only the
selected merged patch is applied in each forward. A decision from the first 64
tokens is held for each unrelated-text damage window, matching arm I.

`e2e_fsa.py --address-ceiling --load-patch ... --steps 0` supports a single global
patch: as trained and named-entity first-hop intervention. Entity bank selection
and isolated oracle routing are explicitly unavailable, with null address
accuracy. It rejects wake/repair/training and per-person patches, writes only the
evaluation JSON, and checks exact revocation. Use the same saved configuration
flags as ordinary `--load-patch` evaluation. All existing defaults are unchanged.

CPU validation, run from the repository root:

```bash
OMP_NUM_THREADS=1 experiments/oscillation/.venv/bin/python -m pytest -q --tb=short --show-capture=no experiments/end_to_end/tests
git diff --check
```

Result: **649 passed, 1 skipped** (11 new C26 tests). Coverage includes sparse
code cardinality, deterministic projection/fitting without RNG changes, multiple
bank activation/null selection, question-token capture, oracle rewriting,
conditional denominators, the exact ten-point threshold, both patch families,
TRAIN-only fitting, unchanged source files, global eval-only writes, and bit-exact
revocation. No CUDA/model evaluation or git commit was performed.

Exact model-host commands (Bash; set paths to existing artifacts and disjoint
token JSON panels, with at least 6,080 anchor tokens):

```bash
: "${PATCH_ROOT:?Directory containing the six R6_v4 patch directories}"
: "${MODEL_EXPORT:?Export path matching saved patch metadata}"
: "${MODEL_STEP:?Export step matching saved patch metadata}"
: "${ANCHOR:?Training anchor token JSON}"
: "${WIKITEXT:?Disjoint WikiText token JSON}"
: "${FINEWEB:?Disjoint FineWeb token JSON}"
: "${CODE:?Disjoint code token JSON}"
banks=(
  --patches
  "$PATCH_ROOT/R6_v4_full_p0/final.pt"
  "$PATCH_ROOT/R6_v4_full_p1/final.pt"
  "$PATCH_ROOT/R6_v4_full_p2/final.pt"
  --partition-patches
  "$PATCH_ROOT/R6_v4_part_p0/final.pt"
  "$PATCH_ROOT/R6_v4_part_p1/final.pt"
  "$PATCH_ROOT/R6_v4_part_p2/final.pt"
  --export "$MODEL_EXPORT" --export-step "$MODEL_STEP"
  --anchor "$ANCHOR" --eval-set "$WIKITEXT"
  --fineweb "$FINEWEB" --code "$CODE"
  --merge sum_clip --fast-moe exact --gen-bs 1
)
python experiments/end_to_end/branch_eval.py "${banks[@]}" \
  --arms AC --ac-indexer-layers 26,28 --out out/c26_ac.json
for layer in 26 4; do
  python experiments/end_to_end/branch_eval.py "${banks[@]}" \
    --arms DG,I --dg-layer "$layer" --indexer-layer "$layer" \
    --dg-expansion 8 --dg-density 0.3333333333333333 \
    --dg-sparsity 0.015 --dg-seed 0 --dg-steps 200 \
    --out "out/c26_dg_i_L${layer}.json"
done
```

For an existing global patch, append `--address-ceiling --load-patch
"$GLOBAL_PATCH" --steps 0 --out out/c26_global_ac.json` to its ordinary reload
command, retaining its saved model/world/layer/scale flags and omitting wake and
repair steps. No bank decomposition is inferred from a global patch.
