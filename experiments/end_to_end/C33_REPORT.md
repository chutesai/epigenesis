# C33 — hierarchical attribute banks

Implemented deterministic, balanced contiguous groups of each person's sorted
attributes. `--train-person P --train-attr-group G --attr-groups K` scopes the
training pool, validation and audit; full-world TEST remains available alongside
`test_gen_bank`. Patch metadata and reload checks include both group options.
Partitions support `people*K` parts with bank ID `P*K+G`, including whole units.

`branch_eval.py --arms H,FLAT --patches ...` accepts a complete bank grid in any
file order. H uses centered training centroids at two full-block inputs, thresholds
at both levels, and a global beam ranked by summed `log((cosine+1)/2)` affinities.
These scores are not calibrated probabilities. FLAT has the same activation cap.
Layers and thresholds are selected using VAL accuracy minus VAL null activation,
then locked. An attribute per bank is excluded from selector fitting/selection
when another training attribute remains. Reports include exact bank-set accuracy,
true-bank recall, shared/held-out attribute accuracy, binding-aware composition,
resident bytes, and actual key comparisons per query. For 3×4 banks and top-k1=2,
H compares 3–11 keys depending on abstention; FLAT compares 12. Capturing residuals,
merging patches and the separately labeled child-null probes are outside this cost.

Damage and optional benchmarks/subject text use each locked routing policy. Base
and patched preservation NLL both use batch size 1. Independent general queries
exclude TRAIN anchor, preservation and validation; exact one-sided 95% binomial
bounds are reported, with conclusions suppressed below 3,000 available queries.
Window independence is an assumption. Empty-bank, standalone-bank and exact
revocation checks run through real serving. Failed controls suppress verdicts.

CPU validation: **800 passed, 1 skipped** in the complete end-to-end suite.
The C33 integration test trains all **12** tiny CPU banks, reloads one, rejects a
group mismatch, and evaluates H/FLAT with 3,000 distinct synthetic text windows.
It also changes TEST prompts without changing the validation-selected settings,
and checks short-corpus/failed-control suppression. Synthetic separable keys give
**100% selection** and null abstention at both levels. Tiny-model recall is not
evidence about the external language model. Existing working-tree edits were
preserved; no commit was run.

```bash
C33_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C33_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests
OMP_NUM_THREADS=1 "$C33_PYTHON" experiments/end_to_end/branch_eval.py --arms H,FLAT --positive-control
git diff --check
```

Full-model recipe below was **not run**: this workspace lacks the external
checkpoint, `/workspace/malleable/mlf` loader and GPU runtime. Set the token JSON
paths to distinct corpora (anchor ≥6,080 tokens; validation/final evaluation needs
the trainer's reserved text ranges). The negative corpus should provide at least
3,000 distinct 63-token windows after the preservation and 64-window VAL reserves
(≥213,472 tokens in a single source before deduplication). Budget/optimizer settings
below are pilot settings. Training layer IDs are MoE ordinals; selection layer IDs
are full-block inputs, with their mapping printed by the evaluator.

```bash
C33_ANCHOR=/path/to/anchor_tokens.json
C33_EVAL=/path/to/wikitext_tokens.json
C33_FINEWEB=/path/to/fineweb_tokens.json
C33_CODE=/path/to/code_tokens.json
C33_NULLS=/path/to/independent_general_tokens.json
C33_RUNS=results/c33
mkdir -p "$C33_RUNS"
C33_PATCHES=()
for C33_PERSON in 0 1 2; do
  for C33_GROUP in 0 1 2 3; do
    C33_PART=$((C33_PERSON * 4 + C33_GROUP))
    C33_BANK="$C33_RUNS/p${C33_PERSON}_g${C33_GROUP}"
    "$C33_PYTHON" experiments/end_to_end/e2e_fsa.py \
      --method fsa_forced --fmt ternary --layers 12,14,17,20 \
      --seed 0 --people 3 --facts-per-person 16 --shared-attrs 16 --values disjoint \
      --train-person "$C33_PERSON" --train-attr-group "$C33_GROUP" --attr-groups 4 \
      --slot-partition 12 --slot-part "$C33_PART" --partition-unit \
      --sparse-state --slot-frac 0.5 --patch-scale row --budget 20000 \
      --optim bop --bop-tau 0.7 --steps 300 --answer-only --aug --aug-all \
      --anchor "$C33_ANCHOR" --local-lambda 10 --eval-set "$C33_EVAL" \
      --eval-every 25 --train-audit 128 --patch-dir "$C33_BANK" \
      --out "${C33_BANK}.json"
    C33_PATCHES+=("$C33_BANK/final.pt")
  done
done
"$C33_PYTHON" experiments/end_to_end/branch_eval.py \
  --arms H,FLAT --patches "${C33_PATCHES[@]}" \
  --h-layer1 4 --h-layers2 12,14,17,20,26,28 \
  --h-top-k1 2 --h-top-k2 2 --h-beam 4 \
  --indexer-layers 4,12,14,17,20,26,28 \
  --anchor "$C33_ANCHOR" --eval-set "$C33_EVAL" \
  --fineweb "$C33_FINEWEB" --code "$C33_CODE" \
  --negative-text "$C33_NULLS" --negative-n 3000 --out "$C33_RUNS/h_vs_flat.json"
```
