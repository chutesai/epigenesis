# C54: S1 entity firing screen

Implemented `reviews/brainstorm_2026-10-12/tasks/c54_brief.md` and the authoritative
S1 in `experiments/epigenesis/designs/SCR_cheap_screens.md`. Existing working-tree
changes were preserved. No `git commit` was run.

## Implementation and interpretation

`census.py --firing-screen` runs the forward-only S1 screen at MoE inventory
indices 12,14,17,20. The ordinary C51 census remains available with the flag off.
The screen reuses its actual dispatch capture, duplicate-route coalescing,
projection-hook checks, padding exclusion, and exception-safe restoration. It
observes the frozen base only. Only `fwhq_sel` contributes FineWeb negatives;
other supplied streams are overlap-audited and do not contribute S1 statistics.
Use a single FineWeb stream for the commands below. With multiple streams,
`--tokens` is divided into per-stream full-context budgets as in C51.

`c54.py` labels exact character spans and half-open token spans using tokenizer
offsets on the actual independently encoded prompt and answer. A token overlapping
a mention is included, including a leading-space BPE token; BOS and unlabelled
words are excluded. Conflicting entity labels on one token fail explicitly.
The tiny character-tokenizer fallback is checked rather than guessed. Names
in relation answers are labelled too, up to the actual teacher-forced input
boundary `(prompt + answer)[:-1]`. The final answer token is not forwarded.

All twelve `TRAIN_AUG_T` templates from C51 are retained. Additional single-person
and recent-person conversation prompts come directly from
`build_address_pronoun_items`, using its train-only templates, mention PIDs and
filler construction. Pronoun spans come from `{pos}`, `{Pos}`, `{subj}`, and
`{Subj}` substitutions and inherit the builder's subject/antecedent PID. Thus
an earlier person's name is labelled as that earlier person while the final
pronoun is labelled as the recent person; no nearest-string heuristic or held-out
QA/VAL/TEST template is used. The existing world has no aliases; the annotation
API accepts a PID-to-alias mapping and tests cover alias boundaries.

The default dev namespace/seed matches C51 (`CensusDev510051 PersonNNNNN`).
`--factqa-relations` adds linked sister relations and renames their answer values
consistently. `--factqa-world arm` instead screens the exact ordinary world that
`e2e_fsa --seed ... --people ... --facts-per-person ... --relation-world` will train.
Dev-world lists should not be reused for different entity names in the small arm.

Per unit and population the screen accumulates routed count, positive-firing
count and fp64 sum of `relu(z)^2`. Statistics report both all-population-token
rates/means and routed-conditional rates/means. AUROC uses all labelled entity
span tokens versus all other-entity span tokens, and separately versus all
FineWeb tokens. Non-dispatched units receive an exact zero score. Unlabelled QA
words never become other-entity negatives. Router weights do not rescale the
hidden-unit firing score. Unrouted experts are included as all-zero units.

Histograms have **32,641 fixed bins**: exact zero and the intervals corresponding
to every positive finite BF16 representation. They are stored as sorted sparse
`unit * 32641 + bin` integer indices with int64 counts, merged between forwards;
no token activations or token-by-unit matrix survives a forward. Histogramming
`relu(z)` preserves AUROC order for `relu(z)^2`. Native BF16 outputs yield exact
pairwise AUROC, including ties, across the entire BF16 dynamic range. Higher
precision values are truncated into the same bins and tagged as inexact; pairs
with uncertain within-bin ordering give rigorous lower/upper AUROC bounds. The
estimate is their midpoint. Threshold counts distinguish estimated, certified
(lower bound >= threshold), and unresolved (lower < threshold <= upper) units.
Uncertainty at 0.9 cannot silently trigger the positive screen decision.
Rank lists and uniform random samples retain only bounded candidate pools.
Memory grows with occupied fixed histogram cells, not the number of tokens.

Entity-v-other AUROC is the authoritative top-unit ranking, ordered by lower
bound, estimate, and stable bank/unit identity. FineWeb also gets an independent
top list. The JSON reports the top 32 units and their three populations' stats,
counts at 0.9/0.8, the number of top-8 units certified at 0.9, and the S1 decision:
`value_only_control`, `train_addresses`, or `unresolved`. The top-8 list always
contains the eight highest ranked measured units if eight are available; it does
not claim that all eight meet 0.9 when fewer do.

## Artifacts and value-only support

For `--out out/C54/dev.npz`:

- `dev.npz`: each expert's mapping, each entity/FineWeb population's raw counts,
  sparse histograms and inexact counts, plus `[estimate, lower, upper]` AUROC arrays
  for both comparisons and JSON metadata. Other-entity raw stats are sums of the
  remaining entity populations.
- `dev.summary.json`: per-entity rankings, statistics, threshold counts and inputs.
- `dev.spans.json`: the exact generated prompts, teacher-forced token IDs, span
  kinds, character/token intervals and PID labels.
- `dev.entity<PID>.top8.units.json` and `.random8.units.json`: eight units per entity
  and a deterministic uniform random eight-unit control from the same layer/unit
  inventory (seed 510051).
- `dev.top8.units.json` and `.random8.units.json`: unions for a single shared
  three-person bank. Both union sizes are reported; shared top units can make the
  top union smaller than 24. Random draws are per entity without replacement.

Unit files use `{"version": 1, "units": {"<MoE name>.experts.<ID>": [<unit>, ...]}}`.
The existing `attach(restrict=...)` already masks up rows/down columns, but no
CLI option accepted an explicit unit-list file. The only C54 harness changes
add `--restrict-units FILE`, its value-only validation/loading and its saved
identity. It requires ordinary FSA and `--down-only`, creates restrictions only
for listed down columns, and uses `attach()` to make every unlisted bank empty.
Base forced/free-zero mask legality remains enforced. Unknown banks, out-of-layer
banks, duplicate/out-of-range IDs, empty lists and incompatible address/selection
modes fail. The file's SHA256 is in the recovery configuration; patch metadata
records the selection and digest. The flag off adds no recovery/patch metadata.

## Box commands (v57/)

Run from `/workspace/malleable`. Stage `census.py`, `c54.py`, the minimally changed
`e2e_fsa.py`, and all existing harness dependencies together in `v57/`. Inputs
below use the same tokenizer/export/backend as C51. The FineWeb file must contain
at least 12,998,656 uint32 tokens (6,347 complete 2,048-token contexts).
No CUDA/export experiment was launched on this host.

Dev relation-world screen, using C51's four entities and forty facts each:

```bash
set -euo pipefail
mkdir -p out/C54
python -u v57/census.py \
  --firing-screen --screen-layers 12,14,17,20 \
  --export data/hf/exports/725B-tokens_step58814 --export-step 58814 \
  --eda-backend naive_fp32 \
  --stream fwhq_sel=data/census/fwhq_sel.u32 \
  --tokens 12998656 --context-length 2048 --batch-size 4 \
  --tokenizer data/tokenizer.json \
  --factqa-seed 510051 --factqa-people 4 --factqa-facts 40 \
  --factqa-world dev --factqa-relations \
  --anchor data/anchor-wikitext.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --ckpt-tokens 0 --out out/C54/dev.npz | tee out/C54/dev.log
```

This generates 2,560 prompts: 160 facts × (12 training formats + 4 pronoun
conversations). S1 does not reuse the SC0 resume checkpoint; an interrupted
screen is restarted. The existing SC0 checkpoint/resume behavior remains intact.

Screen the exact 3p×100 small-arm relation world before selecting its controls:

```bash
python -u v57/census.py \
  --firing-screen --screen-layers 12,14,17,20 \
  --export data/hf/exports/725B-tokens_step58814 --export-step 58814 \
  --eda-backend naive_fp32 \
  --stream fwhq_sel=data/census/fwhq_sel.u32 \
  --tokens 12998656 --context-length 2048 --batch-size 4 \
  --tokenizer data/tokenizer.json \
  --factqa-seed 510051 --factqa-people 3 --factqa-facts 100 \
  --factqa-world arm --factqa-relations \
  --anchor data/anchor-wikitext.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --ckpt-tokens 0 --out out/C54/arm.npz | tee out/C54/arm.log
```

The arm screen generates 4,500 prompts: 300 facts × (12 formats + 3 pronoun
conversations). Both training arms below use that same seed/world, 300 value-only
updates and the union of eight units per entity. They use all eligible forced
zero slots of those down columns, with no additional slot sampling or global
support budget. Bop gamma .05, calibrated tau multiplier .5, local lambda 10,
full fact batch, twelve formats and identical final evaluation are fixed recipe
choices for this new control, not measured performance claims.

```bash
for c54_arm in top8 random8; do
  python -u v57/e2e_fsa.py \
    --export data/hf/exports/725B-tokens_step58814 --export-step 58814 \
    --eda-backend naive_fp32 \
    --method fsa_forced --fmt ternary --optim bop --bop-gamma .05 --bop-tau .5 \
    --layers 12,14,17,20 --down-only \
    --restrict-units "out/C54/arm.${c54_arm}.units.json" \
    --people 3 --facts-per-person 100 --shared-attrs 0 --values disjoint \
    --relation-world --seed 510051 --steps 300 --fact-batch 300 --val-facts 300 \
    --answer-only --aug --aug-all --micro 6 \
    --budget 0 --slot-frac 1 --sparse-state --expert-checkpoint --fast-moe exact \
    --local-lambda 10 --rehearsal-frac 0 --no-churn --no-pareto \
    --eval-every 100 --qa-probe --qa-n 120 --zero-patch-control \
    --anchor data/anchor-wikitext.json \
    --eval-set data/eval-wikitext.json \
    --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
    --ckpt-every 100 \
    --out "out/C54/value_${c54_arm}.json" \
    --patch-dir "out/C54/value_${c54_arm}_patches" \
    | tee "out/C54/value_${c54_arm}.log"
done
```

If the screen certifies no unit at 0.9, S1's finding is that addresses need
training; the paired runs can still measure the requested top-eight control but
are not evidence for the high-AUROC gate. Report final held-out GEN, QA/pronoun
and genuine two-hop results, all three paired damage panels, live slots, reserve
slots, selected union sizes and wall time for both arms. Fewer than eight
certified units per entity and overlapping entity selections are visible in the
screen summary. For isolated entity writes, use the corresponding per-entity
file and `--train-person PID` instead of the combined union.

## Validation

CPU/macOS, Python 3.12 and PyTorch 2.8.0. Tests cover planted entity-selective
units, exact BF16 pairwise AUROC, uncertain FP32 within-bin pairs, all-zero ties,
non-dispatched zero scores, chunk-invariant histogram merges, actual duplicate
routes and padding, token offset boundaries, aliases, recent-person antecedents,
relation-answer spans, full dev/arm entry points and output artifacts, restricted
mask legality, real Bop writes and unit-file content identity on resume. Existing
C49/C51 and harness default/RNG parity tests are included.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q experiments/end_to_end/tests/test_c54.py \
  experiments/end_to_end/tests/test_c51.py experiments/end_to_end/tests/test_c49.py \
  experiments/end_to_end/tests/test_fast_moe.py experiments/end_to_end/tests/test_e2e_fsa.py
```

The combined regression command above, extended with `test_c44.py` and
`test_c45.py`, passed **392 tests**, with the existing CUDA-only dispatch test
skipped and the existing LoRA scalar-conversion warning. A final exact-0.9
threshold-boundary case was then added; the final C54 file passed **14 tests**.
Python syntax checks, `git diff --check`, shell syntax of every command block
and parsing all three box commands through the real CLI parsers passed.
GPU timing, full-export selectivity and the small-arm empirical verdict remain
for the box commands.
