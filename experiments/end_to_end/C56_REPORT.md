# C56: minimal review fixes and final v57 commands

Implemented `reviews/brainstorm_2026-10-12/tasks/c56_brief.md`. Existing workspace
changes were preserved. No `git commit` was run.

## Changes

- C50 uses a dedicated device-local `torch.Generator`, seeded with `--seed +
  560050`, for both off-topic windows and in-topic negatives. C44 checkpoints
  save/restore it as `training.objects.preservation_rng`, alongside token counters
  and remainders. KL no longer advances either the global/local-penalty RNG or
  the factorial anchor RNG. Tests compare actual local-penalty indices with
  `--offtopic-kl .03 --local-lambda 10` versus `--offtopic-kl 0` at seed 1.
- Sibling hard negatives use entities absent from `train_idx`, plus novel
  near-sibling names. Absent-bank rows exclude written pairs and all known facts
  of trained entities, so validation holdouts cannot become base-KL targets.
  Previously written resident pairs remain protected.
- C52 defaults to `--distractor-mode all`: 75% relevant-only visibility, 25%
  relevant plus every other bank, before independent presence dropout. The old
  uniform-k distribution remains available with `--distractor-mode uniform`;
  a smaller `--max-distractors` requires that mode. J/F use the same activation
  schedule and SHA256; tests exercise the real joint runner in both arms.
- Census defaults are 10,000 minimum routed tokens per expert/domain and
  selection A / confirmation B = 70/30 (`--split-fraction .3` is B's share).
  A-only calibration estimates the layer median of **nonzero** phi using the
  existing bounded histogram. Tau uses the middle order statistics' bin upper
  edges, averaged for an even sample (about 2.2% multiplicative bin width);
  zeros are excluded and an all-zero layer or
  median overflow fails explicitly. Tau stays fixed for measurement.
- Energy ranks, 10%/20% reserve selections, A/B stability and answer-position
  reachability select fractions within each expert. Near-dormant thresholds
  likewise use each expert/domain's p1, with low-count cells excluded. NPZ
  `layerN_Ee_rank` / `_rank_B` index the full layer expert/unit mapping. Legacy
  flat `layerN_rank` / `_rank_B` concatenate expert-ordered rank blocks; their
  prefix must not be used as a pooled reserve selection.
- Batch agreement checks bottom **20% per expert**, reports each expert, and
  requires >=95% in every measured expert. A failed expert fails the gate even
  if pooled overlap passes. Empty/unobserved selections are unmeasured; they
  cannot certify a pass. Census checkpoint version 3 and the selection policy
  identity reject old p99/global-ranking checkpoints; start a new census.

## Model-box commands

Run from `/workspace/malleable` in Bash with the current harness and all its
existing helper modules staged together in `v57/`, including C50/C52/C54,
`rank1_solve.py`, `teacher_screen.py`, `census.py`, `eda_backend.py`, and
`c44_runtime.py`. These are final recipes; no CUDA/725B jobs ran on this host.
These commands supersede the earlier C50–C55 report recipes for this launch.

```bash
set -euo pipefail
mkdir -p out/C52_J out/C52_F out/C53 out/C54 out/C56
```

### GL_A2_s1

The C50 queue recipe, moved to `v57/`, with explicit export/backend/output paths:

```bash
python -u v57/e2e_fsa.py --method fsa_forced --optim bop --bop-gamma 0.05 --bop-tau 0.7 --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1 --offtopic-kl 0.03 --offtopic-kl-weight 1.0 --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 --export-step 58814 --eda-backend naive_fp32 --patch-dir out/GL_A2_s1_patches --out out/GL_A2_s1.json
```

### SC4 J and F

Start from the same three disjoint R20 FULL banks. Their complete writable masks
must be disjoint, as enforced by the loader. The arms differ only in distractor
gradients and output destinations.

```bash
python v57/e2e_fsa.py \
  --joint-banks out/R20_full_p0/final.pt out/R20_full_p1/final.pt out/R20_full_p2/final.pt \
  --method fsa_forced --fmt ternary --optim bop \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --layers 12,14,17,20 --people 3 --relation-world \
  --steps 150 --seed 0 --joint-schedule-seed 0 \
  --distractor-mode all --p-iso .75 --max-distractors 2 --presence-dropout .2 --presence-target base \
  --answer-only --aug --fact-batch 40 --micro 1 \
  --bop-gamma .05 --bop-tau .5 --budget-freeze 150 --refresh-every 10 \
  --local-lambda 10 --anchor data/anchor-wikitext.json --anchor-len 96 --anchor-bs 2 \
  --sparse-state --no-churn --expert-checkpoint --fast-moe exact \
  --eval-set data/eval-wikitext.json --eval-windows 40 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --in-topic-panel data/in-topic-725B.json \
  --fact-class-metrics --decisive-eval --qa-n 120 --joint-negative-n 1000 --gen-bs 1 \
  --patch-dir out/C52_J --out out/C52_J.json

python v57/e2e_fsa.py \
  --joint-banks out/R20_full_p0/final.pt out/R20_full_p1/final.pt out/R20_full_p2/final.pt \
  --joint-frozen-distractors \
  --method fsa_forced --fmt ternary --optim bop \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --layers 12,14,17,20 --people 3 --relation-world \
  --steps 150 --seed 0 --joint-schedule-seed 0 \
  --distractor-mode all --p-iso .75 --max-distractors 2 --presence-dropout .2 --presence-target base \
  --answer-only --aug --fact-batch 40 --micro 1 \
  --bop-gamma .05 --bop-tau .5 --budget-freeze 150 --refresh-every 10 \
  --local-lambda 10 --anchor data/anchor-wikitext.json --anchor-len 96 --anchor-bs 2 \
  --sparse-state --no-churn --expert-checkpoint --fast-moe exact \
  --eval-set data/eval-wikitext.json --eval-windows 40 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --in-topic-panel data/in-topic-725B.json \
  --fact-class-metrics --decisive-eval --qa-n 120 --joint-negative-n 1000 --gen-bs 1 \
  --patch-dir out/C52_F --out out/C52_F.json

python - <<'PY'
import json
from pathlib import Path
j = json.loads(Path('out/C52_J.json').read_text())
f = json.loads(Path('out/C52_F.json').read_text())
assert j['schedule']['sha256'] == f['schedule']['sha256']
assert j['schedule'] == f['schedule']
print('J/F schedule.sha256:', j['schedule']['sha256'])
PY
```

### S1 firing screen and small arm

The dev screen and exact small-arm screen use their original C54 worlds. Run the
small-arm top8/random8 controls after inspecting `decision` and
`top8_qualified_at_09`: a top-eight file with fewer than eight certified units
per entity is not an eight-unit high-AUROC positive control.

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

### S2 rank-one solve, continuous/ternary evaluation, polish and polished evaluation; S3 teacher screen

The S2 source is training-only FineWeb, checked against every supplied panel.
The explicit final load-eval measures the polished patch independently at zero
steps. S3 uses no patch and reports C34 oracle-evidence ceilings, including the
dated-conflict panel; its gate uses forward-chain and within-person conjunction.

```bash
set -euo pipefail
C53_PY=/venv/main/bin/python
C53_EXPORT=/workspace/malleable/data/hf/exports/725B-tokens_step58814
C53_BACKEND=naive_fp32
mkdir -p out/C53

# S2: full covariance, subject expert-input keys, no held-out tuning.
"$C53_PY" v57/rank1_solve.py \
  --export "$C53_EXPORT" --export-step 58814 --eda-backend "$C53_BACKEND" \
  --seed 0 --layers 12,14,17,20 --projection up \
  --cov-tokens data/rehearsal-fineweb.json --cov-max-tokens 10000 \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json \
    data/eval-code.json data/heldout-tokens.json \
  --cov-rank 0 --cov-window 256 --ridge .001 \
  --v-steps 10 --v-lr .05 --v-batch 0 --v-l2 .001 \
  --row-batch 8 --spectrum-rows 3 --solve-device cuda:0 \
  --out out/C53/rank1

C53_EVAL=(
  --export "$C53_EXPORT" --export-step 58814 --eda-backend "$C53_BACKEND"
  --method fsa_forced --layers 12,14,17,20 --seed 0
  --people 1 --facts-per-person 40 --shared-attrs 0 --values disjoint
  --sparse-state --slot-frac 1 --patch-scale row --budget 0
  --steps 0 --eval-every 0 --no-pareto --no-churn
  --eval-set data/eval-wikitext.json
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json
  --fast-moe exact --base-cache out/C53/base_cache --gen-bs 1
)
"$C53_PY" v57/e2e_fsa.py "${C53_EVAL[@]}" --fmt bf16 --optim adam \
  --load-patch out/C53/rank1/continuous.pt \
  --patch-dir out/C53/continuous_eval --out out/C53/continuous.json
"$C53_PY" v57/e2e_fsa.py "${C53_EVAL[@]}" --fmt ternary --optim bop \
  --load-patch out/C53/rank1/ternary.pt \
  --patch-dir out/C53/ternary_eval --out out/C53/ternary.json

# Positive budget creates empty support; reload populates only imported codes.
# No refresh means Bop can only delete/rebirth those coordinates for all 30 steps.
C53_NNZ=$("$C53_PY" - <<'PY'
import torch
s = torch.load('out/C53/rank1/ternary.pt', map_location='cpu', weights_only=False)
print(sum(len(e['idx']) for e in s['entries'].values()))
PY
)
if [ "$C53_NNZ" -gt 0 ]; then
  "$C53_PY" v57/e2e_fsa.py "${C53_EVAL[@]}" --fmt ternary --optim bop \
    --load-patch out/C53/rank1/ternary.pt --steps 30 \
    --budget "$C53_NNZ" --budget-freeze -1 --bop-gamma .05 --bop-tau .7 \
    --answer-only --aug --aug-all --fact-batch 40 --micro 6 \
    --expert-checkpoint --anchor data/anchor-wikitext.json \
    --anchor-bs 2 --anchor-len 96 --local-lambda 10 \
    --patch-dir out/C53/polished --out out/C53/polished.json
  "$C53_PY" v57/e2e_fsa.py "${C53_EVAL[@]}" --fmt ternary --optim bop \
    --load-patch out/C53/polished/final.pt \
    --patch-dir out/C53/polished_eval --out out/C53/polished_eval.json
else
  printf '%s\n' 'No ternary support: S2 fails; there are no coordinates to polish.'
fi

# S3: direct base-only oracle evidence, actual relation world, all four families.
"$C53_PY" v57/teacher_screen.py \
  --export "$C53_EXPORT" --export-step 58814 --eda-backend "$C53_BACKEND" \
  --seed 0 --people 3 --facts-per-person 40 --shared-attrs 30 --values disjoint \
  --relation-world --qa-n 120 --gen-stop-k 2 --out out/C53/teacher.json

```

### Census (725B)

Each of the four input files needs at least 12,998,656 tokens (6,347 full
2,048-token windows). Total measurement budget is 51,994,624 tokens. Per domain,
A contains 4,443 windows = 9,099,264 tokens; B contains 1,904 windows = 3,899,392
tokens. The 999,424-token A calibration and 499,712-token batch-agreement replay
are additional forwards over existing prefixes. Factqa is an additional dev
routing diagnostic. The 10,000-token observation threshold applies separately
to each expert/domain/split. Inspect per-expert agreement/unobserved counts
before using reserve selections.

```bash
python -u v57/census.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --stream cp_wikimedia=data/census/cp_wikimedia.u32 \
  --stream fwhq_sel=data/census/fwhq_sel.u32 \
  --stream pd_cp_pre1929=data/census/pd_cp_pre1929.u32 \
  --stream cp_stackexchange=data/census/cp_stackexchange.u32 \
  --tokens 51994624 --context-length 2048 --batch-size 4 \
  --min-routed 10000 --split-fraction 0.3 \
  --tokenizer data/tokenizer.json \
  --factqa-seed 510051 --factqa-people 4 --factqa-facts 40 \
  --anchor data/anchor-wikitext.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --ckpt-tokens 1000000 --checkpoint out/C56/census-725B.resume.pt \
  --out out/C56/census-725B.npz

```

After interruption, repeat the exact census command with `--resume auto`.
Use the same backend for census, reserve training and write evaluation.

### PG_eval_fla2

Load-evaluate the existing naive-backend `F40_bopck_s0` patch with FLA and the
C55 evaluation-only override. The world, layers and full-slot recipe match
`reviews/experiments/facts40/raw/F40_bopck_s0.log`; anchor length 256 is the
historical harness default. Damage includes the two additional panels, so
compare them against an otherwise identical naive-backend load-eval. The
result must record `backend_crosseval={saved: naive_fp32, eval: fla_kda_2t}`;
the original patch's backend provenance is retained.

```bash
python -u v57/e2e_fsa.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend fla_kda_2t --allow-backend-crosseval \
  --load-patch out/F40_bopck_s0_patches/final.pt --steps 0 \
  --method fsa_forced --fmt ternary --optim bop --layers 14,15 \
  --seed 0 --people 1 --facts-per-person 40 --shared-attrs 0 --values disjoint \
  --budget 300000 --slot-frac 1 --patch-scale row --sparse-state \
  --support-score netcost --cost-rel 1 --bop-gamma .05 --bop-tau .7 \
  --local-lambda 10 --rehearsal-frac .2 --answer-only --aug \
  --no-pareto --no-churn --expert-checkpoint --fast-moe exact \
  --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 256 \
  --eval-set data/eval-wikitext.json --eval-windows 40 --eval-every 0 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --gen-bs 1 --gen-stop-k 2 --base-cache base_cache \
  --patch-dir out/PG_eval_fla2_patches --out out/PG_eval_fla2.json
```

## Validation

Broad CPU regression: **548 passed, 1 skipped** in 147.50 seconds. The skip is
the existing CUDA-only dispatch benchmark; the existing LoRA scalar-conversion
warning remains. Coverage includes actual local-anchor draws with KL on/off,
hard-negative holdout exclusions, process interruption/resume with preservation
RNG equality, default/full-set and uniform-k visibility, real J/F schedule
SHA256 equality, per-expert ranks and A/B confirmation, bottom-20% batch checks,
census resume, firing screens, rank-one solve/reload/polish, teacher screens and
cross-backend load-eval.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q --tb=short \
  experiments/end_to_end/tests/test_c50.py \
  experiments/end_to_end/tests/test_c51.py \
  experiments/end_to_end/tests/test_c52.py \
  experiments/end_to_end/tests/test_c49.py \
  experiments/end_to_end/tests/test_c44.py \
  experiments/end_to_end/tests/test_c54.py \
  experiments/end_to_end/tests/test_c53.py \
  experiments/end_to_end/tests/test_c55.py \
  experiments/end_to_end/tests/test_e2e_fsa.py \
  experiments/end_to_end/tests/test_branch_eval.py \
  experiments/end_to_end/tests/test_fast_moe.py \
  experiments/end_to_end/tests/test_c45.py \
  experiments/end_to_end/tests/test_c34.py
```

After refining the histogram median to average the two middle order statistics
for even samples, C51/C49/C54 were rerun: **59 passed** in 4.00 seconds,
including odd/even nonzero medians, zero exclusion and overflow rejection.

All **15 expanded model commands** pass Bash syntax and their actual CLI parsers
and applicable recipe validation. Expansion checks include both S1 arms, the
S2 shell array and polish budget, and the explicit polished load-eval. Parsed
J/F arguments match except for the freeze flag and output paths. Census token
budgets match the values above. Python compilation, `git diff --check`, and
whitespace checks on the changed untracked files pass. Real FLA kernels,
725B recalls, damage, GPU timings and screen verdicts remain unmeasured here.
