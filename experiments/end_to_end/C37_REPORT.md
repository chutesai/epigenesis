# C37 novel-idea learning experiments

All seven opt-ins are implemented in `e2e_fsa.py`, with a selector-free
`branch_eval.py --learning-eval` path. Defaults remain unchanged. No commit.
Existing C32–C36 workspace changes were preserved.

| Experiment | Measurements and controls |
| --- | --- |
| Sibling environment | Independently draws each frozen bank's presence every update. Same student mask for p=0/1/random arms; siblings never enter optimizer/export. Exhaustive configurations report focus GEN, worst GEN, isolated GEN and damage in training harness. Eval uses additive saved deltas at full strength. |
| Robust trits | Forward-only live-code dropout/sign corruption, one seeded draw per update held through checkpoint backward. Separate `--activation-noise Q` arm injects Gaussian linear-output noise with energy matched to shadow Q-rate live-trit sign flips. Stored masters, EMA, support and slot budget are untouched by the noise. Final clean QA/test GEN plus 10%/25% exact-count deletion, five draws each, with restoration. |
| Hard facts | Random code values matched to original tokenizer answer lengths. Twenty paired capital overrides, filtered by clean true-value base agreement before writing; accepted count and every base answer recorded. Final unseen QA, test GEN, slots/fact and untouched-knowledge ripple probes. Saved worlds reload exactly. |
| Value operands | Numeric >50, +3 and string-instrument membership. Reports patched memory, base literal, patched literal, base memory and patched direct recall. Mastery precondition requires direct and base-literal accuracy >=95%. Empty operation sets explicitly unavailable. |
| Dark audit | Eval only. 300 fixed geography/continent questions, with targeted overrides removed, plus feasible shared geographical name-fragment probes. Base/patched GEN, per-token base-to-patch KL p99/p99.9/mean/max, worst-window delta NLL, and bounded greedy template search scored on base continuations. |
| Transplant | Normal cross-export loading still refuses mismatches. Explicit transplant maps sparse codes to the NEW forced-zero reserve, drops illegal coordinates, derives NEW row scales, and rejects flips/reset/address patches. Reports dropped fraction and recall/damage before/after <=10 Bop repair updates; repair births may use the new legal mask within the existing budget. Mask Jaccard is computed when a source `--mask-compare` artifact is supplied, otherwise unavailable. |
| Evidence skill | Disposable train people/attribute, dated/composition evidence episodes built with the oracle builders, changing values each update and randomized dated-clause order. Answer-only loss; held-out dated evidence validation. Final base, patched, no-context and base-no-context dated/composition results; damage and decision. Ordinary fact-bank runs provide a separate ceiling reference. |

Evidence composition retains the historical oracle presence/stop score for the
2.8%/8.3% comparison, with a separate terse ordered-correctness diagnostic.
Uniform attribute-pool guessing is labeled as a reference probability; it is
not a calibrated free-form model prior. The matched base-no-context result is
also reported. No full-model accuracy or scientific pass is claimed from CPU
controls. Neighbour wording repetitions do not constitute 300 independent facts.
The trigger search is a fixed bounded diagnostic, not an exhaustive guarantee.

Validation commands (cached Python avoids the sandbox's unwritable uv cache):

Final full CPU suite: **909 passed, 1 skipped** (one existing tensor-to-scalar
warning); C37 suite: **39 passed**. Both CLI mechanics controls pass.
Python parsing, documented shell syntax, and `git diff --check` pass.

```bash
C37_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C37_PYTHON" -m pytest -q experiments/end_to_end/tests
"$C37_PYTHON" experiments/end_to_end/e2e_fsa.py --c37-control --out /tmp/c37-control.json
"$C37_PYTHON" experiments/end_to_end/branch_eval.py --c37-control --out /tmp/c37-branch-control.json
git diff --check
```

Full-model commands below are **not run here**: the external lambda loader,
exports, tokenizer and GPU are unavailable. Run from the repository root.
Use the established headline layers/slot budget/step count for the matched
arms; the explicit example uses layers 14,15, budget 200000, 300 updates.
Supply disjoint WikiText/FineWeb/code panels and an anchor file.

```bash
C37_PYTHON=python
C37_TRAIN=experiments/end_to_end/e2e_fsa.py
C37_BRANCH=experiments/end_to_end/branch_eval.py
C37_WIKI=data/wikitext.json
C37_FINEWEB=data/fineweb.json
C37_CODE=data/code.json
C37_ANCHOR=data/anchor.json
mkdir -p results/c37
C37_COMMON=(--layers 14,15 --steps 300 --budget 200000 --answer-only
  --micro 12 --eval-every 50 --no-pareto --qa-probe --qa-n 120
  --anchor "$C37_ANCHOR" --local-lambda .01 --eval-set "$C37_WIKI"
  --extra-eval-sets "$C37_FINEWEB,$C37_CODE")
C37_BOP=(--method fsa_forced --optim bop)

# Siblings: independent clean banks, then new bank in three environments.
C37_WORLD=(--people 3 --facts-per-person 100 --shared-attrs 100
  --slot-partition 3 --partition-unit)
for person in 0 1; do
  "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" "${C37_WORLD[@]}" \
    --train-person "$person" --slot-part "$person" \
    --patch-dir "results/c37/sibling$person" --out "results/c37/sibling$person.json"
done
for p in 0 1 .5; do
  "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" "${C37_WORLD[@]}" \
    --train-person 2 --slot-part 2 --sibling-p "$p" \
    --sibling-patches results/c37/sibling{0,1}/final.pt \
    --patch-dir "results/c37/environment$p" --out "results/c37/environment$p.json"
done

# Robust common-checkpoint continuation; keep compute and budget identical.
"$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" --steps 150 \
  --patch-dir results/c37/partial --out results/c37/partial.json
for noise in clean drop10 drop25 corrupt activation; do
  C37_NOISE=()
  case "$noise" in
    drop10) C37_NOISE=(--slot-dropout .1);;
    drop25) C37_NOISE=(--slot-dropout .25);;
    corrupt) C37_NOISE=(--trit-corrupt .001);;
    activation) C37_NOISE=(--activation-noise .001);;
  esac
  "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" \
    --steps 150 --load-patch results/c37/partial/final.pt "${C37_NOISE[@]}" \
    --patch-dir "results/c37/$noise" --out "results/c37/$noise.json"
done

# Prior-supported vs arbitrary-code FSA/LoRA, seeds 0 and 1.
for seed in 0 1; do
  for world in ordinary surprisal; do
    C37_HARD=()
    if [ "$world" = surprisal ]; then C37_HARD=(--hard-facts surprisal); fi
    for method in fsa_forced lora; do
      C37_OPTIM=(--optim bop)
      if [ "$method" = lora ]; then C37_OPTIM=(--optim adam --rank 16 --budget 0); fi
      "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" --method "$method" \
        "${C37_OPTIM[@]}" --seed "$seed" "${C37_HARD[@]}" --value-operands \
        --patch-dir "results/c37/$world-$method-$seed" --out "results/c37/$world-$method-$seed.json"
    done
  done
done
"$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" \
  --hard-facts override --people 1 --facts-per-person 20 \
  --patch-dir results/c37/override --out results/c37/override.json

# Audit existing FSA and LoRA winners by supplying matching training config.
"$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" --steps 0 \
  --load-patch results/c37/ordinary-fsa_forced-0/final.pt --dark-audit --value-operands \
  --out results/c37/dark.json
"$C37_PYTHON" "$C37_BRANCH" --learning-eval \
  --patches results/c37/environment.5/final.pt results/c37/sibling{0,1}/final.pt \
  --eval-set "$C37_WIKI" --fineweb "$C37_FINEWEB" --code "$C37_CODE" \
  --out results/c37/environment-eval.json

# Base upgrade: preserve source world/options; repeat repair=0 and 10.
C37_NEW_EXPORT=/path/to/new/export
C37_NEW_STEP=6445
for repair in 0 10; do
  "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" --steps 0 \
    --load-patch results/c37/ordinary-fsa_forced-0/final.pt --transplant \
    --export "$C37_NEW_EXPORT" --export-step "$C37_NEW_STEP" --repair-steps "$repair" \
    --patch-dir "results/c37/transplant$repair" --out "results/c37/transplant$repair.json"
done

# Evidence skill vs ordinary fact-bank ceiling on exactly the same held-out world.
for kind in skill fact; do
  C37_SKILL=()
  if [ "$kind" = skill ]; then C37_SKILL=(--evidence-skill); fi
  "$C37_PYTHON" "$C37_TRAIN" "${C37_COMMON[@]}" "${C37_BOP[@]}" \
    --people 3 --facts-per-person 40 --shared-attrs 40 "${C37_SKILL[@]}" \
    --patch-dir "results/c37/evidence-$kind" --out "results/c37/evidence-$kind.json"
  "$C37_PYTHON" "$C37_BRANCH" --learning-eval --evidence-skill \
    --patches "results/c37/evidence-$kind/final.pt" \
    --eval-set "$C37_WIKI" --fineweb "$C37_FINEWEB" --code "$C37_CODE" \
    --out "results/c37/evidence-$kind-ceiling.json"
done
```

Preregistered comparisons from Muse/Astra; repeat sibling/robust/operand/skill
runs across seeds before interpreting a difference:

| Experiment | Advance / kill rule |
| --- | --- |
| Siblings (Astra #4) | Random p=.5: worst GEN >= better p=0/1 +10 points, isolated cost <=2 points. Kill if isolated training already meets target everywhere or randomization sacrifices isolation. Muse's phantom variant is not substituted for the actual sibling intervention requested here. |
| Robust (Astra #7; Muse #4) | Primary: >=5 points clean unseen QA at same slots/compute; synthetic tolerance alone is insufficient. Dropout: kill every positive-p arm if clean cost >3 points or 25%-deletion improvement <10 points; pass deletion within5 of clean at clean cost<=2. Compare final live slots, not only configured caps. |
| Hard facts (Muse #3) | Scope headline claim if code-value FSA–LoRA unseen gap <1/3 ordinary gap or slots/fact >3x ordinary. Ripple loss >5 points requires a different override regime. Report the base-gated accepted count; a reduced set is not a full 20-fact replication. |
| Operands (Astra #1) | Direct and literal >=95% precondition. >=20-point operation gap establishes deficit; <5 points on both mastered operations rejects bottleneck; otherwise inconclusive. |
| Dark (Muse #8) | Stop if neighbour loss <2 points, p99.9 <=2x LoRA and triggers comparable; any >=5x mean-implied scale merits further gating. Compare four FSA winners and LoRA; the search runs within a fixed budget. |
| Transplant (Muse #9) | Pass native recall retention >=80% before repair or >=95% after. Kill repair if ten updates recover <50% of native-minus-transplant gap. Native reference and fresh-new-base ceiling remain separate matched runs. |
| Skill (Muse #2) | Kill dated gain <15 points, no-context >chance+5, or any panel >+.003. Pass dated>=50%, composition>=40%, no-context at chance; all three damage panels required. |

The activation-noise comparator uses the same shadow trit perturbation rate
and matches output energy per forward. It performs an additional local matrix
multiply, so equal update counts do not imply equal wall time; report measured
compute costs alongside transfer. No serving behavior is added.
