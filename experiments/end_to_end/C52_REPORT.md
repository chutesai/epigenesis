# C52: joint J/F continue-training and N3 presence dropout

Implemented the opt-in entry point in `e2e_fsa.py`, its joint runner in `c52.py`,
and tiny-fixture regressions in `tests/test_c52.py`. `census.py` was not edited.
No `git commit` was run. Existing working-tree changes were preserved.

## Training and saved state

`--joint-banks F1 F2 ...` attaches trainable, reserve-sized sparse FSA banks to
one frozen base. Each retains its saved world/scope, row scales, codes, support
budget and Bop EMA/threshold. Complete writable masks, including dormant slots,
are reconstructed from each saved partition/sampling recipe and checked for
overlap **before any parametrization is registered**. Shape, codes, backend,
export, fact ownership and world compatibility are checked too. Ordinary
row-scaled ternary routed FSA banks are supported; occupied flips, unit/address
overlays and unrelated training objectives are rejected explicitly.

Existing v1 inference patches store live codes and scales, without continuous
masters, dormant support, support scores or EMA. A matching completed-step C44
`resume.pt` beside an input `final.pt` restores those tensors and its calibrated
Bop threshold; its codes must agree with the inference patch. Without training
state, the loader uses stored codes as Bop masters, live support and zero EMA,
then calibrates a separate answer-gradient threshold for each bank. This fallback
is recorded in `source_states`; it does not claim to recover missing state.
`--bop-tau-abs` explicitly overrides all bank thresholds when supplied.

Each final bank is saved as `<patch-dir>/bankN/final.pt`, using the existing
`e2e-fsa-patch-v1` inference entries plus optional `joint_state` version 1. The
extra state preserves the full reserve indices, masters, support, score, EMA,
budget and threshold for further joint continuation. Existing resident/reload
consumers can use its ordinary entries. Input hashes, arm, schedule seed/hash,
visibility counts, per-bank budgets, source-state provenance and update diagnostics
are recorded in the result; each bank's metadata also records its joint schedule.

The saved fact world is authoritative. Explicit world/layer/budget arguments must
agree with it. Every update includes a question from each bank and a sampled
same-attribute, different-value conflict group when available. `--fact-batch`
is therefore a requested minimum that may be exceeded to cover those groups;
zero includes every owned fact. Non-augmented training uses the ordinary training
templates; `--aug` samples three of the augmented templates per fact, and
`--aug --aug-all` includes all of them. Loss covers every answer/stop token for
every bank's query, with an equal mean per example.

The dedicated sampler uses separate seeded streams for facts/wordings and bank
visibility. Its draw never depends on arm, logits or optimizer state. With
probability `--p-iso .75`, the relevant bank is alone; otherwise it is accompanied
by uniformly sampled `k ~ U(1,M)` distinct other banks. Default M is all others;
`--max-distractors` can reduce it. This follows the brief and authoritative
Refinement's uniform-k rule. At three banks, mixed examples have one **or** two
distractors; it does not replace uniform-k with all-two on every mixed example.

Arm J sends gradients through every active bank. `--joint-frozen-distractors`
selects F: non-relevant deltas are detached only for that example, while the banks
remain trainable on their own examples. Forward/backward finishes before changing
visibility, including expert checkpoint replay. One example is executed per
microbatch: `--micro` is an upper bound, and grouping is not currently used. This
increases forward/backward dispatch overhead and sacrifices batching throughput;
memory remains bounded by one example's graph plus all banks' training state.

On `--presence-dropout .2` of own queries, the relevant bank is removed after the
ordinary activation draw. `--presence-target base` uses full-vocabulary
`KL(base || student)`, with a detached, no-grad, all-banks-off teacher on the same
teacher-forced answer prefixes. `--presence-target unknown` instead supervises
` Unknown.` including its stop. Isolated dropped queries have no active trainable
bank and therefore no patch gradient; F also has no patch gradient on mixed
dropped queries because every remaining bank is a frozen distractor. These cases
are counted, not backpropagated through artificially activated banks.

Each bank's Bop calibration and support score use answer gradients before local
preservation. Bop updates and support refresh stay within its own reserve and
budget. Optional `--local-lambda` applies the existing output penalty to each
bank alone on identical separately seeded anchor draws in J and F. Anchor/eval
32-gram overlap checks remain enforced.

## Evaluation

Final evaluation is unconditional in joint mode and records:

- Each bank alone on its own held-out TEST phrasings, all banks composed, and
  composed recall by bank.
- All-active C34 task panels, including cross-person conjunction and genuine
  sister-relation forward chains; only owned, supported tasks are included.
- Every ordered own-person/wrong-bank activation condition.
- At least 1,000 distinct never-written entities, evaluated with base and all
  banks active. This reuses decisive_eval's negative templates, value-boundary
  matcher, relation-value exclusions and refusal/assertion classifier. Both
  `false_answer_rate` (asserted answers) and stored-value false recall are reported,
  together with generations. No shorter internal negative pool substitutes for
  the 1,000-query panel.
- Off-topic paired NLL on heldout, `--eval-set` and `--extra-eval-sets`; C49
  knowledge QA, concept-text NLL and written-fact retention when
  `--in-topic-panel` is supplied.
- C50 exact/semantic splits for GEN and composition, without metric-only
  training effects. Splits are always included for the joint experiment.
- Per-bank revoke logits/weights equality against an independently rebuilt set
  of remaining ordinary resident banks, exact pristine-base recovery with all
  banks removed, and remaining-bank recall changes relative to all-resident
  recall. The independent recall gate is a loss of at most one percentage point.

This is a final-only experimental runner, without legacy Pareto selection or
C44 interrupted-joint-run resume. Unsupported checkpoint/objective flags fail
instead of being silently ignored. Saved banks can be continued in another run;
that starts a new explicitly seeded schedule.

## Exact box commands (v57)

Stage the harness and its dependencies together in `v57/`, including `c52.py`,
`c50.py`, `decisive_eval.py`, `in_topic.py`, `token_streams.py`, `eda_backend.py`,
`c44_runtime.py`, `mls.py` and `fast_moe.py`. Run from the experiment directory on
the CUDA model host. The panel must have been built by C49 for the same export
and backend. Paths below use the established anchor/eval filenames. Budgets and
fact counts are inherited from the input banks.

**Launch precondition:** all three input banks must have disjoint complete
writable masks. The documented older FULL-bank recipe has unpartitioned sampled
reserves, which can overlap (identical sampling seeds on the same base produce
identical reserves). The actual R20 files are not in this local workspace, so
their layout could not be certified. These exact requested FULL-input commands
will reject overlapping masks rather than remap/delete saved writes or silently
change to the block/unit bank set. This is a prerequisite to resolve on the model
host if its FULL files use that older layout.

```bash
python v57/e2e_fsa.py \
  --joint-banks out/R20_full_p0/final.pt out/R20_full_p1/final.pt out/R20_full_p2/final.pt \
  --method fsa_forced --fmt ternary --optim bop \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --layers 12,14,17,20 --people 3 --relation-world \
  --steps 150 --seed 0 --joint-schedule-seed 0 \
  --p-iso .75 --max-distractors 2 --presence-dropout .2 --presence-target base \
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
  --p-iso .75 --max-distractors 2 --presence-dropout .2 --presence-target base \
  --answer-only --aug --fact-batch 40 --micro 1 \
  --bop-gamma .05 --bop-tau .5 --budget-freeze 150 --refresh-every 10 \
  --local-lambda 10 --anchor data/anchor-wikitext.json --anchor-len 96 --anchor-bs 2 \
  --sparse-state --no-churn --expert-checkpoint --fast-moe exact \
  --eval-set data/eval-wikitext.json --eval-windows 40 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --in-topic-panel data/in-topic-725B.json \
  --fact-class-metrics --decisive-eval --qa-n 120 --joint-negative-n 1000 --gen-bs 1 \
  --patch-dir out/C52_F --out out/C52_F.json
```

Verify identical `schedule.sha256` and `schedule.counts` in J/F before comparing
results. Compare `evaluation.composed_gen.recall` with mean individual recall and
the untouched R20 independent-bank reference. A no-dropout joint reference is
needed to apply N3's preregistered <=2-point composed-GEN cost criterion; these
two J/F runs alone do not establish that causal contrast.

## Validation

CPU/macOS with PyTorch 2.8.0, using
`/private/tmp/epigenesis-ladder-venv/bin/python -m pytest`:

- C52 focused tests: 23 passed. Covers J/F gradients with and without expert
  checkpointing, inactive-bank silence, sampler distribution/cardinality,
  arm-identical exposures, base KL and explicit unknown targets, complete-mask
  overlap rejection, sampled-mask recovery, state round trips, matching C44 state
  recovery, exact revoke and final evaluation integration (including C49/C50).
- Final broader run of C52/e2e/C34/C35/C38/C49/C50: **398 passed**, with one
  existing tensor-to-scalar warning. An earlier run encountered two census
  failures during the concurrent C51 edits; both passed in isolation and in the
  final broader run. No census fix was made by C52.
- C45 backend coverage plus C52: **54 passed**. The loader-count assertion now
  allows additional entrypoints while checking every builder/flag/announcement;
  the joint runner announces its backend through the shared wrapper. C44
  recovery regressions: **16 passed** in the earlier combined C44/C45/C52 run.
- A direct before/after run of the saved pre-C52 harness's parser/main against
  the same tiny fixture produced identical RNG state, acquisition results,
  tensor/metadata trees and **entire final.pt file bytes**. Legacy execution
  removes the new default parser fields before checkpoint/configuration capture.
- Both reported box commands were parsed and validated with the real parser.
  Python compilation and `git diff --check` passed.

No 8B model training or GPU result is claimed.
