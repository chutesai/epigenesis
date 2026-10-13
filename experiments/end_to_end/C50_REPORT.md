# C50: preservation KL and exact/semantic fact metrics

Implemented the opt-in objective and measurement split from
`reviews/brainstorm_2026-10-12/tasks/c50_brief.md`. Existing working-tree changes
were preserved. No `git commit` was run.

## Training objective

`--offtopic-kl F` draws general-text windows from `--rehearsal-set`, or the
training portion of `--anchor` when no rehearsal set is provided. It uses the
existing `anchor_indices` sampler: global torch RNG normally, the existing
seeded anchor generator under factorial recipes. Source overlap checks and the
anchor validation-tail exclusion remain in force. KL is full-vocabulary
`KL(base || student)` in fp32, on next-token positions after BOS. The teacher
forward runs without gradients and with FSA, LoRA, frozen residents and
train-only teacher adapters disabled; switches are restored before the student
forward and backward. Teacher logits are computed per sampled window, without
a vocabulary-sized persistent cache.

`--intopic-neg-frac I` adds sibling-entity questions and absent-bank attribute
questions. Siblings include unwritten entities in the current world and novel
near-sibling names. Absent-bank questions ask a current entity about an attribute
outside the new bank's scope. Answers are greedy continuations of the frozen
no-context base, at most 32 tokens per window or through EOS, with KL only on
answer positions. Teacher distributions collected during generation are reused
for the loss. This is distribution preservation, without invented gold answers.
Both negative kinds use `--offtopic-kl-weight W`, default 1.0.

Fractions count **tokens**, not examples or windows. For a positive batch with
`A` supervised assistant tokens, requested negative quotas are
`A*F/(1-F-I)` and `A*I/(1-F-I)`. Fractional remainders carry across updates,
bounding cumulative rounding error below one token per kind. The last window
is truncated to its quota. Assistant counts exclude prompt tokens, include
answer/stop and self-study answer tokens, and exclude saturation-masked facts.
The existing fact loss is scaled by `1-F-I`; negative loss is
`W * summed_KL / (A + actual_negative_tokens)`. The existing local/drift terms
retain their own weights. With `--answer-only`, this gives the intended token
mixture; without it, the existing fact CE also supervises prompts, while the
negative dose is still determined exclusively by assistant tokens.

Previously written resident/sibling facts, including worlds recorded in
`c37_facts`, are excluded from virgin-base KL targets. Loaded patches are also
included in the protected scope when KL is requested during repair. Protection
uses entity/attribute pairs independently of values, so conflicting rewrites
cannot become hard negatives. General-text windows mentioning any protected
entity are conservatively excluded; an entirely excluded source fails. Resident
answers are not re-anchored to the virgin base. Mask-only reservations are also
excluded conservatively.

Negative fractions must be finite in `[0,1)`, sum to less than one, and are
mutually exclusive with positive `--rehearsal-frac`. The weight must be finite
and nonnegative. The MLS transaction path and activation-noise hooks are rejected
with KL: they cannot provide this gradient objective and frozen-base teacher
respectively. Other gradient optimizers use the existing training loop. Support
scores and Bop threshold calibration still use the answer gradient before the
negative backward passes.

`RESULT` records the three flags and `c50`; patch metadata/checkpoints record
`c50` and the training recipe. Accounting includes requested/realized fractions,
actual assistant/off-topic/in-topic tokens, protected pairs, available hard
negative counts, source/window count, cumulative weighted loss, teacher and
student forward counts, and elapsed seconds. Timing includes teacher answer
generation and both KL forwards/backwards, with CUDA synchronization at the
timed region boundaries; the existing training wall-clock logs include the
same work. Token remainders/counters survive C44 process recovery. Elapsed
seconds naturally differ across uninterrupted/resumed processes.

## Fact classes and metrics

`--fact-class-metrics` enables measurement without changing training; requesting
either KL fraction enables it automatically. The deterministic rule is recorded
as `fact_class_rule`, with every global fact index recorded in `fact_classes`:

- **Exact:** the value contains any digit, or is a whitespace-free token
  containing `_`, `/`, `:`, or a backslash; consists of at least two uppercase
  letters; or contains an uppercase letter immediately after a lowercase letter.
- **Semantic:** all other values, including ordinary title-case names and word
  values such as `viola` or `blue-green`.

Multi-answer tasks are exact if any target fact is exact. C34 reverse lookup
uses the stored fact's value rather than its generated owner name; forward
chains classify the terminal fact and exclude the bridge. This is a diagnostic
classification rule, without a precision/allocation change.

GEN recall/match/strict, QA probes (including two-fact correctness and both
presence diagnostics), MCQ accuracy/confusion, and C34 generation/negative
panels have `by_fact_class` reductions. Each bucket contains its item count;
empty buckets have no measured rate. C34 per-item rows also record the class.
Splits reuse the same generations/logits as the aggregate metrics. Top-level
`FINAL`/`RESULT` include `by_fact_class`; the branch evaluator exposes splits
through its arm panels, C34 suite and top-level outputs, and accepts the same
measurement flag. Opted-in base-cache identities include the class rule code.
All options off preserves legacy training, output schema, patch tensors and RNG.

## Exact GL_A2_s1 queue line (v56/)

This is the `GL_A_s1` line from
`reviews/experiments/gap_ladder/design_review_muse.md` section 3, with the harness
prefix changed to `v56/`, the two requested flags appended, and the queue label
changed to `GL_A2_s1`. `--rehearsal-frac 0`, final-step selection and all other
recipe tokens are retained. Deploy `c50.py` alongside the changed `e2e_fsa.py`,
`branch_eval.py`, `decisive_eval.py` and their existing dependencies in `v56/`.

```text
v56/e2e_fsa.py --method fsa_forced --optim bop --bop-gamma 0.05 --bop-tau 0.7 --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1 --offtopic-kl 0.03 --offtopic-kl-weight 1.0|GL_A2_s1
```

The line does not add in-topic negatives to this specific arm; their flag
defaults to zero. The review's 0.05 operating point is available separately;
this requested arm uses 0.03. The command is a queue record, with output paths
supplied by the existing runner.

## Validation

CPU/macOS, Python 3.12 and PyTorch 2.8.0. Focused tests cover zero KL with patch
disabled, detached teacher and patch gradients, token quotas, resident exclusion,
exception-safe switch restoration, flag validation/exclusivity, class rules,
GEN/QA/MCQ/C34/branch outputs, metadata, reload, no-extra-forward measurement,
legacy patch/RNG parity, and real process interruption/resume.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q experiments/end_to_end/tests/test_c50.py experiments/end_to_end/tests/test_c34.py experiments/end_to_end/tests/test_branch_eval.py experiments/end_to_end/tests/test_e2e_fsa.py
```

The final combined regression passed **316 tests in 73.71 seconds**, with the
existing LoRA scalar-conversion warning. This includes the real process-resume
case and the full C34/PK branch entry point with class metrics enabled. Python
compilation and `git diff --check` pass. The queue line was checked against the
reference tokens and parsed through the actual harness parser. No CUDA/export
experiment, GPU timing or empirical recall/damage improvement was measured on
this host.
