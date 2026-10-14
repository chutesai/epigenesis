# C60: legacy column energy and rank-one device fixes

Both GPU crash causes are fixed. `c59.prepare()` now receives the original
bank metadata/model from J/F and single-patch continuation. For netcost banks,
it calls the original `collect_col_energy` on BASE with the original anchor
file and window length: remove the 510-token validation tail, construct windows
with the original exclusive range stop, then consume the first 32 windows.
The bank's seed is recorded in the replay log; this statistic's window selection
is deterministic and consumes no RNG draws. Continuation command anchor settings
are never substituted for the original settings.

Older training recipes recorded `anchor` but omitted `anchor_len`. When needed,
the loader recovers missing settings from the original run's matching
JSON/manifest, including `configuration.anchor_len`. Existing checkpoint hash
and bank identity checks apply, including seed inside manifest configuration.
New recipes record `anchor_len`, and recovered settings persist in exported
continuation checkpoints. Missing original settings fail rather than guessing.
Keep each R20 original result/manifest beside its bank, as in C59.

`collect_col_energy` now honors its module argument, selecting the requested
bank's parametrization even when several banks share a linear. Other banks'
statistics are not overwritten. BASE enable flags and temporary hooks are
restored on failure. All saved column energies are checked against replay
(`rtol=1e-5`, `atol=1e-8`); mismatches raise and saved values remain authoritative.
Missing columns receive the replayed statistic. `C60-COL-ENERGY` logs the anchor,
window length/count, seed, recomputed module names, and verified count. Original
tau/gamma, lambda, energy tau multipliers, support and early-growth requirements
remain enforced; no learned calibration is recomputed.

In `rank1_solve.SubjectCapture._linear`, the value correction uses `.to(output)`
before `index_copy`, transferring both dtype and device while preserving its
gradient. A CUDA-less CPU/meta-device regression intercepts the actual
`index_copy` operation and checks that target, row indices and source share the
output device. Its CPU case also verifies selected rows and correction gradients.

## Box commands

Stage the harness and dependencies, including `c52.py` and `c59.py`, under
`/workspace/malleable/v59/` and run from `/workspace/malleable` on the CUDA host.
These are the C59 commands unchanged except for the `v59/` script prefix.
The existing output names are intentionally retained. Check both
`C60-COL-ENERGY` and `C59-CALIBRATION` lines before interpreting J/F.

```bash
/venv/main/bin/python -u v59/e2e_fsa.py \
  --joint-banks out/R20_unit_p0/final.pt out/R20_unit_p1/final.pt out/R20_unit_p2/final.pt \
  --method fsa_forced --fmt ternary --optim bop \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --layers 12,14,17,20 --people 3 --relation-world \
  --steps 150 --seed 0 --joint-schedule-seed 0 \
  --p-iso .75 --distractor-mode all --max-distractors 2 \
  --presence-dropout .2 --presence-target base \
  --answer-only --aug --fact-batch 40 --micro 1 --budget-freeze -1 \
  --local-lambda 10 --anchor data/anchor-wikitext.json --anchor-len 96 --anchor-bs 2 \
  --sparse-state --no-churn --expert-checkpoint --fast-moe exact \
  --eval-set data/eval-wikitext.json --eval-windows 40 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --in-topic-panel data/in-topic-725B.json \
  --fact-class-metrics --decisive-eval --qa-n 120 --joint-negative-n 1000 --gen-bs 1 \
  --patch-dir out/C52_Ju_v58 --out out/C52_Ju_v58.json

/venv/main/bin/python -u v59/e2e_fsa.py \
  --joint-banks out/R20_unit_p0/final.pt out/R20_unit_p1/final.pt out/R20_unit_p2/final.pt \
  --joint-frozen-distractors \
  --method fsa_forced --fmt ternary --optim bop \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --layers 12,14,17,20 --people 3 --relation-world \
  --steps 150 --seed 0 --joint-schedule-seed 0 \
  --p-iso .75 --distractor-mode all --max-distractors 2 \
  --presence-dropout .2 --presence-target base \
  --answer-only --aug --fact-batch 40 --micro 1 --budget-freeze -1 \
  --local-lambda 10 --anchor data/anchor-wikitext.json --anchor-len 96 --anchor-bs 2 \
  --sparse-state --no-churn --expert-checkpoint --fast-moe exact \
  --eval-set data/eval-wikitext.json --eval-windows 40 \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --in-topic-panel data/in-topic-725B.json \
  --fact-class-metrics --decisive-eval --qa-n 120 --joint-negative-n 1000 --gen-bs 1 \
  --patch-dir out/C52_Fu_v58 --out out/C52_Fu_v58.json
```

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q --tb=short \
  experiments/end_to_end/tests/test_c60.py \
  experiments/end_to_end/tests/test_c59.py \
  experiments/end_to_end/tests/test_c52.py \
  experiments/end_to_end/tests/test_c53.py
```

**78 passed in 6.85 seconds.** Coverage includes BASE replay for each of three
resident banks, partially and fully saved energies, mismatch rejection,
legacy manifest window-length recovery and identity rejection, preserved RNG
and masters/EMA/scores, missing original settings, strict tau-scale/lambda
requirements, real J and F continuation/export, single-bank continuation,
collector exception cleanup, and CPU/meta `index_copy` placement/gradients.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q --tb=short experiments/end_to_end/tests/test_c58.py
```

**15 passed in 2.91 seconds.** Running C58 after C53 in the same process causes
two existing teacher-recovery failures (FP32 and BF16), with 37 passing tests.
Both failures reproduce with C60's `.to(output)` change removed in memory;
the separate C58 run passes. That existing test-order issue is outside these
minimal crash fixes.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q --tb=short \
  experiments/end_to_end/tests/test_e2e_fsa.py \
  experiments/end_to_end/tests/test_c44.py \
  experiments/end_to_end/tests/test_c45.py \
  experiments/end_to_end/tests/test_c49.py \
  experiments/end_to_end/tests/test_c50.py
```

**344 passed, 1 failed in 111.03 seconds**, with one existing tensor-to-scalar
warning. The failure is
`test_c45.py::test_every_model_entrypoint_has_flag_and_uses_shared_builder`:
the concurrent, protected `composition_contract.py` lacks the `.announce(` text
required by that static test. This was also documented in C59; the C57 file
was left untouched.

Python syntax and changed-file whitespace checks pass. The box commands are
checked against the original C59 commands, Bash syntax, the harness parser and
C52 validation. `.gitignore` adds only C60 allowlist entries for the report,
regression file and previously ignored `rank1_solve.py`.

No GPU/R20 experiment was run locally. No `git commit` was run. The concurrent
C57 files `build_in_topic_panel.py`, `serialize_patch.py`,
`composition_contract.py`, and `branch_eval.py` were not edited. Other existing
workspace changes were preserved.
