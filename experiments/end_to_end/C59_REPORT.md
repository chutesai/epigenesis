# C59: restore calibration for joint continuation

C52 J/F now restores each bank's recorded absolute Bop threshold and terminal
gamma instead of calibrating on the converged bank's first answer gradient.
Saved support stays fixed for the entire continuation: `budget_freeze=-1`, no
refresh, and births are gated by that support even when the original budget was
zero. Saved grace, death-threshold multiplier, EMA behavior, energy threshold
multipliers, column energy, netcost lambda, masters, support scores and ages are
restored when present. Legacy inference-only banks retain their saved live
coordinates as support and start with zero EMA; imported slots are mature when
grace is enabled. Their original threshold must still be available.

The shared implementation is `c59.py`; the entrypoints are `c52.py` and
`e2e_fsa.py`. New regular Bop checkpoints carry `meta.bop_calibration` plus
`continue_state`; new joint checkpoints retain `joint_state` version 1 with a
calibration record and the additional training tensors. Inference entries keep
the existing format. These training payloads increase physical checkpoint file
size; serving-code storage estimates continue to describe inference storage.
Dense saved banks can be converted to sparse joint state, including mapping
per-row energy tau multipliers to their corresponding reserve slots.

Calibration recovery checks saved continuation/joint calibration, matching-step
C44 `resume.pt`, then saved metadata or the original run's JSON/manifest. For
`out/R20_unit_p0/final.pt`, keep `out/R20_unit_p0.json` (and, if available,
`out/R20_unit_p0.manifest.json`) beside the original run directory. Equivalent
`manifest.json`/`meta.json` inside the bank directory are accepted. A recorded
positive `bop_tau_abs` is usable; the CLI sentinel `bop_tau_abs=0` in an older
manifest is not. Calibration identity fields and any supplied checkpoint hash
are checked. Missing tau, required energy tensors, or netcost lambda fail
clearly rather than recalibrating. Original annealing and late-gamma schedules
resolve to the saved terminal thresholds; continuation does not restart them.
Saved calibration is authoritative over the joint command's Bop flags.

Both continuation paths abort if nnz exceeds 120% of the loaded bank's nnz
on updates 1–10 (inclusive). The error identifies the bank/path, update, baseline
and current count before a final checkpoint is exported. Exactly 20% growth is
allowed; zero saved nnz permits no early growth.

The single-patch Bop continuation path restores the same calibration/state and
freezes support. A recorded step-zero S2 analytic rank-one patch without a Bop training recipe
starts its first Bop run using the existing initial calibration. Other step-zero
initializers require an explicit absolute threshold. It still gets frozen
imported support and the early-growth guard. This preserves S2 polish's intended
first-training behavior without allowing converged Bop banks to recalibrate.

## Corrected J/F box commands (v58)

Stage the current harness and dependencies in `/workspace/malleable/v58/`,
including **both `c52.py` and the new `c59.py`**. Run from
`/workspace/malleable` on the CUDA host. Use the disjoint whole-unit R20 banks,
not the overlapping older FULL-bank reserves. Calibration comes from each
original bank/run record; no common replacement tau or recalibration multiplier
is supplied. Budget/support and original world/scope are inherited per bank.
The explicit `--budget-freeze -1` documents the saved-support policy and is also
enforced by the loader. Output names preserve the failed launch artifacts.

```bash
/venv/main/bin/python -u v58/e2e_fsa.py \
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

/venv/main/bin/python -u v58/e2e_fsa.py \
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

Check the three `C59-CALIBRATION` lines before interpreting either run. J/F must
restore identical per-bank calibrations and have identical schedule hashes and
counts. The early guard is a corruption/failure detector, not a claim that loss
is universally monotonic during mixed-bank training.

## Validation

The tiny-fixture regression trains three banks with real Bop to a fixed point
on their own facts, exports and reloads them, then executes both real J and F
runners with `P_iso=1`, no presence dropout and all own facts per update. For all
10 continuation updates, per-bank nnz stays **64 / 58 / 64**, own-fact loss
stays **5.8862356**, and every saved support tensor stays identical. The fixture's
recorded tau/gamma are restored despite different command defaults. Other tests
cover original-run calibration recovery, invalid/missing state, dormant support,
energy/netcost/grace round trips, dense-to-sparse row thresholds, the exact guard
boundary, and injected runaway births aborting both real entrypoints before
export. The existing C3 age fixture now retains its source budget during continuation
and starts from its first Bop update with all 20 supported codes live. It no
longer changes the saved budget to zero to enable accumulation; the existing
separate accumulation tests retain that coverage.

No GPU/8B run or R20 result is claimed. No `git commit` was run. The concurrent
C58/C57 files named in the request were not edited.

Validation commands/results:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q --tb=short \
  experiments/end_to_end/tests/test_c59.py \
  experiments/end_to_end/tests/test_c52.py \
  experiments/end_to_end/tests/test_c53.py \
  experiments/end_to_end/tests/test_e2e_fsa.py \
  experiments/end_to_end/tests/test_c44.py \
  experiments/end_to_end/tests/test_c45.py \
  experiments/end_to_end/tests/test_c49.py \
  experiments/end_to_end/tests/test_c50.py
```

- Broad run: **410 passed, 1 failed** in 111.74 seconds. The failure is
  `test_c45.py::test_every_model_entrypoint_has_flag_and_uses_shared_builder`:
  concurrently edited `composition_contract.py` lacks the `.announce(` text
  required by that static scan. That protected C57 file was left untouched.
  One existing tensor-to-scalar warning remains.
- Final focused C59/S2 run: **42 passed** in 4.75 seconds after tightening the
  initial-Bop exception and validating dense row-threshold conversion.
- Both box commands pass Bash syntax, the actual harness parser and C52
  validation. Python syntax and whitespace checks pass for the C59 changes.
- `.gitignore` gained only the three C59 allowlist entries; pre-existing
  concurrent changes were preserved.
