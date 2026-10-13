# C47: gap-ladder B1 and B2

Implemented review §4 B1 and B2 in `e2e_fsa.py`. B3 (quinary Bop) is not implemented. Existing working-tree changes were preserved. No git commit was run.

## Implementation

- `--fmt quinary` projects masters with `round(2m).clamp(-2, 2)/2`, producing `{0, ±alpha/2, ±alpha}`. Ternary is the even sub-lattice. The existing STE reaches off-support masters. Quinary netcost scoring uses the minimum nonzero magnitude 0.5, as the existing half-step format does. Exports contain int8 codes `k` in half-alpha units with `fmt="quinary"`; both patch loading and resident overlays divide by two. Dense accounting uses log2(5) bits/slot and sparse accounting uses two value bits/nonzero. Bop continues to assert plain ternary and explicitly says quinary-Bop is unimplemented.
- `--patch-scale rowlearn` requires sparse FSA under Bop. Per-linear fp32 training parameter `logr[out_features]` starts at `log(alpha_row)`. Free slots use `exp(logr)[rows]`; occupied flips retain `-2w`. Slot costs use the current learned magnitude, detached from autograd. Bop updates M through the existing transition logic. A separate no-decay AdamW group updates only logr each training step, independently of Bop accumulation; logr is excluded from the lr-zero SGD placeholder. Pareto snapshots include logr, and resume checkpoints include its optimizer moments.
- Rowlearn saves fp16 logr with `scale="rowlearn"`. Loading casts back to fp32; scale rounding is bounded by `expm1(max |logr - fp16(logr)|)` relative error. Resident overlays decode the same scales. Scales multiply only the patch, so revocation returns the exact base weights/logits. Zero-alpha rows retain log scale negative infinity and exact zero magnitude.
- `fsa_storage(..., n_scales, scale_bits=32)` retains legacy block-scale accounting; rowlearn passes 16. Scale overhead is exactly two bytes times the number of patched output rows in dense, sparse and entropy estimates. `patch_bytes` remains the minimum nominal dense/sparse representation; `patch_file_bytes` includes PyTorch serialization overhead and actual indices/metadata.
- Effective rowlearn scale LR defaults to 1e-3, and is recorded in the training recipe, patch metadata, result and comparison manifest. The parser retains the existing `--scale-lr 0` automatic sentinel so legacy block scales still inherit the main LR. Default fmt, scale mode, optimizer settings, backend recording and other gates are unchanged. Sparse block scales, energy tau and expert profiling remain rejected. Specialized ternary/row-only feature gates are not broadened.

## Validation

C47 focused tests: **17 passed**. They cover quantization/grid nesting, STE gradients on unsupported slots, dense/sparse bit-exact quinary roundtrips, resident decoding, dtype/storage formulas, two-step arm-C ternary and arm-D quinary Adam sparse-budget training, cap enforcement, hand-computed row-scale delta/gradient/cost math including occupied flips, frozen-scale Bop births/deaths and real control training, fp16 roundtrip bounds, exact revoke, optimizer separation and validation gates.

A separate before/after comparison used a copy of the pre-edit harness, running sparse Bop, dense Adam, and dense Adam with block scales for four steps each with recovery checkpoints. All three produced **byte-identical final.pt and result JSON**, identical checkpoint state, and identical Python/torch RNG state. All parser defaults matched exactly.

All **12** process interruption/resume cases passed, including the three added cases: rowlearn with explicit scale LR .002, rowlearn during Bop gradient accumulation, and sparse-budget Adam quinary. The resumed result JSON, final patch state and complete recovery state match uninterrupted runs bit-exactly on CPU.

Resident decoding continues to accept legacy entries omitting `fmt`, treating them as ternary. Final full-suite sweep: **1,132 passed, 1 skipped**, with one existing scalar-conversion warning, in **336.99 seconds**.

Python compilation and `git diff --check` pass. All four queue lines were parsed against the harness parser and compared to the review: each Dq line differs from its matched GL_C line only in fmt, and Drs differs from CTRL only in patch_scale/scale_lr (apart from v55).

Validation environment: Python 3.12.14, PyTorch 2.8.0, CPU. Full-suite command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q experiments/end_to_end/tests
```

No CUDA device or 8B model is available on this host. The commands below are prepared for the GPU harness; no GL_D experiment scores are claimed.

## Exact GPU queue command lines (v55)

These lines use the review §3 GL_CTRL_s1 command verbatim except the harness prefix and each treatment. The `|name` suffix is the existing queue format, not a shell pipe. GL_Dq changes only the Adam optimizer/LR and quinary format relative to CTRL; compare its precision effect against the matched-LR GL_C arms. GL_Drs retains Bop and adds rowlearn with scale LR 1e-3.

```text
v55/e2e_fsa.py --method fsa_forced --optim adam --lr 0.005 --fmt quinary --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0.2 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1|GL_Dq_lr005_s1
v55/e2e_fsa.py --method fsa_forced --optim adam --lr 0.02 --fmt quinary --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0.2 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1|GL_Dq_lr02_s1
v55/e2e_fsa.py --method fsa_forced --optim adam --lr 0.05 --fmt quinary --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0.2 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1|GL_Dq_lr05_s1
v55/e2e_fsa.py --method fsa_forced --optim bop --bop-gamma 0.05 --bop-tau 0.7 --patch-scale rowlearn --scale-lr 1e-3 --local-lambda 10 --support-score netcost --cost-rel 1 --rehearsal-frac 0.2 --no-churn --budget 1000000 --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 --val-facts 60 --conv-test --conv-facts 60 --steps 400 --eval-every 100 --micro 6 --no-pareto --sparse-state --expert-checkpoint --layers 3,16,22,29 --slot-frac 0.5 --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 --answer-only --eval-set data/eval-wikitext.json --aug --extra-eval-sets data/eval-fineweb.json,data/eval-code.json --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1|GL_Drs_s1
```

## Selection and reporting protocol

Run the three quinary LRs at equal slots (budget 1M), and report all three final-step `trajectory[-1].val_gen` numbers. Choose the winner using only final-step validation GEN; tie-break by lower `trajectory[-1].val_dnll`, then lower LR. State in the GPU run log that test fields were not read during selection. Compare each quinary run to its corresponding GL_C LR arm, and compare the separately validation-selected Dq winner with the validation-selected GL_C winner. Compare Drs with GL_CTRL_s1. Keep `--no-pareto`, master clip 0, refresh every 10 and freeze 200; score the final step.

For every run report final test GEN, conv GEN, final val GEN, WikiText/FineWeb/Code dNLL versus base, nnz, realized support size from REFRESH events, patch_bytes, storage.fsa.pos_bits, births/deaths/evictions totals, cost_lambda and wall time. Report rowlearn's scale overhead as `2 * sum(entry.shape[0])` bytes. Record the deployed v55 code-tree hash in each result directory and preserve the existing eda_backend fields (default naive_fp32).

If Dq wins, add an equal-bytes budget only as a secondary result using `K_quinary = K_ternary * (pos_bits + 1)/(pos_bits + 2)` with pos_bits from GL_CTRL storage; do not substitute this for the equal-slots headline. B3 remains deferred until B1 provides evidence that precision matters.
