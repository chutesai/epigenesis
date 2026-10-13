# C44: training recovery and heartbeats

Implemented opt-in `--ckpt-every N`, `--ckpt-minutes M`, and `--resume auto` in `e2e_fsa.py`. Both cadence options default to zero. Recommend `--ckpt-every 100 --ckpt-minutes 60 --resume auto` from the first launch, then repeat the same command after interruption. The first cadence reached saves at a completed training-step boundary. A checkpoint-enabled run also saves its final training step, before final evaluation.

`<patch-dir>/resume.pt` is the latest snapshot only. Writes use a temporary file in the same directory, flush/fsync, and `os.replace`; failed writes preserve the prior checkpoint and remove the temporary file. Resume validates the parsed argv/config SHA-256 before building the model, normalizes input paths and ignores `--resume`, `--out`, `--ckpt-every`, `--ckpt-minutes`, and `--patch-dir`, refuses mismatches, and prints `RESUMED {"step": ...}`. Keep the same explicit `--patch-dir` if changing `--out`; otherwise the default checkpoint directory follows the output stem.

The snapshot includes registered adapter parameters/buffers and unregistered tensor/scalar/container state: sparse indices, masks/support, masters/codes, score/energy EMAs, Bop evidence/weights/tstat, birth/death/age and C41 diagnostics, LoRA/fold parameters, resident overlays, optimizer moments/groups, and outstanding gradients. It saves accumulation and phase cursors, calibrated tau and netcost lambda, adaptive local lambda, exposures, queues and the current batch, failing-fact promotion state, trajectory, learning curves, Pareto/best state, and evaluation-checkpoint bookkeeping. All Python, NumPy, torch CPU/CUDA RNG states and private batch/composition/anchor/binding/sibling/chaos/saturation/curriculum/burst streams are restored after deterministic setup. Unit-burst history, crossing evidence, per-fact attribution and activation-noise accounting are included. Input-file SHA-256 fingerprints (including resident patches and the export manifest) and resident bookkeeping must match the reconstructed inputs. There is no separate scheduler object in this training path; its step-based schedules resume from the saved step and configuration.

Factorial LR search saves completed trials and the active candidate's entire partial training snapshot in the root `resume.pt`. Resume skips finished candidates, reconstructs the current candidate and resumes its step; stable candidate-order validation-only selection is unchanged. A temporary child `resume.pt` is used to hydrate an interrupted candidate and removed afterward. The coordinator retains only one durable latest checkpoint. It prints the final selected `RESULT` as well as the existing candidate results.

A central process-wide monotonic-clock heartbeat rate-limits iteration messages to once per 120 seconds; model builds emit explicit before/after messages. C42 selector fitting loops also heartbeat. Calls at each progressing inner iteration emit on the first opportunity after that interval, including across phase changes. Logs have `HEARTBEAT {"phase": ..., "done": ..., "total": ..., "t": ...}`; `t` is process elapsed seconds. Unknown generator lengths use JSON null for total. Coverage includes training microbatches and steps, wake/allocation passes, teacher-forced evaluation, generation tokens/batches (including val/test GEN, conv and QA), MCQ, damage windows/subject text/dark audit, benchmark questions/choices, shared residual feature capture, branch arms/routing/damage, decisive panels and residual generation, two-hop pointer training/cleanup/preservation, and fast-MoE benchmark repeats/warmups. A blocked model call, GPU kernel or disk write cannot emit a cooperative heartbeat; the absent heartbeat is the hang signal. When iterations complete within 180 seconds, the 120-second rate limit gives the requested maximum 300-second gap while progressing.

## Entry-point recovery coverage

| Entry point / stage | Resumable? |
| --- | --- |
| `e2e_fsa.py` value/repair training with Bop, Adam/AdamW or SGD | Yes, at completed steps, including partial Bop accumulation and Adam-to-Bop transitions. |
| `e2e_fsa.py --factorial-lrs` | Yes, between candidates and within the active candidate. |
| `e2e_fsa.py` resident training | Yes; frozen overlays and private sibling RNG/bookkeeping are saved. |
| `e2e_fsa.py --optim mls` | **No.** Checkpoint/resume options explicitly reject MLS; its covariance solver state is not serialized. |
| Wake/allocation/calibration/setup before value training | **No independent mid-stage recovery.** These stages replay before restoring a value-training checkpoint. |
| Final evaluation, or evaluation-only `e2e_fsa.py` modes | **No evaluation cursor recovery.** Final evaluation can replay from the completed-training checkpoint. Pure evaluation modes do not create training checkpoints. |
| `branch_eval.py` | **No.** Heartbeats only; selector training and evaluation arms replay. |
| `two_hop.py` | **No.** Heartbeats only; pointer training and evaluation replay. |
| `decisive_eval.py` suite | **No independent recovery.** Heartbeats; reruns during final evaluation. |
| `bench_fast_moe.py`, residual feature collection | **No.** Heartbeats only. |

C46 supplies the queue runner and hang janitor in `ops/`; only checkpoint-enabled e2e training resumes, while other entry points replay.

## Validation

CPU process tests use the real main/train/eval path with the repository's tiny MoE/tokenizer loader fixture. Each interrupted process uses the original N-step horizon, exits immediately after a durable checkpoint at K, and a separate process resumes to N. Changing `--steps` from K to N is deliberately refused: it changes the configuration and final-step/annealing behavior.

The tests compare final tensor storage bits, the complete trajectory and JSON result bytes, printed `RESULT`, optimizer/diagnostic/cursor/RNG checkpoint state, and every final candidate patch. Cases cover sparse Bop with support refresh, calibrated tau and C41 tstat; LoRA/Adam; factorial LR search interrupted inside its second candidate with composition and private anchor draws; resident training; outstanding Bop accumulation; Adam-to-Bop transition; random unit bursts; adaptive netcost/local penalty, saturation and nonempty Pareto state; and train-only LoRA teacher distillation/fade. Separate checks cover checkpointing versus disabled checkpoints, missing/mismatched resume, step/time cadence, failed atomic writes, all global RNG streams and shared heartbeat rate limiting.

Final full CPU regression run: **1,052 passed, 1 skipped**, one existing scalar-conversion warning, in 309.65 seconds (Python 3.12, PyTorch 2.8.0). This includes **13 C44 checks**, with nine independent-process recovery cases. `py_compile` passed for all changed Python entry points/helpers, and `git diff --check` passed.

Equivalent test invocation with dependencies available:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/end_to_end/tests
```

Dependencies were supplied from the existing local cache by a temporary launcher. An exploratory PyTorch 2.11 CPU run exposed an existing `test_masked_solve.py` CUDA mock incompatibility; that test and the complete suite pass on the cached PyTorch 2.8 environment.

 CUDA/8B recovery was not executed on this CPU host. The guarantee is mechanically exact state restore on GPU; GPU kernels nondeterministic. GPU bit-exact numerics are unproven. `max_alloc_gb` is allocator telemetry excluded from numerical exactness, with its running peak restored from the checkpoint. No GPU performance or learning improvement is claimed.

No git commit was run. Existing working-tree changes were preserved.
