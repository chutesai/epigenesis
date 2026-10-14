# C64: patch-training speed profiler

Implemented `profile_patch_train.py`; no performance fixes or training-source edits. The reported live-job observation is **86–240 seconds per Bop update, roughly 20% GPU utilisation, one saturated CPU core**. No GPU box was available in this workspace: those observations are supplied context, not measurements from this profiler. CPU regression results are below. Real phase timings and speedups remain to be measured on the box.

## Exact box command

Run from `/workspace/malleable`. The profiler and its matching harness dependencies belong in `v60/`; it imports `v60/e2e_fsa.py` through Python's script-directory search path. Data paths remain relative to `/workspace/malleable`. This preserves the complete GL_ECTRL_s1 recipe argv, including `--steps 800 --eval-every 100`; the adapter stops after 20 updates and removes evaluation/checkpoint code from the training loop independently of those flags.

```bash
cd /workspace/malleable
/venv/main/bin/python -u v60/profile_patch_train.py \
  --profile-updates 20 --profile-out out/C64_ECTRL_s1_profile -- \
  --method fsa_forced --optim bop --bop-gamma 0.05 --bop-tau 0.7 \
  --local-lambda 10 --support-score netcost --cost-rel 1 \
  --rehearsal-frac 0.2 --no-churn --budget 1000000 --budget-freeze 200 \
  --people 3 --facts-per-person 100 --shared-attrs 30 --fact-batch 40 \
  --val-facts 60 --conv-test --conv-facts 60 --steps 800 --eval-every 100 \
  --micro 6 --no-pareto --sparse-state --expert-checkpoint \
  --layers 3,16,22,29 --slot-frac 0.5 \
  --anchor data/anchor-wikitext.json --anchor-bs 2 --anchor-len 96 \
  --answer-only --eval-set data/eval-wikitext.json --aug \
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json \
  --train-audit 120 --strata --fast-moe exact --base-cache base_cache --seed 1 \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --patch-dir out/C64_ECTRL_s1_unused_patches --out out/C64_ECTRL_s1_unused.json
```

Recipe provenance: `C64_ECTRL_ARGV.txt` matches the GL_ECTRL_s1 row in `reviews/experiments/gl_ectrl_drs_test/raw/ladder_design.md:160`. That design row uses the historical `v54/` script prefix; the requested reference invocation uses `v58/e2e_fsa.py` with the same recipe tokens, plus explicit export/backend/output arguments shown above. To reconstruct that full reference command, replace `v60/profile_patch_train.py --profile-updates 20 --profile-out out/C64_ECTRL_s1_profile --` with `v58/e2e_fsa.py` and use `--patch-dir out/GL_ECTRL_s1_patches --out out/GL_ECTRL_s1.json`. The 800-step, seed-1 result is independently described in `experiments/epigenesis/LEADERBOARD.md:47`. Output paths for C64 intentionally avoid the ladder artifacts.

Outputs: `out/C64_ECTRL_s1_profile.json`, `.md`, `.pstats`, and one `.updateN.trace.json` per update plus a separate `.eval21.trace.json`. JSON/Markdown are refreshed after each completed session. Setup time and source SHA-256 are in JSON metadata. There are no patch saves. The `--recipe-file` option also accepts the ladder row, strips the script prefix and `|GL_ECTRL_s1` label, and supplies an unused `--out` if absent.

## Measurement contract

The profiler imports `e2e_fsa`, copies `main`'s AST into a private namespace, and makes these structural changes only: removes baseline/pristine evaluation; bounds the outer loop with `min(total_steps, 20)`; cuts the periodic evaluation/checkpoint tail; replaces the post-training scoring/export tail with one timed evaluation pass. It inserts timing scopes around the retained code and wraps imported forward helpers, optimizer zeroing/SGD step, and in-process `torch.autograd.backward`. Missing expected AST boundaries fail explicitly. Loss conditions match by `ast.dump` of the test and parsed expected expression, ignoring redundant parentheses and source locations. All five expected loss conditions (rehearsal, local/gate penalty, drift, preservation KL and route KL) must occur in the retained training prefix; otherwise adaptation raises with the missing expressions before compiling or running training. Bop, support selection, parametrizations, gradient scaling, loss expressions, RNG seeding, rehearsal, anchor/local penalties, micro-batches and expert checkpointing all use the original implementation.

`a.steps` and `total_steps` remain 800. This avoids changing annealing denominators, the final-update Bop accumulation condition, or schedule boundaries. Support refreshes occur after updates **1 and 11** in this prefix (`refresh_due` uses 1, 1+N, …), with freeze still 200. Periodic audits are omitted with evaluation; the ECTRL defaults do not use audit-driven training weights or promotion. Token pools and panel preparation still run during setup; batch resampling/token tensor construction is measured per update.

| JSON phase/metric | Meaning |
| --- | --- |
| `data_tokenise_wall_s` | Original `make_batch`, including batch sampling and `batchify`; initial pool tokenization is setup time. |
| `forward_wall_s` | Forward and loss construction for facts, rehearsal and local penalties, excluding nested backward/data/optimizer scopes. |
| `backward_wall_s` | Original autograd backward, including checkpoint replay. |
| `bop_bookkeeping_wall_s` | Remaining original training-loop work: tau calibration, score EMA, Bop births/deaths, net-cost calibration, support refresh, age/event bookkeeping and training logging. |
| `optimizer_wall_s` | Original zero-grad and SGD step, if invoked. Ordinary Bop updates have no SGD step; Bop itself is in bookkeeping. |
| `eval_wall_s` | One separate no-grad acquisition pass (`with_train=True`, including test and conversational generation), plus WikiText/FineWeb/code NLL panels. No baseline or damage deltas. |
| `naive_eda_wall_s` | Inclusive `_scan` forward/recomputation wall time; a **subset** of forward/backward/eval, never an additive phase. |
| `cuda_kernel_launches` | Actual device kernel events in each update trace, excluding copies and GPU annotations. |
| `naive_eda_kernel_launches` | Kernel events linked by runtime correlation to launches inside `_scan` ranges or autograd backward ranges matched by forward sequence number. |
| `naive_eda_cuda_kernel_s` | Sum of attributed device kernel durations, distinct from wall time. |
| `unattributed_wall_s` | Session/profiler/timing overhead outside exclusive phase scopes. |

`torch.profiler` enables CPU and CUDA activities on CUDA boxes, with `with_stack=True`; CPU tests enable CPU activities. In-process cProfile covers the update/eval sessions, excluding model load and report processing. Reports include top 30 cumulative cProfile functions, top 30 CPU ops by self time, and top 30 actual CUDA kernels by summed device time. Raw traces retain stacks, sequence numbers and launch correlations for inspection. One trace per update bounds retained profiler events to a single update; naïve EDA can still make each trace very large.

CUDA synchronization at phase boundaries makes wall times attributable but removes overlap; cProfile and stack tracing also slow Python dispatch. Treat these as **instrumented** timings, not unbiased production latency. The first update includes cold-path costs, and refresh updates should be compared separately with ordinary updates. EDA attribution covers the reference Python `_scan` and its autograd-generated backward nodes; it depends on Kineto runtime correlations/sequence metadata. The box adapter refuses CUDA profiling if no `eda_kernels._scan` function can be found. Check trace metadata and counts before treating attribution as complete. Kernels lacking usable correlation metadata can appear in total launch counts without EDA attribution. CPU-only zero launch counts mean CUDA was unavailable, not zero CUDA work on the production model.

Skipping baseline evaluation can alter warm caches. Timing/synchronization can also change GPU execution ordering. No arithmetic is rewritten, but the CPU equivalence regression is not a claim of bit-identical CUDA training: the leaderboard already reports seed-1 Bop divergence from update 2 across jobs.

## Static bottlenecks and minimal fix candidates

These are conditional engineering estimates, **not measured speedups**. For a measured affected wall fraction `f` and phase acceleration `s`, the end-to-end bound is `1 / ((1-f) + f/s)`. Low GPU utilisation alone does not identify which phase owns the idle time.

| Candidate, ordered by likely implementation scope | Static evidence and profiler trigger | Conditional speedup estimate | Numerics risk |
| --- | --- | --- | --- |
| Batch Bop counters and vectorize masked writes | `e2e_fsa.bop_step` around line 2104 loops over every patch module, does many small boolean/pointwise ops, uses dynamic boolean indexing, and calls `int(birth.sum())`/`int(death.sum())` per module. Those scalar reads synchronize CUDA. `--no-churn` removes history arrays but retains aggregate counters. Trigger: bookkeeping wall dominated by scalar extraction, index ops and idle launch gaps. | 2–5× affected bookkeeping; if it owns 40% of wall, 1.25–1.47× overall. | Low for device-side integer totals with one final host read and unchanged EMA/update order. Masked-write fusion needs exact handling of absent gradients, birth/death precedence, EMA reset, legal signs and support. Floating EMA fusion/FMA can change threshold crossings. |
| Reduce per-parameter Python loops and repeated support merges | Main loop around lines 8023 and 8202 updates score tensors per module; first refresh calibrates medians through repeated top-K merges. `refresh_support` around line 1624 does per-module positive-count host reads/top-Ks, `unique().tolist()`, mask reconstruction and eviction counters. `refresh_support_event` around line 5301 adds three more per-module scalar-count passes. Trigger: updates 1/11 markedly slower; cProfile top-K bookkeeping, copies and scalar reads dominate. | 2–4× affected loop/refresh work; at 20% wall fraction, 1.11–1.18× overall. Refresh acceleration affects two of these 20 updates and vanishes after freeze 200. | Low for caching immutable module lists and aggregating integer counts. Medium for regrouping global top-K: preserve tie policy and candidate order, exact medians for cost calibration, support cardinality, eviction/master/EMA resets. `foreach` EMA updates require verifying arithmetic equivalence. Avoid flattening the entire slot pool: the existing streamed selector deliberately limits memory. |
| Larger micro-batches, starting 6→12 where memory allows | `fact_loss_backward` around line 7116 loops over the sampled 40 facts × three augmented phrasings, about 20 fact micro-batches at size 6, each with a forward/backward. Rehearsal separately uses one window/backward, and local anchors add another pass. Trigger: large forward/backward launch counts and CPU launch gaps, with activation-memory headroom. | Ideally up to 2× fewer fact forward dispatches for 6→12; plausible 1.2–1.8× affected fact work, 1.1–1.6× overall if it dominates. Token-time recurrence launches do not necessarily halve if padded lengths grow. | Medium: loss denominator can remain unchanged, but GEMM shapes and gradient accumulation order change; near-threshold Bop flips can diverge. Padding/routing, remainder batches and expert checkpoint memory need checks. Rehearsal batching also changes reduction order. |
| Capture fixed-shape forward/backward segments with CUDA graphs | `fast_moe.py` grouped-routing preparation around line 90 transfers counts to CPU and calls `.tolist()`; exact dispatch retains expert loops. Variable sequence padding/routing, checkpointing, local hooks, per-module scalar reads and dynamic boolean indexing prevent a simple whole-update capture. Trigger: most wall lies between tiny launches while device kernel time is small. | 2–5× captured launch-bound segments; if 60% of wall is removable dispatch overhead, 1.43–1.92× overall for 2–5× segment speedup. | Low arithmetic risk only after capture preserves shapes, buffers, hook/checkpoint behavior and original accumulation order. Medium implementation risk: varying routes, changing support, gradient buffers and batch sampling must be staged outside capture. Graph RNG/state lifetimes require validation. This is a larger change than counter aggregation. |
| `fla_kda_2t` when EDA is launch-bound | The locally readable external model's `mesh/model/eda_kernels.py:_scan` (around line 330) loops over every token, with decay, erase/read, residual, write and three einsums. `eda_recurrence` dispatches `naive_fp32` directly to this FP32 scan. This must be confirmed against the box implementation recorded by the profiler. Trigger: EDA-attributed launches/device ops dominate forward/backward and show per-token tiny-kernel dispatch. The harness already exposes `--eda-backend fla_kda_2t` through `eda_backend.py`. | Plausible 3–10× affected EDA work; at 70% wall fraction, 1.88–2.70× overall. No claim about the actual box without measuring backend forward **and backward**. | High relative to preserving this exact recipe: accelerated recurrence changes precision/reduction order; the external EDA wrapper casts q/k/e to the value dtype for accelerated CUDA backends. Test outputs, gradients, births/support and acquisition/damage over seeds. Backend provenance and cache identity are intentionally strict. Treat as a separate recipe experiment, not a numerically identical substitution. |

First inspect normal updates versus 1/11, the CUDA total/EDA launch counts, self CPU operator time versus kernel time, and cProfile `_scan` versus `bop_step`/`refresh_support`/routing. If EDA dominates, vectorizing Bop cannot explain an order-of-magnitude gain. If bookkeeping dominates, start with integer counter aggregation and immutable-loop caching before changing the EDA backend. None of these fixes has been implemented.

## CPU verification

```bash
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q tests/test_c64.py
```

Eleven tests pass on CPU with PyTorch 2.8.0: real tiny-MoE imported training matches ordinary `E.main` exactly for masters, EMAs and support under net-cost scoring, 20% rehearsal, local loss, sparse state and augmentation; update/eval separation and output artifacts are checked; baseline evaluation and patch saving are forbidden in a second prefix run; a synthetic trace checks forward/backward CUDA launch attribution and excludes memcpy; an actual CPU `_scan` scope checks profiling and restoration; the ladder recipe and unchanged imported Bop/support helpers are verified; a real 20-update tiny-model prefix preserves the 800-step horizon and refreshes exactly at updates 1 and 11. No CUDA timing, graph/backend experiment, or production-speedup assertion is made.

## C64b review fix

Addressed the blocking issue in `C64_REVIEW_FIX.md`: `ast.unparse` drops the redundant parentheses in `(local is not None and lam > 0) or gate is not None`, so the former exact-string matcher silently left that branch in bookkeeping. The normalized AST match now places the entire local/gate branch inside `forward`, including anchor sampling, hook setup/cleanup, scalar loss reductions and their host synchronization. Nested backward/data/optimizer scopes retain exclusive accounting.

The new regression captures the transformed AST passed to `compile` from the real `e2e_fsa.main` source and checks that all six matching branches (route KL appears twice) have an immediate `forward` wrapper. It does not use `inspect.getsource` on the adapted function, which returns the original harness. Five additional parameterized cases remove each expected condition from the real source and require an explicit error naming it. All six new cases failed against the old adapter, including the local-penalty wrapper assertion, and pass after the fix. The review's remaining recipe, clock, output and command checks were passes; no additional defects were listed. The 8B/CUDA profile has not been launched.
