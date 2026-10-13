# C51: corrected census energy and confirmation checks

Implemented `reviews/brainstorm_2026-10-12/tasks/c51_brief.md` in `census.py`,
with tiny-fixture tests. Existing working-tree changes were preserved. No
`git commit` was run.

## Measurements and dispatch

For each routed unit, `phi = relu(z)^2`. The census retains C49's raw
`routed`, `fired`, `relu2_sum`, and `down_norm` fields and adds `relu4_sum`,
`weighted_relu4_sum`, and `tail`. All accumulators and activation reductions
are fp64. Conditional `energy` is now the corrected
`||down[:,j]||^2 * sum(relu(z_j)^4) / routed_count`. The former formula is
available as `legacy_energy`. `overall_energy` is also corrected; the old
frequency and mean-ReLU-squared fields remain.

`unconditional_energy` divides the sum of
`w_e(x)^2 * ||down[:,j]||^2 * relu(z_j)^4` by **all stream tokens**, including
those not dispatched to that expert. This incorporates both routing probability
and the actual expert weight. Conditional frequency/activation/tail statistics
use routed tokens; the corresponding `unconditional_*` fields use all stream
tokens. These unconditional frequencies are event probabilities and are not
multiplied by gate weights. Energy includes the squared gate weight. Energy is
per unit in the expert output coordinate space, before any latent-up projection,
and does not include cross terms between units or experts.

The model sources read were
`reviews/brainstorm_2026-10-12/mlf_ref/router.py:830-984` and
`mlf_ref/moe.py:6761-6802,7926-8102`. The sigmoid router selects top-k using
logits plus balancing bias in its ordinary global-top-k mode; magnitude uses
raw sigmoid gates, divided by the selected-gate sum (clamped at `1e-6`), then
multiplied by `scaling_factor`. The census reads dispatch **after** this
selection/scaling; it does not reconstruct router decisions. Each result records
actual top-k, router selection mode and scaling factor per layer, so no fixed
value of k is assumed.

The fallback dispatch groups assignments by expert, evaluates each expert on
ascending unique token IDs, sums duplicate assignments' weights for that token,
then multiplies its output by this summed weight. The census uses that same
summed weight before squaring and counts each expert/token once. Tail padding
introduced by `_run_expert` is omitted. As in the harness's
`benchmark_dispatch`, the census disables fused frozen/trainable grouped GEMMs
and DeepGEMM in favor of the model's per-expert fallback, which exposes projection
hooks. It restores all overrides on exit, including exceptions. Unsupported
backends that bypass these hooks fail explicitly. The simple tiny fixture uses
its existing unweighted, Linear/argmax top-1 route.

A separate calibration pass estimates each layer's pooled p99 of `phi`, including
zeros, over the first up-to-1M **selection-A** stream tokens. Domain prefixes are
balanced, with the same context and main batch size as the census. Calibration
never uses B and does not add to census counts. The bounded histogram has a
zero bin and 2048 log2 bins from `2^-32` to `2^32`, plus overflow. Tau is the
upper edge of the p99 bin (roughly 2.2% bin width); a p99 in overflow fails rather
than silently truncating. The recorded threshold remains fixed for the counted
pass, which reports conditional and unconditional `P(phi > tau)` with a strict
inequality. The summary records tau, estimator, requested/actual calibration
size and domain targets. Small fixtures use the available A prefix.

## Observation, A/B confirmation, and diagnostics

`--min-routed` defaults to 2000. An expert/domain cell below that count is
unobserved; its derived statistics are NaN even if it routed a few tokens.
Raw counts/sums remain available. Rankings require observed cells in **every**
natural-text domain in the relevant split; no low-count or unrouted expert is
ranked as quiet. Near-dormant and domain-specific counts use the observed
A-union-B cells, with the existing strict domain-layer p1 rule. The summary
reports unobserved cell counts globally and per domain, both combined and for
A/B, along with routed counts. Unknown-unit counts remain separately available.

`--split-fraction` defaults to 0.5 and specifies B's share of each domain budget.
B is the suffix of each source; its window count is rounded down, and A receives
the remaining prefix. Batches stop at this boundary. `layerN_rank` is ordered by
A's overall unconditional energy, pooled using actual stream-token counts.
`layerN_rank_B` independently ranks B. Both index the original full expert/unit
mapping arrays and exclude unobserved experts. Bottom fractions are
`floor(fraction * observed_ranked_units)` with stable expert/unit tie breaking.
The summary reports Jaccard and set/intersection counts for bottom 10% and 20%.
An empty union has null Jaccard. NPZ fields with `_A`/`_B` retain each split's
raw counts and derived statistics.

The separate `factqa` stream uses `make_world` and `build_items` from `e2e_fsa`,
all twelve `TRAIN_AUG_T` training templates (including training QA), and no
validation/test templates. Default seed 510051 is reserved for development
(seeds below 100000 are rejected). Four synthetic dev entities have 40 facts
each; their census-only `CensusDev510051 PersonNNNNN` namespace is disjoint from
both the ordinary test people and synthetic training names. This produces 1920
prompts. Each is teacher-forced independently at batch one, without padding or
packing. The report includes seed, entities, templates, actual token count,
per-expert routed counts, routing share across expert/token dispatches, and
routing probability per prompt token. At the last prompt token that predicts
the first answer token, it records how many A-selected bottom-10% units belong
to dispatched experts, per prompt and as a distribution. This is dispatch
reachability, irrespective of whether those units fire. The total number of
selected units is recorded, including zero when no units have enough evidence.
Factqa is a small routing diagnostic outside the natural-text token budget;
it does not contribute to the natural-text rankings or dormant classifications.

The batch check independently replays balanced domain prefixes totaling up to
500k tokens, first at batch one and then at the configured batch size. Each run
ranks by unconditional energy and applies the same observation threshold.
Per-layer bottom-decile membership overlap is intersection / batch-one selected
count. The pre-registered pass threshold is **>=0.95 in every layer**. Empty
batch-one selections are unmeasured/null, never an automatic pass. The summary
records the two batch sizes, exact sample counts, membership counts/overlaps,
and per-layer/overall pass status. This check runs even when the main batch is
one. Probe passes do not change the primary census. A failure is reported;
the census does not initiate training or select a subsequent experimental arm.

Existing anchor/eval 32-gram checks also cover the generated factqa sequences.
Natural-text operands are the complete consumed prefix of each source, with
all source grams including cross-context boundaries; factqa operands are each
actual teacher-forced sequence, independently. References are the supplied
anchor and each eval panel, with the existing every-16-position reference-gram
denominator and >1% rejection rule. No cross-prompt grams are manufactured.
Export shards, streams, tokenizer, anchor and eval panel contents are hashed.

Checkpoint version 2 retains combined and A/B counters, stream denominators,
frozen tau/calibration histograms, cursors and RNG state. Identity includes the
new threshold/split/dev-world settings, generated prompts/templates, sample
sizes and tokenizer. C49 version-1 checkpoints are rejected because they cannot
recover fourth moments, dispatch weights or A/B counts. Calibration is replayed
if interrupted before the first measurement checkpoint; diagnostics are replayed
from the final measurement checkpoint after an interruption.

## Exact box command (v56/)

Run from `/workspace/malleable`, with `census.py` and its existing dependencies
staged together in `v56/`, and the four specified uint32 inputs containing
13,000,000 tokens each. This consumes **12,998,656 tokens per domain** (6347
full 2048-token contexts), totaling **51,994,624**. The final 1344 tokens per
file are unused; requesting 52M with equal full-window rounding would exceed
some 13M inputs. Each domain has 6,500,352 A tokens and 6,498,304 B tokens.
The calibration contains 999,424 tokens (249,856/domain); the batch check
contains 499,712 tokens (124,928/domain). Both are additional forwards over
existing prefixes, not additional source-token budget. Factqa is additional.
Use the same recorded EDA backend in later reserve training/evaluation.

```bash
python -u v56/census.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --stream cp_wikimedia=data/census/cp_wikimedia.u32 \
  --stream fwhq_sel=data/census/fwhq_sel.u32 \
  --stream pd_cp_pre1929=data/census/pd_cp_pre1929.u32 \
  --stream cp_stackexchange=data/census/cp_stackexchange.u32 \
  --tokens 51994624 --context-length 2048 --batch-size 4 \
  --min-routed 2000 --split-fraction 0.5 \
  --tokenizer data/tokenizer.json \
  --factqa-seed 510051 --factqa-people 4 --factqa-facts 40 \
  --anchor data/anchor-wikitext.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --ckpt-tokens 1000000 --checkpoint out/C51/census-725B.resume.pt \
  --out out/C51/census-725B.npz
```

After interruption, repeat this exact command with `--resume auto`. Output is
`out/C51/census-725B.npz`, its `.summary.json`, and layer-distribution PNGs;
`--no-plots` disables PNGs. Checkpoint/output parent directories are created by
the program. No GPU census was launched on this host.

## Validation

CPU/macOS, Python 3.12 and PyTorch 2.8.0. Analytical fixture cases cover rare
bursts that reverse the old ranking, squared down norms, actual top-k weights,
duplicate-weight coalescing, routing probability and padding exclusion. Other
cases cover frozen p99/tail counts, A-only selection and B Jaccard, low-count
exclusions, batch-agreement pass and failure, dev-only template/entity
construction, answer reachability, hook restoration and new checkpoint state.
The existing C49 main interruption/resume test compares complete artifacts.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python \
  -m pytest -q experiments/end_to_end/tests/test_c51.py \
  experiments/end_to_end/tests/test_c49.py experiments/end_to_end/tests/test_fast_moe.py
```

This focused run passed **93 tests**, with the existing CUDA-only dispatch
benchmark skipped. Python compilation, `git diff --check`, and parsing the
box command through the real census parser pass; the exact budgets and probe
counts were also checked. The broader `test_c45.py` backend suite passed 30
tests and failed its repository-wide loader inventory assertion on the separate
`c52.py` module, which lacks the standalone announcement expected by that
pre-existing test. C51 does not change that module or the inventory test.
The repository's existing ignore allowlist was extended for the C51 test and
report so both new files are visible to version control.

Full-export p99s, routing shares, rank stability and the
empirical 95% agreement verdict remain to be measured by the box command.
