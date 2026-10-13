# C49: quiet-unit census and split in-topic damage

Implemented SC0 Part 1 / build item 1 and the DM panel. No SC0 reserve training,
address-row overlay, or reserve write experiment was built. No git commit was
run. Existing working-tree changes were preserved.

## Census

`census.py` loads the export through the same `eda.build_model` wrapper and
`ops.eval.bench_saves.build_model` arguments as `e2e_fsa.py`: `cuda:0`, requested
export step, `expert_format="bf16"`, and the explicit `--eda-backend` (default
`naive_fp32`; `fla_kda_2t` has no fallback). It runs without gradients or weight
changes and hooks each routed expert's `up_proj` output before activation.

`--stream name=path` accepts little-endian uint32 files (memory mapped) or JSON
lists of uint32 IDs. Binary streams remain on disk; JSON lists are loaded into
memory. Context length defaults to 2048, batch size to 1. The token budget is
rounded down to full contexts and divided among domains as evenly as possible
(the difference is at most one context). Too-short inputs fail rather than wrap
or repeat. The command below uses 50,003,968 tokens, exactly 12,500,992 per domain,
to meet the design's >=50M requirement without partial contexts.

Before any model forward, every source gram in the consumed prefix is scanned
against the anchor and all `data/eval-*.json` panels, or explicit
`--eval-panels`. Any per-domain/reference fraction >1% fails. This uses the
existing harness denominator: reference 32-grams sampled every 16 positions,
including repeated reference occurrences; source grams are exhaustive and cross
context/chunk boundaries. BLAKE2b-128 hashes bound memory to the reference panels;
no full training gram set is built.

Per expert/domain, CPU fp64 tensors retain routed-token counts, firing counts
(`z > 0`) and `sum(relu(z)^2)`. Reductions and down-column norms use fp64. Output
energy is `norm(down[:, j]) * mean(relu(z_j)^2)`, without router-weight scaling.
The combined-domain mean is weighted by each expert's routed-token count.

Outputs are `<out>.npz`, `<out>.summary.json` and one energy-distribution PNG per
layer (disable PNGs with `--no-plots`). NPZ keys include `L0_E0_routed`, `_fired`,
`_relu2_sum`, `_down_norm`, `_frequency`, `_mean_relu2`, `_energy`, and pooled
`_overall_*` arrays. `layer0_rank` indexes the flattened expert/unit arrays
`layer0_expert` and `layer0_unit`, sorted by pooled energy. Layer summaries contain
frequency, mean-activation and energy percentiles globally and per domain,
routed counts per expert, near-dormant-everywhere counts, domain-specific counts,
and unknown counts. Quiet means **strictly below** that domain's layer 1st
percentile. Domain-specific means quiet in >=3 domains and active in >=1.
Unrouted expert/domain pairs have NaN derived statistics and cannot establish
either classification. If a percentile is zero, tied zeros do not satisfy the
design's strict "below" definition; the raw zeros remain visible in the NPZ.

`--ckpt-tokens` defaults to 1M. At the next completed batch after the cadence is
reached, an atomic temporary-file/fsync/replace save writes domain cursors,
accumulators, down norms, configuration (including domain-to-file mapping and
requested token budget), input SHA-256 fingerprints and Python /
NumPy / torch CPU/CUDA RNG states. The final cursor is saved too. The default
checkpoint is `<out>.resume.pt`; `--checkpoint` can fix its location. Restart
with the identical command plus `--resume auto`; a missing checkpoint or changed
input/backend/context/batch/target is rejected before model construction.
Export fingerprints cover all files, including weight shards. C44's shared
heartbeat now also covers fingerprint reads; census scans, hook setup, forward
batches and output layers emit progress heartbeats. Checkpoints recover forward
batches; output generation can be replayed from the final checkpoint.

## In-topic panel

`build_in_topic_panel.py` constructs reviewed QA about real concepts in the
world's attribute-value pools: cities, instruments, professions and colours.
The built-in catalogue has 290 prompts over 24 concepts (ten fixed framings of
29 properties). It also reuses C37 neighbour probes when they mention a value
concept. Synthetic names/numbers receive no invented real-world gold answers.
Additional reviewed `{concept, prompt, answers}` rows can be supplied with
`--qa-candidates`. Candidate duplicates and invalid concepts fail.

The GPU pass uses C37's `answer_item`, greedy generation and clean answer/stop
scoring, with reviewed aliases, 32 new tokens and stop horizon 2. The filtering
batch size is recorded and reused for base/patched QA scoring. Only
base-correct questions are retained; fewer than 200 aborts before writing a
panel. Candidate base accuracy, retained accuracy/answers and concept coverage
are recorded. The builder does not manufacture correct answers from base output.
If the built-in catalogue does not produce 200 eligible questions on the export,
add reviewed candidates rather than weakening the minimum.

Text consists of unique, fixed-length FineWeb windows mentioning at least one
value concept (default 40 x 512). **Every** 32-gram in candidate windows and
training/anchor/reference sources is checked. Overlapping windows are excluded;
insufficient disjoint windows fails. The small candidate panel supplies the
in-memory gram index while larger training sources are scanned. Supply all
training/rehearsal/census corpora via `--training-text`; the builder also excludes
the anchor and supplied/default eval panels. The artifact records export,
backend, tokenizer SHA, source SHAs, QA selection, raw text IDs and concepts.

`--in-topic-panel FILE` is available in `e2e_fsa.py` and `branch_eval.py`, including
`--learning-eval`. It checks the export/step/backend/tokenizer, >=200 eligible QA,
concept mentions and strict disjointness from the runtime anchor, rehearsal,
route-KL and active address-calibration inputs. The pristine base must still
answer every retained question correctly. Baseline and patched text NLL use the
same windows, with fp64 paired differences and SE. These final-only panels are
excluded from fitting, validation selection and the existing base cache.

Both entry points print `IN-TOPIC` and include `in_topic` in `FINAL`, `RESULT` and
the saved result JSON. Fields are `written_facts`, `knowledge_qa`, `text_dnll`,
panel identity and a combined `passed` flag. Gates are:

- Each previously written bank: TEST-template clean GEN loss <=1 percentage
  point, without a noise exemption.
- Knowledge QA: paired loss <=1 item per 100, or its paired 95% normal CI covers
  zero. Per-item correctness/answers, lost/gained counts, SE and CI are recorded.
- Text: paired mean dNLL <=+0.003; per-window differences and paired SE recorded.

E2e compares all earlier resident/sibling banks before the new student with all
banks resident afterward; mask-only reservations are not written-fact banks.
Branch evaluation compares each loaded bank alone against all banks resident
under the configured merge rule, and clears that activation afterward. No prior
banks means `written_facts.applicable=false`. Existing off-topic damage metrics
and gates remain separate. The panel is opt-in; these gates report eligibility
without changing training or choosing checkpoints based on held-out results.

## Exact box commands (v56/)

Run from the model-box experiment root containing `v56/`, `data/` and `out/`.
Deploy the existing harness with the changed `e2e_fsa.py`, `branch_eval.py`,
`c44_runtime.py` and the new `census.py`, `token_streams.py`, `in_topic.py` and
`build_in_topic_panel.py` together in `v56/`. Install the repository requirements
plus the existing box's model/tokenizer/backend dependencies. Matplotlib is
needed for the default plots.

The four `data/c49/*.u32` names below are **staging inputs**, not claims that those
files already exist: prepare >=12,500,992 tokens each with the export tokenizer,
using WikiText train, FineWeb-Edu training text, Python training code and
`pd_cp_*` long_flat books. Stage independent held-out FineWeb text as
`data/c49/fineweb-heldout.u32`; it must supply 40 concept windows disjoint from
those training sources. Existing anchors/panels use the established filenames.

```bash
set -euo pipefail
C49_PY=python
C49_EXPORT=/workspace/malleable/data/hf/exports/725B-tokens_step58814
C49_BACKEND=naive_fp32
mkdir -p out/C49
C49_STREAMS=(
  --stream wikitext=data/c49/wikitext-train.u32
  --stream fineweb=data/c49/fineweb-train.u32
  --stream code=data/c49/python-train.u32
  --stream books=data/c49/books-long-flat.u32
)
C49_CENSUS=(
  --export "$C49_EXPORT" --export-step 58814 --eda-backend "$C49_BACKEND"
  "${C49_STREAMS[@]}" --tokens 50003968 --context-length 2048 --batch-size 1
  --anchor data/anchor-wikitext.json
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json
  --ckpt-tokens 1000000 --checkpoint out/C49/census-725B.resume.pt
  --out out/C49/census-725B.npz
)
"$C49_PY" v56/census.py "${C49_CENSUS[@]}"
# After interruption, repeat in the same shell/configuration:
# "$C49_PY" v56/census.py "${C49_CENSUS[@]}" --resume auto

"$C49_PY" v56/build_in_topic_panel.py \
  --export "$C49_EXPORT" --export-step 58814 --eda-backend "$C49_BACKEND" \
  --fineweb data/c49/fineweb-heldout.u32 --anchor data/anchor-wikitext.json \
  --training-text data/c49/wikitext-train.u32 data/c49/fineweb-train.u32 \
    data/c49/python-train.u32 data/c49/books-long-flat.u32 data/rehearsal-fineweb.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --context-length 512 --text-windows 40 --batch-size 16 \
  --out data/in-topic-725B.json
```

If additional reviewed QA is needed, repeat the builder command with
`--qa-candidates data/c49/reviewed-concept-qa.json`. To use FLA, set
`C49_BACKEND=fla_kda_2t` before building the arrays and panel; use that backend for
all patch training/evaluation too. For another export, rebuild the panel and
census with its actual path/step and separate output/checkpoint files.

This concrete sequential smoke recipe creates an earlier bank, then adds a
second bank while reporting both damage columns:

```bash
C49_WRITE=(
  --export "$C49_EXPORT" --export-step 58814 --eda-backend "$C49_BACKEND"
  --method fsa_forced --fmt ternary --optim bop --seed 0
  --layers 3,16,22,29 --people 3 --facts-per-person 100 --shared-attrs 30
  --budget 1000000 --fact-batch 40 --steps 400 --eval-every 100 --micro 6
  --answer-only --aug --no-pareto --sparse-state --slot-frac .5
  --expert-checkpoint --fast-moe exact --local-lambda 10
  --support-score netcost --cost-rel 1 --rehearsal-frac 0
  --anchor data/anchor-wikitext.json --anchor-len 96
  --eval-set data/eval-wikitext.json
  --extra-eval-sets data/eval-fineweb.json,data/eval-code.json
  --in-topic-panel data/in-topic-725B.json
  --ckpt-every 100 --ckpt-minutes 60
)
"$C49_PY" v56/e2e_fsa.py "${C49_WRITE[@]}" --train-person 0 \
  --patch-dir out/C49/prior_p0 --out out/C49/prior_p0.json
"$C49_PY" v56/e2e_fsa.py "${C49_WRITE[@]}" --train-person 1 \
  --resident-patch out/C49/prior_p0/final.pt \
  --patch-dir out/C49/new_p1 --out out/C49/new_p1.json
```

For an all-resident branch stress test, use the established three independent
full-mask banks (they must match this export/backend):

```bash
"$C49_PY" v56/branch_eval.py \
  --export "$C49_EXPORT" --export-step 58814 --eda-backend "$C49_BACKEND" \
  --patches out/R6_v4_full_p0/final.pt out/R6_v4_full_p1/final.pt out/R6_v4_full_p2/final.pt \
  --modes sum --merge sum_clip --anchor data/anchor-wikitext.json \
  --eval-set data/eval-wikitext.json --fineweb data/eval-fineweb.json --code data/eval-code.json \
  --fast-moe exact --in-topic-panel data/in-topic-725B.json --out out/C49/resident.json
```

## Validation

CPU/macOS, Python 3.12.14, PyTorch 2.8.0, NumPy 2.5.4. C49 focused tests:
**25 passed in 6.89 seconds**. Coverage includes planted globally dormant and
three-domain-quiet units, down-zero columns, fp64 counters, preactivation-square
semantics on the tiny ReLU fixture, unrouted experts, rankings, both token file
formats, budget rounding, 1% overlap boundary, chunk/context boundaries,
interruption/resume through the actual census entry point, swapped domain/file
mapping and export-payload change
rejection, reviewed base-correct QA filtering, strict unsampled-gram text
exclusion, paired gates/SE, and real builder/e2e/branch/learning entry points.

The existing C45 entry-point inventory assertion was updated from 15 to 17;
its shared-builder/backend checks remain intact. The final combined C45/C49
focused sweep passed **56 tests in 4.43 seconds**, including the final domain/file
mapping and concept-matching checks.

A separate real Matplotlib 3.10.8 smoke produced a valid 17,333-byte layer PNG
from the planted fixture. Matplotlib and its dependencies were supplied from
local cached packages in the temporary test environment; repository dependencies
were not changed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/end_to_end/tests/test_c49.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/end_to_end/tests
```

Full CPU regression: **1,196 passed, 1 skipped** in 373.15 seconds, with the
existing LoRA scalar-conversion warning. Python compilation and `git diff --check`
pass. The report's five box invocations passed `bash -n` and each actual command's
argument parser through a loader-free CLI checker; no box command was executed
against a GPU/export. Final census/panel matcher refinements were also checked
by the focused sweep above.
No CUDA export census, GPU QA-filter pass, 50M-token timing or empirical DM gate
outcome has been measured on this host. The box commands above remain to be run;
no generated production panel is presented as GPU-validated.
