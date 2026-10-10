<p align="center">
  <img src="figures/epigenetic-llm.jpg" width="900" alt="Epigenetic Language Models: a base model, knowledge intercalated into its structural zeros, and the personalized individual that results.">
</p>

<h1 align="center">Epigenetic Language Models</h1>

<p align="center"><b>Lifelong learning by intercalating knowledge into structural zeros.</b></p>

<p align="center">The base weights never change. New knowledge is written into the zero slots a compressed model already has, with no new parameters, and can be erased bit-exactly when a user asks to be forgotten. The method is <b>Free-Slot Adaptation (FSA)</b>.</p>

<p align="center">
  📄 <a href="paper/freeslot.pdf"><b>Read the paper (PDF)</b></a> <i>(AI-assisted)</i>
  &nbsp;·&nbsp; <a href="#results-on-the-real-8b-model-2026-10-10">Results</a>
  &nbsp;·&nbsp; <a href="#best-algorithms-so-far">Algorithms</a>
  &nbsp;·&nbsp; <a href="#how-free-slot-adaptation-works">How it works</a>
  &nbsp;·&nbsp; <a href="#reproduce">Reproduce</a>
  &nbsp;·&nbsp; <a href="ROADMAP.md">Roadmap</a>
</p>

> [!NOTE]
> **The paper and this write-up were generated with AI assistance**, from experiments run and verified by the authors. Treat the prose as a draft under human review, not a peer-reviewed result.

## Results on the real 8B model (2026-10-10)

Teaching 40 facts about “Tamsin Vey” to the lambda 8B ternary MoE, with 12 training formats and two
held-out phrasings (80 items). Every run uses a single seed; **seeds pending** for the headline configuration.
TF is teacher-forced exact answer-token recall. GEN is free-running greedy recall: the whole answer must
match at a word boundary, with a terminator or end-of-sequence within two tokens after the answer.
Damage is ΔNLL on 40×512 held-out WikiText windows (± paired SE). Final checkpoints are step 300,
except repair, which adds 50 steps with births disabled.

| method | TF | GEN | general-text damage | patch | inference overhead |
|---|---|---|---|---|---|
| FSA Bop + penalty + budget + rehearsal (best) | 98.75% (79/80) | 98.75% (79/80) | -0.0001 ± 0.0017 | 250k slots (0.33–0.84 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop + penalty + budget + energy threshold | 98.75% (79/80) | 97.5% (78/80) | +0.0013 ± 0.0019 | 239k slots (0.32–0.81 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop + penalty + budget + repair | 97.5% (78/80) | 97.5% (78/80) | +0.0029 ± 0.0016 | 251k slots (0.33–0.85 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop + penalty + budget | 96.25% (77/80) | 95% (76/80) | -0.0003 ± 0.0021 | 244k slots (0.32–0.82 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop + penalty (rerun) | 97.5% (78/80) | 97.5% (78/80) | -0.0011 ± 0.0019 | 736k slots (0.82–2.48 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop + penalty (earlier) | 100% (80/80) | — | +0.0015 ± 0.0017 | 740k slots (0.83–2.50 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop τ0.7 (run 1) | 100% (80/80) | — | +0.0266 ± 0.0035 | 812k slots (0.89–2.74 MB) | same packed kernel; synthetic serving measured below |
| FSA Bop τ0.7 (repeat) | 100% (80/80) | — | +0.0237 ± 0.0036 | 670k slots (0.76–2.26 MB) | same packed kernel; synthetic serving measured below |
| FSA Learn-then-prune, decay 0.3 | 91.25% (73/80) | — | +0.0250 ± 0.0032 | 542k slots (0.64–1.83 MB) | same packed kernel; synthetic serving measured below |
| FSA Learn-then-prune, decay 0.5 | 88.75% (71/80) | — | +0.0129 ± 0.0028 | 523k slots (0.62–1.77 MB) | same packed kernel; synthetic serving measured below |
| FSA Fp4 + penalty | 96.25% (77/80) | — | +0.0050 ± 0.0015 | 21.27M slots (18.17–79.75 MB) | needs fp4 slot support |
| LoRA r1 + penalty (baseline, fp32 / measured int4) | 93.75% (75/80) / 93.75% (75/80) | 95% (76/80) / 93.75% (75/80) | -0.0002 ± 0.0019 / +0.0021 ± 0.0017 | 1,376,256 params (2.75 MB bf16 / 0.69 MB int4) | extra matmul per token |
| LoRA r2 + penalty (baseline, fp32 / measured int4) | 96.25% (77/80) / 97.5% (78/80) | 95% (76/80) / 95% (76/80) | +0.0019 ± 0.0014 / +0.0021 ± 0.0017 | 2,752,512 params (5.51 MB bf16 / 1.38 MB int4) | extra matmul per token |
| LoRA r4 + penalty (baseline, fp32 / measured int4) | 96.25% (77/80) / 97.5% (78/80) | 95% (76/80) / 96.25% (77/80) | -0.0001 ± 0.0020 / +0.0003 ± 0.0016 | 5,505,024 params (11.01 MB bf16 / 2.76 MB int4) | extra matmul per token |
| LoRA r16 + penalty (baseline) | 97.5% (78/80) | — | +0.0005 ± 0.0018 | 22,020,096 params (44.04 MB bf16) | extra matmul per token |

<p align="center">
  <img src="figures/fig_e2e_frontier.png" width="760" alt="Held-out recall versus general-text damage for every comparable run on the real 8B model.">
  <br>
  <img src="figures/fig_e2e_storage.png" width="640" alt="Held-out recall versus patch size for FSA and LoRA.">
</p>

Patch sizes include positions and values: **E** entropy-codes positions over 66,807,179 free slots plus
one sign bit per ternary slot; **I** stores a 26-bit index plus the value. Sizes use decimal MB.
LoRA bf16 uses two bytes per parameter; int8/int4 factor sizes include one bf16 absmax scale per rank vector.
Their recall and damage were measured after fake quantization. Every run revokes bit-exactly: logits match
the base on one 512-token window and test recall returns to 0/80. “—” means generated recall was not measured.

- **The best result is Bop + penalty + budget + rehearsal:** 79/80 on both recall metrics from 249,755 slots,
  with ΔNLL −0.0001 ± 0.0017. At a comparable byte budget, the best FSA patch recalled 79/80 teacher-forced and 79/80 generated items from 0.33 MB entropy-coded or 0.84 MB indexed, versus 75–78/80 teacher-forced and 75–77/80 generated for every LoRA rank and measured precision tried (r1–r4, 0.69–11.01 MB for int4–bf16). The gap is 1–4 items out of 80 on one seed: suggestive, not established; **seeds pending**. The closest LoRA size is r1 int4 at 0.69 MB: 75/80 TF and 75/80 GEN versus FSA's 79/80 on both. These are two alternative FSA encodings, not an exact equality of byte budgets.
- **Damage is within noise of zero for the best penalized FSA runs and fp32 penalized LoRA:** |ΔNLL| ≤ 0.003,
  SE about 0.002. At this sample size no damage difference is detectable between those methods. The measured
  r1 bf16/int8 evaluations are exceptions (+0.0040/+0.0049); all precision results are in the findings.
  Plain Bop without the penalty costs +0.024 to +0.031; the older rows remain as history.
- **The earlier 80/80 TF result was not reproduced exactly:** its same-configuration rerun gives 78/80 TF
  and 78/80 GEN at −0.0011 ± 0.0019. Variation between executions is a couple of items on the same seed.
- **Where each method stands today:** FSA has the best observed recall at a small patch size; the recall gap
  is suggestive on one seed and does not establish that FSA beats LoRA overall. Penalized damage is comparable.
  Synthetic legal patches show no detectable serving cost; learned patches still need a serving benchmark.
  Multi-session concept learning is the EP-1 experiment in [`experiments/epigenesis/`](experiments/epigenesis/).

## Serving: does a patch cost anything at inference?

Measured on the released lambda GGUF, with the patch written into the packed pair8 experts
([`experiments/serving/`](experiments/serving/README.md)). Patches are synthetic, matching the learned patches' size and
spread (670k slots, plus a 3M-slot stress test, over all experts of MoE layers 14–15). Speeds are median tokens/s; the
patched columns are patched ÷ base measured in the same round (median [min, max] over rounds).

| device and kernel | base decode | 670k patch | 3M patch | base prefill (512 tok) | 670k / 3M prefill |
|---|---:|---:|---:|---:|---:|
| Mac M5 Max, Metal (`lut9`) | 174.8 | 0.999 [0.96, 1.01] | 0.999 [0.97, 1.03] | 1,075 | 0.999 / 1.001 |
| Mac M5 Max, CPU 8 threads (`lut8`) | 102.2 | 1.005 | 0.990 | 371 | 1.002 / 1.009 |
| Mac M5 Max, CPU 8 threads (`lut`) | 66.0 | 1.006 | 0.998 | 148 | 1.004 / 0.999 |
| Snapdragon 8 Elite phone, Q8 trunk (`lut8`) | 57.0 | 1.002 [0.97, 1.03] | 0.982 [0.95, 1.03] | 209 | 0.996 / 1.000 |

| | base | 670k patch | 3M patch |
|---|---|---|---|
| file size | 2,133,006,592 B | identical | identical |
| peak memory (Metal / CPU / phone) | 2.75 GiB / 3.29 GiB / 3.24 GB | identical | identical |
| bytes changed in the file | — | 0.99 MB | 4.3 MB |
| revoke | — | byte-identical original (sha256 match) | byte-identical original |

Every ratio's range crosses 1.0, so no slowdown is measurable. That is expected: each packed block costs one table
lookup whatever it holds, and a patch never turns an all-zero block nonzero. Applying a 670k-slot patch takes ~46 ms in
memory (apply A → revoke A → apply B: ~137 ms); today's runtime repacks weights at load, so switching users means
patching the file and reloading (~2 s) until a live hook is added. An unmerged rank-16 LoRA in this runtime is
estimated (not measured; the runtime has no adapter support) at +0.5–1.5% per token plus 44 MB per adapter and new
code in every backend; merging it would turn 64 MB of packed experts into 906 MB of bf16.

## Best algorithms so far

**Writers (how slots get filled):**
1. **Bop, latent-free ternary** (Helwegen et al. 2019, extended to {−1, 0, +1}). Each free slot keeps an exponential
   moving average of its gradient; a slot is born (0 → ±1) or dies (±1 → 0) only when that average is consistent and
   exceeds a threshold τ, and the average resets on every change. τ directly sets the flip rate, so a small gradient
   never turns into a full ±α step. Best ternary result so far when combined with the local penalty, budget and rehearsal below.
2. **Learn-then-prune.** fp32 masters + straight-through ternary, AdamW with weight decay on the masters. The model
   learns everything early (~50 steps), then decay pulls weakly supported masters back through the threshold: the
   patch shrank from ~20M to 0.5M slots while recall held.
3. **fp4 slots + local output penalty** (non-ternary option): the penalty is the relative energy of the patch's direct
   output change on general text, Σ‖x·ΔWᵀ‖² / Σ‖x·Wᵀ‖², which avoids KL's routing-chaos floor because it measures each patched layer's direct output change.

**Preservation and allocation in the new Bop runs (single seed):**
- **Local output penalty, λ=10:** penalize the relative energy of each patched projection's direct output change
  on general text, using the same penalty described above.
- **300k-slot budget:** every 10 steps, reselect the allowed slots by gradient evidence minus expected
  general-text cost (α² times input-column energy), directing births toward useful, less disruptive slots.
- **Rehearsal 0.2:** optimize 0.8 × fact loss + 0.2 × ordinary next-token loss on WikiText. These windows share
  no 32-token sequences with the damage panel, but come from the same corpus; damage on other kinds of text is untested.
- **Energy threshold variant:** multiply each row's birth threshold by clamp((e_row / median e)^0.5, 0.25, 4),
  so rows sensitive to general text need more evidence. **Repair variant:** 50 extra penalty steps with births
  disabled, allowing deaths only. Neither variant establishes an improvement on one seed.

**Ingredients that mattered for every method:**
- Loss on the answer tokens only. Whole-sentence loss trained models to recite template boilerplate and caused most of
  the early damage.
- Diverse training phrasings including Q/A forms (12 formats, 3 sampled per fact per step). This closed the
  validation→test gap for both FSA and LoRA.
- A broad footprint: all experts of the patched layers, up and down projections. Small footprints starved capacity.
- Damage measured as ΔNLL on a large held-out pool, checkpoints chosen on separate validation text. Any single weight
  change in this model already moves KL-to-base to ~0.02 through routing near-ties, so KL cannot measure damage.

<p align="center">
  <img src="figures/fig_e2e_trajectory.png" width="820" alt="Validation recall, damage and filled slots over training for learn-then-prune, Bop and LoRA.">
  <br>
  <img src="figures/fig_chaos_floor.png" width="760" alt="KL to the base model from random flips and from a single-weight perturbation.">
</p>

**What did not work:** KL penalties (they fight the routing-chaos floor and only cost recall); ternary + smooth penalty +
decay (recall peaks, then gets pruned away); tiny-magnitude fp4 (did not learn in our run; its smallest steps sit near bf16 resolution); patch-only
block scales (no better than plain ternary); continuous values on the slot mask at a single late layer (did not beat
LoRA at the same placement).

## Why "epigenetic"

Epigenetic changes alter how a genome is expressed without changing the DNA, and they can be reversed. Here the base weights play the role of the genome: they are frozen and never edited. What the model learns after deployment lives in a separate, sparse layer written into slots that were exactly zero, so removing it restores the original model bit for bit. That is the whole analogy; everything else on this page is measured.

## What is pair8?

`pair8` is an extreme weight format. Take the weights in **aligned blocks of 8** (four adjacent pairs). The rules:

- every weight is **ternary** — `{ -1, 0, +1 }` times a block scale;
- **at most 2 of the 4 pairs** in a block may be non-zero.

That second rule forces **≥ 50% of all weights to be exact structural zeros**. A block has only **417 reachable states**, so it encodes in **≈ 1.088 information-bits per weight** (1.125 bits when packed) — vs 16 bits for bf16. The zeros aren't a rounding artifact; they're *guaranteed by the format*, which is exactly what makes them a predictable reserve.

```
one pair8 block of 8 weights  (· = forced zero,  ■ = ternary ±1)
┌─────┬─────┬─────┬─────┐
│ ■ ■ │ · · │ ■ ■ │ · · │   ← ≤ 2 of 4 pairs active  ⇒  ≥ 4 of 8 weights are zero
└─────┴─────┴─────┴─────┘
```

## How Free-Slot Adaptation works

<p align="center">
  <img src="figures/freeslot_html_render.png" width="820" alt="Free-Slot Adaptation: a compressed model's built-in empty shelf slots are filled with new knowledge while the original weights never move.">
</p>

```
Base model:   pair8 ternary MoE. In every 8-weight block at most 2 of 4 pairs are active,
              so at least half of the weights are exact zeros.
Free slots:   the zeros INSIDE pairs that are already active. Filling one keeps the block a legal
              pair8 block, so the model stays in its packed format and runs on the same kernel.

Adapt:        1. Freeze every original weight, scale, norm and router.
              2. Train only the free slots of the chosen experts (ternary values ±α of the row).
              3. The patch is the list of filled slots and their signs.

Revoke:       zero the filled slots. The weights are bit-identical to the base again.
```

**Guaranteed zeros vs writable slots.** The format guarantees at least half of the weights are zero, but only the zeros
*inside already-active pairs* can be filled without breaking the packed format. On the lambda 8B, the 256 experts of
MoE layers 14–15 hold ~453M expert weights and **66.8M writable free slots (~14.7%)** across up and down projections. The best
patches use under 1M of them.

**What it costs.** No new tensors and no extra compute per token: the patch lives inside tensors the base kernel
already multiplies. The cost is storage for the patch, which must record which slots it filled (0.33–2.74 MB for the
ternary patches above, versus 44 MB for a bf16 LoRA at the same layers), and a measured amount of general-text damage.

## Limitations

- **Damage is measured on one corpus, with one seed per run.** The best penalized FSA and fp32 LoRA runs
  are within noise of zero at this sample size; this does not establish damage-free adaptation. Rehearsal uses
  the same corpus as evaluation with no shared 32-token sequences, so preservation is measured in-distribution.
  Other kinds of text are untested. Plain Bop costs +0.024 to +0.031. **Seeds pending** for the headline configuration.
- **Generated recall is measured for the new runs.** GEN checks the whole answer and a clean stop under
  free-running greedy generation; older runs have TF only. A 1–4-item difference on one seed is suggestive,
  not established, and this 40-fact task does not establish broader generalization.
- **Multi-tenant serving.** An FSA patch lives inside the weights, so one copy of the weights serves one user's patch at
  a time. LoRA adapters can be batched across many users on one shared base. FSA fits single-tenant and on-device use;
  serving many users at once needs per-user weight copies or patch swapping between batches (cost being measured).
- **One task so far.** The real-model results are 40 facts about one fictional person. Whether slots accumulate
  *concepts* across many sessions without accumulating interference is the open question EP-1 tests.
- **Revoke scope.** Zeroing the slots restores the base exactly. Removing one session's influence after later sessions
  have learned on top of it is not guaranteed by zeroing that session's slots alone.
- **Format.** The ternary writers stay in the packed pair8 format; the fp4 option does not and needs slot-level fp4 support.
- **Privacy.** Knowledge in weights can be elicited from the model; a per-user model is the safe deployment.

## Reproduce

```bash
# needs one 32 GB GPU, the lambda 8B export and the mesh-lambda code (see the harness header)
python experiments/end_to_end/e2e_fsa.py --method fsa_free --optim bop --bop-tau 0.7 --bop-gamma 0.05 \
    --layers 14,15 --answer-only --aug --anchor <anchor.json> --eval-set <eval.json> --steps 300 --out bop.json
python experiments/end_to_end/e2e_fsa.py --method lora --rank 16 --lr 3e-4 --layers 14,15 --local-lambda 100 \
    --answer-only --aug --anchor <anchor.json> --eval-set <eval.json> --steps 300 --out lora.json
python figures/make_e2e_figs.py experiments/end_to_end/data      # rebuild the figures from the saved runs
```

Every run, including the ones that did not work, is described in
[`experiments/end_to_end/E2E_FINDINGS.md`](experiments/end_to_end/E2E_FINDINGS.md).

## Prior art

[PackNet](https://arxiv.org/abs/1711.05769) · [Latent weights do not exist (Bop)](https://arxiv.org/abs/1906.02107) ·
[LoRA](https://arxiv.org/abs/2106.09685) · [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) ·
[Cartridges](https://arxiv.org/abs/2506.06266) · BitNet b1.58 (ternary). FSA is specific to *format-reserved* sparse
capacity: the free slots exist because of the quantization format.

## How claims were checked

Every claim here went through repeated rounds of adversarial review, and the write-up was revised until the findings stopped changing.

## License & attribution

Code: Apache-2.0. Paper & figures: CC-BY-4.0. By [Chutes.ai](https://chutes.ai) / Parallax. Paper and write-up AI-assisted (see the note at the top).
