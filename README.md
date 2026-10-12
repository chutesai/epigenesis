<p align="center">
  <img src="figures/epigenetic-llm.jpg" width="900" alt="Epigenetic Language Models: a base model, knowledge intercalated into its structural zeros, and the personalized individual that results.">
</p>

<h1 align="center">Epigenetic Language Models</h1>

<p align="center"><b>Lifelong learning by intercalating knowledge into structural zeros.</b></p>

<p align="center">The base weights never change. New knowledge is written into the zero slots a compressed model already has, with no new parameters, and can be erased bit-exactly when a user asks to be forgotten. The method is <b>Free-Slot Adaptation (FSA)</b>.</p>

<p align="center">
  📄 <a href="paper/freeslot.pdf"><b>Read the paper (PDF)</b></a> <i>(AI-assisted)</i>
  &nbsp;·&nbsp; <a href="#results-forced-zero-mask-2026-10-12">Results</a>
  &nbsp;·&nbsp; <a href="#research-frontier">Research frontier</a>
  &nbsp;·&nbsp; <a href="#best-algorithms-so-far">Algorithms</a>
  &nbsp;·&nbsp; <a href="#how-free-slot-adaptation-works">How it works</a>
  &nbsp;·&nbsp; <a href="#reproduce">Reproduce</a>
  &nbsp;·&nbsp; <a href="ROADMAP.md">Roadmap</a>
</p>

> [!NOTE]
> **The paper and this write-up were generated with AI assistance**, from experiments run and verified by the authors. Treat the prose as a draft under human review. The results have not been peer-reviewed.

## TL;DR

This repo studies reversible learning in a frozen model’s zero weights, one user at a time.

FSA writes only into the **pair8 forced-zero mask**: the two pairs pair8 did not select in each 8-weight block, exactly half of every expert weight matrix. The mask is deterministic, derived from the released model and shipped with it. The base weights never change, and zeroing the mask restores the exact pair8 base.

Results on the lambda 8B ternary MoE, forced-zero mask, free-generation exact match (GEN). The first two rows are held-out test
phrasings; the last two are validation and dev numbers from runs still in progress:

| task | FSA | LoRA r16 (same layers) |
|---|---|---|
| 40 facts about one person | **100%** (2 seeds) | 90.0–93.75% (3 seeds) |
| 2 people × 50 conflicting facts | **97.0–99.5%** (3 seeds) | 94.5–95.5% (3 seeds) |
| 3 people × 100 conflicting facts | 37.5% best val (2-layer), 49% val at step 100 (4 layers) | 57.5–62.5% val |
| new rules (EP-1, dev set) | **95.25%** with sleeping-unit gating | 81% at step 250 (mid-layer LoRA); dense all-layer fine-tuning 96% at +0.20 damage |

On the small and medium tasks FSA beats LoRA with WikiText damage within noise of zero. It still trails LoRA when hundreds of facts about several people collide; closing that gap is the current work ([research frontier](#research-frontier)). The forced-zero patch can give a block more than two active pairs, so serving needs a sparse overlay or a wider block encoding; that path is not built or benchmarked yet.

## Why this matters

There are three common ways to give a deployed model new knowledge, and each has a cost:

| | put it in the prompt | retrieval (RAG) | fine-tune / LoRA | FSA (this repo) |
|---|---|---|---|---|
| where the knowledge lives | the context window | an external database | new adapter weights | the pair8 forced-zero mask of the existing weights |
| new parameters | none | a separate store | yes | none |
| extra compute per token | grows with the context | retrieval + longer context | an extra matmul (unmerged) | forced-zero: sparse overlay, pending; earlier in-pair variant: none measured |
| deployed model file | unchanged | unchanged | adapter file, or a merged bf16 copy | base file unchanged plus a small patch (overlay format pending) |
| undo one user's knowledge | drop the text | delete the records | drop the adapter | zero the mask: exact base |

The goal is a personal model that gradually absorbs what one user teaches it, runs on-device, and can be returned to its original state exactly. We call this an *epigenetic* model: the base weights
are the genome and are never edited; what the model learns afterwards is a reversible layer on top
([why the name](#why-epigenetic)).

## Where it stands (2026-10-12)

- **Works:** single-person facts and two-person conflicting facts, written into the forced-zero mask. FSA beats LoRA r16 on held-out
  test GEN on every seed, with WikiText damage within noise of zero. The 40-fact patch re-checked on held-out FineWeb is
  also clean (−0.0007 ± 0.0018) ([results](#results-forced-zero-mask-2026-10-12)).
- **Works, dev set only:** learning new rules. Sleeping-unit gating reaches 95.25% on a 400-item dev set at +0.001 damage
  ([EP-1](#ep-1-learning-rules-instead-of-facts)). Held-out test numbers are pending.
- **Main gap:** many conflicting facts. With 3 people × 100 shared attributes, FSA peaks at 37.5% validation recall on two layers
  versus LoRA’s 57.5–62.5%, and its recall falls after the peak. Spreading the patch over four layers reaches 49% by step 100, so
  placement is part of the answer ([research frontier](#research-frontier)).
- **Pending:** the serving path for forced-zero patches (sparse overlay or wider block encoding), code-panel damage for the
  2-layer winners, realistic-question evaluation, router-drift measurements and multi-session revoke-scope tests.
  See [limitations](#limitations).
- **History:** every result before 2026-10-11 used an *in-pair* write set (zeros inside already-selected pairs), which is disjoint
  from the forced-zero mask. Those results, including the serving benchmarks, are kept in
  [earlier results](#earlier-results-in-pair-variant-2026-10-10).

The next three sections explain the name, the pair8 format and the method; detailed results, the research frontier,
serving numbers and algorithms follow.

## Why "epigenetic"

Epigenetic changes alter how a genome is expressed without changing the DNA, and they can be reversed. Here the base weights play the role of the genome: they are frozen and never edited. What the model learns after deployment lives in a separate, sparse layer written into slots that were exactly zero, so removing it restores the original model bit for bit. That is the whole analogy; everything else on this page is measured.

## What is pair8?

`pair8` is an extreme weight format. Take the weights in aligned blocks of 8 (four adjacent pairs). The rules:

- every weight is ternary — `{ -1, 0, +1 }` times a block scale;
- at most 2 of the 4 pairs in a block may be non-zero.

That second rule forces ≥ 50% of all weights to be exact structural zeros. A block has only 417 reachable states, so it encodes in ≈ 1.088 information-bits per weight (1.125 bits when packed) — vs 16 bits for bf16. The zeros aren't a rounding artifact; they're *guaranteed by the format*, which is exactly what makes them a predictable reserve.

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
Write mask:   the two pairs pair8 did not select: 4 of every 8 weights (at least half).
              This deterministic forced-zero mask ships with the model.

Adapt:        1. Freeze every original weight, scale, norm and router.
              2. Train only slots in the mask (ternary values ±α of the row).
              3. Store filled positions and signs as a removable patch.

Revoke:       zero the mask. The exact pair8 base is restored.
```

The mask rule is fixed: a block’s selected pairs are the ones holding a base nonzero, filled up to two in ascending pair
index if fewer hold one. It is computed once from the released export (sha256 `25a85fd2…`) and never recomputed from patched
weights. Patch values are plain ternary at the row’s scale α.

Filling forced-zero pairs can give a block more than two nonzero pairs, making it illegal pair8.
Serving needs a sparse overlay beside the packed kernel or blocks re-encoded in a wider format.
This path is not built or benchmarked yet.

### What it costs

Patch storage records positions and signs. The best forced-zero patches use 175k–760k slots, a few hundred kilobytes to
about 2 MB, versus 44 MB for a bf16 LoRA r16 at the same two layers. Serving cost for the forced-zero overlay is pending.
The earlier in-pair variant measured no serving cost at all ([serving](#serving-in-pair-variant)).

## Results: forced-zero mask (2026-10-12)

Model: the lambda 8B ternary MoE (725B-token export). Writer for the headline rows: Bop with a local output penalty (λ=10),
a netcost slot budget and 0.2 WikiText rehearsal, on MoE layers 14–15 unless noted. Runs started after the damage-panel
finding below drop rehearsal. Training uses 12 phrasings per fact with loss on the answer tokens only.
Test items are held-out phrasings. **GEN** (headline) is free-running greedy generation: the whole answer must match and the
model must stop cleanly. **TF** is teacher-forced answer-token recall. Damage is ΔNLL ± paired SE on 40×512 held-out WikiText
windows. Every patch revokes bit-exactly.

**40 facts about one person (80 test items)**

| method | GEN | TF | damage (WikiText) | slots | seeds |
|---|---|---|---|---|---|
| **FSA Bop recipe** | **100% / 100%** | 100% / 100% | −0.0006 ± 0.0020 / +0.0019 ± 0.0014 | 175k / 188k | 2 |
| FSA, random 25% of the mask | 97.5% | 100% | −0.0000 ± 0.0020 | 252k | 1 |
| FSA, single layer 12 | 98.75% | 98.75% | +0.0014 ± 0.0020 | 225k | 1 |
| LoRA r16 + penalty | 93.75% / 92.5% / 90.0% | 96.25% / 93.75% / 88.75% | +0.0012 to +0.0015 | 22M params | 3 |
| LoRA r4 / r1 | 96.25% / 93.75% | 96.25% / 93.75% | ~0 | 5.5M / 1.4M params | 1 each |

The 2-layer FSA patch (seed 0) re-measured on held-out FineWeb-Edu gives −0.0007 ± 0.0018.

**2 people × 50 shared attributes with conflicting values (200 test items)**

| method | GEN | TF | damage (WikiText) | slots | seeds |
|---|---|---|---|---|---|
| **FSA Bop recipe, 1M budget** | **97.0% / 99.5% / 99.5%** | 96.5% / 99.5% / 99.5% | −0.0015 / +0.0000 / −0.0026 | 733k–757k | 3 |
| FSA, random 25% of the mask | 97.5% | 98.0% | +0.0010 ± 0.0021 | 761k | 1 |
| LoRA r16 | 94.5% / 95.5% / 95.0% | 96.0% / 94.5% / 95.5% | +0.0010 / +0.0021 / +0.0028 | 22M params | 3 |

On the earlier in-pair write set the same FSA recipe scored 64% on this task. Moving to the forced-zero mask turned a loss into
a win.

**3 people × 100 conflicting facts (main gap; validation only, runs in progress)**

| method | best val GEN (step) |
|---|---|
| LoRA r16, layers 14–15 | 57.5% (1000) / 62.5% (900) |
| FSA Bop recipe, layers 14–15, 1M budget | 37.5% (500) / 25% (400), then declining |
| FSA Bop, 4 random layers (3, 16, 22, 29) | 49% (100), 43% at step 200 |

Every attribute here conflicts across people: 60 attributes are three-way and 60 two-way. FSA’s recall peaks and then falls,
which points at the budget evicting slots that already hold learned facts.

### Placement

Layer matters. On 40 facts with one layer each (test TF): layer 12 98.75%, layer 17 96.3%, layer 20 95%, layer 9 92.5%,
layer 25 88.75%, layer 6 86.25%. Middle layers (12–20) are best. Choosing layers by causal tracing was the worst option on the 300-fact task
(7–10% validation), consistent with reports that causal tracing does not predict where edits work.

### Damage panels

Early multi-layer patches trained with WikiText rehearsal looked *better* than the base on WikiText (−0.024 to −0.040). On
held-out panels that turned out to be adaptation toward prose, not preservation:

| 4-layer 40-fact patch | WikiText | FineWeb-Edu | Python code |
|---|---|---|---|
| causal-trace layers | −0.040 | −0.008 ± 0.002 | **+0.0076 ± 0.0017** |
| every 4th layer | −0.029 | −0.0076 ± 0.0021 | **+0.0065 ± 0.0020** |
| random layers | −0.024 | −0.0056 ± 0.0015 | +0.0001 ± 0.0016 |

Fact runs now use no WikiText rehearsal, and damage is reported on all three panels.

### EP-1: learning rules instead of facts

EP-1 asks whether the mask can hold *rules*: support-ticket policies (category plus condition → action) that must be applied to new
tickets. The model cannot apply them even with all rules in its context: 0.30 with the original rules (0.27 ticket form, 0.34 Q&A form)
and 0.27 dev / 0.32 test with coarser rules, against chance 0.25. So it cannot teach itself. EP-1 therefore trains on ground-truth targets (the pre-registered fallback).

**Sleeping-unit gating** writes only into hidden units of the relu² experts that rarely fire on general text: the 10% least
active. A hinge term pushes those units’ pre-activations negative on general text, encouraging them to stay off outside the new context.
Ternary writes go into the forced-zero mask of those units’ rows.

| arm (400-item dev set, chance ≈ 0.25, base 0.265) | dev accuracy | damage |
|---|---|---|
| **sleeping-unit gating, patch scale 1.0α** | **95.25%** (step 400) | +0.0011 |
| sleeping-unit gating, 0.5α | 87.5% | +0.0026 |
| gating on the most active 10% instead | 91.5% | +0.014 |
| gating on a random 10% | 89.5% (step 300, running) | +0.005 |
| gating plus the facts recipe (budget, local penalty, rehearsal) | 32–34% | ~+0.001 |
| no gating, ternary writes at 0.25α | 99.5% | +0.041 |
| dense fine-tuning of all layers (placement study) | 96.25% (step 350) | +0.20 |

Every tested gated 10% learns the rules; choosing the units that are asleep on general text kept damage at noise level. The facts
recipe’s constraints choke rule learning. These are dev-set numbers used for checkpoint selection, so they are optimistic;
held-out test numbers and a second seed are pending. Gating does not help facts (at most 22% on the 300-fact task).

The first EP-1 attempt (v1, in-pair write set) was not a valid test: its in-context teacher had the answer leaked into its
context, and with a non-leaking context the teacher scored only 0.30 ([findings](experiments/epigenesis/EP1_V1_FINDINGS.md)).

## Research frontier

The open problem is many conflicting facts. The current diagnosis:

- **Collisions, not capacity (hypothesis).** The mask has far more slots than the patches use. The suspected difficulty is that
  “Tamsin’s tram stop” and “Ilse’s tram stop” look almost the same to the model, so the patch needs units that recognise each person-attribute
  combination. Batch fact-editing methods report the same collapse when many edits share a subject.
- **Too little exposure.** With 40 facts per step, each fact is seen about half as often as on the two-person task.
- **Destructive churn.** The slot budget re-chooses slots every few steps and can evict slots that already hold a learned fact,
  zeroing their evidence. That matches the peak-then-decline curves.

Directions being tested (in progress; none has yet shown an improvement over the baseline):

| direction | idea | source |
|---|---|---|
| spread placement | write across 4–8 middle layers instead of 2, with a larger budget | batch fact editing spreads edits over several layers |
| keep-live + binding loss | never evict slots holding learned facts; penalise giving one person’s value for another’s | our churn and conflict diagnostics |
| saturation sampling | stop drilling a fact once it is answered cleanly; resample it when forgotten | exposure analysis |
| context-teacher distillation | the base model with the fact sheet in context is the teacher; train the patch to match its full answer distribution | self-study distillation ([Cartridges](https://arxiv.org/abs/2506.06266)) |
| Hebbian write | sum output-direction × whitened input key per fact, keep the mask positions, round to ternary; use as a start for Bop | associative memory; closed-form editors |
| closed-form masked solve | solve for the patch with general-text covariance whitening, then round to ternary with error compensation | [MEMIT](https://arxiv.org/abs/2210.07229), [AlphaEdit](https://arxiv.org/abs/2410.02355) |
| masked Gauss–Newton ternary steps | accept a block of code changes only when the true loss drops | discrete second-order search |
| wake, then write | first train units that fire only on a person-attribute combination, then write the answers from them | sleeping-unit gating result |
| router stability | measure and limit how much a patch changes expert routing downstream | [MoEEdit](https://arxiv.org/abs/2602.10965) |
| learned editor | a small network that maps a fact to a patch in one pass, trained on patches found by the methods above | [MEND](https://arxiv.org/abs/2110.11309), [MALMEN](https://arxiv.org/abs/2311.04661) |

Evaluation is being tightened as well. Teacher-forced scores can overstate edits badly, so GEN is the headline and a
realistic-question set is being added: unseen phrasings, multi-fact questions and indirect references to the person. The win
condition for the 300-fact task is at least 5 GEN points over LoRA r16 on the same layers, damage ≤ +0.003 on every panel, exact
revoke and 2–3 seeds.

## Earlier results: in-pair variant (2026-10-10)

Everything in this section and the serving section used the in-pair write set: zeros inside pairs pair8 had already selected
(66,807,179 slots at layers 14–15). That set is disjoint from the forced-zero mask. Filling it keeps pair8 legal, which is why its
serving cost could be measured. It is kept as history; the forced-zero mask above is the design.

All reported FSA runs use the in-pair variant (66,807,179 in-pair slots at layers 14–15), a different set of base zeros (inside the selected pairs, disjoint from the forced-zero mask) that keeps pair occupancy legal; plain ternary patches keep pair8 legal. Forced-zero-mask results are pending. Below, “slots” means in-pair slots.

Teaching 40 facts about “Tamsin Vey” to the lambda 8B ternary MoE, with 12 training formats and two
held-out phrasings (80 items). The headline has three seeds with different fact values and training randomness: seed 1 differs from seed 0
on 33/40 attribute values, and seed 2 on 34/40. Other runs, including all LoRA baselines, have one seed each.
TF is teacher-forced exact answer-token recall. GEN is free-running greedy recall: the whole answer must
match at a word boundary, with a terminator or end-of-sequence within two tokens after the answer (the model may keep generating after that).
Damage is ΔNLL on 40×512 held-out WikiText windows (± paired SE). Final checkpoints are step 300,
except repair, which adds 50 steps with births disabled.
All evaluations run one sequence at a time: 16-prompt batching flipped 15/32 greedy base continuations; 1–2 item recall differences are within noise.
The legacy 5×512 KL/top-1 set overlaps training anchors (57% of 32-token sequences); the 40×512 ΔNLL panel, validation and conversation text have zero overlap, enforced by the harness.

| method | TF | GEN | general-text damage | patch | format |
|---|---|---|---|---|---|
| FSA Bop + penalty + budget + rehearsal (best, seed 0) | 98.75% (79/80) | 98.75% (79/80) | -0.0001 ± 0.0017 | 249,755 slots (0.33–0.84 MB) | packed ternary |
| Same headline recipe, seed 1 | 98.75% (79/80) | 98.75% (79/80) | -0.0015 ± 0.0019 | 245,281 slots (0.32–0.83 MB) | packed ternary |
| Same headline recipe, seed 2 | 100% (80/80) | 100% (80/80) | +0.0006 ± 0.0020 | 255,585 slots (0.33–0.86 MB) | packed ternary |
| FSA Bop + penalty + budget + energy threshold | 98.75% (79/80) | 97.5% (78/80) | +0.0013 ± 0.0019 | 239k slots (0.32–0.81 MB) | packed ternary |
| FSA Bop + penalty + budget + repair | 97.5% (78/80) | 97.5% (78/80) | +0.0029 ± 0.0016 | 251k slots (0.33–0.85 MB) | packed ternary |
| FSA Bop + penalty + budget | 96.25% (77/80) | 95% (76/80) | -0.0003 ± 0.0021 | 244k slots (0.32–0.82 MB) | packed ternary |
| FSA Bop + penalty (rerun) | 97.5% (78/80) | 97.5% (78/80) | -0.0011 ± 0.0019 | 736k slots (0.82–2.48 MB) | packed ternary |
| FSA Bop + penalty (earlier) | 100% (80/80) | — | +0.0015 ± 0.0017 | 740k slots (0.83–2.50 MB) | packed ternary |
| FSA Bop τ0.7 (run 1) | 100% (80/80) | — | +0.0266 ± 0.0035 | 812k slots (0.89–2.74 MB) | packed ternary |
| FSA Bop τ0.7 (repeat) | 100% (80/80) | — | +0.0237 ± 0.0036 | 670k slots (0.76–2.26 MB) | packed ternary |
| FSA Learn-then-prune, decay 0.3 | 91.25% (73/80) | — | +0.0250 ± 0.0032 | 542k slots (0.64–1.83 MB) | packed ternary |
| FSA Learn-then-prune, decay 0.5 | 88.75% (71/80) | — | +0.0129 ± 0.0028 | 523k slots (0.62–1.77 MB) | packed ternary |
| FSA Fp4 + penalty | 96.25% (77/80) | — | +0.0050 ± 0.0015 | 21.27M slots (18.17–79.75 MB) | fp4 slots |
| LoRA r1 + penalty (baseline, fp32 / measured int4) | 93.75% (75/80) / 93.75% (75/80) | 95% (76/80) / 93.75% (75/80) | -0.0002 ± 0.0019 / +0.0021 ± 0.0017 | 1,376,256 params (2.75 MB bf16 / 0.69 MB int4) | adapter |
| LoRA r2 + penalty (baseline, fp32 / measured int4) | 96.25% (77/80) / 97.5% (78/80) | 95% (76/80) / 95% (76/80) | +0.0019 ± 0.0014 / +0.0021 ± 0.0017 | 2,752,512 params (5.51 MB bf16 / 1.38 MB int4) | adapter |
| LoRA r4 + penalty (baseline, fp32 / measured int4) | 96.25% (77/80) / 97.5% (78/80) | 95% (76/80) / 96.25% (77/80) | -0.0001 ± 0.0020 / +0.0003 ± 0.0016 | 5,505,024 params (11.01 MB bf16 / 2.76 MB int4) | adapter |
| LoRA r16 + penalty (baseline) | 97.5% (78/80) | — | +0.0005 ± 0.0018 | 22,020,096 params (44.04 MB bf16) | adapter |

<p align="center">
  <img src="figures/fig_e2e_frontier.png" width="760" alt="Held-out recall versus general-text damage for every comparable run on the real 8B model.">
  <br>
  <img src="figures/fig_e2e_storage.png" width="640" alt="Held-out recall versus patch size for FSA and LoRA.">
</p>

Ternary in-pair patches use the same packed kernel; synthetic serving is measured below. Fp4 slots need
slot-level fp4 support. Unmerged LoRA adapters add a matrix multiplication per token.

Patch sizes include positions and values: E entropy-codes positions over 66,807,179 in-pair slots plus
one sign bit per ternary slot; I stores a 26-bit index plus the value. Sizes use decimal MB.
The storage figure uses measured bf16 recall for LoRA r1/r2/r4 (95%, 95%, 97.5%);
r16: accuracy of fp32 factors, shown at bf16 size because no bf16 evaluation is available.
LoRA bf16 uses two bytes per parameter; int8/int4 factor sizes include one bf16 absmax scale per rank vector.
Their recall and damage were measured after fake quantization. Every run revokes bit-exactly: logits match
the base on one 512-token window and test recall returns to 0/80. “—” means generated recall was not measured.

- Across seeds with different fact values, the headline’s three-seed mean is 99.2% TF and GEN (range 98.75–100%), with mean ΔNLL −0.0003
  (range −0.0015 to +0.0006) and 245k–256k slots. The figures show three individual seed markers;
  frontier bars are each seed’s paired SE. All three revoke bit-exactly.
- FSA recalls 79–80/80 on all three seeds with different fact values versus LoRA’s 75–78/80 TF and 75–77/80 GEN across tested
  ranks 1–4 and precisions. Gaps of 1–2 items are within noise; 3–4 items may matter, but LoRA’s run-to-run spread is unmeasured:
  ranks 1/2/4/16 still have one seed each. The nearest LoRA size is r1 int4 at 0.69 MB, 75/80 on both
  metrics; the alternative FSA encodings bracket that size rather than matching its byte budget exactly.
- Damage is within noise of zero for the best penalized FSA runs and fp32 penalized LoRA: |ΔNLL| ≤ 0.003,
  SE about 0.002. At this sample size no damage difference is detectable between those methods. The measured
  r1 bf16/int8 evaluations are exceptions (+0.0040/+0.0049); all precision results are in the findings.
  Plain Bop without the penalty costs +0.024 to +0.031; the older rows remain as history.
- The earlier 80/80 TF result was not reproduced exactly: its same-configuration rerun gives 78/80 TF
  and 78/80 GEN at −0.0011 ± 0.0019. Variation between executions is a couple of items on the same seed, within noise.
- Where each method stands today: FSA has the best observed recall at a small patch size; the recall gap
  holds across three FSA seeds with different fact values, with 1–2 item gaps within noise; single-seed LoRA baselines do not establish that FSA beats LoRA overall. Penalized damage is comparable.
  Synthetic in-pair patches show no detectable serving cost; learned patches still need a serving benchmark.
  These in-pair conclusions are superseded by the forced-zero results above, where FSA beats three LoRA seeds.

On the 40-fact in-conversation test, the question follows about 545 tokens of unrelated chat.
Generated recall is 85% for the headline versus 40% for the unbudgeted base (single seed, 40 items); short-prompt recall does not distinguish them beyond noise.

### Bigger-facts test (in-pair variant)

This run used the in-pair write set; the forced-zero results for this task are in the 3 people × 100 table above.

Three fictional people each have 100 facts, with 30 shared attributes and conflicting values across people
for every fact. Training samples 40 facts per step for 600 steps; testing uses 600 held-out items.
Both runs use one seed. LoRA values below come from the run log’s final evaluation, which follows the earlier checkpoint
evaluations. Its result JSON is not yet written because checkpoint-selection evaluation is still running.

| method | TF | GEN | ΔNLL ± paired SE | slots / parameters |
|---|---|---|---|---:|
| FSA headline recipe, budget raised to 1M | 18.2% | 18.2% | +0.0046 ± 0.0024 | 875,105 slots |
| LoRA r16 + penalty 100, fp32 factors | 63.0% | 65.7% | +0.0017 ± 0.0023 | 22,020,096 params |
| Same LoRA, measured bf16 | 64.3% | 64.3% | +0.0034 ± 0.0018 | same factors |
| Same LoRA, measured int8 | 64.7% | 65.2% | +0.0027 ± 0.0018 | same factors |
| Same LoRA, measured int4 | 64.8% | 66.3% | +0.0029 ± 0.0020 | same factors |

FSA fits only 22.9% of its training facts, gives another person’s value on conflicting attributes 14.2%
of the time, and reaches 8.3% teacher-forced in-conversation recall. The failure to fit the training facts points to a capacity or optimization
limit as well as a generalization gap. Its support never settled: 18.8M births, 12.2M deaths and 5.7M
evictions produced an 875k patch. LoRA does much better, and its factors quantize to int4 without loss
on this task. The in-pair budget reruns were superseded by the forced-zero mask.

### EP-1 v1: not a valid test of multi-session consolidation

The FSA arms use the in-pair write set; fp4 and learned scales leave the original pair8 codebook.

In the pre-registered five-session experiment, each session’s self-study was distilled into a patch.
On 100 four-way concept items, the unpatched model scores 0.24 (chance 0.25).
Session-5 accuracy is 0.32 for ternary slots with learned block scales, 0.32 for fp4 slots, 0.32 for
ternary Bop, and 0.28 for LoRA. Bop was added after pre-registration with owner approval.

Every arm failed the thesis: none gained the required 10 concept accuracy points over the unpatched
model. Only that criterion failed. Lure recall (0/50), damage bounds, retention, no lure ratchet and
exact revoke all passed; revoke restored bitwise weights, legal format and identical logits.
Session-5 damage is within noise of zero for the ternary arms; fp4 is +0.0074 ± 0.0020.
v1's in-context ceiling (1.00) was invalid: each probe's context held only the correct policy's source episode, so it already contained the answer. With a non-leaking context (all four rules of the category) the unpatched model picks the right action only 0.30 of the time (ticket 0.27, Q/A 0.34; chance 0.25). The identifiability gate was therefore invalid, and the self-distillation teacher was never competent on the probed skill, so v1 cannot tell whether slots consolidate concepts. EP-1 now uses ground-truth supervised targets on the forced-zero mask ([EP-1 above](#ep-1-learning-rules-instead-of-facts)). Multi-session consolidation remains open.

## Serving (in-pair variant)

All serving measurements below use synthetic ternary in-pair patches, including the same kernel, file size, memory and byte-identical packed-file revoke checks. The forced-zero design needs a sparse overlay or wider block encoding; that path is not built or benchmarked, and its serving cost, file size and memory are pending.

Measured on the released lambda GGUF, with the in-pair patch written into the packed pair8 experts
([`experiments/serving/`](experiments/serving/README.md)). Synthetic patches use uniformly drawn slots with random signs at the size of the earlier Bop patches
(670k) and a 3M stress test, over all experts of MoE layers 14–15; learned patches
(now ~250k slots) have not been benchmarked. Speeds are median tokens/s; the
patched columns are patched ÷ base measured in the same round (median [min, max] over rounds).

| device and kernel | base decode | 670k patch | 3M patch | base prefill (512 tok) | 670k / 3M prefill |
|---|---:|---:|---:|---:|---:|
| Mac M5 Max, Metal (`lut9`) | 174.8 | 0.999 [0.96, 1.01] | 0.999 [0.97, 1.03] | 1,075 | 0.999 / 1.001 |
| Mac M5 Max, CPU 8 threads (`lut8`) | 102.2 | 1.005 | 0.990 | 371 | 1.002 / 1.009 |
| Mac M5 Max, CPU 8 threads (`lut`) | 66.0 | 1.006 | 0.998 | 148 | 1.004 / 0.999 |
| Snapdragon 8 Elite phone, Q8 trunk (`lut8`) | 57.0 | 1.002 [0.97, 1.03] | 0.982 [0.95, 1.03] | 209 | 0.996 / 1.000 |

| | base | 670k patch | 3M patch |
|---|---|---|---|
| Mac model file size | 2,133,006,592 B | unchanged | unchanged |
| phone Q8-trunk file size | 1,656,608,512 B | unchanged | unchanged |
| peak memory (Metal / CPU / phone) | 2.75 GiB / 3.29 GiB / 3.24 GB | identical | identical |
| bytes changed in the file | — | 0.99 MB | 4.3 MB |
| revoke | — | byte-identical original (sha256 match) | byte-identical original |

Paired median ratios stay within about 2% of 1.0, with no consistent direction; brackets show min–max
over rounds. On the Metal text-prefill run the ranges touch 1.0 ([0.993, 1.000] and [0.991, 1.000]).
No slowdown was detected; a min–max range is not a significance test and does not prove zero cost. That is expected: each packed block costs one table
lookup whatever it holds, and a patch never turns an all-zero block nonzero. Applying a 670k-slot patch takes ~46 ms in
memory (apply A → revoke A → apply B: ~137 ms); today's runtime repacks weights at load, so switching users means
patching the file and reloading (~2 s) until a live hook is added. An unmerged rank-16 LoRA in this runtime is
estimated (not measured; the runtime has no adapter support) at +0.5–1.5% per token plus 44 MB per adapter and new
code in every backend; merging it would turn 64 MB of packed experts into 906 MB of bf16.

## Best algorithms so far

### Writers

1. Bop, latent-free ternary (Helwegen et al. 2019, extended to {−1, 0, +1}). Each writable slot keeps an exponential
   moving average of its gradient; a slot is born (0 → ±1) or dies (±1 → 0) only when that average is consistent and
   exceeds a threshold τ, and the average resets on every change. τ directly sets the flip rate, so a small gradient
   never turns into a full ±α step. Best ternary result so far when combined with the local penalty, budget and rehearsal below.
2. Learn-then-prune. fp32 masters + straight-through ternary, AdamW with weight decay on the masters. The model
   learns everything early (~50 steps), then decay pulls weakly supported masters back through the threshold: the
   patch shrank from ~20M to 0.5M slots while recall held.
3. fp4 slots + local output penalty (non-ternary option; in-pair runs only): the penalty is the relative energy of the patch's direct
   output change on general text, Σ‖x·ΔWᵀ‖² / Σ‖x·Wᵀ‖², which avoids KL's routing-chaos floor because it measures each patched layer's direct output change.
4. Sleeping-unit gating (rules, forced-zero mask): pick the 10% of relu² hidden units least active on general text, write
   ternary values only into their rows' forced-zero slots, and add a hinge that keeps their pre-activations negative on
   general text. Best rule learner so far (95.25% dev at +0.001); it does not help facts.

### Preservation and allocation in the Bop facts recipe

- Local output penalty, λ=10: penalize the relative energy of each patched projection's direct output change
  on general text, using the same penalty described above.
- 300k-slot budget: the allowed slots are re-chosen every 10 steps up to step 200
  (the training default; that setting is not stored in the result files), by gradient evidence minus expected
  general-text cost (α² times input-column energy), directing births toward useful, less disruptive slots.
- Rehearsal 0.2: optimize 0.8 × fact loss + 0.2 × ordinary next-token loss on WikiText. These windows share
  no 32-token sequences with the damage panel, but come from the same corpus. With multi-layer patches this hid real
  damage on code ([damage panels](#damage-panels)), so new fact runs drop rehearsal and report FineWeb and code too.
- Energy threshold variant: multiply each row's birth threshold by clamp((e_row / median e)^0.5, 0.25, 4),
  so rows sensitive to general text need more evidence. Repair variant: 50 extra penalty steps with births
  disabled, allowing deaths only. Neither variant establishes an improvement on one seed.

### Ingredients that mattered for every method

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

### What did not work

On the forced-zero mask:

- Straight-through ternary with Adam masters: 2.5–8.5% on the two-person task. A fold-aware variant matched LoRA’s recall
  at about 10× the damage.
- Projecting a trained LoRA into the mask in one shot (least squares, box-constrained, ternary, GPTQ-style, multiscale):
  at most 19% even before rounding, against LoRA’s 79% on the same task.
- Patch scales below half the row scale: 0.25α gives 31% and 0.125α 4% on 40 facts; 0.5α works (97.5%).
- Sleeping-, specific- or random-unit gating for facts: at most 22% validation on the 300-fact task.
- Causal-trace layer choice: 7–10% validation on the 300-fact task.

Earlier, on the in-pair write set: KL penalties (they fight the routing-chaos floor and only cost recall); ternary + smooth penalty +
decay (recall peaks, then gets pruned away); tiny-magnitude fp4 (did not learn in our run; its smallest steps sit near bf16 resolution); patch-only
block scales (no better than plain ternary); continuous values on the slot mask at a single late layer (did not beat
LoRA at the same placement).

Churn controls did not improve on the headline. Against the unbudgeted, unrehearsed Bop + penalty base
(78/80 TF and GEN, −0.0011 ± 0.0019, 735,955 slots), keeping newborn slots alive for 20 steps gives
80/80, +0.0024 ± 0.0017 and 543,145 slots; 50 steps gives 79/80, +0.0052 ± 0.0017 and 769,165 slots.
Requiring 1.5× the birth threshold for death gives 79/80, +0.0029 ± 0.0019 and 783,165 slots.
Recall differences are only 1–2 items, within noise; damage is slightly higher, with grace 50 about three SE above zero.

Utility-based flip pruning also failed as a free size reduction. Slots were ranked by a first-order
utility estimate and removed in 10%, then 5%, then 2.5% chunks only if validation recall held.
The headline accepted no chunk (0% removed); removing its “least useful” 10% raised fact loss despite
a predicted decrease, and validation TF recall fell to 81%. The unbudgeted base accepted one 2.5%
chunk, keeping 717,557 slots, but test TF fell from 78/80 to 75/80 and GEN from 78/80 to 77/80
(the one-item GEN drop is within noise). The first-order estimate is wrong at a trained minimum
(fact loss ≈ 0.01): gradients are near zero, while reverting a flip is a full-α step. Reverting the
predicted least-useful 10% raised base fact loss by +0.258 versus a predicted −0.025; for the headline,
+0.150 versus −0.013. Just 7.5% of the base patch’s flips carry half the positive first-order utility.
The pruning failure diagnoses the estimate, rather than proving there is no removable slack; exact
leave-out by expert and by layer is running.

## Limitations

- Damage. The forced-zero headline runs were trained with WikiText rehearsal and are measured mainly on WikiText; the
  40-fact patch is also clean on FineWeb, but code-panel checks of the 2-layer winners are pending. Multi-layer patches
  trained with rehearsal showed real code damage (up to +0.0076).
- Evaluation. GEN uses held-out phrasings of the trained questions. Realistic questions (new phrasings, multi-fact,
  indirect references) and router-drift measurements are not done yet; teacher-forced scores are known to overstate edits.
- Seeds. Facts winners have 2–3 seeds; EP-1 gating has one seed and dev-set numbers only.
- Capacity. FSA trails LoRA on 3 people × 100 conflicting facts (37.5% versus 57.5–62.5% validation).
- Multi-tenant serving. A patch lives inside the weights, so one copy of the weights serves one user's patch at
  a time. LoRA adapters can be batched across many users on one shared base. FSA fits single-tenant and on-device use.
- Revoke scope. Zeroing the mask restores the base exactly. Removing one session's influence after later sessions
  have learned on top of it is not guaranteed by zeroing that session's slots alone; multi-session tests are pending.
- Format. Forced-zero patches can exceed pair8’s two-pair limit, so they need a sparse overlay or wider block encoding;
  serving cost, file size and memory are not measured yet.
- Privacy. Knowledge in weights can be elicited from the model; a per-user model is the safe deployment.

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
[Cartridges](https://arxiv.org/abs/2506.06266) · [ROME](https://arxiv.org/abs/2202.05262) ·
[MEMIT](https://arxiv.org/abs/2210.07229) · [AlphaEdit](https://arxiv.org/abs/2410.02355) ·
[MEND](https://arxiv.org/abs/2110.11309) · [MALMEN](https://arxiv.org/abs/2311.04661) ·
[MoEEdit](https://arxiv.org/abs/2602.10965) · [Does localization inform editing?](https://arxiv.org/abs/2301.04213) ·
BitNet b1.58 (ternary). FSA is specific to *format-reserved* sparse
capacity: the free slots exist because of the quantization format.

## How claims were checked

Every claim here went through repeated rounds of adversarial review, and the write-up was revised until the findings stopped changing.

## License & attribution

Code: Apache-2.0. Paper & figures: CC-BY-4.0. By [Chutes.ai](https://chutes.ai) / Parallax. Paper and write-up AI-assisted (see the note at the top).
