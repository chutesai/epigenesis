# Results — the dense fp32 mechanism assay

> **Scope caveat (read first).** Every number here is from `src/freeslot.py`, which trains **dense fp32
> weights** on a **random 50% mask** in a 3-layer fact-memorizing MLP (917,504 weights, ρ=0.5, base
> saturated with 128k synthetic facts). **pair8/ternary is not in the adaptation loop, and there is no
> real model.** `K` = retained bits = `N·log₂V − ΣCE/ln2` (a log-loss improvement over a uniform
> predictor, *not* a semantic-knowledge measurement). See [`../ROADMAP.md`](../ROADMAP.md) for the gating
> experiments that decide whether any of this transfers to the real format.

## 1. FSA reserve capacity ([`fs_packnet`](../experiments/capacity/fs_packnet.json))

| new facts | knowledge acquired | base retained (masked) | base retained (unmasked) | reserve filled |
|---:|---:|---:|---:|---:|
| 500 | 5.0 k | 1.000 | −42.0 | 499 k bits |
| 2,000 | 20.0 k | 1.000 | −45.4 | 499 k |
| 8,000 | 80.0 k | 1.000 | −32.3 | 499 k |
| 32,000 | 316.6 k | 1.000 | −16.9 | 499 k |
| **128,000** | **490.6 k** (ceiling) | 1.000 | −2.7 | 499 k |
| 256,000 | 256.5 k (overflow) | 1.000 | −1.6 | 499 k |

Retention is 1.000 **only with the mask** (by construction). FSA fills ~the whole reserve regardless of
load, so it is storage-efficient only near saturation (a sparse-reserve-training opportunity).

## 2. Tuned LoRA — capacity scales with rank (best LR per rank)

| rank | LR | peak knowledge | adapter params | adapter bits (FP16) | peak knowledge/bit |
|---:|---:|---:|---:|---:|---:|
| 16 | 2e-4 | 85 k | 53 k | 0.85 Mb (0.11 MB) | 0.10 |
| 64 | 1e-4 | 388 k | 213 k | 3.4 Mb (0.43 MB) | 0.11 |
| 256 | 1e-4 | **1,050 k** | 852 k | 13.6 Mb (1.70 MB) | 0.077 |
| 1024 | 1e-4 | **1,360 k** | 3.4 M | 54.5 Mb (6.82 MB) | 0.025 |

LoRA is **more capable** (r≥256 out-acquires FSA's ~490k ceiling) — once its LR is tuned. The earlier
"LoRA collapses" was an LR-instability artifact; the effective LR must scale **down** with fact-load.

## 3. FSA vs LoRA — storage efficiency, counted honestly (fp32-vs-fp32)

Both methods were **trained in fp32**. Counting each in the representation it was actually trained in, at
saturation:

| method | knowledge | fp32 bits trained | knowledge / fp32 bit |
|---|---:|---:|---:|
| FSA (490 k in 458,752 reserve weights) | 490 k | 14.7 Mb | **0.033** |
| LoRA r=64 (388 k in 213 k params) | 388 k | 6.8 Mb | **0.057** |
| LoRA r=256 (1,050 k in 852 k params) | 1,050 k | 27 Mb | 0.039 |

**LoRA is more storage-efficient, not less.** A ternary-reserve win (FSA stored as trits) is a *prediction*
contingent on [Gate 0](../ROADMAP.md), and **genuinely open** — the format assay measures *from-scratch* QAT
at matched loads, not a *frozen-base* reserve, so it cannot forecast this; even a pessimistic ternary yield
(priced at the corrected ~1.58 bits/trit) needn't fall below fp32 LoRA's 0.057. The earlier "~10×" figure
divided fp32 knowledge by hypothetical trit storage — a unit conversion, now retracted.

## 4. Middle ground — `soft` (consistency-penalized reserve, evaluated UNMASKED)

| new facts | knowledge | base retained (unmasked, no task-ID) |
|---:|---:|---:|
| 500 | 5.0 k | 0.98 |
| 2,000 | 20.0 k | 0.94 |
| 8,000 | 79 k | 0.78 |
| 32,000 | 260 k | 0.12 |
| 128,000 | 204 k | 0.08 |

A single always-on model with **graceful degradation** — good for modest skills. λ is a plain
retention↔acquisition dial (λ=3 over-preserves and tanks acquisition at high load; λ=0.3 forgets sooner),
so no further λ sweeps. `softfreeze` (letting the base drift even 0.5%) **craters** a saturated base.

## Figures
[`fig1_retention`](../figures/fig1_retention.png) · [`fig2_reserve`](../figures/fig2_reserve.png) ·
[`fig3_vs_lora`](../figures/fig3_vs_lora.png) · [`fig4_soft`](../figures/fig4_soft.png). Regenerate:
`python figures/make_figs2.py experiments/capacity`.
