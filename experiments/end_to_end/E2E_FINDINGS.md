# End-to-end FSA on the full lambda 8B: running findings

Harness: [`e2e_fsa.py`](e2e_fsa.py). The setup teaches 40 facts about a fictional person by training only a patch on
the routed experts. Training uses 3 phrasings per fact; recall is tested on 2 held-out phrasings, one in Q/A form.
Preservation is measured against the unpatched model on real held-out text. Every comparison below uses a single seed per run; repeats use the same seed and do not test robustness across seeds.

## v1: no regularization, loss on every token of the training sentence

| arm | held-out phrasing recall | held-out NLL (base 2.890) | KL | top-1 | revoke |
|---|---|---|---|---|---|
| FSA free, ternary, lr .01/.02/.05 | 71 / 69 / 76% | 4.48–4.59 | 1.56–1.68 | 0.49 | bit-exact |
| FSA full, ternary | 45% | 14.9 | 11.8 | 0.015 | bit-exact |
| LoRA r4 / r16 / r64 | 46–55% | 3.39–3.83 | 0.52–0.91 | 0.61–0.68 | bit-exact |

- **FSA free generalizes better.** It reaches 69–76% recall on phrasings it never trained on, against 46–55% for LoRA,
  and is far more confident on the answers (answer log-prob −0.5 against −1.6 to −1.9).
- **Every arm damages the model, and none is usable as-is.** FSA used 31–41M flipped slots to store 40 facts.
- Two causes:
  - The loss covered the template words too, so every arm was trained to predict boilerplate like "Everyone who
    knows … knows that her".
  - Adam normalizes per coordinate, which turns persistent small gradients into full ±α flips.

## v2: KL(base‖patched) penalty on wikitext anchors, still with whole-sentence loss

| arm | flips | recall | NLL | KL | top-1 |
|---|---|---|---|---|---|
| FSA tern λ1 / 10 / 30 | 7.6M / 7.6M / 7.2M | 48 / 28 / 20% | 2.908–2.914 | 0.059–0.062 | 0.88 |
| FSA fp4 λ1 / 10 | 51M / 52M | 54 / 28% | 2.91 | 0.067–0.070 | 0.88 |
| LoRA r16 λ1 / 10 / 30 | — | 53 / 36 / 30% | 2.888–2.894 | 0.030–0.039 | 0.91–0.93 |

- The penalty cut KL about 25×, but **KL did not fall further with larger λ; only recall fell.** That pattern
  pointed to a floor.

## The chaos floor: KL to the base model is not a damage metric at this scale

Random ternary flips in free slots, with no training:

| layer | k=1 | k=1,000 | k=1,000,000 |
|---|---|---|---|
| 2 | KL 0.027 | 0.027 | 0.028 |
| 15 | 0.022 | 0.023 | 0.026 |
| 28 | 0.007 | 0.015 | 0.023 |

Held-out NLL stays within ±0.005 of base in every case.

Direct perturbation of **one** zero weight in layer 15 (expert 0, down_proj) by ε·α:

| ε | KL | top-1 |
|---|---|---|
| ≤ 0.01 | exactly 0 (absorbed by bf16 rounding) | 1.000 |
| 0.1 | 0.0105 | 0.978 |
| 1 (one ternary flip) | 0.0238 | 0.939 |

Two *different* single flips disagree with each other by KL 0.027, as much as either disagrees with the base. The
divergence is flat across sequence position.

**Interpretation.** Once any weight changes by more than the bf16 resolution, the model moves onto a different but
equally good trajectory. Mean NLL is unchanged, but about 6% of top-1 tokens change. The likely driver is
discrete MoE routing near-ties, propagated through the recurrent EDA state.
- KL ≈ 0.02–0.027 against the base is therefore the floor for **any** always-on edit, whatever its size or method.
- The original preservation target of KL ≤ 0.01 is unreachable by construction.
- A KL penalty mostly fights that floor and throws away recall.

**Consequences for the protocol (v4 onward):**
- Damage is measured as ΔNLL on 40×512 disjoint wikitext tokens, with a paired standard error.
- KL is reported only against the floor.
- Preservation during training uses a smooth **local** penalty instead of KL: the relative energy of the patch's
  direct output change on general text, Σ‖x·ΔWᵀ‖² / Σ‖x·Wᵀ‖² over patched linears. Routing chaos cannot reach this
  quantity.

Re-reading v2 on this basis: FSA λ1 costs about +0.018 nats and LoRA about +0.004 nats, measured on 5 windows with
noise of about ±0.005. Both are small, and LoRA is ahead.

## v4: answer-only loss; damage = ΔNLL on 40×512 disjoint wikitext windows (± paired SE)

| arm | held-out recall | ΔNLL |
|---|---|---|
| **FSA, all experts L14–15, up+down, wd 0.1, no penalty** | **54%** | **+0.086 ± 0.006** |
| FSA, 32 selected experts, L28, down-only, 500k budget | 29% | +0.006 ± 0.002 |
| FSA, same at L15 | 19% | +0.010 ± 0.003 |
| FSA, same at L15, fp4 | 15% | +0.008 ± 0.002 |
| FSA, same at L15, local penalty 100 / 1000 | 0% (no learning) | ≈0 |
| LoRA r16, local penalty 100 | 35–38% | +0.001 ± 0.002 |
| LoRA r16, no penalty | 56% | +0.66 ± 0.02 |

**First end-to-end FSA advantage.** Without a penalty, FSA reaches the same recall as LoRA (54% vs 56%) with about
**8× less damage** (+0.086 vs +0.66).

Weight decay on the masters prunes the patch from 37M to 4M slots during training while validation recall stays at
94–97%. Damage was still falling at step 300 (validation KL 2.4 → 0.17), so acquire-then-prune looks like a natural
FSA recipe.

Three caveats:
- LoRA with the local penalty holds a different point on the frontier (lower recall, no damage). A full frontier
  comparison is still needed.
- The validation→test gap (94% → 54%) comes mostly from the Q/A test format, which is absent from training.
- The local penalty as calibrated for LoRA stops FSA learning entirely. A single ternary birth costs a full α²·E[h²],
  but the penalty has zero gradient at zero. This is the main point of the round-2 brainstorm.

## Round-2 brainstorm (two independent AI-assisted reviews) → v6

Both reviews converged on the same plan:
1. Select births by **actual-flip gain minus preservation cost**, with the budget as a ceiling.
2. Run **continuous values on the same free mask** as the decisive diagnostic for whether the reserve can beat LoRA.
3. Fix checkpoint selection, so it no longer minimizes KL.
4. Fix the local-penalty denominator, which used the patched output.
5. Add Q/A and other training formats for both methods.
6. Try tiny-magnitude fp4.

A code review of the v6 harness found 7 bugs, all fixed; the worst were wrong-sign births and stale optimizer moments
on eviction.

## v6 results (augmented formats, Pareto selection on validation recall and validation ΔNLL)

| arm | held-out recall | ΔNLL | patch |
|---|---|---|---|
| **FSA all experts, learn-then-prune (wd 0.3, no penalty, ternary)** | **91%** | **+0.025 ± 0.003** | **542k ternary slots** |
| FSA L28 down-only, cost-aware selection, ternary | 60% | +0.064 | 1.6M |
| FSA L28 down-only, continuous values (diagnostic) | 49% | +0.047 | 2M |
| FSA L28 down-only, tiny fp4 | 0% | +0.007 | never learned |
| **LoRA L14–15 + local penalty 100** | **97.5%** | **+0.0005 ± 0.002** | 22M fp32 params |
| LoRA L14–15, no penalty | 97.5% | +0.94 | |
| LoRA L28 down-only + penalty | 38% | +0.001 | |

- **Augmented Q/A formats closed the validation→test gap for both methods.**
- **LoRA with the penalty wins the fact task outright**: ~98% recall at no measurable damage.
- **Learn-then-prune is the best ternary recipe so far**: 91% held-out recall from a 542k-slot ternary patch
  (0.6–1.8 MB including slot positions) at +0.025 damage, bit-exact revoke, no penalty at all. Damage was still falling at step 300
  (+0.028 at step 250), so longer pruning may close more of the gap to LoRA.
- **Continuous slots at L28 lost to LoRA at the same placement** (49%/+0.047 vs 38%/+0.001). The free-slot mask is
  less damage-efficient than a low-rank adapter for fact memorization at that placement, independent of ternary.
- Conclusion for the fact task: FSA can approach LoRA's recall at ~50× its damage with a far smaller patch and the
  same kernel; it does not beat LoRA on quality here. The epigenesis task (concepts across sessions, EP-1) is where a
  high-rank distributed patch has a structural case.


**Superseded by v9–v11:** the v6 conclusions above describe the earlier runs. Bop with the local penalty
now reaches damage within noise of penalized LoRA; the budgeted rehearsal run has the best observed recall
per byte. One seed does not establish that FSA beats LoRA overall; seeds pending.

## v7–v8: Bop and stronger learn-then-prune decay

Single seed per run, same 12-format protocol; final step 300. Recall here is TF only.

| arm | TF | ΔNLL ± paired SE | slots | E / I MB |
|---|---|---|---:|---|
| FSA Bop τ0.7 (run 1) | 100% (80/80) | +0.0266 ± 0.0035 | 812,420 | 0.89 / 2.74 |
| FSA Bop τ0.7 (repeat) | 100% (80/80) | +0.0237 ± 0.0036 | 669,705 | 0.76 / 2.26 |
| FSA Bop τ0.9 | 100% (80/80) | +0.0314 ± 0.0033 | 588,673 | 0.68 / 1.99 |
| FSA Learn-then-prune, decay 0.3 | 91.25% (73/80) | +0.0250 ± 0.0032 | 542,205 | 0.64 / 1.83 |
| FSA Learn-then-prune, decay 0.5 | 88.75% (71/80) | +0.0129 ± 0.0028 | 523,229 | 0.62 / 1.77 |
| FSA Fp4 + penalty | 96.25% (77/80) | +0.0050 ± 0.0015 | 21,265,606 | 18.17 / 79.75 |

- Bop τ0.7 recalls 80/80 in both executions at +0.0266/+0.0237 damage. The “+wd” run is a repeat:
  decay does not apply to Bop, which has no real weights to decay. This is same-seed execution variation.
- τ0.9 also recalls 80/80 at +0.0314, so a larger threshold is not a simple damage control.
- Decay 0.5 reduces learn-then-prune damage to +0.0129 but misses nine items. Fp4 with the local penalty
  reaches 77/80 at +0.0050, with a much larger patch that leaves the ternary format.

## v9/v12: Bop with the local output penalty

Single seed per run; same τ0.7 and λ=10, no slot budget. Final step 300.

| arm | TF | GEN | ΔNLL ± paired SE | slots | E / I MB |
|---|---|---|---|---:|---|
| Bop + penalty (rerun) | 97.5% (78/80) | 97.5% (78/80) | -0.0011 ± 0.0019 | 735,955 | 0.82 / 2.48 |
| Bop + penalty (earlier) | 100% (80/80) | — | +0.0015 ± 0.0017 | 739,549 | 0.83 / 2.50 |

- The earlier v9 result remains 80/80 TF; v12 gives 78/80 TF and GEN. Full recall was not reproduced
  exactly: same-seed execution variation is a couple of items.
- Both damage estimates are within noise of zero. No damage difference from penalized LoRA is detectable
  at this sample size, superseding the older lower-damage-LoRA conclusion.

## v11: budget, rehearsal, repair, energy thresholds, and small-rank LoRA

All comparisons use one seed per run; **seeds pending** for the headline configuration. The final endpoint
is step 300, except v11b, which reports step 350 after 50 repair steps. TF requires greedy argmax agreement
on every answer token under teacher forcing. GEN is free-running greedy generation: the whole answer
must match at a word boundary, and a terminator or end-of-sequence must occur within two tokens after it.
GEN tests answer production and stopping; older runs have TF only (GEN “—”).

| arm | TF | GEN | ΔNLL ± paired SE | slots / params | E / I MB or bf16 MB |
|---|---|---|---|---:|---|
| Bop + penalty + budget + rehearsal (best) | 98.75% (79/80) | 98.75% (79/80) | -0.0001 ± 0.0017 | 249,755 | 0.33 / 0.84 |
| Bop + penalty + budget + energy threshold | 98.75% (79/80) | 97.5% (78/80) | +0.0013 ± 0.0019 | 238,662 | 0.32 / 0.81 |
| Bop + penalty + budget + repair | 97.5% (78/80) | 97.5% (78/80) | +0.0029 ± 0.0016 | 250,538 | 0.33 / 0.85 |
| Bop + penalty + budget | 96.25% (77/80) | 95% (76/80) | -0.0003 ± 0.0021 | 243,935 | 0.32 / 0.82 |
| LoRA r1 + penalty λ=100, fp32 as trained | 93.75% (75/80) | 95% (76/80) | -0.0002 ± 0.0019 | 1,376,256 | 2.75 bf16 |
| LoRA r2 + penalty λ=100, fp32 as trained | 96.25% (77/80) | 95% (76/80) | +0.0019 ± 0.0014 | 2,752,512 | 5.51 bf16 |
| LoRA r4 + penalty λ=100, fp32 as trained | 96.25% (77/80) | 95% (76/80) | -0.0001 ± 0.0020 | 5,505,024 | 11.01 bf16 |

The budget caps filled slots at 300k. The allowed set is re-chosen every 10 steps up to step 200
(the harness default; that setting is not stored in the result files), by gradient evidence
minus expected general-text cost (α² times input-column energy). Rehearsal uses 0.8 × fact loss + 0.2 ×
ordinary next-token loss. Its WikiText windows share no 32-token sequences with the damage panel, but
come from the same corpus: near-zero damage is measured in-distribution, with other text untested.
The energy variant multiplies each row's birth threshold by clamp((e_row / median e)^0.5, 0.25, 4),
where e_row is the expected general-text output energy of one birth. Repair keeps λ=10 and disables
births for 50 additional Bop steps. The local penalty measures direct relative output-change energy.

Measured factor-precision evaluations (same single-seed adapters, no retraining): int8/int4 use fake
quantization with one bf16 absmax scale per rank vector. Sizes include those scales; bf16 is params × 2 B.

| baseline | precision | TF | GEN | ΔNLL ± paired SE | MB |
|---|---|---|---|---|---:|
| LoRA r1 + penalty | bf16 | 95% (76/80) | 93.75% (75/80) | +0.0040 ± 0.0020 | 2.75 |
| LoRA r1 + penalty | int8 | 95% (76/80) | 95% (76/80) | +0.0049 ± 0.0018 | 1.38 |
| LoRA r1 + penalty | int4 | 93.75% (75/80) | 93.75% (75/80) | +0.0021 ± 0.0017 | 0.69 |
| LoRA r2 + penalty | bf16 | 95% (76/80) | 96.25% (77/80) | +0.0018 ± 0.0017 | 5.51 |
| LoRA r2 + penalty | int8 | 96.25% (77/80) | 95% (76/80) | +0.0013 ± 0.0014 | 2.76 |
| LoRA r2 + penalty | int4 | 97.5% (78/80) | 95% (76/80) | +0.0021 ± 0.0017 | 1.38 |
| LoRA r4 + penalty | bf16 | 97.5% (78/80) | 95% (76/80) | -0.0010 ± 0.0018 | 11.01 |
| LoRA r4 + penalty | int8 | 96.25% (77/80) | 95% (76/80) | +0.0015 ± 0.0017 | 5.51 |
| LoRA r4 + penalty | int4 | 97.5% (78/80) | 96.25% (77/80) | +0.0003 ± 0.0016 | 2.76 |

- At a comparable byte budget, the best FSA patch recalled 79/80 teacher-forced and 79/80 generated items from 0.33 MB entropy-coded or 0.84 MB indexed, versus 75–78/80 teacher-forced and 75–77/80 generated for every LoRA rank and measured precision tried (r1–r4, 0.69–11.01 MB for int4–bf16). The gap is 1–4 items out of 80 on one seed: suggestive, not established; **seeds pending**. The closest LoRA size is r1 int4 at 0.69 MB: 75/80 TF and 75/80 GEN versus FSA's 79/80 on both. These are two alternative FSA encodings, not an exact equality of byte budgets.
- The best penalized FSA and fp32 penalized LoRA estimates have |ΔNLL| ≤ 0.003 with SE about 0.002:
  no damage difference is detectable at this sample size. This summary does not cover r1 bf16/int8,
  whose measured +0.0040/+0.0049 damage is shown above; quantized accuracy is measured.
- Every FSA and LoRA run revokes bit-exactly: patch-off logits equal the base on one 512-token window,
  and test recall returns to 0/80. Revocation does not establish preservation while enabled.
- Generated recall removes the TF-only limitation for the new runs. Small item gaps and in-distribution
  damage on this single task remain limitations; seeds pending.

E entropy-codes positions plus one sign bit over 66,807,179 free slots; I uses a 26-bit index plus the
value per slot. Both are position-inclusive estimates in decimal MB, rather than serialized training files.
