# End-to-end FSA on the full lambda 8B: running findings

Harness: [`e2e_fsa.py`](e2e_fsa.py). The setup teaches 40 facts about a fictional person by training only a patch on
the routed experts. Training uses 3 phrasings per fact; recall is tested on 2 held-out phrasings, one in Q/A form.
Preservation is measured against the unpatched model on real held-out text.

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
| LoRA r16, local penalty 100 | 35–38% | +0.001 ± 0.002 |
| LoRA r16, no penalty | 56% | +0.66 ± 0.02 |
| **FSA, all experts L14–15, up+down, wd 0.1, no penalty** | **54%** | **+0.086 ± 0.006** |
| FSA, 32 selected experts, L28, down-only, 500k budget | 29% | +0.006 ± 0.002 |
| FSA, same at L15 | 19% | +0.010 ± 0.003 |
| FSA, same at L15, fp4 | 15% | +0.008 ± 0.002 |
| FSA, same at L15, local penalty 100 / 1000 | 0% (no learning) | ≈0 |

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

Both models converged on the same plan:
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
| **LoRA L14–15 + local penalty 100** | **97.5%** | **+0.0005 ± 0.002** | 22M fp32 params |
| LoRA L14–15, no penalty | 97.5% | +0.94 | |
| LoRA L28 down-only + penalty | 38% | +0.001 | |
| **FSA all experts, learn-then-prune (wd 0.3, no penalty, ternary)** | **91%** | **+0.025 ± 0.003** | **542k ternary slots** |
| FSA L28 down-only, cost-aware selection, ternary | 60% | +0.064 | 1.6M |
| FSA L28 down-only, continuous values (diagnostic) | 49% | +0.047 | 2M |
| FSA L28 down-only, tiny fp4 | 0% | +0.007 | never learned |

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
