# End-to-end FSA on the full lambda 8B: running findings

**Write-set scope.** The FSA design writes only into the deterministic pair8 forced-zero mask shipped with the model: the two unselected pairs, 4 of every 8 weights (at least half). Zeroing that mask restores the exact pair8 base. Filling those pairs can exceed pair8’s two-pair limit, so serving needs a sparse overlay beside the packed kernel or wider block encoding. That path is not built or benchmarked; forced-zero results, serving cost, file size and memory are pending.

All reported FSA adaptation runs use the **in-pair variant** (one exception: the v1 "FSA full" control wrote all structural zeros, outside pair8): zeros inside already-active pairs, with 66,807,179 in-pair slots at layers 14–15 (late-layer ablations use their stated placement). These zeros sit inside the selected pairs, disjoint from the forced-zero mask, and filling them keeps pair occupancy legal; plain ternary patches keep pair8 legal, while fp4 and learned scales leave the codebook. In-pair and forced-zero slots lie in different pairs. Below, “slots” and historical “free” labels denote the in-pair write set; the full-support control is labeled separately. Same-kernel, no-extra-compute, unchanged file size or memory, and byte-identical packed-file revoke claims cover only plain ternary in-pair patches; packed serving measurements use synthetic patches.

Harness: [`e2e_fsa.py`](e2e_fsa.py). The setup teaches 40 facts about a fictional person by training only a patch on
the routed experts. Training uses 3 phrasings per fact; recall is tested on 2 held-out phrasings, one in Q/A form.
Preservation is measured against the unpatched model on real held-out text. Historical comparisons use one seed per run; same-seed repeats do not test robustness across seeds. The headline recipe now has three seeds with different fact values, reported in v12 below.

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

Random ternary flips in in-pair slots, with no training:

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
2. Run **continuous values on the same in-pair mask** as the decisive diagnostic for whether the reserve can beat LoRA.
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
- **Continuous slots at L28 lost to LoRA at the same placement** (49%/+0.047 vs 38%/+0.001). The in-pair mask is
  less damage-efficient than a low-rank adapter for fact memorization at that placement, independent of ternary.
- Conclusion for the fact task: FSA can approach LoRA's recall at ~50× its damage with a far smaller patch and the
  same kernel for the ternary in-pair variant; it does not beat LoRA on quality here. The structural case for a high-rank distributed patch on the epigenesis task (concepts across sessions,
  EP-1) was a pre-v1 hypothesis; EP-1 v1 then failed its thesis, but v1 turned out not to be a valid test (see the EP-1 v1 section).


**Superseded by v9–v11:** the v6 conclusions above describe the earlier runs. Bop with the local penalty
now reaches damage within noise of penalized LoRA; the budgeted rehearsal run has the best observed recall
per byte. The headline now holds across three FSA seeds with different fact values (v12 below), but LoRA’s seed spread is unmeasured.

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

The table below reports seed 0; the headline’s two additional seeds are in v12 below. LoRA baselines still use one seed each. The final endpoint
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

- The headline recalls 79–80/80 TF and GEN across three seeds with different fact values, versus 75–78/80 TF and 75–77/80 GEN
  for ranks 1–4 at measured precisions. LoRA’s run-to-run spread is unmeasured; the FSA consistency
  does not establish an overall advantage. The nearest LoRA size is r1 int4 at 0.69 MB, 75/80 on
  both metrics; FSA’s two position encodings bracket that size rather than match it exactly.
- The best penalized FSA and fp32 penalized LoRA estimates have |ΔNLL| ≤ 0.003 with SE about 0.002:
  no damage difference is detectable at this sample size. This summary does not cover r1 bf16/int8,
  whose measured +0.0040/+0.0049 damage is shown above; quantized accuracy is measured.
- Every FSA and LoRA run revokes bit-exactly: patch-off logits equal the base on one 512-token window,
  and test recall returns to 0/80. Revocation does not establish preservation while enabled.
- Generated recall removes the TF-only limitation for the new runs. Small item gaps and in-distribution
  damage remain limitations; LoRA seeds are still pending.

E entropy-codes positions plus one sign bit over 66,807,179 in-pair slots; I uses a 26-bit index plus the
value per slot. Both are position-inclusive estimates in decimal MB, rather than serialized training files.


## v12: headline seeds, churn ablations and utility pruning

The headline is Bop τ0.7 + local penalty 10 + a 300k gain-minus-cost slot budget + rehearsal 0.2.
Endpoints are step 300. E / I are position-inclusive decimal MB, as above.

| seed | TF | GEN | ΔNLL ± paired SE | slots | E / I MB |
|---|---|---|---|---:|---|
| 0 (v11c) | 79/80 | 79/80 | -0.0001 ± 0.0017 | 249,755 | 0.33 / 0.84 |
| 1 | 79/80 | 79/80 | -0.0015 ± 0.0019 | 245,281 | 0.32 / 0.83 |
| 2 | 80/80 | 80/80 | +0.0006 ± 0.0020 | 255,585 | 0.33 / 0.86 |

Each headline seed varies both fact values and training randomness: seed 1 differs from seed 0 on
33/40 attribute values, and seed 2 on 34/40. Mean TF and GEN are both 99.2% (range 98.75–100%); mean ΔNLL is −0.0003 (range −0.0015 to
+0.0006), using 245k–256k slots. All three revoke bit-exactly. The frontier and storage figures plot
three individual headline markers, with each frontier marker’s paired SE. LoRA r1/r2/r4/r16 remain
single seed: the recall gap is more consistent on the FSA side, but LoRA’s run-to-run spread is unmeasured.
Rehearsal and evaluation remain on the same corpus, with no shared 32-token sequences.

Churn ablations use the unbudgeted, unrehearsed Bop τ0.7 + penalty 10 base, not the headline.
Grace N prevents death within N steps of birth; death τ×1.5 requires 1.5× the birth threshold.

| arm | TF | GEN | ΔNLL ± paired SE | slots |
|---|---|---|---|---:|
| Grace 20 | 80/80 | 80/80 | +0.0024 ± 0.0017 | 543,145 |
| Grace 50 | 79/80 | 79/80 | +0.0052 ± 0.0017 | 769,165 |
| Death τ×1.5 | 79/80 | 79/80 | +0.0029 ± 0.0019 | 783,165 |
| Base (v12) | 78/80 | 78/80 | -0.0011 ± 0.0019 | 735,955 |

None improves on the headline. Damage is slightly higher than the base (grace 50 is about three SE
above zero), while recall differs by only 1–2 items. These are negative results.

After training, utility pruning ranks slots by a first-order estimate and tries 10%, then 5%, then 2.5%
chunks, accepting only when validation recall holds.

| patch | removed | slots kept | test TF before → after | test GEN before → after |
|---|---|---:|---|---|
| Headline (v11c) | 0%; no chunk accepted | 249,755 | 79/80 → 79/80 | 79/80 → 79/80 |
| Unbudgeted base (v12) | one 2.5% chunk | 717,557 | 78/80 → 75/80 | 78/80 → 77/80 |

For the headline’s “least useful” 10%, the estimate predicted lower fact loss, but removing them
raised it and reduced validation TF recall to 81%. The patches have little removable slack under
this estimate; pruning is not a free size reduction.

**Why pruning stopped.** The stop rule did not fire spuriously. Every tried chunk was measured, and
every chunk but one cost validation recall:

| patch | chunk | flips reverted | val TF / GEN after (before 98.75 / 98.75) | fact loss before → after |
|---|---|---:|---|---|
| unbudgeted base | 10% | 73,595 | 82.5 / 83.75% | 0.008 → 0.266 |
| | 5% | 36,797 | 91.25 / 90% | 0.008 → 0.066 |
| | 2.5% (accepted) | 18,398 | 98.75 / 98.75% | 0.008 → 0.015 |
| | next 2.5% | 17,938 | 77.5 / 75% | 0.015 → 0.300 |
| headline (v11c) | 10% | 24,975 | 81.25 / 82.5% | 0.011 → 0.161 |
| | 5% | 12,487 | 90 / 88.75% | 0.011 → 0.088 |
| | 2.5% | 6,243 | 92.5 / 91.25% | 0.011 → 0.046 |

The utility distribution is not flat in the first-order sense. 7.5% of the base patch's flips carry
half the positive first-order utility, 30% carry 90% and 50% carry 99% (headline: 12% / 43% / 67%).
But 38% (base) and 20% (headline) of flips have *negative* first-order utility, and the estimate is
wrong at this point. Exact group leave-out of 10% of the flips (fact-loss increase, measured vs predicted):

| group of flips reverted | base: predicted → exact | headline: predicted → exact |
|---|---|---|
| lowest-utility 10% | −0.025 → **+0.258** | −0.013 → **+0.150** |
| random 10% | +0.010 → +0.069 | +0.015 → +0.157 |
| middle 10% | +0.003 → +0.005 | +0.008 → +0.040 |
| highest-utility 10% | +0.080 → +1.970 | +0.070 → +2.220 |
| lowest 1% | −0.006 → −0.001 | −0.004 → +0.010 |

The trained patch sits at a fact-loss minimum (loss ≈ 0.01), where gradients are near zero and carry no
removal information, while a ternary flip is a full-α step. The reverts are far outside the linear regime
(the top decile's exact cost is 25–30× the prediction). The flips the estimate calls harmful are not removable as a group: on the base patch reverting them costs more
than reverting a random 10% (+0.258 vs +0.069); on the headline the two are similar (+0.150 vs +0.157). The ranking keeps only its coarse
order (top ≫ middle). Exact leave-out by whole expert and by linear is running; results to follow.

The accepted 2.5% chunk on the base passed validation but lost test items (78 → 75/80 TF): with 80
validation items, a pass/fail rule on recall alone is itself noisy.

Sources: seed/churn JSONs in `data/` and matching raw logs in `reviews/paper_r6/raw_logs/`;
seed 0 also matches `reviews/paper_r5/raw_logs/v11c_rehearsal02.log`. Pruning JSONs are
`data/prune_v11c_u.json` and `data/prune_v12base_u.json`, checked against their raw logs.

**Churn.** Bop flip churn, from exact per-slot counters (v12) and from the every-25th-step log samples
for older runs, which only show the trend. Unbudgeted runs: births are mostly one-shot. The v12 base had
1.09M births, 352k deaths (89% within 50 steps of birth) and 152k rebirths for a 736k final patch. 112k
slots were born at least twice, 26k at least three times; one slot was born 25 times. 89% of the final
nonzeros are first births. Maturity grace cuts flip-flopping (slots born ≥ 2 times: 113k → 32k at grace 20,
20k at grace 50) without improving recall or damage; the tested controls showed no benefit
(single seed, unbudgeted base, recall changes within noise). Budgeted runs churn far more, because each support
refresh evicts live slots. The headline seeds made 3.8–3.9M births and 1.2–1.4M Bop deaths, plus
2.3M budget evictions, to end with 245–256k slots, about 15 births per surviving slot. Across eight sampled steps (100–275), v11c and v12 base averaged 2,143 and 258 births per sampled
step, respectively (about 8×). These sparse samples do not establish means over all training steps. The 1M-budget bigger-facts run made 18.8M births, 12.2M deaths
and 5.7M evictions for an 875k patch: its support never settled, which fits its failure to fit the
training facts.

**In-conversation questions.** The same test question asked after ~545 tokens of unrelated chat
(40 facts) separates patches that look identical on short prompts. Headline (v11c): 85% generated
recall. Unbudgeted base (v12): 40%. The pruned patches score the same as their parents. Single seed, 40 items.

**Evaluation protocol notes.**
- On this model, eval batch composition alone changes outputs. Batching 16 prompts instead of one
  changed base-model logits by up to 2–5 and flipped 15/32 greedy continuations, while 4 identical
  rows in one batch agreed exactly. This is the chaos floor again: kernel shapes change bf16 rounding,
  and that flips routing. All v11+ evals therefore run one sequence at a time, as v1–v10 did.
  Recall differences of one or two items between otherwise identical configs are at this noise level.
- The legacy 5 × 512 held-out set behind the KL / top-1 columns shares 57% of its 32-token sequences
  with the training anchors. The 40 × 512 ΔNLL windows, validation windows and conversation text share
  none, and the harness now aborts on any overlap there. KL / top-1 on the legacy set should not be read
  as held-out for penalty runs.

## Bigger-facts test: capacity / optimization failure

Three fictional people × 100 facts = 300 facts; 30 attributes are shared across all three people,
and every fact has conflicting values across people. Each step samples 40 facts; training runs for
600 steps, with 600 held-out test items. Both methods use one seed.

| method | TF | GEN | ΔNLL ± paired SE | slots / parameters |
|---|---|---|---|---:|
| FSA headline recipe, budget raised to 1M | 18.2% | 18.2% | +0.0046 ± 0.0024 | 875,105 slots |
| LoRA r16 + penalty 100, fp32 factors | 63.0% | 65.7% | +0.0017 ± 0.0023 | 22,020,096 params |
| Same LoRA, measured bf16 | 64.3% | 64.3% | +0.0034 ± 0.0018 | same factors |
| Same LoRA, measured int8 | 64.7% | 65.2% | +0.0027 ± 0.0018 | same factors |
| Same LoRA, measured int4 | 64.8% | 66.3% | +0.0029 ± 0.0020 | same factors |

FSA’s training-fact recall is only 22.9%. On conflicting attributes it gives another person’s value
14.2% of the time; teacher-forced in-conversation recall is 8.3%. The recipe fails to fit its training
facts, so the limitation is capacity or optimization, not only generalization. LoRA does much
better, and its factors quantize to int4 without loss on this task. Budget-scaled FSA reruns
are pending. This is a measured failure of the current recipe.

FSA source: `data/v12big_fsa_c.json` and `reviews/paper_r6/raw_logs/v12big_fsa_c.log`.
LoRA source: `reviews/paper_r6/raw_logs/v11big_lora_r16.log`. The harness prints the final adapter's
three QUANT lines *before* its FINAL line; QUANT lines after FINAL belong to checkpoint-selection
candidates (the first is step 400, 40% recall). An earlier version of this table took those by mistake.

## Beat-LoRA push on conflicting facts (in-pair write set; superseded by the forced-zero mask)

Task: 2 fictional people × 50 attributes, every attribute held by both with different values (100
conflicting facts), 40 facts sampled per step, 400 steps, layers 14–15; 200 held-out test prompts. GEN is the
headline; ΔNLL on the 40 × 512 panel. One seed each unless stated. All FSA rows write the **in-pair** set.

| arm | test TF | test GEN | ΔNLL ± SE | patch |
|---|---|---|---|---|
| LoRA r16 + penalty 100 | 96.0% | 94.5% | +0.0010 ± 0.0017 | 22.0M fp32 params |
| Fold-aware LoRA (forward = α·ternary(mask⊙BA/α), STE), in-pair | 95.5% | 95.0% | **+0.0206 ± 0.0029** | 3.97M slots |
| Bop + penalty + 350k netcost budget + rehearsal | 64.0% | 67.0% | +0.0051 ± 0.0021 | 322k slots |
| same + conflict-group (paired) batches | 52.0% | 51.0% | +0.0025 ± 0.0018 | 322k slots |
| static slot pool by per-person signed gradient at init | 1.0% | 1.0% | +0.0034 ± 0.0018 | 272k slots |
| fold with periodic error-feedback (every 30 steps) | 0% | 0% | 0 | 0 slots |

- The fold-aware arm ties LoRA on recall at about 20× its damage; no in-pair arm beat LoRA. On 1 person × 40
  facts the fold-aware arm reached 97.5% GEN at +0.0175 ± 0.0026 (1.8M slots), again far more damage than LoRA.
- The error-feedback fold never wrote a slot. The in-pair component of the adapter update is tiny relative to the
  ternary step: |ΔW|/α on in-pair slots has median 0.009 and 90th percentile 0.05, so a 0.5 threshold is never
  crossed.
- Validation and test phrasings differ: LoRA r16's validation recall was 76–79% while its test recall was
  94.5–96%. Compare arms on test numbers only.

**Where LoRA's update lives.** Splitting the energy of the trained LoRA r16 update (3 × 100 task) by position
class: in-pair zeros 14.7%, zero pairs that may be opened while keeping pair8 (≤ 2 active pairs) 4.2%, zero
pairs beyond pair8 47.3%, base nonzeros 33.8%. In layers 14–15, 94.8% of 8-weight blocks already have 2 active
pairs. The continuous LoRA update restricted to in-pair slots recalls 0% (ΔNLL −0.0010); its ternary projection
on the top 250k in-pair slots also 0%. The in-pair set cannot express that adapter's solution; this motivated
the forced-zero mask (the two unselected pairs per block, 50% of weights), which holds the 51.5% of the energy
on the non-selected pairs.

**Placement facts.** Routed experts use relu² with up/down only (no gate). Each MoE layer has 128 routed experts
on a 384-d latent and one shared expert on d_model 1152; the shared expert is dense bf16 with no zeros, so it
offers no write set. LoRA and FSA in all runs above patch exactly the same modules.

**Pruning a trained patch.** First-order utility (−Δw·∂L/∂w) is unreliable at the trained minimum: reverting
the 10% of flips it ranked least useful raised the training fact loss by +0.26 (predicted −0.025), and the top
10% raised it by +1.97 (predicted +0.08). Every pruning chunk except one 2.5% chunk lost validation recall.

**Protocol notes.** Eval batch composition changes outputs on this model (bf16 rounding flips routing), so all
evals are batch-1. Bop flip churn on budgeted runs is large (headline seeds ≈ 3.9M births and 2.3M budget
evictions for a 250k-slot patch); maturity grace and a 1.5× death threshold reduced flip-flopping without
improving recall or damage.

Forced-zero re-evaluation (mask built from the same 725B export, sha256 25a85fd2…e215d, verified bit-for-bit;
the harness also asserts W₀[mask] = 0 at attach) is running: the continuous → box-constrained → ternary
feasibility ladder, the 40-fact headline with 3 seeds vs LoRA r1/r4/r16, the conflict tasks with fold-aware STE
and Bop vs LoRA r16 with random-placement controls and 2 seeds, causal tracing for placement, and smaller patch
scales.

## EP-1 v1: five-session self-distillation

The pre-registered multi-session experiment distilled each session’s self-study into a patch.
On 100 four-way test concept items the unpatched model scores 0.24 (chance 0.25).

| arm | session-5 concept accuracy | concept gain | thesis |
|---|---:|---:|---|
| Ternary in-pair slots with learned block scales | 0.32 | +8 points | failed |
| Fp4 slots | 0.32 | +8 points | failed |
| Ternary Bop (added after pre-registration, owner-approved) | 0.32 | +8 points | failed |
| LoRA | 0.28 | +4 points | failed |

Every arm failed only the required concept gain of at least +10 points. Lure recall (0/50), damage
bounds, retention, no lure ratchet and exact revoke passed in every arm. Revocation restores bitwise
weights, legal format and identical logits. Session-5 damage is within noise of zero for both ternary
arms; fp4 is +0.0074 ± 0.0020. v1's in-context ceiling (1.00) was invalid: each probe's context held only the correct policy's source episode, so it already contained the answer. With a non-leaking context (all four rules of the category) the unpatched model picks the right action only 0.30 of the time (ticket 0.27, Q/A 0.34; chance 0.25). The identifiability gate was therefore invalid, and the self-distillation teacher was never competent on the probed skill, so v1 cannot tell whether slots consolidate concepts. EP-1 v2.1 uses coarser categorical rules and a full-table context, or a ground-truth supervised target if the teacher still fails. The multi-session thesis is not established. Details: `experiments/epigenesis/EP1_V1_FINDINGS.md`.
Source: `experiments/epigenesis/out/ep1/summary.json` (also copied in `reviews/paper_r6/ep1/`).
