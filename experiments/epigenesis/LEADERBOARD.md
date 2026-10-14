# FSA forced-zero leaderboard

Write set: pair8-forced zeros only (mask sha 25a85fd2…), plain ternary at row scale α unless noted, revoke = zero mask.
Test numbers are held-out test phrasings; "val" marks validation-only (in-progress) numbers, which are not comparable
with test. ΔNLL is paired ± SE on 40×512 held-out WikiText windows unless a panel is named. One seed unless listed.

## Progress digest

**Digest 2026-10-14 ~08:15Z.**
- **Landed (reviewed):**
  - GL_ECTRL_s1: 1M budget, 800 steps; 99.5% test; code +0.001.
  - GL_Drs_s1: 99.5%; code +0.007, fails the gate.
  - GL_E600_s1: dips to 48% at step 500, then recovers; 99.5% test; code +0.0026.
  - Measured packed sizes with the scale derived from the base: 1M 1.66 MB (≈53× smaller than LoRA r16), 4M 2.66 MB, 16M 3.31 MB.
  - S2 rank-one: the solver is invalid, so there is no substrate verdict (parked).
- **Code reviewed GO and deployed (v60):**
  - in-topic panel: PANEL_C61 is building on C with an 8M-token fresh FW-HQ held-out slice; 40 windows, zero census overlap.
  - CC1 composition contract: C61_CC1_unit_s0 is running on A; the full-mask pair is queued.
  - derived-scale serializer.
  - PR3 probes: queued on B behind the profiler.
  - C64 speed profiler: running on B. It also decides whether same-argv runs are deterministic.
- **Running:**
  - R20 composition decisive evals on B, about 16–21 h in, generation-bound. Early THREE-METRICS, all banks summed: full 25.7%, block 5.3% (unreviewed).
  - J/F v59 at about step 76 of 150.
  - Ladder seeds, all one seed:
    - ECTRL s0/s2 on A, about 240 s per update;
    - EB4M s1 on Q;
    - LoRA800 s1 on A;
    - B16M s0 on A;
    - B4M s0/s2;
    - CTRLv/Av/Adampg (decline arms; no new requeues).
- **Correction:** the ECTRL/CTRL divergence from update 2 is no longer attributed to GPU nondeterminism. Its cause is not isolated yet.

**2026-10-14 ~06:30Z: GL_E600_s1 (800 steps, slot selection frozen at step 600) recovered to 100% val GEN at step 800** (val curve 36.7/50/51.7/69.2/48.3/74.2/92.5/100 at steps 100–800, val ΔNLL +0.086 at step 500). **Test GEN 99.5% (597/600); damage WikiText −0.023 (rehearsal), FineWeb −0.004 ± 0.002, code +0.0026 ± 0.0016** (point estimate under the 0.003 gate, upper bound not); 0.81M nnz. Result review confirmed-with-caveats (`reviews/experiments/gl_e600_test/`): numbers and the freeze-600 vs freeze-200 one-variable contrast verified; the reviewer notes this pair alone does not identify *why* the mid-run dip happens (short training vs run variance). Third data point that the step-300→400 "decline" is short-training / run-to-run variance rather than an instability: the 800-step 1M run never declined, and this run recovers fully by step 800. One seed; see the decline rows below.

**Seed policy (2026-10-14):** every gap-ladder (GL_*) row on this page is a single seed unless stated otherwise. Runs of the same recipe and seed diverge from Bop update 2 (cause not isolated: GPU nondeterminism or step-count-dependent path; same-argv replay via the C64 profiler will decide; GPU nondeterminism), so single-seed differences of a few points are within run-to-run variance. Promotion requires two seeds, and damage gates are judged on the one-sided 95% upper bound, not the point estimate.

**2026-10-14: derived-scale packed sizes (result review confirmed-with-caveats; `reviews/experiments/c62_sizes_result/`).** The packed file no longer stores the per-row fp32 scale α; the loader re-derives it from the base (α = mean |nonzero base weight| of the row), and the derived α is verified bit-identical on load. Every file also round-trips bit-identically. Size is the file as shipped, JSON metadata header included, in decimal MB (bytes / 10^6). Per fact = size / facts written. Ratio = LoRA r16 bf16 (88,080,384 B) / FSA size.

| patch | slots | packed, scale derived (shipped) | packed, scale stored | vs LoRA r16 | per fact (derived) |
|---|---|---|---|---|---|
| GL_CTRL_s1 (1M budget, 400 steps) | 0.81M | 1.66 MB (1,659,543 B) | 7.15 MB | ≈ 53× | 5.5 KB |
| GL_ECTRL_s1 (1M budget, 800 steps) | 0.81M | 1.66 MB (1,657,298 B) | 7.14 MB | ≈ 53× | 5.5 KB |
| GL_B4M_s1 (4M) | 1.60M | 2.66 MB | 8.14 MB | ≈ 33× | 8.9 KB |
| GL_B16M_s1 (16M) | 2.13M | 3.31 MB | 8.80 MB | ≈ 27× (26.6×) | 11.0 KB |
| GL_Drs_s1 (per-row learned scale; learned fp16 scale kept, 2.75 MB) | 0.77M | 4.45 MB | 9.94 MB | ≈ 20× | 14.8 KB |
| R20 per-person banks (100 facts each) | — | 1.51–1.64 MB each | 7.0–7.1 MB each | ≈ 54–58× | ≈ 15–16 KB |
| F40 (40 facts, layers 14,15; bopck s0) | 0.175M | 0.51 MB | 3.25 MB | — (vs 4-layer LoRA r16) | ≈ 12.7 KB |
| LoRA r16 (R4_lora16_rand4_s1) | 44M params | 88.08 MB bf16 | 88.08 MB bf16 (176.7 MB fp32 saved) | 1× | 294 KB |

- **Where the bytes go (1M patch, 1.66 MB):** 1.03 MB positions, 0.10 MB signs, 0.53 MB JSON metadata header (manifest plus framing). The header counts as part of the patch, because the file does not load without it.
- **Caveats from the review:** CTRL's scale-stored file is 7.15 MB; 7.14 MB is ECTRL's. The 88 MB LoRA numerator comes from the ladder reference log, not the serial JSONs. The round trip was not re-executed by the reviewer. ECTRL is still one seed.
- **Follow-up (not blocking):** compact the 0.53 MB JSON header (binary or gzip'd metadata, or per-unit lists into the position stream).

**2026-10-14: measured packed sizes, scale stored (superseded by the derived-scale entry above).** FSA patch sizes before this entry were estimated encodings or the unpacked `final.pt`. A lossless packed file now exists for each patch below (delta-coded coordinates + packed signs; auxiliary tensors stored verbatim); every file round-trips bit-identically. Sizes are decimal MB; per fact = size / facts in the task.

| patch | slots | estimated (old) | packed (measured) | packed / est. | per fact (packed) |
|---|---|---|---|---|---|
| GL_CTRL_s1 (1M budget, 400 steps) | 0.81M | 2.95 MB | 7.15 MB (final.pt 10.45 MB) | 2.4× | 23.8 KB |
| GL_ECTRL_s1 (1M budget, 800 steps) | 0.81M | 2.95 MB | 7.14 MB | 2.4× | 23.8 KB |
| GL_B4M_s1 (4M) | 1.60M | 5.80 MB | 8.14 MB | 1.4× | 27.1 KB |
| GL_B16M_s1 (16M) | 2.13M | 7.74 MB | 8.80 MB | 1.14× | 29.3 KB |
| GL_Drs_s1 (per-row learned scale) | 0.77M | 5.55 MB | 9.94 MB | 1.8× | 33.1 KB |
| R20 per-person banks (100 facts each) | — | — | 7.0–7.1 MB each | — | ≈ 70 KB |
| F40 (40 facts, layers 14,15; bopck s0) | 0.175M | 0.63 MB | 3.25 MB | 5.2× | ≈ 81 KB |
| LoRA r16 (R4_lora16_rand4_s1) | 44M params | 88.1 MB bf16 | 88.1 MB bf16 (176.7 MB fp32 saved) | — | 294 KB |

- **Where the bytes went (1M patch):** 5.5 MB of the 7.15 MB was the per-row fp32 scale (α = mean |nonzero base weight| of the row, 1,376,256 rows × 4 B). α is derivable from the base, so the derived-scale files above drop it.
- **Against LoRA r16 (88 MB bf16), with the scale stored:** ≈ 12× smaller at 1M, ≈ 11× at 4M, ≈ 10× at 16M. Replaced by the derived-scale ratios above (≈ 53×, ≈ 33×, ≈ 27×).
- With the scale stored, the per-person banks and the F40 patch cost more per fact than the 300-fact patch (≈ 70 and ≈ 81 KB against 23.8 KB), because each file stored the per-row scale for every row of its layers.

**2026-10-14 ~02:00Z: GL_ECTRL_s1, 1M budget trained for 800 steps.** Independent result review: confirmed-with-caveats (`reviews/experiments/gl_ectrl_drs_test/`).
- **Test GEN 99.5% (597/600)**, scored at the final step.
- **Damage:** WikiText −0.028 (rehearsal), FineWeb −0.007 ± 0.002, **code +0.0010 ± 0.0019**, labelled clean by the point estimate; the one-sided 95% upper bound is above 0.003.
- **Size:** 0.81M nnz, 1.66 MB packed with the scale derived from the base (7.14 MB with the scale stored; old estimate 2.95 MB), ≈ 53× smaller than LoRA r16 bf16.
- **Val GEN curve, every eval:**

  | step | 100 | 200 | 300 | 400 | 500–800 |
  |---|---|---|---|---|---|
  | val GEN | 27.5% | 47.5% | 96.7% | 100% | 100% |

  No 300→400 decline at all.
- **Caveat (important):** ECTRL and CTRL (same recipe, same seed, 400 steps) diverge from Bop update 2 (cause not isolated: GPU nondeterminism or a step-count-dependent path; the C64 profiler replays the exact ECTRL argv for 20 updates and will show whether same-argv runs match). ECTRL is already at 100% val at step 400, where CTRL is at 93.3%. So 91.3% → 99.5% mixes extra steps with run-to-run variance. The 400-step "decline" may be run variance, not instability.
- One seed. ECTRL seeds 0 and 2 are queued. It becomes the headline point only if at least one more seed reaches ≥ 99%.
- **Wave 2 queued:**
  - GL_ECTRL_s0 and GL_ECTRL_s2;
  - GL_EB4M_s1 (800 steps at 4M);
  - GL_LoRA800_s1: LoRA r16 at 800 steps, current harness, same split. Fairness control for the "training length, not budget" reading.

**GL_Drs_s1** (Bop + one learned fp16 scale per output row, 1M, 400 steps): test GEN 99.5%; code +0.0071 ± 0.0018 (fails the gate); 0.77M nnz, 4.45 MB packed with the base-derivable part re-derived (the learned fp16 scale is kept, 2.75 MB; 9.94 MB scale stored; old estimate 5.55 MB). Confirmed-with-caveats, same review.

**S2 whitened rank-one: solver invalid, no substrate verdict.** The unconstrained dense control also scored 0% GEN, with ΔNLL ≈ 0. Parked (coordinator decision).

**2026-10-13, late: gap-ladder updates (3p×100, seed 1, val GEN).**
- Freeze arms (800 steps): freeze 600 collapses to 48% at step 500 (val ΔNLL +0.086); never freezing reaches about 50% (step 200–300); freeze 200 reaches 100%.
- Post-freeze decline: A (no rehearsal) 97.5% → 91.7% between steps 300 and 400; the original recipe 95.8% → 90.8%. Decline-fix arms (slower Bop EMA after the freeze; rising threshold) are running.
- Drs (ternary slots + one learned fp16 scale per output row, Bop): 94.2% at step 300, still running.

**2026-10-13 ~21:00Z: sequential revoke: KILL for resident training. Mask-only is the default sequential-write protocol from now on.**

Setup: A = person-0 bank (R6_v4_full_p0); B = person-1 bank, 300 steps, absgrad support, layers MoE 12,14,17,20, seed 0. Person-1 in-scope test GEN, n = 200:

| B variant | B with A loaded | B alone (A revoked) | all-split alone |
|---|---|---|---|
| B_res: trained with A resident | 99.0% | **0.0%** | 0/600 |
| B_mask: A's coordinates excluded, A not applied | n/a | 99.5% | 33.2% |
| B_base: trained on the plain base | n/a | 100% | 33.3% |

- Revoke gates are bit-exact. B is active in the "alone" condition (held KL 0.027).
- Dependence (revoked − masked) is −99.5 points, paired fact-clustered 95% CI [−100, −98.5]: pre-registered **KILL**. Capacity cost (masked − base) is −0.5.
- Caveats from the independent result review (confirmed-with-caveats, `reviews/experiments/seqrev_result/`):
  - nnz differs (191k res / 316k mask / 690k base);
  - B_res realized a Bop threshold about 3.6× higher;
  - A's own JSON is missing (in-run A recall 0.98).
- **Rule:** disjoint slots give mechanical revoke; functional independence requires training against the base forward.
- **Consequences:**
  - CC1 and J/F must report AB−A and canary leakage.
  - J/F's joint-gradient arm must report the same dependence number. Its "repair" is expected to create this coupling.

Also from today:
- **fla parity, eval of the same F40 patch:** naive vs fla test GEN identical (100% both). ΔNLL difference: WikiText 0.0009 (within the 0.001 threshold), FineWeb 0.0018 (outside it). Naive stays the backend.
- **S2 rank-one solve:** the first evaluation of the continuous solve scored test GEN 0%. Ternary and polished results are pending, and the cause (a mask with no isolating direction vs a solver issue) is not yet determined.

**2026-10-13: serialized patch bytes (3p×100, seed 1).** Superseded by the 2026-10-14 derived-scale packed sizes above; both packed columns are added here for comparison. The estimated column is the old encoding estimate (positions + signs; LoRA = bf16 parameters). The `final.pt` the harness saves stores unpacked int64 indices plus codes. Per fact = size / 300 facts.

| run | estimated (old) | packed, scale derived | packed, scale stored | saved final.pt | per fact (derived / saved) |
|---|---|---|---|---|---|
| GL_CTRL_s1 (1M budget) | 2.95 MB | 1.66 MB | 7.15 MB | 10.44 MB | 5.5 / 34.8 KB |
| GL_B4M_s1 (4M) | 5.80 MB | 2.66 MB | 8.14 MB | 14.38 MB | 8.9 / 47.9 KB |
| GL_B16M_s1 (16M) | 7.74 MB | 3.31 MB | 8.80 MB | 17.06 MB | 11.0 / 56.9 KB |
| R4_lora16_rand4_s1 (LoRA r16) | 88.1 MB (bf16) | 88.1 MB (bf16) | 88.1 MB (bf16) | 176.7 MB (fp32) | 294 / 589 KB |

**2026-10-13 ~18:10Z: GL_B16M_s1, 16M budget cap: test GEN 99.8% (599/600).**
- Damage: WikiText −0.039 (rehearsal), FineWeb −0.002 ± 0.002, **code +0.0074 ± 0.0015** (fails the 0.003 gate; lower than 4M's +0.011 and LoRA's +0.008).
- Size: realized 2.13M nnz (13% of the cap), 3.31 MB packed with the scale derived from the base (8.80 MB scale stored; old estimate 7.7 MB), ≈ 27× smaller than LoRA r16 bf16.
- One seed, final-step scoring; independent result review confirmed-with-caveats (`reviews/experiments/gl_b16m_test/`).
- Budget ladder (seed 1, test GEN):

  | arm | test GEN | size (packed scale derived / scale stored / saved file) |
  |---|---|---|
  | 1M | 91.3% | 1.66 / 7.15 / 10.44 MB |
  | 4M | 99.2% | 2.66 / 8.14 / 14.38 MB |
  | 16M | 99.8% | 3.31 / 8.80 / 17.06 MB |
  | LoRA r16 | 96.2% | 88 MB bf16 / 88 MB bf16 / 176.7 MB fp32 |

- **Freeze arms (800 steps):**

  | freeze at | val GEN (step 400) | val GEN (step 500) |
  |---|---|---|
  | 200 (control) | 100% | 100% |
  | 600 | 69.2% | |
  | never (800) | 52.5% (step 200) | |

  Keeping the support refreshing longer slows learning; freeze 200 stays the best schedule.
- **Optimizers:** Adam-based writers remain far behind Bop at matched budget and steps.
  - Adam STE ternary: lr 5e-2 → test 72.8% (val 80% at step 400), code +0.012; lr 2e-2 → val 48% at step 300. The lr 5e-3 arm was stopped as dominated.
  - Quinary under Adam: ≤7% val at steps 100–200. Two dominated LRs were stopped; lr 5e-2 is still running.
  - The quinary-precision question stays open until a Bop-native quinary writer exists (design B3, not built).

**2026-10-13 ~16:30Z: gap ladder, first test results.** 3p×100, seed 1, same 600-item test split, all **final-step** scored (`--no-pareto`, including LoRA). Independent result review: confirmed-with-caveats (`reviews/experiments/gl_b4m_test/result_review_grok.md`).

| run | test GEN | WikiText / FineWeb / code ΔNLL (±SE ≈ 0.002) | realized nnz / bytes |
|---|---|---|---|
| **GL_B4M_s1** (4M budget cap) | **99.2%** (595/600) | −0.036 (rehearsal) / −0.005 / **+0.011 FAIL** | 1.60M / 2.66 MB packed (8.14 MB scale stored) |
| GL_CTRL_s1 (1M, 93.2% recipe rerun) | 91.3% | −0.030 / −0.008 / +0.004 | 0.81M / 1.66 MB packed (7.15 MB scale stored) |
| GL_A_s1 (no rehearsal) | 90.3% | +0.008 / **+0.013 / +0.015** | 0.83M / 3.0 MB est. (not packed) |
| LoRA r16 (R4_lora16_rand4_s1, v34) | 96.2% | +0.0003 / +0.004 / **+0.008 FAIL** | 44M params / 88 MB bf16 |

**Caveats:**
- One seed each; B4M seeds 0 and 2 are queued.
- B4M's code damage fails the 0.003 gate, and so does LoRA's (within about 1 SE of each other).
- LoRA ran on harness v34, the ladder on v54.

**Val trajectory (decline check):** B4M val GEN 100% at step 300 → 99.2% at step 400 (little decline). CTRL 91.7% → 93.3%. A 97.5% → 91.7% (decline).

**Scoring rule from now on:** new arms use `--save-at 100,200,300,400` and report both final-step and best-val-step test. GL_CTRLv, GL_Av and the damping arms (GL_Adampg, GL_Adampt, GL_B4Mdampg) are queued.

**Combine-winners plan:** B4M + the best preservation term once A2 (off-topic KL) reports; B4M + damping. Other ladder arms are mid-run:

| arm | val GEN |
|---|---|
| B16M | 100% (step 400) |
| C Adam STE | 22–48% (step 300) |
| Dq quinary | ≤7% (step 100) |
| Drs rowlearn | 36.7% (step 100), val ΔNLL +0.077 |
| E (800 steps) control | 100% (step 400) |
| E600 | 51.7% (step 300) |

**2026-10-13 ~12:40Z: launches.**

*Gap ladder, val GEN at step 300 (test at step 400 pending):*

| arm | val GEN | val ΔNLL |
|---|---|---|
| CTRL | 91.7% | |
| A (no rehearsal) | 97.5% | +0.003 |
| B4M | 100% | −0.042, rehearsal |
| B16M | 78.3% (step 200) | |
| C, Adam STE | 0–17.5% | |

Adam STE is failing.

*fla parity, training (40 facts, 100 steps):* test GEN 97.5% naive vs 100% fla, within noise. fla is only 1.3× faster in training (55 vs 43 s/step). **Decision (coordinator): keep naive_fp32 for all comparisons.** The eval-parity re-eval runs for the record (PG_eval_fla2, C55 cross-eval flag).

*Composition suite (R20):* the full family is running on B (C34 J2 decomposition, pointer, 3 ceilings). Block/unit p2 banks are in their final evals, and J1/J3/J4/J6 plus 6 ceilings follow.

*v57 launch cards (all naive_fp32, 725B):*
- A: SC4 J and F joint continue-training, 150 steps from the R20 full banks, P_iso 0.75. A non-isolated example has the relevant bank plus all other banks; J lets gradients into the distractors, F freezes them; N3 dropout at 0.2. Then S3 teacher screen, S2 whitened rank-one solve (continuous / ternary / polished), PG_eval_fla2.
- C: 725B quiet-unit census (4 domains × 13M tokens, A/B 70/30, min-routed 10k, energy ‖v‖²·E[relu⁴], unconditional and conditional). Then the S1 firing screen plus the small top-8-AUROC vs random-8 value-only arm.
- D: ladder arm A2 (off-topic KL-to-base 0.05 of supervised tokens, own RNG stream, no rehearsal).
- The in-topic panel (DM) is not built yet (needs held-out training text), so J/F report off-topic damage only for now.

*Killed:* R6_v4_addr10 ×3 (no log output for 23 h).

*Design docs* (`experiments/epigenesis/designs/`; all independently reviewed FIX, revisions adopt the edits): SC0 census/reserve, SC1 feasibility allocation, SC2 coverage-first context distillation, SC3 paired cancellation, SC4 joint banks (J/F card), SC5 expand-then-answer, SC6 nursery expert, SCR cheap screens, DM split damage.

**2026-10-13 ~08:20Z digest**

**Gap ladder, interim** (3p×100, seed 1; val GEN at step 100, compared with control GL_CTRL at 35.8%; test at run end):

| arm | val step 100 | val step 200 | val ΔNLL |
|---|---|---|---|
| CTRL (93.2% recipe, current harness) | 35.8% | | +0.028 |
| A no rehearsal | 40.0% | 50.8% | +0.005 / +0.000 |
| B 4M budget | 60.8% | | −0.022 |
| **B 16M budget** | **72.5%** | | −0.018 |
| C Adam STE lr 5e-2 | 9.2% | | −0.012 |
| C Adam STE lr 5e-3 | patch still empty at step 75 (masters below threshold) | | |
| E control (800 steps, freeze 200) | 27.5% | | +0.012 |

- Budget is the strongest early lever. The original recipe's 300-step consolidation jump (≈50% → 95%) means step-100 numbers are not final; test GEN decides.
- Adam STE looks clearly worse than Bop so far.

**Cheap-write support-turnover result, seed 1:** R25_t10_b80k_turnover_s1 test GEN 85% (s0 100%). The locked-support screen gave 0–5%; damage clean (+0.0037 / +0.0018 / +0.0004).

**2×2 factorial (1 person × 100 facts, matched Adam recipe, val-selected LR).**
- dense_tern_s0: test GEN 100% but **damage WikiText +2.04, FineWeb +1.69, code +0.76**. Dense ternary low-rank (row-α scaled ternary LoRA) destroys the base at every tried LR; s2 shows the same (+1.75 at step 100).
- dense_fp_s0: test GEN 17.5% (LR grid 1e-4..1e-3 too small for 200 steps); damage +0.028 / +0.014 / +0.013.
- Sparse cells are still running.
- Interim reading: on the forced-zero support, ternary writes are safe; ternary low-rank is not. The fp low-rank cell is LR-limited at 200 steps, so that comparison is not yet decisive.

**R20 bank set (relation world) damage at FINAL:**
- full p0: WikiText +0.0032, FineWeb +0.0007, code +0.0022;
- unit p0: +0.0025 / +0.0003 / +0.0029;
- block p0: **+0.0066 / +0.0061 / +0.0036 (fails the 0.003 gate)**.

p1 banks are done at step 300 (95.8–100% val); p2 are at steps 200–300. The decomposition suite (C34 J1/J2, selectors, INH/CONJ/SLOT, pointer with C40 fix, C29, 9 ceilings; 15 jobs, v55) is staged and launches when the p2 banks finish.

**Sequential revoke:** B_res val 100% (step 300), B_mask val 99.2% (step 300). The base control and the test-split paired CI follow.

**Ops.**
- All five boxes now run `fsaq4.sh` plus `lock_janitor3.sh`: race-free claims, owner-checked locks, and a heartbeat hang rule that applies only to v55 jobs.
- Legacy fsaq2 job subshells finish and clean up on their own.

**Finisher side tasks (coordinator):**
- `needle_32k_100` with the v3 retrieval (fixed-shape greedy decoding, error guard) on export 7401: **33% exact**, one shape, 3.05 s per 32k forward on a 5090. That is the 8k-stage reference for the S3 gate (0.8).
- D GPU 1 is released back to the FSA queue.

**2026-10-13 ~05:40Z digest** (725B base unless noted)

**Cheap writes: decided. The failure was support lock-in, not budget, capacity or latency.** 10 facts, 40 steps, layers MoE 12,14,17,20, test GEN:

| run | budget | support policy | test GEN | WikiText / FineWeb / code ΔNLL |
|---|---|---|---|---|
| R14_t10_B/D (screen) | 80k / 40k | freeze 15 / 10, keep-live | 0–5% (5 runs) | clean |
| R23_t10_b120000_s0 | 120k | screen policy | 75% | +0.0013 / −0.0004 / +0.0014 |
| **R25_t10_b80k_turnover_s0** | **80k** | **no freeze within 40 steps, refresh every 10, no keep-live** | **100%** | +0.0016 / +0.0006 / +0.0005 |
| R21_t10_long (200 steps) | 300k | turnover | 100% | +0.0040 / +0.0022 / +0.0004 |

- At the screen's own 80k budget, letting the support turn over takes 10 facts from 0% to 100% in 40 steps (one seed; s1 queued).
- The 1-fact positive control passes (100%).
- Write-recipe rule from here: **do not lock the support early for small writes.**
- T-stat Bop and unit bursts stay NO-GO; the diagnosis that motivated them is refuted.

**3p×100 FSA 93.2% vs LoRA 96.2% (test, seed 1): independent result review, confirmed-with-caveats** (`reviews/experiments/p3x100_test/result_review_grok.md`).
- Both runs used the identical base and test items: the BASE objects match, 600 items.
- Recipe differences: 400 vs 700 steps, rehearsal 0.2 vs 0, local λ 10 vs 100, harness v32 vs v34.
- FSA val GEN peaked at 95.8% at step 300 and fell to 90.8% at the scored step 400. Late-training decline is a live suspect; arm E's trajectories will show it.

**Gap-closing ladder (3p×100, seed 1; owner priority; bar = LoRA 96.2%).**
- Running: GL_CTRL (93.2% recipe on the current harness), A (no rehearsal), B4M, B16M, C (Adam STE at lr 5e-3 / 2e-2 / 5e-2), E (800 steps, freeze 200/600/800).
- Built and reviewed, then queued: D (C47).
  - quinary {−2..2}·α/2 under Adam, against the matched C arms;
  - ternary plus a learned fp16 per-row scale under Bop (GL_Drs).
- Design review and implementation check (both GO): `reviews/experiments/gap_ladder/`, `reviews/experiments/c47_c48_check/`.
- Results are due in about 8–10 h; E takes about 16 h.
- **fla EDA parity gate** (owner-approved, never silent, `--eda-backend` C45) is queued on A:
  - F40_bopck_s0 load-eval under both backends;
  - a 40-fact, 100-step training under both.

  It passes if GEN counts are identical and |ΔNLL difference| ≤ 0.001 on WikiText and FineWeb. If it passes, new batches run under fla, expected about 3–5× faster.

**Decisive-suite banks (R20, relation world).**
- p0 banks are done at step 300: full 99.2%, block 98.3%, unit 97.5% val GEN.
- p1 banks are at step 250 (97.5–100%).
- p2 banks run on C; full p2 shows val ΔNLL +0.014 at step 200, to watch.
- The decomposition suite launches when all 9 banks exist.

**Sequential revoke (R22_seqrev, v52, absgrad support).** B with A resident reached val 100% at step 300; the mask-only control is at 80.8% (step 100); the base control is queued.

**2×2 factorial (R23_fac, in its LR search).** The dense-ternary cell reaches val 100% but with **val ΔNLL +1.76**, wrecking the base. Dense ternary LoRA is not a viable format at these LRs; it awaits the selected-LR final.

**R16 attribute-group bank p0_g0:** test GEN 8.3% of the 300-fact test, i.e. its 25 in-scope facts. Damage clean (WikiText +0.0005, FineWeb −0.0013, code +0.0010).

**Preempted for the ladder:** R17 ×5 (superseded by ladder arm E) and R19 ×2.
- R19 at step 50: late bank MoE 26–29 reached 80% val GEN vs 99.2% mid.
- Deferred with placeholder logs: R12_*m13 (late-address arms) and R8.

**Ops.**
- Two suspected hangs were diagnosed as slow, silent naive-EDA phases, not deadlocks (`reviews/experiments/hang_2026-10-13/NOTES.md`). The R13 jobs keep running.
- C44 (resumable checkpoints + heartbeats), C45 (explicit EDA backend), C46/C48 (heartbeat hang janitor + race-free runner fsaq4) are reviewed GO and deployed as v55.
- The runner switch-over is pending a quiet moment.
- Runners now export PYTHONFAULTHANDLER=1, so a stuck job can be stack-dumped with `kill -ABRT`.

**Coordinator side tasks (finisher; D GPU 1).**
- Long-context NLL table posted (in summary: 8–16k −0.282 nats and 16–32k −0.352 vs 725B; most gain early in the 16k stage).
- needle_32k_100 built and verified.
- The `retrieval()` and `heldout()` doc_context recompile crash was found and fixed by fixed-shape windows. The S2 long_val heldout passed with the fix (98 windows, no crash); the live file crashed.
- The S3 v3 package (guarded retrieval, `guarded_heldout`) is in `reviews/finisher_needle/`.

**2026-10-13 ~02:50Z: LoRA wins 3p×100 test; gap-closing ladder is now top priority (owner ruling).**
- **Spread-layer LoRA r16 (R4_lora16_rand4_s1, layers 3,16,22,29, seed 1): test GEN 96.2%** (QUANT bf16 line 96.3%) against **FSA Bop 93.2%** (R4_rand4_rep_s1, same seed, same test split, 725B). FSA loses this task by 3 points on one seed each.
- Owner: "if LoRA wins anything ever it's because this approach is lacking something still." 96.2% is the bar.
- **Gap-closing ladder** (3p×100, seed 1, one variable per arm vs the 93.2 recipe; each arm reports test GEN, three damage panels and bytes written). It runs ahead of R17/R19/R12/R8 and of anything outside the decisive composition suite.

  | arm | variable |
  |---|---|
  | A | no rehearsal |
  | B | budget 4M and 16M slots |
  | C | latent-continuous slots with straight-through ternary, Adam (Bop control) |
  | D | quinary {−2..2} slots, and ternary plus a learned fp16 per-row scale, compared at equal bytes |
  | E | freeze at 600 and never |

  - The winners combine into one recipe, run on seeds 0 and 1.
  - **Coverage audit (F), from code:** FSA and LoRA attach to exactly the same matrices, the routed-expert `up_proj`/`down_proj` in `attach()`. Neither touches attention; shared experts only with `--shared`. So coverage does not explain the gap.
  - The design review (confounds, LR selection, byte accounting, what must be built) is running; launches follow it.

**2026-10-13 ~00:10Z digest** (725B base unless noted; val = validation only)

**New results**

- **Cheap writes: the 40-step failure is a support/budget problem, not capacity or a broken writer.**
  - 1-fact positive control (R21_t1_posctrl): **test GEN 100%** (100% val from step 5). WikiText +0.0003; code **+0.0062 (fails the 0.003 code gate)**.
  - 10 facts, 300k budget, freeze 200, refresh every 10 (R21_t10_long): val GEN 5% at step 10 → 70% at step 15 → **100% from step 20** (still 100% at step 70), val ΔNLL ≤ 0.
  - Budget sweep (R23_t10_b120000_s0, 40 steps): 55% val GEN at step 20.
  - Log diagnosis (`reviews/novel_ideas_2026-10-12/ideas3_diagnosis.md`): the failed 40k/80k screens filled their whole support at update 2. Their surviving codes were all born at step 2. The "evidence-latency" premise of the T-stat idea is therefore refuted for this screen.
- **Per-person bank status (R20, relation world, layers MoE 12,14,17,20):**
  - full p0: val 100% at step 100;
  - block p0: 99.2% at step 100;
  - unit p0: 97.5% at step 50.
  - The p1 banks are training; the p2 banks run on C.
- **Attribute-group bank R16_grp_p0_g0:** val 98% at step 150 (25 facts, 1/12 unit partition).
- **Sequential revoke, B trained with A resident (R22_seqrev_B_res):** val 100% at step 50; A loaded under v52. Masked and base controls queued behind it.
- **Spread-layer LoRA seed 1 (R4_lora16_rand4_s1):** val 97.5% at step 700. Its test phase has run for about 4 h with no FINAL yet.
- **Finisher-6445-prior banks (R7_fin_v4_*):** val 99–100% but damage is far worse than on 725B. part_p1: WikiText +0.034, FineWeb **+0.048**, code **+0.026**, all failing the gate. full_p2 has val ΔNLL +0.0003, so the damage differs across banks. Open question: why the same recipe damages the finisher base more. Do not use finisher-base banks until that is understood.
- **Per-person bank test GEN ≈ 33% each** (R6_v4_part_s1_p0/p1 0.332/0.333, R6_r4_full_p0 0.327). This is expected: one person's bank against a 3-person test set, so about ⅓ of items are in scope.

**Reviews and decisions**

- **muse_ideas3 §1–4 (coordinator request), built as C41/C42/C43 and design-reviewed by two independent reviewers** (`reviews/experiments/c41_c42_ideas3/`):
  - **T-stat Bop: NO-GO now.** Its premise is refuted by the logs; the birth-matched control is degenerate under support saturation.
  - **Unit bursts: NO-GO, dropped.** They only birth vacant supported slots, which are saturated.
  - **LLR selector + hierarchy fixes: FIX → GO.** C43 applied the fixes:
    - the rescue kill rule now uses the positive-query rescue rate;
    - the false-fire verdict uses the exact 95% upper bound, with 10k nulls per split where available;
    - simulated-100 is diagnostic only.
  - **Queued:** C42_dump3 (Q, 3 R6_v4 banks; the 12-bank dump waits for R16) and R25_t10_b80k_turnover_s0/s1 (C). R25 separates "more slots" from "support turnover": 80k budget under the 300k run's policy, with `--write-diagnostics`.
  - §5 (gate-trained pointers) is still gated on the R20 `oracle_entity` chain ceiling.
- **Harness v54** (C39–C43) is on all boxes. Legacy behaviour is byte-identical, verified by digest tests.

**Coordinator side tasks (finisher; box D GPU 1 reserved)**

- Long-context NLL by position (dense, own rope_theta, 40 docs × 32k) is running on 725B / 6445 / 7401 / 8260. Script: `experiments/long_context/long_nll.py`.
- The needle set `needle_32k_100.jsonl` was built on ny and the tokenizer verified. A `retrieval()` dry run on export 7401 follows the NLL job.

**Process**

- Everything runs slowly: about 60–90 s per step plus about 1 h of setup, so a 300-step bank takes about 7 h. GPU utilization is about 20% fleet-wide.

**2026-10-12 ~21:50Z: steward takeover, queue cleanup.**
- **Queues re-prioritized to the refocused plan.** Out-of-scope pending items are commented `#OOS` and have placeholder logs (benchmarks R9/R15, EP-1 extras, masked LS R5, unit-k sweep R8, finisher-6445 extras R7_fin, R10 separability reruns, hygiene R4_hyg).
- **Killed (exact PID, out of scope):** R5_ms_rand4c (masked LS, dead recipe), R9_bench_sub_s0 and R9_bench_P4every4 (benchmarks), R8_u100_k2 (k-sweep, step 50 after 7.7 h). Also killed: R12_lateaddr_p2, the wrong-layer (MoE L26/28) late-address arm that the stale queue order had launched.
- **Sequential revoke fixed and requeued (Q, top).** The first launch died because resident training correctly rejects `netcost`. The independent design review (`reviews/experiments/c38_decisive/design_review_muse.md`) gave FIX-BEFORE-TRUSTING. Requeued as three matched arms, all with `absgrad` support and `--bootstrap 1000`:
  - R22_seqrev_B_res: B trained with A resident, then A revoked;
  - R22_seqrev_B_mask: A's coordinates excluded, zero delta (capacity control);
  - R22_seqrev_B_base: B trained on the base.

  Decision rule: PASS when the bit-exact gate passes and the paired 95% CI of (revoked − masked) covers 0.
- **2×2 factorial requeued as R23_fac_* (B, C, D), FIX-BEFORE-TRUSTING items applied.**
  - The old queue trained 40 facts (no `--facts-per-person`); the new runs train 100.
  - The analysis now accepts equal-size per-cell LR grids (C39), with three points per cell: ternary 0.003/0.01/0.03, sparse fp 3e-4/1e-3/3e-3, dense fp 1e-4/3e-4/1e-3.
  - Seeds 0–2.
  - R22_fac_lora_reh and R22_fac_fsa_lam100 are held until the winners are known.
  - The sparse cells use the matched Adam recipe, not production Bop, so the headline must say so. The gap metric is labelled "TF→GEN gap".
- **LoRA-bank composition control queued (Q):** R22_lora_p0/p1/p2 are r16 per-person banks on the R6_v4 world with the canonical LoRA recipe (lr 3e-4, λ 100), followed by branch_eval `--modes oracle router sum --merge sum_delta`. R22_fsa_banks_sumdelta is the same evaluation on the R6_v4 FSA banks (matched arithmetic).
- **Decisive-suite design review (R20 bank set): core suite GO** (`reviews/experiments/c36_decisive_suite/design_review_muse.md`). The review found two issues and C40 fixes both (harness v53, on all boxes):
  - `two_hop.py` rebuilt a legacy world for relation-world banks (3 gold facts wrong);
  - selector-gated ("served") negative queries did not exist; only forced-bank leakage did.

  Pass/kill rules for each arm are pre-registered in that review (§4). The R20 jobs J1–J6 plus 9 single-bank ceilings launch on box A once all 9 banks exist. The R20 p2 banks moved to box C to train in parallel.
- **Older R13 outputs (legacy world, pre-C36 tree):**
  - Kept: single-value GEN, selector false-activation, damage.
  - Relabelled: two-fact numbers become presence only; the "pronoun" and "two-hop" QA panels become single-mention and indirect reference.
  - Discarded as two-hop evidence.
- **R1_lora16_big_rand4 killed (hung).** It printed nothing for 4 h after its step-600 TRAJ (normal cadence is 25 min). The matched-seed LoRA test number comes from R4_lora16_rand4_s1, now in its final test phase. Seed 0 is requeued as R24_lora16_rand4_s0 (D).
- **One-fact positive control (R21_t1_posctrl): PASS.** It reaches 100% val GEN at step 5 and 100% test GEN at step 40, with WikiText ΔNLL +0.0003. The 40-step write pipeline works, so the 10-fact failure is not a broken writer.
- **Coordinator request: muse_ideas3 §1–5 queued as refinements of existing slots.**
  - C41 (write recipe): T-stat Bop and cross-projection unit bursts with their controls. It first checks the latency diagnosis from churn logs.
  - C42 (selector and hierarchy): per-bank LLR + margin null; LLR beam, sibling-conditional children and priced flat rescue.
  - Both are being built now; independent design review follows before any launch.
  - §5 (gate-trained pointers) is gated on the text oracle. The R20 J1/J2 `forward_chain` `oracle_entity` ceiling is that oracle; if it does not rescue hop 2, §5 stops.
- **Lock janitor redeployed on all 5 boxes** (none was running; D GPU3 had a stale lock). It skips GPUs 4–7 on C and D and only removes empty lock dirs.

**2026-10-12 ~21:30Z: handover snapshot.** The fleet keeps running from per-box queues (`fsaq*.sh`).

**Top results** (GEN headline; 725B base; seeds stated; test unless marked val):

| # | arm | task | result | damage | seeds |
|---|---|---|---|---|---|
| 1 | Global Bop on layers 3,16,22,29 (1M budget, freeze at step 200, WikiText rehearsal 0.2) | 3p×100 | **test GEN 93.2%**; val 95.0 / 95.8 | WikiText −0.025, FineWeb −0.005, code +0.009 | s1 test; s0/s1 val |
| 2 | Spread-layer LoRA r16 (same layers) | 3p×100 | val 95.0 (s300), 94.2 (s400–600); test pending | WikiText −0.005 val | s0 |
| 3 | Per-person banks, layers 12,14,17,20, full mask and ⅓ block partition | 3p×100, one person each | val 96.7–100 individual; composed recall pending (branch_eval R13 running) | FineWeb ≤ +0.003, code ≤ +0.004 | s0 (block s1 running) |
| 4 | EP-1 sleep gating, 20% units | rules | dev 99.25 / test 100 (57/57) | +0.0025 | s0 |
| 5 | EP-1 sleep gating, 10% | rules | dev 95.25 / 94.25, test 98.2 / 98.2 | +0.001 / +0.002 | s0, s1 |
| 6 | Bop + λ10 + 300k budget, layers 14,15 | 40 facts | 100 / 100 (LoRA r16: 93.8 / 92.5 / 90.0) | all panels clean | s0, s1 |
| 7 | Bop 1M recipe, layers 14,15 | 2p×50 | 97.0 / 99.5 / 99.5 (LoRA 94.5 / 95.5 / 95.0) | clean | s0–2 |
| 8 | Realistic QA on the 40-fact patches, FSA vs LoRA | held-out | unseen 90 / 88 vs 65 / 62; reverse lookup 51 / 44 vs 22.5 / 30; pronoun 75 vs 47.5 | clean | 2 vs 2 seeds (pronoun 1 vs 1) |
| 9 | 4 layers, no rehearsal | 40 facts | 98.75 | FineWeb +0.0018, code +0.0014 (clean) | s0 |
| 10 | Per-person banks on finisher-6445-prior | 3p×100, one person | val 95.8–100 | FineWeb up to +0.013 (worse than 725B) | s0 |

**Pending, deeply queued:**
- composed recall F/G, router C, indexer I, addressing ceiling AC, DG, INH/CONJ/SLOT, indexed composition, two-hop pointer (R13, on Q);
- whole-unit and relation-world bank set (R20) with the decisive suite;
- 12 attribute-group banks for hierarchical selection (R16);
- freeze 60 / 200 / never on test (R17);
- sequential revoke (R22_seqrev);
- 2×2 factorial with LR search (R22_fac);
- late (MoE 26–29) vs mid bank (R19);
- unit-k sweep and init trap (R8);
- late-address banks (R12);
- finisher-7401 oracle evidence and separability (R18);
- LoRA r16 spread-layer test number.

**2026-10-12 ~20:35Z**

**3p×100 TEST result (725B).** Global FSA Bop on layers 3,16,22,29 (1M budget, freeze at step 200, WikiText rehearsal 0.2), seed 1:
- **test GEN 93.2%** (TF 93.3%);
- damage: WikiText −0.025 (rehearsal), FineWeb −0.005, code +0.0094 (acceptable band).

For comparison, the earlier 2-layer LoRA test was 65.7%. The matched spread-layer LoRA (R1_lora16_big_rand4) has val 94.2% and its test number is still pending. The FSA vs LoRA verdict waits on it.

**EP-1 rules.** Sleep gating with 20% units reaches **99.25% dev / 100% test** at ΔNLL +0.0025. That beats LoRA (91–95% dev) and has about 1/100 the damage of ungated ternary (single seed; seed 1 queued).

**Joint masked least-squares gate.** Continuous solve 0.8% (3,16,22,29) and 0.0% (14,15 both projections) val GEN, so this recipe fails. Positive control pending.

**Ten-fact cheap-Bop screen (A–D × 3 seeds).** All arms 0–5% test GEN at 40 steps, so it fails its ≥90% kill rule. The zero-patch control passes. Cheap writes need more steps or budget than 40 steps at 40k–80k slots. A one-fact positive control is still owed.

**Infrastructure and process.**
- Benchmark zero-patch control passes. The ARC gain is real for both FSA and LoRA, likely a Q/A-format effect.
- A lock-janitor race caused one OOM (R17_frzNever_s0). The janitor is fixed (requires no compute apps, lock older than 30 min and 3 idle checks), and the run is requeued.
- Finisher post-replay export **step-00007401** (BACKUP_RECEIPT) is on all 5 boxes. Oracle-evidence ceiling and separability runs on it are queued or running.
- New coherent bank set (R20: full / block / unit × 3 people, `--relation-world`, harness v51) is queued for the decisive evaluation suite.

**Reviews.**
- The C34 decisive eval suite got an independent design review: FIX-BEFORE-TRUSTING. The strict two-fact scorer accepted negated and listed answers. Fixed in v51 (C36).
- C35 recipe features (sequential revoke, 2×2 factorial, LoRA banks, composition curriculum, synthetic people): all FIX-BEFORE-TRUSTING; fixes are being implemented (C38).
- C37 novel-idea experiments: under review.

**Owner refocus (20:50Z).** Work is restricted to the documented hierarchical FSA plan in ARCHITECTURE.md. Dropped:
- both "novel ideas" waves, including the constructive mechanisms (the Muse coding run was stopped);
- the exploration pilots (VSA, Kuramoto, etc.; R13_explore removed from the queue);
- extra EP-1 sweeps.

Kept: per-entity whole-unit banks with composed recall and decomposition, the selector (positive/inhibitory keys, null, gating), hierarchical selection, graph pointers and the reverse index, addressing (late reads, DG, conjunctive gates, slots), the write recipe (freeze plus the 10-fact screen), and the decisive checks.

**Superseded (20:35Z):** constructive wave (owner request). Muse is building standalone mechanisms:
- self-evidence readout;
- corrective overlay banks with revoke-by-dependency;
- shield units;
- write-time inference closure;
- dream textbook.

The second tier (authority, preference banks, active teaching, value codec, evidence banks, meta-banks, algorithm banks, state banks, surprise-only writes) follows.

**2026-10-12 ~18:55Z** (725B base unless noted)

**Oracle-evidence ceiling.** The base model has the relevant fact lines in context, with no patch (GEN on 600 test items):

| condition | GEN |
|---|---|
| needed facts only | 73.0% |
| needed facts + 20 distractors | 88.7% |
| conflicting dated old/new facts, "use the latest" | **2.8%** |
| two-person composition (n = 60) | 8.3% |

The 725B base cannot resolve dated conflicts or compose from context. Per-person weight banks (96–100% individual GEN) exceed in-context retrieval on this model. finisher-6445-prior is queued.

**Per-person banks (final, layers 12,14,17,20, 3p×100).** Individual val GEN: full mask 96.7/99.2/99.2%, block-partition 98.3/98.3/100%.

| bank set | WikiText ΔNLL | FineWeb ΔNLL | code ΔNLL | label |
|---|---|---|---|---|
| full mask | +0.002 to +0.005 | +0.0005 to +0.003 | −0.0003 to +0.0019 | clean to borderline |
| block partition | n/a | +0.002 to +0.0025 | +0.004 | acceptable, pending bench |
| spread layers 3,16,22,29, block-partition p0 | n/a | +0.017 | +0.015 | **fail** |

Mid layers are much cleaner than spread placements that include layer 3.

**Composed recall (F/G), router C, indexer I, addressing ceiling, DG, INH/CONJ/SLOT, indexed composition (reverse index, intersection, hypothesis beams), two-hop pointer with cleanup, exploration pilots.**
- Queued on box Q against the six v4 banks (harness v48, all C32 fixes, `--positive-control` run first).
- Launches as soon as the last block-partition bank file is copied over.
- Hierarchical selection (person → attribute group, H vs FLAT): 12 attribute-group banks are queued (3 people × 4 groups, unit partition).

**Whole-unit partition rerun (R11, 2 seeds) and seed-1 block banks:** queued or running.

**Global FSA vs spread-layer LoRA, 3p×100 on 3,16,22,29.**
- Bop seed 1: val 95.8% at step 300, 90.8% at step 400 (end). Damage FineWeb −0.005, code +0.0094 (acceptable band, pending bench).
- LoRA: 94.2% at steps 400–600.
- Test numbers are printing at run end.

**Benchmark ARC anomaly persists after the C28 fix.**
- F40 FSA patch: ARC-Easy 0.56 → 0.635.
- trace-top4 patch: 0.56 → 0.67.
- Base is always exactly 0.56, so a cached or stale base value is suspected.
- A zero-patch control (v48 `--zero-patch-control`) and a no-fast-moe/no-cache rerun are queued. Benchmark verdicts stay void.

**10-fact cheap-Bop screen.** Arms B and D are near 0% val GEN at steps 20–35 of 40. The 40k–80k budgets on 4 layers at slot-frac 0.5 look too slow for 40 steps. Final numbers are pending. A likely verdict: cheap writes need more steps or a larger birth rate.

**GPU allocation.** GPUs 4–7 on boxes C and D were freed for the oscillation agent at 18:25Z. Long jobs were killed and requeued on our own GPUs; the two block-partition banks finishing on C GPUs 4 and 7 release their locks automatically.

**2026-10-12 ~17:00Z** (725B base unless noted)

- **The step-300 consolidation jump replicates on a second seed.**
  - Bop on 3,16,22,29 with a 1M budget, freeze at step 200: 50.8% → 95.8% val GEN between steps 200 and 300 (seed 1). The original seed 0 went 43.3% → 95.0%.
  - Matched LoRA r16 on the same layers: 95.0% (step 300), 94.2% (400), 94.2% (500).
  - Global FSA and spread-layer LoRA are therefore tied on validation. Test numbers come at run end.
- **Realistic QA, FSA vs LoRA** (2-layer 40-fact patches, held-out free generation; result review: confirmed-with-caveats):

  | probe | FSA (2 seeds) | LoRA r16 (2 seeds) | n |
  |---|---|---|---|
  | unseen phrasing | 90.0 / 88.3 | 65.0 / 61.7 | 120 |
  | two-hop | 50.8 / 44.2 | 22.5 / 30.0 | 120 |
  | **pronoun** (new; seed 0 only) | **75.0** | **47.5** | 40 |

  Damage on FineWeb and code is clean for both.
- **Benchmark harness bug fixed.** Base and patched scoring used different grouped-GEMM dispatch; fixed in harness v45. ARC/HellaSwag/MBPP numbers from before v45 are void. Reruns are queued (R9b_*).
- **Rehearsal attribution, 4-layer 40-fact runs:**
  - WikiText rehearsal: WikiText −0.040, FineWeb −0.008, code +0.0076;
  - FineWeb rehearsal: FineWeb −0.0084, code −0.0013;
  - no rehearsal: FineWeb +0.0018, code +0.0014 (clean).

  The ΔNLL "gain" follows the rehearsal corpus.
- **Process.**
  - Retroactive result reviews by an independent reviewer: 4 headline claims confirmed-with-caveats. The banks review is pending.
  - Design review of the queued branching/addressing arms: B, D and E-unit pass. C/F/G/I/AC/DG/INH/CONJ/SLOT/pointer, the late address arm, the unit k-sweep and the masked LS gate are FIX-BEFORE-TRUSTING. Fixes are being implemented (C31 done, C32 running).
  - The 10-fact cheap-Bop screen had a refresh-schedule bug (freeze 10 and 15 identical). It was relaunched with the corrected schedule as 12 runs, A–D × 3 seeds.
- **Bases.** finisher-6445-prior runs are in progress (banks, 40-fact pair, oracle evidence, separability).
- **GPU allocation.** GPUs 4–7 on boxes C and D are ceded to the oscillation agent: 32 GPUs for this track.

**2026-10-12 14:40Z.** Every result below is on the **725B** base unless it is labelled "finisher-6445-prior". That is the pre-crash finisher export copied from ny. It will be re-pinned to the first post-replay finisher export once that has a BACKUP_RECEIPT.

**Branching track: individual banks on 3p×100**

Validation GEN, seed 0; each bank holds one person's 100 facts.

| arm | layers | step | p0 | p1 | p2 |
|---|---|---|---|---|---|
| B full mask | 12,14,17,20 | 200 | 98.3 | 98.3 | 100 (s100) |
| E block-partition ⅓ | 12,14,17,20 | 150 | 95.8 | 100 | 97.5 |
| B full mask | 3,16,22,29 | 100 | 96.7 | 100 | 88.3 (s50) |
| E block-partition ⅓ | 3,16,22,29 | 150 | 98.3 | 90.0 (s50) | 81.7 (s50) |

- **Composed recall (F/G), router (C) and indexer (I): pending.** These need all three banks finished (step 300, about 1.5 h), then branch_eval.
- **Partition bug.** The architecture review found that the block partition lets banks share hidden units. The E/F/G numbers above are therefore labelled "block-partition".
- **Unit-level partition rerun is queued** (R11_v4_upart, 2 seeds; whole hidden units per bank).

**The step-300 consolidation jump replicates (16:15Z).** R4_rand4_rep_s1 (Bop on 3,16,22,29, 1M budget, freeze at step 200) went from 50.8% val GEN at step 200 to **95.8% at step 300**, with a different seed from the original (43.3% → 95.0%).

**Global single patch on 3,16,22,29: Bop vs LoRA.** Both reach 95% val GEN at step 300, so they are tied on validation.
- The Bop replicate R4_rand4_rep_s0 matches the original exactly at step 100 (49.17% both, deterministic).
- Seed 1 is at 50.8% at step 200.
- LoRA is at 94.2% at step 400.
- Test numbers are pending.

**Realistic QA on the 2-layer 40-fact patches** (unseen phrasing / two-hop):

| patch | unseen | two-hop | seeds |
|---|---|---|---|
| FSA Bop | 88–90% | 44–51% | s0, s1 |
| LoRA r16 | 65% | 22.5% | s0 |

- More LoRA seeds and the pronoun probe are queued (R9_bench_*, with ARC-Easy, HellaSwag and MBPP-MC).
- The lenient two-fact scorer is added in v42. The strict scorer gave ≈0 for both methods.

**Separability (725B).**
- The original probe trained only on cloze templates. The fixed probe (v41) adds question-form training templates, a name-token baseline and a bounded novelty score. Its reruns on 725B and finisher-6445-prior are queued (R10_sep_*).
- From the original run: person-from-name is 89% at L4; pronoun → person is near chance mid-network and 68–77% at L26–28, so coreference binds late.
- AUROC sign: a code review found the AUROC sign correct. The low values came from ridge extrapolation on general text, which made unrelated inputs look confident; the new bounded centroid-cosine score is primary.
- The joint 300-class probe is data-limited (about 3 prompts per class). Attribute-given-person will be reported separately.
- Next: address reads at L26/28, via indexer arm I at layers 26/28, and address units at L26/28 with values at 12–20.

**Queued screens:**
- the 10-fact cheap-Bop screen, arms A–D × 2 seeds, 40 steps, all 12 formats;
- the oracle-evidence ceiling on 725B and finisher-6445-prior;
- the 30-item context-teacher gate;
- the joint masked least-squares gate;
- unit-constrained k-sweep (k = 1–32) and the address-init trap.

**Leaderboard label check.** The 95% Bop row is correct: Cb_bop_big_rand4 is a Bop log (fsa_forced/bop, births lines). The matched LoRA (R1_lora16_big_rand4) independently also reached 95%. Both rows are now stated.

**2026-10-12 12:25Z — branching track, first evals (725B base, seed 0; validation GEN, step 50 of 300).**

Each row trains one person's 100 facts alone (3p×100 world, Bop, 1M budget, no rehearsal).

| arm | layers | person 0 | person 1 | person 2 |
|---|---|---|---|---|
| B: full forced mask | 12,14,17,20 | **100** | **97.5** | (pending) |
| E: disjoint ⅓ block partition | 12,14,17,20 | **99.2** | **97.5** | **96.7** |
| E: disjoint ⅓ partition | 3,16,22,29 | 94.2 | (pending) | (pending) |

- **Individual recall is essentially solved at step 50.** It holds even when each person gets only a third of the forced slots. The capacity per bank is not the limit.
- The decisive numbers are still to come from branch_eval, once all three patches per arm finish:
  - composed recall with all three disjoint banks resident in one forward (F);
  - router accuracy (C);
  - two-person composition (G);
  - confidently-wrong-without-fallback.
- **Matched LoRA r16 on the same spread layers 3,16,22,29 reached 95.0% val GEN at step 300** (R1_lora16_big_rand4). The earlier 3p×100 "gap" was mostly placement on two adjacent layers, for both methods. FSA has no demonstrated advantage over LoRA here yet; test numbers decide.
- **Global single-patch comparison, validation GEN:**
  - Bop on 3,16,22,29 at step 100: 2M budget 74.2%; seed 1 40.8%; no-rehearsal replicate 51.7%;
  - the 2M run then dropped to 63.3% at step 200, during the eviction phase;
  - matched LoRA r16 on 3,16,22,29: 70.8% at step 200;
  - the 95% step-300 jump awaits replication at step 300.
- The finisher export step-00006445 (ny, BACKUP_RECEIPT verified) is copied and SHA256-verified on all 5 boxes. Mask comparison, oracle-evidence and separability runs are queued on both bases.

**2026-10-12 08:25Z.** Headline metric is GEN (free generation, exact answer plus clean stop). TF is secondary everywhere.

**Locked-panel re-evaluation of the 4-layer 40-fact patches trained with WikiText rehearsal:**

| patch | WikiText | FineWeb | code |
|---|---|---|---|
| trace-top4 | −0.040 | −0.008 ± 0.002 | **+0.0076 ± 0.0017** |
| every-4th | −0.029 | −0.0076 ± 0.0021 | **+0.0065 ± 0.0020** |
| random-4 | −0.024 | −0.0056 ± 0.0015 | +0.0001 ± 0.0016 |

The 2-layer headline patch is −0.0007 ± 0.0018 on FineWeb. The negative WikiText ΔNLL is rehearsal-driven adaptation toward prose, and two of the three patches show 3–4.5 SE damage on code. Verdict: those negative ΔNLL numbers are not preservation. All new runs use no rehearsal and report all three panels.

**Other updates:**
- **Masked Gauss–Newton ternary optimizer (MLS):** after a discrete step-size fix it is mechanically working. It accepts about 20k code changes per iteration at about 0.1–0.2 nats of real loss decrease. It costs about 108 s/iteration and reached 5% val GEN at step 20 on 40 facts, behind Bop. A 150-step screen is queued.
- **Fast-MoE exact dispatch:** bit-identical logits, 1.5× on eval and 1.6× on greedy generation, but only 1.16× on training steps.
- **Base-evaluation cache:** saves roughly 10–40 min per run.
- **Queued:** Hebbian/delta-rule ternary write (with a Bop warm start), Cartridges-style context-teacher distillation (teacher gate check first), LoRA fade-out homotopy, and tuned LoRA baselines (lr 1e-3, λ 10) on the same 4 layers.

**2026-10-12 05:45Z.**
- **Rehearsal hides damage on multi-layer patches.**
  - With WikiText rehearsal, the 4-layer causal-trace 40-fact patch scores WikiText ΔNLL −0.040 ± 0.002, but only −0.008 ± 0.002 on held-out FineWeb.
  - The same placement without rehearsal (Cb_bop40_tracetop4_noreh, test 100/100) gives **+0.0079 ± 0.0023 on WikiText**, about 3.4 SE of damage.
  - The 2-layer headline patch is −0.0007 ± 0.0018 on FineWeb.
  - New rule: fact runs use no WikiText rehearsal, and damage is reported on FineWeb and code panels as well as WikiText.
- **Bop on 3p×100 is unstable.** Validation GEN peaks and then falls:
  - 2-layer: 37.5% at step 500, then 22.5% at step 600;
  - 4 random layers: 50% at step 100, then 43% at step 200.
  - LoRA r16 2-layer seed 1 is at 61.7% at step 800.
  - This matches the brainstorm diagnosis of destructive budget eviction and low exposure per fact.
- Wave R2 is queued: layers 12,14,17,20, 2.3M budget, binding-loss × keep-live factorial, continuous box ladder, matched LoRA.
- Harness v25–v27 adds the binding loss, keep-live, stratified audits, curriculum, epoch scoring, distillation, saturation sampling and chaos options.

**2026-10-12 03:55Z.** Search loop restarted. Persistent per-box queue runners (`fsaq2.sh`) were deployed so freed GPUs pick up
queued work. Held-out non-WikiText ΔNLL panels (FineWeb-Edu, Python code; 90k tokens each) were built to test the
negative ΔNLL on multi-layer runs. Wave R1 is running: 3p×100 placement (mid-4, mid-8, random-4 with a 2M budget, τ 0.5,
no rehearsal, sleeping gating), a 4-layer LoRA r16 control, and panel re-evaluations. AI-assisted brainstorm round 1 is
in progress.

## Suspicious: benchmark gain from a 40-fact patch (15:20Z)

The trace-top4 40-fact patch (layers 6,7,10,11, WikiText rehearsal 0.2) gives these benchmark results:

| benchmark | base | patched | change |
|---|---|---|---|
| ARC-Easy | 0.56 | **0.67** | +0.11 ± 0.027 (26 gained, 4 lost) |
| HellaSwag | 0.470 | 0.465 | unchanged |
| MBPP-MC | 0.450 | 0.455 | unchanged |

Code ΔNLL is +0.0076 ± 0.0017. The damage gate labels the code panel **fail**, because the benchmarks are not all unchanged.

An 11-point ARC gain from writing 40 fictional facts is implausible as knowledge. The likely cause is rehearsal-driven prose adaptation shifting length-normalised log-likelihoods. A scorer artifact is also possible. **Update 19:20Z: the scorer is consistent.** The v48 zero-patch control passes on every scoring path, including the benchmarks: an all-zero patch reproduces base exactly. The benchmark gains appear for both FSA and LoRA:

| patch | ARC-Easy, base → patched | HellaSwag | MBPP-MC |
|---|---|---|---|
| 2-layer FSA 40-fact | 0.56 → 0.635 | flat | flat |
| trace-top4 FSA | 0.56 → 0.67 | flat | flat |
| LoRA r16 40-fact | 0.56 → 0.67 | flat | −0.03 |

Most likely this is a **format effect**: training on many "Q: … A: <answer>." items raises the likelihood the model assigns to short answer options in ARC's Question/Answer format. It is not knowledge. Whether "unchanged within noise" counts a benchmark *gain* as a gate failure is an owner call. Under the current literal rule, every patch "fails" the acceptable band because ARC moved.

**Superseded (15:45Z):** "this is almost certainly a scorer bug, not an effect." The LoRA r16 40-fact patch on layers 14,15 gives the *identical* ARC-Easy shift, 0.56 → 0.67 (27 gained, 5 lost). Unrelated patches producing the same patched accuracy means base and patched are being scored differently. A bug hunt is running (C28). All --bench numbers are void until it is fixed.

Not to be cited until checked against:
- the matched no-rehearsal and FineWeb-rehearsal patches (R2_f40_r4_noreh / _fwreh, finishing);
- the 2-layer patches;
- a review of the ARC scorer.

## Damage gate (owner ruling 2026-10-12 12:45Z)

Damage is measured as per-panel ΔNLL on WikiText, FineWeb and code. Each panel gets a label:
- **clean:** ΔNLL ≤ +0.003;
- **acceptable:** ΔNLL ≤ +0.01, and ARC-Easy, HellaSwag and MBPP-MC (about 200 items each) are unchanged within noise with all banks resident;
- **fail:** ΔNLL > +0.01.

Re-ranking the 40-fact 4-layer patches trained with WikiText rehearsal. The benchmark evaluation is pending (harness v40), so "acceptable" is provisional.

| patch | test GEN | WikiText | FineWeb | code | gate |
|---|---|---|---|---|---|
| random-4 (3,16,22,29) | 98.75 | −0.024 | −0.006 | +0.0001 | clean |
| every-4th (8 layers) | 97.5 | −0.029 | −0.008 | +0.0065 | acceptable? (needs bench) |
| trace-top4 (6,7,10,11) | 100 / 98.75 | −0.040 | −0.008 | +0.0076 | acceptable? (needs bench) |
| 2-layer headline F40_bopck_s0/s1 | 100 / 100 | −0.0006 / +0.0019 | −0.0007 / −0.0007 | +0.0003 / −0.0001 | clean |

## Hygiene panel for the 2-layer forced-mask winners (2026-10-12 ~11:10Z; load-patch re-evaluation, 725B base)

| patch | test GEN | WikiText ΔNLL | FineWeb ΔNLL | code ΔNLL | unseen-phrasing QA | indirect QA | two-fact both | route Jaccard (mean / downstream layers) |
|---|---|---|---|---|---|---|---|---|
| F40_bopck_s0 | 100 | −0.0006 ± .0020 | −0.0007 ± .0018 | +0.0003 ± .0017 | 90.0 | 50.8 | 1.7 | 0.986 / 0.97–0.98 (FineWeb) |
| F40_bopck_s1 | 100 | +0.0019 ± .0014 | −0.0007 ± .0016 | −0.0001 ± .0015 | 88.3 | 44.2 | 0.0 | ≈0.99 |
| C_bopck_sub_s0 (2p×50) | 97.0 | −0.0015 | −0.0020 ± .0019 | −0.0001 ± .0018 | (pending) | | | |
| C_bopck_sub_s1 (2p×50) | 99.5 | +0.0000 | −0.0011 ± .0020 | −0.0005 ± .0016 | (pending) | | | |

- Damage on every locked panel is within 1 SE of zero.
- Realistic-QA generalisation is partial:
  - unseen question phrasings: 88–90%;
  - indirect reference ("the person whose X is Y — what is her Z?"): 44–51%;
  - two-fact questions answered completely: ≈0%. Scoring may be strict here; check the samples.
- **Matched LoRA r16 (F40_lora16_s0) on the same probes:** unseen phrasing 65.0%, two-hop 22.5%, two-fact 0%, FineWeb +0.0002 ± .0018, code +0.0009 ± .0014.
  - FSA leads on realistic QA by about 24 points (unseen) and about 22–28 points (two-hop).
  - That is 2 FSA seeds against 1 LoRA seed, n = 120 each.
  - The probe scores free generation on phrasings never trained, so this is the first non-TF advantage evidence.
  - More LoRA seeds are needed before claiming it.
- The downstream routing Jaccard of 0.97–0.98 per layer clears the ≥0.87 gate.

## EP-1 rules: sleeping-unit gating sweep (forced ternary, layers 14,15; dev n=400, test n=57; 725B)

| arm | seed | dev acc (final / best) | select ΔNLL | test acc |
|---|---|---|---|---|
| **sleep gating, 20% units, β 1.0** | 0 | **99.25 / 99.25** | +0.0025 | **100** (57/57) |
| sleep gating, 10%, β 1.0 | 0 | 95.25 / 95.25 | +0.0011 | 98.2 |
| sleep gating, 10%, β 1.0 | 1 | 94.25 / 96.0 | +0.0020 | 98.2 |
| sleep gating, 5%, β 1.0 | 0 | 95.75 / 95.75 | +0.0025 | 98.2 |
| random gating, 10%, β 1.0 | 0 / 1 | 95.0 / 93.5 | +0.0036 / +0.0088 | 96.5 / 96.5 |
| sleep / random gating, 10%, β 0.5 | 0 | 83.75 / 82.75 | −0.0018 / +0.0022 | 84.2 / 82.5 |
| LoRA ref (same layers) | 0 / 1 | 94.75 / 91.25 | +0.0009 / +0.0009 | 98.2 / 96.5 |
| ungated ternary | 0 | 99.0 | **+0.234** | 100 |

Reading of the table:
- Sleep gating with 20% of units matches ungated accuracy (99%) at about 1/100 of its damage.
- It also beats LoRA dev on both LoRA seeds.
- Sleep beats random gating on damage in both seeds: +0.002 vs +0.004–0.009.
- β 0.5 hurts accuracy.

Caveats:
- 20% is one seed.
- Test n=57 cannot separate arms above about 96%.
- The 20% seed-1 run is queued next.

## Layer indexing (authoritative; 21:45Z)

**Convention.** Every claim must say which index it uses.
- "MoE L*i*" = the *i*-th MoE ordinal (0–31). This is what the harness `--layers` / `--address-layers` take.
- "block B*j*" = the full-block index (0–63) of `model.layers`. This is what the separability probe, indexer `--indexer-layer`, DG `--dg-layer` and `--h-layer*` take.
- Mapping: MoE L*i* = block B(2*i*+1).

**Corrections applied.**
- The coreference finding is at **blocks B26–B28 ≈ MoE L12–L13**.
- The late-address arms R12_* had used `--address-layers 26,28` (MoE L26/L28 = blocks B53/B57). That is the wrong layer for this finding, so they were killed and requeued as R12_*m13 with address units at **MoE L13 (block B27)** and values at MoE L12,14,17,20.
- R19_late_moe26_p0 deliberately tests a truly late bank at MoE L26–29 (blocks B53–B59) and is labelled as such.

## Layer indexing note (19:40Z)

The model's MoE modules are `layers.1, layers.3, …, layers.63`, so MoE ordinal *i* is full block *2i+1*.
- Separability-probe and indexer layers are **full-block** indices.
- `--layers` for patches uses **MoE ordinals**.

Consequences:
- "Pronoun coreference binds at L26–28" refers to full blocks 26–28, which is about MoE 12–13. That is already inside the v4 banks' layers (MoE 12,14,17,20, i.e. full blocks 25,29,35,41).
- A literal "MoE 26–29" bank (full blocks 53–59) is queued as the truly-late comparison (R19_late_moe26_p0 vs R19_mid_p0, person 0, same facts, with name/pronoun/indirect probes).

## Glossary (frozen 2026-10-12)

- **two-hop**: a question that requires a hop through a stored relation fact. Example: "<A>'s sister is <B>" → "What is <A>'s sister's <attr>?". The older probe "the person whose X is Y — what is her Z?" is an **indirect / reverse-lookup** probe, not two-hop. Its earlier "two-hop" label is renamed.
- **bank**: a separately trained, separately revocable patch for one scope (person, or person × attribute group). It is written into the forced-zero mask (or a partition of it) and gated at full strength, never blended.
- **within noise**: |Δ| < 2 paired SE, or a 95% bootstrap CI that covers 0. With n=40–120 items, 1–2 item differences are within noise.

## Joint masked least-squares gate (C14 masked_solve, 3p×100, layers 3,16,22,29)

The continuous masked solve scores **0.8% val GEN** (R5_ms_rand4c, before ternarization). That is below the 45% gate.

Per the design review, this is a verdict on *this solver recipe*, not a capacity or geometry impossibility. Its targets are residual interventions mapped to experts by a prescribed allocation. The positive control (recovering a known writable update) has not been run yet. The 14,15 placement and positive control are pending.

## Review status (owner requirement, 16:05Z)

Every arm gets three checks:
- a design review before launch (a fresh read-only code/design review);
- a result review before any verdict, by a reviewer who did not write the code;
- a positive control before any kill.

Reviews are stored under `reviews/experiments/<arm>/`.

| claim / arm | design-reviewed | result-reviewed |
|---|---|---|
| 40 facts: FSA 100% vs LoRA 89–96% | retro (n/a) | **confirmed-with-caveats** (all numbers verified from FINAL lines; LoRA r1/r4 single seed; n=80) |
| 2p×50: FSA 97–99.5% vs LoRA 94.5–95.5% | retro (n/a) | **confirmed-with-caveats** (unequal recipes: FSA had rehearsal 0.2 + λ10, LoRA λ100; s1_lora16 lacks a RESULT line) |
| Realistic-QA FSA-vs-LoRA gap | retro (n/a) | **confirmed-with-caveats** (items not independent: 3 phrasings/fact; 2 seeds each) |
| Per-person banks (individual recall) | design_batch1: B/E PASS | **confirmed-with-caveats** (val phrasing transfer at ceiling, 1 seed; not test or composed recall) |
| EP-1 sleeping-unit gating 95% | retro (n/a) | **confirmed-with-caveats** (numbers verified; sleep ≈ random ≈ LoRA within noise, so "sleep matters" is NOT established; test n=57; seed-1 pending) |
| Branching B (oracle), D (merge), E-unit (oracle partitioned) | **PASS** (design_batch1) | pending |
| C router, F all-resident, G composition, I indexer, AC, DG, INH/CONJ/SLOT, pointer, late address, unit k-sweep/init-trap, masked LS | **FIX-BEFORE-TRUSTING** (design_batch1); fixes being implemented (C31/C32); positive controls required | results from pre-fix code are labelled "pre-fix" |
| 10-fact cheap-Bop screen | **FIX** (freeze 10 and 15 refreshed identically; γ schedule not implemented); runs pulled before launch, relaunch after C31 | — |

## Exploration budget (about 20% of GPUs; low-prior, high-upside pilots)

Each pilot runs a positive control first and has a pre-stated kill criterion. The code is being written (C30, `exploration.py`).

| pilot | prior | kill criterion | control | result |
|---|---|---|---|---|
| VSA binding keys | 15% | not better than dense-cosine indexer (arm I) at equal bytes | pending | pending |
| Phase / resonance addressing | 10% | ≤ arm I | pending | pending |
| Kuramoto bank selection | 5% | not ≥5 pts over softmax-argmax, or >2× false activations | pending | pending |
| Interference retrieval (signed superposition) | 5% | composed recall < 50% of oracle | pending | pending |
| 4th tensor mode (CP-factored address) | 10% | ≤ arm I | pending | pending |
| Expand-and-sparsify (DG) address | (queued, branch_eval DG) | per C26 | | |

## 40 facts, one person (test, 80 items)

| arm | TF | GEN | ΔNLL ± SE | slots | seeds | run |
|---|---|---|---|---|---|---|
| Bop + λ10 local penalty + 300k netcost budget + rehearsal 0.2, L14,15 | 100 / 100 | 100 / 100 | −0.0006 ± .0020 / +0.0019 ± .0014 | 175k / 188k | s0, s1 | F40_bopck_s0/s1 (A) |
| same, random 25% of reserve | 100 | 97.5 | −0.0000 ± .0020 | 252k | s0 | F40_bop_rand25_s0 (A) |
| same, causal-trace top-4 layers 6,7,10,11, slot-frac .5 | 100 | 98.75 | −0.0401 ± .0023 (suspect) | 225k | s0 | P4_bop40_trace_top4_sf (Q) |
| same, random 4 layers 3,16,22,29 | 98.75 | 98.75 | −0.0244 ± .0017 (suspect) | 235k | s0 | P4_bop40_rand4_sf (Q) |
| single layer L12 | 98.75 | 98.75 | +0.0014 ± .0020 | 225k | s0 | P_bop_40_L12 (B) |
| sleeping gating 10% + Bop | 97.5 | 96.25 | −0.0015 ± .0019 | 208k | s0 | GA_bop40_sleep10_g (A) |
| LoRA r16 + penalty 100 | 96.25 / 93.75 / 88.75 | 93.75 / 92.5 / 90.0 | +0.0014 / +0.0015 / +0.0012 | 22M params | s0,s1,s2 | F40_lora16_s* (A) |
| LoRA r4 / r1 | 96.25 / 93.75 | 96.25 / 93.75 | ~0 | | s1 | F40_lora4_s1, F40_lora1_s1 |

## 2 people × 50 conflicting facts (test, 200 items)

| arm | TF | GEN | ΔNLL ± SE | slots | seeds | run |
|---|---|---|---|---|---|---|
| Bop recipe, 1M budget, L14,15 | 96.5 / 99.5 / 99.5 | 97.0 / 99.5 / 99.5 | −0.0015 / +0.0000 / −0.0026 | 733k–757k | s0,s1,s2 | C_bopck_sub_s0/s1 (C), s2 (Q) |
| same, random 25% of reserve | 98.0 | 97.5 | +0.0010 ± .0021 | 761k | s0 | C_bop_sub_rand25 (C) |
| LoRA r16 | 96.0 / 94.5 / 95.5 | 94.5 / 95.5 / 95.0 | +0.0010 / +0.0021 / +0.0028 | 22M | s0,s1,s2 | s1_lora16, D_lora16_sub_s1/s2 |

## 3 people × 100 conflicting facts (main gap; validation GEN, 120 items; test pending)

Validation phrasings are harder than test for this task (an earlier LoRA run had 46.7% val against 65.7% test GEN), so these numbers are not comparable with test.

| arm | val GEN by step | seed | run |
|---|---|---|---|
| **Bop 1M budget, random 4 layers 3,16,22,29, slot-frac .5, WikiText rehearsal .2, τ .7** | 49.2 (100), 43.3 (200), **95.0 (300)** | s0 | Cb_bop_big_rand4 (killed in the 07:37Z prune) |
| same, seed 1 | 51.7 (200) | s1 | Bb_bop_big_rand4_s1 (killed) |
| same, τ .5 | 45.8 (100) | s0 | R1_big_rand4_tau05 |
| Bop every 4th of 8 layers, slot-frac .25 | 25.0 (100), 36.7 (200) | s0, s1 | Db/Bb_bop_big_every4 (killed) |
| Bop recipe L14,15 (2 layers) | best 37.5 (500), then 20.0 (700) / best 27.5 (700) | s0 / s1 | D_bopck_big_s0/s1 (killed) |
| LoRA r16 L14,15, λ100 | 57.5 (1000) / 62.5 (900) | s0 / s1 | D_lora16_big_s0/s1 |
| LoRA r16 on 3,16,22,29 | 5.8 (100) | s0 | R1_lora16_big_rand4 |

**Attribution check (14:00Z), after a coordinator query.** Both runs below reached 95.0% val GEN at step 300 on layers 3,16,22,29:
- **Bop**: Cb_bop_big_rand4. Its log shows `--method fsa_forced --optim bop` (queue qCb), 13 "bop births" lines and TRAJ 0.50 → 0.433 → 0.95 at steps 100/200/300.
- **LoRA r16**: R1_lora16_big_rand4, TRAJ 0.058 → 0.708 → 0.95 at the same steps.

So the 95% Bop row is correct, and the matched LoRA reaches the same number. The comparison is a tie on validation until test numbers land.

**Suspicious jump, under verification.** The 4-layer run went from 43% to 95% between steps 200 and 300. Step 200 is the default budget-freeze step: support refresh, and with it eviction, stops there. Births flattened (12.0M → 12.9M) and train loss fell to 0.10–0.14. This fits the destructive-eviction diagnosis. It is one seed, validation only, and trained with WikiText rehearsal, so damage must be checked on FineWeb and code.

Queued at the head of the queues (08:35Z):
- an exact v24 replicate (R4_rand4_rep_s0);
- seed 1 with locked panels (R4_rand4_rep_s1);
- the same without rehearsal (R4_rand4_rep_noreh);
- freeze at step 60 (R4_rand4_freeze60).

## Dead or failing ideas (forced mask)

- STE ternary with Adam masters: 2.5–8.5% on 2p×50. Fold-aware STE: ties LoRA on recall at ~10× damage.
- One-shot projection of a trained LoRA into forced slots (least squares, box, ternary, GPTQ, multiscale): ≤19% even
  unquantized, versus LoRA 79% (2p×50).
- Patch scale 0.25α / 0.125α: 31% / 4% (40 facts). 0.5α works (97.5%).
- Sleeping / specific / random gating on 3p×100: ≤8% val at step 200.

## Pruned runs (killed by exact PID, partial logs kept)

2026-10-12 ~04:00Z, dead ends:
- C_bop_sub_scale025 (5–7% val at s150; 0.25α already failed on 40 facts).
- C_t2_sub_s05 (0% at s150).

2026-10-12 ~06:40Z, rejected placement or gating on 3p×100, all ≤18% val:
- Cb_bop_big_tracetop4_s1, Db_bop_big_tracetop4 (trace placement);
- Cb_bop_big_gate_rand, Bb_bop_big_gate_spec, Bb_bop_big_gate_spec_s1 (gating).

2026-10-12 07:45Z, owner/coordinator prune:
- **Fact gating (rejected by synthesis):** GC_bop_sub_rand10_g, GC_bop_sub_spec10_g, GD_bop_big_spec10_g, GA_bop40_spec10_g_s1, R1_big_rand4_gate_sleep.
- **STE:** C_ste_sub_s05.
- **Causal-trace placement:** P4_bop40_trace_top4_s1_sf, Bb_bop_sub_tracetop4.
- **Superseded old-recipe 3p×100 runs** (1M budget, WikiText rehearsal; validation peaked then decayed): D_bopck_big_rand25, Cb_bop_big_rand4, Bb_bop_big_rand4_s1, Bb_bop_big_every4_s1, Db_bop_big_every4, D_bopck_big_s0 and D_bopck_big_s1.
  - D_bopck_big_s0/s1 last validation: 22.5%/24.2% at s600, after peaks of 37.5%/24.2%.
- **Extra seeds or arms on solved tasks:** C_bopck_sub_s3, C4_bop_sub_all_f003.
- **Never launched** (marked skipped): Bb_bop_sub_gate_spec, Bb_bop_sub_gate_rand, Bb_lora16_big_tracetop4.

Standing rule from now on: an arm is killed at its first eval checkpoint if it is more than 15 points below the best comparable arm at the same step.

2026-10-12 ~09:00–09:40Z, pruned: reallocated to the branching track (owner decision).
- Global 3p×100 knob sweeps and algorithm screens:
  - R1_big_rand4_tau05, R1_big_mid4/_s1, R1_big_mid8, R1_big_rand4_k2m_s1;
  - R2_v4_bind/keep/bindkeep(_s1)/base, R2_v4_box/_down;
  - R3_v4_sat/distill/epoch/fisher/cur/chaos/fade/meta/pair/up2;
  - R4_rand4_freeze60.
- Masked Gauss–Newton screens: R3_v4_mls, R3_L1415_mls, R4_f40_mls.
- Smokes: R2_smoke_wake.
- Solved-task extras: Bb_bop_sub_rand4, Bb_bop_sub_gate_spec.
- LoRA controls for the deprecated 12,14,17,20 placement: R2_lora16_v4, R3_lora16_v4_s1, R4_lora16_v4_lr1e3/lam10. A duplicate seed, R4_lora16_rand4_s0v34.
- Removed from queues before launch: the Hebbian arms (R4_hebb40, R4_hebbB) and LoRA-fade (R3_L1415_fade).

2026-10-12 13:35Z, pruned: superseded by the branching track or by spread-layer LoRA (coordinator status review).
- **D_lora16_big_s0/s1** (2-layer LoRA, 3p×100, killed at step 1100+).
  - Last val GEN: s0 best 60.0% at s1100; s1 best 64.2% at s1100.
  - Superseded by LoRA r16 on 3,16,22,29: 95.0% at s300.
- **Global-patch budget and ablation variants:** R1_big_rand4_k2m (74.2% at s100, 63.3% at s200), R1_big_rand4_noreh (48.3%, 37.5%), R4_rand4_keeplive, R4_rand4_freeze100 (43.3% at s100), R4_rand4_rep_noreh (51.7% at s100), R4_rand4_rep_s2.
  - Only R4_rand4_rep_s0/s1 remain as the global-FSA vs spread-LoRA replicate.
- **R4_ctx_teacher_check** (Cartridges teacher gate).
  - Stuck for 5 h in batch-1 greedy generation over 1,200 items with fact-sheet contexts and no KV cache.
  - No keep rate was produced. It will be re-run on a 60-item subsample.
