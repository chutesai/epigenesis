# EP-1 pre-registration (written before any GPU run; 2026-10-09)

Hypothesis under test: a frozen pair8-ternary genome (lambda 8B) can accumulate a user's *concepts* (policies applied
to new cases), vocabulary and style across 5 sessions through consolidation into free ternary slots, without
accumulating one-offs and without measurable general damage, at parity with a regularized LoRA at the same placement.

## Fixed before the run

- Corpus: `make_user.py --seed 20261009` (40 policies, 100 recurrent facts in 3/5 sessions each, 100 one-off lures
  20/session, closed status vocabulary + answer shape, 5 chunked sessions). Dev/test probes built before synthesis,
  disjoint entity pools; test probes are touched once per session for the retention curve and never used for
  selection, netcost gain or synthesis.
- Self-study: generated once per session by the genome with the source chunk in its window; frozen; shared by every
  arm. Verifier and gates as in DESIGN.md §4. Null-KL threshold = 95th percentile of the length-matched unrelated
  context's per-token KL.
- Arms: genome, icl (ceiling, not shippable), tmid, fp4mid, loramid. KV-prefix skipped (EDA recurrence makes a
  prefix-only-at-MSA baseline broken; stated, not tested).
- Loss: 0.4 gated consolidation KL / 0.4 general-text KL to the session-start individual / 0.2 replay KL
  (`--mix balanced`); T=1; top-32 teacher approximation; answer-only; assistant turns masked; replay bounded (256),
  never lure text, wiped on revoke.
- Writer: masters rehydrated from committed slots and Adam moments reset at every session boundary; router, router
  biases, alpha and the indexer frozen; tmid budget ceiling 2M slots (ceiling, not target); block scales init 0.25
  alpha; wd on labile slots only; promotion = survived one prune cycle + session dev concept gain.
- Per-session rollback if the selection-pool dNLL (16x512, disjoint from the locked panel) exceeds +0.005 vs session
  start. Checkpoint selection = Pareto on (dev concept, selection dNLL) with dev lure > genome+5 as a hard discard.
- Steps 48 per session (160-token micro-batches on the naive EDA backend; see DESIGN.md section 6); 8 self-study items +
  2 anchor windows (96 tokens) + 4 replay items per step; teacher context = the source episode; self-study answers <= 32 tokens; one seed (pilot).
- Smoke before the run: sessions 1-2, 10 steps per arm (FSA lr x5), revoke after every session; not scored.
- Identifiability gate: icl dev concept must exceed genome dev concept by >= 10 points, else the corpus is declared
  too weak and no arm is scored.

## Success criteria (all must hold for the thesis to pass; measured on TEST probes after session 5)

1. Concept accuracy >= genome + 10 points on the 100 four-way test items.
2. Lure recall <= genome + 5 points (hard; a method that gains concepts by memorizing lures fails).
3. Locked wikitext 40x512 mean dNLL <= +0.01 nats/token AND the paired 95% interval excludes +0.05.
4. Retention: test concept accuracy after session 5 >= 80% of the best earlier session's test concept accuracy;
   lure recall does not ratchet upward across sessions (ratchet = monotone non-decreasing with a net increase).
5. Revoke passes: after zeroing slots + scales + masters + moments + replay + ledger, every patched weight tensor is
   bitwise the genome's and pair8-legal, logits on 32 fixed prompts equal the genome's within bf16 noise (max |diff| <= 1e-2; exact equality reported), and concept
   / lure metrics return to the genome's within noise.
6. Greedy agreement with the genome on the locked windows is reported every session; no criterion in EP-1.

## Decision versus LoRA (pre-registered; not promotable after the fact)

- "Match LoRA" (operational justification): tmid or fp4mid test concept within 3 points of loramid, dNLL no worse
  than loramid by more than 0.005, lure constraint held, revoke passes.
- "Beat LoRA" (quality claim): +5 concept points over loramid at dNLL no worse than loramid.
- Reported either way: distillation gap (icl concept minus arm concept), slots/bytes, participation ratio.

## Not claimed

Lifelong capacity, selective per-session unlearning, privacy from weights, or any quality advantage not meeting the
"beat LoRA" bar. One seed is a pilot; three seeds are needed to claim.

## SECONDARY exploratory comparison: owner-approved utility-release ablation

This additive, default-off comparison is outside the primary success criteria and the pre-registered LoRA decision.
`tmid_util01` and `tmid_util03` follow `tmid` in every respect except manual labile master decay is off (wd = 0)
and a release runs at the start of sessions K >= 2. Each calibrates its own netcost lambda in session 1.

After load/rehydration and before training caches and the router superset, accumulate acquisition+replay KL gradients
through the enabled patch's normal STE on up to `--util-items 64` uniformly sampled own-data items without replacement
(seed + session, independent sampler). Replay uses stored teacher records; all accepted current self-study items use
the pre-release session-start individual's `cache_items` teacher records and gates. Loss is the item mean of
`epi_common.topk_kl`, with the arm's acquisition/replay mix coefficients and acquisition gates; zero-gated items
contribute zero, and there is no general-text term. Score frozen slots too, before gradient protection.
Utility is `|M * M.grad|` for committed ternary M (scale cancels in `|w * E[dL/dw]|`). Filled slots are `delta != 0`.
For L14 and L15 separately, release filled slots with utility strictly below rho times that layer's filled-slot
mean: rho = 0.1 or 0.3, derived from the arm name. Zero master, support and frozen bits; assert no optimizer state
exists yet; clear gradients and CUDA cache. Release counts are logged and recorded per session. Training rollback
returns to the post-release start, with all subsequent training/evaluation/revoke rules as in `tmid`.

Hypothesis: utility release keeps recurring facts while freeing reserve, with lure recall no worse than `tmid`
and final panel dNLL no worse than `tmid + 0.005`. Report recurring-fact test verbatim EM per session and
session-5/best-earlier ratio, lure EM trajectory, final panel dNLL with paired 95% CI, and final slot count versus
`tmid`. An absent final session or zero earlier maximum makes the ratio undefined. These observations do not
modify the primary thesis pass/fail criteria.

Run after the main EP-1 arms on whatever GPUs free up, with a separate ablation smoke first. Select explicitly
with `ARMS="tmid_util01 tmid_util03"`; the runner's default remains `"tmid fp4mid loramid"`. Resume into the main
run directory after it completes to preserve the decay comparator and include all arms in the summary.

## Post-hoc added arm (2026-10-10, owner-approved, after the main run started)

`tbop` is an additive, default-off exploratory arm, **not part of the original
thesis decision**. It ports the end-to-end harness's best fact-injection v11c
configuration (recovered from result JSON; box run: `e2e_fsa_v11.py`): fsa_free,
ternary, layers 14/15 up+down, all experts, Bop gamma .05, row patch scale without
block scales, wd 0, netcost cost_rel 1, ceiling 300000 including frozen slots,
local output penalty lambda 10 with frozen-base denominator, answer-only,
rehearsal fraction .2. Tau = .7 times the first answer-gradient RMS divided by .8.

For EP-1, fact loss is acquisition/replay gated teacher KL with balanced .4:.2
renormalized, then weighted .8; general-text KL is replaced by .2 patched next-token
CE on two anchor windows. Tau and netcost lambda calibrate only in session 1 and
persist unchanged across sessions, including rollback. Refresh every 10 reference
steps (EP-1 steps 1,11,... through index 200) uses CPU top-K; session boundaries
rehydrate committed ternary M and reset EMA. Tmid frozen/promotion, superset,
rollback, selection, replay, evaluation and revoke rules apply; revoke clears EMA.
No Adam, decay, annealing or block scales. DESIGN.md documents every EP-1 mapping.

Compare versus tmid and loramid on the same pre-registered metrics, criteria and
LoRA comparison thresholds above. `summary.json` labels tbop post-hoc; any reported
criteria pass or LoRA match/beat is exploratory and cannot alter the original
thesis decision. Run explicitly with `run_arm_chain.sh tbop RUN_DIR [--smoke]`;
existing default arms remain unchanged.

## Amendment v2 (2026-10-10, written after the v1 results; v1 verdicts above stand as recorded)

v1 failed the thesis for every arm including LoRA (results and diagnosis: EP1_V1_FINDINGS.md). The failure is
attributed to the consolidation signal (self-study prompts named the rule; ~1 item per policy per session; a
single-episode teacher that is right on only 50% of probe-format cases). v2 changes the signal only; arms, metrics,
success criteria, rollback (+0.005), lure discard and checkpoint selection are unchanged.

Changes:
1. Teacher context = the full rule table of the case's category (all four policies with rule and rationale text),
   instead of the single source episode.
2. Training items in probe format without the codename (half ticket family, half Q/A family), 32 new cases per
   policy per appearance, from the self-study name pool; deduplication only of identical cases. Targets are the
   true resolution / action tokens; the loss is forward KL to the teacher's distribution on those tokens (top-32 +
   tail bucket), no null-KL gate for these items. Items are kept only if the teacher, scored by cloze over the four
   options exactly as the ICL arm, picks the correct option.
3. 150 steps per session, evaluation every 25 steps.
4. tmid at lr 0.02 and 0.05 (two arms); other arms' hyperparameters unchanged.

Gates, in order (each must pass before the next step runs; failures are reported, not tuned around):
- G1 teacher: on session-1 policies, the category-table teacher keeps >= 0.80 of generated items.
- G2 pilot: session 1 only, loramid and tmid (lr 0.02 and 0.05); at least one arm reaches >= 0.70 dev concept
  accuracy on the trained policies at some evaluation under the +0.005 dNLL limit.
- Only then: the 5-session v2 run with arms genome, ICL, tmid (both lr), fp4mid, loramid, tbop, and tmid_util01 /
  tmid_util03 (secondary, labelled), and revoke-scope v2.

Revoke-scope v2: policies in the v2 form above; facts by supervised cross-entropy in the 12-format augmentation.
Gate: condition A reaches >= 0.80 generated recall on A items (steps raised until it does, recorded); then the seven
conditions are rerun unchanged.

## Amendment v2.1 (2026-10-10, after gate G1 failed; v1 and v2 records above stand)

G1 result: keep rate 0.30 (bar 0.80). v1's ICL ceiling (1.00) and identifiability gate are recorded as INVALID: the
ICL context contained only the correct policy's episode, which leaks the answer.

Changes:
1. Corpus: coarse rules. Each policy's condition is one categorical attribute stated literally on the ticket; the four
   policies of a category partition that attribute's values; no numeric ranges. Seed unchanged; probes rebuilt
   before any synthesis, disjoint pools as before.
2. ICL, self-study and every teacher see the full category rule table, never the correct policy's episode alone. The
   ICL ceiling and the identifiability gate (ICL - genome >= 10 points) are re-measured on this context.
3. Gates unchanged: G1 teacher keep >= 0.80; G2 pilot >= 0.70 dev concept under the +0.005 dNLL limit.
4. Fallback, fixed now: if G1 fails on the coarse corpus, EP-1 switches to the external ground-truth supervised
   target (d2 form: answer-only cross-entropy on the true resolution/action, 6 formats) for every arm. That is
   supervised continual learning with revocable slots, not self-distillation, and is reported as such.

### v2.1 addendum: MDL / compression score (secondary metric + secondary arm; primary verdicts unchanged)

**Pre-run change (2026-10-11, before the MDL arm run):** reserved wording, bounded sampling/windows,
whole-group deletion and explicit replacement provenance below supersede the earlier MDL release specification.

Metric, per arm and session, in bits:
- held-out user text H_k: EVERY episode kind (policy intro/recur, fact chat, lure ticket and chit-chat) re-rendered
  deterministically using reserved templates in make_user.py, a held-out customer name pool and fresh ticket ids (same policies, facts and lures; never used for training, self-study or probes).
- bits_saved = sum over H_1..H_k tokens of (NLL_genome - NLL_patched) / ln 2.
- general_cost = mean locked-panel dNLL (nats/token) / ln 2 x N_ref, with N_ref = 1,000,000 tokens (stated reference
  amount of general text); reported with its paired 95% interval propagated.
- patch_bits = the same accounting as the slot byte count (budgeted ternary/fp4: nnz x (log2(reserve) + value bits) +
  32 bits per patch-only scale; unbudgeted fp4: reserve x 4; LoRA: factor parameters x 32).
- net MDL gain = bits_saved - general_cost - patch_bits. Reported for every arm; no pass/fail criterion.

Secondary arm tmid_mdl: tmid plus, at the start of every session k >= 2, MDL-based release:
- Per earlier session, draw a fixed seeded sample of 16 held-out texts (seed + session namespace), reused for
  all baselines and leave-outs. This release sample does not replace the all-text secondary metric H_k.
- Use BOS-prefixed windows of at most min(2048, --micro-tokens) tokens, resetting context at each window;
  every document token is a target exactly once. NLL sums exclude BOS and right padding. Utility gradients and
  leave-out scoring use the identical windows. Code inspection: MeshModel/EDA support optional cu_seqlens/doc_context boundaries, but the active
  epi_gpu.logits_of(model, x) path supplies only token IDs and no such boundaries. Choose the safe row-based
  packing path: one document window per row,
  right-pad rows, with total padded tokens per forward <= --micro-tokens. No cross-document attention is allowed.
- Rank groups by held-out gradient utility and evaluate at most --mdl-max-groups=256 exact leave-outs. Release
  scoring uses effective nonzero coordinates; deletion clears the ENTIRE (layer, expert, projection, 64-row
  block): all values including shadows, support, frozen flags, netcost scores and patch-only scales.
  Empty groups use absent scales (-inf log scales in memory); serialization omits their numeric scale payload
  and slot accounting charges zero scale bits. Initialize to alpha_row * .25 only when future support admission
  reoccupies the block. A proposal STE uses the init scale for gradients without changing the empty forward.
- Release when exact leave-out bits_saved on the sampled prior text is below storage bits. Training-item
  gradient attribution observes actual acquisition/replay backward passes, with micro-batches
  <= --micro-tokens: a policy or fact owns a birth only with a strict majority of absolute gradient mass on
  the birth group's coordinates (unlabelled items count in the denominator). Ledger mdl_concepts stores group ->
  concept id -> {version, session}; unversioned corpus concepts have version 1. Record provenance with each
  candidate's committed births, so selecting an earlier checkpoint cannot use later gradients. Recurrence preserves provenance.
- Conflict comparisons span physical groups, including competitors outside the selected deletion set, and run
  ONLY for explicit corpus replacements: replacements=[{concept, old_version, new_version, session}]. Keep
  the group with greater recent-session leave-out compression; ties keep both. A group representing multiple
  conflicts is kept if it wins or ties any comparison. The coarse corpus has no replacements, so this exception
  is a no-op there. Record conflicts and measured release-phase seconds in the printed release result/ledger.
Compared against tmid, tmid_util01 and tmid_util03 on primary metrics and net MDL gain. Other arms' training
paths are unchanged by this pre-run MDL change.

### v2.1 gate results (2026-10-10) and fallback activation

On the coarse corpus (seed 20261009, rules=coarse), with the full category table in context:
- genome dev concept 0.20 (test 0.23); ICL dev 0.27 (test 0.32): the identifiability gate (ICL - genome >= 10
  points) FAILS (+7).
- G1 teacher keep rate 0.33 (ticket 0.29, Q/A 0.36): FAILS (bar 0.80).
The base genome cannot apply even single-attribute lookup rules from context, so it cannot serve as a teacher for the
concept channel. Per the fallback fixed above, EP-1 v2.1 switches every arm to the external ground-truth supervised
target (d2 form: answer-only cross-entropy on the true resolution / action, 6 formats, 32 cases per policy per
appearance). This is supervised continual learning with revocable slots, not self-distillation from context; results
will be reported under that name. G2 (>= 0.70 dev concept under +0.005 dNLL, session 1, loramid and tmid lr 0.02 /
0.05) applies to the d2 target on the coarse corpus.

## Amendment v2.2 (2026-10-11, after G2 failed at 150 steps; earlier records stand)

G2 at 150 steps (coarse corpus, supervised target): best dev concept under +0.005 dNLL: LoRA 0.54 (still rising),
tmid lr 0.02 0.45, tmid lr 0.05 0.52; bar 0.70 not met.
Change: G2 is rerun at 600 steps per session, evaluation every 50 steps; for tmid arms the support refresh stops after
step 300 (support frozen, values keep training). Bar (0.70) and dNLL limit (+0.005) unchanged.
Decision rule: pass -> the 5-session v2.1 study at this step budget (arms run in parallel); fail -> EP-1 is reported
as "concepts not acquirable by this model at this budget by any method, LoRA included", and tuning stops.
Secondary pilot arm (labelled, not part of the gate decision): `lorafold` = fold-aware STE adapter on the same
free-slot mask (forward delta = alpha_row * Q(mask * BA / alpha_row), straight-through backward, Q = ternary round at
0.5 alpha; the served patch is the ternary slot codes only), rank 16, layers 14-15 up+down, local output penalty 10.
rehearsal 0.2 (next-token CE on general text), provisional until the FSA damage-control sweep reports.

## Amendment v2.3 (2026-10-11): canonical write set = pair8-forced zeros (owner rule)

All slot arms (tmid and its lr/util/mdl variants, tbop, fp4mid, lorafold) and revoke-scope write ONLY the
pair8-forced zeros: per 8-weight block, all 4 weights of the 2 pairs the base did not select. The mask is the shipped,
deterministic `pair8_forced` mask built from the export by the FSA harness's build_forced_mask.py (selected = the
block's nonzero pairs, filled to 2 with the lowest-index zero pairs), loaded from its file and verified by sha256;
EP-1 does not recompute or fork it. Patch values are plain ternary (alpha_row x {-1,0,+1}); revoke = zero the masked
positions -> the exact base. A patched expert is no longer pair8-format (it may use all 4 pairs); the revoke test
still requires the post-revoke weights to be bitwise the base and pair8-legal. LoRA arms are unchanged.
All v1 and v2/v2.1 slot-arm results (and the 150-step G2 pilot) used the earlier in-pair free-slot set (zeros
inside the selected pairs); they are recorded as such and not compared as if they used this mask. The v2.2 G2 pilot
is rerun for the slot arms on this mask; the LoRA G2 pilot (unaffected) continues.
Value format under v2.3: tmid and its variants drop the patch-only block scales (plain alpha_row x ternary, like tbop
and lorafold). fp4mid keeps its fp4 values (it is the precision comparison arm) on the same forced mask.

### G2 result (2026-10-11)
loramid, supervised d2 target, coarse corpus, 600-step pilot: dev concept 0.82 at step 250 (ticket 0.83, Q/A 0.81,
n = 56) with selection dNLL +0.0008: G2 PASSES. The 5-session v2.3 study (supervised objective, 600 steps/session,
eval every 50, ternary refresh through 300, forced write set for slot arms) proceeds with all registered arms; the
ternary-arm pilots (tmid lr 0.02 / 0.05, lorafold) are reported alongside but do not change the gate decision.

## Amendment v2.4 (2026-10-11): evaluation-set size for gates and the ladder (criteria unchanged)

The 56-item dev subset (trained policies only) has SE about 6.5 points, and single evaluations swung by 10-20 points.
For G2, the forced-mask ladder and every later pilot, dev concept accuracy is measured on a large dev set of 400
items: new cases of the trained policies from a reserved name pool disjoint from training, self-study, dev and test
pools, half ticket form and half Q/A form, deterministic from the corpus seed. Reported as mean +- binomial SE; a
curve is judged by its smoothed value = mean of the last 3 evaluations (and the best smoothed value over the run),
never a single evaluation. Bars (0.70), dNLL limit (+0.005) and the test-set criteria are unchanged. Earlier ladder
runs did not save checkpoints and cannot be re-scored; they are rerun with the large set.

## Secondary addendum: concept/rule placement study (2026-10-11, owner-approved; exploratory, no pass/fail)

Task: the v2.x coarse-rule supervised task (d2 target), judged on the 400-item dev set (v2.4 reporting: mean +- SE,
3-eval smoothed values); one seed.
1. WHERE. LoRA rank 16 on every layer's trainable linears (all MoE experts' up/down of all 32 MoE layers, and the
   sequence mixers' projections: attention q/k/v/o, EDA projections, MSA projections; routers, norms, embeddings
   excluded), trained to its best smoothed dev accuracy. The MSA indexer is fixed. Attribution at that checkpoint: (a) zero each layer's LoRA
   delta alone and measure the dev drop; (b) zero bands (early / middle / late thirds; attention-family-only;
   expert-only); (c) causal tracing on concept probes: corrupt the ticket's key attribute token(s) (swap to another
   value of the same attribute) and restore each layer's clean output, reporting the restored probability of the
   correct option per layer. Output: a per-layer attribution map (reported first).
2. ATTENTION SHARE. LoRA attention/mixer-only vs expert-only vs both, same rank, steps and learning rate; dev
   accuracy and dNLL. Free-slot FSA can only write experts, so this bounds what any slot method can reach.
3. TWO-BAND FSA (forced mask, Bop, plain ternary). Band A = middle MoE layers (facts/support), band B = later MoE
   layers (application); staged: stage 1 trains band A on the task; stage 2 trains band B with the stage-1 patch in
   place on application data (many phrasings, held-out formats). Regularizers: local penalty 10, rehearsal 0.2, an
   optional smaller patch scale. Compared with single-band FSA, LoRA(all) and LoRA(experts-only). Band boundaries
   are taken from the attribution map of part 1 (chosen before part 3 runs, recorded here when known).
Queued after the current ladder; runs on NEW-B GPUs under the lock convention.

### Ladder rung: sleeping-neuron gating (2026-10-11, secondary/diagnostic)
Writes are restricted to expert hidden units that are rarely active on general text (relu^2 pre-activation > 0 on a
small fraction of anchor tokens; the threshold and per-expert unit budget are taken from the FSA harness's writer,
which EP-1 reuses rather than reimplements). Their up-proj forced slots are trained so they fire on the rule-key
contexts and their down-proj forced slots carry the answer; a hinge penalty keeps their pre-activation below -margin
on general text. Ternary at full alpha and at beta = alpha/2. Controls with identical budgets: randomly chosen units
and the most-active units. Reported like the other rungs (400-item dev, smoothed accuracy, dNLL). Hypothesis: relu^2
gating keeps damage near zero while allowing full-alpha learning.
