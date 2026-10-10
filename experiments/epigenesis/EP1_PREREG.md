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
