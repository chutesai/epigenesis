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
