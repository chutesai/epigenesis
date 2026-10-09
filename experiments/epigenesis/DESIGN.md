# EP-1 harness design: epigenetic consolidation on the lambda 8B

Internal design note for the EP-1 harness. Spec sources: three independent AI-assisted reviews (internal) of the
concept and plan, which converged on the pre-registered experiment, the gates/masks/loss, the evidence weighting,
the frozen teacher and the split-before-synthesis rule, plus their consolidated measurement-trap list. Reused code: `experiments/end_to_end/e2e_fsa.py`
(byte-identical to the dev box's `e2e_fsa_v7.py`).

## 0. Environment

| item | value |
|---|---|
| dev box | 8x RTX 5090 (32 GB). Working dir `/workspace/malleable/epi/` (harness copy), model/tokenizer/data under `/workspace/malleable/{mlf,data}` |
| python | `/venv/main/bin/python` (torch 2.14, tokenizers; no `datasets`/`pandas`) |
| model | `ops.eval.bench_saves.build_model(EXPORT, device="cuda:0", model_config=None, step=58814, expert_format="bf16", eda_kernel_backend="naive_fp32")`, ~14 GiB bf16; no KV cache: generation = full recompute per token (batched, right-padded, causal) |
| harness import | `sys.path.insert(0, "/workspace/malleable"); import e2e_fsa_v7 as fsa` (FSAParam, LoRAParam, attach, set_enabled, netcost support, LocalOutputPenalty, nll_windows, logits_of, batchify, encode) |
| GPU policy | use only GPUs with <1 GB used and no `e2e_fsa` process (map PIDs to GPUs via `nvidia-smi --query-compute-apps=pid,gpu_uuid`); never kill anything we did not start; exact PID only |
| wikitext pools | rehearsal anchors = `anchor-wikitext.json[0:150000]`; checkpoint-selection pool = `anchor-wikitext.json[-16*511:]` (16x512, disjoint from the anchor region); locked test panel = `eval-wikitext.json[0:40*511]` (40x512, never a selection target) |
| public sanity | `arc_easy_200.json` (200 four-choice ARC-Easy test items, seed 20261009, fetched from HF on the Mac, copied to the box) |

## 1. Module layout (all in `experiments/epigenesis/`)

```
epi_common.py   pure-torch pieces, CPU-testable: tokenization helpers, top-k teacher records, gated KL loss,
                mixing ratios, pair8 legality check, claim hashing, one-off regex, participation ratio
make_user.py    deterministic synthetic user corpus + probes (seed published in the JSON)
selfstudy.py    per-session frozen self-study set + null-KL calibration (genome + context as generator)
consolidate.py  one arm x one session (teacher/student KL training, births, rollback, promotion, replay)
evaluate.py     all metrics, revoke test, checkpoint selection helpers; standalone for genome / icl arms
run_ep1.sh      5-session protocol for all arms on free GPUs; smoke mode
tests/test_cpu.py  unit tests: corpus invariants, loss on tiny tensors, pair8 legality, mixing
EP1_PREREG.md   pre-registered success criteria (written before any GPU run)
```

Outputs (ignored by git): `out/<run>/corpus.json`, `out/<run>/selfstudy/s<k>.json` (+ `.pt` teacher records),
`out/<run>/<arm>/s<k>/{state.pt,metrics.json,replay.json,ledger.json,log}`, `out/<run>/summary.json`.

## 2. Data flow

```
make_user.py --seed S --out out/R/corpus.json
   corpus: policies[40], facts[100], lures[100], style, sessions[5] (episodes -> chunks), probes {dev,test}
selfstudy.py --corpus ... --session k --out out/R/selfstudy/s{k}.json     (one GPU job per session, 5 in parallel)
   genome + source-episode context generates answers; verifier; null-KL threshold; frozen set shared by every arm
consolidate.py --arm A --session k --run out/R                             (sequential over k per arm)
   loads arm state s{k-1}, trains one session, evaluates, writes s{k}
evaluate.py --arm genome|icl --run out/R                                   (no training; icl = context in window)
run_ep1.sh  -> summary.json, EP1 table, retention curves
```

Everything the probes need is built in `make_user.py` BEFORE any synthesis; self-study never sees probe items
(disjoint entity pools per split; asserted in tests).

## 3. Synthetic user corpus (`make_user.py`)

Fictional parcel-desk operations lead ("Fennmark Parcel Desk"; no real org). Deterministic from `--seed` (default
20261009). All text comes from handwritten templates + seeded sampling; no LLM at build time.

**Case attributes.** category (10 fictional ticket categories), tier (bronze/silver/gold/platinum), amount (currency,
1-5000), age_days (0-60), region (6 invented hub names), channel (email/phone/chat/portal), customer name (from
split-specific name pools), ticket id `FP-#####`.

**Policies (concepts), 40 = 10 categories x 4 mutually exclusive conditions.** Each category gets one splitting
attribute (amount bucket, tier, age bucket or region group), and 4 conditions partition it. Each policy has: codename
("the Lantern rule"), rule sentence (category + condition -> action), rationale sentence (templated per action), and
3 worked cases (ticket + resolution). Actions come from a closed vocabulary of 14 (`approve a full refund`, `request
photo evidence`, `escalate to the carrier desk`, ...); the 4 policies of a category get 4 distinct actions assigned by
the seed, so a 4-way probe's options are the category's four actions (hardest distractors; chance 25%; not guessable
from priors, since the assignment is random).

**Status vocabulary + answer shape (checkable style).** Every resolution in the corpus is written as
`STATUS: <RESOLVED|PENDING|ESCALATED|DECLINED> | ACTION: <action> | NOTE: <one sentence>`; the status is a fixed
function of the action. Style compliance = regex shape + closed vocabulary.

**Recurrent facts, 100.** (subject, relation, value) with unguessable values (hub leads, SLA hours, carrier codes,
depot towns, ...), each scheduled into exactly 3 of the 5 sessions, stated declaratively in user chat turns, with
2 phrasings. **Lures, 100.** one-time codes `XXXX-XXXX`, tracking ids `TRK-########`, case references; 20 per session,
each in exactly one session, embedded in ticket texts. Lures are what a fact-cramming writer would memorize.

**Sessions, 5.** Each session is a list of episodes: policy intro (user explains codename + rule + rationale, then one
worked ticket + resolution), policy recurrence (new worked ticket + resolution naming the policy), fact chat turns,
lure-bearing tickets, chit-chat (assistant filler; masked everywhere). Schedule: each policy is introduced in session
1-3 and recurs with NEW cases in 2 later sessions (3 worked cases total, 24 policy episodes per session); facts 60
mentions/session; lures 20/session. Sessions are ~6-8k tokens, so **sessions are chunked**: episodes are packed
greedily into chunks of <=1536 tokens (the session record). **The teacher's context unit is the source episode**
(~100-300 tokens: the policy statement + worked case, the fact turn, the lure ticket), not the whole chunk: the model
has no KV cache, so every generated token re-runs the full prefix, and a measured 1x2048 forward is 6.6 s (a 1.7k-token
context would make one session's self-study cost ~5 GPU-hours; the episode context costs minutes). The episode is
the grounded evidence the teacher needs; every probe carries `source_episode` (ICL ceiling uses the same unit).

**Probes, built first, disjoint splits.** Dev and test each: 100 concept items (new ticket from a split-specific
name pool, 4 options = the category's actions, shuffled; half in ticket form `... Resolution: STATUS: ... | ACTION:`,
half in Q/A form `Q: Under the user's policies, which action applies ...? A:`), 50 verbatim items (recurrent facts,
declarative-cloze and Q/A families), 50 lure items (same families). 20 fixed style prompts (tickets) and 32 fixed
revoke prompts (mixed corpus/wikitext/ARC text). The genome's expected concept accuracy is chance.

## 4. Self-study (`selfstudy.py`)

Generator = **genome** with the source episode in its window (so the set is identical for every arm; per-arm teacher
*distributions* are recomputed by the arm's own session-start individual in `consolidate.py`). Greedy decoding,
batched across items (rows = episode + prompt + partial answer, right-padded), <=32 new tokens,
stop at newline/EOS.

Item kinds (per session):
- `apply`: "Apply <codename> to this new ticket" = new ticket (self-study name pool, disjoint from probes) +
  `Resolution:`; 2 per policy episode. Verifier: mean answer-token entropy <= `--ent-max` (2.0 nats) AND
  (source entailment: the ACTION field equals the action the source policy prescribes for that ticket, OR agreement
  of a second teacher draw at T=0.7 on the ACTION field).
- `style`: "Resolve this ticket in the user's format" for tickets with no policy hook; verifier = shape regex + status
  vocabulary + entropy.
- `factqa`: Q/A on claims whose claim-hash has recurred in >=2 sessions so far (recurrence counted on SOURCE
  occurrences in the corpus ledger up to this session, never on paraphrases); 1 per recurrent claim; verifier =
  entropy + second-draw agreement. Lures never recur, so they are never eligible; additionally any answer containing
  a one-off pattern (`[A-Z0-9]{4}-[A-Z0-9]{4}`, `TRK-\d+`, `FP-\d{5}`) is rejected.
- `modelq`: the genome is asked for a question about the episode (2 per policy chunk-group, capped); the questions are **inputs only** (masked); the
  teacher answers them; same verifier as factqa plus the one-off filter.
Dedup by claim hash (normalized source span + ACTION); cap `--per-source 3` items per source episode. Assistant
chit-chat is never a target. Record for each accepted item: source episode id, prompt ids, answer ids, verifier branch, and
the genome-teacher top-k (k=32) logits + logsumexp per answer token (fp16) in `s{k}.pt`.

**Null-KL threshold.** For the accepted items, run the genome with a length-matched *unrelated* context (a wikitext
anchor slice of the same token length) and the genome with no context; per-token KL(unrelated-ctx || no-ctx) over
answer tokens. Threshold = 95th percentile; written to `s{k}.json` and reused per arm.

Cost: no KV cache, so each generated token is a forward over the batch. Budget per session ~110 generations x <=48
tokens in batches of 8 rows x ~1.7k tokens, plus second draws only for items not settled by entailment/shape; the
smoke measures the per-forward time and `run_ep1.sh` runs the 5 sessions' synthesis in parallel on free GPUs.

## 5. Consolidation (`consolidate.py`, one arm x one session)

**Arms.** `genome` (no training), `icl` (context in window, no write; evaluation only), `tmid` (fsa_free ternary,
L14-15 up+down, netcost births inside the router superset, patch-only block scales block 16 init 0.25 alpha, wd 0.3
on labile slots only, no smooth penalty), `fp4mid` (same placement; fp4 on the 0.25-alpha grid, local penalty 10,
wd 0.1, unbudgeted: the recipe that learned at zero damage), `loramid` (r16, L14-15 up+down, local penalty 100).
**KV-prefix: skipped.** A learned prefix only at the 6 MSA layers does not define a prefix state for the EDA
recurrence or the sliding window; a prefix would corrupt the recurrent state and measure the wrong thing.

**Session start.** Build genome; attach the arm parameterization to ALL experts of L14-15 (prior slots must stay
applied); load `s{k-1}/state.pt` (slot values, block scales, LoRA factors, frozen mask, replay, ledger). Rehydrate:
`M = committed quantized value` (ternary +-1 -> M=+-1; fp4 -> value in alpha units); Adam re-created (moments reset).
Freeze teacher = this individual (patch enabled). Compute and cache (CPU, fp16): (a) teacher top-k records with
context for every accepted item; (b) context-free session-start records on the same answer tokens (the gate and the
student's starting point); (c) anchor records: session-start individual on 64 anchor windows x 256 tokens;
(d) per-token gate `w_t = 1[KL(teacher_ctx || start_noctx)_t > null_thr] * 1[H(teacher)_t <= ent_max]`.
Router superset: profile L14/L15 expert token counts of the current individual on the session's context-free
self-study prompts; superset = smallest expert set covering 90% of routed tokens. Births allowed only in superset
experts; prior slots elsewhere stay applied but untrainable this session.

**Loss (per step, by loss-bearing-token mass).** `L = a*KL_acq + g*KL_gen + r*KL_rep` with `--mix balanced`
(0.4/0.4/0.2, default), `acquire` (0.5/0.25/0.25), `retain` (0.5/0.3/0.2) in (acquire, general, replay) order. Session 1
(empty replay) reassigns r to g. Each term is a mean over its own gated tokens of KL(teacher || student) at T=1 over k+1 buckets:
the teacher's top-32 ids (full-vocab teacher log-probs, student log-probs gathered at those ids) plus one tail bucket
holding each side's remaining mass; exact for the bucketed distributions, zero when the student matches the teacher,
and student mass leaving the teacher's top-k is charged to the tail (`epi_common.topk_kl`).
`KL_acq`: student = prompt + answer WITHOUT context; teacher record (a); weights (d). `KL_gen`: anchor windows vs (c)
(stop-grad session-start individual, NOT the genome). `KL_rep`: replay items vs their stored records. fp4mid/loramid
add the smooth local penalty (`fsa.LocalOutputPenalty`) on anchor windows.

**Writer.** AdamW on masters (lr 0.02 FSA / 1e-3 LoRA; scale lr 0.002, no decay on log-scales); decoupled wd applied
manually only to labile (born-this-session) coordinates; frozen slots get zero grad, no decay, no eviction.
tmid births by netcost (`fsa.support_scores` + `refresh_support`, EMA of signed dL/dM, lambda calibrated once in
session 1 with `cost_rel 1` and persisted in the ledger) every 8 steps until step 48, ceiling = `--budget` (2M slots
across the run; the per-session labile cap is budget minus frozen). Steps per session: `--steps 48`, each step = 8
self-study items (uniform over source episode, cap per source), 2 anchor windows (96 tokens), 4 replay items;
micro-batched to <= `--micro-tokens` (160) tokens with gradient accumulation.

**Checkpoint selection, rollback, promotion.** Every 16 steps and at the end: dev concept (100), dev lure (50),
selection-pool dNLL (16 windows) vs session start. Discard checkpoints with dev lure > genome + 5 points. Among the
rest with dNLL <= +0.005, pick the highest dev concept (ties: smaller patch). If none qualifies, **roll back** the
session to its start state (logged as `rolled_back`). Promotion at session end: labile slots that survived the last
support refresh are promoted to frozen iff the chosen checkpoint's dev concept exceeds the session-start dev concept;
otherwise they stay labile (decay and eviction apply next session). No contradiction/unfreeze rule in EP-1 (the
corpus has no corrections; noted as a limitation).

**Replay buffer.** Entries = accepted `apply`/`style` items and `factqa` items for claims with recurrence >= 2, with
their teacher records from the session they were accepted; cap 256 (uniform over policies, newest first); never lure
text (one-off regex + recurrence gate); consolidation-only; wiped on revoke; its contents are written to
`replay.json` as personal state.

**State contract.** Semantic state = committed slot values (+ block scales / LoRA factors). Operational metadata =
ledger (frozen mask, births per session, netcost lambda, superset per session, claim recurrence counts). Temporary =
masters, moments, teacher records (deleted at session end). Revoke clears all three.

## 6. Memory and time plan (32 GB, naive EDA fp32 backend; measured on GPU 0, 2026-10-09 11:45Z)

| measurement | value |
|---|---|
| model bf16 | 14.5 GiB |
| + fp32 masters L14-15 all 256 experts up+down (481M incl. block scales) | 16.8 GiB |
| forward (no grad) 1x512 / 4x512 / 1x2048 / 2x2048 / 1x4096 | 3.1 s / 2.3 s / 6.6 s / 8.6 s / 12.2 s, peak <= 15.6 GiB |
| train step (fwd+bwd+Adam) 1x256 | 10.0 s, peak 25.1 GiB |
| train step 2x256, 1x1024, 1x2048 | OOM (the naive EDA scan stores per-step state for backward) |

First smoke (12:50Z): resident state after step 1 (model + fp32 masters + Adam moments + grads + netcost
score/support) is ~23.4 GiB and a 256-token micro-batch added ~8 GiB of activations: 29.4 GiB at step 1, OOM at
step 2. Consequences: training micro-batches are capped at **160 tokens** (`--micro-tokens 160`; one student row
of ~135 tokens per micro-batch); anchor windows are 96 tokens; teacher and
anchor records are computed under `no_grad` in 4096-token micro-batches; top-k records (k=32, fp16) instead of
full-vocab logits. Per step ~1.5k loss-bearing tokens = ~7 micro-batches = ~70 s; 48 steps/session = ~55 min
training + ~25 min teacher records and evaluation per session per arm (verified in the smoke, which prints per-phase
times and peak memory). Self-study with episode contexts: ~60-80 generations x <=32 tokens in batches of 8 rows x
~400 tokens = ~2 s per decode step = ~10 min per session. `run_ep1.sh` refuses the full run if the smoke peak exceeded
30 GiB.

## 7. Evaluation (`evaluate.py`, after every session)

- test concept accuracy (4-way, cloze: mean log-prob of each option's tokens after the prompt, argmax), split by
  ticket-form and Q/A-form; dev likewise (dev is used for selection, test only for the retention curve)
- verbatim recall (teacher-forced greedy exact match of the value tokens; dev/test, both families)
- lure recall (same scoring; must stay <= genome + 5)
- locked wikitext 40x512: mean dNLL vs genome with paired SE (windows as the unit) and the 95% paired interval;
  greedy-token agreement with the genome on the same windows (genome argmax cached once per run)
- ARC-Easy 200 (cloze over the 4 options, length-normalized), same items for every arm
- style compliance: greedy generation (<=40 tokens) on the 20 fixed ticket prompts, shape regex + closed vocabulary
- patch: slot count, surviving births, bytes (slots + scale metadata; LoRA fp32 factors), participation ratio over
  experts `(sum n_e)^2 / sum n_e^2` and over rows
- retention: concept(test) after session 5 / best earlier session; lure trajectory must not ratchet upward
- revoke (after session 5): zero slots + scales + masters + moments + replay + ledger; check (i) every patched
  linear's effective weight tensor (the pair8 bf16 tensor that would be packed) is bitwise equal to the genome's
  and pair8-legal, (ii) logits on the 32 fixed prompts equal within bf16 noise (report max |diff| and `torch.equal`),
  (iii) concept/lure metrics back to the genome's. `logits_pass` = max |diff| <= 1e-2 (bf16 noise tolerance,
  fixed here); exact equality is reported separately. Lure "ratchet" = monotone non-decreasing lure recall across
  sessions with a net increase.

## 8. Run protocol (`run_ep1.sh`)

1. `make_user.py` -> corpus; `tests/test_cpu.py` must pass (also run on the Mac before deploy).
2. Self-study for sessions 1-5 on free GPUs (parallel; each job owns one GPU via `CUDA_VISIBLE_DEVICES`).
3. `evaluate.py --arm genome` and `--arm icl` (identifiability gate: icl dev concept must beat genome by >= 10
   points, else the run stops and reports).
4. Per arm (`tmid`, `fp4mid`, `loramid`): sessions 1..5 sequentially on one GPU each; `summary.json` collects the
   per-session metrics; `--smoke` = sessions 1-2, 10 steps (lr x5 for FSA arms so masters cross the ternary threshold), 8
   self-study items, eval subsets, revoke test after every smoke session; SMOKE_OK requires peak <= 30 GiB, nonzero
   committed patches, tmid births + calibration/refresh, rehydration in session 2, revoke pass.
5. Launch detached: `setsid nohup ... </dev/null > log 2>&1 &`, no `cd X &&` prefix; exact-PID bookkeeping in
   `out/<run>/pids`.

Throughput is unknown until the smoke (no GPU is free at design time); ETAs are reported from the smoke's measured
per-phase times.
