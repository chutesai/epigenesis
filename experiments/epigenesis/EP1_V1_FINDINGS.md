# EP-1 v1 findings

## Pre-registered verdicts

Test probes after session 5: 100 four-way concept items. Genome 0.24, ICL 1.00; the ICL-beats-genome-by-10-points gate passed.

| arm | test concept | lure | locked ΔNLL [95% CI] | greedy agreement | revoke |
|---|---:|---:|---|---:|---|
| tmid | 0.32 | 0 | −0.0001 [−0.0029, +0.0027] | 0.931 | pass |
| fp4mid | 0.32 | 0 | +0.0074 [+0.0035, +0.0114] | 0.893 | pass |
| loramid | 0.28 | 0 | +0.0009 [−0.0027, +0.0045] | 0.924 | pass |
| tbop (added after pre-registration) | 0.32 | 0 | −0.0007 [−0.0042, +0.0027] | 0.931 | pass |

- Concept ≥ genome + 10 fails for every arm (best +8, about 1.7 SE from chance), so the thesis fails.
- Lure and ΔNLL criteria hold, but lure is uninformative because nothing was learned (verbatim recall is 0 everywhere). fp4mid's damage interval excludes zero.
- Revoke passes everywhere: weights bitwise equal to the genome, logit max |diff| 0.
- "Match LoRA" is formally met by tmid and tbop but uninterpretable near chance; no arm beats LoRA.

## Why consolidation failed

LoRA fit its own targets (acquisition KL 4.84 → 0.11), so the storage method is not the bottleneck. The training signal is:

- Self-study `apply` prompts name the rule ("Apply the Kestrel rule to this new ticket…"); the probes give only the ticket. The arms learned codename → action, not category + condition → action.
- Claim-hash dedup left about one `apply` item per policy per session (23–25 per session).
- Without the codename, the genome with only the source episode in context picks the correct action 50% of the time, so a probe-format teacher is weak.

## Ablation (session 1, 150 steps, 16 cases per policy, test n ≈ 57)

Genome: 0.20 dev, 0.24 test.

| data | LoRA dev peak → test | tmid dev peak → test |
|---|---|---|
| d0: original self-study set | 0.38 → 0.375 | 0.30 → 0.41 |
| d1: probe-format teacher KL, no codename | 0.48 → 0.51 | 0.36 → 0.42 |
| d2: supervised cross-entropy, 6 formats | 0.48 → 0.37 | 0.34 → 0.21 |

No setting reached the 0.70 target. Dev curves are noisy and non-monotone, consistent with overfitting 16 cases per policy plus evaluation noise. Supervised ground truth (d2) does not beat d1, so the model is not learning the numeric-range and region-set decision boundaries from this few cases.

## Revoke-scope v1: uninformative

Condition A never learned its items (generated recall 0.0 in every condition), so the revoke variants cannot be compared. A wrote 1.13M slots; 59% of B's writes touched A's slots, but 785k of those zeroed an A slot and only 3.2k overwrote one. That is B's eviction and decay, not semantic correction.

## EP-1 v2 (proposed)

1. Teacher context: all four policies of the case's category (about 600–1000 tokens) instead of one source episode. Gate: the teacher keeps ≥ 0.80 of items; otherwise coarsen the corpus conditions first.
2. d1-form data: no codename, half ticket form and half Q/A form, 32 cases per policy per appearance, dedup only on identical cases.
3. 150 steps per session, eval every 25 steps; v1's selection, +0.005 ΔNLL rollback and lure discard unchanged; tmid at learning rates 0.02 and 0.05.
4. Pilot gate: session 1 only, LoRA and tmid; run the 5-session study only if some arm reaches ≥ 0.70 dev concept.
5. Arms: genome, ICL, tmid, fp4mid, loramid, tbop, plus tmid_util01 and tmid_util03 as labelled secondary comparisons. Success criteria unchanged.
6. Revoke-scope v2: policies in d1 form with the whole-category teacher, facts trained with the 12-format supervised recipe; raise steps until A reaches ≥ 80% generated recall, then rerun the seven conditions.

## v2 gate G1 failed, and v1's ICL ceiling was invalid (2026-10-10)

G1 (teacher keep rate with the full category rule table in context): 232 of 768 items kept, 0.30 (ticket form 0.27,
Q/A form 0.34) against chance 0.25 and the 0.80 bar. With all four rules of a category in its window, the genome
cannot apply them to a new ticket.

This invalidates v1's ICL ceiling and identifiability gate. Each probe's ICL context was the source episode of the
correct policy alone, so the context already contained the right action; the 1.00 measured that leak, not rule
application. The same leak applies to v1's self-study teacher and to the ablation's d1 teacher (50% keep even so).
With a non-leaking context the real in-context ceiling on these rules is about 0.30, so no v1 or ablation arm had a
teacher worth distilling on the probed skill.

## v2.1 gate 1 on coarse rules (2026-10-10)

Even with one categorical attribute per rule (tier or channel, stated on the ticket) and the full category table in
context, the genome scores 0.27 dev / 0.32 test (genome without context 0.20 / 0.23), and the teacher keep rate is
0.33. The identifiability gate and G1 both fail, so EP-1 switches to ground-truth supervised targets (pre-registered
fallback).
