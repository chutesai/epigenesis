# C36 — C34 independent design review fixes

Applied `reviews/experiments/c34_eval_suite/design_review_muse.md` to the current
C35 tree. CLI defaults remain unchanged; no commit was run. C34's opt-in JSON
schema changes deliberately to distinguish correctness from presence.

| Review item | Fix |
| --- | --- |
| §1 training-budget confounding | Every saved checkpoint records optimizer/parameters, configured recipe, actual value/wake steps, budget, augmentation, local penalty, fact batch and factual/self-study/wake/allocation exposures. C34/PK refuse missing or unequal recipe, steps, selection policy or normalized exposure counts across all full/block/unit files. Token totals and global fact IDs are recorded separately because the matched fictional people have different name lengths/IDs. |
| §1 comparison manifest | Branch JSON includes each checkpoint's SHA-256, effective mask SHA-256, initialization, exposures, selection policy and file bytes, plus locked selectors and their bytes, including PK reserves. |
| §1 family matching | Each supplied block/unit family has its own solo, pinned-independent and merged arms, using exactly its own files. Cross-family paired comparisons are removed. |
| §1 merge order | `Branches.activate` merges in sorted bank-ID order, including `last_write`. |
| §2 held-out wording | All seven C34 templates have a disjointness assertion against training, augmentation, self-study, address, validation, test and QA wording; conjunction wording no longer repeats self-study/QA stems. |
| §2 pronouns | Original panel renamed `single_mention_pronoun`; matched earlier/later target panels include a same-gender competitor and an explicit arrival-order cue. These measure cued antecedent retrieval, not unrestricted pronoun resolution. |
| §2 effective sample size | Panels report distinct facts, people and relations. C34 paired comparisons report counts/deltas only, with SE and within-noise verdict unset, avoiding iid inference on overlapping facts. |
| §2 legacy composition | Cross-person conjunctions now enter decomposition even in legacy worlds; each family reports the multi-bank subset and paired subset separately. |
| §2 partial bank support | Ceiling, format, rescore and routing reports split supported/unsupported items. |
| §3 scorer/item agreement | Primary `correct` parses a terse first sentence: exactly two candidate values in question order, with no negation, swaps, duplicates or listings. Prompts request that format. Name-dependent binding is reserved for attributed arm-G items. |
| §3 misleading labels/probes | Presence diagnostics renamed `presence_stop`/`presence`; `GEN` removed from C34 ceilings. Shared QA/branch labels are `two_fact_correct`, `two_fact_presence_stop`, `two_fact_presence`. Tests include the exact listing, negation and swap review probes, plus terse gold answers. |
| §4 multi-layer estimand | Retained the review's permitted pinned construction, renamed `per_layer_pinned_independent_outputs`. Definitions and comparisons explicitly state that later pins erase propagation, solo trajectories/router weights differ, and gaps cannot identify pure merge nonlinearity. A two-patched-layer fixture demonstrates the erased propagation. |
| §4 dispatch fail-open/alignment | Every forward verifies nonempty actual dispatch at every MoE layer, that every dispatched touched Linear fires, and sufficient output rows. Experts assigned zero tokens legitimately skip Linear calls. Empty patches still verify their registered expert path. Both residual replay and conditioned recall reject bypass/empty/short-row paths and clean up hooks/activation on failure. Unmatched-row counting remains. Alignment wording now says token positions. |
| §5 negative effective n | Sixteen fixed plausible absent attributes, rotated foreign facts and distinct wording produce ≥16 unique queries per feasible category per bank; generator/report enforce the minimum and every rate includes n. Explicit `qa-n=0` and impossible foreign scopes remain unavailable. Small positive `qa-n` values retain the negative minimum. |
| §5 near misses | Seeded sampling varies mutation position/replacement and rejects names or values already in the world. |
| §5 refusal echoes | Relation values and the queried name are excluded from both leakage and emitted-value confidence. |
| §5 confidence/hallucination | Reports joint probability and per-token geometric mean; ≥0.8 threshold uses the latter. `asserted_answer` measures generic first-sentence answers independently of stored-value leakage for both base and patched. |
| §5 serving interpretation | Negative results explicitly say `forced-bank leakage`, bypassing selector abstention. |

Validation: full end-to-end CPU suite **868 passed, 1 skipped**. Final C34/C36
regressions **48 passed**, including all 26 C36 cases and two negative-report
tests added after the full suite collected. C34/PK CLI positive controls pass; `git diff --check` is
clean. Full-model/GPU dispatch has not been run in this workspace; runtime guards
fail closed on unsupported paths.
