# C40 — relation-world and served-negative fixes

| Item | Fix |
| --- | --- |
| F1: two-hop world/gold mismatch | `two_hop.main` passes `meta.get("relation_world", False)` as the `relations` argument to `E.make_world`. Shared `branch_eval.sister_relations` derives sister targets from stored relation values and asserts complete, known targets consistent with the three-person cycle. Pointer training/evaluation targets therefore use the stored sister fact. C29 uses the same derivation. Legacy metadata retains the original world and external cycle. |
| F2: selector-gated negatives | Every category now reports `served` beside the unchanged `base`/`patched` forced-bank results, with the same `false_recall`, `confidently_wrong`, `asserted_answer`, and `n` metrics. `suite` forwards its locked selector; empty routes generate with base and contribute to both metrics and abstention counts/rates. Per-item served rows record routes and abstentions. Without a selector, serving explicitly means all banks resident at full strength. |

Validation: **185 passed** across C34, C36, C38, C39, the existing two-hop
regressions (C23/C27/C29/C32), and C40. The **10 C40 cases** exercise production
world reconstruction with relation/legacy metadata, stored-fact pointer gold,
relation consistency failures, production suite selector wiring, base generation
on abstention, metric/count correctness, unchanged forced-bank fields, and resident
activation of all banks. `git diff --check` passed. Validation used tiny CPU
fixtures; no full-model/GPU evaluation was run. No commit was run.
