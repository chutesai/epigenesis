# C61 — C57 pre-launch review fixes

## C61b — independent re-review follow-up

Addressed the remaining CC1 bug and advisory-number mismatch in `C61_REVIEW_FIX.md`. This follow-up edits only `composition_contract.py`, `tests/test_c61.py`, and this report. No git commits or GPU runs were made.

- Unknown-query emissions now match every stored non-relation value in the checkpoint world, regardless of person ID or loaded-bank membership. Canary matching remains scoped to the canary's target values. The separate any-leak bound still intersects emitted fact IDs with loaded facts only.
- Off-topic panels now reuse `e2e_fsa.extra_eval_panels` while retaining the `wikitext`, `fineweb`, and `code` report labels: exactly 40 windows, each BOS plus 511 tokens. Surplus tokens are truncated and short files rejected by the harness function. These rows remain advisory.
- Regression tests score generated strings rather than assign `emitted_rate`. They check every non-relation value in the three-person production world, exclude relation-only and unstored strings, count multi-value answers once, and exercise person IDs 0, 1, 2, and 7 at both 10/1000 (no kill) and 11/1000 (kill). Non-loaded values leave the loaded-fact numerator and denominator unchanged. Off-topic tests verify harness-function reuse, window equality, and matching CPU-model paired ΔNLL.

C61b verification (local CPU environment):

```bash
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_c61.py -q --tb=short
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_c61.py tests/test_c57.py -q
/private/tmp/epigenesis-ladder-venv/bin/python -m py_compile composition_contract.py tests/test_c61.py
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q
git diff --check -- composition_contract.py tests/test_c61.py C61_REPORT.md
```

Focused C61 regression run: **30 passed in 1.42s**. Final C61/C57 integration run: **56 passed in 2.79s**. Full suite: **1487 passed, 1 failed, 1 skipped in 382.53s**. The sole failure is the previously documented `tests/test_ops_scripts.py::test_leftover_group_is_bounded_after_leader_exit_without_heartbeat`: an eight-second timeout waiting for the job lock and runner ownership to clear. That test imports the existing ops scripts, not CC1; no ops file was edited. The full run includes the final loaded-fact denominator assertions. Compilation and scoped whitespace checks pass. No production ΔNLL or independent GO re-review is claimed.

The original C61 account below is historical; its claim that all CC1 fixes were complete was corrected by the independent re-review.

## Original C61 implementation record

Implemented the panel and all ten numbered CC1 fixes from `C57_REVIEW_FIX.md`, using the adopted Edits in `../../reviews/experiments/cc1_review/design_review_grok.md` and DM v2. No GPU run or production score is claimed. No git commits were made. `e2e_fsa.py` and the approved `serialize_patch.py` were not edited; pre-existing workspace changes were preserved.

## Changes and locations

| Review item | Implementation |
|---|---|
| Panel census certificate | `in_topic.py:54` reconciles explicit `--census-read PATH:END_TOKEN` declarations against every supplied frozen census record. Missing streams, short declarations, changed recorded source hashes, and cursors beyond file length are refused. Missing records/fingerprints leave the panel report-only. Repeated records/streams use the largest end, including resumed reads. Native resume records conservatively require the reserved target as well as the end cursor, covering calibration and agreement passes. `in_topic.py:116` verifies certificate/source hashes and repeats the stride-one 32-gram scan on load, including a certified report-only DM artifact used by CC1. Gate-ready manifests lacking this provenance are refused. |
| Separate held-out text | `build_in_topic_panel.py:251` adds repeated census-read/record options and `--heldout-text`; `build_in_topic_panel.py:266` takes the separate file in its entirety, independently excluding the original census/training streams. Census references cannot shorten existing training references. `build_in_topic_panel.py:336` records the retained-window scan and requires completeness for `gate_ready`. The old numeric prefix flag cannot make a gate-ready panel. |
| Written-fact reference | `in_topic.py:337` records `meta.alone_resident_test_gen` when present. `in_topic.py:376` compares co-resident TEST GEN to that stored alone-resident fraction. Otherwise the bank row explicitly records `stored_reference_available=false`, is report-only, and cannot pass. `in_topic.py:424` no longer rescores the alone bank under caller generation flags. Historical checkpoints without a stored reference remain report-only. |
| CC1 1: all-family overlap | `composition_contract.py:41`, `:189` reject either overlapping slots or sign conflicts for full-mask and whole-unit inputs before scoring. The CLI writes a mechanical `kill` report for an overlapping family before loading a model (`:556`). No overlapping inputs are merged with sum clipping. |
| CC1 2: real unload/order/revoke set | `composition_contract.py:197` edits resident ternary masters and reset masks directly. Apply preserves resident codes; unload clears the removed bank's code coordinates and its reset rows/columns, restoring the pristine base contribution and keeping surviving codes. `:226` applies A/B in actual order from a fresh base; it never replays a saved remaining-bank checkpoint. `:235` uses branch_eval's first 15 eval tokens plus the exact two pristine prose prompts. Effective weights and all these logits are compared bitwise (`:279`). |
| CC1 3: replacement unavailable | Only A/B are required or loaded (`:517`, `:539`). The legacy `--a-prime` argument is ignored. A′B is not scored and gets an explicit unavailable table row (`:431`). |
| CC1 4: canary census | `composition_contract.py:84` builds exactly ten held-out surfaces for every owned non-sister fact. No shared-value filtering, training/validation surfaces, archive padding, or reverse-lookup value canaries. Fact-level any-surface bounds are separate under B and AB−A (`:172`, `:353`); two leaking facts kill. Reverse lookup has a separate name-emission table (`:128`, `:292`). |
| CC1 5: unknown entities | `composition_contract.py:102` uses `c52.absent_items`, with at least 1000 distinct virgin names. `:292` scores stored non-relation value emissions; `:353` kills above 1% of queries, records forced-bank/all-loaded as one measurement, and separately reports resident-fact any-emission bounds. Probe-rate and fact-rate table rows have their own denominators (`:431`). |
| CC1 6: CC1 in-topic rules | `composition_contract.py:332` uses base-correct, non-primed value QA clustered by concept. A concept event is a positive loss of its frame-mean score; more than 1 event per 100 concepts kills. Text kills on paired excluded mean >+0.003 and promotes only if mean+1.645×SE ≤+0.003. These decisions do not use DM `passed`. A certified, calibrated 24-concept artifact can be screened even though DM's 40-pair readiness remains false; its exact bound still blocks promotion. Without the certified panel, rows are unavailable. The CLI pins export step 58814 and naive_fp32. |
| CC1 7: retention/both/conflicts | `composition_contract.py:128` uses the complete non-sister fact census: both TEST templates and all three unseen QA templates. Retention kills use fact-mean loss, with exact any-loss promotion bounds (`:150`, `:353`). No relational-retention kill. The fixed-seed conjunction sample has 200 unique items on the production roster and covers every A/B fact, including sister links (`:110`); small CPU worlds report their smaller exhausted pair set without padding. Only p0→p1 chains are built, with `n_relations` reported separately. Conflicting identifiers cover the full three-person `conflict_map`; any other person's emitted value is confusion, displayed separately. |
| CC1 8: paired seeds | `composition_contract.py:469` retains every seed's panels, estimates, bounds, decisions, and denominators. Any seed's kill or mechanical failure kills. No seed averaging or pooled binomial n. Incomplete/non-C61 evidence cannot promote. |
| CC1 9: controls | `composition_contract.py:244` contains CPU controls using real FSA effective weights and the live edit/scoring operators: disjoint one-fact AB−A≡B; a planted B code emits A's value and is flagged by a fact-clustered canary; overlap AB−A≠B. Failed controls void the run (`unavailable`, `void=true`) instead of killing the hypothesis. |
| CC1 10: outcomes/table | `composition_contract.py:431` emits configuration, panel, n_strings, n_clusters, estimate, `upper95` (one-sided 95% upper), and decisions from the four required values. A′B and the independent repaired bank-C pairing are unavailable. A repaired pair cannot promote; the old generic unseen-report path refuses to substitute another jointly trained pair. |

`tests/test_c61.py` covers each numbered bug, all three controls, unknown/concept/text kill lines, actual census ends and source hashes, chunk-boundary overlap, separate-file selection, native resume targets, 200-item roster coverage, control failure voiding, three-person confusion, and CLI overlap kill without model loading. `tests/test_c57.py` updates the old assertions to the adopted contract while retaining real CPU model/CLI integration. `tests/test_c32.py:123` updates an old QA mock to accept the existing harness keyword `fact_class_metrics`; no runtime numerics changed. The local `.gitignore` keeps the new test/report visible without editing the repository-level ignore file outside this workspace.

## Verification

Commands run locally from this directory:

```bash
/private/tmp/epigenesis-ladder-venv/bin/python -m py_compile composition_contract.py build_in_topic_panel.py in_topic.py
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_c61.py -q
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_c61.py tests/test_c57.py tests/test_c49.py tests/test_c45.py tests/test_c32.py::test_damage_and_benchmark_follow_routes_and_match_base_batch_size -q
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_c61.py tests/test_c57.py -q
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest tests/test_ops_scripts.py::test_leftover_group_is_bounded_after_leader_exit_without_heartbeat -q
git diff --check -- build_in_topic_panel.py in_topic.py tests/test_c57.py tests/test_c32.py
```

C61 regression coverage: **21 CPU tests**, all passing in the final focused run. Focused harness/integration run before the final extra two regression cases: **101 passed**. Final focused run: **104 passed in 5.34s**; paired-focused run: **46 passed in 1.95s**. Full suite: **1445 passed, 1 failed, 1 skipped in 396.68s**. The outstanding failure is `tests/test_ops_scripts.py::test_leftover_group_is_bounded_after_leader_exit_without_heartbeat`, which timed out after 8 seconds waiting for existing shell-runner cleanup. Its isolated retry also failed (8.66s). That test imports only the existing ops scripts, not the C61 modules; no ops runtime or test was edited. Python compilation and scoped whitespace checks pass. The final focused run also checks the last census domain-order and complete-record safeguards and conjunction coverage edits made during the broad run.

## Exact box commands (not run here)

From `/workspace/malleable` after deploying these files and tests into `v58/`:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -m pytest -q v58/tests
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -c 'import sys; sys.path.insert(0,"v58"); import composition_contract as C; r=C.positive_controls(); print(r); assert r["passed"]'
```

The next command uses the actual C51 checkpoint to derive declarations, rather than guessing the 13M budget's per-stream end cursors. The fresh input `data/census/fwhq_fresh_holdout.u32` must be a separately materialized FineWeb-HQ slice. Include every additional census invocation's checkpoint in `records`; checkpoints and streams must remain accessible when loading the panel. Native checkpoint target coverage is intentionally conservative. A JSON census record is also supported with `streams`, recorded `cursors`, and `inputs` mapping absolute source paths to recorded SHA256 hashes. A plain declaration without independent recorded cursors/fingerprints remains report-only.

```bash
/venv/main/bin/python - <<'PY'
import subprocess
import sys
from pathlib import Path
import torch
records = [Path('out/C51/census-725B.resume.pt')]
ends = {}
for path in records:
    r = torch.load(path, map_location='cpu', weights_only=False)
    identity = r['identity']
    target_by_domain = dict(zip(identity['domains'], identity['targets']))
    for domain, source in identity['streams'].items():
        end = max(r['cursors'][domain], target_by_domain[domain])
        ends[source] = max(ends.get(source, 0), end)
command = [sys.executable, 'v58/build_in_topic_panel.py',
    '--export', '/workspace/malleable/data/hf/exports/725B-tokens_step58814',
    '--export-step', '58814', '--eda-backend', 'naive_fp32',
    '--fineweb', 'data/census/fwhq_sel.u32',
    '--heldout-text', 'data/census/fwhq_fresh_holdout.u32',
    '--anchor', 'data/anchor-wikitext.json',
    '--training-text', 'data/census/cp_wikimedia.u32', 'data/census/fwhq_sel.u32',
    'data/census/pd_cp_pre1929.u32', 'data/census/cp_stackexchange.u32', 'data/rehearsal-fineweb.json',
    '--eval-panels', 'data/eval-wikitext.json', 'data/eval-fineweb.json', 'data/eval-code.json',
    '--calibration-patch', 'out/GL_CTRL_s1', '--context-length', '512',
    '--text-windows', '40', '--batch-size', '1', '--out', 'data/in-topic-C61-725B.json']
for path in records:
    command += ['--census-record', str(path)]
for source, end in sorted(ends.items()):
    command += ['--census-read', source+':'+str(end)]
subprocess.run(command, check=True)
PY
```

With the built artifact, screen the two families separately:

```bash
/venv/main/bin/python v58/composition_contract.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --a out/R20_unit_p0/final.pt --b out/R20_unit_p1/final.pt \
  --in-topic-panel data/in-topic-C61-725B.json \
  --anchor data/anchor-wikitext.json --eval-set data/eval-wikitext.json \
  --fineweb data/eval-fineweb.json --code data/eval-code.json \
  --minimum-probes 1000 --bootstrap 5000 --out out/C61_CC1_unit_s0.json

/venv/main/bin/python v58/composition_contract.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --a out/R20_full_p0/final.pt --b out/R20_full_p1/final.pt \
  --in-topic-panel data/in-topic-C61-725B.json \
  --anchor data/anchor-wikitext.json --eval-set data/eval-wikitext.json \
  --fineweb data/eval-fineweb.json --code data/eval-code.json \
  --minimum-probes 1000 --bootstrap 5000 --out out/C61_CC1_full_s0.json
```

Before the artifact exists, omit `--in-topic-panel`; its QA/text rows are unavailable and cannot promote. Full-mask overlaps produce a mechanical kill artifact before any GPU model is loaded. Neither A′ nor a substitute unseen repair pair should be supplied.

Only after a separately trained paired seed-1 screen exists:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/composition_contract.py paired \
  --reports out/C61_CC1_unit_s0.json out/C61_CC1_unit_s1.json \
  --bootstrap 5000 --out out/C61_CC1_unit_paired.json

CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/composition_contract.py paired \
  --reports out/C61_CC1_full_s0.json out/C61_CC1_full_s1.json \
  --bootstrap 5000 --out out/C61_CC1_full_paired.json
```

Disjoint slots guarantee mechanical removal only. The ten-surface canaries measure knowledge leakage. Fact/QA upper bounds at the current roster can block promotion even when point-estimate kill lines do not fire. Fresh panel data, additional reviewed concepts for DM gate readiness, stored alone-resident GEN where needed, and an independently trained repair bank C are external artifacts, not numbers fabricated by this change.
