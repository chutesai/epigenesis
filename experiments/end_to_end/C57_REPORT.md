# C57 — DM panel, lossless patch files, and CC1

Implemented in `build_in_topic_panel.py`, `in_topic.py`, `serialize_patch.py`,
and `composition_contract.py`, with CPU tests in `tests/test_c57.py` and revised
DM assertions in `tests/test_c49.py`. `e2e_fsa.py` is unchanged. No git commit
was run. The new files have explicit `.gitignore` exceptions.

Validation uses the real harness and Branches on tiny CPU models, including
the script entry points. Production GPU panels, checkpoint compression ratios,
and CC1 scores have **not** been measured here: the box data/checkpoints are not
present in this checkout. Unrelated README, architecture, and figure changes in
the working tree were preserved.

## 1. Fixed, two-directional DM panel

- Real concepts are the union of the **fixed `ATTRS` pools** selected by
  `in_topic.REAL_ATTRIBUTES`, stripping a leading `the `. Names, fictional
  titles/places, numeric fields, and all `EXTRA_ATTRS` pseudo-values are excluded.
  This rule and the full concept list are stored in the manifest, independent of
  the assigned per-seed world. The built-in reviewed QA covers 24 of these
  concepts: cities, instruments, jobs, and colours, each with a matched
  same-category non-value neighbour. Ten fixed frames give 290 value questions
  and 290 neighbour questions. Gold aliases, categories, pair IDs, counts per
  concept, and prompts are persisted. Generation uses casefolded reviewed aliases
  with C37's clean short-answer scorer; long explanatory answers remain outside
  this short-answer construct.
- Both base-correct **and base-incorrect** rows survive the builder. Evaluation
  reports losses, gains, and net change in both strata, and the value-minus-neighbour
  difference-in-differences with a concept-pair bootstrap. Answers that are pool
  value strings themselves are reported separately and excluded from that estimand.
  Construction and evaluation use the same stored order, backend, batch **1**,
  32 new tokens, and stop limit 2. The JSON has a canonical content hash, plus
  source/tokenizer hashes; loaders verify the content hash and mention masks.
- Calibration includes a pristine zero-patch repeat and uniformly sampled random
  writes into original forced-zero slots. The random control uses the supplied
  saved patch's realized per-linear nnz and original attachment/scaling recipe,
  not its trained positions or signs. Its source hash, nnz, seed, churn, and
  text noise are recorded. No final-panel score selects a training update.
- QA gating requires >=40 matched non-primed concept pairs, exact zero-repeat
  scores, a checked declared census-prefix boundary, and a random-control
  difference-in-differences fitting the registered
  margins, and random-control text SE <=0.001. The equivalence rule is point
  change >=-0.01 **and** the 90% concept-cluster interval's lower bound >=-0.025.
  The built-in 24-pair catalogue is therefore **report-only**, even with 580
  strings. Broader human-reviewed matched rows can be added with
  `--qa-candidates`; each needs `concept`, `pair`, `category`, `stratum`,
  `prompt`, and nonempty `answers`. More paraphrases do not meet the concept
  minimum. Two paired seeds are required for promotion.
- Text comes exclusively from the last **1,000,000 tokens** of
  `data/census/fwhq_sel.u32`. Its complete preceding prefix, the other three
  census streams, anchor, rehearsal, and off-topic panels are scanned against
  every retained window's every 32-gram. The source file can safely also appear
  in `--training-text`: its reference is the prefix, not the held-out tail.
  Matching is case-insensitive, word-bounded, longest multiword first, with the
  fixed polysemy exclusion list recorded in the manifest (including Hobart,
  teal, and periwinkle). Inclusive NLL is reported alongside NLL excluding
  matched tokens **and the following token**; the excluded number gates at
  one-sided upper 95% <=+0.003 and paired SE <=0.001.
- The fixed text target is 256 x 512. If the tail yields fewer disjoint matches,
  >=2 windows are retained with the shortage recorded; insufficient precision
  blocks a gate verdict. The tolerance is equal to off-topic, separately measured.
  Previously written facts come from resident/evaluated banks, using their
  TEST templates and alone-resident GEN references, with fact-cluster loss bounds
  under the common retention clause. A legacy v1 one-directional panel remains
  readable for reporting and cannot pass the QA preservation gate.

The manifest states scope (new launches after 2026-10-13, no retrofit of existing
ladder/SC registrations), a selector-gated forced-bank off-topic reference cap
of +0.01 with selector accuracy reporting, and construction/evaluation passes.
CC1 uses explicit residency and reports forced-bank off-topic damage as advisory.

Run from `/workspace/malleable`, with the updated harness deployed as `v58/`:

```bash
/venv/main/bin/python v58/build_in_topic_panel.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --fineweb data/census/fwhq_sel.u32 \
  --anchor data/anchor-wikitext.json \
  --training-text data/census/cp_wikimedia.u32 data/census/fwhq_sel.u32 \
    data/census/pd_cp_pre1929.u32 data/census/cp_stackexchange.u32 \
    data/rehearsal-fineweb.json \
  --eval-panels data/eval-wikitext.json data/eval-fineweb.json data/eval-code.json \
  --calibration-patch out/GL_CTRL_s1 \
  --context-length 512 --text-windows 256 --batch-size 1 \
  --out data/in-topic-725B.json
```

`--calibration-patch out/GL_CTRL_s1` resolves either
`out/GL_CTRL_s1/final.pt` or the harness default
`out/GL_CTRL_s1_patches/final.pt`. The same resolution is supported by the
serializer. The panel can then be consumed by the existing
`v58/e2e_fsa.py --in-topic-panel data/in-topic-725B.json` and branch evaluator.

**Census range check:** the command reserves the entire pre-tail prefix. Actual
census runs must respect that reservation. C51_REPORT's old A+B prefix is
12,998,656 tokens; if that census was run, additionally pass
`--census-prefix-tokens 12998656`. This checks that the last-1M slice starts after
that prefix and rejects overlap. Such a file must contain at least 13,998,656
tokens. A 13M-token file already consumed by the old full-prefix census cannot
also provide a disjoint last-1M panel; the script exposes that inconsistency
rather than claiming disjointness. The actual box file lengths/current census
cursors were not available locally.

## 2. Actual lossless packed files

Each sparse ternary FSA patch is encoded three ways, counting the entire file:

| Encoding | Position/value payload |
|---|---|
| coordinate | unsigned LEB128 position deltas + packed sign bits |
| block | delta-coded touched pair8 block IDs + 8-bit occupancy masks + packed signs |
| whole-unit | delta-coded unit IDs + dense packed trits, five base-3 digits per byte; up rows/down columns |

Auxiliary tensors are stored with their original dtype, shape, and exact bytes:
alpha, learned scales, gates, unit-reset indices, address starts, flip indices,
and any additional saved tensor state. All other metadata and compatibility
information are preserved recursively. Each candidate is decoded to
`e2e-fsa-patch-v1` and asserted bit-identical before selection; this assertion
includes NaN bit patterns and signed zero. Tensor changes abort serialization.
The shortest whole-file candidate is written, read back from disk, and asserted
again. Checksums detect payload corruption. No model/CUDA is needed.

Every encoding gets its actual byte count, `final.pt` ratio, and keep/kill status.
Savings <2x are marked **kill**, including if that encoding is the shortest;
the measured artifact is still written with `promotable=false`. The report
includes saved `final.pt` bytes, recorded harness estimate, and packed/estimate
ratio. Estimates are read from sibling result JSON, or explicit `--results` in
patch order, including the harness's nested `storage.fsa.bytes` schema. Missing
estimates are explicitly unavailable; they are not fabricated from base shapes.

Exact CPU box commands:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/serialize_patch.py \
  --patches out/GL_CTRL_s1 out/GL_B4M_s1 out/GL_B16M_s1 \
  --out-dir out/C57_packed/GL --out out/C57_serial_GL.json

CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/serialize_patch.py \
  --patches out/R20_unit_p0/final.pt out/R20_unit_p1/final.pt out/R20_unit_p2/final.pt \
  --out-dir out/C57_packed/R20_unit --out out/C57_serial_R20_unit.json

CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/serialize_patch.py \
  --patches out/R20_full_p0/final.pt out/R20_full_p1/final.pt out/R20_full_p2/final.pt \
  --out-dir out/C57_packed/R20_full --out out/C57_serial_R20_full.json
```

Marginal rows report adjacent-patch Delta bytes / Delta distinct facts, both
packed bytes and `final.pt` bytes. These arms generally have equal written-fact
counts, so a zero Delta facts gives `null`, not an infinite efficiency claim.
Without `--retained-facts`, the denominator is explicitly **written facts,
retention not measured**. Supply one independently measured distinct retained-fact
count per patch with `--retained-facts` to compare retained-fact efficiency.
Paraphrase accuracy is not converted into a distinct retained-fact count.

The packed artifact is directly reloadable with
`serialize_patch.decode_patch(Path(packed_path).read_bytes())`, followed by
`e2e_fsa.load_patch(model, state)` on a compatible attached model. The CPU test
does that real harness load for all three encodings and compares effective weights.

## 3. CC1 configurations and decisions

The driver validates the R20 world, export/step/backend, MoE placement,
scale identity, linear sets, and p0/p1 ownership. Whole-unit A/B also require
K=3 partitions, distinct parts, disjoint writes, and runtime verification that
each saved write is inside its declared original-export partition. Full-mask
is a separate labelled family. The permitted full/block p0 replacement is
labelled by its actual partition type.

It scores base, A, B, AB, AB-A, AB-B, AprimeB, and BA. Each configuration builds
a fresh model. Every activation clears `Branches.cache` and its active-key shortcut.
Unload/replacement use the harness's subset activation to reconstruct the
remaining saved codes; effective weights are hashed across **all linears**, and
three fresh probe forwards are compared byte-for-byte to independent directly
loaded configurations. AprimeB has an additional fresh direct reference model.
This tests the actual Branches serving semantics. Per-forward recurrent state
is fresh; effective weight subtraction is not the serving operator.

Every row scores each bank's full direct and unseen panels and target-owned
sister-chain probes, both-bank cross-person conjunctions/chains (filtered so no
third-person target enters that panel), conflicting identifiers, >=1,000 unique
never-written/absent-attribute requests, >=1,000 canary strings per bank, and
DM/off-topic text. Canaries include all direct registered phrasings, unseen,
the two conversation-pronoun families, C34 pronoun families, and reverse lookups.
Extra request labels meet the string minimum in tiny worlds and are explicitly
recorded; they never increase the fact denominator. Sister names and values
shared with another bank are excluded from value canaries. Full per-item artifacts
retain prompts, fact IDs, generated text, and emissions. Confusion scoring retains
the complete three-person world, including the unloaded person's competing values.

Retention uses equal underlying fact clusters, with a one-sided 95% paired
bootstrap and a conservative exact any-fact-loss envelope, so an all-retained
small panel cannot manufacture a zero uncertainty bound. Canary leakage means
**any surface** emitted the fact's value and uses a one-sided exact 95% bound
on facts, not strings. Zero leakage at 100 facts still has upper bound 2.95%;
it does not certify a 1% contract. Observed >1-point retention loss or >1% fact
leakage kills; an upper-bound-only excess blocks promotion under the design's
inconclusive-interval clause. Mechanical mismatches kill, and a failed positive
control invalidates the run. Replacement also tests retention of B. In-topic
text uses its excluded-token upper bound; unresolved/report-only QA blocks
promotion. No single-seed report can promote.

The mandatory note appears in every artifact: disjoint slots guarantee
**MECHANICAL removal, not KNOWLEDGE removal**; J/F joint gradients can teach B
information from A, which canaries measure.

Exact box commands, retaining the authoritative design's permitted replacements:

```bash
/venv/main/bin/python v58/composition_contract.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --a out/R20_unit_p0/final.pt --b out/R20_unit_p1/final.pt \
  --a-prime out/R20_full_p0/final.pt \
  --in-topic-panel data/in-topic-725B.json \
  --anchor data/anchor-wikitext.json --eval-set data/eval-wikitext.json \
  --fineweb data/eval-fineweb.json --code data/eval-code.json \
  --qa-n 120 --minimum-probes 1000 --bootstrap 5000 \
  --out out/CC1_unit_s0.json

/venv/main/bin/python v58/composition_contract.py \
  --export /workspace/malleable/data/hf/exports/725B-tokens_step58814 \
  --export-step 58814 --eda-backend naive_fp32 \
  --a out/R20_full_p0/final.pt --b out/R20_full_p1/final.pt \
  --a-prime out/R20_block_p0/final.pt \
  --in-topic-panel data/in-topic-725B.json \
  --anchor data/anchor-wikitext.json --eval-set data/eval-wikitext.json \
  --fineweb data/eval-fineweb.json --code data/eval-code.json \
  --qa-n 120 --minimum-probes 1000 --bootstrap 5000 \
  --out out/CC1_full_s0.json
```

Console output has one compact configuration table; JSON also contains full
scores, clustered bounds, mechanics, ownership/mask identities, and input hashes.
The design's 8–16 GPU-hour screen budget is unverified; this implementation adds
the fresh direct replacement reference and scores the full canary census.

After genuine paired seed-1 screens exist, aggregate without a GPU:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/composition_contract.py paired \
  --reports out/CC1_unit_s0.json out/CC1_unit_s1.json \
  --bootstrap 5000 --out out/CC1_unit_paired.json

CUDA_VISIBLE_DEVICES='' /venv/main/bin/python v58/composition_contract.py paired \
  --reports out/CC1_full_s0.json out/CC1_full_s1.json \
  --bootstrap 5000 --out out/CC1_full_paired.json
```

Aggregation checks seeds 0/1, different trained A/B file hashes, family,
training identity, panel hash, and replacement labels. Facts are clustered
within seed with equal cluster counts; fixed concepts and windows are averaged
across seeds before uncertainty is measured. Every mechanical/control check must
pass in both seeds.

For J/F patches, C52 metadata automatically marks the run repaired; `--repaired`
also supports other repair formats. Promotion additionally needs paired unseen
screens supplied to the aggregator with `--unseen-pair-reports`. Each screen pins
`--repair-training-pairings`, a frozen JSON list of SHA256 pairs actually used
in repair, and the aggregator verifies that the tested pair is absent and differs
from the primary pair. Without this evidence, repaired pairs remain blocked.

## Tests

Local command (preinstalled CPU environment):

```bash
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q \
  experiments/end_to_end/tests/test_c57.py \
  experiments/end_to_end/tests/test_c49.py \
  experiments/end_to_end/tests/test_c52.py \
  experiments/end_to_end/tests/test_branch_eval.py \
  experiments/end_to_end/tests/test_c34.py
```

The same checks on the box, from `/workspace/malleable`:

```bash
CUDA_VISIBLE_DEVICES='' /venv/main/bin/python -m pytest -q \
  v58/tests/test_c57.py v58/tests/test_c49.py v58/tests/test_c52.py \
  v58/tests/test_branch_eval.py v58/tests/test_c34.py
```

Coverage includes all three encodings, empty entries, both projection axes,
signed zero/NaN preservation, checksum failures, disk length selection and
<2x rejection, actual harness reload, marginal zero-denominator handling,
two-directional QA, matched-concept clustering, tail/prefix disjointness and
declared-census overlap rejection, excluded mention/following-token damage,
the random forced-zero control, real Branches mechanical checks on nine fresh
CPU models, singleton/tiny-panel uncertainty, main CLI JSON artifacts, paired
aggregation, and frozen-panel tamper rejection. Existing DM/resident/joint and
C34/branch evaluations are exercised as regression checks.

Final result: **121 passed in 41.37s**. Python compilation and scoped
`git diff --check` also passed; `e2e_fsa.py` has no diff.
