# C42 — calibrated selectors and rescued hierarchy

Implemented opt-in C42 modes in `branch_eval.py`, with CPU analysis in
`c42_selector.py` and GPU extraction/serving in `c42_runtime.py`. Existing selector
modes and defaults are unchanged. This task did not edit `e2e_fsa.py` or run
`git commit`; existing and concurrent working-tree changes were preserved.

Each bank fits a balanced binary logistic map on CAL scores, with balanced
normalization as well as class weights. Equal class priors make its output an
estimated LLR rather than population-prior posterior odds. One positive score
uses two coefficients; adding a cached inhibitory score uses three. JSON ledgers
store raw-score coefficients per bank, and selector revocation removes the key
and its coefficients. Calibration is regularized and approximate. Ordinary
queries fire exactly one bank iff its LLR is strictly positive and its margin
strictly exceeds the VAL-selected delta. Registered multi-bank types can add
other positive-LLR banks after that gate; unregistered types remain single-bank.
The supplied study and GEN commands evaluate single-answer questions.

TRAIN fits centered cosine keys. CAL contains separate name, pronoun, and indirect
queries plus other banks' queries, independent text, and null queries. VAL alone
sets cutoffs, layers and the winning hierarchy. TEST is locked. Global cosine
and per-bank z-score arms use the smallest empirical VAL cutoff satisfying the
query false-fire cap of 0.001. Z-score moments use each bank's CAL negatives,
including other banks. LLR retains the absolute zero null and tunes its margin.
Reports give routing recall, exact route accuracy, false-fire rates with exact
one-sided 95% bounds, abstention, and separate forms/negative categories. TEST
cap violations and fewer than 3,000 negatives prevent a matched-rate keep verdict;
an empirical cutoff is not a population-rate guarantee.

Hierarchy arms are flat LLR, current log-affinity hierarchy, sibling-trained LLR
beam, and the same beam with flat rescue. Parent keys/calibration include global
nulls. Child multinomial keys and their calibration use only sibling queries
under their parent, with no general anchor or queries from other parents.
Paths rank by parent LLR plus conditional child LLR. The parent beam permits a
strong child to compensate a weaker parent; the final path uses zero and margin
nulls. Rescue fires when the best path is nonpositive or lacks the required
margin, and pays all N flat comparisons even when flat then abstains. Rescue
confidence choices use a fixed 17-quantile VAL grid plus zero and the no-rescue
cutoff. Validation path scores are reused across these choices.

The current hierarchy control preserves centered centroid children and summed
`log((cosine+1)/2)` rankings, with a predeclared cosine threshold grid and final
beam cap one for the study's single-answer queries. Parent beam width defaults
to two. The child feature layer is shared with the VAL-selected flat arm;
parent layers are searched on VAL. The winning hierarchy is chosen on VAL
accuracy, then lower false-fire and lower cost, before TEST evaluation or GEN.

Comparison counts include every parent and every visited child's keys. Reports
express the exact average as `depth * effective_branching + rescue_rate * N`:
effective branching includes the beam's visited children. At 3 parents, 4 children
and a two-parent beam, the base cost is 3 + 2×4 = 11, not 8. Positive and negative
costs/rescue rates are also reported. The aggregate rescue kill rule uses the
actual TEST query mixture; null-heavy evaluation can make rescue frequent
because a nonpositive path explicitly requests flat rescue.

Simulated-100 partitions each real bank's fact IDs into nonempty sub-banks;
all phrasings of a fact keep the same owner across TRAIN/CAL/VAL/TEST. Twelve
real banks yield twelve parents with eight or nine children, rather than
inventing patches or claiming a 10×10 trained layout. Insufficient facts produce
an explicit unavailable result. Real-bank inhibitory scores cannot be relabeled
as sub-bank scores; that simulation requires corresponding sub-bank scores.
Scaling output compares `1 - product(1 - p_b)` from per-bank VAL qualification
rates with measured TEST query firing and its interval. It also exposes raw
LLR-positive evidence rates/union firing, and extrapolates the real-12 VAL rates
to 100 when both caches are supplied. Margin-qualified events are mutually
exclusive, so independence is an approximation; the point prediction does not
include uncertainty in estimated VAL rates.

JSON encodes the pre-registered rules: LLR must improve real-12 recall by at
least five percentage points at the matched false-fire cap; hierarchy is killed
if its VAL-selected winner loses over two points of real-12 TEST route accuracy,
or simulated-100 rescue exceeds 40%. Scaling mismatches are also recorded.
Insufficient evidence is reported as unresolved, not a successful experiment.

Feature dump collects every requested full-block input together on the frozen
base, forwarding each query once. It saves CPU features, TRAIN anchor features,
queries, labels, fact ownership, layers, model identity and checkpoint SHA-256s.
CAL/VAL/TEST general windows are disjoint, deduplicated, and exclude preservation
windows and the TRAIN anchor. Null prompts cover untaught entities, untaught
attributes, and explicitly restricted wrong scopes; ordinary questions about
other resident facts are positives. Token-identical cross-split prompts fail
validation. GEN validates cache/checkpoint/export identity, refits selectors on
CPU, then serves the locked flat and best hierarchical choices through real
patch loading. Routes are fixed for the complete generated answer. Standalone,
empty-bank, pristine-base weight and bit-exact revocation controls are checked.

Local validation: **15 C42 tests passed; complete end-to-end suite: 1,025 passed,
1 skipped.** The first full run had a source-inspection failure while another
task changed `e2e_fsa.py`; it passed in isolation and in the fresh full rerun.
Coverage includes one-/two-score calibration,
known density ratios, strict margin/multi-bank gates, sibling keys, exact rescue
cost, TEST independence, simulated-100 scaling, and cache leakage checks. A tiny
three-bank integration test writes fixture patches, captures all layers once
(using synthetic residuals), runs real composed GEN, checks serving/revocation,
and rejects a stale cache. These are mechanics checks, not LM-quality evidence.

Exact local commands:

```bash
C42_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C42_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests/test_c42.py
OMP_NUM_THREADS=1 "$C42_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests
OMP_NUM_THREADS=1 "$C42_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests/test_e2e_fsa.py::test_sparse_sgd_rms_denominator_uses_full_shapes
git diff --check
```

A standalone CPU CLI smoke run also exercised 3, 12 and simulated-100 fixtures:

The separable three-/twelve-bank fixture achieved 100% flat LLR routing recall
and zero false fires on its 64 toy nulls. Simulated-100 reused bank-level features
across fact splits, so it intentionally lacked sub-bank separation: flat LLR
recall was 16.7%, and rescue was 78.3%. The JSON correctly killed hierarchy for
that rescue rate and left the real-12 LLR recall rule unresolved because toy
negative samples were insufficient. These fixture outcomes do not decide the
pre-registered real-model experiment.

```bash
OMP_NUM_THREADS=1 "$C42_PYTHON" - <<'PY'
import sys
from pathlib import Path
import torch
sys.path.insert(0, 'experiments/end_to_end/tests')
from test_c42 import fixture
root = Path('/tmp/c42-smoke')
root.mkdir(exist_ok=True)
for n in (3, 12):
    torch.save(fixture(n, 10), root / f'features{n}.pt')
PY
OMP_NUM_THREADS=1 "$C42_PYTHON" experiments/end_to_end/c42_selector.py \
  --caches /tmp/c42-smoke/features3.pt /tmp/c42-smoke/features12.pt \
  --simulate-100 --out /tmp/c42-smoke/analysis.json
```

**Full-model recall/false-fire, hierarchy kill outcomes, and composed GEN remain
unmeasured.** This workspace lacks the box's checkpoints, external model loader
and GPU runtime. Use the following exact commands from the model-box experiment
root containing `v54/`, `out/`, and `data/`. Deploy the changed `branch_eval.py`
and the new `c42_selector.py`/`c42_runtime.py` alongside the existing harness
modules in `v54/`. Supply the original disjoint TRAIN anchor and, if the three
panel files lack 9,000 unique post-preservation windows, an additional general
text token JSON. No LM training command is required.

```bash
C42_ANCHOR=data/anchor-wikitext.json
C42_NULLS=data/c42-general-tokens.json
# C42_NULLS must be an existing independent corpus, not resampled panel windows.
# Each of CAL/VAL/TEST requires >=3,000 unique general windows; TRAIN anchor >=6,080 tokens.
C42_P3=(out/R6_v4_full_p{0,1,2}/final.pt)
C42_P12=(out/R16_grp_p{0,1,2}_g{0..3}/final.pt)
C42_COMMON=(--anchor "$C42_ANCHOR"
  --eval-set data/eval-wikitext.json --fineweb data/eval-fineweb.json
  --code data/eval-code.json --negative-text "$C42_NULLS"
  --negative-n 3000 --c42-layers 4,26,28 --fast-moe exact)

OMP_NUM_THREADS=1 python v54/branch_eval.py --c42-mode dump \
  --patches "${C42_P3[@]}" "${C42_COMMON[@]}" \
  --c42-cache out/C42_features3.pt --out out/C42_dump3.json
OMP_NUM_THREADS=1 python v54/branch_eval.py --c42-mode dump \
  --patches "${C42_P12[@]}" "${C42_COMMON[@]}" \
  --c42-cache out/C42_features12.pt --out out/C42_dump12.json

# Fast CPU analysis: frozen features only, no model loader or GPU.
OMP_NUM_THREADS=1 python v54/c42_selector.py \
  --caches out/C42_features3.pt out/C42_features12.pt \
  --simulate-100 --beam 2 --out out/C42_analysis.json

# GPU selection-in-the-loop GEN on the twelve real banks.
OMP_NUM_THREADS=1 python v54/branch_eval.py --c42-mode gen \
  --patches "${C42_P12[@]}" --c42-cache out/C42_features12.pt \
  --h-top-k1 2 --fast-moe exact --gen-bs 16 --gen-max-new 64 \
  --out out/C42_gen12.json
```

The model/export defaults must match saved checkpoint metadata; explicit
`--export`/`--export-step` overrides must be identical for dump and GEN. The
additional null-corpus path above is an input to provide on the box, not a
claimed artifact created here.
