# C41: statistical Bop and cross-projection unit bursts

Implemented opt-in write refinements in `e2e_fsa.py`, `c41_birth_match.py`, and `tests/test_c41.py`. No model-scale experiments were run. No commit was made. Existing workspace changes were preserved.

## Diagnosis first

Read Muse §1–§2 fully and wrote `reviews/novel_ideas_2026-10-12/ideas3_diagnosis.md` before editing the optimizer. It contains all TRAJ cumulative-birth curves, validation GEN, major-wave counts for fully logged runs, sampled-wave lower bounds for older runs, and birth-event median proxies.

The proposed latency explanation is not established:

- All R14 failures birth their entire 40k/80k support at update 2. By 40, 40k arms have 45,791/45,484 cumulative births and 0% test GEN; 80k arms have 93,815/91,055 births and 0%/5% test GEN. They have one/two major waves respectively.
- With no live evictions, deaths cannot remove enough update-2 codes to move the final surviving last-birth median: **it is exactly 2 in all four failed screens**.
- The one-fact control reaches 100% validation GEN at 5, before its later waves. R21 long reaches 70% at 15 and 100% at 20 after its second wave; it has a much larger support and turnover. Its log ends at 32, so its advertised 200-step horizon is unfinished.
- R6 has near-perfect person-conditioned validation GEN by the first sample at 50, despite millions of later births. R4's large recall gain from 200 to 300 accompanies only 8.6% more cumulative births. Neither total births nor wave count alone explains these observations. Old R6/R4 logs cannot measure surviving birth-step medians.

## Implementation choices and logging

`--bop-tstat Z` adds a second-moment EMA at the same gamma. Actual per-slot sums of EMA weights and squared weights give bias correction and `n_eff`, including gamma switches, missing-gradient zero observations, resets, sparse coordinates, and support evictions. Birth requires `abs(m_hat) > floor` and `abs(m_hat) > Z * SE`; death mirrors the rule against the live sign. At least two effective observations are required (`n_eff > 1`), because a single observation has zero estimated variance and would otherwise admit any spike. With gamma=1, effective sample size never exceeds one. `--bop-tstat-floor F` is an **absolute gradient floor**, defaulting to `0.3 * calibrated tau`; existing row/occupied-slot threshold multipliers and death multipliers also scale the floor. Grace, repair, support, promotion checks, and keep-EMA remain active. Statistical arms are separate from pair-birth and unit-burst.

`--unit-burst M --unit-burst-thr T` uses the hidden-unit coordinates of an expert's up rows and down columns. Its pooled score is `sqrt(sum(up net_gain) * sum(down net_gain)) / tau`, where `net_gain = max(abs(EMA) - cost_lambda * slot_cost, 0)`. `slot_cost` is the existing row-scale-squared times input activation energy, divided by frozen-base output energy. In the netcost recipe this is the existing signed-score cost convention; the calibrated cost lambda is held as in ordinary support ranking. The default unit threshold is 1 in tau units. A crossing writes top-M **vacant, writable, supported, nonzero-EMA** slots on each side, using their own descent signs. Units need M eligible slots on both sides; there are no partial or one-sided bursts. Each unit bursts once, then uses ordinary Bop. Bursts do not allocate support, change the budget, or bypass freeze/keep-live.

`--unit-burst-random` chooses the same number of eligible units and the same M+M sizes as that update's evidence policy. For a **separate training arm**, use `--unit-burst-replay evidence_result.json`: changed gradients would otherwise change the schedule. Replay preserves per-expert counts and sizes at every applied wall update. If the random arm cannot supply enough eligible units, it fails explicitly; that run is not a matched control. Its dedicated seeded generator does not consume the training RNG.

BOP records `unit_bursts` (expert, count, M on each side, unit IDs), `unit_burst_births`, and `unit_burst_additional_births`. Ordinary births and bursts can overlap; overall code births count their union once. A zero additional-birth count is a useful no-op finding, especially given R14's initial support saturation. Distinct units per global fact count units born while that fact supplied a nonzero answer-loss slot gradient in the update/accumulation window. Both burst arms use one-example microbatches for this attribution; this is gradient evidence, not exclusive fact ownership. Their actual forward/backward overhead must be included in cost comparisons.

`--write-diagnostics` enables int32 per-slot last-birth wall updates, independent of `--no-churn`; tstat/burst enable it automatically. TRAJ and FINAL include the supported-live histogram, known count, unknown/imported count, and median of known birth steps. Death/eviction clears a timestamp; rebirth replaces it. The tensors are training metadata and do not change ternary bank serialization or revoke behavior. Diagnostics also print cumulative code births on every BOP line and total births by step 40 in FINAL/RESULT.

The birth-match helper accepts RESULT JSON or BOP logs, rejects incomplete/duplicate horizons, excludes REFRESH support additions, checks available seed/budget/recipe identity, ranks **measured** standard-Bop taus by relative birth-count error, and can emit a tau only within a chosen tolerance. It does not assume births are monotonic in tau, invent an unmeasured tau, or use GEN to select a control.

## Validation

Validation command:

```bash
/private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q experiments/end_to_end/tests
```

C41 alone: **36 passed**. Full end-to-end suite: **1,025 passed, 1 skipped** in 96.15 seconds. `git diff --check` and `bash -n` on every suggested command block also passed. Tests cover consistent sub-tau births versus ordinary magnitude Bop, absolute floors, spike suppression, bias correction and gamma switches, mirrored death/grace/repair, legal occupied flips, dense/sparse support, birth/rebirth/eviction histograms, atomic bursts, top-M signs, activation-energy costs, overlap with ordinary births, random controls with changed evidence, unmatchable replay, measured birth matching, and production-loop TRAJ/FINAL logging with `--no-churn`. Four frozen pre-C41 byte digests verify dense/sparse Bop state, birth/death histories and churn under both reset and retained EMAs. The new flags default off; no statistical/timestamp tensors or random draws are introduced into ordinary runs.

## Exact suggested CUDA-host commands

Run from the repository root on the host with the 725B-token export. These commands reconstruct R14 from its RESULT metadata and the corrected commands in `reviews/brainstorm_2026-10-12/tasks/c31_out.md`: layers 12,14,17,20; all 12 formats; full fact batch; tau .5; gamma .05; sparse half-slot reserve; netcost support; local lambda 10; no rehearsal. The supplied logs lack literal shell command headers. The 40k control freezes at 10; the 80k control freezes at 15; both keep live codes. Do not import R21's different support/turnover policy into these comparisons.

```bash
set -euo pipefail
c41_data=/workspace/malleable/data
c41_logs=reviews/novel_ideas_2026-10-12/ideas3_logs
mkdir -p out
c41_common=(
  --export "$c41_data/hf/exports/725B-tokens_step58814" --export-step 58814
  --method fsa_forced --fmt ternary --optim bop --layers 12,14,17,20
  --people 1 --shared-attrs 0 --values disjoint
  --answer-only --aug --aug-all --bop-gamma .05 --bop-tau .5
  --support-score netcost --cost-rel 1 --local-lambda 10 --rehearsal-frac 0
  --sparse-state --slot-frac .5 --fast-moe exact --expert-checkpoint
  --keep-live --refresh-every 10 --no-churn --no-pareto --write-diagnostics
  --eval-every 5 --qa-probe --qa-n 10 --zero-patch-control
  --anchor "$c41_data/anchor-wikitext.json"
  --eval-set "$c41_data/eval-wikitext.json"
  --extra-eval-sets "$c41_data/eval-fineweb.json,$c41_data/eval-code.json"
)

# Arguments: facts horizon budget seed freeze arm micro [extra flags]
c41_run() {
  local c41_f=$1 c41_h=$2 c41_k=$3 c41_s=$4 c41_fr=$5 c41_arm=$6 c41_micro=$7
  shift 7
  local c41_tag="out/c41_f${c41_f}_h${c41_h}_k${c41_k}_s${c41_s}_${c41_arm}"
  python experiments/end_to_end/e2e_fsa.py "${c41_common[@]}" \
    --facts-per-person "$c41_f" --fact-batch "$c41_f" --val-facts "$c41_f" \
    --steps "$c41_h" --budget "$c41_k" --seed "$c41_s" --budget-freeze "$c41_fr" \
    --micro "$c41_micro" --out "${c41_tag}.json" --patch-dir "${c41_tag}_patches" \
    "$@" | tee "${c41_tag}.log"
}

# 10 facts / 40 steps: both budgets and seeds 0,1,2.
for c41_k in 40000 80000; do
  if [[ "$c41_k" == 40000 ]]; then c41_fr=10; c41_letter=D; else c41_fr=15; c41_letter=B; fi
  for c41_s in 0 1 2; do
    c41_prefix="out/c41_f10_h40_k${c41_k}_s${c41_s}"
    # Standard-reuse arm: seeds 0/1 already exist; seed 2 is missing.
    if [[ "$c41_s" -lt 2 ]]; then
      cp "$c41_logs/R14_t10_${c41_letter}_s${c41_s}.log" "${c41_prefix}_standard.log"
      c41_standard="${c41_prefix}_standard.log"
    else
      c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" standard 4
      c41_standard="${c41_prefix}_standard.json"
    fi
    c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" tstat 4 --bop-tstat 3

    # Calibration candidates: select by births only, independently per budget/seed.
    c41_candidates=("$c41_standard")
    for c41_tau in .25 .125 .0625 .03125; do
      c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" "calib_tau${c41_tau}" 4 --bop-tau "$c41_tau"
      c41_candidates+=("${c41_prefix}_calib_tau${c41_tau}.json")
    done
    python experiments/end_to_end/c41_birth_match.py \
      --target "${c41_prefix}_tstat.json" --candidates "${c41_candidates[@]}" --horizon 40 \
      > "${c41_prefix}_birth_match.json"
    c41_matched_tau=$(python experiments/end_to_end/c41_birth_match.py \
      --target "${c41_prefix}_tstat.json" --candidates "${c41_candidates[@]}" \
      --horizon 40 --tolerance .05 --emit-tau)
    c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" birth_matched 4 --bop-tau "$c41_matched_tau"

    c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" burst 1 --unit-burst 8 --unit-burst-thr 1
    c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" random_burst 1 \
      --unit-burst 8 --unit-burst-thr 1 --unit-burst-random \
      --unit-burst-replay "${c41_prefix}_burst.json"
    # Additional control for comparing bursts against standard with identical microbatch size.
    c41_run 10 40 "$c41_k" "$c41_s" "$c41_fr" standard_micro1 1
  done
done
```

The reused standard logs lack new survivor instrumentation. Tstat and birth-matched arms retain the reconstructed micro=4 recipe. Burst versus random has identical micro=1, and `standard_micro1` supplies batching parity for a claim about bursts versus ordinary Bop. The extra control does not replace the historical standard-reuse arm. If calibration finds no tau within 5%, refine the measured tau grid or report that no matched control exists; do not call the nearest unmatched arm matched. Saturated supports can make a lower tau ineffective.

40-fact/100-step horizon: use 160k and 320k budgets to retain the screen's 4k/8k slots per fact. This is a stated experiment design, not an existing measured baseline. Use three seeds and a common micro=1 across all arms; freeze at 40 for both budgets to permit four support refreshes, then compare optimization on fixed support. The burst versus random pair is the required test; the other three arms extend the statistical comparison to this horizon.

```bash
# Uses c41_common and c41_run defined above.
for c41_k in 160000 320000; do
  for c41_s in 0 1 2; do
    c41_prefix="out/c41_f40_h100_k${c41_k}_s${c41_s}"
    c41_run 40 100 "$c41_k" "$c41_s" 40 standard 1
    c41_run 40 100 "$c41_k" "$c41_s" 40 tstat 1 --bop-tstat 3
    c41_candidates=("${c41_prefix}_standard.json")
    for c41_tau in .25 .125 .0625 .03125; do
      c41_run 40 100 "$c41_k" "$c41_s" 40 "calib_tau${c41_tau}" 1 --bop-tau "$c41_tau"
      c41_candidates+=("${c41_prefix}_calib_tau${c41_tau}.json")
    done
    # Match this horizon at 100, while each RESULT also reports births through 40.
    python experiments/end_to_end/c41_birth_match.py \
      --target "${c41_prefix}_tstat.json" --candidates "${c41_candidates[@]}" --horizon 100 \
      > "${c41_prefix}_birth_match.json"
    c41_matched_tau=$(python experiments/end_to_end/c41_birth_match.py \
      --target "${c41_prefix}_tstat.json" --candidates "${c41_candidates[@]}" \
      --horizon 100 --tolerance .05 --emit-tau)
    c41_run 40 100 "$c41_k" "$c41_s" 40 birth_matched 1 --bop-tau "$c41_matched_tau"
    c41_run 40 100 "$c41_k" "$c41_s" 40 burst 1 --unit-burst 8 --unit-burst-thr 1
    c41_run 40 100 "$c41_k" "$c41_s" 40 random_burst 1 \
      --unit-burst 8 --unit-burst-thr 1 --unit-burst-random \
      --unit-burst-replay "${c41_prefix}_burst.json"
  done
done

# One seed continued to 300 with the same freeze/recipe from the start.
# Fresh 300-step runs preserve their full EMA and once-per-unit history. Loading only a
# 100-step bank would omit optimizer evidence, birth timestamps and burst history.
c41_run 40 300 320000 0 40 finish_standard 1
c41_run 40 300 320000 0 40 finish_burst 1 --unit-burst 8 --unit-burst-thr 1
c41_run 40 300 320000 0 40 finish_random_burst 1 \
  --unit-burst 8 --unit-burst-thr 1 --unit-burst-random \
  --unit-burst-replay out/c41_f40_h300_k320000_s0_finish_burst.json
```

Report held-out test GEN at the declared horizon (not only validation GEN), all three paired damage panels, live slots, cumulative and per-update code births, support-refresh curves, survivor histograms, burst counts/additional births, per-fact distinct units, and actual compute. The existing one-fact positive control already succeeds; repeat its recipe if the environment/model changes. Muse's decision bars remain: tstat at least 50% screen GEN and at least 10 points over a truly birth-matched control; unit-burst at least 10 points over random at the truncated horizons, with no more than 3 points lost to standard at the 300-step finish. No improvement is claimed from CPU mechanism tests alone.
