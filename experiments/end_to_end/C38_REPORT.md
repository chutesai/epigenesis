# C38: C35 design-review fixes

Built on the completed C37 tree without reverting its experiments. No commit.
The referenced review ends at §1f; the additional requirements are taken from
`reviews/brainstorm_2026-10-12/tasks/c38_brief.md` and its overall-verdict table.

| Review item | Fix and evidence |
|---|---|
| 1.1 dense residents | Dense floating-point LoRA occupancy above 50% of any target raises unless `--allow-dense-resident`; `RESIDENT-SLOTS` prints each target's count and fraction. |
| 1.2 revoke gate | Resident runs independently compute B-only weights from pristine originals and the student parametrization, use `torch.equal` against revoked weights, exercise A on/off over several prompts, and hard-fail on any discrepancy. Final all-off weights and several logits are checked against pre-attach base snapshots. |
| 1.3 circular branch baseline | `branch_eval` captures three logits probes and target weight hashes before attaching anything. It checks both attachment and final revocation against that pristine snapshot, even with a cached scoring baseline. A test mutates the base during bank construction and requires failure. |
| 1.4 bf16 evidence | Test trains and exports B in bf16, loads it alone on a separate pristine model, enables A and revokes it, and requires identical effective weights and logits for three distinct prompts. |
| 1.5 capacity control | `--resident-mask-only` excludes the same actual coordinates as resident-on training, but enforces zero resident delta in every forward. The analysis tool reports revoked-minus-masked dependence, masked-minus-full capacity cost, all three final nnz counts, and an explicit nnz mismatch marker. |
| 1.A original recall | Every resident's original-world recall disables the other residents; a two-resident test makes joint and isolated measurements differ. |
| 1.B auxiliary base contamination | Resident and mask-only runs reject route-KL, context teacher/check, netcost/epoch/Fisher/energy, address/unit, gate/expert, wake/fold/MLS/drift and sibling recipes. Parameterized tests cover named auxiliary paths. |
| 2 factorial | A private anchor generator gives identical local/rehearsal window draws across all four cells despite different initialization RNG consumption. Conflicting explicit recipe flags raise. `--factorial-lrs` trains the same candidate grid with equal updates in each cell and chooses by final validation GEN, with stable tie-breaking. Ternary LoRA optimizes dimensionless codes and uses the same base-row alpha threshold/gradient units as FSA. Saved entries record the coordinate convention; legacy entries retain their original interpretation. |
| 2 uncertainty/capacity | Bootstrap runs retain fact-level train TF and test GEN observations, compute a paired generalization-gap CI, and support between-cell, gap and factorial-interaction CIs. Selected results record actual nnz and bytes. Comparisons are explicitly format-native; they do not claim matched nnz or storage capacity. The analysis tool refuses mismatched candidate grids, updates, data, recipe or anchor draws. |
| 3 LoRA banks | Ternary resident alpha must agree with the actual pristine base. The solo control temporarily rebuilds a standalone training parametrization, then compares its weight bits and generation against bank assembly; a deliberately corrupted sum fails the control. `--merge sum_delta` adds both FSA and LoRA saved deltas in fp32 followed by one cast; default clipping/product behavior is labelled format-native. |
| 4 curriculum | Auxiliary paired tasks classify lexical relations between stored values; inverse tasks verify/reject identity assertions; null tasks classify covered/missing entries. These operations and categorical answer formats differ semantically from evaluation's two-value/entity/abstention QA. They do not train evaluation answer pairs, even when the underlying stored facts overlap. Direct facts retain exactly the direct arm's augmentation, RNG draws and example counts; `--answer-only` is honored. Auxiliary examples and increased token budget are recorded separately. |
| 5 synthetic/dose | Synthetic values are globally unique seeded random 16-letter pseudo-words from a private RNG, without seed/person/attribute indices. Resident fact counts and training exposures, occupied-union slots, fixed B fact count/updates/exposures, final B nnz/bytes and loss tokens are recorded. Dose analysis rejects changing B's world or optimization/exposure budget. |

CPU verification uses the repository's Tiny model, including real training,
save/load, bank assembly, damage panels and revocation paths. Full-model/GPU
experiments have not been run; this report contains no empirical full-model claim.

Validation: **945 passed, 1 skipped** in the full CPU suite (one existing
tensor-to-scalar warning). The strengthened nonzero-B bf16 assertion also
passes separately. Python syntax, documented Bash syntax and `git diff --check`
pass.

```bash
C38_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C38_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests
git diff --check
```

The following protocol replaces the confounded C35 command blocks. Set the
token paths to three disjoint files and use the usual external GPU model loader.
Repeat with seeds 1 and 2. Equal search budget does not make the formats equal
in storage or final nnz; those quantities must accompany every recall contrast.

```bash
C38_PYTHON=python3
C38_TRAIN=experiments/end_to_end/e2e_fsa.py
C38_EVAL=experiments/end_to_end/branch_eval.py
C38_ANALYSIS=experiments/end_to_end/c38_analysis.py
C38_ANCHOR=/absolute/path/to/anchor_tokens.json
C38_WIKI=/absolute/path/to/wikitext_tokens.json
C38_FINEWEB=/absolute/path/to/fineweb_tokens.json
C38_CODE=/absolute/path/to/code_tokens.json
C38_COMMON=(--layers 12,14,17,20 --seed 0 --steps 300
  --local-lambda 10 --answer-only --aug --fact-batch 8 --micro 2
  --anchor "$C38_ANCHOR" --eval-set "$C38_WIKI"
  --extra-eval-sets "$C38_FINEWEB,$C38_CODE" --eval-every 50
  --qa-probe --qa-n 120 --bootstrap 1000 --no-pareto)
C38_WORLD=(--people 3 --facts-per-person 100 --shared-attrs 100 --values disjoint)

# Equal per-cell search: five candidates, 300 updates each, validation selection.
for cell in sparse_tern sparse_fp dense_tern dense_fp; do
  "$C38_PYTHON" "$C38_TRAIN" "${C38_COMMON[@]}" "${C38_WORLD[@]}" \
    --factorial-cell "$cell" --factorial-lrs .001 .005 .02 .1 .5 \
    --patch-dir "results/c38/$cell" --out "results/c38/$cell.json"
done
"$C38_PYTHON" "$C38_ANALYSIS" factorial results/c38/{sparse_tern,sparse_fp,dense_tern,dense_fp}.json \
  --out results/c38/factorial_comparison.json

# Sequential: same fixed LR and B training budget in all three B arms.
"$C38_PYTHON" "$C38_TRAIN" "${C38_COMMON[@]}" "${C38_WORLD[@]}" \
  --factorial-cell sparse_tern --lr .02 --train-person 0 \
  --patch-dir results/c38/A --out results/c38/A.json
for arm in resident masked full; do
  C38_RESIDENT=()
  if [ "$arm" = resident ]; then C38_RESIDENT=(--resident-patch results/c38/A/final.pt); fi
  if [ "$arm" = masked ]; then C38_RESIDENT=(--resident-mask-only results/c38/A/final.pt); fi
  "$C38_PYTHON" "$C38_TRAIN" "${C38_COMMON[@]}" "${C38_WORLD[@]}" \
    --factorial-cell sparse_tern --lr .02 --train-person 1 "${C38_RESIDENT[@]}" \
    --patch-dir "results/c38/B_$arm" --out "results/c38/B_$arm.json"
done
"$C38_PYTHON" "$C38_ANALYSIS" sequential results/c38/B_{resident,masked,full}.json \
  --out results/c38/sequential_comparison.json

# Matched bank sum arithmetic for both formats; inspect independent solo controls.
# Supply previously trained per-person files. Also run without --merge sum_delta
# to measure each format's native merge, labelled separately.
"$C38_PYTHON" "$C38_EVAL" --patches results/c38/banks/p{0,1,2}/final.pt \
  --modes oracle sum router --merge sum_delta --anchor "$C38_ANCHOR" \
  --eval-set "$C38_WIKI" --fineweb "$C38_FINEWEB" --code "$C38_CODE" \
  --bootstrap 1000 --out results/c38/bank_sum_delta.json

# Resident dose varies; B always has exactly 100 facts and 300 updates.
# Scale A updates with A fact count so its expected exposures/fact stay constant.
for people in 1 3 10 30; do
  "$C38_PYTHON" "$C38_TRAIN" "${C38_COMMON[@]}" --factorial-cell sparse_tern --lr .02 \
    --synthetic-people "$people" --facts-per-person 100 --shared-attrs 100 --steps "$((300*people))" \
    --patch-dir "results/c38/dose/$people/A" --out "results/c38/dose/$people/A.json"
  for arm in resident masked full; do
    C38_RESIDENT=()
    if [ "$arm" = resident ]; then C38_RESIDENT=(--resident-patch "results/c38/dose/$people/A/final.pt"); fi
    if [ "$arm" = masked ]; then C38_RESIDENT=(--resident-mask-only "results/c38/dose/$people/A/final.pt"); fi
    "$C38_PYTHON" "$C38_TRAIN" "${C38_COMMON[@]}" --factorial-cell sparse_tern --lr .02 --seed 1 \
      --synthetic-people 1 --facts-per-person 100 --shared-attrs 100 "${C38_RESIDENT[@]}" \
      --patch-dir "results/c38/dose/$people/B_$arm" --out "results/c38/dose/$people/B_$arm.json"
  done
done
"$C38_PYTHON" "$C38_ANALYSIS" dose results/c38/dose/{1,3,10,30}/B_resident.json \
  --out results/c38/dose/comparison.json
```

For a direct/curriculum comparison, use the same world, LR, seed, steps,
`--aug`, `--answer-only`, and batch size; add `--comp-curriculum` only to the
curriculum arm. Read `composition_accounting` alongside recall: direct examples
are matched, but auxiliary examples increase total loss tokens. It is an
additional-supervision experiment, not a claim of equal total token budget.

Old synthetic checkpoints whose values encode indices are not evidence for
the revised memorization experiment; regenerate their worlds and patches.
