# C35 training recipes

The design-review fixes and replacement comparison protocols are in
[C38_REPORT.md](C38_REPORT.md). The historical commands below predate those
fixes and should not be used to support decisive comparison claims.

Implemented in `e2e_fsa.py` and `branch_eval.py`; existing training defaults remain unchanged. No commits or latency work.

- Repeatable `--resident-patch` stacks frozen saved deltas on every post-attachment training/evaluation forward. Actual resident coordinates are excluded from new FSA support and new LoRA deltas. Residents use buffers and never enter the optimizer or new patch export. `sequential_revoke` reports full scoring paths (B recall, held-out damage, extra panels, optional QA/benchmarks) with residents on, residents revoked, and residents alone. Resident-only scoring also includes each saved resident's original-world recall. `resident_slots` records individual nonzero counts, the occupied union, and file hashes; overlapping residents count once in the union. Final full revocation checks the pristine base. New writes with wake/unit/fold/MLS/drift recipes are rejected explicitly.
- `--factorial-cell {sparse_tern,sparse_fp,dense_tern,dense_fp}` chooses forced sparse support or LoRA r16, ternary or floating-point values, Adam, full eligible support, and row scales. All cells use lr=.001 unless supplied explicitly. The local penalty remains the supplied `--local-lambda` in every cell. Ternary LoRA rounds/clamps the actual product B@A to {-alpha,0,+alpha}, with STE and base-derived per-row alpha; its penalty uses that same delta. Factors, format, scales and exclusions round-trip in patch files. `c35_recipe` records the recipe in results/checkpoints; the existing manifest records full arguments/exposures.
- `--bootstrap 1000` adds percentile 95% `ci95` intervals to test GEN and QA metrics, using an isolated RNG and resampling complete fact clusters. QA uses the target `fi` (the first fact for paired answers), keeping each pair score intact. Empty panels have unavailable bounds. Both evaluators expose these intervals.
- Per-person LoRA banks support `--modes oracle sum router`, including the usual recall, composition, damage, selection, byte counts, zero control and revocation metrics. Sum activates every stored delta at its saved strength; ternary products are quantized individually before addition; active products sum in fp32 before one cast/addition to bf16 base weights. The router reads the frozen base and locks settings on validation. LoRA banks are independent writes on the base; FSA partition and C34 arms retain their existing restrictions.
- `--comp-curriculum` samples three training items per selected fact from direct/paired/inverse/null groups. `--comp-{direct,paired,inverse,null}-weight` controls relative sampling probability (default 1 each; zero excludes a group). Pairs use only taught facts of that person; inverse answers give the name; null answers say `I don't know.` for an attribute absent from the full world. One cyclic pair per fact bounds the pool size. New prompt templates are disjoint from evaluation probes, and the composition batch scores answer tokens. Existing attribute/paired/saturation/conflict/promotion recipes are rejected when incompatible.
- `--synthetic-people N` uses N deterministic synthetic people instead of the named PEOPLE list, overrides `--people`, supplies unique fact values, and extends the attribute pool when necessary. World metadata lets bank evaluation reconstruct exactly the same facts.

CPU validation: 22 C35 tests cover actual training in every factorial cell, LoRA person filtering, frozen/disjoint/repeated residents, original/base revocation, product STE/penalty/export, composition labels and sampling, 100/300/1000/3000-fact worlds and attribute extension, clustered uncertainty, and both LoRA bank formats through all three evaluation policies. Full CPU suite: **844 passed, 1 skipped** (one existing tensor-to-scalar warning); `git diff --check`, Python compilation, and shell syntax checks of the documented commands pass.

```bash
C35_PYTHON=/Users/jdurbin/.cache/uv/environments-v2/e0817003e2ecbca1/82395b30f9e793b9/bin/python
OMP_NUM_THREADS=1 "$C35_PYTHON" -m pytest -q --show-capture=no experiments/end_to_end/tests
git diff --check
```

Full-model recipes below were **not run** here: the external model loader/checkpoint and GPU runtime are unavailable. Run from the repository root, supplying three disjoint token files and the usual external loader/tokenizer installation. The default export is `/workspace/malleable/data/hf/exports/725B-tokens_step58814`, step 58814. Shell arrays keep shared flags identical; output JSON and patch directories are explicit. Repeat the entire protocol with seeds 1 and 2 for replication.

```bash
C35_PYTHON=python3
C35_ANCHOR=/absolute/path/to/train_anchor_tokens.json
C35_WIKI=/absolute/path/to/wikitext_tokens.json
C35_FINEWEB=/absolute/path/to/fineweb_tokens.json
C35_CODE=/absolute/path/to/code_tokens.json
C35_TRAIN=experiments/end_to_end/e2e_fsa.py
C35_EVAL=experiments/end_to_end/branch_eval.py
C35_MATCHED=(--layers 12,14,17,20 --seed 0 --steps 300 --lr .001
  --local-lambda 10 --answer-only --aug --fact-batch 8 --micro 2
  --anchor "$C35_ANCHOR" --eval-set "$C35_WIKI"
  --extra-eval-sets "$C35_FINEWEB,$C35_CODE" --eval-every 50
  --qa-probe --qa-n 120 --bootstrap 1000 --no-pareto)
C35_WORLD=(--people 3 --facts-per-person 100 --shared-attrs 100 --values disjoint)

# Sequential A -> B -> revoke A, with B-on-base control; compare final checkpoints.
"$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
  --factorial-cell sparse_tern --train-person 0 \
  --patch-dir results/c35/A --out results/c35/A.json
"$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
  --factorial-cell sparse_tern --train-person 1 --resident-patch results/c35/A/final.pt \
  --patch-dir results/c35/B_resident --out results/c35/B_resident.json
"$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
  --factorial-cell sparse_tern --train-person 1 \
  --patch-dir results/c35/B_base --out results/c35/B_base.json

# Matched 2x2 generalization-gap experiment: all other arguments/data order stay fixed.
for cell in sparse_tern sparse_fp dense_tern dense_fp; do
  "$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
    --factorial-cell "$cell" --patch-dir "results/c35/$cell" --out "results/c35/$cell.json"
done
# Extra arms: LoRA with 0.2 rehearsal, FSA with lambda=100.
"$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
  --factorial-cell dense_fp --rehearsal-frac .2 \
  --patch-dir results/c35/lora_rehearsal --out results/c35/lora_rehearsal.json
"$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
  --factorial-cell sparse_tern --local-lambda 100 \
  --patch-dir results/c35/fsa_lambda100 --out results/c35/fsa_lambda100.json

# Person banks, with and without composition training, for FSA and LoRA.
# dense_fp can be replaced with dense_tern to evaluate ternary LoRA banks.
for cell in sparse_tern dense_fp; do
  for curriculum in direct comp; do
    C35_COMP=()
    if [ "$curriculum" = comp ]; then
      C35_COMP=(--comp-curriculum --comp-direct-weight 1 --comp-paired-weight 1
        --comp-inverse-weight 1 --comp-null-weight 1)
    fi
    for person in 0 1 2; do
      "$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" "${C35_WORLD[@]}" \
        --factorial-cell "$cell" --train-person "$person" "${C35_COMP[@]}" \
        --patch-dir "results/c35/banks/$cell/$curriculum/p$person" \
        --out "results/c35/banks/$cell/$curriculum/p$person.json"
    done
    "$C35_PYTHON" "$C35_EVAL" --modes oracle sum router \
      --patches "results/c35/banks/$cell/$curriculum/"p{0,1,2}/final.pt \
      --anchor "$C35_ANCHOR" --eval-set "$C35_WIKI" --fineweb "$C35_FINEWEB" --code "$C35_CODE" \
      --bootstrap 1000 --out "results/c35/banks/$cell/$curriculum/eval.json"
  done
done

# Capacity/damage dose: 100, 300, 1000, 3000 facts, then an equally sized B world.
# Compare B with residents revoked to independently trained seed-1 B at each size.
for people in 1 3 10 30; do
  "$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" --factorial-cell sparse_tern \
    --synthetic-people "$people" --facts-per-person 100 --shared-attrs 100 \
    --patch-dir "results/c35/capacity/$people/A" --out "results/c35/capacity/$people/A.json"
  "$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" --factorial-cell sparse_tern --seed 1 \
    --synthetic-people "$people" --facts-per-person 100 --shared-attrs 100 \
    --resident-patch "results/c35/capacity/$people/A/final.pt" \
    --patch-dir "results/c35/capacity/$people/B" --out "results/c35/capacity/$people/B.json"
  "$C35_PYTHON" "$C35_TRAIN" "${C35_MATCHED[@]}" --factorial-cell sparse_tern --seed 1 \
    --synthetic-people "$people" --facts-per-person 100 --shared-attrs 100 \
    --patch-dir "results/c35/capacity/$people/B_base" --out "results/c35/capacity/$people/B_base.json"
done
```

For sequential dependence, use `sequential_revoke.new_with_residents.acq.test_gen_person`, `new_residents_revoked.acq.test_gen_person`, and the B-base `patched.test_gen_person`. For capacity runs without a person filter, use `test_gen`. Compare damage `big.dnll` and `big_extra` against resident occupied-union slot count, and inspect resident-only `original_world_recall`. The factorial gap is final train TF versus test GEN/QA under the same recipe; intervals are individual metric intervals, not a CI for the train–test difference or between-cell contrasts.
