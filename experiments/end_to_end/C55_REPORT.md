# C55: parity-gate cross-backend evaluation

Added `--allow-backend-crosseval` to `e2e_fsa` and `branch_eval`. A saved `naive_fp32` patch can be evaluated with `--eda-backend fla_kda_2t`, and a saved `fla_kda_2t` patch can be evaluated with `--eda-backend naive_fp32`. Without the flag, C45's mismatch rejection remains in effect.

Cross-evaluation prints `CROSS-EVAL saved=<backend> eval=<backend>` and records `backend_crosseval: {saved: X, eval: Y}` in the output file and RESULT/FINAL JSON. `eda_backend` continues to identify the evaluation backend. Same-backend evaluation does not gain a cross-evaluation marker. Original patch files retain their saved provenance.

The opt-in check is limited to patch loads: loaded/transplant patches, residents, mask-only residents, siblings, normal branch banks, and branch learning-environment evaluation. Branch serving and its decisive-evaluation path use the same check. Other artifact checks, export validation, mixed-bank rejection, and base-cache validation remain strict. C42 feature-cache workflows retain their existing backend restrictions. Base caches continue to use the selected evaluation backend in keys and manifests.

The flag is evaluation-only: nonzero value (`--steps`), wake, or repair training raises before the e2e runner reads patches or builds a model. This also rejects same-backend training with the flag, preventing an evaluation override from becoming a training option.

Example: append `--steps 0 --allow-backend-crosseval --eda-backend fla_kda_2t --load-patch <naive-patch.pt>` to the matching e2e configuration. For branch evaluation, append `--allow-backend-crosseval --eda-backend fla_kda_2t` to the matching branch command. Reverse the backend for the opposite parity measurement.

## Validation

- C55 and C45: **54 passed**. Tests exercise both directions through real CPU CLI flows, normal and learning branch evaluation, loaded and resident/sibling patches, RESULT/FINAL metadata, banners, original-file provenance, evaluation-backend cache manifests, same-backend behavior, default rejection, training rejection before I/O, and strict cache/mixed-bank checks.
- Broader regressions (`test_e2e_fsa`, `test_branch_eval`, C34, C44, C48, C52): **331 passed**, with one existing PyTorch tensor-to-scalar warning.
- Python compilation and `git diff --check` pass.
- The CPU harness supplies fake FLA imports; real FLA GPU kernel numerics and the parity gate itself were not exercised on this host.

`census.py` was not edited. Existing workspace edits were preserved. No git commit was run.
