# C46: checkpoint, backend and queue review fixes

Applied the requested fixes from `reviews/experiments/c44_c45_ops/design_review_muse.md`. Existing working-tree changes were preserved. No git commit was run.

## C44 recovery and heartbeats

Resume now compares SHA-256 content fingerprints for anchor/evaluation panels (including every extra panel), teacher/sibling/loaded/resident patches, and `<export>/manifest.json`. Mask comparison, calibration/rehearsal/routing inputs and unit-burst replay files are also covered. Hashing streams files in 1 MiB blocks. Absent optional/default files are recorded as null, so subsequent creation or deletion also changes identity. Fingerprinting runs only with checkpoint/resume enabled. A mismatch aborts before model construction; older checkpoints without fingerprints are refused.

Configuration identity normalizes input paths and excludes `resume`, `out`, `ckpt_every`, `ckpt_minutes`, and `patch_dir`. Moving a checkpoint or changing save cadence is permitted; changing the training horizon or backend still fails validation. `max_alloc_gb` is telemetry excluded from numerical exactness. A running allocator peak is checkpointed, restored, and combined with the new process's peak for trajectory/final reporting.

`object_state` raises for unsupported public or private state containers rather than dropping them. Explicit exceptions cover reconstructed UnitBurst banks, coordinates, module-name references and hooks, and MatchedActivationNoise hooks/deltas (redrawn before use). PyTorch Module's standard internal registries remain reconstructed or handled by `state_dict`. The synchronous, by-reference `values` contract is documented beside `training_state`.

All shared model-builder calls emit HEARTBEAT before and after construction. C42 centroid, LBFGS closure, Newton/line-search, bank/parent fitting, layer/hierarchy and cutoff/grid loops now heartbeat without RNG draws. Reports and `--resume` help state: **mechanically exact state restore on GPU; GPU kernels nondeterministic**. CPU bit-equality tests remain valid; GPU bit-exact numerics have not been established.

## C45 backend comparison

C41 birth matching includes `eda_backend` in identity and uses `eda.require_same`; legacy records default to naive_fp32. `mask_diagnostics` validates the saved mask's backend against the model before comparing bits. The C45 report describes the default as **numerically identical; artifacts gain a field**, conditional on the external builder's prior default being naive_fp32.

## Queue operations

Added `ops/fsaq4.sh`, `ops/lock_janitor3.sh` and their shared `ops_common.sh`. The supplied `*.orig` files are unchanged. Deploy all three new scripts together. Production defaults remain `/workspace/malleable`, `/venv/main/bin/python`, 90 seconds between launches, 60 seconds between queue scans, and 120 seconds between janitor scans. `FSAQ_ROOT`/`PY` and interval variables allow isolated tests.

The runner preserves `<script> <args>|<name>`, `SH <command>|<name>`, appended queue lines, log-based once-only execution, and `<queue>.stop`. It validates names, skips reserved GPUs, atomically claims `locks/job_<name>` before GPU selection, launches a gated `setsid` session, and stores `pid|pgid|starttime|name` in `run_own/<name>` and both ownership locks. Both scripts use singleton flock locks, with a shared flock protecting ownership changes. Cleanup checks the complete identity before removing records/locks, including delayed cleanup from a previous launch. Job environments include PYTHONFAULTHANDLER=1 and core dumps are disabled. SH descendants retain the same process group; cleanup waits for that group to finish.

The janitor retains its SKIP GPU argument and reserved-GPU policy. Stale removal requires a dead owner, an old lock, three consecutive idle checks, and another ownership/GPU-idle check under the shared lock. A live process group or failed process inspection retains the lock.

Hang handling requires a HEARTBEAT line and log age above HANG_MIN (default 45 minutes). The retry cap is checked before any signal; at the cap the janitor only alerts. PID starttime and session identity are rechecked immediately before signaling. SIGABRT targets the whole group, followed by bounded waits and SIGKILL if needed. A surviving group is alerted and its log retained. Only verified death permits a rename to `<name>.log.hang_<YYYYmmddTHHMMSSZ>_<n>`; archive collisions are refused. A retry marker preserves hangcount across janitor retries, while fresh launches reset it even if old archives remain. Janitor messages go to its own log, with no tee into the job log.

Only checkpoint-enabled e2e training resumes from `out/<name>_patches/resume.pt`; other jobs replay. Heartbeats remain cooperative: a single blocking model/kernel/filesystem call cannot emit an intermediate heartbeat. No CUDA/8B or real FLA run was available on this CPU host.

## Validation

Final focused C46/ops run: **29 passed** in 24.52 seconds, using Python 3.12.4 and cached PyTorch 2.8.0. The 19 C46 checks cover every requested input flag, export manifest and extra panels, cadence/path normalization, allocator-peak restore, public/private unsupported state, backend mismatches/legacy defaults, model-build and fitting heartbeats, and a real interrupted/resumed activation-noise training process. The 10 ops checks run Bash with fake nvidia-smi/python, actual sessions and process-group signals. They cover duplicate scheduling (including log removal while live), both singleton locks, matching-only cleanup, SH group kill/relaunch, cap-before-signal, simulated survival after kill, live/dead stale owners, idle/busy/reserved/SKIP GPU rules, changed starttime, fresh hangcount reset and normal completion cleanup.

The macOS host lacks Linux setsid/flock commands and blocks `/bin/ps`. Test adapters use os.setsid, fcntl.flock and targeted libproc inspection of fixture-owned pids; production Linux uses the normal utilities and `/proc` starttime. The alive-after-kill case suppresses janitor signals to simulate an uninterruptible process. Retry-cap tests confirm no SIGABRT delivery.

Final full end-to-end regression suite: **1,112 passed, 1 skipped**, with one existing scalar-conversion warning, in 322.60 seconds. The final sweep includes every new C46/ops test. Singleton polling sleeps close inherited lock descriptors, and the startup/restart regression waits for actual acquisition/release so it also verifies restart behavior.

Python compilation, `bash -n` for all three ops scripts, and `git diff --check` pass. GPU/FLA numerical validation remains unexecuted.

Equivalent test command with dependencies installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q experiments/end_to_end/tests
```
