# C48: C46 recheck fixes

Applied the FIX items in `reviews/experiments/c46_recheck/design_review_muse.md`.
Changed `c44_runtime.py`, `ops/fsaq4.sh`, `ops/lock_janitor3.sh`, and
`ops/ops_common.sh`; added checkpoint regressions, extended the real Bash/process
tests and documented operations in [ops/README.md](ops/README.md). Added narrow
`.gitignore` exceptions for the new report, test and runbook. `e2e_fsa.py` was not
edited, and concurrent C47 changes were preserved. No commit was run.

## Checkpoint input identity

Checkpoint-enabled runs now stream SHA-256 hashes for the export's complete
regular-file inventory, including nested payload shards and relay manifests,
as well as `bench_dir` contents. This is intentionally broader than parsing the
external builder's manifest schema: no claim is made that its manifests already
contain content hashes. Replacing a shard or benchmark JSON while leaving the
manifest unchanged is rejected before model construction. File additions and
removals also change identity. Input directory paths are normalized, and
fingerprinting still runs only when checkpoint/resume is enabled.

The export and benchmark directories should be immutable input trees; hashing
all their files conservatively rejects changes to auxiliary files too, and adds
one streamed read of the export payload per checkpoint-enabled process startup.
Older checkpoints with only a manifest fingerprint fail the existing input
identity check and need a fresh run. Checkpoint serialization and training
behavior are otherwise unchanged.

## Ops review items

| Review item | Change |
| --- | --- |
| N1: ownerless locks | Aged ownerless GPU locks need three idle observations plus an age/ownership/GPU-use recheck under `state.lock`. Aged ownerless job locks get the same locked age/ownership recheck. Plain `rmdir` preserves nonempty directories; malformed existing owner files remain protected. |
| N2: launch failure | Kill both the candidate session group and leader; wait for the leader; archive the failed launch log with a unique timestamp suffix; release GPU/job claims only against the passed placeholder identity. The ready gate has a bounded 600 × 0.05-second wait and exits 75 if never opened. |
| N3: stale mtime decision | Recheck log age under `state.lock`, immediately after PID/starttime/group verification and before SIGABRT. A fresh heartbeat while waiting for the lock prevents the kill. |
| N4: argument contract | Ordinary queue arguments use unquoted `$cmd` array expansion: IFS word splitting followed by globbing in `FSAQ_ROOT`, matching fsaq2. Embedded quotes remain literal; SH lines retain shell parsing and earlier pipes. `PY` is explicitly validated as one executable path, including paths containing spaces, and cannot contain additional flags. |
| N5: numeric settings | Validate all integer thresholds/retry counts/wait counts and fractional-second intervals before doing work. Integer leading zeroes and values longer than nine digits are rejected to avoid octal interpretation/overflow; invalid settings exit 2. Persisted hangcounts use the same integer restrictions. |
| N6: loading helpers | Resolve PATH/basename invocations and then canonicalize through `readlink -f`; check `source` and fail immediately if the common file is absent. |
| N7: leftover groups | After leader exit, allow 60 one-second grace checks, then TERM and ten one-second checks, then KILL and a bounded death check. Retain claims if death cannot be verified. This covers SH jobs without heartbeats. |

Additional requested fixes: strip carriage returns before queue validation;
reject every name starting with `.`; uniquify colliding hang archives instead of
giving up. Archive move errors are explicitly reported. Both singleton locks,
identity-checked cleanup, cap-before-kill behavior and retry counts remain intact.

The runbook contains the exact one-time migration command for empty legacy
`locks/gpu*` directories. It requires stopped old schedulers and no live legacy
owners/workloads, refuses remaining `run_own` records, holds runner/state locks,
checks GPU idleness three times and removes only still-ownerless empty
directories. The command was exercised only on fixture locks, not deployment
state. The queue contract and single-path `PY` requirement are documented there.

## Validation

Python 3.12.14, PyTorch 2.8.0, CPU/macOS; process tests use fake GPU/Python commands
and fixture implementations of Linux utilities, with real sessions, flock
primitives and process-group signals.

- Final C48/ops sweep: **49 passed in 23.75 seconds**. Includes unchanged-manifest
  payload replacement/addition/deletion, normalized directory paths, legacy
  ownerless lock recovery, locked ownership rechecks, failed-session cleanup and
  retry, orphan gate expiry, fresh-heartbeat race, word splitting/globs/CRLF/dot
  names, single-path PY, all invalid numeric knobs, symlink/PATH/basename loading,
  TERM-resistant leftovers requiring KILL, unique hang archives, and the exact
  documented migration command.
- C44/C46 checkpoint recovery regressions: **35 passed**, including real
  interrupted/resumed training. The combined run with the then-current 44 C48/ops
  cases passed **79 tests in 116.90 seconds**; the final five additional ops cases
  are included in the final 49-test sweep above.
- `bash -n` for all three scripts, Python compilation and `git diff --check` pass.

Final focused command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q experiments/end_to_end/tests/test_ops_scripts.py experiments/end_to_end/tests/test_c48.py
```

Recovery validation command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/epigenesis-ladder-venv/bin/python -m pytest -q experiments/end_to_end/tests/test_ops_scripts.py experiments/end_to_end/tests/test_c48.py experiments/end_to_end/tests/test_c44.py experiments/end_to_end/tests/test_c46.py
```

No GPU/model smoke test or production deployment was performed. Process-group
cleanup still cannot catch descendants that create another session; a blocking
GPU query can still delay operations holding `state.lock`. The requested fixes
do not change those existing limitations.
