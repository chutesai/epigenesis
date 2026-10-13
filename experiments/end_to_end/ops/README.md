# Queue runner and janitor

Deploy `fsaq4.sh`, `lock_janitor3.sh`, and `ops_common.sh` together in one directory.
Production requires Linux `flock`, `setsid`, `readlink -f`, procps `ps`, `/proc`,
and `nvidia-smi`. The scripts resolve symlinks and PATH invocations to find the
shared file, and exit immediately if it cannot be sourced. Run one queue runner
and one janitor per `FSAQ_ROOT` (default `/workspace/malleable`).

`PY` must be a **single executable path**, for example `/venv/main/bin/python`.
A path containing spaces works. `PY='/venv/main/bin/python -u'` is rejected;
Python unbuffered output is already enabled through `PYTHONUNBUFFERED=1`.

Queue lines have the form `<script> <args>|<name>` or `SH <shell command>|<name>`.
For ordinary lines, arguments use **IFS word splitting followed by pathname
expansion in `FSAQ_ROOT`, exactly like fsaq2's unquoted `$cmd`**. Thus
`patches/*.pt` expands, unmatched globs remain literal, and quotes embedded in a
queue line are literal characters rather than shell quoting. A matched filename
containing spaces remains one argument. Use `SH` for shell quoting, pipelines,
redirection or other shell syntax; SH jobs must write their own outputs. The
last `|` separates the job name, so earlier SH pipes remain intact. Carriage
returns are stripped before parsing/validation. Names contain only letters,
digits, `_`, `-`, and `.`, and cannot start with `.`. Spaces are not trimmed from
names. Existing `logs/<name>.log` prevents a rerun.

Checkpoint-enabled e2e queue lines should include
`--ckpt-every 100 --ckpt-minutes 60 --resume auto` from their first launch; their
default recovery file is `out/<name>_patches/resume.pt`. Other jobs replay after a
janitor retry. The runner does not insert recovery flags.

The janitor defaults to a 45-minute stale-heartbeat threshold, two retries,
1800-second stale-lock age and 120-second polling. It checks GPU idleness three
times before removing aged ownerless GPU locks, then rechecks age, ownership and
GPU use under `state.lock`. Ownerless job locks use the age threshold and a
locked recheck. Only empty ownerless directories are removed with `rmdir`;
existing malformed owner records or extra contents require inspection.

A launch gate expires after 600 checks spaced 0.05 seconds apart if its supervisor
never opens it. Failed session establishment kills the candidate group and
leader, releases only matching placeholder claims, and archives its log as
`<name>.log.launchfail_<UTC timestamp>` for retry. Hang archives use
`<name>.log.hang_<UTC timestamp>_<retry>`. Both archive types append `.1`, `.2`,
etc. on collisions. Move failures are reported. After a leader exits, remaining
group members get 60 one-second grace checks, SIGTERM and ten more one-second
checks, then SIGKILL. Claims are released only after group death is verified.

Numeric environment settings are checked before scheduling or signaling.
`HANG_MIN`, `MAX_RETRY`, `STALE_LOCK_SECONDS`, `ABRT_WAIT_STEPS`, `KILL_WAIT_STEPS`,
`GATE_WAIT_STEPS`, `GROUP_GRACE_STEPS`, and `GROUP_TERM_STEPS` accept unsigned
decimal integers of at most nine digits, without leading zeroes (except `0`).
`FSAQ_SPACING_SECONDS`, `FSAQ_POLL_SECONDS`, `JANITOR_POLL_SECONDS`,
`KILL_WAIT_SECONDS`, `GATE_WAIT_SECONDS`, and `GROUP_WAIT_SECONDS` accept
nonnegative seconds, including fractions such as `.05`. Invalid settings exit 2.
The new gate/group knobs allow accelerated process tests; production defaults
give roughly 30 seconds for the gate and 60/10 seconds for group cleanup.

## One-time fsaq2 migration

Stop the old queue runner and other schedulers first, and wait for their jobs to
finish. Confirm that no live legacy owner/workload exists. Old fsaq2 `run_own`
files contain no identity, so any remaining record must be investigated before
this sweep. Execute the following on the deployment host, before starting
fsaq4. It refuses any remaining ownership record, takes the runner singleton and
shared state locks, observes each GPU idle three times, and uses only `rmdir` on
still-ownerless empty `locks/gpuN` directories. This does not delete logs,
ownership files or nonempty lock directories. The janitor subsequently handles
aged empty `locks/job_*` directories.

```bash
FSAQ_ROOT=/workspace/malleable OPS_DIR=/workspace/malleable/ops bash <<'BASH'
set -u
cd "$FSAQ_ROOT" || exit 1
source "$OPS_DIR/ops_common.sh" || exit 1
mkdir -p locks run_own
exec 9>locks/fsaq4.lock
flock -n 9 || { echo 'queue runner is active' >&2; exit 1; }
state_lock
for record in run_own/* run_own/.[!.]* run_own/..?*; do
  if [ -e "$record" ] || [ -L "$record" ]; then
    echo "inspect remaining ownership record: $record" >&2
    state_unlock; exit 1
  fi
done
for d in locks/gpu*; do
  [ -d "$d" ] || continue
  g=${d#locks/gpu}
  [[ "$g" =~ ^[0-9]+$ ]] || continue
  [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ] || continue
  idle=true
  for check in 1 2 3; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v g="$g" '$1+0 == g {gsub(/ /,"",$2); print $2}')
    if ! [[ "$used" =~ ^[0-9]+$ ]] || [ "$used" -ge 1000 ]; then
      idle=false; break
    fi
    [ "$check" = 3 ] || sleep 2
  done
  if "$idle" && [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ]; then
    rmdir "$d" 2>/dev/null && echo "removed empty legacy $d"
  fi
done
state_unlock
BASH
```

The migration command is documented for deployment; it has not been run against
production locks in this repository task.
