#!/bin/bash
# lock_janitor3.sh [SKIP_GPUS]. Retries require verified session death.
# Only checkpoint-enabled e2e jobs resume (out/<name>_patches/resume.pt); other jobs replay.
set -u
OPS_PATH=$0
if [[ "$OPS_PATH" != */* ]] && [ ! -f "$OPS_PATH" ]; then
  OPS_PATH=$(command -v "$OPS_PATH") || exit 1
fi
OPS_PATH=$(readlink -f "$OPS_PATH") || exit 1
OPS_DIR=$(cd "$(dirname "$OPS_PATH")" && pwd) || exit 1
source "$OPS_DIR/ops_common.sh" || exit 1
HANG_MIN=${HANG_MIN:-45}; MAX_RETRY=${MAX_RETRY:-2}; STALE_LOCK_SECONDS=${STALE_LOCK_SECONDS:-1800}
ABRT_WAIT_STEPS=${ABRT_WAIT_STEPS:-12}; KILL_WAIT_STEPS=${KILL_WAIT_STEPS:-12}
KILL_WAIT_SECONDS=${KILL_WAIT_SECONDS:-5}; JANITOR_POLL_SECONDS=${JANITOR_POLL_SECONDS:-120}
validate_uint HANG_MIN MAX_RETRY STALE_LOCK_SECONDS ABRT_WAIT_STEPS KILL_WAIT_STEPS || exit 2
validate_seconds KILL_WAIT_SECONDS JANITOR_POLL_SECONDS || exit 2
cd "${FSAQ_ROOT:-/workspace/malleable}" || exit 1
mkdir -p locks run_own logs
exec 9>locks/lock_janitor3.lock
flock -n 9 || { echo 'lock_janitor3 already running' >&2; exit 1; }
SKIP=${1:-}
declare -a idle
alert() { echo "janitor: $* $(date -u)" >>logs/lock_janitor.log; }
while true; do
  [ -e janitor.stop ] && exit 0
  while IFS=',' read -r g used; do
    g=${g//[[:space:]]/}; used=${used//[[:space:]]/}
    [[ "$g" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
    reserved_gpu "$g" && continue
    d="locks/gpu$g"
    [ -d "$d" ] || { idle[$g]=0; continue; }
    owner=$(cat "$d/owner" 2>/dev/null) || owner=''
    # Legacy/crash-window empty directories have no owner. Malformed existing
    # owner files remain fail-safe: only a missing owner permits plain rmdir.
    ownerless=false
    if [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ]; then ownerless=true; fi
    if { "$ownerless" || { owner_identity "$owner" && ! owner_alive "$owner"; }; } &&
       [ "$used" -lt 1000 ] && [ "$(file_age_seconds "$d")" -gt "$STALE_LOCK_SECONDS" ]; then
      idle[$g]=$(( ${idle[$g]:-0} + 1 ))
    else idle[$g]=0; fi
    if [ "${idle[$g]:-0}" -ge 3 ]; then
      state_lock
      # Re-query GPU use and ownership inside the same lock as runner/cleanup.
      current_used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, -v g="$g" '$1+0 == g {gsub(/ /,"",$2); print $2}')
      if [[ "$current_used" =~ ^[0-9]+$ ]] && [ "$current_used" -lt 1000 ] &&
         [ "$(file_age_seconds "$d")" -gt "$STALE_LOCK_SECONDS" ]; then
        if "$ownerless"; then
          if [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ] && rmdir "$d" 2>/dev/null; then
            alert "removed ownerless stale lock gpu$g"
          fi
        elif owner_matches "$d/owner" "$owner" && ! owner_alive "$owner"; then
          IFS='|' read -r pid pgid start name <<< "$owner"
          if ! group_alive "$pgid"; then
            release_dir "$d" "$owner"
            release_dir "locks/job_$name" "$owner"
            if owner_matches "run_own/$name" "$owner"; then rm -f "run_own/$name"; fi
            alert "removed stale lock gpu$g"
          fi
        fi
      fi
      state_unlock; idle[$g]=0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  for d in locks/job_*; do
    [ -d "$d" ] && [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ] &&
      [ "$(file_age_seconds "$d")" -gt "$STALE_LOCK_SECONDS" ] || continue
    state_lock
    if [ ! -e "$d/owner" ] && [ ! -L "$d/owner" ] &&
       [ "$(file_age_seconds "$d")" -gt "$STALE_LOCK_SECONDS" ] && rmdir "$d" 2>/dev/null; then
      alert "removed ownerless stale lock ${d##*/}"
    fi
    state_unlock
  done
  for record in run_own/*; do
    [ -f "$record" ] || continue
    name=${record##*/}; valid_name "$name" || continue
    owner=$(cat "$record")
    owner_identity "$owner" || continue
    IFS='|' read -r pid pgid start recorded_name <<< "$owner"
    [ "$recorded_name" = "$name" ] || continue
    log="logs/$name.log"
    [ -f "$log" ] && grep -aqm1 '^HEARTBEAT ' "$log" || continue
    [ "$(file_age_seconds "$log")" -gt "$((HANG_MIN * 60))" ] || continue
    state_lock
    count=$(cat "logs/$name.hangcount" 2>/dev/null) || count=0
    [[ "$count" =~ ^(0|[1-9][0-9]*)$ && ${#count} -le 9 ]] || { alert "invalid hangcount $name"; state_unlock; continue; }
    if [ "$count" -ge "$MAX_RETRY" ]; then
      alert "HANG $name at retry cap $count/$MAX_RETRY; alert only"
      state_unlock; continue
    fi
    # Slow log scanning is finished. Re-verify pid starttime/group immediately before signals.
    if ! owner_matches "$record" "$owner"; then
      state_unlock; continue
    fi
    if ! owner_group_matches "$owner"; then state_unlock; continue; fi
    # A blocked call may have completed and heartbeated while we waited for state.lock.
    [ "$(file_age_seconds "$log")" -gt "$((HANG_MIN * 60))" ] || { state_unlock; continue; }
    alert "HANG $name pid $pid; SIGABRT group $pgid; retry $((count+1))/$MAX_RETRY"
    kill -ABRT -- "-$pgid" 2>/dev/null || true
    for ((i=0; i<ABRT_WAIT_STEPS; i++)); do
      group_alive "$pgid" || break
      sleep "$KILL_WAIT_SECONDS" 9>&-
    done
    if group_alive "$pgid"; then
      # If the leader survived, require the original starttime again before escalation.
      if kill -0 "$pid" 2>/dev/null && ! owner_alive "$owner"; then
        alert "identity changed for $name; no escalation or retry"
        state_unlock; continue
      fi
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
    for ((i=0; i<KILL_WAIT_STEPS; i++)); do
      group_alive "$pgid" || break
      sleep "$KILL_WAIT_SECONDS" 9>&-
    done
    if group_alive "$pgid"; then
      alert "$name still alive after kill; no rename or retry"
      state_unlock; continue
    fi
    # Keep state.lock until rename: old supervisor cleanup cannot race a replacement.
    if owner_matches "$record" "$owner"; then
      retry=$((count+1))
      archive=$(unique_archive "$log.hang_$(date -u +%Y%m%dT%H%M%SZ)_$retry")
      if mv "$log" "$archive"; then
        printf '%s\n' "$retry" >"logs/$name.hangcount"
        printf '%s\n' "$archive" >"logs/$name.hangretry"
        rm -f "$record"
        release_dir "locks/job_$name" "$owner"
        for d in locks/gpu*; do [ -d "$d" ] && release_dir "$d" "$owner"; done
        echo "$name hang $(date -u)" >>logs/killed_by_fsa_owner.log
      else
        alert "$name hang archive move failed; retained ownership for inspection"
      fi
    fi
    state_unlock
  done
  sleep "$JANITOR_POLL_SECONDS" 9>&-
done
