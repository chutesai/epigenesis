#!/bin/bash
# fsaq4.sh <queuefile>: <script> <args>|<name>, or SH <shell command>|<name>.
# Appended lines are picked up each scan. Existing logs skip jobs; <queue>.stop stops scheduling.
set -u
OPS_PATH=$0
if [[ "$OPS_PATH" != */* ]] && [ ! -f "$OPS_PATH" ]; then
  OPS_PATH=$(command -v "$OPS_PATH") || exit 1
fi
OPS_PATH=$(readlink -f "$OPS_PATH") || exit 1
OPS_DIR=$(cd "$(dirname "$OPS_PATH")" && pwd) || exit 1
source "$OPS_DIR/ops_common.sh" || exit 1
FSAQ_SPACING_SECONDS=${FSAQ_SPACING_SECONDS:-90}; FSAQ_POLL_SECONDS=${FSAQ_POLL_SECONDS:-60}
GATE_WAIT_STEPS=${GATE_WAIT_STEPS:-600}; GATE_WAIT_SECONDS=${GATE_WAIT_SECONDS:-0.05}
GROUP_GRACE_STEPS=${GROUP_GRACE_STEPS:-60}; GROUP_TERM_STEPS=${GROUP_TERM_STEPS:-10}
GROUP_WAIT_SECONDS=${GROUP_WAIT_SECONDS:-1}
validate_uint GATE_WAIT_STEPS GROUP_GRACE_STEPS GROUP_TERM_STEPS || exit 2
validate_seconds FSAQ_SPACING_SECONDS FSAQ_POLL_SECONDS GATE_WAIT_SECONDS GROUP_WAIT_SECONDS || exit 2
PY=${PY:-/venv/main/bin/python}
cd "${FSAQ_ROOT:-/workspace/malleable}" || exit 1
# PY is one executable path, including paths with spaces; never a command plus flags.
[[ "$PY" == */* && -f "$PY" && -x "$PY" ]] || { echo 'PY must be a single executable path' >&2; exit 2; }
mkdir -p locks run_own logs out
exec 9>locks/fsaq4.lock
flock -n 9 || { echo 'fsaq4 already running' >&2; exit 1; }
Q=${1:?usage: fsaq4.sh queuefile}
ulimit -c 0
free_gpu() {
  local g used
  while IFS=',' read -r g used; do
    g=${g//[[:space:]]/}; used=${used//[[:space:]]/}
    [[ "$g" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ ]] || continue
    reserved_gpu "$g" && continue
    if [ "$used" -lt 1000 ] && mkdir "locks/gpu$g" 2>/dev/null; then
      printf '%s\n' "$g"; return
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}
launch_job() {
  local cmd=$1 name=$2 gpu=$3 placeholder=$4 pid pgid start owner i archive
  local -a argv
  exec 9>&- 8>&-
  # The gated session cannot finish before its ownership records are installed.
  if [[ "$cmd" == SH\ * ]]; then
    argv=(bash -c "${cmd#SH }")
  else
    # Match fsaq2's $cmd expansion: IFS word splitting, then pathname expansion.
    # Embedded quotes are literal characters; SH lines alone get shell parsing.
    argv=( $cmd )
    argv=("$PY" "${argv[@]}" --out "out/$name.json")
  fi
  setsid bash -c 'gate=$1; steps=$2; seconds=$3; shift 3
    for ((i=0; i<steps; i++)); do
      [ -f "$gate" ] && exec "$@"
      sleep "$seconds"
    done
    exit 75' \
    job "locks/job_$name/ready" "$GATE_WAIT_STEPS" "$GATE_WAIT_SECONDS" env OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    TRITON_CACHE_DIR="/tmp/triton_$name" CUDA_VISIBLE_DEVICES="$gpu" "${argv[@]}" \
    >"logs/$name.log" 2>&1 </dev/null &
  pid=$!
  start=$(process_start "$pid")
  for i in {1..100}; do
    pgid=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
    [ "$pgid" = "$pid" ] && [ -n "$start" ] && break
    sleep 0.05
    start=$(process_start "$pid")
  done
  if [ "$pgid" != "$pid" ] || [ -z "$start" ]; then
    echo "cannot establish session for $name; releasing launch claim" >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    state_lock
    if owner_matches "locks/job_$name/owner" "$placeholder"; then
      archive=$(unique_archive "logs/$name.log.launchfail_$(date -u +%Y%m%dT%H%M%SZ)")
      mv "logs/$name.log" "$archive" || echo "cannot archive launch failure for $name" >&2
    fi
    release_dir "locks/gpu$gpu" "$placeholder"
    release_dir "locks/job_$name" "$placeholder"
    state_unlock
    return 1
  fi
  owner="$pid|$pgid|$start|$name"
  state_lock
  printf '%s\n' "$owner" >"run_own/$name"
  printf '%s\n' "$owner" >"locks/gpu$gpu/owner"
  printf '%s\n' "$owner" >"locks/job_$name/owner"
  touch "locks/job_$name/ready"
  state_unlock
  echo "launched $name on GPU $gpu $(date -u +%H:%MZ)"
  wait "$pid" || true
  # Bound leftovers after the leader exits, even for SH jobs without heartbeats.
  for ((i=0; i<GROUP_GRACE_STEPS; i++)); do
    group_alive "$pgid" || break
    sleep "$GROUP_WAIT_SECONDS"
  done
  if group_alive "$pgid"; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for ((i=0; i<GROUP_TERM_STEPS; i++)); do
      group_alive "$pgid" || break
      sleep "$GROUP_WAIT_SECONDS"
    done
  fi
  if group_alive "$pgid"; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi
  for ((i=0; i<GROUP_TERM_STEPS; i++)); do
    group_alive "$pgid" || break
    sleep "$GROUP_WAIT_SECONDS"
  done
  if group_alive "$pgid"; then
    echo "$name group still alive after cleanup; retaining claims" >&2
    return 1
  fi
  cleanup_owner "$name" "$gpu" "$owner"
}
while true; do
  [ -e "$Q.stop" ] && { echo 'stop file, exiting'; exit 0; }
  while IFS= read -r item <&3 || [ -n "$item" ]; do
    item=${item//$'\r'/}
    [ -z "$item" ] && continue
    [[ "$item" == \#* ]] && continue
    [[ "$item" == *'|'* ]] || { echo "invalid queue line: $item" >&2; continue; }
    cmd=${item%|*}; name=${item##*|}
    valid_name "$name" || { echo "invalid job name: $name" >&2; continue; }
    state_lock
    if [ -e "logs/$name.log" ] || ! mkdir "locks/job_$name" 2>/dev/null; then
      state_unlock; continue
    fi
    # Claim the name BEFORE selecting/claiming a GPU.
    G=$(free_gpu)
    if [ -z "$G" ]; then
      rmdir "locks/job_$name"; state_unlock; break
    fi
    # A janitor retry marker preserves the count; all other launches start fresh.
    if [ -f "logs/$name.hangretry" ]; then
      rm -f "logs/$name.hangretry"
    else rm -f "logs/$name.hangcount"; fi
    owner="$$|$$|$(process_start $$)|$name"
    printf '%s\n' "$owner" >"locks/job_$name/owner"
    printf '%s\n' "$owner" >"locks/gpu$G/owner"
    launch_job "$cmd" "$name" "$G" "$owner" &
    state_unlock
    sleep "$FSAQ_SPACING_SECONDS" 9>&-
  done 3<"$Q"
  sleep "$FSAQ_POLL_SECONDS" 9>&-
done
