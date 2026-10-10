#!/bin/bash
# Add one arm to an existing run, owning one shared GPU lock for the entire chain.
set -eu
ARM=${1:?Usage: run_arm_chain.sh ARM RUN_DIR [--smoke]}
RUN=${2:?Usage: run_arm_chain.sh ARM RUN_DIR [--smoke]}
case "$ARM" in tmid|fp4mid|loramid|tmid_util01|tmid_util03|tbop) ;; *) echo "Unknown arm: $ARM" >&2; exit 2;; esac
SM=''
SESSIONS='1 2 3 4 5'
if [ "$#" -gt 2 ]; then
  [ "$#" -eq 3 ] && [ "$3" = --smoke ] || { echo 'Expected optional --smoke' >&2; exit 2; }
  SM=--smoke; SESSIONS='1 2'
fi
EPI=${EPI:-$(cd "$(dirname "$0")" && pwd)}
PY=${PY:-/venv/main/bin/python}
LOCKS=${LOCKS:-$EPI/../locks}
mkdir -p "$RUN/status" "$RUN/logs" "$LOCKS"
RUN=$(cd "$RUN" && pwd)
# one writer per arm, and a chain always starts from a fresh arm directory (session 1 calibrates tau/lambda once;
# reusing an old config would silently carry a previous run's calibration)
exec 7> "$RUN/$ARM.chain.lock"
flock -n 7 || { echo "Another chain holds $RUN/$ARM" >&2; exit 1; }
[ ! -e "$RUN/$ARM" ] || { echo "$RUN/$ARM exists: remove it to start a fresh chain" >&2; exit 1; }
CLAIM=''
release_gpu(){
  if [ -n "$CLAIM" ] && [ "$(cat "$LOCKS/gpu$GPU/epi_owner" 2>/dev/null)" = "$CLAIM" ]; then
    rm -f "$LOCKS/gpu$GPU/epi_owner"
    rmdir "$LOCKS/gpu$GPU" 2>/dev/null || true
  fi
}
trap release_gpu EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
while [ -z "$CLAIM" ]; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u) || { sleep 5; continue; }
  while IFS=', ' read -r idx uuid used; do
    [ "$used" -lt 1000 ] || continue
    echo "$busy" | grep -q "$uuid" && continue
    [ -e "$LOCKS/gpu$idx" ] && continue
    if mkdir "$LOCKS/gpu$idx" 2>/dev/null; then
      GPU=$idx
      CLAIM="$GPU:$(basename "$RUN").$$.$(date +%s%N).$RANDOM"
      echo "$CLAIM" > "$LOCKS/gpu$GPU/epi_owner"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  [ -n "$CLAIM" ] || sleep 5
done
export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
cd "$EPI"
for k in $SESSIONS; do
  rc=0
  set -- consolidate.py --run "$RUN" --arm "$ARM" --session "$k"
  [ -z "$SM" ] || set -- "$@" "$SM"
  "$PY" "$@" > "$RUN/logs/${ARM}_s$k.log" 2>&1 || rc=$?
  echo "$rc" > "$RUN/status/${ARM}_s$k"
  [ "$rc" -eq 0 ] || exit "$rc"
done
