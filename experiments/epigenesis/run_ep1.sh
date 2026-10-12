#!/bin/bash
# Detached, status-checked EP-1 jobs; GPU capacity is reclaimed while waiting.
set -eu
MODE=${1:?smoke|full}
case "$MODE" in smoke|full) ;; *) echo 'Expected smoke or full' >&2; exit 1;; esac
V22=${V22:-0}
MAX_GPUS=${MAX_GPUS:-4}
case "$MAX_GPUS" in ''|*[!0-9]*|0) echo 'MAX_GPUS must be positive' >&2; exit 1;; esac
if [ "$V22" = 1 ]; then
  ARMS=${ARMS:-"tmid tmid_lr005 fp4mid loramid tbop tmid_util01 tmid_util03 tmid_mdl lorafold"}
  STUDY_ARGS='--write-set forced --objective supervised --steps 600 --eval-every 50 --refresh-until 300'
  CORPUS_ARGS='--rules coarse'
else
  ARMS=${ARMS:-"tmid fp4mid loramid"}
  STUDY_ARGS=''
  CORPUS_ARGS=''
fi
export V22 MAX_GPUS STUDY_ARGS
for arm in $ARMS; do
  case "$arm" in tmid|tmid_lr005|fp4mid|loramid|tbop|lorafold|tmid_util01|tmid_util03|tmid_mdl) ;; *) echo "Unknown arm: $arm" >&2; exit 1;; esac
done
export ARMS
NAME=${2:-ep1}
if [ "$MODE" = smoke ]; then NAME=${2:-ep1_smoke}; fi
EPI=${EPI:-/workspace/malleable/epi}
PY=${PY:-/venv/main/bin/python}
RUN=$EPI/out/$NAME
TOK=${TOK:-/workspace/malleable/data/tokenizer.json}
LOCKS=${LOCKS:-$EPI/../locks}   # shared mkdir-lock convention with the FSA schedulers (/workspace/malleable/locks)
if [ -e "$RUN" ] && [ "${RESUME:-0}" != 1 ]; then
  echo "Run already exists: $RUN (set RESUME=1 to resume)" >&2; exit 1
fi
if [ "$MODE" = full ] && [ ! -s "$EPI/out/${SMOKE:-ep1_smoke}/SMOKE_OK" ]; then
  echo "Full mode requires out/${SMOKE:-ep1_smoke}/SMOKE_OK" >&2; exit 1
fi
mkdir -p "$RUN/logs" "$RUN/status" "$RUN/selfstudy" "$LOCKS"
# one runner per run directory for its whole lifetime
exec 8> "$RUN/run.lock"
flock -n 8 || { echo "Another runner holds $RUN" >&2; exit 1; }
cd "$EPI"
touch "$RUN/pids" "$RUN/jobs" "$EPI/claimed"
# GPU claims are shared ACROSS runs ($EPI/claimed + $EPI/claim.lock) so two runners never pick the same idle GPU
[ "$MODE" = smoke ] && rm -f "$RUN/SMOKE_OK"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export RUN PY
log(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
free_gpu(){
  local busy idx uuid used
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  while IFS=', ' read -r idx uuid used; do
    [ "$used" -lt 1000 ] || continue
    echo "$busy" | grep -q "$uuid" && continue
    grep -q "^$idx:" "$EPI/claimed" && continue
    [ -d "$LOCKS/gpu$idx" ] && continue          # held by another scheduler (shared mkdir convention)
    echo "$idx"; return 0
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
}
# A claim is "N:token"; the token is written into $LOCKS/gpuN/epi_owner, so a release (by the job's own epilogue
# or by the reaper) only touches a lock and registry entry that this claim still owns: idempotent, never frees a
# later claimant's GPU.
release_gpu(){ # caller holds claim.lock; arg N:token
  local n=${1%%:*}
  grep -vx "$1" "$EPI/claimed" > "$EPI/claimed.tmp" || true
  mv "$EPI/claimed.tmp" "$EPI/claimed"
  if [ "$(cat "$LOCKS/gpu$n/epi_owner" 2>/dev/null)" = "$1" ]; then
    rm -f "$LOCKS/gpu$n/epi_owner"; rmdir "$LOCKS/gpu$n" 2>/dev/null || true
  fi
}
reap_locked(){
  local pid gpu name
  : > "$RUN/jobs.tmp"
  while read -r pid gpu name; do
    if [ -f "$RUN/status/$name" ]; then
      :   # finished: the job's own wrapper already released its GPU lock and claim
    elif ! kill -0 "$pid" 2>/dev/null; then
      # died without running its wrapper epilogue: release on its behalf
      release_gpu "$gpu" || return 1      # release first: a status file means "GPU already released"
      echo 1 > "$RUN/status/$name"
    else
      echo "$pid $gpu $name" >> "$RUN/jobs.tmp"
    fi
  done < "$RUN/jobs"
  mv "$RUN/jobs.tmp" "$RUN/jobs"
}
claim_locked(){
  local gpu
  reap_locked
  # Count this runner's live GPU jobs while holding the shared claim lock.
  if [ "${V22:-0}" = 1 ]; then
    [ "$(wc -l < "$RUN/jobs")" -lt "${MAX_GPUS:-4}" ] || return 0
  fi
  gpu=$(free_gpu)
  # atomic cross-scheduler claim: mkdir $LOCKS/gpuN; a failure means someone took it between the check and now
  if [ -n "$gpu" ] && mkdir "$LOCKS/gpu$gpu" 2>/dev/null; then
    local claim="$gpu:$(basename "$RUN").$$.$(date +%s%N).$RANDOM"
    echo "$claim" > "$LOCKS/gpu$gpu/epi_owner"; echo "$claim" >> "$EPI/claimed"; echo "$claim"
  fi
}
with_lock(){ ( flock -x 9 || exit 1; "$@" ) 9> "$EPI/claim.lock"; }
claim(){
  # `var=$(...) 9>file` would open fd 9 only after the substitution ran, so the lock lives in a subshell helper
  local gpu
  while :; do
    gpu=$(with_lock claim_locked) || { echo "claim lock failed" >&2; return 1; }
    if [ -n "$gpu" ]; then echo "$gpu"; return; fi
    sleep 5
  done
}
check_job(){
  local name=$1 artifact=$2
  [ -f "$RUN/status/$name" ] && [ "$(cat "$RUN/status/$name")" = 0 ] &&
    [ -s "$artifact" ] && [ "$artifact" -nt "$RUN/status/$name.start" ] || {
      echo "Job failed or artifact stale: $name ($artifact)" >&2; return 1;
    }
}
launch(){ # claim(N:token) name command...
  local claim=$1 gpu=${1%%:*} name=$2 pid; shift 2
  rm -f "$RUN/status/$name"
  touch "$RUN/status/$name.start"
  # the job releases its own GPU (lock dir + claim) when it exits, so a dead runner never strands a GPU
  JOB_CLAIM=$claim EPI=$EPI LOCKS=$LOCKS CUDA_VISIBLE_DEVICES=$gpu setsid nohup bash -c '"$@"; rc=$?
    # release (only what this claim still owns) + status, atomically under the lock; if this epilogue dies, the
    # token-checked release done by the reaper is idempotent
    ( flock -x 9 || exit 1; n=${JOB_CLAIM%%:*}
      { grep -vx "$JOB_CLAIM" "$EPI/claimed" || true; } > "$EPI/claimed.$$" && mv "$EPI/claimed.$$" "$EPI/claimed" &&
      if [ "$(cat "$LOCKS/gpu$n/epi_owner" 2>/dev/null)" = "$JOB_CLAIM" ]; then
        rm -f "$LOCKS/gpu$n/epi_owner"; rmdir "$LOCKS/gpu$n" 2>/dev/null || true; fi &&
      echo "$rc" > "$RUN/status/$JOB_NAME" ) 9> "$EPI/claim.lock"
    exit "$rc"' ep1 "$@" \
    > "$RUN/logs/$name.log" 2>&1 < /dev/null &
  pid=$!
  # from here the detached job owns the claim and releases it on exit, so the runner must never release it
  # (signals are not deferred: ignored signals would be inherited by the job and block exact-PID stops)
  PENDING=''
  echo "$pid $claim $name" >> "$RUN/pids"
  (flock -x 9 || exit 1; echo "$pid $claim $name" >> "$RUN/jobs") 9> "$EPI/claim.lock" ||
    { echo "job registration failed for $name (pid $pid); it releases its own GPU on exit" >&2; exit 1; }
  log "$name on gpu$gpu pid $pid"
}
wait_job(){
  while [ ! -f "$RUN/status/$1" ]; do
    with_lock reap_locked
    sleep 5
  done
  with_lock reap_locked
  check_job "$1" "$2"
}
export -f check_job
# a claim taken but not yet handed to a launched job is released if the runner dies or is stopped
PENDING=''
trap '[ -n "$PENDING" ] && with_lock release_gpu "$PENDING"' EXIT
trap 'exit 143' TERM INT
if [ -s "$RUN/jobs" ]; then  # RESUME: refuse live jobs, release GPUs of dead ones, then forget them
  while read -r pid gpu name; do
    if kill -0 "$pid" 2>/dev/null; then echo "Cannot resume: tracked job $name (pid $pid) is still alive" >&2; exit 1; fi
  done < "$RUN/jobs"
  with_lock reap_locked
fi
if [ "$MODE" = smoke ]; then SM=--smoke; SESSIONS='1 2'; SSLIM='--limit 12'; else SM=''; SESSIONS='1 2 3 4 5'; SSLIM=''; fi
export SESSIONS SM
# provenance: reuse of finished artifacts on RESUME requires the same corpus and generator/evaluator code
prov(){ { [ "$V22" != 1 ] || echo "$V22"; cat "$RUN/corpus.json" make_user.py selfstudy.py evaluate.py epi_gpu.py epi_common.py epi_mask.py; } | sha256sum | cut -c1-16; }
[ -s "$RUN/corpus.json" ] || $PY make_user.py --seed 20261009 $CORPUS_ARGS --tokenizer "$TOK" --out "$RUN/corpus.json" > "$RUN/logs/corpus.log" 2>&1
if [ "$V22" = 1 ] && ! "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("rules") != "coarse")' "$RUN/corpus.json"; then
  "$PY" make_user.py --seed 20261009 --rules coarse --tokenizer "$TOK" --out "$RUN/corpus.json" > "$RUN/logs/corpus.log" 2>&1
fi
PROV=$(prov)
if [ "${RESUME:-0}" = 1 ] && [ "$(cat "$RUN/provenance" 2>/dev/null)" != "$PROV" ]; then
  log "provenance changed: nothing is reused"; rm -f "$RUN"/status/selfstudy_s* "$RUN"/status/genome "$RUN"/status/icl
fi
echo "$PROV" > "$RUN/provenance"
# tests run with the runner's own control variables removed (they would leak into the runner unit tests)
env -u V22 -u MAX_GPUS -u ARMS -u RESUME -u SMOKE -u STUDY_ARGS -u RUN -u EPI -u LOCKS -u SM -u SESSIONS \
  $PY -m pytest -q tests > "$RUN/logs/tests.log" 2>&1 || { tail -3 "$RUN/logs/tests.log" >&2; log "CPU tests failed"; exit 1; }
finished(){ # RESUME=1 only: status 0 and every listed artifact present -> reuse instead of rerun
  [ "${RESUME:-0}" = 1 ] && [ -f "$RUN/status/$1" ] && [ "$(cat "$RUN/status/$1")" = 0 ] || return 1
  local f; shift; for f in "$@"; do [ -s "$f" ] || return 1; done
}
if [ "$V22" != 1 ]; then
for k in $SESSIONS; do
  finished "selfstudy_s$k" "$RUN/selfstudy/s$k.json" "$RUN/selfstudy/s$k.pt" && { log "reusing selfstudy_s$k"; continue; }
  gpu=$(claim); PENDING=$gpu
  JOB_NAME=selfstudy_s$k; export JOB_NAME
  launch "$gpu" "$JOB_NAME" "$PY" selfstudy.py --corpus "$RUN/corpus.json" --session "$k" \
    --out "$RUN/selfstudy/s$k.json" --pt "$RUN/selfstudy/s$k.pt" $SSLIM
done
for k in $SESSIONS; do wait_job "selfstudy_s$k" "$RUN/selfstudy/s$k.json"; done
fi
for arm in genome icl; do
  finished "$arm" "$RUN/$arm/metrics.json" $([ "$arm" = genome ] && echo "$RUN/genome/panel.pt $RUN/genome/revoke_logits.pt") && { log "reusing $arm"; continue; }
  gpu=$(claim); PENDING=$gpu; JOB_NAME=$arm; export JOB_NAME
  launch "$gpu" "$arm" "$PY" evaluate.py --run "$RUN" --arm "$arm" $SM
  wait_job "$arm" "$RUN/$arm/metrics.json"
done
# arms always rerun on resume: clear their artifacts so stale sessions can never satisfy check_job
for arm in $ARMS; do rm -rf "$RUN/$arm"; rm -f "$RUN/status/${arm}_s"* "$RUN/status/chain_$arm"*; done
# The ICL identifiability gate guards self-distillation (the teacher must carry the concept in context). Under the
# pre-registered v2.1 supervised fallback (V22=1) training targets are ground truth, so the gate is recorded, not
# enforced (it failed on the coarse corpus: EP1_PREREG.md v2.1 gate results).
if [ "$MODE" = full ] && [ "$V22" != 1 ]; then
  "$PY" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); g=json.loads((r/"genome/metrics.json").read_text()); i=json.loads((r/"icl/metrics.json").read_text()); sys.exit(0 if i["dev"]["concept"]["acc"]-g["dev"]["concept"]["acc"] >= .10-1e-12 else "Identifiability gate failed")' "$RUN"
fi
arm_chain(){
  local arm=$1 k name rc
  for k in $SESSIONS; do
    name=${arm}_s$k
    rm -f "$RUN/status/$name"
    touch "$RUN/status/$name.start"
    SESSION_JOB=$name bash -c '"$@"; rc=$?; echo "$rc" > "$RUN/status/$SESSION_JOB"; exit "$rc"' ep1 \
      "$PY" consolidate.py --run "$RUN" --arm "$arm" --session "$k" $SM $STUDY_ARGS \
      > "$RUN/logs/$name.log" 2>&1
    rc=$?
    [ "$rc" = 0 ] || return "$rc"
    check_job "$name" "$RUN/$arm/s$k/metrics.json" || return 1
  done
}
export -f arm_chain
# When both are requested, finish the main arms before claiming GPUs for the exploratory arms.
for stage in main ablation; do
  stage_arms=''
  for arm in $ARMS; do
    if [ "${V22:-0}" = 1 ]; then
      [ "$stage" = main ] || continue
    else
    case "$stage:$arm" in main:tmid_util*|main:tmid_mdl|ablation:tmid|ablation:fp4mid|ablation:loramid) continue;; esac
    fi
    stage_arms="$stage_arms $arm"
  done
  for arm in $stage_arms; do
    gpu=$(claim); PENDING=$gpu; JOB_NAME=chain_$arm; export JOB_NAME
    launch "$gpu" "$JOB_NAME" bash -c 'arm_chain "$1"' ep1 "$arm"
  done
  for arm in $stage_arms; do
    wait_job "chain_$arm" "$RUN/$arm/s${SESSIONS##* }/metrics.json"
    for k in $SESSIONS; do check_job "${arm}_s$k" "$RUN/$arm/s$k/metrics.json"; done
  done
done
"$PY" evaluate.py --run "$RUN" --summary > "$RUN/logs/summary.log" 2>&1
cat "$RUN/logs/summary.log"
if [ "$MODE" = smoke ]; then
  "$PY" -c '
import json,re,sys
import os
from pathlib import Path
r=Path(sys.argv[1]); paths=list((r/"logs").glob("*.log"))+list(r.glob("*/s*/log"))
peaks=[float(x) for p in paths for x in re.findall(r"\"(?:refresh_peak_gib|peak_gib)\"\s*:\s*([0-9.eE+-]+)",p.read_text())]
if not peaks or max(peaks)>30: sys.exit("Smoke peak absent or >30 GiB")
timings={}; coverage=[]; steps=[]
for p in (r/"logs").glob("*.log"):
 for line in p.read_text().splitlines():
  try: d=json.loads(line.removeprefix("selfstudy "))
  except ValueError: continue
  if "seconds" in d: timings.setdefault(p.stem,{}).setdefault(d.get("phase","selfstudy" if p.stem.startswith("selfstudy") else "step"),[]).append(d["seconds"])
  if "step" in d and "seconds" in d: steps.append(d["seconds"])
  if d.get("phase")=="smoke_coverage": coverage.append(dict(job=p.stem,**d))
arms=os.environ.get("ARMS","tmid fp4mid loramid").split()
coverage=[c for c in coverage if c["job"] in {f"{a}_s{k}" for a in arms for k in (1,2)}]
if sorted(c["job"] for c in coverage)!=sorted(f"{a}_s{k}" for a in arms for k in (1,2)) or not all(c["revoke_pass"] for c in coverage): sys.exit("Missing smoke coverage or revoke failure")
# effective writes: every fsa session must save a nonzero committed patch; tmid must have born slots
# a 10-step smoke cannot commit ternary slots in session 1 (support is chosen at the step-8 refresh and masters need
# ~25 steps to cross the threshold; lorafold codes appear after ~50 steps), so each FSA arm must commit a nonzero
# patch in at least one smoke session; lorafold is exempt (its code path is covered by unit tests)
fsa_arms={c["job"].rsplit("_s",1)[0] for c in coverage if not c["job"].startswith(("loramid","lorafold"))}
for arm in sorted(fsa_arms):
    if not any(c["nnz_committed"]>0 for c in coverage if c["job"].rsplit("_s",1)[0]==arm): sys.exit(f"FSA arm {arm} committed an empty patch in every smoke session")
if not all(c["births"]>0 for c in coverage if c["job"].startswith("tmid")): sys.exit("tmid smoke had no births")
if not all(c["rehydrated"] for c in coverage if c["job"].endswith("s2")): sys.exit("Missing rehydration")
if not all(c["refreshes"]>0 and c["lambda"] is not None for c in coverage if c["job"].startswith("tmid")): sys.exit("Missing ternary calibration/refresh coverage")
manifest=dict(peak_gib=max(peaks),timings=timings,smoke_coverage=coverage)
(r/"SMOKE_OK").write_text(json.dumps(manifest,indent=2)+"\n")
seconds=sum(steps)/len(steps)
print(f"Smoke peak: {max(peaks):.2f} GiB; ETA per arm: 48 steps x {seconds:.2f} seconds x 5 sessions = {48*seconds*5/3600:.2f} hours")
' "$RUN"
fi
log done
