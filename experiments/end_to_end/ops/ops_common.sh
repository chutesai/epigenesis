#!/bin/bash
# Shared identity and serialization helpers. Production targets Linux (/proc).
valid_name() { [[ "$1" =~ ^[A-Za-z0-9_.-]+$ && "$1" != .* ]]; }
validate_uint() {
  local key value
  for key in "$@"; do
    value=${!key}
    # Reject leading zeroes too: Bash arithmetic would interpret them as octal.
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 9 ]] || {
      echo "invalid $key: expected an unsigned decimal integer (at most 9 digits)" >&2
      return 2
    }
  done
}
validate_seconds() {
  local key value
  for key in "$@"; do
    value=${!key}
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ || "$value" =~ ^[.][0-9]+$ ]] || {
      echo "invalid $key: expected nonnegative seconds" >&2
      return 2
    }
  done
}
unique_archive() {
  local base=$1 candidate=$1 suffix=0
  while [ -e "$candidate" ] || [ -L "$candidate" ]; do
    suffix=$((suffix+1)); candidate="$base.$suffix"
  done
  printf '%s\n' "$candidate"
}
process_start() {
  local stat rest
  if [ -r "/proc/$1/stat" ]; then
    stat=$(cat "/proc/$1/stat") || return 1
    rest=${stat##*) }
    # starttime is field 22, i.e. field 20 after pid and comm.
    printf '%s\n' "$rest" | awk '{print $20}'
  else
    # Allows CPU process tests on hosts without /proc.
    LC_ALL=C ps -p "$1" -o lstart= 2>/dev/null | tr -d ' \n'
  fi
}
owner_identity() {
  local pid pgid start name extra
  IFS='|' read -r pid pgid start name extra <<< "$1"
  [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ && -n "$start" && -z "$extra" ]] &&
    [ "$pid" -gt 1 ] && [ "$pgid" -gt 1 ] && valid_name "$name"
}
owner_alive() {
  owner_identity "$1" || return 1
  local pid pgid start name current
  IFS='|' read -r pid pgid start name <<< "$1"
  kill -0 "$pid" 2>/dev/null || return 1
  current=$(process_start "$pid")
  [ -n "$current" ] && [ "$current" = "$start" ]
}
owner_group_matches() {
  owner_alive "$1" || return 1
  local pid pgid start name actual
  IFS='|' read -r pid pgid start name <<< "$1"
  actual=$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')
  [ "$actual" = "$pgid" ] && [ "$pid" = "$pgid" ]
}
group_alive() {
  # Zombies cannot execute or write; don't let them prevent a verified retry.
  local snapshot
  # Failed inspection is not evidence of death.
  snapshot=$(ps -eo pgid=,stat=) || return 0
  printf '%s\n' "$snapshot" | awk -v g="$1" '$1 == g && $2 !~ /^Z/ {found=1} END {exit !found}'
}
state_lock() { exec 8>locks/state.lock; flock -x 8; }
state_unlock() { flock -u 8; exec 8>&-; }
owner_matches() { [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]; }
release_dir() {
  # Caller holds state.lock, so a new owner cannot race this comparison/removal.
  if owner_matches "$1/owner" "$2"; then
    rm -f "$1/owner" "$1/ready"
    rmdir "$1" 2>/dev/null || true
  fi
}
cleanup_owner() {
  local name=$1 gpu=$2 owner=$3
  state_lock
  if owner_matches "run_own/$name" "$owner"; then rm -f "run_own/$name"; fi
  release_dir "locks/gpu$gpu" "$owner"
  release_dir "locks/job_$name" "$owner"
  state_unlock
}
reserved_gpu() {
  case " ${SKIP:-} " in *" $1 "*) return 0;; esac
  grep -qx "$1" reserved_gpus.txt 2>/dev/null
}
file_age_seconds() {
  local stamp
  stamp=$(stat -c %Y "$1" 2>/dev/null) || stamp=$(stat -f %m "$1") || return 1
  printf '%s\n' "$(( $(date +%s) - stamp ))"
}
