"""Runner scheduling and job checks with mock GPU queries and detached wrappers."""
import os
import shutil
import sys

# macOS has no util-linux flock: a real stand-in for `flock -x|-n FD`. A BSD flock lock belongs to the open file
# description, which the caller shares, so it stays held after this helper exits (released when the fd closes).
FLOCK_STUB = f'''#!{sys.executable}
import fcntl, sys
fd = int(sys.argv[-1]); op = fcntl.LOCK_EX | (fcntl.LOCK_NB if '-n' in sys.argv else 0)
try:
    fcntl.flock(fd, op)
except OSError:
    sys.exit(1)
'''
from pathlib import Path
import subprocess

import pytest

RUNNER = Path(__file__).resolve().parents[1] / 'run_ep1.sh'


def runner_functions():
    source = RUNNER.read_text()
    return source[source.index('log(){'):source.index('export -f check_job')]


def shell(tmp_path, script):
    (tmp_path / 'status').mkdir()
    (tmp_path / 'logs').mkdir()
    for name in ('claimed', 'jobs', 'pids'):
        (tmp_path / name).touch()
    # Real command/status handling; mock platform-specific lock/detach commands and GPU hardware.
    setup = '''
set -eu
export RUN
EPI=$RUN; export EPI
LOCKS=$RUN/locks; export LOCKS; mkdir -p "$LOCKS"
touch "$EPI/claimed"
setsid(){ "$@"; }
sleep(){ /bin/sleep .02; }
nvidia-smi(){
  case "$1" in --query-gpu=*) echo '0, GPU-fake, 0';; *) :;; esac
}
'''
    env = dict(os.environ, RUN=str(tmp_path))
    if shutil.which('flock') is None:            # detached wrappers run a fresh bash: stub flock on PATH too
        stub = tmp_path / 'bin'
        stub.mkdir(exist_ok=True)
        (stub / 'flock').write_text(FLOCK_STUB)
        (stub / 'flock').chmod(0o755)
        env['PATH'] = f"{stub}:{env['PATH']}"
    return subprocess.run(['bash', '-c', setup + runner_functions() + script],
                          env=env, text=True, capture_output=True, timeout=10)


def test_single_gpu_scheduler_reaps_before_second_claim(tmp_path):
    result = shell(tmp_path, '''
gpu=$(claim)
[ "${gpu%%:*}" = 0 ]
first=$gpu
JOB_NAME=first; export JOB_NAME
launch "$gpu" first bash -c 'sleep 1.1; echo new > "$RUN/item"'
gpu=$(claim)
[ "${gpu%%:*}" = 0 ] && [ "$gpu" != "$first" ]
check_job first "$RUN/item"
[ ! -s "$RUN/jobs" ]
[ "$(cat "$RUN/claimed")" = "$gpu" ]
[ "$(cat "$LOCKS/gpu0/epi_owner")" = "$gpu" ]
# a stale release of the first claim must not free the second claimant's GPU
release_gpu "$first"
[ -d "$LOCKS/gpu0" ] && [ "$(cat "$RUN/claimed")" = "$gpu" ]
[ "$(wc -l < "$RUN/pids")" -eq 1 ]
''')
    assert result.returncode == 0, result.stderr


def test_job_status_and_stale_artifact_rejected(tmp_path):
    result = shell(tmp_path, '''
echo stale > "$RUN/item"
/bin/sleep .05
touch "$RUN/status/old.start"
echo 0 > "$RUN/status/old"
if check_job old "$RUN/item"; then exit 3; fi
JOB_NAME=failed; export JOB_NAME
launch 0 failed bash -c 'exit 7'
while [ ! -f "$RUN/status/failed" ]; do sleep 5; done
[ "$(cat "$RUN/status/failed")" = 7 ]
if check_job failed "$RUN/item"; then exit 4; fi
''')
    assert result.returncode == 0, result.stderr


def test_existing_run_and_full_smoke_gate(tmp_path):
    run = tmp_path / 'out/used'
    run.mkdir(parents=True)
    # the runner itself runs these tests under RESUME=1; the guard under test is the non-resume path
    env = {k: v for k, v in os.environ.items() if k not in ('RESUME', 'SMOKE')}
    env['EPI'] = str(tmp_path)
    result = subprocess.run(['bash', str(RUNNER), 'smoke', 'used'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Run already exists' in result.stderr
    result = subprocess.run(['bash', str(RUNNER), 'full', 'new'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'SMOKE_OK' in result.stderr


@pytest.mark.parametrize('arms', ['tmid fp4mid loramid', 'tmid_util01 tmid_util03',
                                 'tmid fp4mid loramid tmid_util01 tmid_util03'])
def test_smoke_manifest_peak_refresh_and_phase_timings(tmp_path, arms):
    import json
    import sys
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'selfstudy_s1.log').write_text('selfstudy ' + json.dumps({'seconds': 2., 'peak_gib': 22.}) + '\n')
    for arm in arms.split():
        for session in (1, 2):
            rows = [dict(step=1, seconds=3., peak_gib=25.), dict(phase='cache', seconds=1., peak_gib=24.),
                    dict(phase='smoke_coverage', births=10, evictions=0, nnz_committed=7, refreshes=1,
                         **{'lambda': .2}, rehydrated=session == 2, revoke_pass=True)]
            (logs / f'{arm}_s{session}.log').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    arm_log = tmp_path / 'tmid/s1/log'
    arm_log.parent.mkdir(parents=True)
    arm_log.write_text('{"refresh_peak_gib": 29.5}\n')
    script = RUNNER.read_text().split('"$PY" -c \'\nimport json,re,sys\n', 1)[1].split("\n' \"$RUN\"", 1)[0]
    code = 'import json,re,sys\n' + script
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                            env=dict(os.environ, ARMS=arms), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / 'SMOKE_OK').read_text())
    assert manifest['peak_gib'] == 29.5
    assert manifest['timings']['selfstudy_s1']['selfstudy'] == [2.]
    assert len(manifest['smoke_coverage']) == 2 * len(arms.split())
    assert '48 steps x 3.00 seconds x 5 sessions' in result.stdout
    ternary = next(a for a in arms.split() if a.startswith('tmid'))
    path = logs / f'{ternary}_s1.log'
    original = path.read_text()
    path.write_text(original.replace('"births": 10', '"births": 0'))
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                            env=dict(os.environ, ARMS=arms), capture_output=True, text=True)
    assert result.returncode != 0 and 'no births' in result.stderr
    path.write_text(original)
    arm_log.write_text('{"refresh_peak_gib": 30.1}\n')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                            env=dict(os.environ, ARMS=arms), capture_output=True, text=True)
    assert result.returncode != 0 and '>30 GiB' in result.stderr


def test_resume_with_dead_tracked_job_proceeds(tmp_path):
    """A dead pid in jobs must not abort a RESUME run; jobs are cleared."""
    run = tmp_path / 'out/ep1_smoke'
    run.mkdir(parents=True)
    (run / 'jobs').write_text('999999 0:tok old\n')
    env = {k: v for k, v in os.environ.items() if k != 'SMOKE'}
    env.update(EPI=str(tmp_path), RESUME='1', PY='false', LOCKS=str(tmp_path / 'locks'))
    if shutil.which('flock') is None:            # macOS: no util-linux; a no-op stub suffices here
        stub = tmp_path / 'bin'
        stub.mkdir()
        (stub / 'flock').write_text(FLOCK_STUB)
        (stub / 'flock').chmod(0o755)
        env['PATH'] = f"{stub}:{env['PATH']}"
    (tmp_path / 'claimed').write_text('0:tok\n')
    (tmp_path / 'locks' / 'gpu0').mkdir(parents=True)
    (tmp_path / 'locks' / 'gpu0' / 'epi_owner').write_text('0:tok\n')
    result = subprocess.run(['bash', str(RUNNER), 'smoke'], env=env, capture_output=True, text=True, timeout=20)
    # it must get past the resume guard and fail later at the (fake) interpreter, with jobs cleared
    assert (run / 'jobs').read_text() == ''
    assert 'Cannot resume' not in result.stderr
    # the dead job's GPU claim and lock dir are released, not stranded
    assert (tmp_path / 'claimed').read_text() == '' and not (tmp_path / 'locks' / 'gpu0').exists()


@pytest.mark.parametrize('arms', ['tmid fp4mid loramid', 'tmid_util03 tmid fp4mid loramid tmid_util01',
                                 'tmid_util01 tmid_util03'])
def test_selected_arm_chains_wait_for_primary_before_ablation(tmp_path, arms):
    source = RUNNER.read_text()
    block = source[source.index('# When both are requested'):source.index('"$PY" evaluate.py --run "$RUN" --summary')]
    setup = '''
set -eu
SESSIONS='1 2'
claim(){ echo fake; }
launch(){ echo "launch $2"; }
wait_job(){ echo "wait $1"; }
check_job(){ :; }
'''
    result = subprocess.run(['bash', '-c', setup + block], env=dict(os.environ, ARMS=arms, RUN=str(tmp_path)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    primary = [a for a in arms.split() if not a.startswith('tmid_util')]
    utility = [a for a in arms.split() if a.startswith('tmid_util')]
    expected = [f'{event} chain_{a}' for group in (primary, utility) for event in ('launch', 'wait') for a in group]
    assert result.stdout.splitlines() == expected


def test_pending_claim_released_when_runner_stops(tmp_path):
    result = shell(tmp_path, '''
PENDING=''
trap '[ -n "$PENDING" ] && with_lock release_gpu "$PENDING"' EXIT
gpu=$(claim); PENDING=$gpu
[ -d "$LOCKS/gpu0" ] && grep -qx "$gpu" "$EPI/claimed"
exit 0
''')
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'locks' / 'gpu0').exists()
    assert (tmp_path / 'claimed').read_text() == ''
