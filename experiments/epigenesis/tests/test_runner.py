"""Runner scheduling and job checks with mock GPU queries and detached wrappers."""
import os
from pathlib import Path
import subprocess

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
flock(){ :; }
setsid(){ "$@"; }
sleep(){ /bin/sleep .02; }
nvidia-smi(){
  case "$1" in --query-gpu=*) echo '0, GPU-fake, 0';; *) :;; esac
}
'''
    return subprocess.run(['bash', '-c', setup + runner_functions() + script],
                          env=dict(os.environ, RUN=str(tmp_path)), text=True, capture_output=True, timeout=10)


def test_single_gpu_scheduler_reaps_before_second_claim(tmp_path):
    result = shell(tmp_path, '''
gpu=$(claim)
[ "$gpu" = 0 ]
JOB_NAME=first; export JOB_NAME
launch "$gpu" first bash -c 'sleep 1.1; echo new > "$RUN/item"'
gpu=$(claim)
[ "$gpu" = 0 ]
check_job first "$RUN/item"
[ ! -s "$RUN/jobs" ]
[ "$(cat "$RUN/claimed")" = 0 ]
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
    env = dict(os.environ, EPI=str(tmp_path))
    result = subprocess.run(['bash', str(RUNNER), 'smoke', 'used'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'Run already exists' in result.stderr
    result = subprocess.run(['bash', str(RUNNER), 'full', 'new'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'SMOKE_OK' in result.stderr


def test_smoke_manifest_peak_refresh_and_phase_timings(tmp_path):
    import json
    import sys
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'selfstudy_s1.log').write_text('selfstudy ' + json.dumps({'seconds': 2., 'peak_gib': 22.}) + '\n')
    for arm in ('tmid', 'fp4mid', 'loramid'):
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
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / 'SMOKE_OK').read_text())
    assert manifest['peak_gib'] == 29.5
    assert manifest['timings']['selfstudy_s1']['selfstudy'] == [2.]
    assert len(manifest['smoke_coverage']) == 6
    assert '48 steps x 3.00 seconds x 5 sessions' in result.stdout
    arm_log.write_text('{"refresh_peak_gib": 30.1}\n')
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and '>30 GiB' in result.stderr
