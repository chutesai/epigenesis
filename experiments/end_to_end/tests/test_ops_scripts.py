"""Run real bash runners/janitors with fake GPU/python commands and real process groups."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

OPS = Path(__file__).resolve().parents[1]/'ops'


def wait_for(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.03)
    assert check(), 'timed out waiting for operation'


def lock_available(path):
    import fcntl
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
    return True


@pytest.fixture
def ops(tmp_path):
    bindir = tmp_path/'bin'
    bindir.mkdir()
    def stub(name, code):
        path = bindir/name
        path.write_text(f'#!{sys.executable}\n'+code)
        path.chmod(0o755)
        return path
    stub('nvidia-smi', "import os\nfrom pathlib import Path\nprint(Path(os.environ['FSAQ_ROOT'], 'gpus').read_text(), end='')\n")
    # Darwin lacks the Linux utilities. These stubs implement their actual primitives.
    stub('setsid', '''import os, sys
os.setsid()
with open(os.path.join(os.environ['FSAQ_ROOT'], 'sessions'), 'a') as f: f.write(str(os.getpid())+'\\n')
os.execvp(sys.argv[1], sys.argv[1:])
''')
    if sys.platform == 'darwin':
        # /bin/ps is unavailable in the managed sandbox. Inspect only this fixture's
        # known pids using macOS's targeted API; real sessions/signals stay in use.
        stub('ps', '''import ctypes, json, os, sys
from pathlib import Path
class Info(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ('flags','status','xstatus','pid','ppid','uid','gid',
        'ruid','rgid','svuid','svgid','rfu')] + [('comm', ctypes.c_char*16), ('name', ctypes.c_char*32)] + [
        (n, ctypes.c_uint32) for n in ('nfiles','pgid','pjobc','e_tdev','e_tpgid')] + [
        ('nice',ctypes.c_int32), ('sec',ctypes.c_uint64), ('usec',ctypes.c_uint64)]
lib = ctypes.CDLL('/usr/lib/libproc.dylib')
def info(pid):
    value = Info()
    return value if lib.proc_pidinfo(pid, 3, 0, ctypes.byref(value), ctypes.sizeof(value)) else None
if '-p' in sys.argv:
    pid = int(sys.argv[sys.argv.index('-p')+1]); value = info(pid)
    if not value: sys.exit(1)
    print(str(value.sec)+'-'+str(value.usec) if 'lstart=' in sys.argv else value.pgid)
else:
    root = Path(os.environ['FSAQ_ROOT'])
    sessions, launches = root/'sessions', root/'launches'
    pids = set(map(int, sessions.read_text().splitlines())) if sessions.exists() else set()
    if launches.exists(): pids.update(json.loads(s)['pid'] for s in launches.read_text().splitlines())
    for pid in pids:
        value = info(pid)
        if value: print(value.pgid, 'Z' if value.status == 5 else 'S')
''')
    stub('flock', '''import fcntl, os, sys
from pathlib import Path
mode, fd = sys.argv[1], int(sys.argv[2])
if mode == '-x' and os.environ.get('FLOCK_NOTIFY'): Path(os.environ['FLOCK_NOTIFY']).touch()
flags = fcntl.LOCK_UN if mode == '-u' else fcntl.LOCK_EX
if mode == '-n': flags |= fcntl.LOCK_NB
try: fcntl.flock(fd, flags)
except BlockingIOError: sys.exit(1)
''')
    python = stub('python', '''import json, os, signal, sys, time
from pathlib import Path
root = Path(os.environ['FSAQ_ROOT'])
assert os.environ['PYTHONFAULTHANDLER'] == '1'
import resource
assert resource.getrlimit(resource.RLIMIT_CORE)[0] == 0
def abort(signum, frame):
    with (root/'aborts').open('a') as f: f.write(str(os.getpid())+'\\n')
signal.signal(signal.SIGABRT, abort)
with (root/'launches').open('a') as f:
    f.write(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], argv=sys.argv[1:]))+'\\n')
print('HEARTBEAT fake', flush=True)
if sys.argv[1] == 'exit': sys.exit(0)
while not (root/'finish').exists(): time.sleep(.03)
''')
    (tmp_path/'gpus').write_text('0, 0\n1, 0\n')
    (tmp_path/'queue').write_text('block|job\n')
    env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'], PY=str(python),
        FSAQ_ROOT=str(tmp_path), FSAQ_SPACING_SECONDS='.05', FSAQ_POLL_SECONDS='.05',
        JANITOR_POLL_SECONDS='.05', HANG_MIN='0', ABRT_WAIT_STEPS='1', KILL_WAIT_STEPS='3',
        KILL_WAIT_SECONDS='.03', STALE_LOCK_SECONDS='0', GROUP_GRACE_STEPS='2',
        GROUP_TERM_STEPS='3', GROUP_WAIT_SECONDS='.03')
    processes = []
    def start(script, *args, **extra):
        p = subprocess.Popen(['bash', str(OPS/script), *args], env=dict(env, **extra),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        processes.append(p)
        return p
    def records():
        import json
        p = tmp_path/'launches'
        return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
    def owner():
        p = tmp_path/'run_own/job'
        return p.read_text().strip() if p.exists() else ''
    def age_log():
        log = tmp_path/'logs/job.log'
        wait_for(lambda: log.exists() and 'HEARTBEAT ' in log.read_text())
        os.utime(log, (time.time()-120,)*2)
    yield tmp_path, env, start, records, owner, age_log
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for r in records():
        try: os.killpg(r['pgid'], signal.SIGKILL)
        except ProcessLookupError: pass
    sessions = tmp_path/'sessions'
    recorded_groups = {r['pgid'] for r in records()}
    for pid in map(int, sessions.read_text().splitlines() if sessions.exists() else []):
        if pid in recorded_groups:
            continue
        live = subprocess.run(['bash', '-c',
            f'source {shlex.quote(str(OPS/"ops_common.sh"))}; group_alive "$1"', 'test', str(pid)],
            cwd=tmp_path, env=env)
        if live.returncode != 0:
            continue
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    for p in processes:
        try: p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()


def test_double_launch_and_singleton(ops):
    root, env, start, records, owner, age = ops
    (root/'queue').write_text('block|job\nblock|job\nblock|../bad\n')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and len(records()) == 1)
    duplicate = start('fsaq4.sh', 'queue')
    assert duplicate.wait(timeout=3) == 1
    time.sleep(.2)
    assert len(records()) == 1
    assert (root/'locks/job_job').is_dir()
    assert not (root/'bad.log').exists()
    # Log removal alone must not permit a second launch while the claim is live.
    (root/'logs/job.log').unlink()
    time.sleep(.2)
    assert len(records()) == 1


def test_cleanup_only_removes_matching_identity(ops):
    root, env, *_ = ops
    for directory in ('locks/gpu0', 'locks/job_job', 'run_own'):
        (root/directory).mkdir(parents=True, exist_ok=True)
    for file in ('locks/gpu0/owner', 'locks/job_job/owner', 'run_own/job'):
        (root/file).write_text('new owner\n')
    subprocess.run(['bash', '-c', f'source {shlex.quote(str(OPS/"ops_common.sh"))}; '
                    'cleanup_owner job 0 "old owner"'], cwd=root, env=env, check=True)
    assert (root/'run_own/job').read_text() == 'new owner\n'
    assert (root/'locks/gpu0/owner').read_text() == 'new owner\n'
    assert (root/'locks/job_job/owner').read_text() == 'new owner\n'


def test_hang_kills_whole_sh_group_and_relaunches(ops):
    root, env, start, records, owner, age = ops
    # Shell leader plus two children, all in the recorded session.
    (root/'queue').write_text(f'SH {shlex.quote(env["PY"])} block & {shlex.quote(env["PY"])} block & wait|job\n')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and len(records()) == 2)
    first_owner = owner()
    first_group = int(first_owner.split('|')[1])
    age()
    start('lock_janitor3.sh')
    wait_for(lambda: len(records()) == 4)
    assert owner() != first_owner
    assert all(r['pgid'] == first_group for r in records()[:2])
    assert all(r['pgid'] != first_group for r in records()[2:])
    archives = list((root/'logs').glob('job.log.hang_*_1'))
    assert len(archives) == 1
    assert 'janitor:' not in archives[0].read_text()
    assert (root/'logs/job.hangcount').read_text().strip() == '1'
    dead = subprocess.run(['bash', '-c', f'source {shlex.quote(str(OPS/"ops_common.sh"))}; '
        'group_alive "$1"', 'test', str(first_group)], cwd=root, env=env)
    assert dead.returncode == 1
    # Late cleanup with the old identity must preserve the replacement records.
    gpu = records()[0]['gpu']
    subprocess.run(['bash', '-c', f'source {shlex.quote(str(OPS/"ops_common.sh"))}; '
        'cleanup_owner job "$1" "$2"', 'test', gpu, first_owner], cwd=root, env=env, check=True)
    assert owner() != first_owner and owner()


def test_retry_cap_alerts_without_killing(ops):
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    old = owner()
    (root/'logs/job.hangcount').write_text('2\n')
    age()
    start('lock_janitor3.sh')
    wait_for(lambda: (root/'logs/lock_janitor.log').exists() and
             'alert only' in (root/'logs/lock_janitor.log').read_text())
    assert owner() == old and len(records()) == 1
    os.kill(int(old.split('|')[0]), 0)
    assert (root/'logs/job.log').exists()
    assert not list((root/'logs').glob('job.log.hang_*'))
    assert not (root/'aborts').exists()


def test_alive_after_kill_does_not_rename(ops):
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    old = owner()
    age()
    # Simulate an uninterruptible process by suppressing only the janitor's signals.
    startup = root/'no_kill.sh'
    startup.write_text('kill() { case "$*" in *ABRT*|*KILL*) return 0;; *) builtin kill "$@";; esac; }\n')
    start('lock_janitor3.sh', BASH_ENV=str(startup))
    wait_for(lambda: (root/'logs/lock_janitor.log').exists() and
             'still alive after kill' in (root/'logs/lock_janitor.log').read_text())
    assert owner() == old and len(records()) == 1
    assert (root/'logs/job.log').exists()
    assert not list((root/'logs').glob('job.log.hang_*'))


def test_stale_lock_with_live_owner_is_preserved(ops):
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    old = owner()
    gpu = records()[0]['gpu']
    lock = root/f'locks/gpu{gpu}'
    os.utime(lock, (time.time()-3600,)*2)
    start('lock_janitor3.sh', HANG_MIN='45')
    time.sleep(.5)  # > three accelerated stale checks, while fake GPU stays idle.
    assert lock.is_dir() and (lock/'owner').read_text().strip() == old


def test_reserved_gpu_and_fresh_hangcount_reset(ops):
    root, env, start, records, owner, age = ops
    (root/'reserved_gpus.txt').write_text('0\n')
    (root/'logs').mkdir()
    (root/'logs/job.hangcount').write_text('2')
    (root/'logs/job.log.hang_old_1').write_text('old archive')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    assert records()[0]['gpu'] == '1'
    assert not (root/'logs/job.hangcount').exists()


def test_stale_dead_owner_needs_idle_gpu_and_honors_skip(ops):
    root, env, start, *_ = ops
    for gpu in (0, 1):
        lock = root/f'locks/gpu{gpu}'
        lock.mkdir(parents=True)
        (lock/'owner').write_text(f'99999999|99999999|dead|job{gpu}\n')
        os.utime(lock, (time.time()-3600,)*2)
    (root/'gpus').write_text('0, 2000\n1, 0\n')
    janitor = start('lock_janitor3.sh', '1')
    singleton = root/'locks/lock_janitor3.lock'
    wait_for(lambda: singleton.exists() and not lock_available(singleton))
    duplicate = start('lock_janitor3.sh')
    assert duplicate.wait(timeout=3) == 1
    time.sleep(.5)
    assert all((root/f'locks/gpu{g}').is_dir() for g in (0, 1))
    janitor.terminate(); janitor.wait(timeout=3)
    wait_for(lambda: lock_available(singleton))
    start('lock_janitor3.sh')
    wait_for(lambda: not (root/'locks/gpu1').exists())
    assert (root/'locks/gpu0').is_dir()


def test_changed_starttime_prevents_hang_signals(ops):
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    old = owner()
    fields = old.split('|')
    fields[2] = 'different-starttime'
    (root/'run_own/job').write_text('|'.join(fields)+'\n')
    age()
    start('lock_janitor3.sh')
    time.sleep(.5)
    os.kill(int(fields[0]), 0)
    assert not (root/'aborts').exists()
    assert (root/'logs/job.log').exists()
    assert not list((root/'logs').glob('job.log.hang_*'))


def test_completed_python_job_cleans_up_and_log_prevents_rerun(ops):
    root, env, start, records, *_ = ops
    (root/'queue').write_text('exit|job\n')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: records())
    wait_for(lambda: not (root/'run_own/job').exists() and not (root/'locks/job_job').exists())
    assert not (root/f'locks/gpu{records()[0]["gpu"]}').exists()
    time.sleep(.2)
    assert len(records()) == 1 and (root/'logs/job.log').exists()


def test_ownerless_locks_require_age_idle_checks_and_empty_directory(ops):
    root, env, start, *_ = ops
    for directory in ('locks/gpu0', 'locks/gpu1', 'locks/job_orphan', 'locks/job_nonempty',
                      'locks/job_malformed', 'locks/job_fresh'):
        (root/directory).mkdir(parents=True)
        os.utime(root/directory, (time.time()-3600,)*2)
    (root/'locks/job_nonempty/keep').touch()
    (root/'locks/job_malformed/owner').write_text('malformed')
    # Restore ages changed by adding content; the fresh lock stays younger than threshold.
    for name in ('job_nonempty', 'job_malformed'):
        os.utime(root/'locks'/name, (time.time()-3600,)*2)
    os.utime(root/'locks/job_fresh', None)
    (root/'gpus').write_text('0, 2000\n1, 0\n')
    janitor = start('lock_janitor3.sh', STALE_LOCK_SECONDS='60', JANITOR_POLL_SECONDS='.2')
    wait_for(lambda: not (root/'locks/job_orphan').exists())
    # One pass cannot remove a legacy GPU lock; three idle observations are required.
    assert (root/'locks/gpu1').exists()
    wait_for(lambda: not (root/'locks/gpu1').exists())
    assert all((root/d).exists() for d in ('locks/gpu0', 'locks/job_nonempty',
                                         'locks/job_malformed', 'locks/job_fresh'))
    janitor.terminate(); janitor.wait(timeout=3)


def test_ownerless_gpu_rechecks_ownership_under_lock(ops):
    import fcntl
    root, env, start, *_ = ops
    lock = root/'locks/gpu0'
    lock.mkdir(parents=True)
    os.utime(lock, (time.time()-3600,)*2)
    notify = root/'waiting'
    with (root/'locks/state.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        start('lock_janitor3.sh', FLOCK_NOTIFY=str(notify))
        wait_for(notify.exists)
        (lock/'owner').write_text('new owner')
        fcntl.flock(stream, fcntl.LOCK_UN)
    time.sleep(.2)
    assert (lock/'owner').read_text() == 'new owner'


def test_session_failure_releases_claims_and_allows_retry(ops):
    root, env, start, records, owner, _ = ops
    startup = root/'bad_ps.sh'
    startup.write_text('ps() { case "$*" in *-p*) return 1;; *) command ps "$@";; esac; }\n')
    runner = start('fsaq4.sh', 'queue', BASH_ENV=str(startup))
    wait_for(lambda: (root/'logs/job.log').exists())
    (root/'queue.stop').touch()
    wait_for(lambda: list((root/'logs').glob('job.log.launchfail_*')), timeout=15)
    assert not records() and not owner()
    assert not (root/'locks/job_job').exists()
    assert not (root/'locks/gpu0').exists()
    assert not (root/'logs/job.log').exists()
    runner.wait(timeout=3)
    (root/'queue.stop').unlink()
    start('fsaq4.sh', 'queue')
    wait_for(lambda: records() and owner())
    assert len(records()) == 1


def test_gate_expires_if_supervisor_dies_before_ready(ops):
    root, env, start, records, _, _ = ops
    startup = root/'slow_ps.sh'
    # Delay only the job's PGID check; startup identity lookup still succeeds.
    startup.write_text('ps() { case "$*" in *pgid=*) command sleep 3;; esac; command ps "$@"; }\n')
    runner = start('fsaq4.sh', 'queue', BASH_ENV=str(startup),
                   GATE_WAIT_STEPS='5', GATE_WAIT_SECONDS='.03')
    wait_for(lambda: (root/'sessions').exists() and (root/'locks/job_job/owner').exists())
    gate_pid = int((root/'sessions').read_text().splitlines()[0])
    # Kill the supervisor plus scheduler, leaving the separately established gate session.
    os.killpg(runner.pid, signal.SIGKILL)
    runner.wait(timeout=3)
    wait_for(lambda: subprocess.run(['bash', '-c',
        f'source {shlex.quote(str(OPS/"ops_common.sh"))}; group_alive "$1"', 'test', str(gate_pid)],
        cwd=root, env=env).returncode == 1)
    assert not records() and not (root/'locks/job_job/ready').exists()


def test_fresh_heartbeat_while_waiting_for_lock_prevents_signal(ops):
    import fcntl
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    old = owner()
    age()
    notify = root/'waiting'
    with (root/'locks/state.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        start('lock_janitor3.sh', HANG_MIN='1', FLOCK_NOTIFY=str(notify))
        wait_for(notify.exists)
        with (root/'logs/job.log').open('a') as log:
            log.write('HEARTBEAT recovered\n')
        fcntl.flock(stream, fcntl.LOCK_UN)
    time.sleep(.3)
    assert owner() == old and len(records()) == 1
    assert not (root/'aborts').exists()
    assert not list((root/'logs').glob('job.log.hang_*'))


def test_queue_word_splitting_globs_crlf_and_dot_names(ops):
    root, env, start, records, *_ = ops
    (root/'patches').mkdir()
    for name in ('a.pt', 'b.pt', 'space name.pt'):
        (root/'patches'/name).touch()
    (root/'queue').write_bytes(b'exit|.\r\nexit|..\r\nexit|.hidden\r\n'
                              b'exit\tpatches/*.pt missing?.pt "two words"|job\r\n')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: records())
    assert records()[0]['argv'] == ['exit', 'patches/a.pt', 'patches/b.pt',
        'patches/space name.pt', 'missing?.pt', '"two', 'words"', '--out', 'out/job.json']
    assert not (root/'logs/.hidden.log').exists()
    assert not (root/'logs/...log').exists()


def test_py_is_one_path_including_spaces(ops):
    root, env, start, records, *_ = ops
    bad = start('fsaq4.sh', 'queue', PY=env['PY']+' -u')
    assert bad.wait(timeout=3) == 2
    spaced = root/'python with spaces'
    spaced.symlink_to(env['PY'])
    start('fsaq4.sh', 'queue', PY=str(spaced))
    wait_for(lambda: records())


@pytest.mark.parametrize('script,key,value', [
    ('lock_janitor3.sh', 'MAX_RETRY', '2x'), ('lock_janitor3.sh', 'HANG_MIN', '45m'),
    ('lock_janitor3.sh', 'HANG_MIN', '08'), ('lock_janitor3.sh', 'HANG_MIN', '999999999999999999999'),
    ('lock_janitor3.sh', 'STALE_LOCK_SECONDS', '-1'), ('lock_janitor3.sh', 'ABRT_WAIT_STEPS', '1x'),
    ('lock_janitor3.sh', 'KILL_WAIT_STEPS', '08'), ('lock_janitor3.sh', 'KILL_WAIT_SECONDS', 'NaN'),
    ('lock_janitor3.sh', 'JANITOR_POLL_SECONDS', '1s'),
    ('fsaq4.sh', 'FSAQ_SPACING_SECONDS', '-1'), ('fsaq4.sh', 'FSAQ_POLL_SECONDS', '1s'),
    ('fsaq4.sh', 'GATE_WAIT_STEPS', '08'), ('fsaq4.sh', 'GATE_WAIT_SECONDS', 'infinity'),
    ('fsaq4.sh', 'GROUP_GRACE_STEPS', 'x'), ('fsaq4.sh', 'GROUP_TERM_STEPS', '-1'),
    ('fsaq4.sh', 'GROUP_WAIT_SECONDS', '1x'),
])
def test_invalid_numeric_settings_exit_before_work(ops, script, key, value):
    root, env, start, records, *_ = ops
    process = start(script, *(['queue'] if script == 'fsaq4.sh' else []), **{key: value})
    assert process.wait(timeout=3) == 2
    assert not records() and not (root/'locks').exists()


@pytest.mark.parametrize('script', ['fsaq4.sh', 'lock_janitor3.sh'])
def test_symlink_path_and_missing_common_fail_fast(ops, script):
    root, env, start, records, owner, _ = ops
    linked = root/'bin'/('linked_'+script)
    linked.symlink_to(OPS/script)
    process = start(str(linked), *(['queue'] if script == 'fsaq4.sh' else []))
    if script == 'fsaq4.sh':
        wait_for(lambda: records() and owner())
    else:
        wait_for(lambda: (root/'locks/lock_janitor3.lock').exists() and
                 not lock_available(root/'locks/lock_janitor3.lock'))
    process.terminate(); process.wait(timeout=3)
    isolated = root/'isolated'
    isolated.mkdir()
    copy = isolated/script
    copy.write_text((OPS/script).read_text())
    broken = start(str(copy), *(['queue'] if script == 'fsaq4.sh' else []))
    assert broken.wait(timeout=3) == 1


def test_leftover_group_is_bounded_after_leader_exit_without_heartbeat(ops):
    root, env, start, records, owner, _ = ops
    (root/'queue').write_text(f'SH trap "" TERM; {shlex.quote(env["PY"])} block >/dev/null & exit 0|job\n')
    start('fsaq4.sh', 'queue')
    wait_for(lambda: records())
    wait_for(lambda: not (root/'locks/job_job').exists() and not owner())
    assert not (root/f'locks/gpu{records()[0]["gpu"]}').exists()
    assert 'HEARTBEAT' not in (root/'logs/job.log').read_text()
    dead = subprocess.run(['bash', '-c', f'source {shlex.quote(str(OPS/"ops_common.sh"))}; '
        'group_alive "$1"', 'test', str(records()[0]['pgid'])], cwd=root, env=env)
    assert dead.returncode == 1


def test_hang_archive_collision_uses_unique_name_and_retries(ops):
    root, env, start, records, owner, age = ops
    start('fsaq4.sh', 'queue')
    wait_for(lambda: owner() and records())
    startup = root/'fixed_date.sh'
    startup.write_text('date() { if [ "${*: -1}" = "+%Y%m%dT%H%M%SZ" ]; then '
                       'echo FIXED; else command date "$@"; fi; }\n')
    existing = root/'logs/job.log.hang_FIXED_1'
    existing.write_text('preserve existing archive')
    age()
    start('lock_janitor3.sh', BASH_ENV=str(startup))
    wait_for(lambda: len(records()) == 2)
    assert existing.read_text() == 'preserve existing archive'
    assert (root/'logs/job.log.hang_FIXED_1.1').exists()
    assert (root/'logs/job.hangcount').read_text().strip() == '1'


@pytest.mark.parametrize('script', ['fsaq4.sh', 'lock_janitor3.sh'])
@pytest.mark.parametrize('via_bash', [False, True])
def test_scripts_invoked_through_path(ops, script, via_bash):
    root, env, *_ = ops
    (root/'queue.stop').touch()
    (root/'janitor.stop').touch()
    command = ['bash', script, 'queue'] if via_bash else ['bash', '-c', 'exec "$1" queue', 'test', script]
    result = subprocess.run(command, env=dict(env, PATH=str(OPS)+os.pathsep+env['PATH']),
                            cwd=root, capture_output=True, timeout=3)
    assert result.returncode == 0, result.stderr.decode()


def test_documented_legacy_sweep_refuses_records_and_preserves_nonempty_or_busy_locks(ops):
    root, env, *_ = ops
    for directory in ('run_own', 'locks/gpu0', 'locks/gpu1', 'locks/gpu2', 'locks/gpu3'):
        (root/directory).mkdir(parents=True, exist_ok=True)
    (root/'locks/gpu2/owner').write_text('existing owner')
    (root/'locks/gpu3/keep').touch()
    (root/'gpus').write_text('0, 0\n1, 2000\n2, 0\n3, 0\n')
    readme = (OPS/'README.md').read_text()
    shell = readme.split("<<'BASH'\n", 1)[1].split('\nBASH\n', 1)[0]
    # Retain all three independent GPU queries but shorten only their spacing.
    startup = root/'quick_sweep.sh'
    startup.write_text('sleep() { command sleep .01; }\n')
    sweep_env = dict(env, OPS_DIR=str(OPS), BASH_ENV=str(startup))
    record = root/'run_own/legacy'
    record.touch()
    result = subprocess.run(['bash', '-c', shell], env=sweep_env, capture_output=True, timeout=5)
    assert result.returncode == 1 and b'inspect remaining ownership record' in result.stderr
    assert (root/'locks/gpu0').exists()
    record.unlink()
    result = subprocess.run(['bash', '-c', shell], env=sweep_env, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode()
    assert not (root/'locks/gpu0').exists()
    assert (root/'locks/gpu1').is_dir()
    assert (root/'locks/gpu2/owner').read_text() == 'existing owner'
    assert (root/'locks/gpu3/keep').exists()
