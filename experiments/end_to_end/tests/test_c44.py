"""C44: real process interruption/recovery through the CPU training harness."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_e2e_fsa import E, cpu_harness
import c44_runtime as R


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    elif isinstance(a, (torch.Generator,)):
        equal_state(a.get_state(), b.get_state())
    elif hasattr(a, 'getstate'):
        equal_state(a.getstate(), b.getstate())
    else:
        assert a == b


def worker(root, flags):
    from test_e2e_fsa import TOK
    torch.set_num_threads(1)
    np.random.seed(1234)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patcher = pytest.MonkeyPatch()
    encode = type(TOK).encode
    patcher.setattr(type(TOK), 'encode', lambda self, text, **kw: encode(self, text.replace('—', '-'), **kw))
    run, paths = cpu_harness.__wrapped__(root, patcher)
    if '--resident-case' in flags:
        flags.remove('--resident-case')
        resident = root / 'resident_patches' / 'final.pt'
        if not resident.exists():
            run(['--steps', '2', '--train-person', '0', '--sparse-state'], 'resident')
        flags += ['--resident-patch', str(resident), '--train-person', '1']
    if '--teacher-case' in flags:
        flags.remove('--teacher-case')
        teacher = root / 'teacher_patches' / 'final.pt'
        if not teacher.exists():
            run(['--steps', '2', '--method', 'lora', '--optim', 'adam', '--budget', '0',
                 '--bop-tau-abs', '0', '--rank', '2'], 'teacher')
        flags += ['--teacher-patch', str(teacher), '--distill-mu', '.2', '--teacher-fade', '.5']
    if '--anchor-case' in flags:

        flags.remove('--anchor-case')
        flags += ['--anchor', paths['anchor'], '--anchor-len', '16']
    if '--pareto-case' in flags:
        flags.remove('--pareto-case')
        patcher.setattr(E, 'PARETO_EM', (-1.,))
        original_main = E.main
        def main(argv=None):
            argv = list(sys.argv[1:] if argv is None else argv)
            return original_main([flag for flag in argv if flag != '--no-pareto'])
        patcher.setattr(E, 'main', main)
    original = R.atomic_save
    stop = int(os.environ.get('C44_STOP_STEP', '0'))
    def save(path, state):
        original(path, state)
        if stop and state['step'] >= stop:
            os._exit(77)
    patcher.setattr(R, 'atomic_save', save)
    run(flags, 'run')


def process(root, flags, stop=0):
    env = dict(os.environ, C44_STOP_STEP=str(stop), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root), *flags],
                               env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert completed.returncode == (77 if stop else 0), completed.stdout
    return completed.stdout


@pytest.mark.parametrize('flags,stop', [
    (['--sparse-state', '--patch-scale', 'rowlearn', '--scale-lr', '.002', '--refresh-every', '2'], 3),
    (['--sparse-state', '--patch-scale', 'rowlearn', '--budget', '0', '--bop-accum', '4'], 3),
    (['--sparse-state', '--optim', 'adam', '--fmt', 'quinary', '--bop-tau-abs', '0'], 3),
    (['--sparse-state', '--refresh-every', '2', '--bop-tau-abs', '0', '--bop-tstat', '2', '--write-diagnostics'], 3),
    (['--method', 'lora', '--optim', 'adam', '--budget', '0', '--bop-tau-abs', '0', '--rank', '2'], 3),
    (['--factorial-cell', 'sparse_fp', '--factorial-lrs', '.001', '.002', '--optim', 'adam',
      '--budget', '0', '--bop-tau-abs', '0', '--anchor-case', '--local-lambda', '1', '--comp-curriculum'], 8),
    (['--sparse-state', '--resident-case'], 3),
    (['--sparse-state', '--teacher-case'], 3),
    (['--sparse-state', '--anchor-case', '--support-score', 'netcost', '--local-lambda', '1',
      '--local-target', '.0001', '--saturation-sampling', '--pareto-case', '--refresh-every', '2'], 3),
    (['--sparse-state', '--budget', '0', '--bop-accum', '4', '--bop-tstat', '2'], 3),
    (['--sparse-state', '--adam-phase', '2', '--bop-tstat', '2'], 3),
    (['--sparse-state', '--anchor-case', '--unit-burst', '1', '--unit-burst-thr', '.01', '--unit-burst-random'], 3),
])
def test_process_resume_is_bit_exact(tmp_path, flags, stop):
    args = ['--steps', '6', '--ckpt-every', '1', *flags]
    uninterrupted = process(tmp_path, args)
    out = tmp_path / 'run_result.json'
    expected_json = out.read_bytes()
    (tmp_path/'expected.json').write_bytes(expected_json)
    expected_result = [line for line in uninterrupted.splitlines() if line.startswith('RESULT ')][-1]
    paths = sorted((tmp_path / 'run_patches').rglob('final.pt'))
    expected = {str(path): torch.load(path, weights_only=False) for path in paths}
    expected_checkpoint = torch.load(tmp_path / 'run_patches' / 'resume.pt', weights_only=False)
    process(tmp_path, args, stop)
    checkpoint = torch.load(tmp_path / 'run_patches' / 'resume.pt', weights_only=False)
    assert checkpoint['step'] == stop
    resumed = process(tmp_path, [*args, '--resume', 'auto'])
    assert 'RESUMED ' in resumed
    equal_state(json.loads(expected_json), json.loads(out.read_bytes()))
    assert out.read_bytes() == expected_json
    assert [line for line in resumed.splitlines() if line.startswith('RESULT ')][-1] == expected_result
    for path in paths:
        equal_state(expected[str(path)], torch.load(path, weights_only=False))
    equal_state(expected_checkpoint, torch.load(tmp_path / 'run_patches' / 'resume.pt', weights_only=False))
    assert not list((tmp_path / 'run_patches').rglob('*.tmp'))
    if '--factorial-lrs' in flags:
        assert checkpoint['candidate'] == 1 and checkpoint['active']['step'] == 2
        assert 'lr_0/result.json' not in resumed  # candidate 0 was skipped
        assert len(list((tmp_path / 'run_patches').rglob('resume.pt'))) == 1


def test_heartbeat_shared_rate_limit_and_progress(monkeypatch):
    now, lines = [0.], []
    hb = R.Heartbeat(clock=lambda: now[0], emit=lambda line, **kw: lines.append(line))
    for phase, at in [('train', 119), ('gen', 120), ('qa', 121), ('bench', 239), ('damage', 240), ('train', 360)]:
        now[0] = at
        hb(phase, at, 400)
    rows = [json.loads(line.removeprefix('HEARTBEAT ')) for line in lines]
    assert [r['t'] for r in rows] == [120, 240, 360]
    assert [r['phase'] for r in rows] == ['gen', 'damage', 'train']
    monkeypatch.setattr(R, 'heartbeat', hb)
    now[0] = 480
    assert list(R.progress(range(3), 'two_hop')) == [0, 1, 2]
    assert json.loads(lines[-1].removeprefix('HEARTBEAT ')) == dict(phase='two_hop', done=1, total=3, t=480)


def test_checkpoint_config_and_atomic_failure(tmp_path, monkeypatch):
    def manager(*extra):
        return R.Checkpoints(E.build_parser().parse_args(['--out', str(tmp_path/'x.json'),
            '--patch-dir', str(tmp_path), '--ckpt-every', '2', *extra]))
    m = manager()
    assert manager('--resume', 'auto').saved is None
    assert not m.due(1) and m.due(2)
    m.save(2, payload=torch.tensor([3]))
    previous = m.path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('write failed')
    with monkeypatch.context() as patcher:
        patcher.setattr(torch, 'save', fail)
        with pytest.raises(OSError, match='write failed'):
            m.save(4)
    assert m.path.read_bytes() == previous and not list(tmp_path.glob('*.tmp'))
    loaded = manager('--resume', 'auto', '--out', str(tmp_path/'different.json'))
    assert loaded.saved['step'] == 2
    with pytest.raises(SystemExit, match='config mismatch.*steps'):
        manager('--resume', 'auto', '--steps', '7')
    with pytest.raises(SystemExit, match='finite'):
        manager('--ckpt-minutes', 'nan')
    m = manager('--ckpt-every', '0', '--ckpt-minutes', '1')
    with monkeypatch.context() as patcher:
        patcher.setattr(R.time, 'monotonic', lambda: m.last_time+60)
        assert m.due(1)


def test_checkpointing_preserves_training_and_rng(cpu_harness):
    run, _ = cpu_harness
    _, first = run(['--steps', '4', '--sparse-state'], 'without')
    rng = torch.get_rng_state().clone()
    state = torch.load(first, weights_only=False)
    _, second = run(['--steps', '4', '--sparse-state', '--ckpt-every', '1'], 'with')
    equal_state(state, torch.load(second, weights_only=False))
    assert torch.equal(rng, torch.get_rng_state())


def test_all_global_rng_streams_restore():
    saved = R.rng_state()
    expected = (random_draws(), torch.rand(3))
    R.restore_rng(saved)
    actual = (random_draws(), torch.rand(3))
    assert expected[0] == actual[0]
    equal_state(expected[1], actual[1])
    R.restore_rng(saved)


def random_draws():
    import random
    return (random.random(), float(np.random.rand()))


if __name__ == '__main__':
    worker(sys.argv[1], sys.argv[2:])
