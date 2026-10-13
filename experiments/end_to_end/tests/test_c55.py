"""C55 opt-in cross-backend parity evaluation using real CPU CLI flows."""
import json
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, Tiny, cpu_harness
from test_branch_eval import B, state
from test_c45 import fake_fla
import eda_backend as K


@pytest.mark.parametrize('saved,evaluated', [K.CHOICES, tuple(reversed(K.CHOICES))])
def test_loaded_patch_cross_eval(cpu_harness, fake_fla, capsys, saved, evaluated):
    run, _ = cpu_harness
    (fake_fla.parent / 'tokenizer.json').write_text('{}')
    _, patch = run(['--steps', '1', '--eda-backend', saved], 'source')
    capsys.readouterr()
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        run(['--steps', '0', '--load-patch', str(patch), '--eda-backend', evaluated], 'refused')
    capsys.readouterr()
    cache = fake_fla.parent / 'cache'
    result, _ = run(['--steps', '0', '--load-patch', str(patch), '--eda-backend', evaluated,
                     '--allow-backend-crosseval', '--base-cache', str(cache)], 'cross')
    crossing = dict(saved=saved, eval=evaluated)
    assert result['backend_crosseval'] == crossing
    assert result['eda_backend'] == evaluated
    assert torch.load(patch, weights_only=False)['meta']['eda_backend'] == saved
    manifest = json.loads((cache / (result['base_cache_key'] + '.json')).read_text())
    assert manifest['eda_backend'] == evaluated
    lines = capsys.readouterr().out.splitlines()
    assert f'CROSS-EVAL saved={saved} eval={evaluated}' in lines
    reports = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith(('RESULT ', 'FINAL '))]
    assert len(reports) >= 2
    assert all(report['backend_crosseval'] == crossing for report in reports)


@pytest.mark.parametrize('patch_flag', ['--load-patch', '--resident-patch', '--resident-mask-only', '--sibling-patches'])
@pytest.mark.parametrize('training', [['--steps', '1'], ['--steps', '0', '--repair-steps', '1'],
                                     ['--steps', '0', '--wake-steps', '1']])
def test_training_rejected_before_patch_io(patch_flag, training):
    with pytest.raises(ValueError, match='evaluation-only'):
        E.main(['--out', 'unused', '--allow-backend-crosseval', patch_flag, 'missing.pt', *training])


@pytest.mark.parametrize('patch_flag', ['--resident-patch', '--resident-mask-only', '--sibling-patches'])
def test_resident_cross_eval(cpu_harness, fake_fla, patch_flag):
    run, _ = cpu_harness
    _, patch = run(['--steps', '1'], 'source')
    result, _ = run(['--steps', '0', patch_flag, str(patch), '--eda-backend', 'fla_kda_2t',
                     '--allow-backend-crosseval'], 'cross')
    assert result['backend_crosseval'] == dict(saved=K.DEFAULT, eval='fla_kda_2t')


@pytest.mark.parametrize('saved,evaluated', [K.CHOICES, tuple(reversed(K.CHOICES))])
@pytest.mark.parametrize('learning', [False, True])
def test_branch_cli_cross_eval(cpu_harness, fake_fla, monkeypatch, capsys, saved, evaluated, learning):
    run, paths = cpu_harness
    (fake_fla.parent / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(Tiny, 'layers', property(lambda self: self.moes), raising=False)
    patches = []
    for person in range(3):
        _, patch = run(['--steps', '1', '--train-person', str(person), '--eda-backend', saved], f'person{person}')
        patches.append(str(patch))
    capsys.readouterr()
    out = fake_fla.parent / 'branch.json'
    flags = ['--patches', *patches, '--out', str(out), '--eda-backend', evaluated,
             '--fast-moe', 'off', '--gen-max-new', '1']
    flags += (['--learning-eval'] if learning else ['--arms', 'B,D', '--anchor', paths['anchor'],
        '--eval-set', paths['eval'], '--fineweb', paths['extra'], '--code', paths['extra'],
        '--qa-n', '0', '--g-n', '0', '--router-layer', '0,1',
        '--base-cache', str(fake_fla.parent / 'branch_cache')])
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        B.main(flags)
    capsys.readouterr()
    B.main(flags + ['--allow-backend-crosseval'])
    result = json.loads(out.read_text())
    assert result['eda_backend'] == evaluated
    crossing = dict(saved=saved, eval=evaluated)
    assert result['backend_crosseval'] == crossing
    if not learning:
        manifests = list((fake_fla.parent / 'branch_cache').glob('*.json'))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text())['eda_backend'] == evaluated
    lines = capsys.readouterr().out.splitlines()
    assert f'CROSS-EVAL saved={saved} eval={evaluated}' in lines
    reports = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith(('RESULT ', 'FINAL '))]
    assert len(reports) >= 2 and all(report['backend_crosseval'] == crossing for report in reports)


def test_flag_does_not_relax_cache_or_mixed_bank_checks(tmp_path):
    args = SimpleNamespace(eda_backend='fla_kda_2t', allow_backend_crosseval=True, steps=0)
    banks = [state(p, [p], [1]) for p in range(3)]
    assert len(B.validate_patches(banks, eda_backend=args.eda_backend, crosseval_args=args)) == 3
    banks[0]['meta']['eda_backend'] = 'fla_kda_2t'
    with pytest.raises(ValueError, match='mixed EDA backends'):
        B.validate_patches(banks, eda_backend=args.eda_backend, crosseval_args=args)
    E.cached_base(tmp_path, 'key', lambda: {}, 'cpu')
    with pytest.raises(ValueError, match='base-cache manifest'):
        E.cached_base(tmp_path, 'key', lambda: {}, 'cpu', args.eda_backend)


def test_same_backend_flag_is_unmarked(cpu_harness):
    run, _ = cpu_harness
    _, patch = run(['--steps', '1'], 'source')
    result, _ = run(['--steps', '0', '--load-patch', str(patch), '--allow-backend-crosseval'], 'same')
    assert 'backend_crosseval' not in result
