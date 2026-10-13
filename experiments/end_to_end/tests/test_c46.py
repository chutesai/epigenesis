"""C46 checkpoint identity, strict state, backend and heartbeat regressions."""
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, tiny
import c44_runtime as R
import c41_birth_match as births
import c42_selector as selector
import eda_backend as eda


def args(root, *flags):
    return E.build_parser().parse_args(['--out', str(root/'x.json'), '--patch-dir', str(root),
                                       '--ckpt-every', '1', *flags])


@pytest.mark.parametrize('flag', ['--anchor', '--eval-set', '--extra-eval-sets', '--teacher-patch',
    '--sibling-patches', '--load-patch', '--resident-patch', '--resident-mask-only', '--mask-compare'])
def test_resume_refuses_changed_input_contents(tmp_path, flag):
    source = tmp_path/'input.json'
    source.write_text('original')
    m = R.Checkpoints(args(tmp_path, flag, str(source)))
    m.save(1)
    assert m.inputs[str(source)] == R.hashlib.sha256(b'original').hexdigest()
    source.write_text('changed')
    with pytest.raises(SystemExit, match='input contents changed'):
        R.Checkpoints(args(tmp_path, flag, str(source), '--resume', 'auto'))


def test_export_manifest_and_all_extra_panels_are_fingerprinted(tmp_path):
    export = tmp_path/'export'
    export.mkdir()
    manifest = export/'manifest.json'
    manifest.write_text('{}')
    panels = [tmp_path/'a.json', tmp_path/'b.json']
    for p in panels:
        p.write_text('[]')
    flags = ['--export', str(export), '--extra-eval-sets', ','.join(map(str, panels))]
    m = R.Checkpoints(args(tmp_path, *flags))
    m.save(1)
    assert all(str(p) in m.inputs for p in [manifest, *panels])
    manifest.write_text('{"changed": true}')
    with pytest.raises(SystemExit, match='input contents changed'):
        R.Checkpoints(args(tmp_path, *flags, '--resume', 'auto'))


def test_resume_accepts_cadence_and_path_spelling_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('input.json').write_text('[]')
    R.Checkpoints(args(tmp_path, '--anchor', './input.json')).save(1)
    m = R.Checkpoints(args(tmp_path, '--anchor', str(tmp_path/'input.json'), '--resume', 'auto',
                          '--ckpt-every', '8', '--ckpt-minutes', '10', '--patch-dir', './'))
    assert m.saved['step'] == 1
    assert not {'ckpt_every', 'ckpt_minutes', 'patch_dir'} & m.config.keys()


def test_running_memory_peak_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 7 * 2**30)
    m = R.Checkpoints(args(tmp_path))
    m.save(1)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 2 * 2**30)
    resumed = R.Checkpoints(args(tmp_path, '--resume', 'auto'))
    assert resumed.memory_peak() == 7
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 9 * 2**30)
    assert resumed.memory_peak() == 9


@pytest.mark.parametrize('key', ['history', '_history'])
def test_unsupported_state_raises_and_known_rebuilt_state_is_excluded(key):
    with pytest.raises(TypeError, match=f'SimpleNamespace.{key}'):
        R.object_state(SimpleNamespace(**{key: deque([torch.ones(1)])}))
    for name, keys in [('UnitBurst', ('banks', 'coords', 'names', 'hooks')),
                       ('MatchedActivationNoise', ('deltas', 'hooks'))]:
        obj = type(name, (), {})()
        for key in keys:
            setattr(obj, key, {object(): torch.ones(1)})
        obj.counter = 3
        assert R.object_state(obj) == {'counter': 3}


def test_birth_matching_rejects_backend_mix_and_defaults_legacy(tmp_path):
    def write(name, backend=None):
        state = dict(optim='bop', bop_tau=.1, bop_events=[dict(update=1, births=3)])
        if backend:
            state['eda_backend'] = backend
        path = tmp_path/name
        path.write_text(json.dumps(state))
        return path
    target, legacy = write('target', eda.DEFAULT), write('legacy')
    assert births.compare_births(target, [legacy], 1)['best']['relative_error'] == 0
    with pytest.raises(ValueError, match='mixed EDA backends'):
        births.compare_births(target, [write('fla', 'fla_kda_2t')], 1)


def test_mask_diagnostics_checks_backend_before_comparing(tmp_path):
    mask = tmp_path/'mask.pt'
    torch.save(dict(format='pair8-forced-mask-v1', eda_backend='fla_kda_2t'), mask)
    with pytest.raises(ValueError, match='EDA backend mismatch.*mask-compare'):
        E.mask_diagnostics(tiny(), mask)


def test_model_build_emits_before_and_after(monkeypatch):
    calls = []
    monkeypatch.setattr(R, 'heartbeat', lambda *a, **kw: calls.append((a, kw)))
    model = SimpleNamespace()
    eda.build_model(lambda **kw: (model, None, 'cpu'), SimpleNamespace())
    assert calls == [(('model_build', 0, 1), {'force': True}),
                     (('model_build', 1, 1), {'force': True})]


def test_selector_inner_fits_heartbeat(monkeypatch):
    calls = []
    monkeypatch.setattr(R, 'heartbeat', lambda *a: calls.append(a))
    monkeypatch.setattr(selector, 'heartbeat', lambda *a: calls.append(a))
    x = torch.tensor([[-2.], [-1.], [1.], [2.]])
    labels = torch.tensor([0, 0, 1, 1])
    selector.logistic_llr(x, labels.bool())
    selector.SiblingKeys.fit(x, labels)
    assert any(c[0] == 'selector_logistic_fit' for c in calls)
    assert any(c[0] == 'selector_sibling_fit' for c in calls)


def test_activation_noise_resume_rebuilds_ephemeral_deltas(tmp_path):
    from test_c44 import test_process_resume_is_bit_exact as verify_resume
    verify_resume(tmp_path, ['--sparse-state', '--activation-noise', '.2'], 3)
