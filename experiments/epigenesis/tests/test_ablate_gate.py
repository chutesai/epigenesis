"""Sleeping-unit contracts using the imported FSA helpers and tiny CPU experts."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate as a


class EDALayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(8, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            for lin in (self.up_proj, self.down_proj):
                lin.weight.zero_()
                lin.weight[:, [0, 2]] = 1.

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)).square())


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = torch.nn.Module()
        self.moe.experts = torch.nn.ModuleList([EDALayer(), None])

    def forward(self, x):
        return self.moe.experts[0](x)


def harness():
    # Independent enable flags, as with epi_gpu's e2e_fsa_v7 module.
    import e2e_fsa
    source = SimpleNamespace(enabled=True)
    def set_enabled(on):
        source.enabled = on
    return SimpleNamespace(FSAParam=source, set_enabled=set_enabled, moe_layers=e2e_fsa.moe_layers)


def args(**kw):
    base = dict(arm='ladder_tern', layers=[0], projs=['up_proj', 'down_proj'],
                write_set='inpair', patch_beta=1., gate_units_frac=.25,
                gate_select='sleep', gate_lambda=0., gate_margin=0., seed=7)
    return SimpleNamespace(**dict(base, **kw))


@pytest.mark.parametrize('proj', ['up_proj', 'down_proj'])
def test_restrict_rows_and_columns(proj):
    mask = torch.tensor([[True, False, True], [False, True, True], [True, True, False]])
    selected = torch.tensor([False, True, False])
    expected = torch.zeros_like(mask)
    if proj == 'up_proj':
        expected[1] = mask[1]
    else:
        expected[:, 1] = mask[:, 1]
    assert torch.equal(a.restrict_gate_mask(mask, proj, selected), expected)


@pytest.mark.parametrize('write_set', ['inpair', 'forced'])
@pytest.mark.parametrize('mode', ['sleep', 'specific', 'random', 'active'])
def test_selection_passes_through_and_intersects_write_set(monkeypatch, capsys, write_set, mode):
    _, select, hinge = a.gate_reference()
    fsa, model = harness(), TinyModel()
    cfg = args(write_set=write_set, gate_select=mode)
    if write_set == 'forced':
        bits = np.packbits(np.tile([1, 1, 0, 0], (8, 1)).astype(np.uint8).reshape(-1))
        entries = {f'moe.experts.0.{p}': dict(shape=[8, 8], bits=bits) for p in cfg.projs}
        monkeypatch.setattr(a.epi_mask, 'load', lambda: dict(entries=entries, sha256='test'))
    a.attach_ladder(model, fsa, cfg)
    originals = {p: getattr(model.moe.experts[0], p).parametrizations.weight[0].mask.clone() for p in cfg.projs}
    assert all(mask.any() for mask in originals.values())
    key = 'moe.experts.0'
    general = {key: dict(rate=torch.arange(8).float() / 8, mean_sq=torch.arange(8).float())}
    facts = {key: dict(rate=torch.arange(8).flip(0).float() / 8, mean_sq=torch.ones(8))}
    anchors = [[i] for i in range(64)]
    acquisition = [dict(variants=[dict(row=[1, 2, 3]), dict(row=[4, 5])])]
    calls = []
    def stats(model_, layers, rows):
        assert not fsa.FSAParam.enabled
        assert model_ is model and layers == [0]
        calls.append(rows)
        return general if len(calls) == 1 else facts
    selection_calls = []
    def spy_select(st, frac, passed_mode, seed, fs):
        selection_calls.append((st, frac, passed_mode, seed, fs))
        return select(st, frac, passed_mode, seed, fs)
    monkeypatch.setattr(a, 'gate_reference', lambda: (stats, spy_select, hinge))
    gate, info = a.configure_gate(model, fsa, cfg, anchors, acquisition)
    assert gate is None and fsa.FSAParam.enabled
    assert calls == ([anchors[:32], [[1, 2, 3], [4, 5]]] if mode == 'specific' else [anchors[:32]])
    fs = facts if mode == 'specific' else None
    assert selection_calls == [(general, .25, mode, 7, fs)]
    selected = select(general, .25, mode, 7, fs)[key]
    for proj in cfg.projs:
        m = getattr(model.moe.experts[0], proj).parametrizations.weight[0]
        assert torch.equal(m.mask, a.restrict_gate_mask(originals[proj], proj, selected))
    rates = general[key]['rate'][selected]
    assert info['units'] == 2 and info['select'] == mode
    assert info['sel_rate_mean'] == pytest.approx(float(rates.mean()))
    assert info['all_rate_mean'] == pytest.approx(3.5 / 8)
    assert info['never_active_frac'] == pytest.approx(float((rates == 0).float().mean()))
    assert 'GATE ' in capsys.readouterr().out


@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('margin,pre', [(0., -.1), (.5, -.6), (.5, .2)])
def test_reference_hinge_and_checkpoint_replay(checkpointed, margin, pre):
    _, _, GateHinge = a.gate_reference()
    model = TinyModel()
    with torch.no_grad():
        model.moe.experts[0].up_proj.weight[:, :4] = 1.
    if checkpointed:
        assert a.c.checkpoint_recurrences(model) == 1
    gate = GateHinge(model, [0], {'moe.experts.0': torch.tensor([True] + [False] * 7)}, margin)
    x = torch.full((2, 3, 8), pre / 4, requires_grad=True)
    raw, weighted = a.gate_backward(lambda _: model(x), [dict(row=[1, 2, 3])] * 2, 'cpu', gate, 3.)
    expected = max(pre + margin, 0.) ** 2
    assert raw == pytest.approx(expected, abs=1e-7)
    assert weighted == pytest.approx(3 * expected, abs=1e-7)
    expected_grad = torch.zeros_like(x)
    expected_grad[:, :, :4] = 3 * 2 * max(pre + margin, 0.) / 6
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)
    assert not gate.on and not gate.replay and gate.terms == []
    for hook in gate.h:
        hook.remove()


def test_gate_default_off(monkeypatch):
    def fail():
        raise AssertionError('Default-off gate imported the reference')
    monkeypatch.setattr(a, 'gate_reference', fail)
    assert a.configure_gate(None, None, args(gate_units_frac=0), [], []) == (None, None)
    assert a.gate_backward(None, [], 'cpu', None, 10.) == (0., 0.)
    parsed = a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'ladder_tern', '--data', 'd2'])
    assert (parsed.gate_units_frac, parsed.gate_select, parsed.gate_lambda, parsed.gate_margin) == (0., 'sleep', 0., 0.)


def test_real_stats_disable_epi_gpu_patch_state():
    a.gate_reference()
    fsa, model = harness(), TinyModel()
    cfg = args(gate_select='specific')
    a.attach_ladder(model, fsa, cfg)
    with torch.no_grad():
        for proj in cfg.projs:
            getattr(model.moe.experts[0], proj).parametrizations.weight[0].M.fill_(10.)
    # With the patch enabled, up-projection outputs are negative. The base
    # outputs are +2, so activation rates distinguish the two enable states.
    assert (model.moe.experts[0].up_proj(torch.tensor([[1., -2.] * 4])) < 0).all()
    original_forward = model.forward
    def student(tokens):
        x = torch.ones(*tokens.shape, 8)
        x[..., 1::2] = -2.
        return original_forward(x)
    model.forward = student
    gate, info = a.configure_gate(model, fsa, cfg, [[1, 2]] * 64, [dict(row=[3, 4, 5])])
    assert gate is None and fsa.FSAParam.enabled
    assert info['sel_rate_mean'] == info['all_rate_mean'] == info['sel_fact_rate_mean'] == 1.
    assert info['never_active_frac'] == 0.


def test_hinge_ignores_unselected_active_units():
    _, _, GateHinge = a.gate_reference()
    model = TinyModel()
    up = model.moe.experts[0].up_proj
    with torch.no_grad():
        up.weight[:, :4] = 2.5
        up.weight[0, :4] = -.15
    gate = GateHinge(model, [0], {'moe.experts.0': torch.tensor([True] + [False] * 7)}, .5)
    x = torch.ones(1, 8, requires_grad=True)
    gate.start()
    pre = up(x)
    hinge = gate.stop()
    hinge.backward()
    gate.end()
    assert (pre[:, 1:] > 0).all()
    assert float(hinge.detach()) == 0. and not x.grad.any()
    for hook in gate.h:
        hook.remove()


@pytest.mark.parametrize('extra', [['--gate-units-frac', '-.1'], ['--gate-units-frac', '1.1'],
                                  ['--gate-units-frac', 'nan'], ['--gate-lambda', '-1'],
                                  ['--gate-lambda', 'inf'], ['--gate-margin', '-1'],
                                  ['--gate-margin', 'nan'], ['--arm', 'ladder_cont', '--gate-units-frac', '.1']])
def test_gate_cli_validation(extra):
    with pytest.raises(SystemExit):
        a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'ladder_tern', '--data', 'd2'] + extra)


def test_import_has_no_epi_gpu_side_effects():
    code = '''
import sys, types, torch
import epi_gpu
sentinel = types.ModuleType('e2e_fsa_v7')
sentinel.FSAParam = types.SimpleNamespace(enabled=False)
sys.modules['e2e_fsa_v7'] = sentinel
import ablate
state = torch.get_rng_state().clone()
dtype = torch.get_default_dtype()
helpers = ablate.gate_reference()
assert all(h.__module__ == 'e2e_fsa' for h in helpers)
assert epi_gpu.fsa is sentinel and not sentinel.FSAParam.enabled
assert torch.equal(state, torch.get_rng_state())
assert dtype == torch.get_default_dtype()
assert 'lambda_model' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], cwd=Path(a.__file__).parent, check=True, capture_output=True)
