"""CPU contracts for the v2.2 fold-aware arm."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate
import consolidate as c
import evaluate as e
import epi_fold
from epi_common import free_slot_mask


def base():
    return torch.tensor([[1., 0, 1, 0, 0, 0, 0, 0], [2., 0, 2, 0, 0, 0, 0, 0]])


def fold():
    W = base()
    m = epi_fold.FoldLoRAParam(W, 2, free_slot_mask(W))
    with torch.no_grad():
        m.A.copy_(torch.tensor([[0., .5, 0, -.5001, 10, 0, 0, 0],
                               [0., 2., 0, -2., 0, 0, 0, 0]]))
        m.B.copy_(torch.eye(2))
    return m


def test_ste_exact_codes_and_factor_gradients():
    m, W = fold(), base()
    assert torch.equal(m.codes(), torch.tensor([[0., 0, 0, -1, 0, 0, 0, 0],
                                               [0., 1, 0, -1, 0, 0, 0, 0]]))
    assert torch.equal(m.delta_w(), m.codes() * m.alpha)
    assert torch.equal(m(W), W + m.codes() * m.alpha)
    loss = (m.delta_w() * torch.arange(16).reshape(2, 8)).sum()
    loss.backward()
    assert m.A.grad.abs().sum() > 0 and m.B.grad.abs().sum() > 0
    assert not m.A.grad[:, ~m.mask.any(0)].any()


def test_state_keeps_factors_and_committed_codes(tmp_path):
    source = fold()
    path = tmp_path / 'state.pt'
    c.save_state(path, {'m': source}, {}, {'session': 1}, [])
    saved = torch.load(path, weights_only=True)['modules']['m']
    assert set(saved) == {'A', 'B', 'committed'}
    assert torch.equal(saved['committed'], source.codes())
    target = fold()
    with torch.no_grad():
        target.A.zero_()
        target.B.zero_()
    assert c.load_state(path, {'m': target}, {}) == ({'session': 1}, [])
    assert torch.equal(target.A, source.A) and torch.equal(target.B, source.B)
    assert torch.equal(target(base()), source(base()))
    saved['committed'].zero_()
    with pytest.raises(AssertionError, match='committed patch'):
        c.restore({'m': target}, {'m': saved}, {})


def test_attach_mask_layers_switch_and_ternary_accounting(monkeypatch):
    class LoRA(torch.nn.Module):
        enabled = True

    def expert():
        ex = torch.nn.Module()
        for name in ('up_proj', 'down_proj'):
            lin = torch.nn.Linear(8, 2, bias=False)
            lin.weight.data.copy_(base())
            lin.weight.requires_grad_(False)
            setattr(ex, name, lin)
        return ex

    layers = []
    for _ in range(16):
        moe = torch.nn.Module()
        moe.experts = torch.nn.ModuleList([expert(), None])
        layers.append(moe)
    model = torch.nn.ModuleList(layers)
    fsa = SimpleNamespace(moe_layers=lambda model: list(enumerate(model)),
                          LoRAParam=LoRA, FSAParam=type('FSA', (torch.nn.Module,), {}))
    params = epi_fold.attach(model, fsa)
    assert len(params) == 8 and all(p.requires_grad for p in params)
    assert not hasattr(model[13].experts[0].up_proj, 'parametrizations')
    patches = [m for m in model.modules() if isinstance(m, epi_fold.FoldLoRAParam)]
    assert len(patches) == 4 and all(m.A.shape == (16, 8) for m in patches)
    for m in patches:
        assert torch.equal(m.mask, free_slot_mask(base()))
        with torch.no_grad():
            m.A.fill_(1.)
            m.B.fill_(1.)
    monkeypatch.setattr(e, 'gpu', lambda: SimpleNamespace(fsa=fsa))
    stats = e.slot_stats(model)
    assert stats['nnz'] == 16
    assert stats['bytes'] == pytest.approx(16 * (math.log2(16) + math.log2(3)) / 8)
    LoRA.enabled = False
    assert torch.equal(model[14].experts[0].up_proj.weight, base())
    LoRA.enabled = True
    assert not torch.equal(model[14].experts[0].up_proj.weight, base())


def test_revoke_restores_bitwise_weights_and_clears_training(monkeypatch, tmp_path):
    lin = torch.nn.Linear(8, 2, bias=False)
    lin.weight.data.copy_(base())
    m = fold()
    torch.nn.utils.parametrize.register_parametrization(lin, 'weight', m)
    original = lin.parametrizations.weight.original.detach().clone()
    fsa = SimpleNamespace(FSAParam=type('FSA', (torch.nn.Module,), {}),
                          LoRAParam=type('LoRA', (torch.nn.Module,), {}), set_enabled=lambda on: None)
    fake = SimpleNamespace(fsa=fsa, BOS=0, enc=lambda *args: [1],
                           logits_of=lambda model, x: model.weight.sum().expand(1, x.shape[1], 3).clone())
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    monkeypatch.setattr(e, 'concept_acc', lambda *args: {'acc': .25})
    monkeypatch.setattr(e, 'recall', lambda *args: {'em': .1})
    corpus = dict(revoke_prompts=['a'], probes={'test': {'concept': [], 'lure': []}})
    path = tmp_path / 'logits.pt'
    m.enabled = False
    torch.save(e.revoke_logits(lin, None, 'cpu', corpus), path)
    m.enabled = True
    opt = torch.optim.Adam(m.parameters())
    m.delta_w().square().sum().backward()
    opt.step()
    assert opt.state and not torch.equal(lin.weight, original)
    result = e.revoke_test(lin, None, 'cpu', corpus,
                          {'test': {'concept': {'acc': .25}, 'lure': {'em': .1}}}, path, optimizer=opt)
    assert result['pass'] and result['logits_exact']
    assert torch.equal(lin.weight.view(torch.int32), original.view(torch.int32))
    assert not m.A.any() and not m.B.any() and not m.codes().any()
    assert m.A.grad is None and m.B.grad is None and not opt.state


@pytest.mark.parametrize('until,expected', [(None, [8, 16, 24, 32]), (16, [8, 16]),
                                           (15, [8]), (0, [])])
def test_refresh_cutoff_calls(until, expected):
    refresh = Mock()
    for step in range(1, 33):
        if ablate.refresh_due(step, until):
            refresh(step)
    assert [call.args[0] for call in refresh.call_args_list] == expected


def test_cli_additive_defaults():
    argv = ['--run', 'r', '--out', 'o', '--arm', 'tmid', '--data', 'd2']
    assert ablate.parse_args(argv).refresh_until is None
    assert ablate.parse_args(argv + ['--refresh-until', '300']).refresh_until == 300
    args = ablate.parse_args(argv + ['--arm', 'lorafold', '--lr-mult', '5'])
    assert ablate.effective_lr(args) == .001
    with pytest.raises(SystemExit):
        ablate.parse_args(argv + ['--refresh-until', '-1'])
