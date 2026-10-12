"""Canonical shipped write-set contracts, using the reference builder as oracle."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate
import consolidate as c
import epi_mask
import evaluate as e
import revoke_scope as r
from test_consolidate import FSAParam, LoRAParam

# repo layout (experiments/end_to_end) or the box copy of the FSA builder (epi_tbop/fsa_ref)
_here = Path(__file__).resolve().parents[1]
_builder = next(p for p in (_here.parent / 'end_to_end/build_forced_mask.py', _here / 'fsa_ref/build_forced_mask.py')
                if p.exists())
sys.path.insert(0, str(_builder.parent))
spec = importlib.util.spec_from_file_location('reference_builder', _builder)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def base():
    # Two, one, and no nonzero pairs: the shipped file resolves all three.
    return torch.tensor([[1., 0, 1, 0, 0, 0, 0, 0],
                         [0., 0, 0, 0, 2, 0, 0, 0],
                         [0., 0, 0, 0, 0, 0, 0, 0]])


@pytest.fixture
def shipped(tmp_path, monkeypatch):
    entries = {f'layers.0.experts.0.{proj}': dict(shape=list(base().shape), bits=builder.selected_bits(base()))
               for proj in ('up_proj', 'down_proj')}
    stored = dict(format='pair8-forced-mask-v1', export='synthetic', rule='reference builder',
                  sha256=builder.digest(entries), entries=entries)
    path = tmp_path / 'mask.pt'
    torch.save(stored, path)
    monkeypatch.setenv('EPI_FORCED_MASK', str(path))
    return path, stored


def model_and_harness():
    expert = torch.nn.Module()
    for proj in ('up_proj', 'down_proj'):
        lin = torch.nn.Linear(8, 3, bias=False)
        lin.weight.data.copy_(base())
        lin.weight.requires_grad_(False)
        setattr(expert, proj, lin)
    moe = torch.nn.Module()
    moe.experts = torch.nn.ModuleList([expert, None])
    model = torch.nn.ModuleList([moe])
    fsa = SimpleNamespace(FSAParam=FSAParam, LoRAParam=LoRAParam,
                          moe_layers=lambda model: [('layers.0', model[0])],
                          set_enabled=builder.E.set_enabled)
    return model, fsa


def test_reference_digest_and_expansion(shipped):
    path, stored = shipped
    assert epi_mask.load(path)['sha256'] == builder.digest(stored['entries'])
    for entry in epi_mask.load()['entries'].values():
        mask = epi_mask.forced_mask(entry)
        assert mask.dtype == torch.bool and mask.shape == base().shape
        assert torch.equal(mask, builder.E.pair8_forced_mask(base()))
        assert int(mask.sum()) == base().numel() // 2


@pytest.mark.parametrize('corruption', ['bits', 'shape', 'key', 'digest', 'format'])
def test_reject_corrupt_file(shipped, corruption):
    path, stored = shipped
    key = next(iter(stored['entries']))
    if corruption == 'bits':
        stored['entries'][key]['bits'][0] ^= 1
    elif corruption == 'shape':
        stored['entries'][key]['shape'][0] += 1
    elif corruption == 'key':
        stored['entries'][key + 'x'] = stored['entries'].pop(key)
    elif corruption == 'digest':
        stored['sha256'] = '0' * 64
    else:
        stored['format'] = 'other'
    torch.save(stored, path)
    with pytest.raises(ValueError):
        epi_mask.load()


@pytest.mark.parametrize('rank,fmt,budgeted', [(None, 'ternary', True), (None, 'fp4', False), (2, 'ternary', True)])
def test_attach_values_and_reserve_accounting(shipped, monkeypatch, rank, fmt, budgeted):
    model, fsa = model_and_harness()
    params = epi_mask.attach(model, fsa, layers=(0,), fmt=fmt, budgeted=budgeted, rank=rank)
    assert len(params) == (4 if rank else 2)
    monkeypatch.setattr(e, 'gpu', lambda: SimpleNamespace(fsa=fsa))
    for proj in ('up_proj', 'down_proj'):
        lin = getattr(model[0].experts[0], proj)
        m = lin.parametrizations.weight[0]
        assert torch.equal(m.mask, builder.E.pair8_forced_mask(base()))
        with torch.no_grad():
            if rank:
                m.A.fill_(1)
                m.B.fill_(1)
            else:
                assert m.logb is None
                m.M.fill_(.75 if fmt == 'fp4' else 1)
                m.support.copy_(m.mask)
        values = m.delta_w()
        expected = m.mask * m.alpha * (.75 if fmt == 'fp4' else 1)
        assert torch.equal(values, expected)
        assert torch.equal(lin.weight[~m.mask], base()[~m.mask])
    stats = e.slot_stats(model)
    # Fold storage counts committed codes, including codes in a zero-alpha row.
    nnz = 24 if rank else 16
    assert stats['nnz'] == nnz
    expected_bits = 24 * 4 if fmt == 'fp4' else nnz * (math.log2(24) + math.log2(3))
    assert stats['bytes'] == pytest.approx(expected_bits / 8)
    assert epi_mask.metadata(model) == dict(write_set='forced', mask_sha256=shipped[1]['sha256'])


def test_attach_refuses_nonzero_base(shipped):
    model, fsa = model_and_harness()
    model[0].experts[0].up_proj.weight.data[0, 4] = 1
    with pytest.raises(AssertionError, match='zero in the base'):
        epi_mask.attach(model, fsa, layers=(0,))


def test_forced_revoke_is_exact_and_reports_pre_revoke_illegality(shipped, monkeypatch, tmp_path):
    model, fsa = model_and_harness()
    epi_mask.attach(model, fsa, layers=(0,))
    fsa.set_enabled = lambda on: setattr(FSAParam, 'enabled', on)
    fake = SimpleNamespace(fsa=fsa, BOS=0, enc=lambda *args: [1],
                           logits_of=lambda model, x: model[0].experts[0].up_proj.weight.sum().expand(1, 2, 3).clone())
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    monkeypatch.setattr(e, 'concept_acc', lambda *args: {'acc': .25})
    monkeypatch.setattr(e, 'recall', lambda *args: {'em': .1})
    corpus = dict(revoke_prompts=['p'], probes={'test': {'concept': [], 'lure': []}})
    path = tmp_path / 'logits.pt'
    torch.save(e.revoke_logits(model, None, 'cpu', corpus), path)
    for m in model.modules():
        if isinstance(m, FSAParam):
            m.M.data.fill_(1)
            m.support.copy_(m.mask)
    result = e.revoke_test(model, None, 'cpu', corpus,
                          {'test': {'concept': {'acc': .25}, 'lure': {'em': .1}}}, path, replay_dir=tmp_path)
    assert result['pair8_legal_before'] is False
    assert result['pass'] and result['weights_equal'] and result['pair8_legal']
    assert json.loads((tmp_path / 'ledger.json').read_text()) == epi_mask.metadata(model)


def test_scope_revoke_without_scales():
    def state(values):
        return dict(committed=torch.tensor([values]), support=torch.tensor([values]).bool(),
                    frozen=torch.zeros(1, 4, dtype=torch.bool), logb=None)
    a, b, base_state = {'m': state([1, -1, 0, 0])}, {'m': state([1, 1, 1, 0])}, {'m': state([0, 0, 0, 0])}
    revoked = r.revoke_state(a, b, base_state, {'m': 0})
    assert revoked['m']['committed'].tolist() == [[0, 1, 1, 0]]
    assert revoked['m']['logb'] is None


def test_write_set_cli_defaults_and_forced():
    argv = ['--run', 'r', '--arm', 'tmid', '--session', '1']
    assert c.parse_args(argv).write_set == 'inpair'
    assert c.parse_args(argv + ['--write-set', 'forced']).write_set == 'forced'
    argv = ['--run', 'r', '--out', 'o', '--arm', 'lorafold', '--data', 'd2']
    assert ablate.parse_args(argv).write_set == 'inpair'
    assert ablate.parse_args(argv + ['--write-set', 'forced']).write_set == 'forced'


@pytest.mark.parametrize('arm', ['ladder_cont', 'ladder_box', 'ladder_tern'])
@pytest.mark.parametrize('write_set', ['inpair', 'forced'])
def test_ladder_attach_selected_projection_and_all_slots(shipped, arm, write_set):
    model, fsa = model_and_harness()
    args = ablate.parse_args(['--run', 'r', '--out', 'o', '--arm', arm, '--data', 'd2',
                             '--layers', '0', '--projs', 'down', '--write-set', write_set])
    params = ablate.attach_ladder(model, fsa, args)
    expert = model[0].experts[0]
    assert not hasattr(expert.up_proj, 'parametrizations')
    assert len(params) == 1
    m = expert.down_proj.parametrizations.weight[0]
    expected_mask = (epi_mask.forced_mask(next(iter(shipped[1]['entries'].values())))
                     if write_set == 'forced' else ablate.free_slot_mask(base()))
    assert torch.equal(m.mask, expected_mask)
    assert list(m.parameters()) == [m.M]
    assert not hasattr(m, 'support') and not hasattr(m, 'logb')
    # All positive-alpha slots receive a gradient, even with zero ternary codes.
    expert.down_proj.weight.sum().backward()
    assert torch.equal(m.M.grad, expected_mask * m.alpha)
    opt = torch.optim.Adam(params, lr=ablate.effective_lr(args), weight_decay=0.)
    opt.step()
    assert torch.equal(expert.down_proj.weight[~expected_mask], base()[~expected_mask])
    if write_set == 'forced':
        assert epi_mask.metadata(model)['mask_sha256'] == shipped[1]['sha256']
