"""Default-off damage recipe, using the reference hooks and netcost machinery."""
from types import SimpleNamespace

import pytest
import torch

from test_ablate_gate import TinyModel, args
import ablate as a


def reference():
    a.gate_reference()
    import e2e_fsa
    return e2e_fsa


def patched(budget=0, beta=1.):
    fsa = reference()
    model = TinyModel()
    cfg = args(budget=budget, patch_beta=beta)
    params = a.attach_ladder(model, fsa, cfg)
    return model, fsa, params


def test_damage_defaults_and_tmid_budget():
    base = ['--run', 'r', '--out', 'o', '--data', 'd2']
    cfg = a.parse_args(base + ['--arm', 'ladder_tern'])
    assert (cfg.local_lambda, cfg.budget, cfg.rehearsal) == (0., 0, 0.)
    assert a.parse_args(base + ['--arm', 'tmid']).budget == 2000000


@pytest.mark.parametrize('extra', [
    ['--budget', '-1'], ['--local-lambda', '-1'], ['--local-lambda', 'nan'],
    ['--local-lambda', 'inf'], ['--rehearsal', '-.1'], ['--rehearsal', '1.1'],
    ['--rehearsal', 'nan'], ['--rehearsal', 'inf'],
    ['--arm', 'ladder_cont', '--budget', '1'],
    ['--arm', 'ladder_two', '--local-lambda', '1'],
    ['--arm', 'lorafold', '--rehearsal', '.2'],
])
def test_damage_validation(extra):
    with pytest.raises(SystemExit):
        a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'ladder_tern', '--data', 'd2'] + extra)


@pytest.mark.parametrize('fraction', [.1, .2, .7, 1.])
def test_rehearsal_weights_and_gradients(fraction):
    torch.manual_seed(19)
    logits = torch.randn(12, 12, requires_grad=True)
    student = lambda x: logits[x]
    batch = [dict(row=[1, 2, 3, 4], start=2), dict(row=[5, 6, 7], start=1)]
    # No teacher records: positive rehearsal must never use the KL path.
    anchors = [dict(row=[8, 9, 10], start=1), dict(row=[9, 10, 11], start=1)]
    ce = sum(torch.nn.functional.cross_entropy(
        logits[torch.tensor(e['row'][e['start'] - 1:-1])],
        torch.tensor(e['row'][e['start']:]), reduction='sum') for e in batch) / 4
    general = sum(a.c.general_ce(student, torch.tensor([e['row']]), 12) for e in anchors) / 2
    expected = (1 - fraction) * ce + fraction * general
    expected.backward()
    gradient = logits.grad.clone()
    logits.grad = None
    losses = a.training_backward(student, batch, anchors, 'cpu', 12, 'ladder_tern', 'd2', fraction)
    assert losses['loss'] == pytest.approx(float(expected.detach()))
    assert losses['rehearsal_ce'] == pytest.approx(float(general.detach()))
    assert losses['gen'] == losses['tokens']['gen'] == 0
    assert torch.allclose(logits.grad, gradient, atol=1e-6)


@pytest.mark.parametrize('beta', [1., .25])
def test_budget_empty_then_refresh_bounded_and_eviction_resets_adam(monkeypatch, beta):
    model, fsa, params = patched(budget=5, beta=beta)
    mods = fsa.fsa_modules(model)
    assert len(mods) == 2  # Real harness type discovery, including energy hooks.
    opt = torch.optim.Adam(params, lr=.5)
    modules = {str(i): m for i, m in enumerate(mods)}
    frozen = {k: torch.zeros_like(m.mask) for k, m in modules.items()}
    allowed = {k: True for k in modules}
    monkeypatch.setattr(a.c, 'gpu', lambda: SimpleNamespace(fsa=fsa))
    original = model.forward
    model.forward = lambda tokens: original(torch.ones(*tokens.shape, 8))
    den = fsa.collect_col_energy(model, mods, torch.ones(2, 3, dtype=torch.long))
    assert den > 0 and all(m.col_energy.sum() > 0 for m in mods)
    lam = None
    for step in range(1, 33):
        opt.zero_grad(set_to_none=True)
        # Distinct evidence avoids cost-rel=1 ties at the calibration boundary.
        loss = sum((m.delta_w() * torch.arange(m.M.numel()).reshape_as(m.M)).sum() for m in mods)
        loss.backward()
        for m in mods:
            m.score.mul_(.9).add_(m.M.grad, alpha=.1)
        opt.step()
        if step < 8:
            assert all(not m.support.any() and not m.delta_w().any() for m in mods)
        if a.refresh_due(step, 24):
            if lam is None:
                lam = a.c.calibrate_lambda(mods, den, 5)
            assert lam is not None
            a.c.refresh(modules, frozen, allowed, 5, lam, den, opt)
        assert sum(int(m.support.sum()) for m in mods) <= 5
        assert sum(int(torch.count_nonzero(m.delta_w())) for m in mods) <= 5
        for m in mods:
            assert not m.codes()[0][~m.support].any()
    assert any(m.support.any() for m in mods)
    previous = [m.support.clone() for m in mods]
    a.c.refresh_support_cpu(mods, 5, [torch.zeros_like(m.M).cpu() for m in mods], opt)
    for m, evicted in zip(mods, previous):
        assert not m.support.any() and not m.M[evicted].any()
        assert not opt.state[m.M]['exp_avg'][evicted].any()
        assert not opt.state[m.M]['exp_avg_sq'][evicted].any()


def test_netcost_patch_scale_calibration_and_cost():
    model, fsa, _ = patched(budget=5)
    m = fsa.fsa_modules(model)[0]
    m.col_energy = torch.ones(m.M.shape[1])
    m.score.fill_(1.)
    m.M.data.zero_()
    lam = a.c.calibrate_lambda([m], 1., 5)
    score = a.c.netcost_score(m, 1., 1.)
    m.beta_factors = (.25,)
    assert a.c.calibrate_lambda([m], 1., 5) == pytest.approx(lam * 16)
    expected = (1 - (.25 * m.alpha).square() * m.col_energy) * m.mask
    assert torch.allclose(a.c.netcost_score(m, 1., 1.), expected)
    assert torch.allclose(score, (1 - m.alpha.square() * m.col_energy) * m.mask)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_local_and_hinge_share_forward_and_add_gradients(checkpointed):
    model, fsa, params = patched()
    for p in params:
        p.data.fill_(1.)
    if checkpointed:
        assert a.c.checkpoint_recurrences(model) == 1
    local = fsa.LocalOutputPenalty(model)
    gate = fsa.GateHinge(model, [0], {'moe.experts.0': torch.ones(8, dtype=torch.bool)}, .5)
    x = torch.ones(1, 3, 8, requires_grad=True)
    calls = []
    def student(_):
        calls.append(torch.is_grad_enabled())
        return model(x)
    anchors = [dict(row=[1, 2, 3])] * 2
    def run(lw, gw):
        for p in params:
            p.grad = None
        x.grad = None
        values = a.anchor_penalties_backward(student, anchors, 'cpu', local, lw, gate, gw)
        return values, [p.grad.clone() for p in params]
    try:
        _, lg = run(3., 0.)
        _, hg = run(0., 2.)
        calls.clear()
        values, combined = run(3., 2.)
        assert values[0] > 0 and values[1] > 0 and values[2] > 0
        assert any(g.any() for g in lg) and any(g.any() for g in hg)
        assert all(torch.allclose(g, l + h, rtol=1e-5, atol=1e-5) for g, l, h in zip(combined, lg, hg))
        assert calls == [False, False, True, True]  # Both hooks record on each shared forward.
        assert not local.on and not local.replay and local.num == local.den == []
        assert not gate.on and not gate.replay and gate.terms == []
    finally:
        for hook in local.h + gate.h:
            hook.remove()


def test_damage_runner_records_recipe_and_refresh_schedule(tmp_path, monkeypatch):
    from test_ablate import OffsetTokenizer, corpus
    data = corpus.__wrapped__()
    root, out = tmp_path / 'run', tmp_path / 'out'
    (root / 'genome').mkdir(parents=True)
    (root / 'corpus.json').write_text(a.json.dumps(data))
    (root / 'genome/metrics.json').write_text('{}')
    (root / 'genome/panel.pt').touch()
    (tmp_path / 'anchor-wikitext.json').write_text(a.json.dumps([1] * (150000 + 16 * 511)))
    model = TinyModel()
    ref = reference()
    fsa = SimpleNamespace(FSAParam=ref.FSAParam, moe_layers=ref.moe_layers,
                          fsa_modules=ref.fsa_modules, set_enabled=ref.set_enabled,
                          collect_col_energy=ref.collect_col_energy,
                          LocalOutputPenalty=ref.LocalOutputPenalty,
                          nll_windows=lambda *args: [0.])
    original = model.forward
    model.forward = lambda tokens: original(torch.ones(*tokens.shape, 8))
    gpu = SimpleNamespace(fsa=fsa, build=lambda: (model, 'cpu'), load_tokenizer=OffsetTokenizer,
                          BOS=0, VOCAB=256, DATA=tmp_path, logits_of=lambda m, x: m(x))
    monkeypatch.setattr(a.c, 'gpu', lambda: gpu)
    monkeypatch.setattr(a.epi_mask, 'metadata', lambda m: m._epi_write_set.copy())
    for method in ('empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(a.evaluate, 'concept_acc', lambda *args: dict(acc=.5, n=2))
    def backward(*args, rehearsal):
        assert rehearsal == .2
        sum((m.delta_w() * torch.arange(m.M.numel()).reshape_as(m.M)).sum()
            for m in ref.fsa_modules(model)).backward()
        return dict(loss=0.)
    monkeypatch.setattr(a, 'training_backward', backward)
    cfg = a.parse_args(['--run', str(root), '--out', str(out), '--arm', 'ladder_tern',
                        '--data', 'd2', '--layers', '0', '--budget', '5', '--local-lambda', '3',
                        '--rehearsal', '.2', '--patch-beta', '.25', '--steps', '24',
                        '--refresh-until', '16', '--eval-every', '8', '--lr-mult', '50'])
    result = a.run(cfg)
    saved = a.json.loads((out / 'result.json').read_text())
    assert saved['config'] == result['config']
    config = saved['config']
    assert config['budget'] == 5 and config['local_lambda'] == 3 and config['rehearsal_frac'] == .2
    assert config['acquisition_weight'] == .8 and config['general_weight'] == .2
    assert config['cost_rel'] == 1 and config['support_score'] == 'netcost'
    assert saved['netcost_lambda'] == config['netcost_lambda'] > 0
    rows = [a.json.loads(line) for line in (out / 'log.jsonl').read_text().splitlines()]
    train = [row for row in rows if row['phase'] == 'train']
    assert all(row['nnz'] == row['support_size'] == 0 for row in train[:7])
    assert all(row['nnz'] <= row['support_size'] <= 5 for row in train)
    assert any(row['births'] for row in train)
    assert all(row['births'] == row['evictions'] == 0 for row in train if row['step'] not in (8, 16))
