"""C41 statistical decisions, atomic cross-projection bursts and measured controls (CPU)."""
import hashlib
import json

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize as P

from test_e2e_fsa import E, cpu_harness  # noqa: F401
from test_c19 import args
import c41_birth_match as MATCH


def slot(sparse=False, budgeted=False):
    cls = E.SparseFSAParam if sparse else E.FSAParam
    m = cls(torch.ones(4, 6), torch.ones(4, 6, dtype=torch.bool), budgeted=budgeted)
    m.bop_ema = torch.zeros_like(m.M)
    return m


@pytest.mark.parametrize('sparse', [False, True])
def test_consistent_sub_tau_births_only_under_tstat_and_respects_support(sparse):
    standard, statistical = slot(sparse, True), slot(sparse, True)
    for m in (standard, statistical):
        m.support.zero_()
        m.support.flatten()[:3] = True
    for _ in range(20):
        for m in (standard, statistical):
            m.M.grad = torch.full_like(m.M, .5)
        E.bop_step([standard], .05, 1., True)
        E.bop_step([statistical], .05, 1., True, tstat=3.)
    assert standard.M.count_nonzero() == 0
    assert statistical.M.count_nonzero() == 3
    assert (statistical.M.flatten()[:3] == -1).all()
    assert not statistical.M[~statistical.support].any()


def test_tstat_floor_single_observation_spike_and_death_mirror():
    m = slot()
    m.M.grad = torch.full_like(m.M, .5)
    assert E.bop_step([m], .05, 1., False, tstat=3.) == (0, 0)
    assert E.bop_step([m], .05, 1., False, tstat=3.) == (24, 0)
    assert m.bop_v.count_nonzero() == m.bop_weight.count_nonzero() == 0
    m.M.grad.fill_(-.5)
    assert E.bop_step([m], .05, 1., False, tstat=3., death_mult=2.) == (0, 0)
    assert E.bop_step([m], .05, 1., False, tstat=3., death_mult=2.) == (0, 0)  # .5 < mirrored .6 floor
    assert E.bop_step([m], .05, 1., False, tstat=3.) == (0, 24)
    m.M.grad.fill_(.5)
    for _ in range(10):
        assert E.bop_step([m], .05, 1., False, tstat=3., tstat_floor=.6) == (0, 0)
    m = slot()
    for grad in (100., 0., 0., 0., 0.):
        m.M.grad = torch.full_like(m.M, grad)
        assert E.bop_step([m], .05, 1., False, tstat=3.) == (0, 0)


def test_actual_ema_weights_bias_correction_and_gamma_switch():
    m = slot()
    weights, grads = [], []
    for gamma, grad in ((.05, 1.), (.05, -1.), (.2, 2.), (.2, 0.)):
        m.M.grad = torch.full_like(m.M, grad)
        E.bop_step([m], gamma, 100., False, tstat=3.)
        weights = [w * (1 - gamma) for w in weights] + [gamma]
        grads.append(grad)
        w, g = torch.tensor(weights), torch.tensor(grads)
        mean = (w * g).sum() / w.sum()
        v = (w * g.square()).sum() / w.sum() - mean.square()
        ne = w.sum().square() / w.square().sum()
        mag, se, n_eff = E.bop_tstat_evidence(m)
        assert torch.allclose(mag, mean.abs().expand_as(mag))
        assert torch.allclose(n_eff, ne.expand_as(n_eff))
        assert torch.allclose(se, (v / ne).sqrt().expand_as(se))
    E.reset_bop_statistics(m, torch.ones_like(m.mask))
    assert not m.bop_weight.any()


def test_tstat_grace_repair_and_occupied_flip_legality():
    m = slot()
    E.init_churn(m)
    m.M.data.fill_(-1)
    m.M.grad = torch.full_like(m.M, -.5)
    churn = E.new_churn_totals()
    for _ in range(2):
        assert E.bop_step([m], .05, 1., False, tstat=3., grace=3, churn=churn) == (0, 0)
    assert churn['deaths_blocked_by_grace'] == 24
    E.bop_step([m], .05, 1., False, tstat=3., allow_birth=False)
    m.M.grad.fill_(.5)
    for _ in range(3):
        assert E.bop_step([m], .05, 1., False, tstat=3., allow_birth=False) == (0, 0)
    m = E.SparseFSAParam(torch.ones(2, 4), torch.zeros(2, 4, dtype=torch.bool), occupied=True)
    m.bop_ema = torch.zeros_like(m.M)
    for grad in (.8, .8):
        m.M.grad = torch.full_like(m.M, grad)
        E.bop_step([m], .05, 1., False, tstat=3.)
    assert not m.M.any()
    for _ in range(50):
        m.M.grad = torch.full_like(m.M, -.8)
        E.bop_step([m], .05, 1., False, tstat=3.)
    assert (m.M == 1).all()  # occupied flips only admit +1; floor also doubles


@pytest.mark.parametrize('sparse', [False, True])
def test_surviving_histogram_rebirth_eviction_and_unknown_import(sparse):
    m = slot(sparse, True)
    m.support.copy_(m.mask)
    E.init_birth_steps([m])
    m.M.grad = torch.ones_like(m.M)
    E.bop_step([m], 1., .5, True, update=2)
    assert E.surviving_birth_summary([m]) == dict(histogram={'2': 24}, known=24, unknown=0, median=2.)
    m.M.grad.flatten()[0] = -1
    E.bop_step([m], 1., .5, True, update=4)
    assert E.surviving_birth_summary([m])['known'] == 23
    m.M.grad.fill_(1)
    E.bop_step([m], 1., .5, True, update=6)
    summary = E.surviving_birth_summary([m])
    assert summary['histogram'] == {'2': 23, '6': 1}
    E.refresh_support([m], 1, scores=[torch.arange(1, 25).view_as(m.M).float()])
    assert E.surviving_birth_summary([m])['histogram'] == {'2': 1}
    assert (m.birth_step[~m.support] == -1).all()
    m.birth_step.fill_(-1)
    assert E.surviving_birth_summary([m]) == dict(histogram={}, known=0, unknown=1, median=None)


def burst_model(sparse=False, budgeted=False):
    model = nn.Module()
    model.up_proj, model.down_proj = nn.Linear(6, 4, bias=False), nn.Linear(4, 6, bias=False)
    mods = []
    for lin in (model.up_proj, model.down_proj):
        lin.weight.data.fill_(1)
        cls = E.SparseFSAParam if sparse else E.FSAParam
        m = cls(lin.weight, torch.ones_like(lin.weight, dtype=torch.bool), budgeted=budgeted)
        P.register_parametrization(lin, 'weight', m)
        m.bop_ema = torch.zeros_like(m.M)
        m.support.copy_(m.mask)
        mods.append(m)
    # Production names have an expert prefix; supply one for the standalone pair.
    wrapper = nn.Module()
    wrapper.expert = model
    return wrapper, mods


@pytest.mark.parametrize('sparse', [False, True])
def test_burst_births_both_sides_one_time_top_m_and_fact_attribution(sparse):
    model, mods = burst_model(sparse, True)
    burst = E.UnitBurst(model, 2, threshold=.2)
    for m in mods:
        ids = burst.coords[m].view_as(m.M)
        m.M.grad = torch.where(ids == 0, .9, .4)
        burst.fact = 7
        burst.observe(m, m.M.grad)
        burst.fact = None
    assert E.bop_step(mods, .05, 1., True, unit_burst=burst, update=1) == (4, 0)
    assert burst.last == [dict(bank='expert', count=1, up_slots=2, down_slots=2, units=[0])]
    for m in mods:
        assert m.M.count_nonzero() == 2
        assert (burst.coords[m][m.M.flatten() != 0] == 0).all()
        assert (m.M[m.M != 0] == -1).all()
    assert burst.summary([7, 9])['distinct_units_per_fact'] == {'7': 1, '9': 0}
    E.bop_step(mods, .05, 1., True, unit_burst=burst, update=2)
    assert 0 not in [j for e in burst.last for j in e['units']]
    burst.close()


@pytest.mark.parametrize('sparse', [False, True])
def test_random_control_replays_counts_even_when_evidence_changes(sparse, tmp_path):
    model, mods = burst_model(sparse, True)
    burst = E.UnitBurst(model, 2, threshold=.2)
    for m in mods:
        m.M.grad = torch.where(burst.coords[m].view_as(m.M) == 0, .9, .4)
    expected = E.bop_step(mods, .05, 1., True, unit_burst=burst, update=1)
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(dict(bop_events=burst.events)))
    model2, random_mods = burst_model(sparse, True)
    random = E.UnitBurst(model2, 2, threshold=.2, random=True, seed=2, replay=path)
    for m in random_mods:
        m.M.grad = torch.full_like(m.M, .01)  # no unit crosses the threshold here
    assert E.bop_step(random_mods, .05, 1., True, unit_burst=random, update=1) == expected == (4, 0)
    assert random.last[0]['count'] == burst.last[0]['count']
    assert random.last[0]['units'] != burst.last[0]['units']
    assert [int(m.M.count_nonzero()) for m in random_mods] == [2, 2]
    burst.close()
    random.close()


def test_burst_budget_atomicity_repair_energy_score_and_unmatchable_replay(tmp_path):
    model, mods = burst_model(budgeted=True)
    burst = E.UnitBurst(model, 2, threshold=.1)
    for m in mods:
        m.M.grad = torch.full_like(m.M, .5)
        m.col_energy = torch.ones(m.wshape[1])
    mods[1].support.zero_()
    assert E.bop_step(mods, .05, 1., True, unit_burst=burst, update=1) == (0, 0)
    mods[1].support.copy_(mods[1].mask)
    assert E.bop_step(mods, .05, 1., True, unit_burst=burst, update=2, allow_birth=False) == (0, 0)
    burst.den, burst.cost_lambda = 1., 100.
    assert E.bop_step(mods, .05, 1., True, unit_burst=burst, update=3) == (0, 0)
    # Same magnitudes, different activation energy -> only low-cost unit crosses.
    burst.cost_lambda = 1.
    mods[0].alpha[0] = 0
    mods[1].col_energy[0] = 0
    assert E.bop_step(mods, .05, 1., True, unit_burst=burst, update=4) == (4, 0)
    assert burst.last[0]['units'] == [0]
    path = tmp_path / 'schedule.json'
    path.write_text(json.dumps(dict(bop_events=[dict(update=5, unit_bursts=[dict(
        bank='expert', count=4, up_slots=2, down_slots=2)])])))
    random = E.UnitBurst(model, 2, random=True, replay=path)
    mods[1].support.zero_()
    with pytest.raises(ValueError, match='cannot match'):
        E.bop_step(mods, .05, 1., True, unit_burst=random, update=5)
    burst.close()
    random.close()


def test_burst_top_slots_signs_default_random_counts_and_ordinary_overlap():
    model, mods = burst_model()
    burst = E.UnitBurst(model, 2, threshold=.1)
    for m in mods:
        m.M.grad = torch.zeros_like(m.M)
        ix = (burst.coords[m] == 0).nonzero().flatten()[:3]
        m.M.grad.flatten()[ix] = torch.tensor([.9, -.8, .4])
    E.bop_step(mods, .5, 1., False, unit_burst=burst, update=1)
    assert burst.additional_births == 4
    for m in mods:
        ix = (burst.coords[m] == 0).nonzero().flatten()[:3]
        assert m.M.flatten()[ix].tolist() == [-1., 1., 0.]
    random_model, random_mods = burst_model()
    random = E.UnitBurst(random_model, 2, threshold=.1, random=True)
    for m in random_mods:
        m.M.grad = torch.full_like(m.M, 5.)
    assert E.bop_step(random_mods, .5, 1., False, unit_burst=random, update=1) == (48, 0)
    assert sum(e['count'] for e in random.last) == 4
    assert random.additional_births == 0  # all selected burst slots were ordinary births already
    burst.close()
    random.close()


@pytest.mark.parametrize('sparse,keep,expected', [
    (False, False, '941a85a68b5c3d92245ca1c75433da54022b6f027faa57ce895091d4f7e7123d'),
    (False, True, 'cb8035a6f2482ef7e7d7d785db7168c72829ecf457d6aa383d90288af4840d41'),
    (True, False, '39c286a38e2d942f5c03b7ab515300935a16137e7bc2e337f0dff1f206ac37d9'),
    (True, True, '1bece9a339a1100e3aa312c1fd1acadbe4dd6fa56c10ea7568ffd8ae7a40e68d'),
])
def test_legacy_bop_state_is_byte_identical(sparse, keep, expected):
    # Frozen digests from the pre-C41 optimizer: births/deaths, all live/EMA/churn state,
    # two gamma rates, reset/keep-EMA and maturity. A behavioral snapshot, not a new oracle.
    gen = torch.Generator().manual_seed(41)
    mask = torch.rand(4, 6, generator=gen) > .25
    cls = E.SparseFSAParam if sparse else E.FSAParam
    m = cls(torch.ones(4, 6), mask, budgeted=True)
    m.support.copy_(m.mask)
    m.bop_ema = torch.zeros_like(m.M)
    E.init_churn(m)
    churn, trace = E.new_churn_totals(), []
    for u in range(1, 31):
        m.M.grad = torch.randn(m.M.shape, generator=gen)
        trace.append(E.bop_step([m], .2 if u < 16 else .05, .08, True, churn=churn, keep_ema=keep, grace=2))
    data = b''.join(t.numpy().tobytes() for t in [m.M.detach(), m.bop_ema, m.nbirth, m.age, m.lastsign])
    data += json.dumps([trace, churn], sort_keys=True).encode()
    assert hashlib.sha256(data).hexdigest() == expected
    assert not hasattr(m, 'bop_v') and not hasattr(m, 'birth_step')


@pytest.mark.parametrize('flags', [
    ['--bop-tstat', '0'], ['--bop-tstat', 'nan'], ['--bop-tstat-floor', '.3'],
    ['--bop-tstat', '3', '--bop-tstat-floor', '-1'], ['--unit-burst', '-1'],
    ['--unit-burst', '2', '--unit-burst-thr', 'nan'], ['--unit-burst-random'],
    ['--unit-burst', '2', '--unit-burst-replay', 'schedule.json'],
    ['--bop-tstat', '3', '--unit-burst', '2'], ['--unit-burst', '2', '--down-only'],
    ['--bop-tstat', '3', '--pair-birth'], ['--unit-burst', '2', '--bop-tau-scale', 'energy'],
    ['--unit-burst', '2', '--adam-phase', '2'], ['--bop-tstat', '3', '--optim', 'adam'],
])
def test_invalid_c41_flags(flags):
    with pytest.raises(SystemExit):
        E.validate_c41_args(args('--optim', 'bop', *flags))


def test_birth_matching_complete_curves_support_not_births_and_no_guessed_tau(tmp_path, capsys):
    events = [dict(update=u, births=3 if u == 2 else 1, support_births=100) for u in range(1, 42)]
    assert E.bop_birth_totals(events) == dict(horizon=40, births=42)
    assert E.bop_birth_totals(events[:-1])['births'] == 42
    with pytest.raises(ValueError, match='every update'):
        E.bop_birth_totals(events[:39])
    target, candidate = tmp_path / 'target.json', tmp_path / 'candidate.json'
    target.write_text(json.dumps(dict(bop_events=events)))
    candidate.write_text(json.dumps(dict(bop_events=events, optim='bop', bop_tau=.25)))
    report = MATCH.main(['--target', str(target), '--candidates', str(candidate), '--emit-tau'])
    assert report['best']['relative_error'] == 0
    assert capsys.readouterr().out.strip() == '0.25'
    candidate.write_text(json.dumps(dict(bop_events=[dict(update=u, births=0) for u in range(1, 41)],
                                         optim='bop', bop_tau=.125)))
    with pytest.raises(SystemExit):
        MATCH.main(['--target', str(target), '--candidates', str(candidate), '--emit-tau'])
    target.write_text(json.dumps(dict(bop_events=events, seed=0)))
    candidate.write_text(json.dumps(dict(bop_events=events, seed=1, optim='bop', bop_tau=.25)))
    with pytest.raises(ValueError, match='differs from target'):
        MATCH.compare_births(target, [candidate])
    incomplete = tmp_path / 'incomplete.log'
    incomplete.write_text('\n'.join('BOP ' + json.dumps(e) for e in events[:39]))
    with pytest.raises(ValueError, match='every BOP update'):
        MATCH.read_births(incomplete)


@pytest.mark.parametrize('arm', [[], ['--bop-tstat', '3'], ['--unit-burst', '1', '--unit-burst-thr', '.01']])
def test_actual_cpu_training_logs_final_histograms_and_recipe(cpu_harness, capsys, arm):
    run, paths = cpu_harness
    result, _ = run(['--steps', '4', '--write-diagnostics', '--no-churn', '--no-pareto',
                     '--anchor', paths['anchor'], '--anchor-len', '16', *arm], 'c41')
    assert 'c41' in result
    assert result['c41']['cumulative_births'] == sum(e['births'] for e in result['bop_events'])
    for row in result['trajectory']:
        assert row['surviving_birth_steps']['known'] == row['nnz']
    assert result['c41']['surviving_birth_steps']['known'] == result['flipped']
    lines = capsys.readouterr().out.splitlines()
    final = json.loads(next(l[6:] for l in lines if l.startswith('FINAL ')))
    assert final['c41'] == result['c41']
    if '--unit-burst' in arm:
        assert len(result['c41']['distinct_units_per_fact']) == 6
        assert all('unit_bursts' in e for e in result['bop_events'])
    else:
        assert not any('unit_bursts' in e for e in result['bop_events'])


def test_default_recipe_has_no_c41_fields():
    a = args()
    E.validate_c41_args(a)
    assert not E.c41_diagnostics(a)
    recipe = E.training_recipe(a)
    assert not any(key in recipe for key in ('bop_tstat', 'bop_tstat_floor', 'unit_burst'))
