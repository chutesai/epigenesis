"""C59 regressions use real Bop, serialized banks, and the tiny MoE harness."""
import copy
import json

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, cpu_harness
from test_c52 import C, args, base, states
import c59 as R


def test_legacy_recorded_tau_and_terminal_gamma(tmp_path):
    state = states()[0]
    state['meta']['training_recipe'] = dict(optim='bop', bop_tau_abs=0., bop_gamma=.05,
        bop_gamma_late=.02, bop_gamma_switch=2, tau_anneal=2., death_grace=7)
    path = tmp_path/'R20_unit_p0'/'final.pt'
    path.parent.mkdir()
    torch.save(state, path)
    with pytest.raises(ValueError, match='Refusing to recalibrate'):
        R.calibration(state, path)
    (tmp_path/'R20_unit_p0.manifest.json').write_text(json.dumps(dict(optimizer=dict(bop_tau_abs=0.))))
    with pytest.raises(ValueError, match='missing original Bop'):
        R.calibration(state, path)
    (tmp_path/'R20_unit_p0.json').write_text(json.dumps(dict(bop_tau_abs=.003, seed=0, train_person=0)))
    cal = R.calibration(state, path)
    assert cal['tau'] == .006 and cal['gamma'] == .02 and cal['death_grace'] == 7
    assert cal['budget'] == 20 and cal['budget_freeze'] == -1
    assert cal['source'].endswith('R20_unit_p0.json')


@pytest.mark.parametrize('field,value', [('tau', float('nan')), ('gamma', 0), ('death_grace', -1)])
def test_invalid_saved_calibration_fails(field, value):
    state = states()[0]
    state['meta']['training_recipe'].pop('bop_tau_abs')
    state['meta']['bop_calibration'] = dict(tau=.007, gamma=.05)
    state['meta']['bop_calibration'][field] = value
    with pytest.raises(ValueError, match='C59'):
        R.calibration(state)


def test_complete_calibration_and_dormant_support_roundtrip():
    model = base()
    banks = C.attach_banks(E, model, states(budget=64))
    for bank in banks:
        bank.calibration = dict(tau=.007, gamma=.03, budget=64, cost_lambda=.6, den_total=12.,
            death_grace=9, death_tau_mult=2., keep_ema=True, bop_tau_scale='energy', support_score='netcost')
        bank.tau = .007
        for m in bank.modules.values():
            m.support[:2] = True
            m.tau_scale = torch.full_like(m.M, 1.5)
            m.col_energy = torch.ones(m.wshape[1])
            m.age = torch.full_like(m.M, 11, dtype=torch.int8)
            m.bop_ema.normal_(std=.0001)
    exported = [C.export_bank(E,b,C.Schedule(0,3),'J',1) for b in banks]
    restored = C.attach_banks(E,base(),exported)
    for original, bank, state in zip(banks, restored, exported):
        cal = R.calibration(state)
        R.prepare(bank.modules, cal, exact_support=True)
        assert cal['tau'] == .007 and cal['gamma'] == .03 and cal['cost_lambda'] == .6
        for name, m in bank.modules.items():
            for field in R.TENSORS:
                a,b = getattr(original.modules[name],field,None),getattr(m,field,None)
                assert a is b if a is None else torch.equal(a,b)
            assert m.support[1] and m.M[1] == 0
    bad = copy.deepcopy(exported)
    first = next(iter(bad[0]['joint_state']['modules'].values()))
    first['tau_scale'] = None
    broken = C.attach_banks(E,base(),bad)
    with pytest.raises(ValueError,match='missing saved energy'):
        R.prepare(broken[0].modules,R.calibration(bad[0]),exact_support=True)


@pytest.mark.parametrize('initial,current,update,fails', [(100,120,1,False),(100,121,1,True),
    (100,121,10,True),(100,121,11,False),(0,1,1,True)])
def test_guard_boundary(initial,current,update,fails,capsys):
    if fails:
        with pytest.raises(RuntimeError,match='nnz grew >20%'):
            R.check_growth(initial,current,update,'bank 2')
        assert 'bank 2' in capsys.readouterr().out
    else:
        R.check_growth(initial,current,update,'bank 2')


def quiet_joint_eval(monkeypatch):
    monkeypatch.setattr(C,'evaluate',lambda *a,**kw: {})


def converged_banks():
    model = base()
    banks = C.attach_banks(E,model,states(budget=64))
    with torch.no_grad():
        for bank in banks:
            for m in bank.modules.values():
                m.M[:4] = 1
                m.support[:4] = True
    facts,owners,scopes = C.world(E,states())
    a = args()
    a.fact_batch = 0
    schedule = C.Schedule(0,3,p_iso=1,dropout=0)
    rows = C.batch(E,TOK,facts,owners,scopes,schedule,a,BOS_T)
    for bank in banks:
        bank.calibration = R.calibration(bank.state)
        bank.calibration.update(gamma=1., tau=.02)
        bank.tau = bank.calibration['tau']
        R.prepare(bank.modules,bank.calibration,exact_support=True)
    # Original Bop training to a fixed point on each bank's own facts. gamma=1
    # makes fixed-point detection exact, with no latent EMA convergence delay.
    def backward():
        for bank in banks:
            for m in bank.modules.values():
                m.M.grad = None
        loss_sum = 0.
        for row in rows:
            with C.activation(banks,row['active'],row['relevant']):
                loss = C.example_loss(E,model,TOK,banks,row,'cpu')
                loss_sum += float(loss.detach())/len(rows)
                (loss/len(rows)).backward()
        return loss_sum
    for _ in range(100):
        backward()
        changed = sum(sum(E.bop_step(list(b.modules.values()),1.,b.tau,True)) for b in banks)
        if changed == 0:
            break
    assert changed == 0, 'original fixture must reach a Bop fixed point'
    loss = backward()
    return banks,loss


@pytest.mark.parametrize('frozen',[False,True])
def test_converged_banks_isolated_continue_nnz_and_own_fact_loss(cpu_harness,tmp_path,monkeypatch,frozen):
    run,_ = cpu_harness
    quiet_joint_eval(monkeypatch)
    banks, before_loss = converged_banks()
    paths = []
    for i,bank in enumerate(banks):
        path = tmp_path/f'converged{i}.pt'
        torch.save(C.export_bank(E,bank,C.Schedule(0,3),'J',0),path)
        paths.append(str(path))
    result,_ = run(['--joint-banks',*paths,'--steps','10','--fact-batch','0',
        '--p-iso','1','--presence-dropout','0','--refresh-every','1','--budget-freeze','150','--budget','64',
        *(['--joint-frozen-distractors'] if frozen else [])],f'continue_{frozen}')
    initial = [R.nnz(b.modules.values()) for b in banks]
    assert all([b['nnz'] for b in row['banks']] == initial for row in result['trajectory'])
    assert all(row['loss'] <= before_loss+1e-6 for row in result['trajectory'])
    assert all(b['tau'] == .02 and b['gamma'] == 1. for b in result['bop_calibrations'])
    assert result['budget_freeze'] == -1
    for path, bank in zip(result['patch_files'],banks):
        state = torch.load(path,weights_only=False)
        for name,m in bank.modules.items():
            assert torch.equal(state['joint_state']['modules'][name]['support'],m.support.cpu())


def test_joint_missing_calibration_fails_before_training(cpu_harness,tmp_path,monkeypatch):
    run,_ = cpu_harness
    saved = states()
    paths = []
    for i,state in enumerate(saved):
        state['meta']['training_recipe']['bop_tau_abs'] = 0
        path = tmp_path/f'missing{i}.pt'
        torch.save(state,path)
        paths.append(str(path))
    monkeypatch.setattr(C,'batch',lambda *a,**kw: pytest.fail('must not reach training'))
    with pytest.raises(ValueError,match='Refusing to recalibrate'):
        run(['--joint-banks',*paths,'--steps','1'],'missing')


def test_single_patch_continuation_restores_calibration_and_support(cpu_harness,monkeypatch):
    run,_ = cpu_harness
    _,path = run(['--steps','4','--sparse-state','--bop-gamma','.2','--death-grace','3'],'original')
    state = torch.load(path,weights_only=False)
    cal = state['meta']['bop_calibration']
    old_bop = E.bop_step
    def bop(mods,gamma,tau,budgeted,**kw):
        assert gamma == cal['gamma'] and tau == cal['tau']
        assert budgeted and kw['grace'] == 3
        return old_bop(mods,gamma,tau,budgeted,**kw)
    monkeypatch.setattr(E,'bop_step',bop)
    monkeypatch.setattr(E,'refresh_support',lambda *a,**kw: pytest.fail('saved support must stay frozen'))
    result,out = run(['--load-patch',str(path),'--steps','10','--sparse-state','--bop-tau-abs','1e-12',
                      '--budget-freeze','150','--refresh-every','1'],'single_continue')
    assert result['budget_freeze'] == -1
    final = torch.load(out,weights_only=False)
    assert final['meta']['bop_calibration']['tau'] == cal['tau']
    for name,row in state['continue_state']['modules'].items():
        assert torch.equal(row['support'],final['continue_state']['modules'][name]['support'])


@pytest.mark.parametrize('joint',[False,True])
def test_real_continue_guard_aborts_before_export(cpu_harness,tmp_path,monkeypatch,joint,capsys):
    run,_ = cpu_harness
    if joint:
        paths=[]
        for i,state in enumerate(states()):
            path=tmp_path/f'guard{i}.pt'
            torch.save(state,path)
            paths.append(str(path))
        flags=['--joint-banks',*paths]
    else:
        _,path=run(['--steps','2','--sparse-state'],'guard_original')
        flags=['--load-patch',str(path),'--sparse-state']
    def broken_bop(mods,*a,**kw):
        with torch.no_grad():
            for m in mods:
                m.support = torch.ones_like(m.support)
                m.M.fill_(1)
        return 10000,0
    monkeypatch.setattr(E,'bop_step',broken_bop)
    with pytest.raises(RuntimeError,match='C59 abort'):
        run([*flags,'--steps','10'],'guard_failed')
    output=capsys.readouterr().out
    assert 'first 10 updates' in output and 'saved' in output
    assert not (tmp_path/'guard_failed_patches'/'final.pt').exists()
    assert not (tmp_path/'guard_failed_patches'/'bank0'/'final.pt').exists()


def test_dense_bank_state_converts_row_energy_to_sparse_slots():
    state = states(budget=64)[0]
    model = base()
    E.attach(model,[0,1],'fsa_forced',0,budgeted=True,slot_partition=3,slot_part=0)
    modules={name:lin.parametrizations.weight[0] for name,lin in E.patched_linears(model)}
    for m in modules.values():
        m.support.zero_()
    E.load_patch(model,state)
    cal=R.calibration(state)
    cal.update(bop_tau_scale='energy',support_score='netcost',cost_lambda=.4)
    for m in modules.values():
        m.tau_scale=torch.linspace(.25,4,m.wshape[0])[:,None]
        m.col_energy=torch.ones(m.wshape[1])
        m.bop_ema=torch.zeros_like(m.M)
    state['continue_state']=R.snapshot(modules,cal)
    bank=C.attach_banks(E,base(),[state])[0]
    R.prepare(bank.modules,R.calibration(state),exact_support=True)
    for name,m in bank.modules.items():
        assert m.tau_scale.shape == m.M.shape
        assert torch.equal(m.tau_scale,modules[name].tau_scale.flatten()[m.rows()])
        m.M.grad=torch.ones_like(m.M)*.001
    E.bop_step(list(bank.modules.values()),cal['gamma'],cal['tau'],True)


def test_single_missing_original_tau_does_not_use_command_or_new_gradients(cpu_harness,tmp_path):
    run,_=cpu_harness
    _,path=run(['--steps','2','--sparse-state'],'missing_single_source')
    state=torch.load(path,weights_only=False)
    state.pop('continue_state')
    state['meta'].pop('bop_calibration')
    state['meta']['training_recipe']['bop_tau_abs']=0.
    orphan=tmp_path/'orphan.pt'
    torch.save(state,orphan)
    with pytest.raises(ValueError,match='Refusing to recalibrate'):
        run(['--load-patch',str(orphan),'--steps','1','--sparse-state','--bop-tau-abs','1e-12'],'missing_single')
