"""Regression evidence for the C35 design fixes, including production dtype arithmetic."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, _c3_pair8_model, cpu_harness
from test_c35 import ascii_tokenizer, resident_fixture
import branch_eval as B
import c38_analysis as A


def parse(*flags):
    argv = ['--method', 'fsa_forced', '--out', '/tmp/c38-unused.json', *flags]
    a = E.build_parser().parse_args(argv)
    a._explicit_flags = {v.split('=')[0] for v in argv if v.startswith('--')}
    return a


@pytest.mark.parametrize('flags', [('--route-kl', '.1'), ('--ctx-teacher', 'all'),
    ('--ctx-teacher-check',), ('--support-score', 'netcost'), ('--score-epoch',),
    ('--fisher-support',), ('--bop-tau-scale', 'energy'), ('--address-units', '1'),
    ('--units-per-fact', '1'), ('--address-center',), ('--gate-units-frac', '.1'),
    ('--wake-steps', '1'), ('--drift-lambda', '.1')])
def test_residents_reject_auxiliary_base_forwards(flags):
    for resident_flag in ('--resident-patch', '--resident-mask-only'):
        with pytest.raises(SystemExit, match='auxiliary base'):
            E.configure_c35(parse(resident_flag, 'A.pt', *flags))


@pytest.mark.parametrize('flag,value', [('--optim', 'bop'), ('--budget', '20'),
    ('--rank', '8'), ('--fmt', 'bf16'), ('--slot-frac', '.2'), ('--patch-scale-mult', '2')])
def test_factorial_conflicts_are_rejected(flag, value):
    with pytest.raises(SystemExit, match='conflicts'):
        E.configure_c35(parse('--factorial-cell', 'sparse_tern', flag, value))


def test_dense_resident_requires_override_and_alpha_matches_base():
    model = _c3_pair8_model()
    E.attach(model, [0], 'lora', 2)
    for pm in E.lora_modules(model):
        pm.B.data.fill_(.1)
    state = E.export_patch(model)
    E.detach_all(model)
    with pytest.raises(ValueError, match='allow-dense'):
        E.attach_residents(model, [state])
    assert not E.patched_linears(model)
    banks, _ = E.attach_residents(model, [state], allow_dense_resident=True)
    assert all(pm.slots().numel() > .5 * pm.delta_w().numel() for pm in banks[0])
    E.detach_all(model)
    E.attach(model, [0], 'lora', 2, lora_ternary=True)
    state = E.export_patch(model)
    E.detach_all(model)
    first = next(iter(state['entries'].values()))
    first['alpha'] *= 2
    with pytest.raises(ValueError, match='alpha differs'):
        E.attach_residents(model, [state], collect_slots=False)


def test_bf16_trained_saved_b_alone_equals_a_then_revoke_on_multiple_prompts():
    base = _c3_pair8_model().bfloat16().eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    source = copy.deepcopy(base)
    E.attach(source, [0], 'fsa_forced', 0, sparse=True)
    for pm in E.fsa_modules(source):
        pm.M.data.fill_(1)
    resident = E.export_patch(source)
    both = copy.deepcopy(base)
    params, _, _ = E.attach(both, [0], 'fsa_forced', 0, sparse=True, residents=[resident])
    banks, _ = E.attach_residents(both, [resident])
    # B writes in another layer because A occupies the first layer's full support.
    more, _, _ = E.attach(both, [1], 'fsa_forced', 0, sparse=True)
    params += more
    optimizer = torch.optim.Adam(params, lr=.3)
    prompts = [torch.tensor([[BOS_T, *tokens]]) for tokens in ([50,51], [60,61,62], [70,71,72,73])]
    for _ in range(4):
        optimizer.zero_grad()
        E.logits_of(both, prompts[0]).float().square().mean().backward()
        optimizer.step()
    state = E.export_patch(both)
    assert any(entry['val'].numel() for entry in state['entries'].values()), 'B must contain trained nonzero codes'
    alone = copy.deepcopy(base)
    E.attach(alone, [0,1], 'fsa_forced', 0, sparse=True, residents=[resident])
    E.load_patch(alone, state)
    on = [E.logits_of(both,x).detach().clone() for x in prompts]
    with E.resident_state(banks, False):
        assert E.weight_hashes(both) == E.weight_hashes(alone)
        for x in prompts:
            assert torch.equal(E.logits_of(both,x), E.logits_of(alone,x))
        assert any(not torch.equal(y, E.logits_of(both,x)) for y,x in zip(on,prompts))
    assert E.resident_revoke_check(both, banks, prompts)['prompts'] == 3


def test_revoke_weight_failure_hard_gates(monkeypatch):
    model = _c3_pair8_model()
    state = resident_fixture()
    E.attach(model, [0], 'fsa_forced', 0, residents=[state])
    banks, _ = E.attach_residents(model, [state])
    original = E.ResidentParam.forward
    monkeypatch.setattr(E.ResidentParam, 'forward', lambda self,W:
                        original(self,W) if self.enabled else W+1)
    with pytest.raises(RuntimeError, match='bit-exact'):
        E.resident_revoke_check(model, banks, [torch.tensor([[BOS_T,50]])])


def test_slot_mask_control_opportunities_match_and_zero_forward(cpu_harness):
    run, _ = cpu_harness
    _, resident = run(['--steps','2','--train-person','0'], 'mask_source')
    active, active_path = run(['--steps','2','--train-person','1','--resident-patch',str(resident)], 'mask_on')
    masked, masked_path = run(['--steps','2','--train-person','1','--resident-mask-only',str(resident),
                               '--zero-patch-control'], 'mask_off')
    full, _ = run(['--steps','2','--train-person','1'], 'mask_full')
    active_state, masked_state = [torch.load(p, weights_only=False) for p in (active_path,masked_path)]
    assert active_state['meta']['mask_sha256'] == masked_state['meta']['mask_sha256']
    assert masked['zero_patch_control']['passed']
    assert masked['sequential_revoke']['resident_slots']['mask_only']
    assert masked['sequential_revoke']['resident_only']['acq']['test_gen'] == masked['base']['test_gen']
    report = A.sequential(active, masked, full)
    assert report['dependence'] == active['sequential_revoke']['new_residents_revoked']['acq']['test_gen_person']['recall']-A.recall(masked)
    assert len(report['final_nnz']) == 3
    for state in (active_state,masked_state):
        occupied = torch.load(resident, weights_only=False)['entries']
        assert all(not set(e['idx'].tolist()) & set(occupied[name]['idx'].tolist()) for name,e in state['entries'].items())


def test_two_residents_original_recall_is_isolated(cpu_harness, monkeypatch):
    run, _ = cpu_harness
    _, first = run(['--steps','2','--train-person','0'], 'isolation_a')
    _, second = run(['--steps','2','--train-person','1'], 'isolation_c')
    banks = []
    attach, generate = E.attach_residents, E.gen_eval
    def capture(model, states, *args, **kwargs):
        result = attach(model, states, *args, **kwargs)
        if len(states) == 2:
            banks[:] = result[0]
        return result
    def score(model, *args, **kwargs):
        result = generate(model, *args, **kwargs)
        if banks and all(not pm.M.detach().ne(0).any() for pm in E.fsa_modules(model)):
            # Joint and solo measurements provably differ, regardless of Tiny's recall.
            result['recall'] = sum((i+1)*.1 for i,bank in enumerate(banks) if any(m.enabled for m in bank))
        return result
    monkeypatch.setattr(E,'attach_residents',capture)
    monkeypatch.setattr(E,'gen_eval',score)
    result, _ = run(['--steps','2','--train-person','2','--resident-patch',str(first),
                    '--resident-patch',str(second)], 'isolated_b')
    panel = result['sequential_revoke']['resident_only']
    assert panel['acq']['test_gen']['recall'] == pytest.approx(.3)
    assert [r['test_gen']['recall'] for r in panel['original_world_recall']] == [.1,.2]


def test_branch_revoke_baseline_precedes_attachment(cpu_harness, monkeypatch, tmp_path):
    monkeypatch.setattr(Tiny,'layers',property(lambda self:self.moes),raising=False)
    run, paths = cpu_harness
    patches = [str(run(['--steps','1','--train-person',str(pid)], 'pristine_'+str(pid))[1]) for pid in range(3)]
    original = B.Branches.__init__
    def corrupt(self, model, *args, **kwargs):
        original(self, model, *args, **kwargs)
        with torch.no_grad():
            E.patched_linears(model)[0][1].parametrizations.weight.original.add_(.125)
    monkeypatch.setattr(B.Branches,'__init__',corrupt)
    with pytest.raises(RuntimeError,match='pristine baseline'):
        B.main(['--patches',*patches,'--modes','oracle','--anchor',paths['anchor'],
            '--eval-set',paths['eval'],'--fineweb',paths['extra'],'--code',paths['extra'],
            '--fast-moe','off','--router-layer','0,1','--router-none','12','--router-steps','2',
            '--gen-max-new','1','--qa-n','2','--g-n','4','--out',str(tmp_path/'corrupt.json')])


def test_factorial_anchor_rng_is_matched_across_real_training(cpu_harness):
    run, paths = cpu_harness
    runs = []
    for cell in ('sparse_tern','sparse_fp','dense_tern','dense_fp'):
        result, _ = run(['--steps','2','--factorial-cell',cell, '--optim','adam','--budget','0',
            '--method','lora' if cell.startswith('dense') else 'fsa_forced',
            '--bop-tau-abs','0','--local-lambda','1','--anchor',paths['anchor'],'--anchor-len','16'], 'rng_'+cell)
        runs.append(result)
    assert runs[0]['factorial_accounting']['anchor_indices']
    assert all(r['factorial_accounting']['anchor_indices'] == runs[0]['factorial_accounting']['anchor_indices'] for r in runs)
    assert all('final_nnz' in r['factorial_accounting'] for r in runs)


def test_lr_search_uses_validation_with_equal_trial_budget(tmp_path):
    out = tmp_path/'search.json'
    argv = ['--factorial-cell','sparse_tern','--factorial-lrs','.001','.02','.1',
            '--steps','7','--out',str(out)]
    a = E.build_parser().parse_args(argv)
    calls = []
    def runner(flags):
        trial = E.build_parser().parse_args(flags)
        calls.append(trial)
        i = len(calls)-1
        return dict(validation_selection=dict(recall=[.4,.9,.2][i]),
                    patched=dict(test_gen=dict(recall=[1.,0.,1.][i])))
    result = E.factorial_lr_search(argv,a,runner)
    assert result['factorial_lr_search']['selected_lr'] == .02
    assert result['factorial_lr_search']['total_updates'] == 21
    assert all(trial.steps == 7 and trial.factorial_lrs is None for trial in calls)
    assert json.loads(out.read_text()) == result


def test_actual_lr_search_and_factorial_contrasts(cpu_harness):
    run, paths = cpu_harness
    results = []
    for cell in ('sparse_tern','sparse_fp','dense_tern','dense_fp'):
        result, _ = run(['--steps','1','--factorial-cell',cell,'--factorial-lrs','.001','.02',
            '--method','lora' if cell.startswith('dense') else 'fsa_forced','--optim','adam',
            '--budget','0','--bop-tau-abs','0','--bootstrap','100',
            '--anchor',paths['anchor'],'--anchor-len','16','--local-lambda','1'], 'search_'+cell)
        assert result['factorial_lr_search']['total_updates'] == 2
        assert (Path(result['patch_dir'])/'final.pt').exists()
        assert result['factorial_gap']['train_tf_minus_test_gen']['ci95']['clusters'] == 6
        results.append(result)
    report = A.factorial(results,100)
    assert report['contrasts']['interaction']['ci95']['clusters'] == 6
    broken = copy.deepcopy(results)
    broken[1]['factorial_lr_search']['total_updates'] += 1
    with pytest.raises(ValueError,match='budget differ'):
        A.factorial(broken,100)


def test_code_space_scale_and_saved_legacy_compatibility():
    W = torch.tensor([[.01,0,.01,0], [10.,0,10.,0]])
    pm = E.LoRAParam(W,1,True)
    pm.A.data.fill_(1)
    pm.B.data.fill_(.6)
    assert torch.equal(pm.delta_w(), pm.alpha.expand_as(W))
    entry = dict(kind='lora', A=pm.A.detach(), B=pm.B.detach(), alpha=pm.alpha, ternary=True, code_space=True)
    assert torch.equal(E.ResidentParam(W,entry).delta_w(),pm.delta_w())
    entry.pop('code_space')
    expected = ((pm.B@pm.A)/pm.alpha).round().clamp(-1,1)*pm.alpha
    assert torch.equal(E.ResidentParam(W,entry).delta_w(),expected)


def test_independent_lora_control_detects_corrupted_bank_sum(monkeypatch):
    model = _c3_pair8_model()
    E.attach(model,[0],'lora',2)
    for pm in E.lora_modules(model):
        pm.B.data.fill_(.3)
    state = E.export_patch(model,dict(method='lora',train_person=0))
    E.detach_all(model)
    branches = B.Branches(model,{0:state},{0:100},'sum_clip')
    facts = E.make_world(0,1,2,0)
    items = E.build_items(TOK,facts,E.TEST_T,BOS_T)
    answer = TOK.encode(' '+facts[0].val+'.',add_special_tokens=False).ids
    monkeypatch.setattr(E,'greedy_generate',lambda model,prompts,*args,**kw: [answer for _ in prompts])
    a = SimpleNamespace(gen_max_new=100,gen_stop_k=2,gen_bs=1)
    assert B.bank_control(branches,TOK,facts,items,'cpu',a)['solo_reproduced']
    original = B.LoRABankSum.forward
    monkeypatch.setattr(B.LoRABankSum,'forward',lambda self,W: original(self,W)+.1)
    assert not B.bank_control(branches,TOK,facts,items,'cpu',a)['solo_reproduced']


def test_fsa_sum_delta_retains_same_sign_overlap_and_one_bf16_cast():
    model = _c3_pair8_model().bfloat16()
    E.attach(model,[0],'fsa_forced',0,sparse=True)
    for pm in E.fsa_modules(model):
        pm.M.data.fill_(1)
    state = E.export_patch(model,dict(method='fsa_forced'))
    branches = B.Branches(model,{0:state,1:copy.deepcopy(state)},{0:10,1:10},'sum_delta')
    branches.activate((0,1))
    for _,lin in E.patched_linears(model):
        W = lin.parametrizations.weight.original
        delta = lin.parametrizations.weight[0].residents[0].delta_w()
        assert torch.equal(lin.weight,W+(delta+delta).to(W.dtype))
    E.set_enabled(False)
    assert all(torch.equal(lin.weight,lin.parametrizations.weight.original) for _,lin in E.patched_linears(model))
    E.set_enabled(True)


@pytest.mark.parametrize('flags', [[], ['--aug','--answer-only','--train-person','0'],
                                 ['--aug','--balance-people']])
def test_curriculum_semantics_and_direct_exposures(cpu_harness, flags):
    run, _ = cpu_harness
    direct, _ = run(['--steps','2',*flags], 'comp_direct')
    comp, _ = run(['--steps','2',*flags,'--comp-curriculum'], 'comp_aux')
    assert comp['composition_accounting']['direct_examples_per_fact'] == json.loads(open(direct['manifest']).read())['exposures']['direct_examples_per_fact']
    assert set(E.COMP_TASKS.values()).isdisjoint(E.EVAL_COMP_TASKS)
    assert comp['composition_accounting']['augmented'] == ('--aug' in flags)
    assert comp['composition_accounting']['answer_only'] == ('--answer-only' in flags)
    facts = E.make_world(0,3,4,4)
    groups = E.composition_training_items(TOK,facts,E.person_indices(facts,0),BOS_T)
    for kinds in groups.values():
        assert TOK.decode(kinds['paired'][0][0][kinds['paired'][0][1]:]) in {' ascending.', ' descending.', ' equal.'}
        assert {TOK.decode(seq[start:]) for seq,start in kinds['inverse']} == {' verified.',' rejected.'}
        assert {TOK.decode(seq[start:]) for seq,start in kinds['null']} == {' missing.',' covered.'}


def test_random_synthetic_labels_do_not_encode_indices_and_do_not_touch_rng():
    state = random.getstate()
    one = E.make_world(7,per_person=100,shared=100,synthetic_people=30)
    other = E.make_world(8,per_person=100,shared=100,synthetic_people=30)
    assert random.getstate() == state
    assert len({f.val for f in one}) == 3000
    assert all(f.val.isalpha() and len(f.val) == 16 for f in one)
    assert [f.val for f in one] != [f.val for f in other]


def test_paired_gap_ci_and_dose_budget_rejection():
    left = [dict(fi=0,value=1),dict(fi=0,value=1),dict(fi=1,value=0)]
    right = [dict(fi=0,value=0),dict(fi=1,value=0)]
    report = E.paired_cluster_contrast(left,right,1000,9)
    assert report['estimate'] == .5 and report['ci95']['clusters'] == 2
    with pytest.raises(ValueError,match='identical fact'):
        E.paired_cluster_contrast(left,right[:1])
    base = dict(facts=[],seed=0,train_person=None,train_attr_group=None,local_lambda=1,lr=.01,method='fsa_forced',fmt='ternary',
                dose_response=dict(new_facts=100,new_updates=300,new_fact_format_steps=2400,
                                   new_examples_per_fact=[24]*100,new_loss_tokens=5000,
                                   new_examples_per_global_fact={},new_recipe={}))
    changed = copy.deepcopy(base)
    changed['dose_response']['new_facts'] = 300
    with pytest.raises(ValueError,match='exposures differ'):
        A.require_same_new_world([base,changed])
