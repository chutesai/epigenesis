"""C35 CPU coverage of frozen residents, matched recipes, composition and LoRA banks."""
import copy
import json
import random

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, _c3_pair8_model, cpu_harness
import branch_eval as B


@pytest.fixture(autouse=True)
def ascii_tokenizer(monkeypatch):
    original = type(TOK).encode
    monkeypatch.setattr(type(TOK), 'encode', lambda self, text, **kw:
                        original(self, text.replace('—', '-'), **kw))


def args(*flags):
    a = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "/tmp/c35-unused.json", *flags])
    E.configure_c35(a)
    return a


def test_factorial_recipes_match_optimizer_penalty_and_defaults():
    default = args()
    assert not default.resident_patch and not default.lora_ternary and not default.bootstrap
    assert default.lr is None and default.optim == 'adam' and not default.comp_curriculum
    cells = [args('--factorial-cell', cell, '--local-lambda', '10', '--steps', '123', '--layers', '0,1')
             for cell in ('sparse_tern', 'sparse_fp', 'dense_tern', 'dense_fp')]
    assert [(a.method, a.fmt, a.lora_ternary) for a in cells] == [
        ('fsa_forced', 'ternary', False), ('fsa_forced', 'bf16', False),
        ('lora', 'ternary', True), ('lora', 'bf16', False)]
    assert {(a.optim, a.lr, a.local_lambda, a.steps, a.layers, a.rank, a.budget) for a in cells} == {
        ('adam', .001, 10., 123, '0,1', 16, 0)}
    assert args('--factorial-cell', 'sparse_fp', '--lr', '.002').lr == .002
    with pytest.raises(SystemExit, match='requires'):
        args('--method', 'fsa_forced', '--lora-ternary')


def test_ternary_product_forward_gradient_penalty_and_roundtrip():
    W = torch.tensor([[1., 0., -1., 0.], [0., 2., 0., -2.]])
    pm = E.LoRAParam(W, 2, ternary=True)
    with torch.no_grad():
        pm.A.copy_(torch.tensor([[1., -.8, .1, 2.], [0., 0., 0., 0.]]))
        pm.B.copy_(torch.tensor([[1., 0.], [2., 0.]]))
    E.set_enabled(True)
    expected = torch.tensor([[1., -1., 0., 1.], [2., -2., 0., 2.]])
    assert torch.equal(pm.delta_w(), expected)
    assert torch.equal(pm(W), W + expected)
    pm.delta_w().sum().backward()
    assert pm.A.grad.abs().sum() > 0 and pm.B.grad.abs().sum() > 0
    model = _c3_pair8_model()
    E.attach(model, [0], 'lora', 2, lora_ternary=True)
    for m in E.lora_modules(model):
        m.B.data.fill_(2.)
    state = E.export_patch(model)
    delta = [m.delta_w().detach().clone() for m in E.lora_modules(model)]
    for m in E.lora_modules(model):
        m.B.data.zero_()
    E.load_patch(model, state)
    assert all(torch.equal(m.delta_w(), d) for m, d in zip(E.lora_modules(model), delta))
    penalty = E.LocalOutputPenalty(model)
    penalty.start()
    E.logits_of(model, torch.tensor([[BOS_T, 50, 51]]))
    loss = penalty.stop()
    assert loss is not None and loss > 0
    loss.backward()
    penalty.end()
    for h in penalty.h:
        h.remove()


def resident_fixture():
    model = _c3_pair8_model()
    E.attach(model, [0, 1], 'fsa_forced', 0, sparse=True)
    for pm in E.fsa_modules(model):
        pm.M.data[:2] = 1
    return E.export_patch(model, dict(method='fsa_forced', train_person=0))


@pytest.mark.parametrize('sparse', [False, True])
def test_residents_frozen_disjoint_and_revocation_restores_base(sparse):
    state = resident_fixture()
    model = _c3_pair8_model()
    x = torch.tensor([[BOS_T, 50, 51]])
    original = E.logits_of(model, x).detach()
    params, _, _ = E.attach(model, [0], 'fsa_forced', 0, sparse=sparse, residents=[state])
    banks, slots = E.attach_residents(model, [state, state])
    assert all(not list(m.parameters()) for bank in banks for m in bank)
    assert not {id(b) for bank in banks for m in bank for b in m.buffers()} & {id(p) for p in params}
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        if isinstance(pm, E.FSAParam):
            mask = pm._dense(pm.mask).bool() if sparse else pm.mask
            assert not mask.flatten()[slots[name]].any()
            pm.M.data.fill_(1)
    frozen = [[b.clone() for m in bank for b in m.buffers()] for bank in banks]
    E.logits_of(model, x).sum().backward()
    torch.optim.Adam(params, lr=.01).step()
    assert all(torch.equal(b, old) for bank, saved in zip(banks, frozen)
               for b, old in zip([b for m in bank for b in m.buffers()], saved))
    exported = E.export_patch(model)
    assert all(not set(e['idx'].tolist()) & set(slots[n].tolist()) for n, e in exported['entries'].items())
    assert all(n.startswith('moes.0.') for n in exported['entries'])
    E.set_enabled(False)
    assert not torch.equal(E.logits_of(model, x), original)  # residents stay on through student switches
    with E.resident_state(banks, False):
        assert torch.equal(E.logits_of(model, x), original)
    E.set_enabled(True)


def test_resident_block_scales_and_lora_zero_context_keep_frozen_delta():
    W = torch.tensor([[1., 0., 1., 0.]])
    pm = E.FSAParam(W, W.eq(0))
    pm.enable_block_scale(2, 2.)
    pm.M.data.fill_(1)
    entry = dict(kind='fsa', shape=list(W.shape), idx=torch.tensor([1, 3]), val=torch.ones(2),
                 alpha=pm.alpha.squeeze(1), scale='block', logb=pm.logb.detach(), block=2)
    resident = E.ResidentParam(W, entry)
    assert torch.equal(resident.delta_w(), pm.delta_w())
    model = _c3_pair8_model()
    E.attach(model, [0], 'lora', 2)
    for m in E.lora_modules(model):
        m.B.data.fill_(.2)
    state = E.export_patch(model)
    E.detach_all(model)
    E.attach(model, [0], 'lora', 2, residents=[state])
    banks, _ = E.attach_residents(model, [state], allow_dense_resident=True)
    before = [m.B.clone() for bank in banks for m in bank]
    with E.zero_patch(model):
        assert all(torch.equal(m.B, b) for m, b in zip(banks[0], before))


@pytest.mark.parametrize('people, per_person', [(10, 10), (30, 10), (10, 100), (30, 100), (9, 350)])
def test_synthetic_world_unique_reproducible_and_extends_attributes(people, per_person):
    facts = E.make_world(7, per_person=per_person, shared=per_person, synthetic_people=people)
    assert facts == E.make_world(7, per_person=per_person, shared=per_person, synthetic_people=people)
    assert len(facts) == people * per_person
    assert len({f.name for f in facts}) == people
    assert len({f.val for f in facts}) == len(facts)
    assert all(len({f.attr for f in facts if f.pid == pid}) == per_person for pid in range(people))
    assert args('--synthetic-people', '30').people == 30


def test_composition_training_uses_taught_pairs_inverse_and_truly_absent_nulls():
    facts = E.make_world(0, 3, 4, 4)
    ids = E.person_indices(facts, 1)
    groups = E.composition_training_items(TOK, facts, ids, BOS_T)
    for fi, kinds in groups.items():
        assert all(kinds.values())
        paired = TOK.decode(kinds['paired'][0][0])
        assert facts[fi].attr in paired and facts[fi].val not in paired
        assert all(f.val not in paired for f in facts if f.pid != 1)
        inverse = TOK.decode(kinds['inverse'][0][0])
        assert facts[fi].name in inverse and inverse.endswith("verified.")
        null = TOK.decode(kinds['null'][0][0])
        assert null.endswith("missing.")
        assert all(f.attr not in null for f in facts if f.pid == 1)
    weights = dict(direct=0, paired=0, inverse=0, null=1)
    chosen = E.composition_batch(groups, ids, weights, random.Random(2))
    assert len(chosen) == 3 * len(ids)
    assert all(TOK.decode(seq[start:]) in {" missing.", " covered."} for _, (seq, start) in chosen)
    one = E.composition_training_items(TOK, facts, ids[:1], BOS_T)
    assert not one[ids[0]]['paired']


def test_bootstrap_fact_clusters_isolated_rng_and_qa_integration(monkeypatch):
    items = [dict(fi=0)] * 10 + [dict(fi=1)]
    py, th = random.getstate(), torch.get_rng_state()
    ci = E.cluster_bootstrap(items, [1] * 10 + [0], 1000, 5)
    assert ci['clusters'] == 2 and ci['low'] == 0 and ci['high'] == 1
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), th)
    assert ci == E.cluster_bootstrap(items, [1] * 10 + [0], 1000, 5)
    facts = E.make_world(0, 3, 2, 2)
    groups = E.build_qa_items(TOK, facts, BOS_T, 2)
    monkeypatch.setattr(E, 'greedy_generate', lambda model, prompts, *a: [TOK.encode(' wrong.').ids for _ in prompts])
    qa = E.qa_probe_eval(None, TOK, groups, 'cpu', bootstrap=1000)
    assert qa['ci95']['two_fact_presence_stop']['repeats'] == 1000
    gen = E.gen_eval(None, TOK, E.build_items(TOK, facts, E.TEST_T, BOS_T), 'cpu', bootstrap=1000)
    assert gen['ci95']['recall']['clusters'] == len(facts)


@pytest.mark.parametrize('cell', ['sparse_tern', 'sparse_fp', 'dense_tern', 'dense_fp'])
def test_main_factorial_person_and_comp_curriculum(cpu_harness, cell):
    run, paths = cpu_harness
    result, path = run(['--steps', '2', '--factorial-cell', cell, '--optim', 'adam', '--budget', '0',
                        '--method', 'lora' if cell.startswith('dense') else 'fsa_forced', '--bop-tau-abs', '0',
                        '--train-person', '1', '--anchor', paths['anchor'], '--anchor-len', '16',
                        '--local-lambda', '10', '--comp-curriculum', '--qa-probe', '--qa-n', '2',
                        '--bootstrap', '1000'], cell)
    state = torch.load(path, weights_only=False)
    assert result['c35_recipe']['factorial_cell'] == cell and result['lr'] == .001
    assert result['patched']['test_gen_person']['n'] == 4
    assert result['patched']['test_gen_person']['ci95']['recall']['clusters'] == 2
    assert result['qa_probe']['ci95']['two_fact_presence_stop']['repeats'] == 1000
    assert result['revoked']['logits_bit_exact']
    assert state['meta']['c35_recipe']['comp_curriculum']
    if cell.startswith('dense'):
        assert all(e['kind'] == 'lora' for e in state['entries'].values())
        assert any(e['B'].abs().sum() > 0 for e in state['entries'].values())


def test_main_sequential_revoke_and_repeated_residents(cpu_harness):
    run, _ = cpu_harness
    _, path_a = run(['--steps', '2', '--train-person', '0', '--sparse-state'], 'resident_a')
    original = torch.load(path_a, weights_only=False)
    result, path_b = run(['--steps', '2', '--train-person', '1', '--sparse-state',
                          '--resident-patch', str(path_a), '--resident-patch', str(path_a),
                          '--zero-patch-control'], 'resident_b')
    sequential = result['sequential_revoke']
    assert sequential['resident_slots']['patches'] == 2
    assert sequential['resident_slots']['union_slots'] == sum(len(e['idx']) for e in original['entries'].values())
    assert sequential['new_residents_revoked']['acq']['test_gen_person']['n'] == 4
    assert sequential['resident_only']['big'] is not None
    assert result['zero_patch_control']['passed'] and result['revoked']['logits_bit_exact']
    saved_b = torch.load(path_b, weights_only=False)
    for name, e in saved_b['entries'].items():
        assert not set(e['idx'].tolist()) & set(original['entries'][name]['idx'].tolist())
    assert all(torch.equal(e['val'], original['entries'][name]['val'])
               for name, e in torch.load(path_a, weights_only=False)['entries'].items())


@pytest.mark.parametrize('ternary', [False, True])
def test_lora_bank_switch_sum_router_and_base_features(ternary, monkeypatch):
    monkeypatch.setattr(Tiny, 'layers', property(lambda self: self.moes), raising=False)
    model = _c3_pair8_model()
    x = torch.tensor([[BOS_T, 50, 51]])
    base = E.logits_of(model, x).detach()
    E.attach(model, [0], 'lora', 2, lora_ternary=ternary)
    states = {}
    for person in range(3):
        for pm in E.lora_modules(model):
            pm.B.data.fill_(person + 1.)
        states[person] = E.export_patch(model, dict(method='lora', fmt='ternary' if ternary else 'bf16',
            people=3, train_person=person, layers=[0], rank=2, patch_scale='row', lora_ternary=ternary))
    B.validate_patches(list(states.values()))
    E.detach_all(model)
    branches = B.Branches(model, states, {0: 10, 1: 20, 2: 30}, 'sum_clip')
    branches.activate((0, 1, 2))
    for name, lin in E.patched_linears(model):
        expected = lin.parametrizations.weight.original.detach().clone()
        for person in states:
            expected = expected + E.ResidentParam(lin.parametrizations.weight.original.detach(), states[person]['entries'][name]).delta_w()
        assert torch.allclose(lin.weight, expected)
    summed = E.logits_of(model, x).detach()
    branches.activate((1,))
    assert not torch.equal(E.logits_of(model, x), summed)
    features = B.indexer_features(model, [x[0].tolist()], 0, 'cpu')
    assert features.shape[0] == 1
    E.set_enabled(False)
    assert torch.equal(E.logits_of(model, x), base)
    E.set_enabled(True)
    probe = B.fit_router(torch.tensor([[-4., 0.], [4., 0.], [0., 4.], [0., -4.]]), [0, 1, 2, 3], 4, steps=80)
    routes = [() if p == 3 else (p,) for p in probe.predict(torch.tensor([[-5., 0.], [5., 0.], [0., -5.]])).tolist()]
    assert routes == [(0,), (1,), ()]
    for route in routes:
        branches.activate(route)
        assert all(m.enabled == (person in route) for person, bank in branches.lora_banks.items() for m in bank)
    branches.activate(())
    assert torch.equal(E.logits_of(model, x), base)
    assert B.lora_overlap_stats(list(states.values()))['overlapping_slots'] > 0
    bad = copy.deepcopy(list(states.values()))
    bad[1]['entries'][next(iter(bad[1]['entries']))]['B'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='factors'):
        B.validate_patches(bad)


@pytest.mark.parametrize('cell', ['dense_fp', 'dense_tern'])
def test_branch_main_lora_banks_all_policies(cpu_harness, monkeypatch, tmp_path, cell):
    monkeypatch.setattr(Tiny, 'layers', property(lambda self: self.moes), raising=False)
    run, paths = cpu_harness
    patches = []
    seen = []
    encode = E.encode_fact
    def record(tok, template, fact, bos):
        seen.append(fact.pid)
        return encode(tok, template, fact, bos)
    monkeypatch.setattr(E, 'encode_fact', record)
    for person in range(3):
        seen.clear()
        result, path = run(['--steps', '2', '--factorial-cell', cell, '--optim', 'adam', '--budget', '0',
                        '--method', 'lora' if cell.startswith('dense') else 'fsa_forced', '--bop-tau-abs', '0',
                           '--train-person', str(person)], f'{cell}_bank{person}')
        assert set(seen) == {person}
        assert result['revoked']['logits_bit_exact']
        patches.append(str(path))
    out = tmp_path / 'lora_branches.json'
    B.main(['--patches', *patches, '--modes', 'oracle', 'sum', 'router',
            '--anchor', paths['anchor'], '--eval-set', paths['eval'],
            '--fineweb', paths['extra'], '--code', paths['extra'], '--fast-moe', 'off',
            '--router-layer', '0,1', '--router-none', '12', '--router-steps', '20',
            '--gen-max-new', '1', '--qa-n', '2', '--g-n', '4', '--bootstrap', '1000', '--out', str(out)])
    result = json.loads(out.read_text())
    assert {key.split(':')[0] for key in result['arms']} == {'B', 'C', 'D'}
    assert result['zero_patch_control']['passed'] and result['revoked']['logits_bit_exact']
    assert result['selector_selection']['C']['locked']
    for report in result['arms'].values():
        assert report['test_gen']['ci95']['recall']['clusters'] == 6
        assert report['two_fact']['ci95']['two_fact_presence_stop']['repeats'] == 1000
        assert report['unseen']['ci95']['recall']['repeats'] == 1000
        assert report['damage']


def test_lora_sum_accumulates_before_bf16_rounding():
    W = torch.ones(1, 1, dtype=torch.bfloat16)
    entry = dict(kind='lora', A=torch.ones(1, 1), B=torch.full((1, 1), .002))
    residents = [E.ResidentParam(W, entry) for _ in range(3)]
    E.set_enabled(True)
    summed = B.LoRABankSum(residents)(W)
    assert torch.equal(summed, W + torch.tensor([[.006]], dtype=torch.bfloat16))
    assert summed > W
    residents[1].enabled = residents[2].enabled = False
    assert torch.equal(B.LoRABankSum(residents)(W), W)
