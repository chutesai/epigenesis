"""Pure-logic contracts for the session-1 ablation (no model or CUDA required)."""
import copy
import math
import os
from pathlib import Path
import random
import re
import sys
import subprocess
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ablate as a
from make_user import build_corpus, case_matches, SELFSTUDY_POOL, NAME_POOLS, large_dev_concepts


@pytest.fixture(scope='module')
def corpus():
    return build_corpus()


def test_cases_source_only_reproducible_and_satisfy_conditions(corpus):
    source_only = {k: copy.deepcopy(corpus[k]) for k in ('policies', 'sessions')}
    cases = a.build_cases(source_only, cases_per_policy=4)
    assert cases == a.build_cases(source_only, cases_per_policy=4)
    assert cases != a.build_cases(source_only, seed=1, cases_per_policy=4)
    policies = {p['id']: p for p in corpus['policies']}
    eligible = {p['id'] for p in corpus['policies'] if p['intro_session'] <= 1}
    assert len(cases) == len(eligible) * 4
    assert len({c['id'] for c in cases}) == len(cases)
    assert {c['policy_id'] for c in cases} == eligible
    for case in cases:
        policy = policies[case['policy_id']]
        assert case_matches(policy, case['case'])
        assert case['case']['customer'] in SELFSTUDY_POOL
        assert policy['codename'] in case['source_text']
        assert len(case['formats']) == 6
        for rendered in case['formats']:
            assert policy['codename'] not in rendered['prompt']
            lo, hi = rendered['action_span']
            assert rendered['answer'][lo:hi] == policy['action']
            assert policy['action'] not in rendered['prompt']
    assert sum(c['base_format'] == 0 for c in cases) == len(cases) // 2
    assert sum(c['base_format'] == 1 for c in cases) == len(cases) // 2


def test_formats_match_probe_families_and_augmentation(corpus):
    case = a.build_cases(corpus, cases_per_policy=2)[0]
    policy = next(p for p in corpus['policies'] if p['id'] == case['policy_id'])
    ticket = a.render_ticket(case['case'])
    formats = case['formats']
    assert formats[0]['prompt'] == ticket + '\nResolution:'
    assert formats[0]['answer'] == f" STATUS: {policy['status']} | ACTION: {policy['action']} | NOTE: {policy['rationale_text']}"
    assert formats[1]['prompt'] == f"Q: Under the user's policies, which action applies to {ticket}\nA:"
    assert formats[1]['answer'] == ' ' + policy['action']
    assert formats[2]['prompt'] + formats[2]['answer'] == f"User: {ticket}\nWhich action does the desk take?\nAssistant: {policy['action']}"
    assert formats[3]['prompt'] + formats[3]['answer'] == f"{ticket} The desk will {policy['action']}."
    assert formats[4]['prompt'] + formats[4]['answer'] == f"Policy check: {ticket} -> {policy['action']}"
    assert formats[5]['prompt'] + formats[5]['answer'] == f"Q: What should the desk do with this ticket? {ticket}\nA: {policy['action']}"
    with pytest.raises(ValueError):
        a.render_case(case['case'], policy, 6)


class OffsetTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Leading whitespace belongs to the following token, as in BPE.
        spans = [(m.start(), m.end()) for m in re.finditer(r'\s*\S+', text)]
        return SimpleNamespace(ids=list(range(1, len(spans) + 1)), offsets=spans)


def test_action_token_offsets_and_teacher_gate():
    item = dict(id='i', policy_id='p', source_episode='ep')
    target = ' STATUS: PENDING | ACTION: request photo evidence | NOTE: needed'
    lo = target.index('request')
    rendered = dict(prompt='Ticket\nResolution:', answer=target, action_span=(lo, lo + len('request photo evidence')))
    encoded = a.encode_item(item, rendered, OffsetTokenizer(), 0)
    assert encoded['action_positions'] == [4, 5, 6]
    assert encoded['row'][encoded['start']:] == encoded['answer_ids']
    ids = encoded['answer_ids']
    logits = torch.full((len(ids), 20), -10.)
    logits[range(len(ids)), ids] = 10.
    logits[0] = 0  # non-ACTION mismatch must not reject
    assert a.teacher_agrees(logits, ids, encoded['action_positions'])
    for position in encoded['action_positions']:
        wrong = logits.clone()
        wrong[position, 19] = 20.
        assert not a.teacher_agrees(wrong, ids, encoded['action_positions'])
    with pytest.raises(ValueError, match='align'):
        a.teacher_agrees(logits[:-1], ids, [4])
    with pytest.raises(ValueError, match='Empty'):
        a.teacher_agrees(logits, ids, [])


def test_policy_uniform_sampling_and_step_format_draws():
    entries = [dict(policy_id='large', id=i) for i in range(100)] + [dict(policy_id='small', id=101)]
    draws = a.sample_policies(entries, 10000, random.Random(0))
    assert 4700 < sum(e['policy_id'] == 'small' for e in draws) < 5300
    variants = [dict(policy_id='p', format=i) for i in range(6)]
    draws = a.sample_policies([dict(policy_id='p', variants=variants)], 6000, random.Random(1), augment=True)
    assert all(850 < sum(e['format'] == i for e in draws) < 1150 for i in range(6))
    with pytest.raises(ValueError, match='survived'):
        a.sample_policies([], 8, random.Random(0))


def test_probe_filter_uses_trained_ids_and_balances_smoke():
    corpus = dict(probes={split: dict(concept=[dict(policy_id=p, family=f, id=f'{p}{f}{i}')
                        for f in ('ticket', 'qa') for p in ('a', 'b') for i in range(8)])
                        for split in ('dev', 'test')})
    for split in ('dev', 'test'):
        full = a.filtered_probes(corpus, split, {'a'})
        smoke = a.filtered_probes(corpus, split, {'a'}, 10)
        assert len(full) == 16 and len(smoke) == 10
        assert all(p['policy_id'] == 'a' for p in smoke)
        assert sum(p['family'] == 'ticket' for p in smoke) == 5


def test_ce_answer_only_microbatch_gradient_and_token_mean():
    torch.manual_seed(3)
    logits = torch.randn(12, 12, requires_grad=True)
    entries = [dict(row=[1, 2, 3, 4], start=2), dict(row=[5, 6, 7], start=1)]
    expected = sum(torch.nn.functional.cross_entropy(
        logits[torch.tensor(e['row'][e['start'] - 1:-1])],
        torch.tensor(e['row'][e['start']:]), reduction='sum') for e in entries) / 4
    (.6 * expected).backward()
    gradient = logits.grad.clone()
    logits.grad = None
    value, nt = a.ce_backward(lambda x: logits[x], entries, 'cpu', 12, micro_tokens=4)
    assert nt == 4
    assert value == pytest.approx(float(expected.detach()), abs=1e-6)
    assert torch.allclose(logits.grad, gradient, atol=1e-6)
    assert not logits.grad[1].any()  # first prefix token has no CE
    assert not logits.grad[7].any()  # final answer token has no successor
    with pytest.raises(ValueError, match='target'):
        a.ce_backward(lambda x: logits[x], [], 'cpu', 12)


@pytest.mark.parametrize('arm,data', [(arm, data) for arm in ('lorafold', 'loramid', 'tmid')
                                    for data in ('d1', 'd2', 'v2')] +
                                   [(arm, 'd2') for arm in ('ladder_cont', 'ladder_box', 'ladder_tern')])
def test_training_weights_and_rehearsal_gradient(arm, data):
    torch.manual_seed(5)
    logits = torch.randn(12, 12, requires_grad=True)
    student = lambda x: logits[x]

    def entry(row, start):
        return dict(row=row, start=start,
                    record=a.TopKRecord.from_logits(torch.randn(len(row) - start, 12), 4))

    batch = [entry([1, 2, 3, 4], 2), entry([5, 6, 7], 1)]
    anchors = [entry([8, 9, 10], 1), entry([9, 10, 11], 1)]
    if data == 'd2':
        acq = sum(torch.nn.functional.cross_entropy(
            logits[torch.tensor(e['row'][e['start'] - 1:-1])],
            torch.tensor(e['row'][e['start']:]), reduction='sum') for e in batch) / 4
    else:
        acq, _ = a.c.loss_assembly(student, dict(acq=batch), 'cpu', 12, coefficients=(1., 0., 0.))
    if arm == 'lorafold':
        general = sum(torch.nn.functional.cross_entropy(
            logits[torch.tensor(e['row'][:-1])], torch.tensor(e['row'][1:])) for e in anchors) / 2
        expected = .8 * acq + .2 * general
        # CE rehearsal must work without a frozen teacher record.
        anchors = [dict(row=e['row'], start=1) for e in anchors]
    else:
        general, _ = a.c.loss_assembly(student, dict(gen=anchors), 'cpu', 12, coefficients=(0., 1., 0.))
        expected = .6 * acq + .4 * general
    expected.backward()
    gradient = logits.grad.clone()
    logits.grad = None
    losses = a.training_backward(student, batch, anchors, 'cpu', 12, arm, data)
    assert losses['loss'] == pytest.approx(float(expected.detach()), abs=1e-6)
    assert torch.allclose(logits.grad, gradient, atol=1e-6)
    if arm == 'lorafold':
        assert losses['gen'] == 0.
        assert losses['tokens']['gen'] == 0
        assert losses['rehearsal_ce'] == pytest.approx(float(general.detach()), abs=1e-6)
    else:
        assert 'rehearsal_ce' not in losses


def test_best_dev_selection():
    rows = [dict(step=0, dev={'acc': .2}, nnz=0), dict(step=2, dev={'acc': .4}, nnz=10),
            dict(step=4, dev={'acc': .4}, nnz=9), dict(step=6, dev={'acc': .4}, nnz=9)]
    assert a.best_dev_step(rows) == 4


@pytest.mark.parametrize('rules', ['fine', 'coarse'])
def test_large_dev_probe_contract_and_isolation(rules):
    corpus = build_corpus(rules=rules)
    original = copy.deepcopy(corpus)
    ids = {p['id'] for p in a.trained_policies(corpus, 1)}
    random.seed(19)
    rng_state = random.getstate()
    probes = large_dev_concepts(corpus, ids, seed=corpus['seed'])
    assert random.getstate() == rng_state
    assert probes == large_dev_concepts(corpus, reversed(sorted(ids)), seed=corpus['seed'])
    assert probes != large_dev_concepts(corpus, ids, seed=corpus['seed'] + 1)
    assert corpus == original == build_corpus(rules=rules)
    assert len(probes) == len({p['prompt'] for p in probes}) == 400
    assert sum(p['family'] == 'ticket' for p in probes) == 200
    assert {p['policy_id'] for p in probes} == ids
    assert all(set(NAME_POOLS['devlarge']).isdisjoint(pool)
               for name, pool in NAME_POOLS.items() if name != 'devlarge')
    training = a.build_cases(corpus, cases_per_policy=32) + a.build_cases(corpus, cases_per_policy=32, dedup=True)
    other = [p for split in ('dev', 'test') for p in corpus['probes'][split]['concept']]
    assert {p['case']['customer'] for p in probes}.isdisjoint(
        p['case']['customer'] for p in training + other)
    other_prompts = {r['prompt'] for c in training for r in c['formats']} | {p['prompt'] for p in other}
    assert {p['prompt'] for p in probes}.isdisjoint(other_prompts)
    policies = {p['id']: p for p in corpus['policies']}
    for probe in probes:
        policy = policies[probe['policy_id']]
        assert case_matches(policy, probe['case'])
        assert probe['case']['customer'] in NAME_POOLS['devlarge']
        siblings = [p for p in policies.values() if p['category'] == policy['category']]
        ticket = a.render_ticket(probe['case'])
        if probe['family'] == 'ticket':
            assert probe['prompt'] == ticket + '\nResolution:'
            option = lambda p: f"STATUS: {p['status']} | ACTION: {p['action']}"
        else:
            assert probe['prompt'] == f"Q: Under the user's policies, which action applies to {ticket}\nA:"
            option = lambda p: p['action']
        assert len(probe['options']) == 4
        assert set(probe['options']) == {option(p) for p in siblings}
        assert probe['options'][probe['answer_idx']] == option(policy)
        assert probe['source_episode'] in {ep['id'] for s in corpus['sessions'] for ep in s['episodes']}


@pytest.mark.parametrize('n,ids', [(-2, {'p000'}), (3, {'p000'}), (4, set()), (4, {'unknown'})])
def test_large_dev_invalid(corpus, n, ids):
    with pytest.raises(ValueError):
        large_dev_concepts(corpus, ids, n)


def test_large_dev_reproducible_across_process_hash_seeds():
    script = """
import hashlib, json
from make_user import build_corpus, large_dev_concepts
corpus = build_corpus(rules='coarse')
ids = {p['id'] for p in corpus['policies'] if p['intro_session'] <= 1}
items = large_dev_concepts(corpus, ids, seed=corpus['seed'])
print(hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest())
"""
    digests = [subprocess.check_output([sys.executable, '-c', script],
               cwd=Path(a.__file__).parent, env=dict(os.environ, PYTHONHASHSEED=seed), text=True)
               for seed in ('1', '2')]
    assert digests[0] == digests[1]


def test_smoothed_gate_full_windows_and_dnll():
    rows = [dict(dev={'acc': acc}, select_dnll=dnll) for acc, dnll in
            [(1., 0.), (.8, .01), (.9, .005001), (.7, .005), (.2, 0.)]]
    assert a.smoothed_acc(rows[:2]) is None
    assert a.smoothed_acc(rows) == pytest.approx(.6)
    assert a.best_smoothed_acc(rows) == pytest.approx(.8)
    assert a.gate_g2(rows, dev_large=True) == pytest.approx(.8)
    assert a.gate_g2(rows) == 1.
    assert a.best_smoothed_acc(rows[:3]) == 0.
    assert a.gate_g2([], dev_large=True) == 0.


def test_trainable_snapshot_cpu_clone_and_support(tmp_path):
    from test_consolidate import FSAParam
    modules = dict(ladder=a.LadderParam(torch.ones(1, 8), torch.ones(1, 8, dtype=torch.bool),
                                      'ladder_cont', FSAParam),
                   fold=a.epi_fold.FoldLoRAParam(torch.ones(1, 8), 1, torch.ones(1, 8, dtype=torch.bool)))
    modules['ladder'].register_buffer('support', torch.ones(1, 8, dtype=torch.bool))
    state = a.trainable_state(modules)
    assert set(state['ladder']) == {'M', 'mask', 'support'}
    assert set(state['fold']) == {'A', 'B'}
    assert all(t.device.type == 'cpu' and not t.requires_grad for ts in state.values() for t in ts.values())
    with torch.no_grad():
        modules['ladder'].M.fill_(1)
        modules['ladder'].support.zero_()
        modules['fold'].A.zero_()
    assert not state['ladder']['M'].any() and state['ladder']['support'].all()
    path = tmp_path / 'best_state.pt'
    torch.save(dict(step=25, state=state), path)
    saved = torch.load(path, weights_only=True)
    assert saved['step'] == 25
    for name, module in modules.items():
        module.load_state_dict(saved['state'][name], strict=False)
        for key, value in saved['state'][name].items():
            assert torch.equal(module.state_dict()[key], value)


@pytest.mark.parametrize('dev_large', [0, 20])
@pytest.mark.parametrize('arm,steps,init_mode', [('ladder_cont', 4, None), ('ladder_tern', 4, None),
                                               ('ladder_two', 4, None), ('ladder_tern', 0, 'rtn'),
                                               ('ladder_two', 0, 'gptq')])
def test_run_large_dev_smoothing_and_saved_best(corpus, tmp_path, monkeypatch, dev_large, arm, steps, init_mode):
    """Exercise the runner and serialized checkpoint without a GPU model."""
    from test_consolidate import FSAParam
    root, out = tmp_path / 'run', tmp_path / 'out'
    (root / 'genome').mkdir(parents=True)
    (root / 'corpus.json').write_text(a.json.dumps(corpus))
    (root / 'genome/metrics.json').write_text('{}')
    (root / 'genome/panel.pt').touch()
    (tmp_path / 'anchor-wikitext.json').write_text(a.json.dumps([1] * (150000 + 16 * 511)))
    model = torch.nn.Module()
    module = a.LadderParam(torch.ones(1, 8), torch.ones(1, 8, dtype=torch.bool), arm, FSAParam)
    lin = diagnostic_linear(torch.ones(1, 8), module)
    model.add_module('lin', lin)
    dnlls = iter([0., 0., 0., .006, .005, 0.])  # session start, then five evaluations
    fsa = SimpleNamespace(fsa_modules=lambda model: [], set_enabled=lambda enabled: None,
                          nll_windows=lambda *args: [next(dnlls)])
    gpu = SimpleNamespace(fsa=fsa, build=lambda: (model, 'cpu'), load_tokenizer=lambda: OffsetTokenizer(),
                          BOS=0, VOCAB=256, DATA=str(tmp_path), logits_of=lambda *args: lin(torch.ones(1, 8)),
                          topk_records=lambda model, rows, *args: [None] * len(rows))
    monkeypatch.setattr(a.c, 'gpu', lambda: gpu)
    monkeypatch.setattr(a.c, 'checkpoint_recurrences', lambda model: None)
    monkeypatch.setattr(a, 'attach_ladder', lambda *args: list(module.parameters()))
    monkeypatch.setattr(a, 'patched_linears', lambda *args: [(14, 0, 'up_proj', lin)])
    monkeypatch.setattr(a.epi_mask, 'metadata', lambda model: model._epi_write_set.copy())
    for method in ('empty_cache', 'reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    def backward(*args, **kwargs):
        sum(p.sum() for p in module.parameters()).backward()
        return dict(loss=0.)
    monkeypatch.setattr(a, 'training_backward', backward)
    accuracies = iter([.8, .8, .8, .1, .1, .5])
    evaluated = []
    def concept_acc(model, tok, dev, probes):
        evaluated.append(probes)
        return dict(acc=next(accuracies), n=len(probes))
    monkeypatch.setattr(a.evaluate, 'concept_acc', concept_acc)
    extra = []
    if init_mode:
        path = tmp_path / 'continuous.pt'
        torch.save(dict(config=dict(arm='ladder_cont', write_set='inpair', mask_sha256=None),
                        state={'14.0.up_proj': dict(M=torch.full((1, 8), .37), mask=module.mask)}), path)
        extra = ['--init-from', str(path), '--init-mode', init_mode]
    args = a.parse_args(['--run', str(root), '--out', str(out), '--arm', arm, '--data', 'd2',
                         '--steps', str(steps), '--eval-every', '1', '--cases-per-policy', '2',
                         '--dev-large', str(dev_large), '--save-best'] + extra)
    result = a.run(args)
    ids = set(result['config']['trained_policy_ids'])
    expected = (large_dev_concepts(corpus, ids, dev_large, corpus['seed']) if dev_large
                else a.filtered_probes(corpus, 'dev', ids))
    assert all(probes == expected for probes in evaluated[:-1])
    assert evaluated[-1] == a.filtered_probes(corpus, 'test', ids)
    if arm == 'ladder_two':
        committed = torch.load(out / 'committed_state.pt', weights_only=True)['14.0.up_proj']
        assert committed['code1'].dtype == committed['code2'].dtype == torch.int8
        assert torch.equal(committed['beta1'] * committed['code1'] + committed['beta2'] * committed['code2'],
                           module.delta_w())
    if steps == 0:
        assert len(result['trajectory']) == 1 and result['final']['step'] == 0
        assert result['init_mode'] == init_mode
        assert result['init_error']['relative_output_error'] is not None
        assert result['plane_nnz'] == ([0] if arm == 'ladder_tern' else [8, 8])
        assert result['beta_factors'] == ([1.] if arm == 'ladder_tern' else [.5, .125])
        serialized = a.json.loads((out / 'result.json').read_text())
        assert serialized['init_error'] == result['init_error']
        return
    assert result['smoothed'] == pytest.approx(1 / 3)
    assert result['best_smoothed'] == pytest.approx(1.7 / 3)
    for row in result['trajectory']:
        assert row['dev']['n'] == len(expected)
        acc = row['dev']['acc']
        assert row['dev']['se'] == pytest.approx(math.sqrt(acc * (1 - acc) / len(expected)))
    saved = torch.load(out / 'best_state.pt', weights_only=True)
    assert saved['step'] == result['best_state_step'] == 2
    assert saved['smoothed'] == pytest.approx(.8)
    master_name = 'M1' if arm == 'ladder_two' else 'M'
    master = saved['state']['14.0.up_proj'][master_name]
    assert torch.allclose(master, torch.full_like(master, -2 * a.effective_lr(args)))
    assert not torch.equal(master, getattr(module, master_name))
    if arm == 'ladder_two':
        assert set(saved['committed_state']['14.0.up_proj']) == {'code1', 'code2'}
    serialized = a.json.loads((out / 'result.json').read_text())
    assert serialized['best_smoothed'] == result['best_smoothed']


def test_cli_defaults_and_smoke():
    base = ['--run', 'out/R', '--out', 'out/A', '--arm', 'loramid', '--data', 'd1']
    args = a.parse_args(base)
    assert args.dev_large == 0 and not args.save_best
    large = a.parse_args(base + ['--dev-large', '400', '--save-best', '--smoke'])
    assert large.dev_large == 400 and large.save_best
    assert (args.steps, args.eval_every, args.cases_per_policy, args.session, args.lr_mult) == (150, 25, 16, 1, 1.)
    args = a.parse_args(base + ['--smoke'])
    assert (args.steps, args.eval_every, args.cases_per_policy) == (4, 2, 4)


@pytest.mark.parametrize('extra', [['--steps', '0'], ['--eval-every', '-1'], ['--cases-per-policy', '3'],
                                   ['--session', '2'], ['--lr-mult', 'nan'], ['--lr-mult', '0'],
                                   ['--dev-large', '-2'], ['--dev-large', '3']])
def test_cli_invalid(extra):
    with pytest.raises(SystemExit):
        a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'tmid', '--data', 'd2'] + extra)


def test_v2_cases_and_probe_contract(corpus):
    source_only = {k: copy.deepcopy(corpus[k]) for k in ('policies', 'sessions')}
    cases = a.build_cases(source_only, cases_per_policy=32, dedup=True)
    assert cases == a.build_cases(source_only, cases_per_policy=32, dedup=True)
    assert len(cases) == 32 * len(a.trained_policies(corpus, 1))
    assert len({a.json.dumps(c['case'], sort_keys=True) for c in cases}) == len(cases)
    for policy in a.trained_policies(corpus, 1):
        own = [c for c in cases if c['policy_id'] == policy['id']]
        assert sum(c['base_format'] == 0 for c in own) == 16
        for case in own:
            assert case_matches(policy, case['case'])
            assert case['case']['customer'] in SELFSTUDY_POOL
            probe = a.v2_probe(case, list(reversed(corpus['policies'])))
            siblings = sorted((p for p in corpus['policies'] if p['category'] == policy['category']),
                              key=lambda p: p['id'])
            assert probe['context'].split('\n') == [
                f"At Fennmark Parcel Desk we use {p['codename']}. {p['rule_text']} {p['rationale_text']}"
                for p in siblings]
            assert len(probe['options']) == 4
            ticket = a.render_ticket(case['case'])
            if probe['family'] == 'ticket':
                assert probe['prompt'] == ticket + '\nResolution:'
                assert probe['options'] == [f"STATUS: {p['status']} | ACTION: {p['action']}" for p in siblings]
                assert probe['options'][probe['answer_idx']] == f"STATUS: {policy['status']} | ACTION: {policy['action']}"
            else:
                assert probe['prompt'] == f"Q: Under the user's policies, which action applies to {ticket}\nA:"
                assert probe['options'] == [p['action'] for p in siblings]
                assert probe['options'][probe['answer_idx']] == policy['action']
            assert policy['codename'] not in probe['prompt']


def test_v2_dedup_retries_identical_cases_only(corpus, monkeypatch):
    case = dict(a.build_cases(corpus, cases_per_policy=2)[0]['case'])
    small = copy.deepcopy(corpus)
    policy = small['policies'][0]
    small['policies'] = [policy]
    cond = policy['condition']
    if cond['kind'] == 'range':
        cond['max'] = cond['min']
        case[policy['attribute']] = cond['min']
    else:
        cond['values'] = cond['values'][:1]
        case[policy['attribute']] = cond['values'][0]
    different = dict(case, ticket_id=case['ticket_id'] + 'X')
    draws = iter([case, case, different])
    monkeypatch.setattr(a, 'mint_case', lambda *args, **kwargs: dict(next(draws)))
    cases = a.build_cases(small, cases_per_policy=2, dedup=True)
    assert [c['case'] for c in cases] == [case, different]
    assert [c['base_format'] for c in cases] == [0, 1]


@pytest.mark.parametrize('family_index', [0, 1])
def test_v2_separate_encoding_and_true_target_alignment(corpus, family_index):
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=False):
            self.calls.append(text)
            return SimpleNamespace(ids=[ord(c) for c in text])

    case = a.build_cases(corpus, cases_per_policy=2)[family_index]
    probe = a.v2_probe(case, corpus['policies'])
    probe['prompt'] += '  \n'
    tok = RecordingTokenizer()
    item, prefix = a.v2_rows(case, probe, tok, 0)
    answer = ' ' + probe['options'][probe['answer_idx']]
    assert tok.calls == [probe['prompt'].rstrip(), answer, probe['context'], '\n\n']
    assert prefix == [0] + list(map(ord, probe['context'] + '\n\n' + probe['prompt'].rstrip()))
    assert item['row'] == [0] + list(map(ord, probe['prompt'].rstrip() + answer))
    assert item['row'][item['start']:] == item['answer_ids']
    teacher_row = prefix + item['answer_ids']
    assert teacher_row[len(prefix):] == item['row'][item['start']:]
    assert '| NOTE:' not in answer


def test_v2_cloze_gate_and_keep_stats():
    assert a.cloze_correct([-5., -2., -3., -4.], 1)
    assert not a.cloze_correct([-5., -2., -3., -4.], 2)
    assert a.cloze_correct([-2.] * 4, 0)  # concept_acc's first-option tie rule
    hits = {'ticket': [True] * 4, 'qa': [True] * 4 + [False] * 2}
    stats = a.g1_stats(hits)
    assert (stats['attempted'], stats['kept'], stats['keep_rate']) == (10, 8, .8)
    assert stats['by_family']['ticket']['keep_rate'] == 1.
    assert stats['by_family']['qa']['keep_rate'] == pytest.approx(2 / 3)
    assert a.g1_stats({'ticket': [], 'qa': []})['keep_rate'] == 0.


def test_v2_g2_uses_only_evaluations_under_dnll_limit():
    rows = [dict(dev={'acc': .9}, select_dnll=.005001),
            dict(dev={'acc': .7}, select_dnll=.005), dict(dev={'acc': .6}, select_dnll=-.01)]
    assert a.gate_g2(rows) == .7
    assert a.gate_g2(rows[:1]) == 0.
    assert a.gate_g2([]) == 0.


def test_v2_cli_and_lr_defaults():
    base = ['--run', 'r', '--out', 'o', '--arm', 'tmid', '--data', 'v2']
    args = a.parse_args(base)
    assert (args.steps, args.eval_every, args.cases_per_policy, args.session) == (150, 25, 32, 1)
    assert a.effective_lr(args) == .02
    args = a.parse_args(base + ['--g1-only', '--lr-mult', '2.5', '--cases-per-policy', '8'])
    assert args.g1_only and args.cases_per_policy == 8
    assert a.effective_lr(args) == .05
    assert a.parse_args(base + ['--smoke']).cases_per_policy == 4
    assert a.effective_lr(a.parse_args(base + ['--arm', 'loramid'])) == .001
    with pytest.raises(SystemExit):
        a.parse_args(base + ['--data', 'd1', '--g1-only'])


@pytest.mark.parametrize('passed,expected', [(True, 0), (False, 3)])
def test_g1_exit_status(passed, expected, monkeypatch):
    monkeypatch.setattr(a, 'parse_args', lambda: None)
    monkeypatch.setattr(a, 'run', lambda args: dict(g1_pass=passed))
    assert a.main() == expected


@pytest.mark.parametrize('arm', ['ladder_cont', 'ladder_box', 'ladder_tern'])
def test_ladder_forward_gradient_and_disable(arm):
    from test_consolidate import FSAParam
    W = torch.tensor([[2., 0, 2, 0, 0, 0, 0, 0]], dtype=torch.bfloat16)
    mask = torch.tensor([[False, True, False, True, True, True, True, True]])
    values = torch.tensor([[9., -2., 9., -.5, .5, .51, -.51, .2]])
    m = a.LadderParam(W, mask, arm, FSAParam)
    with torch.no_grad():
        m.M.copy_(values)
    assert m.M.dtype == torch.float32 and m.M.shape == W.shape
    expected = values if arm == 'ladder_cont' else (values.clamp(-1, 1) if arm == 'ladder_box'
                                                   else values.sign() * (values.abs() > .5))
    assert torch.equal(m.delta_w(), expected * mask * 2)
    m.delta_w().sum().backward()
    grad_mask = mask & (values.abs() <= 1) if arm == 'ladder_box' else mask
    assert torch.equal(m.M.grad, grad_mask * 2.)
    assert torch.equal(m(W), W + (expected * mask * 2).to(W.dtype))
    assert torch.equal(m(W)[~mask], W[~mask])
    FSAParam.enabled = False
    try:
        assert torch.equal(m(W), W)
    finally:
        FSAParam.enabled = True


@pytest.mark.parametrize('arm,lr', [('ladder_cont', .001), ('ladder_box', .001), ('ladder_tern', .02)])
def test_ladder_cli_and_rates(arm, lr):
    argv = ['--run', 'r', '--out', 'o', '--arm', arm, '--data', 'd2']
    args = a.parse_args(argv)
    assert not args.diag
    assert args.layers == [14, 15] and args.projs == ['up_proj', 'down_proj']
    assert a.effective_lr(args) == lr
    args = a.parse_args(argv + ['--layers', 'all', '--projs', 'down', '--diag', '--lr', '.003'])
    assert args.layers == list(range(32)) and args.projs == ['down_proj'] and args.diag
    assert a.effective_lr(args) == (.02 if arm == 'ladder_tern' else .003)


@pytest.mark.parametrize('extra', [['--layers', 'all'], ['--layers', '32'], ['--layers', '-1'],
                                   ['--layers', '14,14'], ['--layers', 'foo'], ['--lr', 'nan'],
                                   ['--lr', '0'], ['--data', 'd1']])
def test_ladder_cli_invalid(extra):
    with pytest.raises(SystemExit):
        a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'ladder_cont', '--data', 'd2'] + extra)


def diagnostic_linear(W, m):
    lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=W.dtype)
    lin.weight = torch.nn.Parameter(W.clone(), requires_grad=False)
    torch.nn.utils.parametrize.register_parametrization(lin, 'weight', m)
    return lin


def test_diagnostics_actual_delta_aggregation_gradients_and_failure():
    from test_consolidate import FSAParam
    W = torch.tensor([[2., 0, 2, 0, 0, 0, 0, 0]], dtype=torch.bfloat16)
    mask = W == 0
    m = a.LadderParam(W, mask, 'ladder_cont', FSAParam)
    lin = diagnostic_linear(W, m)
    row = a.slot_diagnostics([(14, 0, 'up_proj', lin)], 0, 'eval')
    assert row['grad_rms'] == 0 and row['grad_frac_on_mask'] == 1
    assert row['per_layer'][14]['up']['frac_on_mask'] == 1
    with torch.no_grad():
        m.M[0, 1], m.M[0, 3] = .25, -.5
    m.delta_w().sum().backward()
    row = a.slot_diagnostics([(14, 0, 'up_proj', lin), (14, 1, 'up_proj', lin)], 1, 'eval')
    s = row['per_layer'][14]['up']
    assert s['nnz'] == 4 and s['absmax'] == 1
    assert s['rms_delta'] == pytest.approx(math.sqrt(1.25 / 8))
    assert s['frac_on_mask'] == 1 and row['mask_check'] == row['sign_check'] == 'PASS'
    assert row['grad_rms'] == pytest.approx(math.sqrt(24 / 8))
    assert row['grad_frac_on_mask'] == 1
    # An erroneously served write outside the claimed mask must be visible.
    m.M.data[0, 0] = 1
    m.delta_w = lambda: m.alpha * m.M
    row = a.slot_diagnostics([(14, 0, 'up_proj', lin)], 2, 'eval')
    assert row['mask_check'] == 'FAIL'
    assert row['per_layer'][14]['up']['frac_on_mask'] == pytest.approx(1.5 / 3.5)
    a.json.dumps(row, allow_nan=False)
    m.forward = lambda W: W - m.delta_w().to(W.dtype)
    assert a.slot_diagnostics([(14, 0, 'up_proj', lin)], 3, 'eval')['sign_check'] == 'FAIL'


def test_fold_diagnostics_ba_codes_and_factor_gradients():
    W = torch.tensor([[2., 0, 2, 0, 0, 0, 0, 0]])
    m = a.epi_fold.FoldLoRAParam(W, 1, W == 0)
    lin = diagnostic_linear(W, m)
    with torch.no_grad():
        m.A.copy_(torch.tensor([[1., 2., 1., -3., 0., 0., 0., 0.]]))
    row = a.slot_diagnostics([(15, 0, 'down_proj', lin)], 0, 'eval')
    assert row['per_layer'][15]['down']['ba_frac_on_mask'] == 1
    lin.weight.sum().backward()
    s = a.slot_diagnostics([(15, 0, 'down_proj', lin)], 1, 'first_optimizer_step')['per_layer'][15]['down']
    # BA starts at zero: only B receives a nonzero gradient.
    assert not s['A_grad_nonzero'] and s['B_grad_nonzero']
    with torch.no_grad():
        m.B.fill_(1)
    m.zero_grad()
    lin.weight.sum().backward()
    row = a.slot_diagnostics([(15, 0, 'down_proj', lin)], 2, 'eval')
    s = row['per_layer'][15]['down']
    assert s['ba_frac_on_mask'] == pytest.approx(5 / 7)
    assert s['code_distribution'] == {'-1': 1, '0': 4, '+1': 1}
    assert s['A_grad_nonzero'] and s['B_grad_nonzero']
    assert row['grad_coordinates'] == 'dBA' and row['grad_frac_on_mask'] == pytest.approx(9 / 11)
    assert s['frac_on_mask'] == 1 and row['mask_check'] == row['sign_check'] == 'PASS'


@pytest.mark.parametrize('arm,beta', [('ladder_tern', 1.), ('ladder_tern', .125), ('ladder_two', 1.)])
def test_step_ladder_values_gradients_revoke_and_accounting(arm, beta):
    from test_consolidate import FSAParam
    W = torch.tensor([[2., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.bfloat16)
    m = a.LadderParam(W, W == 0, arm, FSAParam, beta)
    lin = diagnostic_linear(W, m)
    masters = (m.M1, m.M2) if arm == 'ladder_two' else (m.M,)
    with torch.no_grad():
        masters[0].copy_(torch.tensor([[8., -1., -.5, .5, .51, -.51, .2, 1.]]))
        if arm == 'ladder_two':
            masters[1].copy_(torch.tensor([[8., 1., 1., 0., 1., -1., 1., 0.]]))
    expected = sum(f * a.ternary_round(p.detach()) * m.mask
                   for f, p in zip(m.beta_factors, masters)) * 2
    assert torch.equal(m.delta_w(), expected)
    assert torch.equal(lin.weight, W + expected.to(W.dtype))
    assert torch.equal(lin.weight[~m.mask], W[~m.mask])
    m.delta_w().sum().backward()
    for f, master in zip(m.beta_factors, masters):
        assert torch.equal(master.grad, m.mask * (2 * f))
    stats = a.plane_stats({'m': m})
    assert stats['plane_nnz'] == ([4, 5] if arm == 'ladder_two' else [4])
    used = int(torch.stack([c != 0 for c in m.codes()]).any(0).sum())
    assert stats['combined_bits'] == pytest.approx(used * len(masters) * math.log2(3))
    assert a.slot_diagnostics([(14, 0, 'up_proj', lin)], 0, 'eval')['mask_check'] == 'PASS'
    with torch.no_grad():
        for master in masters:
            master.zero_()
    assert torch.equal(lin.weight, W)
    assert not m.delta_w().any()


@pytest.mark.parametrize('factors,expected', [
    ((1.,), [[0, -1, 0, 0, 0, 0, 0, 1, 1]]),
    ((.25,), [[0, -1, -1, 0, 0, 0, 1, 1, 1]]),
    ((.5, .125), [[0, -1, -1, 0, 0, 0, 1, 1, 1], [0, -1, 1, 0, 0, 0, -1, 1, 1]])])
def test_rtn_grid_rounding(factors, expected):
    v = torch.tensor([[-1.7, -.57, -.3, -.0625, 0., .0625, .3, .57, 1.7]])
    mask = torch.ones_like(v, dtype=torch.bool)
    mask[0, 0] = False
    codes = a.initialize_codes(v, mask, factors)
    for code, plane in zip(codes, expected):
        assert torch.equal(code, torch.tensor([plane], dtype=code.dtype))
        assert set(code.flatten().tolist()) <= {-1., 0., 1.}


@pytest.mark.parametrize('factors', [(1.,), (.5, .125)])
def test_gptq_reduces_heldout_linear_output_error(factors):
    rng = torch.Generator().manual_seed(14)
    # Random inputs with correlated columns make feedback consequential.
    mixing = torch.randn(8, 8, generator=rng)
    train = torch.randn(4096, 8, generator=rng) @ mixing
    heldout = torch.randn(4096, 8, generator=rng) @ mixing
    values = torch.randn(32, 8, generator=rng) * .4
    mask = torch.rand(32, 8, generator=rng) > .25
    values *= mask
    def represented(codes):
        return sum(f * q for f, q in zip(factors, codes))
    rtn = a.initialize_codes(values, mask, factors)
    gptq = a.initialize_codes(values, mask, factors, train.T @ train)
    target = heldout @ values.T
    rtn_error = (heldout @ represented(rtn).T - target).square().sum()
    gptq_error = (heldout @ represented(gptq).T - target).square().sum()
    assert gptq_error < rtn_error
    assert all(not code[~mask].any() for code in gptq)


@pytest.mark.parametrize('arm', ['ladder_tern', 'ladder_two'])
@pytest.mark.parametrize('mode', ['rtn', 'gptq'])
def test_continuous_checkpoint_initialization_hooks_and_restoration(tmp_path, arm, mode):
    from test_consolidate import FSAParam
    W = torch.tensor([[2., 0., 0., 0., 0., 0., 0., 0.]])
    module = a.LadderParam(W, W == 0, arm, FSAParam, .25)
    model = torch.nn.Sequential(diagnostic_linear(W, module))
    name = '14.0.up_proj'
    v = torch.tensor([[0., .37, -.57, .12, .78, -.25, .05, .22]])
    path = tmp_path / 'best_state.pt'
    torch.save(dict(config=dict(arm='ladder_cont', write_set='forced', mask_sha256='hash'),
                    state={name: dict(M=v, mask=module.mask)}), path)
    args = SimpleNamespace(init_from=path, init_mode=mode, seed=3)
    masters = tuple(module.parameters())
    batches = [dict(row=[i] * 8) for i in range(1, 25)]
    seen = []
    def student(tokens):
        seen.append(tokens[0, 0].item())
        assert module.arm == 'ladder_cont'
        return model(tokens.float())
    stats = a.initialize_ladder(model, {name: module}, [(14, 0, 'up_proj', model[0])],
                                args, batches, student, 'cpu',
                                dict(write_set='forced', mask_sha256='hash'))
    assert module.arm == arm and tuple(module.parameters()) == masters
    assert len(seen) == 20 and set(seen[:16]).isdisjoint(seen[16:])
    assert not model[0]._forward_hooks
    assert stats['relative_output_error'] is not None
    assert math.isfinite(stats['relative_output_error'])
    error = float((module.delta_w().detach() - v * module.alpha).sum().abs())
    assert stats['relative_output_error'] == pytest.approx(error / float((W + v * module.alpha).sum()), rel=1e-5)
    assert stats['relative_patch_output_error'] == pytest.approx(error / float((v * module.alpha).sum()), rel=1e-5)
    assert all(set(p.detach().flatten().tolist()) <= {-1., 0., 1.} for p in masters)
    if mode == 'rtn':
        assert all(torch.equal(p, q) for p, q in zip(masters, a.initialize_codes(v, module.mask, module.beta_factors)))
    before = module.delta_w().clone()
    with pytest.raises(ValueError, match='mask mismatch'):
        a.initialize_ladder(model, {name: module}, [], args, batches, student, 'cpu',
                            dict(write_set='forced', mask_sha256='different'))
    assert torch.equal(module.delta_w(), before)


@pytest.mark.parametrize('arm', ['ladder_tern', 'ladder_two'])
def test_step_ladder_cli_zero_steps_and_defaults(arm):
    argv = ['--run', 'r', '--out', 'o', '--arm', arm, '--data', 'd2']
    args = a.parse_args(argv)
    assert args.patch_beta == 1. and args.init_from is None
    assert a.effective_lr(args) == .02
    args = a.parse_args(argv + ['--steps', '0', '--init-from', 'best_state.pt', '--init-mode', 'gptq'])
    assert args.steps == 0 and args.init_mode == 'gptq'


@pytest.mark.parametrize('extra', [['--patch-beta', '0'], ['--patch-beta', 'nan'],
                                  ['--patch-beta', '-.5'], ['--steps', '-1'],
                                  ['--arm', 'ladder_two', '--patch-beta', '.5'],
                                  ['--arm', 'ladder_cont', '--init-from', 'best_state.pt']])
def test_step_ladder_invalid_cli(extra):
    with pytest.raises(SystemExit):
        a.parse_args(['--run', 'r', '--out', 'o', '--arm', 'ladder_tern', '--data', 'd2'] + extra)



def test_ladder_recipe_local_penalty_has_replay_protocol():
    cls = a.reference_local_penalty()
    assert all(hasattr(cls, m) for m in ('start', 'stop', 'end'))
