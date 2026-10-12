"""CPU contracts for the secondary MDL metric and release."""
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

import consolidate as c
import evaluate as e
from make_user import NAME_POOLS, HELDOUT_POOL, build_corpus, heldout_text, heldout_sample
from selfstudy import build_items
from test_consolidate import patch


@pytest.mark.parametrize('rules', ['fine', 'coarse'])
def test_heldout_rendering(rules):
    corpus = build_corpus(rules=rules)
    before = copy.deepcopy(corpus)
    texts = heldout_text(corpus, 2, 91)
    assert texts == heldout_text(corpus, 2, 91)
    assert texts != heldout_text(corpus, 2, 92)
    assert corpus == before
    assert 'heldout' not in corpus
    source = corpus['sessions'][1]['episodes']
    assert len(texts) == len(source)
    for ep, text in zip(source, texts):
        for claim in ep['claims']:
            if ep['kind'] not in ('policy_intro', 'policy_recur'):
                assert claim['value'] in text
        if ep.get('policy_id'):
            policy = next(p for p in corpus['policies'] if p['id'] == ep['policy_id'])
            assert policy['action'] in text and policy['category'] in text
        if 'customer ' in text:
            assert any(name in text for name in HELDOUT_POOL)
            assert 'Ticket MDL-91-2-' in text
            assert 'Ticket FP-' not in text
    other = json.dumps(corpus) + json.dumps(build_items(corpus, 2))
    assert all(name not in other for name in HELDOUT_POOL)
    assert all(set(HELDOUT_POOL).isdisjoint(pool) for key, pool in NAME_POOLS.items() if key != 'heldout')
    for key in ('worked', 'selfstudy', 'dev', 'test'):
        assert all(name not in '\n'.join(texts) for name in NAME_POOLS[key])


@pytest.mark.parametrize('nnz,reserve,scales', [(0, 2, 0), (3, 1024, 7), (100, 19, 64)])
def test_group_storage_bits(nnz, reserve, scales):
    assert c.mdl_group_bits(nnz, reserve, scales) == pytest.approx(
        nnz * (math.log2(reserve) + math.log2(3)) + 32 * scales)


def test_mdl_accounting():
    dnll = dict(mean=.001, ci95_lo=-.002, ci95_hi=.003)
    m = e.mdl_bits([10., 20.], [8., 21.], dnll, 4)
    assert m['bits_saved'] == pytest.approx(1 / math.log(2))
    assert m['patch_bits'] == 32
    assert m['N_ref'] == 1000000
    assert m['general_cost'] == pytest.approx({k: v * 1e6 / math.log(2) for k, v in dnll.items()})
    assert m['net'] == pytest.approx(m['bits_saved'] - m['general_cost']['mean'] - 32)
    with pytest.raises(ValueError):
        e.mdl_bits([1], [], dnll, 0)


@pytest.mark.parametrize('name,row,key', [('14.3.up_proj', 63, (14, 3, 'up_proj', 0)),
                                         ('15.2.down_proj', 64, (15, 2, 'down_proj', 1)),
                                         ('15.2.up_proj', 128, (15, 2, 'up_proj', 2))])
def test_group_keys(name, row, key):
    assert c.mdl_group_key(name, row) == key


@pytest.mark.parametrize('saved,storage,conflicts,release', [(1, 2, (), True), (2, 2, (), False),
    (3, 2, (), False), (-1, 0, (), True), (1, 2, [(4, 3)], False), (1, 2, [(3, 4)], True),
    (1, 2, [(3, 3)], False)])
def test_release_rule(saved, storage, conflicts, release):
    assert c.mdl_release_rule(saved, storage, conflicts) is release


def helpers(monkeypatch):
    def logits(model, x):
        return model(x)
    monkeypatch.setattr(e, 'gpu', lambda: SimpleNamespace(BOS=0, VOCAB=3, enc=lambda tok, t: [int(v) for v in t], logits_of=logits))
    monkeypatch.setattr(c, 'gpu', e.gpu)


def test_heldout_nll_and_backward(monkeypatch):
    helpers(monkeypatch)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0., 1., 2.]))
            self.batches = []
        def forward(self, x):
            self.batches.append(tuple(x.shape))
            return self.bias.expand(*x.shape, 3)
    model = Model()
    texts = ['12', '', '221']
    nll = e.heldout_nll(model, None, 'cpu', texts)
    lp = model.bias.detach().log_softmax(0)
    assert nll == pytest.approx([float(-lp[1]-lp[2]), 0, float(-2*lp[2]-lp[1])])
    assert model.bias.grad is None
    assert model.batches == [(2, 4)]
    c.mdl_backward(model, None, 'cpu', texts)
    assert model.bias.grad is not None
    assert float(model.bias.grad.sum()) == pytest.approx(0, abs=1e-6)


def test_group_candidates_and_leaveout_restore(monkeypatch):
    m = patch()
    modules = {'14.0.up_proj': m}
    assert c.mdl_groups(modules) == [((14, 0, 'up_proj', 0), '14.0.up_proj', 0, 2)]
    mask = m.delta().detach() != 0
    before = m.M.detach().clone()
    monkeypatch.setattr(e, 'heldout_nll', lambda *args: [float((m.M.detach() == 0).sum())])
    gain = c.mdl_leaveout(None, None, 'cpu', ['x'], m, mask, 14)
    assert gain == pytest.approx(2 / math.log(2))
    assert torch.equal(m.M, before)
    def fail(*args):
        raise RuntimeError('scoring failed')
    monkeypatch.setattr(e, 'heldout_nll', fail)
    with pytest.raises(RuntimeError):
        c.mdl_leaveout(None, None, 'cpu', ['x'], m, mask, 0)
    assert torch.equal(m.M, before)


def test_mdl_cache_once_per_session(monkeypatch, tmp_path):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_text', lambda corpus, k, seed: [str(k)])
    enabled = [True]
    fsa = SimpleNamespace(set_enabled=lambda value: enabled.__setitem__(0, value))
    monkeypatch.setattr(e, 'gpu', lambda: SimpleNamespace(fsa=fsa))
    calls = []
    def nll(*args):
        calls.append((enabled[0], args[-1]))
        return [1 if enabled[0] else 3 for t in args[-1]]
    monkeypatch.setattr(e, 'heldout_nll', nll)
    panel = {'dnll': dict(mean=0, ci95_lo=0, ci95_hi=0)}
    for session in (1, 2, 2):
        metric = e.mdl_metrics(None, None, 'cpu', {'seed': 1}, tmp_path, session, panel, {'bytes': 0})
        assert metric['bits_saved'] == pytest.approx(2*session / math.log(2))
    assert [text for active, text in calls if not active] == [['1'], ['2']]
    assert json.loads((tmp_path / 'genome/heldout_nll.json').read_text()) == {'1': [3], '2': [3]}


@pytest.mark.parametrize('limit', [0, 1, 2])
def test_release_exact_limit_and_clearing(monkeypatch, limit):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_text', lambda *args: ['x'])
    modules = {f'14.{i}.up_proj': patch() for i in range(2)}
    frozen = {name: m.support.clone() for name, m in modules.items()}
    before = {name: m.M.detach().clone() for name, m in modules.items()}
    model = torch.nn.ModuleList(list(modules.values()))
    def backward(*args):
        for i, m in enumerate(modules.values()):
            m.M.grad = torch.ones_like(m.M) * (i + 1)
    monkeypatch.setattr(c, 'mdl_backward', backward)
    monkeypatch.setattr(e, 'heldout_nll', lambda *args: [1.])
    result = c.mdl_release(model, None, 'cpu', modules, frozen, {'seed': 1}, {},
                           SimpleNamespace(session=2, mdl_max_groups=limit))
    assert result['evaluated'] == limit
    assert result['released'] == limit
    assert result['released_slots'] == 2 * limit
    for i, (name, m) in enumerate(modules.items()):
        assert m.M.grad is None
        if i < limit:
            assert not m.M.any() and not m.support.any() and not frozen[name].any()
            assert torch.isneginf(m.logb).all()
        else:
            assert torch.equal(m.M, before[name])


@pytest.mark.parametrize('replaced', [False, True])
@pytest.mark.parametrize('old_gain,new_gain', [(5., 8.), (8., 5.), (5., 5.)])
def test_cross_group_conflict_and_recurrence(monkeypatch, replaced, old_gain, new_gain):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_text', lambda *args: ['x'])
    modules = {f'14.{i}.up_proj': patch() for i in range(2)}
    model = torch.nn.ModuleList(list(modules.values()))
    frozen = {name: m.support.clone() for name, m in modules.items()}
    monkeypatch.setattr(c, 'mdl_backward', lambda *args: None)
    monkeypatch.setattr(e, 'heldout_nll', lambda *args: [0.])
    def leaveout(model, tok, dev, texts, module, mask, baseline, micro):
        return (old_gain if module is modules['14.0.up_proj'] else new_gain) if texts == ['x'] else 0.
    monkeypatch.setattr(c, 'mdl_leaveout', leaveout)
    provenance = {f'14:{i}:up_proj:0': {'p00': dict(version=i+1 if replaced else 1, session=i+1)} for i in range(2)}
    corpus = dict(seed=1, replacements=[dict(concept='p00', old_version=1, new_version=2, session=2)] if replaced else [])
    result = c.mdl_release(model, None, 'cpu', modules, frozen, corpus,
                           {'mdl_concepts': provenance}, SimpleNamespace(session=3, mdl_max_groups=256))
    assert bool(modules['14.0.up_proj'].M.any()) == (replaced and old_gain >= new_gain)
    assert bool(modules['14.1.up_proj'].M.any()) == (replaced and new_gain >= old_gain)
    assert bool(result['groups'][0]['conflicts']) == replaced


@pytest.mark.parametrize('majority', ['p00', 'f000', None])
def test_birth_training_gradient_majority(monkeypatch, majority):
    m = patch()
    name = '14.0.up_proj'
    model = torch.nn.ModuleList([m])
    state = c.snapshot({name: m}, {name: m.support.clone()})
    state[name]['M'].zero_()
    evidence = c.MDLGradientEvidence({name: m})
    for concept, mass in [('p00', 3 if majority == 'p00' else 1),
                          ('f000', 3 if majority == 'f000' else 1)]:
        evidence.concept = concept
        (m.M.sum() * mass).backward()
    evidence.concept = None
    model.zero_grad(set_to_none=True)
    corpus = dict(policies=[dict(id='p00', version=2)], facts=[dict(id='f000')])
    result = c.mdl_birth_records(model, None, 'cpu', {name: m}, state, corpus, 2, {}, evidence, 5)
    evidence.close()
    expected = {} if majority is None else {majority: dict(version=2 if majority == 'p00' else 1, session=2)}
    assert result == {'14:0:up_proj:0': expected}
    assert m.M.grad is None


def test_summary_mdl_comparisons(tmp_path, capsys):
    baseline = dict(dev={'concept': {'acc': 0}}, test={'concept': {'acc': 0}, 'lure': {'em': 0}},
                    panel={'dnll': {'mean': 0}, 'agreement': 1})
    for arm in ('genome', 'icl'):
        e.write_json(tmp_path / arm / 'metrics.json', baseline)
    for arm, net in [('tmid', 10), ('tmid_mdl', 30), ('tmid_util01', 20), ('tmid_util03', 15)]:
        metric = dict(baseline, session=1, slots={'nnz': 2}, mdl={'net': net})
        e.write_json(tmp_path / arm / 's1/metrics.json', metric)
    result = e.summarize(tmp_path)
    assert result['arms']['tmid_mdl']['secondary_exploratory']
    assert 'criteria' not in result['arms']['tmid_mdl']
    assert set(result['mdl_comparisons']) == {'tmid', 'tmid_util01', 'tmid_util03'}
    assert result['mdl_comparisons']['tmid']['arm']['net_mdl'] == {'1': 30}
    assert result['mdl_comparisons']['tmid']['tmid']['net_mdl'] == {'1': 10}
    output = capsys.readouterr().out
    assert 'net MDL bits' in output and 'tmid_mdl_comparison' in output


@pytest.mark.parametrize('rules', ['fine', 'coarse'])
def test_heldout_disjoint_from_all_training_contexts_and_probes(rules):
    corpus = build_corpus(rules=rules)
    sources = [ep['text'] for session in corpus['sessions'] for ep in session['episodes']]
    sources += [chunk['text'] for chunk in corpus['chunks']]
    for session in range(1, 6):
        sources += [item['prompt'] for item in build_items(corpus, session)]
    sources += [probe['prompt'] for split in corpus['probes'].values() for probes in split.values() for probe in probes]
    grams = {source[i:i+40] for source in sources for i in range(len(source)-39)}
    for session in range(1, 6):
        for text in heldout_text(corpus, session, 91):
            assert text not in sources
            assert not any(text[i:i+40] in grams for i in range(len(text)-39))


def test_fixed_sixteen_text_sample():
    corpus = build_corpus(rules='coarse')
    assert heldout_sample(corpus, 1, 91) == heldout_sample(corpus, 1, 91)
    assert len(heldout_sample(corpus, 1, 91)) == 16
    assert heldout_sample(corpus, 1, 91) != heldout_sample(corpus, 1, 92)
    assert set(heldout_sample(corpus, 1, 91)) <= set(heldout_text(corpus, 1, 91))


def test_bounded_document_rows_no_attention_leak(monkeypatch):
    helpers(monkeypatch)
    class Causal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.2))
            self.sizes = []
        def forward(self, x):
            self.sizes.append(x.numel())
            prefix = x.cumsum(1).float() * self.weight
            return torch.stack((prefix, -prefix, prefix * 0), -1)
    model = Causal()
    texts = ['12', '221', '1' * 20, '']
    packed = e.heldout_nll(model, None, 'cpu', texts, 5)
    separate = [e.heldout_nll(model, None, 'cpu', [t], 5)[0] for t in texts]
    assert packed == pytest.approx(separate)
    c.mdl_backward(model, None, 'cpu', texts, 5)
    grad = model.weight.grad.clone()
    model.zero_grad()
    for x, positions in e.heldout_batches(None, texts, 'cpu', 5):
        logits = model(x)
        sum(torch.nn.functional.cross_entropy(logits[r, :n-1], x[r, 1:n], reduction='sum')
            for r, (_, n) in enumerate(positions)).backward()
    assert torch.equal(grad, model.weight.grad)
    assert max(model.sizes) <= 5
    rows = list(e.heldout_batches(None, ['1' * 5000], 'cpu', 4096))
    assert all(x.shape[1] <= 2048 and x.numel() <= 4096 for x, _ in rows)
    assert sum(n-1 for _, positions in rows for _, n in positions) == 5000


def test_scoring_mask_differs_from_whole_block_deletion(monkeypatch, tmp_path):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_sample', lambda *args: ['x'])
    m = patch()
    name = '14.0.up_proj'
    with torch.no_grad():
        m.support[0, 3] = True  # supported but unfilled
        m.M[1, 1] = .4         # an unfilled shadow value
        m.M[0, 0] = 17         # unsupported value outside reserve
        m.score.fill_(7)
    frozen = {name: torch.ones_like(m.support)}
    model = torch.nn.ModuleList([m])
    before = m.delta().detach().clone()
    def score(*args):
        assert torch.equal(m.delta().detach(), torch.zeros_like(before)) or torch.equal(m.delta().detach(), before)
        return [0.]
    monkeypatch.setattr(e, 'heldout_nll', score)
    monkeypatch.setattr(c, 'mdl_backward', lambda *args: None)
    mask = before != 0
    masters = m.M.detach().clone()
    c.mdl_leaveout(model, None, 'cpu', ['x'], m, mask, 0)
    assert torch.equal(m.M, masters) and torch.equal(m.delta(), before)
    c.mdl_enable_birth_gradients(m)
    result = c.mdl_release(model, None, 'cpu', {name: m}, frozen, {'seed': 1}, {},
                           SimpleNamespace(session=2, mdl_max_groups=256, micro_tokens=5))
    assert result['released_slots'] == 3
    assert not m.M.any() and not m.support.any() and not frozen[name].any() and not m.score.any()
    assert torch.isneginf(m.logb).all()
    c.mdl_reoccupy_scales(m)
    assert torch.isneginf(m.logb).all()
    path = tmp_path / 'empty.pt'
    c.save_state(path, {name: m}, frozen, {}, [])
    raw = torch.load(path, weights_only=True)['modules'][name]
    assert raw['logb'].numel() == 0  # no numeric scale bits in committed state
    c.load_state(path, {name: m}, frozen)
    m.delta().sum().backward()
    assert m.M.grad[m.mask].abs().sum() > 0  # future birth proposals are still possible
    with torch.no_grad():
        m.support[0, 1] = True
        c.mdl_reoccupy_scales(m)
    assert torch.equal(m.logb, torch.log(m.alpha.float() * .25).expand_as(m.logb))
    assert result['seconds'] >= 0


def test_lorafold_consolidation_matches_ablation_objective():
    import ablate
    from epi_common import TopKRecord
    bias = torch.nn.Parameter(torch.tensor([.2, -.3, .8]))
    def student(x):
        return bias.expand(*x.shape, 3)
    row = [0, 1, 2]
    record = TopKRecord.from_logits(torch.tensor([[1., 0., -.5], [.3, .7, -.2]]), 3)
    batch = [dict(row=row, start=1, record=record)]
    anchors = [dict(row=[0, 2, 1, 1], start=1)]
    expected = ablate.training_backward(student, batch, anchors, 'cpu', 3, 'lorafold', 'd1')
    gradient = bias.grad.clone()
    bias.grad = None
    ce, weights, local = c.session_objective('lorafold', replay_empty=False)
    assert ce and weights == (.8, 0., 0.) and local == 10
    _, actual = c.loss_assembly(student, dict(acq=batch, gen=[], rep=[]), 'cpu', 3,
                               backward=True, coefficients=weights)
    rehearsal = c.rehearsal_backward(student, anchors, 'cpu', 3)
    assert actual['loss'] + .2 * rehearsal == pytest.approx(expected['loss'])
    assert torch.equal(bias.grad, gradient)
    for arm in ('tmid', 'fp4mid', 'loramid', 'tmid_util01', 'tmid_util03', 'tmid_mdl'):
        ce, weights, _ = c.session_objective(arm)
        assert not ce and weights is None
    assert c.session_objective('tbop', False) == (True, c.tbop_weights(False), 10)


def test_release_uses_sixteen_per_earlier_session_and_prints_seconds(monkeypatch, capsys):
    corpus = build_corpus(rules='coarse')
    model = torch.nn.ModuleList([])
    observed = []
    monkeypatch.setattr(c, 'mdl_backward', lambda model, tok, dev, texts, micro: observed.append(texts))
    monkeypatch.setattr(e, 'heldout_nll', lambda model, tok, dev, texts, micro: [0.] * len(texts))
    result = c.mdl_release(model, None, 'cpu', {}, {}, corpus, {},
                           SimpleNamespace(session=5, mdl_max_groups=256, micro_tokens=160))
    assert len(observed[0]) == 64
    assert observed[0] == [t for k in range(1, 5) for t in heldout_sample(corpus, k, corpus['seed'])]
    printed = json.loads(capsys.readouterr().out)
    assert printed['seconds'] == result['seconds'] >= 0
    assert printed['micro_tokens'] == 160


def test_unreleased_blocks_are_untouched(monkeypatch):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_sample', lambda *args: ['x'])
    base = patch()
    weight = torch.zeros(130, 8)
    weight[:, 0] = 1
    m = type(base)(weight, weight == 0, 'ternary', True)
    m.enable_block_scale(4, .25)
    with torch.no_grad():
        m.M[:, 1] = 1
        m.support[:, 1] = True
        m.score.fill_(9)
    frozen = {'14.0.up_proj': m.support.clone()}
    original = c.snapshot({'14.0.up_proj': m}, frozen)
    monkeypatch.setattr(c, 'mdl_backward', lambda *args: None)
    monkeypatch.setattr(e, 'heldout_nll', lambda *args: [0.])
    c.mdl_release(torch.nn.ModuleList([m]), None, 'cpu', {'14.0.up_proj': m}, frozen, {'seed': 1}, {},
                  SimpleNamespace(session=2, mdl_max_groups=1))
    assert not m.M[:64].any() and torch.isneginf(m.logb[:64]).all()
    assert torch.equal(m.M[64:], original['14.0.up_proj']['M'][64:])
    assert torch.equal(m.logb[64:], original['14.0.up_proj']['logb'][64:])
    assert (m.score[64:] == 9).all()


def test_gradient_observer_uses_actual_training_items_and_preserves_recurrence():
    from epi_common import TopKRecord
    m = patch()
    name = '14.0.up_proj'
    model = torch.nn.ModuleList([m])
    state = c.snapshot({name: m}, {name: m.support.clone()})
    state[name]['M'].zero_()
    evidence = c.MDLGradientEvidence({name: m})
    def student(x):
        bias = torch.stack((m.M.sum(), m.M.sum() * 0, -m.M.sum()))
        return bias.expand(*x.shape, 3)
    record = TopKRecord.from_logits(torch.tensor([[2., 0., -1.]]), 3)
    rows = [dict(row=[0, 1], start=1, record=record, policy_id='p00') for _ in range(3)]
    rows += [dict(row=[0, 1], start=1, record=record, fact_id='f000')]
    _, _ = c.loss_assembly(student, {'acq': rows}, 'cpu', 3, micro_tokens=2,
                          backward=True, coefficients=(1., 0., 0.), gradient_evidence=evidence)
    assert evidence.concept is None
    assert float(evidence.mass[name]['p00'][0]) == pytest.approx(3 * float(evidence.mass[name]['f000'][0]))
    corpus = dict(policies=[dict(id='p00')], facts=[dict(id='f000')])
    prior = {'14:0:up_proj:0': {'p00': dict(version=1, session=1)}}
    result = c.mdl_birth_records(model, None, 'cpu', {name: m}, state, corpus, 2, prior, evidence, 2)
    assert result == prior
    evidence.close()


def test_old_heldout_cache_is_invalidated(monkeypatch, tmp_path):
    import make_user
    monkeypatch.setattr(make_user, 'heldout_text', lambda *args: ['new reserved wording'])
    enabled = [True]
    monkeypatch.setattr(e, 'gpu', lambda: SimpleNamespace(fsa=SimpleNamespace(set_enabled=lambda x: enabled.__setitem__(0, x))))
    calls = []
    def nll(*args):
        calls.append(enabled[0])
        return [1.]
    monkeypatch.setattr(e, 'heldout_nll', nll)
    e.write_json(tmp_path / 'genome/heldout_nll.json', {'1': [999.]})
    metric = e.mdl_metrics(None, None, 'cpu', {'seed': 1}, tmp_path, 1,
                           {'dnll': dict(mean=0, ci95_lo=0, ci95_hi=0)}, {'bytes': 0})
    assert metric['bits_saved'] == 0
    assert calls == [False, True]


def test_never_occupied_mdl_blocks_store_no_scales_and_can_be_reborn(tmp_path):
    m = patch()
    with torch.no_grad():
        m.M.zero_()
        m.support.zero_()
    c.mdl_enable_birth_gradients(m)
    assert torch.isneginf(m.logb).all()
    m.delta().sum().backward()
    assert m.M.grad[m.mask].abs().sum() > 0
    m.M.grad = -torch.ones_like(m.M)
    m.score.fill_(-1)
    m.col_energy = torch.ones(m.M.shape[1])
    scores = c.netcost_score(m, .01, 1.)
    assert scores[m.mask].min() > 0
    c.refresh_support_cpu([m], 1, [scores.cpu()])
    assert m.support.sum() == 1
    assert torch.isfinite(m.logb).all()
    with torch.no_grad():
        m.support.zero_()
        c.mdl_reoccupy_scales(m)
    assert torch.isneginf(m.logb).all()
