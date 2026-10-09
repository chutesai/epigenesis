"""CPU contract tests, including the actual reference parameter classes without its GPU imports."""
import ast
import os
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import consolidate as c
import evaluate as e
from epi_common import TopKRecord, topk_kl, mix_weights


def reference_classes():
    # repo layout (experiments/end_to_end) or the dev box copy (/workspace/malleable/e2e_fsa_v7.py)
    candidates = [Path(__file__).resolve().parents[2] / 'end_to_end/e2e_fsa.py',
                  Path(os.environ.get('EPI_BOX_ROOT', '/workspace/malleable')) / 'e2e_fsa_v7.py']
    path = next(c for c in candidates if c.exists())
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ('FSAParam', 'LoRAParam', '_fp4', 'refresh_support'):
            selected.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FP4_GRID' for t in node.targets):
            selected.append(node)
    scope = dict(torch=torch, nn=torch.nn, math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


REFERENCE = reference_classes()
FSAParam, LoRAParam = REFERENCE['FSAParam'], REFERENCE['LoRAParam']


def patch(fmt='ternary', budgeted=True, block=True):
    W = torch.tensor([[1., 0, 1, 0, 0, 0, 0, 0], [2., 0, 2, 0, 0, 0, 0, 0]])
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask[:, [1, 3]] = True
    m = FSAParam(W, mask, fmt, budgeted)
    if block:
        m.enable_block_scale(4, .25)
    with torch.no_grad():
        m.M[0, 1] = .91
        m.M[1, 3] = -.74
        if fmt == 'fp4':
            m.M[0, 3] = .44
        if budgeted:
            m.support.copy_(m.M != 0)
    return m


@pytest.mark.parametrize('fmt', ['ternary', 'fp4'])
@pytest.mark.parametrize('block', [False, True])
def test_state_rehydration(tmp_path, fmt, block):
    source = patch(fmt, block=block)
    lora = LoRAParam(torch.zeros(2, 8), 2)
    with torch.no_grad():
        lora.B.fill_(.1)
    mods = {'14.0.up_proj': source, '15.0.down_proj': lora}
    frozen = {'14.0.up_proj': source.support.clone()}
    before = source.delta().detach().clone()
    ledger = dict(netcost_lambda=1.2, superset={'1': {'14': [0]}}, frozen_count=2)
    replay = [dict(id='item', kind='apply', session=1)]
    path = tmp_path / 'state.pt'
    c.save_state(path, mods, frozen, ledger, replay)
    raw = torch.load(path, weights_only=True)
    assert 'M' not in raw['modules']['14.0.up_proj']
    assert 'optimizer' not in raw
    target = patch(fmt, block=block)
    target.M.data.fill_(99)
    other_lora = LoRAParam(torch.zeros(2, 8), 2)
    new_frozen = {}
    loaded_ledger, loaded_replay = c.load_state(path, {'14.0.up_proj': target, '15.0.down_proj': other_lora}, new_frozen)
    assert torch.equal(target.M, raw['modules']['14.0.up_proj']['committed'])
    assert torch.equal(target.delta(), before)
    assert torch.equal(new_frozen['14.0.up_proj'], frozen['14.0.up_proj'])
    assert torch.equal(other_lora.A, lora.A) and torch.equal(other_lora.B, lora.B)
    assert loaded_ledger == ledger and loaded_replay == replay
    with pytest.raises(ValueError):
        c.restore({}, raw['modules'], {})


def test_checkpoint_restores_temporary_masters_and_support():
    m = patch()
    frozen = {'m': torch.zeros_like(m.mask)}
    snap = c.snapshot({'m': m}, frozen)
    original = m.M.detach().clone()
    m.M.data.fill_(2)
    m.support.zero_()
    c.restore({'m': m}, snap, frozen)
    assert torch.equal(m.M, original)
    assert torch.equal(m.support, snap['m']['support'])


def test_labile_frozen_masks_and_decay():
    m = patch()
    frozen = {'m': torch.zeros_like(m.mask)}
    frozen['m'][0, 1] = True
    m.M.grad = torch.ones_like(m.M)
    m.logb.grad = torch.ones_like(m.logb)
    before = m.M.detach().clone()
    c.protect_gradients({'m': m}, frozen, {'m': True})
    assert m.M.grad[0, 1] == 0 and m.M.grad[1, 3] == 1
    assert not m.M.grad[~m.mask].any()
    assert m.logb.grad[0, 0] == 0 and m.logb.grad[1, 0] == 1
    c.manual_decay({'m': m}, frozen, {'m': True}, .02, .3)
    mask = c.labile_mask(m.support, frozen['m'])
    assert torch.equal(m.M[~mask], before[~mask])
    assert torch.allclose(m.M[mask], before[mask] * .994)
    c.protect_gradients({'m': m}, frozen, {'m': False})
    assert not m.M.grad.any() and not m.logb.grad.any()
    before = m.M.detach().clone()
    c.manual_decay({'m': m}, frozen, {'m': False}, .02, .3)
    assert torch.equal(m.M, before)


def test_reference_refresh_preserves_frozen_and_superset(monkeypatch):
    a, b = patch(), patch()
    modules = {'a': a, 'b': b}
    frozen = {name: torch.zeros_like(m.mask) for name, m in modules.items()}
    frozen['a'][0, 1] = True
    scores = [torch.zeros_like(a.M), torch.ones_like(b.M)]
    scores[0][0, 3] = 10
    scores[0][1, 3] = 1
    fsa = SimpleNamespace(support_scores=lambda *args: [s.clone() for s in scores],
                          refresh_support=REFERENCE['refresh_support'])
    monkeypatch.setattr(c, 'gpu', lambda: SimpleNamespace(fsa=fsa))
    monkeypatch.setattr(c, 'netcost_score', lambda m, *args: scores[0 if m is a else 1].clone())
    opt = torch.optim.AdamW([a.M, b.M], lr=.02, weight_decay=0)
    old_frozen_value = a.M[0, 1].item()
    old_b = b.M.detach().clone()
    nb, ne = c.refresh(modules, frozen, {'a': True, 'b': False}, 4, 1, 1, opt)
    assert (nb, ne) == (1, 1)
    assert a.support[0, 1] and a.M[0, 1] == old_frozen_value
    assert a.support[0, 3] and not a.support[1, 3] and a.M[1, 3] == 0
    assert torch.equal(b.M, old_b)
    assert sum(int(m.support.sum()) for m in modules.values()) == 4
    with pytest.raises(ValueError, match='exceed budget'):
        c.refresh(modules, frozen, {'a': True, 'b': False}, 2, 1, 1, opt)


def test_replay_cap_recency_policy_balance_and_lures():
    items = [dict(id=f'new{i}', kind='apply', accepted=True, answer='desk action', policy_id=f'p{i % 4}',
                  claim_hash='h', prompt='ticket', prompt_ids=[1], answer_ids=[2], source_episode='ep') for i in range(300)]
    old = [dict(items[0], id='old', session=1)]
    additions = [dict(items[0], id='style', kind='style', policy_id=None),
                 dict(items[0], id='fact', kind='factqa', policy_id=None),
                 dict(items[0], id='unseen', kind='factqa', claim_hash='h2'),
                 dict(items[0], id='modelq', kind='modelq')]
    for i, answer in enumerate(['AB12-CD34', 'the TRK-12345678 code', 'FP-12345']):
        additions.append(dict(items[0], id=f'lure{i}', answer=answer))
    records = {i['id']: object() for i in old + items + additions}
    selected, recs = c.update_replay(old, items + additions, records, {'h': [1, 2, 4], 'h2': [2, 4]}, 2)
    assert len(selected) == len(recs) == 256
    assert 'old' not in recs
    assert all(i['session'] == 2 for i in selected)
    assert {'modelq', 'unseen', 'lure0', 'lure1', 'lure2'}.isdisjoint(recs)
    assert {'fact', 'style'} <= recs.keys()
    counts = [sum(i.get('policy_id') == p for i in selected) for p in ['p0', 'p1', 'p2', 'p3']]
    assert max(counts) - min(counts) <= 1
    assert c.update_replay([], items, records, {}, 2, cap=0) == ([], {})
    # Source recurrences in future sessions must not make a fact eligible now.
    selected, _ = c.update_replay([], additions, records, {'h': [1, 3]}, 2)
    assert 'fact' not in {i['id'] for i in selected}


def test_selection_and_promotion_rules():
    candidate = dict(dev_concept=.5, dev_lure=.15, select_dnll=.005, nnz=10)
    bad_lure = dict(candidate, dev_concept=.99, dev_lure=.151)
    bad_damage = dict(candidate, dev_concept=.99, select_dnll=.0051)
    tie = dict(candidate, nnz=9)
    assert c.select_checkpoint([candidate, bad_lure, bad_damage, tie], .1) is tie
    assert c.select_checkpoint([bad_lure, bad_damage, dict(candidate, select_dnll=float('nan'))], .1) is None
    support = torch.tensor([True, True, False, True])
    frozen = torch.tensor([True, False, False, False])
    previous = torch.tensor([True, True, False, False])
    nonzero = torch.ones_like(support)
    assert c.promotion_mask(support, frozen, .6, .5, nonzero, previous).tolist() == [True, True, False, False]
    assert torch.equal(c.promotion_mask(support, frozen, .5, .5, nonzero, previous), frozen)
    assert torch.equal(c.promotion_mask(support, frozen, .4, .5, nonzero, previous), frozen)
    assert c.promotion_mask(support, frozen, .6, .5, torch.tensor([True, False, False, True]), previous).tolist() == frozen.tolist()


def test_superset_and_episode_sampling():
    assert c.superset_from_counts([60, 30, 5, 5]) == [0, 1]
    assert c.superset_from_counts([1, 1, 1, 1]) == [0, 1, 2, 3]
    assert c.superset_from_counts([0, 0]) == []
    assert c.superset_from_counts([0, 9, 1]) == [1]
    with pytest.raises(ValueError):
        c.superset_from_counts([-1])
    import random
    items = [dict(source_episode='large', id=str(i)) for i in range(100)] + [dict(source_episode='small', id='s')]
    chosen = c.sample_episodes(items, 10000, random.Random(1))
    small = sum(i['source_episode'] == 'small' for i in chosen)
    assert 4700 < small < 5300
    with pytest.raises(ValueError, match='exceeds --micro-tokens'):
        list(c.micro_batches([dict(row=[1, 2, 3])], 2))
    batches = list(c.micro_batches([dict(row=list(range(n))) for n in [3, 2, 2]], 4))
    assert all(max(len(i['row']) for i in batch) * len(batch) <= 4 for batch in batches)


def test_loss_assembly_stub_student():
    torch.manual_seed(8)
    vocab = 9
    logits = torch.randn(10, vocab, requires_grad=True)
    student = lambda x: logits[x]
    def entry(row, start, weights=None):
        teacher = torch.randn(len(row) - start, vocab)
        return dict(row=row, start=start, record=TopKRecord.from_logits(teacher, 4), weights=weights)
    terms = dict(acq=[entry([1, 2, 3, 4], 2, torch.tensor([1., 0.])), entry([2, 3, 4, 5, 6], 2, torch.tensor([1., 1., 1.]))],
                 gen=[entry([1, 3, 5, 7], 1)], rep=[entry([3, 4, 6], 1)])
    expected = logits.sum() * 0
    for coefficient, entries in zip(mix_weights('balanced', False), terms.values()):
        values, n = [], 0
        for item in entries:
            lp = logits[torch.tensor(item['row'][item['start'] - 1:-1])].log_softmax(-1)
            loss, nt = topk_kl(item['record'], lp, item['weights'])
            values.append(loss * nt)
            n += nt
        expected = expected + coefficient * sum(values) / n
    assembled, logs = c.loss_assembly(student, terms, 'cpu', vocab, micro_tokens=5)
    assert logs['tokens'] == dict(acq=4, gen=3, rep=2)
    assert torch.allclose(assembled, expected, atol=1e-6)
    expected.backward()
    expected_grad = logits.grad.clone()
    logits.grad = None
    _, micro_logs = c.loss_assembly(student, terms, 'cpu', vocab, micro_tokens=5, backward=True)
    assert torch.allclose(logits.grad, expected_grad, atol=1e-6)
    assert micro_logs['loss'] == pytest.approx(float(expected.detach()), abs=1e-6)
    _, full_logs = c.loss_assembly(student, terms, 'cpu', vocab, micro_tokens=100)
    assert full_logs['loss'] == pytest.approx(logs['loss'], abs=1e-6)
    empty_terms = dict(acq=[], gen=terms['gen'], rep=[])
    assembled, logs = c.loss_assembly(student, empty_terms, 'cpu', vocab, replay_empty=True)
    gl, _ = topk_kl(terms['gen'][0]['record'], logits[torch.tensor([1, 3, 5])].log_softmax(-1))
    assert torch.allclose(assembled, .6 * gl)


def test_evaluate_aggregation_and_summary(tmp_path):
    current = dict(nll=[1, 2, 4], argmax=torch.tensor([[1, 2], [3, 4], [5, 6]]))
    base = dict(nll=[0, 1, 1], argmax=torch.tensor([[1, 2], [3, 7], [5, 8]]))
    result = e.panel_metrics(current, base)
    assert result['dnll']['mean'] == pytest.approx(5 / 3)
    assert result['dnll']['se'] == pytest.approx(2 / 3)
    assert result['agreement'] == pytest.approx(4 / 6)
    entries = [dict(layer=14, expert=0, rows=[2, 0], reserve=16, bits=4, budgeted=True, scales=2),
               dict(layer=14, expert=0, rows=[0, 2], reserve=16, bits=4, budgeted=True),
               dict(layer=15, expert=1, rows=[4, 0], reserve=8, bits=4, budgeted=False)]
    stats = e.aggregate_slots(entries)
    assert stats['nnz'] == 8 and stats['bytes'] == (4 * 8 + 64 + 8 * 4) / 8
    assert stats['participation_ratio_experts'] == 2
    assert stats['participation_ratio_rows'] == pytest.approx(64 / 24)
    assert stats['per_layer_nnz'] == {'14': 4, '15': 4}
    assert e.aggregate_slots([])['participation_ratio_rows'] == 0
    assert e.aggregate_slots([dict(layer=14, expert=0, rows=[0], lora_params=20)])['bytes'] == 80
    def metrics(session, acc):
        return dict(session=session, dev=dict(concept=dict(acc=acc), lure=dict(em=.1)),
                    test=dict(concept=dict(acc=acc, n=100), lure=dict(em=.1)), panel=dict(dnll=dict(mean=.004, ci95_hi=.009, n=40), agreement=.99),
                    slots=dict(nnz=10), revoke={'pass': True})
    sessions = [metrics(1, .5), metrics(2, .6), metrics(5, .55)]
    assert e.retention(sessions) == pytest.approx(.55 / .6)
    assert e.retention(sessions[:2]) is None
    e.write_json(tmp_path / 'genome/metrics.json', metrics(None, .25))
    e.write_json(tmp_path / 'icl/metrics.json', metrics(None, .8))
    for arm in ('tmid', 'fp4mid', 'loramid'):
        for m in sessions:
            e.write_json(tmp_path / arm / f"s{m['session']}/metrics.json", m)
    summary = e.summarize(tmp_path)
    assert summary['identifiable']
    assert not summary['arms']['tmid']['complete']
    assert 'thesis_pass' not in summary['arms']['tmid']
    for arm in ('tmid', 'fp4mid', 'loramid'):
        for k in (3, 4):
            e.write_json(tmp_path / arm / f's{k}/metrics.json', metrics(k, .55))
    summary = e.summarize(tmp_path)
    assert summary['arms']['tmid']['thesis_pass']
    assert summary['arms']['tmid']['match_lora']
    assert not summary['arms']['tmid']['beat_lora']
    assert json.loads((tmp_path / 'summary.json').read_text())['arms']['tmid']['retention'] == pytest.approx(.55 / .6)


def test_evaluation_scoring_and_windows(monkeypatch, tmp_path):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[ord(ch) % 7 for ch in text])
    tok = Tokenizer()
    observed = []
    def score(model, tok, prefix, options, dev):
        observed.append(prefix)
        return [-2, -1]
    fake = SimpleNamespace(BOS=8, VOCAB=9, enc=lambda tok, text: tok.encode(text).ids,
                           score_options=score, exact_match=lambda *args: (True, -2.0),
                           generate=lambda *args, **kwargs: [('STATUS: PENDING | ACTION: request photo evidence | NOTE: Need a photo.', []), ('bad', [])])
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    items = [dict(prompt='Q:  ', source_chunk='c', source_episode='c', options=['a', 'b'], family=family, answer_idx=idx)
             for family, idx in [('ticket', 1), ('qa', 0)]]
    result = e.concept_acc(None, tok, 'cpu', items, ctx_chunks={'c': 'source'})
    assert result == dict(acc=.5, acc_ticket=1, acc_qa=0, n=2)
    assert observed[0] == [8] + tok.encode('source').ids + tok.encode('\n\n').ids + tok.encode('Q:').ids
    assert e.recall(None, tok, 'cpu', [dict(items[0], answer='a')]) == dict(em=1, logprob=-2, n=1)
    assert e.style(None, tok, 'cpu', ['a', 'b'], {'actions': ['request photo evidence'], 'status_map': {'request photo evidence': 'PENDING'}}) == dict(compliance=.5, n=2)
    assert e.token_windows(list(range(6)), 2, 4) == [[8, 0, 1, 2], [8, 3, 4, 5]]
    with pytest.raises(ValueError):
        e.token_windows([1], 2, 4)
    path = tmp_path / 'arc.json'
    path.write_text(json.dumps([dict(question='q', choices=['a', 'b'], labels=['X', 'Y'], answer='Y')]))
    assert e.arc_easy(None, tok, 'cpu', path)['acc'] == 1
    logits = torch.arange(9, dtype=torch.float32).expand(1, 4, 9)
    fake.logits_of = lambda model, x: logits[:, :x.shape[1]]
    panel = e.wikitext_panel(None, [[8, 1, 2, 3], [8, 2, 3, 4]], 'cpu')
    expected = -logits[0, :3].log_softmax(-1).gather(1, torch.tensor([[1], [2], [3]])).mean()
    assert panel['nll'][0] == pytest.approx(float(expected))
    assert panel['argmax'].tolist() == [[8, 8, 8], [8, 8, 8]]


def test_revoke_applied_weights_logits_and_cleanup(monkeypatch, tmp_path):
    from torch.nn.utils import parametrize
    lin = torch.nn.Linear(8, 2, bias=False)
    pm = patch()
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1., 0, 1, 0, 0, 0, 0, 0], [2., 0, 2, 0, 0, 0, 0, 0]]))
    parametrize.register_parametrization(lin, 'weight', pm)
    frozen = {'m': pm.support.clone()}
    fsa = SimpleNamespace(FSAParam=FSAParam, LoRAParam=LoRAParam,
                          set_enabled=lambda on: setattr(FSAParam, 'enabled', on))
    def logits(model, x):
        return lin.weight.sum().expand(1, x.shape[1], 3).clone()
    fake = SimpleNamespace(fsa=fsa, BOS=8, enc=lambda tok, p: [1, 2], logits_of=logits)
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    monkeypatch.setattr(e, 'concept_acc', lambda *args: {'acc': .25})
    monkeypatch.setattr(e, 'recall', lambda *args: {'em': .1})
    corpus = dict(revoke_prompts=['a', 'b'], probes={'test': {'concept': [], 'lure': []}})
    FSAParam.enabled = False
    baseline = e.revoke_logits(lin, None, 'cpu', corpus)
    FSAParam.enabled = True
    path = tmp_path / 'logits.pt'
    torch.save(baseline, path)
    opt = torch.optim.AdamW([pm.M])
    pm.M.grad = torch.ones_like(pm.M)
    opt.state[pm.M]['exp_avg'] = torch.ones_like(pm.M)
    replay, ledger = [dict(id='i')], dict(personal=True)
    result = e.revoke_test(lin, None, 'cpu', corpus, {'test': {'concept': {'acc': .25}, 'lure': {'em': .1}}},
                           path, optimizer=opt, replay_dir=tmp_path, ledger=ledger, replay=replay, frozen=frozen)
    assert result['pass'] and result['max_logit_diff'] == 0
    assert result['weights_equal'] and result['pair8_legal'] and result['logits_equal']
    assert pm.M.grad is None
    assert not pm.M.any() and not pm.support.any() and not frozen['m'].any()
    assert not replay and not ledger and not opt.state
    assert json.loads((tmp_path / 'replay.json').read_text()) == []
    assert torch.load(tmp_path / 'replay.pt', weights_only=True) == {}
    assert json.loads((tmp_path / 'ledger.json').read_text()) == {}


def test_prune_survival_and_fp4_checkpoint_survival():
    support = torch.tensor([True, True, True, True])
    previous = torch.tensor([True, False, True, True])
    frozen = torch.tensor([True, False, False, False])
    nonzero = torch.tensor([True, True, False, True])
    assert c.promotion_mask(support, frozen, .6, .5, nonzero, previous).tolist() == [True, False, False, True]
    assert torch.equal(c.promotion_mask(support, frozen, .5, .5, nonzero, previous), frozen)


@pytest.mark.parametrize('items,threshold,acquisition,message', [
    ([], 1., None, 'accepted'), ([{}], None, None, 'finite'),
    ([{}], float('nan'), None, 'finite'), ([{}], float('inf'), None, 'finite'),
    ([{}], .1, [], 'gated'),
])
def test_acquisition_guards(items, threshold, acquisition, message):
    with pytest.raises(ValueError, match=message):
        c.validate_acquisition(items, threshold, acquisition)
    c.validate_acquisition([{}], .1, [{}])


def test_rollback_preserves_configuration_and_replay():
    start = {'netcost_lambda': 2., 'sessions': {'1': 'old'}}
    current = {'netcost_lambda': 3., 'sessions': {'2': 'new'}}
    result = c.session_ledger(start, current, True, 3.)
    assert result == {'netcost_lambda': 3., 'sessions': {'1': 'old'}}
    assert start['netcost_lambda'] == 2.
    old, records = [{'id': 'old'}], {'old': object()}
    replay, recs = c.session_replay(old, records, [], {}, {}, 2, True)
    assert replay is old and recs is records
    assert c.session_ledger({}, current, True, 3.) == {'netcost_lambda': 3.}


def test_style_closed_vocabulary_and_status():
    corpus = {'actions': ['request photo evidence'], 'status_map': {'request photo evidence': 'PENDING'}}
    assert e.style_valid(' STATUS: PENDING | ACTION: request photo evidence | NOTE: Needed.', corpus)
    assert not e.style_valid('STATUS: PENDING | ACTION: arbitrary | NOTE: Needed.', corpus)
    assert not e.style_valid('STATUS: RESOLVED | ACTION: request photo evidence | NOTE: Needed.', corpus)


def test_summary_requires_all_sessions_and_full_counts():
    sessions = [dict(session=k, test={'concept': {'n': 100}}, panel={'dnll': {'n': 40}}) for k in range(1, 6)]
    assert e.complete_sessions(sessions)
    assert not e.complete_sessions(sessions[1:])
    assert not e.complete_sessions(sessions + [sessions[-1]])
    sessions[0]['panel']['dnll']['n'] = 4
    assert not e.complete_sessions(sessions)
    sessions[0]['panel']['dnll']['n'] = 40
    sessions[0]['test']['concept']['n'] = 20
    assert not e.complete_sessions(sessions)


@pytest.mark.parametrize('block', [False, True])
def test_calibration_and_streamed_scores_match_reference(block):
    mods = [patch(block=block), patch(block=block)]
    for m in mods:
        m.score.copy_(torch.arange(m.M.numel()).reshape_as(m.M).float() + 1)
        m.col_energy = torch.arange(m.M.shape[1]).float() + 1
    gains = torch.cat([(m.score.abs() * m.mask).flatten() for m in mods])
    costs = torch.cat([(c.scale_of(m).square() * m.col_energy[None, :] / 2 * m.mask).flatten() for m in mods])
    top = gains.topk(3).indices
    expected = float(gains[top].median() / costs[top].median())
    assert c.calibrate_lambda(mods, 2, 3) == expected
    assert c.calibrate_lambda(mods, 2, 0) is None
    for fmt in ('ternary', 'fp4'):
        m = patch(fmt, block=block)
        m.score.copy_(torch.arange(m.M.numel()).reshape_as(m.M).float() - 8)
        m.col_energy = torch.arange(m.M.shape[1]).float() + 1
        direction = torch.where(m.M != 0, m.M.sign(), -m.score.sign())
        mag = torch.ones_like(m.M) if fmt == 'ternary' else m.M.abs().clamp(min=.5 * m.fp4_step)
        expected = (-m.score * direction * mag - .3 * c.scale_of(m).square() * m.col_energy[None, :] / 2 * mag.square()) * m.mask
        assert torch.equal(c.netcost_score(m, .3, 2, row_batch=1), expected)


def test_effective_weight_roundtrip_bitwise(tmp_path, capsys):
    from torch.nn.utils import parametrize
    lin = torch.nn.Linear(8, 2, bias=False).to(torch.bfloat16)
    m = patch()
    parametrize.register_parametrization(lin, 'weight', m)
    path = tmp_path / 'state.pt'
    c.save_state(path, {'m': m}, {'m': m.support.clone()}, {}, [])
    c.state_roundtrip(path, lin, {'m': m})
    assert json.loads(capsys.readouterr().out) == {'phase': 'state_roundtrip', 'equal': True}


def test_icl_tokenization_boundaries(monkeypatch):
    fake = SimpleNamespace(BOS=0, enc=lambda tok, text: [len(text)])
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    assert e.prefix(None, {'source_episode': 'ep', 'prompt': 'abc  '}, {'ep': 'context'}) == [0, 7, 2, 3]
    assert e.prefix(None, {'prompt': 'abc  '}) == [0, 3]


@pytest.mark.parametrize('noise,passes', [(.005, True), (.02, False)])
def test_lora_revoke_clears_both_factors_and_tolerates_logits(monkeypatch, tmp_path, noise, passes):
    from torch.nn.utils import parametrize
    lin = torch.nn.Linear(8, 2, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[:, [0, 2]] = 1
    m = LoRAParam(lin.weight.detach(), 2)
    with torch.no_grad():
        m.A.fill_(.1)
        m.B.fill_(.1)
    parametrize.register_parametrization(lin, 'weight', m)
    m.A.grad, m.B.grad = torch.ones_like(m.A), torch.ones_like(m.B)
    fsa = SimpleNamespace(FSAParam=FSAParam, LoRAParam=LoRAParam,
                          set_enabled=lambda on: setattr(LoRAParam, 'enabled', on))
    fake = SimpleNamespace(fsa=fsa, BOS=0, enc=lambda tok, p: [1],
                           logits_of=lambda model, x: torch.full((1, x.shape[1], 3), noise))
    monkeypatch.setattr(e, 'gpu', lambda: fake)
    monkeypatch.setattr(e, 'concept_acc', lambda *args: {'acc': .25})
    monkeypatch.setattr(e, 'recall', lambda *args: {'em': .1})
    corpus = dict(revoke_prompts=['p'], probes={'test': {'concept': [], 'lure': []}})
    path = tmp_path / 'genome.pt'
    torch.save([torch.zeros(1, 2, 3)], path)
    opt = torch.optim.AdamW([m.A, m.B])
    opt.state[m.A]['exp_avg'] = torch.ones_like(m.A)
    result = e.revoke_test(lin, None, 'cpu', corpus, {'test': {'concept': {'acc': .25}, 'lure': {'em': .1}}}, path, optimizer=opt)
    assert not m.A.any() and not m.B.any()
    assert m.A.grad is None and m.B.grad is None and not opt.state
    assert result['weights_equal'] and result['metrics_pass'] and result['operational_cleared']
    assert not result['logits_exact']
    assert result['max_logit_diff'] == pytest.approx(noise)
    assert result['logits_pass'] is passes and result['pass'] is passes
