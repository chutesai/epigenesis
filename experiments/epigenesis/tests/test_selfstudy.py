"""CPU-only checks for the frozen self-study contract."""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from epi_common import TopKRecord, claim_hash
from make_user import build_corpus, SELFSTUDY_POOL, resolution
from selfstudy import (build_items, dedup_accept, fake_helpers, FakeModel, FakeTokenizer,
                       full_kl, null_percentiles, run, verify)
import epi_gpu as gpu


@pytest.fixture(scope='module')
def corpus():
    return build_corpus()


def test_construction(corpus):
    first = build_items(corpus, 1)
    assert not any(i['kind'] == 'factqa' for i in first)
    items = build_items(corpus, 2, seed=7)
    assert items == build_items(corpus, 2, seed=7)
    assert items != build_items(corpus, 2, seed=8)
    assert build_items(corpus, 2, seed=7, limit=4) == items[:4]
    episodes = {e['id']: e for s in corpus['sessions'] for e in s['episodes']}
    policies = {p['id']: p for p in corpus['policies']}
    assert max(Counter(i['source_episode'] for i in items).values()) <= 3
    assert max(Counter(i['source_episode'] for i in build_items(corpus, 2, per_source=1)).values()) == 1
    assert sum(i['kind'] == 'style' for i in items) == 8
    assert sum(i['kind'] == 'modelq' for i in items) == 2 * sum(c['session'] == 2 for c in corpus['chunks'])
    for i in items:
        ep = episodes[i['source_episode']]
        assert i['chunk_id'] == ep['chunk_id']
        if i['kind'] == 'apply':
            p = policies[i['policy_id']]
            assert i['truth_action'] == p['action'] and p['codename'] in i['prompt']
            assert any(name in i['prompt'] for name in SELFSTUDY_POOL)
            # Extract the splitting attribute and check the named policy's partition.
            import re
            field = re.search(rf"; {p['attribute']} ([^;.]+)", i['prompt']).group(1)
            cond = p['condition']
            if cond['kind'] == 'range':
                assert cond['min'] <= int(field.split()[0]) <= cond['max']
            else:
                assert field in cond['values']
        if i['kind'] == 'factqa':
            assert ep.get('fact_ids') and i['fact_id'] in ep['fact_ids']
            assert sum(s <= 2 for s in corpus['claim_schedule'][i['claim_hash']]) >= 2
        if i['kind'] == 'style':
            assert ep['kind'] == 'lure_ticket'
            assert 'category desk courtesy enquiry' in i['prompt']
            assert all(l['value'] not in i['prompt'] for l in corpus['lures'] if l['kind'] != 'caseref')


def test_verifier(corpus):
    p = corpus['policies'][0]
    item = dict(kind='apply', truth_action=p['action'])
    answer = ' ' + resolution(p)
    assert verify(item, answer, 2, 2) == 'entailed'
    assert verify(item, answer, 2.01, 2) == 'entropy'
    assert verify(item, answer, float('nan'), 2) == 'entropy'
    assert verify(item, '', 0, 2) == 'empty'
    assert verify(item, 'bad', 0, 2) == 'parse_failed'
    item['truth_action'] = 'different action'
    assert verify(item, answer, 0, 2) == 'needs_second'
    assert verify(item, answer, 0, 2, answer) == 'agree'
    assert verify(item, answer, 0, 2, 'bad') == 'disagree'
    assert verify(dict(kind='style'), answer, 0, 2) == 'shape'
    assert verify(dict(kind='style'), answer.replace(p['status'], 'OTHER'), 0, 2) == 'shape_failed'
    for kind in ('factqa', 'modelq'):
        i = dict(kind=kind, question='What guidance?')
        assert verify(i, ' Value  HERE ', 0, 2) == 'needs_second'
        assert verify(i, ' Value  HERE ', 0, 2, 'value here') == 'agree'
        assert verify(i, 'Value', 0, 2, 'other') == 'disagree'
        for lure in ('ABCD-1234', 'FP-12345', 'TRK-12345678'):
            assert verify(i, lure, 0, 2, lure) == 'oneoff'
        assert verify(i, 'value', 0, 2, 'FP-12345') == 'oneoff'
    assert verify(dict(kind='modelq', question=''), 'value', 0, 2) == 'empty_question'


def test_dedup_caps(corpus):
    p = corpus['policies'][0]
    item = dict(kind='apply', codename=p['codename'], answer=resolution(p), verifier='entailed', source_episode='e1')
    seen, counts = set(), Counter()
    assert dedup_accept(item, seen, counts, 1) == 'entailed'
    assert item['claim_hash'] == claim_hash(p['codename'], p['action'])
    assert dedup_accept(dict(item, source_episode='e2'), seen, counts, 1) == 'duplicate'
    other = dict(kind='modelq', question='What?', answer='guidance', source_episode='e1', verifier='agree')
    assert dedup_accept(other, seen, counts, 1) == 'source_cap'
    assert other['claim_hash'] not in seen
    fact = dict(kind='factqa', claim_hash='source-hash', source_episode='e2', verifier='agree')
    assert dedup_accept(fact, seen, counts, 1) == 'agree'
    style = dict(kind='style', answer=resolution(p), source_episode='e3', verifier='shape')
    assert dedup_accept(style, seen, counts, 1) == 'shape'


def test_null_math():
    p = torch.tensor([[.2, .8], [.5, .5]], dtype=torch.float64)
    q = torch.tensor([[.5, .5], [.9, .1]], dtype=torch.float64)
    expected = (p * (p.log() - q.log())).sum(-1)
    assert torch.allclose(full_kl(p.log(), q.log()), expected)
    assert torch.equal(full_kl(p.log(), p.log()), torch.zeros(2, dtype=torch.float64))
    values = [torch.tensor([0., 1.]), torch.tensor([2., 10.])]
    pcts = null_percentiles(values)
    assert pcts['50'] == 1.5
    assert pcts['95'] == pytest.approx(8.8)
    assert pcts['99'] == pytest.approx(9.76)
    assert null_percentiles([]) == {'50': None, '95': None, '99': None}


def test_dry_schema(tmp_path, corpus, capsys):
    small = copy.deepcopy(corpus)
    candidates = build_items(corpus, 2)
    selected = []
    for kind in ('apply', 'style', 'factqa'):
        selected.append(next(i['source_episode'] for i in candidates if i['kind'] == kind))
    session = next(s for s in small['sessions'] if s['session'] == 2)
    session['episodes'] = [e for e in session['episodes'] if e['id'] in selected]
    for e in session['episodes']:
        e['chunk_id'] = 'smoke'
    small['chunks'] = [dict(session=2, chunk_id='smoke', text='\n\n'.join(e['text'] for e in session['episodes']))]
    source = tmp_path / 'corpus.json'
    source.write_text(json.dumps(small))
    args = SimpleNamespace(corpus=source, session=2, seed=0, per_source=3, limit=None,
                           dry_run=True, ent_max=2., topk=32, max_new=48, gen_batch=8,
                           out=tmp_path/'s2.json', pt=None)
    result = run(args)
    assert 'selfstudy ' in capsys.readouterr().out
    result = json.loads(args.out.read_text())
    assert {'session', 'seed', 'ent_max', 'per_source', 'null_thr', 'null_pcts', 'chunk_tokens', 'items', 'stats'} <= result.keys()
    assert result['null_thr'] == result['null_pcts']['95'] > 0
    assert all(result['stats']['accepted'][kind] >= 1 for kind in ('apply', 'style', 'factqa', 'modelq'))
    records = torch.load(args.out.with_suffix('.pt'), weights_only=True)
    assert set(records) == {i['id'] for i in result['items'] if i['accepted']}
    for i in result['items']:
        assert {'id', 'kind', 'chunk_id', 'source_episode', 'claim_hash', 'prompt', 'answer', 'prompt_ids',
                'answer_ids', 'verifier', 'accepted', 'entropy_mean'} <= i.keys()
        assert isinstance(i['accepted'], bool)
        if i['accepted']:
            rec = TopKRecord.from_state_dict(records[i['id']])
            assert rec.ids.shape == (len(i['answer_ids']), 32)
    assert result['stats']['rejected']['duplicate'] >= 1
    with fake_helpers():
        tok = FakeTokenizer()
        # context unit = source episode: every accepted item's context is its own episode's text
        ep_text = {e['id']: e['text'] for e in session['episodes']}
        for i in result['items']:
            assert i['chunk_id'] == i['source_episode']
            assert result['chunk_tokens'][i['chunk_id']] == gpu.enc(tok, ep_text[i['chunk_id']]) + gpu.enc(tok, '\n\n')
        model = FakeModel(tok, 0)
        prefix = [gpu.BOS] + gpu.enc(tok, 'Prompt:')
        model.script(prefix, ' yes')
        assert gpu.score_options(model, tok, prefix, ['yes', 'no'], 'cpu')[0] > -1
    args.limit = 0
    empty = run(args)
    assert empty['items'] == [] and empty['null_thr'] is None


def test_style_unknown_action_and_inconsistent_status():
    assert verify({'kind': 'style'}, 'STATUS: PENDING | ACTION: arbitrary | NOTE: Needed.', 0, 2) == 'shape_failed'
    assert verify({'kind': 'style'}, 'STATUS: RESOLVED | ACTION: request photo evidence | NOTE: Needed.', 0, 2) == 'shape_failed'


def test_answer_id_alignment():
    from selfstudy import aligned_answer_ids
    tok = FakeTokenizer()
    ids = gpu.enc(tok, ' hello')
    assert aligned_answer_ids(tok, ' hello', ids) is ids
    assert aligned_answer_ids(tok, ' hello', [999]) == ids


def test_lazy_harness_interface(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(gpu, '_harness', lambda: sentinel)
    assert gpu.fsa is sentinel
    with pytest.raises(AttributeError):
        _ = gpu.missing_attribute


@pytest.mark.parametrize('tokens,pieces,expected,ids', [
    ([1], {1: 'hello\ntrailing'}, 'hello', [4]),
    ([1, 2], {1: 'hello', 2: '\n'}, 'hello', [1]),
    ([1, 3], {1: 'hello', 3: '\nEOS'}, 'hello', [1]),
])
def test_generation_stop_alignment_and_eos(monkeypatch, tokens, pieces, expected, ids):
    class Tok:
        def decode(self, output):
            assert 3 not in output, 'EOS must be removed before textual processing'
            return ''.join(pieces[t] for t in output)
        def encode(self, text, add_special_tokens=False):
            assert text == 'hello'
            return SimpleNamespace(ids=ids)
    def logits(model, x):
        out = torch.zeros(x.shape[0], x.shape[1], 5)
        out[:, -1, tokens[x.shape[1] - 1]] = 10
        return out
    monkeypatch.setattr(gpu, 'EOS', 3)
    monkeypatch.setattr(gpu, 'VOCAB', 5)
    monkeypatch.setattr(gpu, 'logits_of', logits)
    assert gpu.generate(None, Tok(), [[0]], 'cpu', max_new=3) == [(expected, ids)]


def test_topk_compresses_each_microbatch(monkeypatch):
    events = []
    original = TopKRecord.from_logits
    def lps(model, rows, starts, dev, micro_tokens):
        events.append(('full', len(rows)))
        return [torch.randn(len(row) - start, 7).log_softmax(-1) for row, start in zip(rows, starts)]
    def compress(cls, logits, k):
        events.append(('compressed', logits.shape[0]))
        return original(logits, k)
    monkeypatch.setattr(gpu, 'logprobs_at', lps)
    monkeypatch.setattr(TopKRecord, 'from_logits', classmethod(compress))
    recs = gpu.topk_records(None, [[0, 1], [0, 1, 2], [0, 1]], [1, 1, 1], 'cpu', k=3, micro_tokens=3)
    assert [r.ids.shape[0] for r in recs] == [1, 2, 1]
    assert [kind for kind, _ in events] == ['full', 'compressed'] * 3
    assert all(r.logp.device.type == 'cpu' for r in recs)
