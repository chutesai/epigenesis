"""EP-1 CPU invariants. Also runnable without pytest: python tests/test_cpu.py."""

from collections import Counter
from itertools import combinations
import io
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from epi_common import (
    MIXES, ONEOFF_RE, SHAPE_RE, STATUS_VOCAB, TopKRecord, claim_hash, free_slot_mask,
    gate_weights, mix_weights, pack_rows, paired_stats, pair8_legal,
    parse_resolution, participation_ratio, topk_kl,
)
from make_user import (
    ACTIONS, CATEGORIES, NAME_POOLS, SELFSTUDY_POOL, STATUS_MAP, build_corpus,
    census, chunk_sessions, mint_case, policy_for_case, render_ticket, word_token_count,
)


def test_corpus():
    corpus = build_corpus(20261009, word_token_count)
    encode = lambda c: json.dumps(c, ensure_ascii=False, indent=2).encode()
    assert encode(corpus) == encode(build_corpus(20261009, word_token_count))
    assert encode(corpus) != encode(build_corpus(20261010, word_token_count))
    assert (len(corpus['policies']), len(corpus['facts']), len(corpus['lures'])) == (40, 100, 100)
    assert len(ACTIONS) == 14 and set(STATUS_MAP.values()) == set(STATUS_VOCAB)
    for a, b in combinations(NAME_POOLS.values(), 2):
        assert set(a).isdisjoint(b)
    assert SELFSTUDY_POOL == NAME_POOLS['selfstudy']

    facts, lures, appearances = {}, {}, {}
    sources = {}
    all_text = []
    for session in corpus['sessions']:
        episodes = session['episodes']
        assert len(episodes) == 108
        assert 23 <= sum(e['kind'].startswith('policy_') for e in episodes) <= 25
        assert sum(len(e.get('fact_ids', [])) for e in episodes) == 60
        assert sum(len(e.get('lure_ids', [])) for e in episodes) == 20
        assert 6000 <= word_token_count('\n\n'.join(e['text'] for e in episodes)) <= 8000
        for ep in episodes:
            sources[ep['id']] = ep
            all_text.append(ep['text'])
            if ep['kind'].startswith('policy_'):
                appearances.setdefault(ep['policy_id'], []).append((session['session'], ep))
            for fid in ep.get('fact_ids', []):
                facts.setdefault(fid, []).append((session['session'], ep))
            for lid in ep.get('lure_ids', []):
                lures.setdefault(lid, []).append((session['session'], ep))
            for line in ep['text'].splitlines():
                if line.startswith('Resolution: '):
                    parsed = parse_resolution(line.removeprefix('Resolution: '))
                    assert parsed and parsed['action'] in ACTIONS
                    assert parsed['status'] == STATUS_MAP[parsed['action']]
    session_texts = ['\n'.join(ep['text'] for ep in s['episodes']) for s in corpus['sessions']]
    for policy in corpus['policies']:
        occurrences = sorted(appearances[policy['id']], key=lambda pair: pair[0])
        assert [s for s, _ in occurrences] == [policy['intro_session'], *policy['recur_sessions']]
        assert [ep['kind'] for _, ep in occurrences] == ['policy_intro', 'policy_recur', 'policy_recur']
        for _, ep in occurrences[1:]:
            assert f"Here is a new application of {policy['codename']} at Fennmark Parcel Desk. {policy['rule_text']}" in ep['text']
        assert len(policy['worked_cases']) == 3
        for worked in policy['worked_cases']:
            assert policy_for_case(corpus['policies'], worked['case']) is policy
            assert worked['ticket'] == render_ticket(worked['case'])
            assert SHAPE_RE.fullmatch(worked['resolution'])
    for fact in corpus['facts']:
        assert len(fact['declarative']) == 2 and fact['qa']['answer'] == fact['value']
        assert len({s for s, _ in facts[fact['id']]}) == 3
        assert corpus['claim_schedule'][claim_hash(fact['subject'], fact['value'])] == sorted(fact['sessions'])
    for lure in corpus['lures']:
        assert ONEOFF_RE.fullmatch(lure['value'])
        assert len(lures[lure['id']]) == 1
        assert sum(lure['value'] in text for text in session_texts) == 1

    seen = []
    for chunk in corpus['chunks']:
        assert word_token_count(chunk['text']) <= 1536
        eps = [sources[eid] for eid in chunk['episode_ids']]
        assert chunk['text'] == '\n\n'.join(ep['text'] for ep in eps)
        assert all(ep['chunk_id'] == chunk['chunk_id'] for ep in eps)
        assert all(ep['id'].startswith(f"s{chunk['session']}-") for ep in eps)
        seen.extend(chunk['episode_ids'])
    assert Counter(seen) == Counter(sources.keys())
    chunk_ids = {c['chunk_id'] for c in corpus['chunks']}
    for parity, split in enumerate(('dev', 'test')):
        probes = corpus['probes'][split]
        assert [len(probes[k]) for k in ('concept', 'verbatim', 'lure')] == [100, 50, 50]
        assert all(name not in '\n'.join(all_text) for name in NAME_POOLS[split])
        for i, item in enumerate(probes['concept']):
            case = item['case']
            assert case['customer'] in NAME_POOLS[split]
            policy = policy_for_case(corpus['policies'], case)
            assert policy['id'] == item['policy_id']
            siblings = [p for p in corpus['policies'] if p['category'] == case['category']]
            expect = ({f"STATUS: {p['status']} | ACTION: {p['action']}" for p in siblings}
                      if item['family'] == 'ticket' else {p['action'] for p in siblings})
            assert set(item['options']) == expect and len(item['options']) == 4
            assert sum(policy['action'] in o for o in item['options']) == 1
            assert policy['action'] in item['options'][item['answer_idx']]
            assert item['source_chunk'] == appearances[policy['id']][0][1]['chunk_id']
            assert item['family'] == ('ticket' if (i // 10) % 2 == 0 else 'qa')
            assert render_ticket(case) in item['prompt']
        for category in CATEGORIES:
            families = Counter(i['family'] for i in probes['concept'] if i['case']['category'] == category)
            assert families == {'ticket': 5, 'qa': 5}
        for kind, table, key, sources_by_id in (
            ('verbatim', corpus['facts'], 'fact_id', facts),
            ('lure', corpus['lures'], 'lure_id', lures),
        ):
            assert [p[key] for p in probes[kind]] == [v['id'] for v in table[parity::2]]
            for i, item in enumerate(probes[kind]):
                assert item['source_chunk'] in chunk_ids
                assert item['source_chunk'] == sources_by_id[item[key]][0][1]['chunk_id']
                assert item['family'] == ('declarative-cloze' if i % 2 == 0 else 'qa')
    assert len(corpus['style_prompts']) == 20
    assert all('category desk courtesy enquiry' in prompt for prompt in corpus['style_prompts'])
    assert len(corpus['revoke_prompts']) == 32
    rng = random.Random(1)
    for category in CATEGORIES:
        for _ in range(20):
            case = mint_case(rng, corpus['policies'], SELFSTUDY_POOL, category)
            policy_for_case(corpus['policies'], case)
            assert case['customer'] in SELFSTUDY_POOL
            text = render_ticket(case)
            assert all(str(value) in text for value in case.values())


def test_chunk_counter():
    sessions = [dict(session=1, episodes=[dict(id=str(i), text='abc') for i in range(3)])]
    # Character counting ensures the packing function actually uses its injected counter.
    chunks = chunk_sessions(sessions, len, max_tokens=8)
    assert [c['episode_ids'] for c in chunks] == [['0', '1'], ['2']]
    assert all(len(c['text']) <= 8 for c in chunks)
    try:
        chunk_sessions(sessions, len, max_tokens=2)
    except AssertionError:
        pass
    else:
        raise AssertionError('Oversize episode was accepted')


def test_topk_loss():
    generator = torch.Generator().manual_seed(72)
    logits = torch.randn(12, 50, generator=generator)
    rec = TopKRecord.from_logits(logits, k=8)
    assert (rec.ids.dtype, rec.logp.dtype, rec.lse.dtype, rec.entropy.dtype) == (
        torch.int32, torch.float16, torch.float32, torch.float32)
    assert rec.ids.shape == rec.logp.shape == (12, 8)
    assert rec.lse.shape == rec.entropy.shape == (12,)
    assert torch.allclose(rec.lse, logits.logsumexp(-1))
    teacher = logits.log_softmax(-1)
    expected_mass = teacher.gather(-1, rec.ids.long()).exp().sum(-1)
    assert torch.allclose(rec.mass(), expected_mass, atol=3e-4)
    assert ((rec.mass() > 0) & (rec.mass() <= 1)).all()
    assert torch.allclose(rec.entropy, -(teacher.exp() * teacher).sum(-1), atol=1e-6)
    assert ((rec.entropy >= 0) & (rec.entropy <= math.log(50))).all()
    stream = io.BytesIO()
    torch.save(rec.state_dict(), stream)
    stream.seek(0)
    restored = TopKRecord.from_state_dict(torch.load(stream, weights_only=True)).to('cpu')
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in rec.state_dict().items())

    # tail-bucket KL: zero when the student equals the teacher; positive when the student's tail mass differs
    loss, n = topk_kl(rec, teacher)
    assert n == 12
    assert abs(loss.item()) < 1e-5
    conditional_student = teacher - expected_mass.log()[:, None]   # all mass on the top-k: tail mismatch
    conditional_loss, _ = topk_kl(rec, conditional_student)
    assert conditional_loss.item() > 0
    full_loss, _ = topk_kl(TopKRecord.from_logits(logits, k=50), teacher)
    assert abs(full_loss.item()) < 2e-3   # fp16 storage of the top-k log-probs
    shifted = logits.clone()
    shifted.scatter_add_(1, rec.ids.long(), torch.full((12, 8), -3.0))
    shifted.requires_grad_()
    shifted_loss, _ = topk_kl(rec, shifted.log_softmax(-1))
    assert shifted_loss.item() > loss.item() > 0
    shifted_loss.backward()
    assert torch.isfinite(shifted.grad).all() and shifted.grad.abs().sum() > 0
    zero_loss, n = topk_kl(rec, shifted.log_softmax(-1), torch.zeros(12))
    assert zero_loss.item() == 0 and n == 0 and zero_loss.requires_grad
    zero_loss.backward()
    weights = torch.tensor([0., 1., 2.] + [0.] * 9)
    weighted, n = topk_kl(rec, teacher, weights)
    individual = [topk_kl(TopKRecord.from_state_dict({k: v[i:i+1] for k, v in rec.state_dict().items()}),
                          teacher[i:i+1])[0] for i in (1, 2)]
    assert n == 2 and torch.allclose(weighted, (individual[0] + 2 * individual[1]) / 3)
    extreme = TopKRecord.from_logits(torch.tensor([[0., -1000., -1000.]]), k=2)
    assert torch.isfinite(extreme.entropy).all()


def test_helpers():
    assert claim_hash('  A Subject! ', ' SOME, value. ') == claim_hash('a subject', 'some value')
    assert claim_hash('ab', 'c') != claim_hash('a', 'bc')
    for value in ('AB12-Z9XY', 'TRK-12345678', 'FP-12345'):
        assert ONEOFF_RE.fullmatch(value)
    assert not ONEOFF_RE.fullmatch('TRK-12345')
    line = 'STATUS: PENDING | ACTION:  Request Photo Evidence  | NOTE: A record is needed.'
    assert parse_resolution(line) == dict(status='PENDING', action='request photo evidence', note='A record is needed.')
    for text in (line + '\n', line.replace('PENDING', 'OTHER'), line + '\nextra',
                 'STATUS: PENDING | ACTION:   | NOTE: Something.'):
        assert parse_resolution(text) is None
    kl = torch.tensor([0., 1., 2., 2., 3.])
    entropy = torch.tensor([0., 0., 1., 1.1, 1.])
    assert torch.equal(gate_weights(kl, entropy, 1., 1.), torch.tensor([0., 0., 1., 0., 1.]))
    for name in MIXES:
        for empty in (False, True):
            mix = mix_weights(name, empty)
            assert math.isclose(sum(mix), 1)
            assert mix[0] == MIXES[name][0]
            assert mix[2] == (0 if empty else MIXES[name][2])
    assert participation_ratio(torch.zeros(4)) == 0
    assert participation_ratio(torch.tensor([0., 2., 0.])) == 1
    assert participation_ratio(torch.ones(7)) == 7
    stats = paired_stats([1., 2., 4.], [0., 1., 1.])
    assert stats['n'] == 3 and stats['se'] > 0
    assert stats['ci95_lo'] <= stats['mean'] <= stats['ci95_hi']
    assert math.isclose(stats['mean'], 5 / 3)
    assert math.isclose(stats['se'], 2 / 3)
    x, mask = pack_rows([[1, 2, 3], [4, 5], []], [1, 0, 0], 'cpu')
    assert x.dtype == torch.long and mask.dtype == torch.bool
    assert x.tolist() == [[1, 2, 3], [4, 5, 0], [0, 0, 0]]
    assert mask.tolist() == [[False, True, True], [True, True, False], [False, False, False]]
    x, mask = pack_rows([], [], 'cpu')
    assert x.shape == mask.shape == (0, 0)


def test_pair8():
    generator = torch.Generator().manual_seed(19)
    blocks = torch.zeros(5, 3, 4, 2)
    for row in range(5):
        for block in range(3):
            for pair in torch.randperm(4, generator=generator)[:2]:
                blocks[row, block, pair, 0] = 1 if torch.rand((), generator=generator) > .5 else -1
    W = blocks.reshape(5, 24)
    assert pair8_legal(W)
    # Inline copy of e2e_fsa.attach's mask: importing the GPU harness is unnecessary.
    pr = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(W.shape)
    assert torch.equal(free_slot_mask(W), (W == 0) & pr)
    assert pair8_legal(W + free_slot_mask(W).float())
    inactive = (~blocks[0, 0].ne(0).any(-1)).nonzero()[0, 0]
    blocks[0, 0, inactive, 0] = 1
    assert not pair8_legal(W)
    assert not pair8_legal(torch.zeros(2, 9))


if __name__ == '__main__':
    for test in (test_corpus, test_chunk_counter, test_topk_loss, test_helpers, test_pair8):
        test()
    print('5 CPU tests passed')
    print(census(build_corpus()))
