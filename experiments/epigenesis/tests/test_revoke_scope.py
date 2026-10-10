"""CPU-only ownership, construction, and generated-recall contracts."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import revoke_scope as r
import make_user as u


def test_ownership_sign_flip_zeroing_new_slot_and_last_writer():
    a = torch.tensor([1, -1, 0, 1, 0], dtype=torch.int8)
    b = torch.tensor([-1, 0, 1, 1, 0], dtype=torch.int8)
    masks = r.ownership_masks(a, b)
    assert {key: mask.tolist() for key, mask in masks.items()} == {
        'A_slots': [True, True, False, True, False],
        'B_writes': [True, True, True, False, False],
        'anti_A': [True, True, False, False, False],
        'overlap': [True, False, False, False, False],
        'A_last': [False, False, False, True, False],
    }
    ledger = r.ownership_ledger({'x': dict(committed=a)}, {'x': dict(committed=b)})
    assert ledger['totals'] == dict(A_slots=3, B_writes=3, anti_A=2, overlap=1,
                                   A_last=1, anti_A_fraction_of_B_writes=2 / 3)
    assert all(mask.device.type == 'cpu' for mask in masks.values())


def test_empty_ledger_fraction():
    state = {'x': dict(committed=torch.zeros(3, dtype=torch.int8))}
    assert r.ownership_ledger(state, state)['totals']['anti_A_fraction_of_B_writes'] == 0


def test_ownership_revoke_preserves_overwrite_and_birth():
    a, b = torch.tensor([1, -1, 0, 1]), torch.tensor([-1, 0, 1, 1])
    assert r.revoke_mask(a, b).tolist() == [False, False, False, True]
    assert r.revoke_mask(a, b, False).tolist() == [True, True, False, True]
    revoked = b.clone()
    revoked[r.revoke_mask(a, b)] = 0
    assert revoked.tolist() == [-1, 0, 1, 0]


@pytest.mark.parametrize('ownership', [True, False])
def test_revoke_block_scales_and_input_immutability(ownership):
    def state(values, scales):
        value = torch.tensor([values], dtype=torch.int8)
        return {'x': dict(committed=value, support=value != 0,
                          frozen=torch.zeros_like(value, dtype=torch.bool), logb=torch.tensor([scales]))}
    base = state([0] * 8, [0., 0., 0., 0.])
    a = state([1, 0, 1, 0, 0, 0, 1, 0], [1., 1., 1., 1.])
    b = state([-1, 0, 1, 0, 1, 0, 0, 0], [2., 2., 2., 2.])
    result = r.revoke_state(a, b, base, {'x': 2}, ownership)['x']
    assert result['committed'].tolist() == [[-1 if ownership else 0, 0, 0, 0, 1, 0, 0, 0]]
    assert result['logb'].tolist() == [[2. if ownership else 0., 0., 2., 0.]]
    assert not result['support'][0, 2]
    assert b['x']['committed'][0, 2] == 1 and b['x']['logb'][0, 1] == 2


@pytest.mark.parametrize('seed', [0, 7, 20261009])
def test_data_construction(seed):
    data = r.build_data(seed)
    assert data == r.build_data(seed)
    a, b = data['A'], data['B']
    assert [p['id'] for p in a['policies']] == [f'p{i:02d}' for i in range(20)]
    assert [f['id'] for f in a['facts']] == [f'f{i:03d}' for i in range(20)]
    assert sorted({p['category'] for p in a['policies']}) == sorted(u.CATEGORIES[:5])
    assert all(sum(p['category'] == category for p in a['policies']) == 4 for category in u.CATEGORIES[:5])
    assert len(b['policies']) == 10 and len(b['facts']) == 20
    for category in u.CATEGORIES[:5]:
        actions = [p['action'] for p in b['policies'] if p['category'] == category]
        assert len(actions) == len(set(actions))
    for before, after in zip(data['replaced_policies'], b['policies']):
        assert before['codename'] == after['codename'] and before['condition'] == after['condition']
        assert before['action'] != after['action']
        assert after['action'] in u.ACTIONS and after['status'] == u.STATUS_MAP[after['action']]
        assert after['action'] not in {p['action'] for p in data['corpus']['policies'] if p['category'] == after['category']}
        ep = next(ep for ep in b['episodes'] if ep.get('policy_id') == after['id'])
        assert f"Update: from now on, {after['rule_text']}" in ep['text']
        assert u.resolution(after) in ep['text'] and u.resolution(before) not in ep['text']
    for before, after in zip(data['replaced_facts'], b['facts'][:10]):
        assert (before['subject'], before['relation']) == (after['subject'], after['relation'])
        assert after['value'] not in {f['value'] for f in data['corpus']['facts']}
        assert after['qa']['answer'] == after['value']
        ep = next(ep for ep in b['episodes'] if after['id'] in ep.get('fact_ids', []))
        assert 'Correction: the ' in ep['text'] and after['value'] in ep['text']
        assert before['value'] not in ep['text']
    subjects = lambda facts: {f['subject'].removesuffix(' ' + f['relation']) for f in facts}
    assert subjects(a['facts']).isdisjoint(subjects(data['new_facts']))
    assert len(subjects(data['new_facts'])) == 10


def test_items_and_disjoint_dev_cases():
    data = r.build_data(3)
    for label in ('A', 'B'):
        items = r.candidates(data, label, 3)
        assert len([i for i in items if i['kind'] == 'apply']) == (40 if label == 'A' else 20)
        assert len([i for i in items if i['kind'] == 'factqa']) == 20
        smoke = r.candidates(data, label, 3, smoke=True)
        assert sum(i['kind'] == 'apply' for i in smoke) == 5
        assert sum(i['kind'] == 'factqa' for i in smoke) == 5
        assert len({i['id'] for i in items}) == len(items)
        for item in items:
            if item['kind'] == 'apply':
                assert any(name in item['prompt'] for name in u.NAME_POOLS['selfstudy'])
    probes = r.make_probes(data, 3)
    assert len(probes) == 50
    for probe in probes:
        if probe['kind'] == 'apply':
            assert any(name in probe['prompt'] for name in u.NAME_POOLS['dev'])
            assert not any(name in probe['prompt'] for name in u.NAME_POOLS['selfstudy'])
        assert bool(probe['old_truth']) == probe['group'].startswith('R2_')


@pytest.mark.parametrize('kind,text,truth,hit', [
    ('apply', 'STATUS: PENDING | ACTION: request photo evidence | NOTE: Check.', 'request photo evidence', True),
    ('apply', 'ACTION: REQUEST PHOTO EVIDENCE', 'request photo evidence', True),
    ('apply', 'ACTION: request photo evidence extra | NOTE: Check.', 'request photo evidence', False),
    ('apply', 'request photo evidence', 'request photo evidence', False),
    ('factqa', 'It is Zevran 8abc.', 'Zevran 8abc', True),
    ('factqa', 'ZEVRAN 8ABC', 'Zevran 8abc', True),
    ('factqa', 'Zevran 8abd', 'Zevran 8abc', False),
])
def test_generated_recall(kind, text, truth, hit):
    assert r.recall_check(kind, text, truth) is hit


def test_old_answer_leakage_is_independent_of_recall():
    text = 'ACTION: offer a desk credit | NOTE: Previously approve a full refund.'
    assert r.recall_check('apply', text, 'offer a desk credit')
    assert r.leakage_check(text, 'approve a full refund')
    assert not r.leakage_check(text, 'request photo evidence')
