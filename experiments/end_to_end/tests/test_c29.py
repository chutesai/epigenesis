"""CPU regressions for C29 indexing, joint inference and frozen-bank integration."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, TOK, Tiny, cpu_harness  # noqa: F401
from test_c23 import OffsetTok
import branch_eval as B
import two_hop as H
import indexed_composition as C


RELATIONS = {0: 1, 1: 2, 2: 0}


def facts():
    return [E.Fact(p, name, pos, subj, attr, value)
            for p, (name, pos, subj) in enumerate(E.PEOPLE[:3])
            for attr, value in [('instrument', ['viola', 'harp', 'flute'][p]),
                                ('tram stop', ['North', 'East', 'West'][p]),
                                ('job', ['doctor', 'painter', 'writer'][p])]]


def test_inverse_many_to_one_conjunction_missing_and_provenance_revocation():
    keys = [(8, 7, 'plays', 'viola'), (9, 3, 'plays', 'viola'),
            (10, 7, 'stop', 'North'), (11, 7, 'plays', 'viola')]
    index = C.InverseIndex(keys)
    assert index.lookup('plays', 'viola') == (3, 7)
    assert index.conjunction([('plays', 'viola'), ('stop', 'North')]) == (7,)
    assert index.conjunction([('plays', 'viola'), ('stop', 'missing')]) == ()
    assert index.revoke([8]).lookup('plays', 'viola') == (3, 7)
    assert index.revoke([8, 11]).lookup('plays', 'viola') == (3,)
    clean = C.InverseIndex(k for k in keys if k[0] not in {8, 11})
    assert clean.postings == index.revoke([8, 11]).postings
    assert clean.storage_bytes() == index.revoke([8, 11]).storage_bytes()
    assert index.keys == tuple(keys)


def test_items_deterministic_disjoint_templates_and_balanced_true_ambiguity():
    world, tok = facts(), OffsetTok()
    index = C.InverseIndex.from_facts(world)
    panels = {}
    for split in ('train', 'test'):
        comp = C.composition_items(tok, world, 1, split)
        pronouns = C.hypothesis_items(tok, world, RELATIONS, 1, split)
        panels[split] = comp+pronouns
        assert comp == C.composition_items(tok, world, 1, split)
        assert {it['kind'] for it in comp} == {'retrieval_two_values', 'intersection'}
        assert any(not it['targets'] for it in comp if it['kind'] == 'intersection')
        for it in comp:
            if it['kind'] == 'retrieval_two_values':
                assert all(value not in it['question'] for value in it['vals'])
                assert it['sources'] == index.lookup(*it['constraints'][0])
            else:
                assert it['targets'] == index.conjunction(it['constraints'])
        grouped = {}
        for it in pronouns:
            assert len(it['slots']) == 2
            assert all(world[slot['pid']*3].pos == 'her' for slot in it['slots'])
            assert all(value not in it['question'] for value in it['vals'])
            if it['ambiguous']:
                grouped.setdefault(it['question'], []).append(it)
        assert grouped and all({it['source'] for it in group} == {0, 2} for group in grouped.values())
        assert all(len(group) == 2 for group in grouped.values())
    assert {it['question'] for it in panels['train']}.isdisjoint(it['question'] for it in panels['test'])


def test_joint_paths_do_not_invent_independent_assignments_and_resolver_posterior_is_real():
    world = facts()
    it = next(it for it in C.hypothesis_items(OffsetTok(), world, RELATIONS, 1, 'test') if it['hop'])
    h, cues, legal = C.evidence(it, (0, 1, 2), C.InverseIndex.from_facts(world), RELATIONS, [1., 0.])
    retained = {(hyp.source, hyp.target) for hyp, ok in zip(h, legal) if ok}
    assert retained == {(0, 1), (2, 0)}
    retrievers = C.JointRetrievers(torch.zeros(10, 4), torch.zeros(4, 4), torch.zeros(4))
    p = C.posterior(retrievers.scores(cues, legal)['indexed'])
    assert int((p > 0).sum()) == 2
    expected = torch.tensor([1., 0.]).softmax(0).double()
    by_source = torch.stack([sum((pi for pi, hyp in zip(p, h) if hyp.source == s), p.new_zeros(()))
                             for s in [slot['pid'] for slot in it['slots']]])
    torch.testing.assert_close(by_source, expected)
    for beam in (2, 4, 8):
        torch.testing.assert_close(C.posterior(retrievers.scores(cues, legal)['indexed'], beam), p)
    assert C.posterior(torch.zeros(10), 4).nonzero().flatten().tolist() == [0, 1, 2, 3]
    for bad in (torch.tensor([float('nan')]), torch.tensor([-torch.inf]), torch.tensor([torch.inf])):
        with pytest.raises(ValueError):
            C.posterior(bad)


def test_answer_marginalization_uses_probabilities_and_sequence_logsumexp():
    # Strongest referent prefers A, but combining the banks prefers B.
    lp = torch.tensor([[.51, .49], [.01, .99]], dtype=torch.double).log()
    weights = torch.tensor([.6, .4], dtype=torch.double)
    ap = C.marginal_answers(weights, lp)
    torch.testing.assert_close(ap, torch.tensor([.31, .69], dtype=torch.double))
    assert int(lp[0].argmax()) == 0 and int(ap.argmax()) == 1
    assert not torch.allclose(ap, (weights[:, None]*lp).sum(0).softmax(0))
    with pytest.raises(ValueError):
        C.marginal_answers([1., 1.], lp)


def test_train_only_learned_joint_retrievers_determinism_and_rng():
    train = C.composition_items(OffsetTok(), facts(), 1, 'train')
    idx = C.InverseIndex.from_facts(facts())
    rows = [C.evidence(it, (0, 1, 2), idx, RELATIONS) for it in train]
    x, legal = torch.stack([r[1] for r in rows]), torch.stack([r[2] for r in rows])
    targets = torch.stack([C._targets(it, r[0]) for it, r in zip(train, rows)])
    rng = torch.get_rng_state()
    learned = C.JointRetrievers.fit(x, legal, targets, 60)
    assert torch.equal(rng, torch.get_rng_state())
    again = C.JointRetrievers.fit(x, legal, targets, 60)
    torch.testing.assert_close(learned.keys, again.keys)
    scores = learned.scores(x, legal)
    assert set(scores) == set(C.BASELINES)
    assert all(torch.isfinite(s[legal]).all() for s in scores.values())
    assert learned.keys.abs().sum() > 0 and learned.hidden.abs().sum() > 0 and learned.out.abs().sum() > 0
    assert learned.storage_bytes() == again.storage_bytes()


class FakeBanks:
    """Deterministic likelihood backend: no gold fields are read."""
    owners = (0, 1, 2)
    names = {0: 'Tamsin Vey', 1: 'Corwin Ashdown', 2: 'Ilsabet Marrow'}
    def __init__(self):
        self.generations, self.forwards, self.mixed, self.closed = 0, 0, 0, False
    def generate(self, item, pid):
        self.generations += 1
        return [0 if pid is None else pid+1]
    def logprobs(self, item, pid, answers, weights=None):
        self.forwards += len(answers)
        if weights is not None:
            self.mixed += len(answers)
            assert torch.isclose(weights.sum(), weights.new_tensor(1.))
            # Actual mixture backend is tested on projection weights below.
            return torch.tensor([-.5, -.7, -.8, -.6], dtype=torch.double)
        return torch.tensor([-.1 if a == [0 if pid is None else pid+1] else -3.
                             for a in answers], dtype=torch.double)
    def text(self, answer):
        return {0: 'No one', 1: 'viola; North; Tamsin Vey',
                2: 'harp; East; Corwin Ashdown', 3: 'flute; West; Ilsabet Marrow'}[answer[0]]
    def close(self):
        self.closed = True


def test_fixed_work_all_baselines_readouts_and_duplicate_deduplication():
    idx = C.InverseIndex.from_facts(facts())
    comp = C.composition_items(OffsetTok(), facts(), 1, 'test')
    pron = C.hypothesis_items(OffsetTok(), facts(), RELATIONS, 1, 'test')
    items = [comp[0], next(it for it in comp if it['kind'] == 'intersection'),
             next(it for it in comp if it['kind'] == 'intersection' and not it['targets']),
             next(it for it in pron if it['ambiguous'])]
    rows = [C.evidence(it, (0, 1, 2), idx, RELATIONS, [0., 0.] if it.get('slots') else None) for it in items]
    learned = C.JointRetrievers.fit(torch.stack([r[1] for r in rows]), torch.stack([r[2] for r in rows]),
                                   torch.stack([C._targets(it, r[0]) for it, r in zip(items, rows)]), 5)
    banks = FakeBanks()
    result = C.evaluate_suite(banks, idx, learned, items, rows, 1234)
    assert banks.closed
    assert len(result['subarms']) == len(C.BASELINES)*len(C.READOUTS)
    resources = [row['resources'] for row in result['subarms'].values()]
    assert all(r == resources[0] for r in resources)
    work = resources[0]['retrieval_work']
    assert work['bank_generations'] == banks.generations == 4*len(items)
    assert work['bank_answer_forwards'] == 16*len(items)
    assert work['amplitude_answer_forwards'] == banks.mixed == 3*len(C.BASELINES)*4*len(items)
    assert banks.forwards == work['bank_answer_forwards']+work['amplitude_answer_forwards']
    assert result['subarms']['indexed:early']['by_kind']['intersection']['accuracy'] == 1
    assert result['subarms']['indexed:oracle_value']['overall']['accuracy'] == 1
    json.dumps(result, allow_nan=False)
    # Duplicate candidate slots must not multiply their probability mass.
    banks = FakeBanks()
    banks.generate = lambda item, pid: [1]
    banks.logprobs = lambda item, pid, answers, weights=None: torch.zeros(len(answers), dtype=torch.double)
    duplicate = C.evaluate_suite(banks, idx, learned, items[:1], rows[:1])
    assert duplicate['subarms']['sum:posterior']['predictions'][0]['confidence'] == 1.


def test_pass_bar_boundaries_best_single_and_confident_errors():
    def row(acc, errors):
        return dict(hypothesis_questions=dict(accuracy=acc), ambiguous=dict(n=10, accuracy=acc, confident_errors=errors))
    reports = {'sum:early': row(.5, 2), 'product:early': row(.55, 1), 'sum:posterior': row(.6, 1)}
    assert C.hypothesis_gate(reports)['passed']
    reports['sum:posterior'] = row(.6, 2)
    assert not C.hypothesis_gate(reports)['passed']
    reports['sum:posterior'] = row(.599, 0)
    assert not C.hypothesis_gate(reports)['passed']


def test_failure_closes_backend_and_input_validation():
    idx = C.InverseIndex.from_facts(facts())
    it = C.composition_items(OffsetTok(), facts(), 1, 'test')[0]
    r = C.evidence(it, (0, 1, 2), idx, RELATIONS)
    retrievers = C.JointRetrievers(torch.zeros(10, 4), torch.zeros(4, 4), torch.zeros(4))
    banks = FakeBanks()
    banks.logprobs = lambda *args: (_ for _ in ()).throw(RuntimeError('failure'))
    with pytest.raises(RuntimeError):
        C.evaluate_suite(banks, idx, retrievers, [it], [r])
    assert banks.closed
    with pytest.raises(ValueError):
        C.evaluate_suite(FakeBanks(), idx, retrievers, [it], [])


def test_inference_answers_do_not_read_held_out_gold():
    index = C.InverseIndex.from_facts(facts())
    item = next(it for it in C.hypothesis_items(OffsetTok(), facts(), RELATIONS, 1, 'test')
                if it['ambiguous'] and not it['hop'])
    row = C.evidence(item, (0, 1, 2), index, RELATIONS, [0., 0.])
    learned = C.JointRetrievers(torch.zeros(10, 4), torch.zeros(4, 4), torch.zeros(4))
    first = C.evaluate_suite(FakeBanks(), index, learned, [item], [row])
    alternate = 2 if item['source'] == 0 else 0
    changed = dict(item, source=alternate, target=alternate, vals=('Never in the answer pool',),
                   answer='Hidden gold replacement')
    second = C.evaluate_suite(FakeBanks(), index, learned, [changed], [row])
    for name in first['subarms']:
        if name.endswith((':oracle_id', ':oracle_value')):
            continue
        a, b = first['subarms'][name]['predictions'][0], second['subarms'][name]['predictions'][0]
        assert a['answer'] == b['answer']
        assert a['confidence'] == b['confidence']
        assert a['joint_hypotheses'] == b['joint_hypotheses']


def test_amplitude_one_forward_sqrt_bank_deltas_and_failure_cleanup(monkeypatch):
    from torch import nn
    from torch.nn.utils import parametrize
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    base = torch.tensor([[1., 0.], [0., 2.]])
    with torch.no_grad():
        model[0].weight.copy_(base)
    pm = E.FSAParam(base, base == 0)
    parametrize.register_parametrization(model[0], 'weight', pm)
    def state(idx, value):
        return dict(entries={'0': dict(kind='fsa', fmt='ternary', scale='row', shape=[2, 2],
            idx=torch.tensor([idx], dtype=torch.int32), val=torch.tensor([value], dtype=torch.int8),
            alpha=torch.tensor([1., 2.]), fp4_step=.25)})
    banks = B.Branches(model, {0: state(1, 1), 1: state(2, -1)}, {0: 1, 1: 1}, 'sum_clip')
    original_states = {p: v['entries']['0']['val'].clone() for p, v in banks.states.items()}
    calls = []
    hook = model[0].register_forward_hook(lambda *args: calls.append(True))
    weights = torch.tensor([0., .25, .75], dtype=torch.double)
    enabled = E.FSAParam.enabled
    with C.amplitude_banks(banks, weights):
        out = model(torch.eye(2))
        expected = base+torch.tensor([[0., .5], [-2*math.sqrt(.75), 0.]])
        torch.testing.assert_close(out, expected.T)
    assert len(calls) == 1 and not pm._forward_hooks
    assert torch.equal(model[0].weight, base)
    with C.amplitude_banks(banks, torch.tensor([1., 0., 0.], dtype=torch.double)):
        assert torch.equal(model[0].weight, base)
    with pytest.raises(RuntimeError):
        with C.amplitude_banks(banks, weights):
            raise RuntimeError('fail')
    assert not pm._forward_hooks and torch.equal(model[0].weight, base)
    assert E.FSAParam.enabled == enabled
    assert all(torch.equal(v, banks.states[p]['entries']['0']['val']) for p, v in original_states.items())
    hook.remove()


def test_branch_c29_cpu_integration_read_only_and_revocation(cpu_harness, monkeypatch, tmp_path, capsys):
    run, paths = cpu_harness
    monkeypatch.setattr(Tiny, 'layers', property(lambda self: self.moes), raising=False)
    encode = type(TOK).encode
    def offsets(self, text, **kwargs):
        text = text.replace('—', '-')
        result = encode(self, text, **kwargs)
        return SimpleNamespace(ids=result.ids, offsets=[(i, i+1) for i in range(len(text))])
    monkeypatch.setattr(type(TOK), 'encode', offsets)
    patches = []
    for person in range(3):
        _, patch = run(['--steps', '1', '--layers', '1', '--train-person', str(person),
                        '--sparse-state', '--slot-frac', '.5'], f'c29p{person}')
        patches.append(str(patch))
    original = {p: Path(p).read_bytes() for p in patches}
    anchor = tmp_path/'long_anchor.json'
    anchor.write_text(json.dumps([40]*8000))
    output = tmp_path/'c29.json'
    fit = C.JointRetrievers.fit.__func__
    train_calls = []
    def training_only(cls, cues, legal, targets, steps=100):
        train_calls.append(len(cues))
        assert len(cues) <= 12 and legal.shape == targets.shape
        return fit(cls, cues, legal, targets, steps)
    monkeypatch.setattr(C.JointRetrievers, 'fit', classmethod(training_only))
    monkeypatch.setattr(sys, 'argv', ['branch_eval.py', '--patches', *patches, '--arms', 'C29',
        '--bio-steps', '2', '--slot-early-layer', '0', '--slot-late-layers', '1', '--c29-max-per-kind', '2',
        '--anchor', str(anchor), '--eval-set', paths['eval'], '--fineweb', paths['extra'], '--code', paths['extra'],
        '--fast-moe', 'off', '--gen-max-new', '1', '--qa-n', '2', '--g-n', '2', '--out', str(output)])
    B.main()
    result = json.loads(output.read_text())
    assert set(result['arms']) == {'C29:full'}
    report = result['arms']['C29:full']
    assert report['n_test'] <= 12
    assert train_calls == [report['n_train']]
    ambiguous = report['subarms']['indexed:early']['ambiguous']
    assert ambiguous['n'] == 4
    assert set(report['subarms']['indexed:early']['by_kind']) == {
        'retrieval_two_values', 'intersection', 'pronoun', 'multi_hop'}
    assert len(report['subarms']) == len(C.BASELINES)*len(C.READOUTS)
    assert set(report['damage']) == {'WikiText', 'FineWeb', 'code'}
    assert result['revoked']['logits_bit_exact']
    assert all(Path(p).read_bytes() == data for p, data in original.items())
    assert 'HYPOTHESIS-PASS' in capsys.readouterr().out
    json.dumps(result, allow_nan=False)
    assert H.parse_args(['--patches', patches[0], '--anchor', str(anchor), '--out', str(output), '--c29']).c29
