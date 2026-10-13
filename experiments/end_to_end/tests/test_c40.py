"""C40 relation-world reconstruction and selector-served negative regressions."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T
import branch_eval as B
import decisive_eval as C
import two_hop as H


@pytest.mark.parametrize('relation_world', [None, False, True])
def test_two_hop_main_rebuilds_training_world_and_pointer_gold(monkeypatch, relation_world):
    meta = dict(seed=0, people=3, facts_per_person=8, shared_attrs=8, values='disjoint')
    if relation_world is not None:
        meta['relation_world'] = relation_world
    trained = E.make_world(0, 3, 8, 8, 'disjoint', bool(relation_world))
    monkeypatch.setattr(H, 'parse_args', lambda argv: SimpleNamespace(
        positive_control=False, patches=['bank.pt'], export='test', export_step=0))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: dict(meta=meta))
    monkeypatch.setattr(B, 'validate_patches', lambda states, **kwargs: {})
    monkeypatch.setattr(E, 'validate_patch_export', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    bench = ModuleType('ops.eval.bench_saves')
    bench.build_model = lambda *args, **kwargs: pytest.fail('model load reached')
    tokenizer = ModuleType('tokenizers')
    tokenizer.Tokenizer = SimpleNamespace(from_file=lambda path: TOK)
    monkeypatch.setitem(sys.modules, 'ops.eval.bench_saves', bench)
    monkeypatch.setitem(sys.modules, 'tokenizers', tokenizer)
    original = H.pointer_items

    class CheckedWorld(Exception):
        pass

    def check_items(tok, facts, relations, templates, bos):
        assert facts == trained
        assert relations == {0: 1, 1: 2, 2: 0}
        items = original(tok, facts, relations, templates, BOS_T)
        assert items
        for it in items:
            assert facts[it['fi']].pid == it['target']
            if relation_world:
                sister = next(f for f in trained if f.pid == it['source'] and f.attr == "sister's name")
                assert facts[it['fi']].name == sister.val
        raise CheckedWorld

    monkeypatch.setattr(H, 'pointer_items', check_items)
    with pytest.raises(CheckedWorld):
        H.main([])


def test_shared_sister_relations_use_values_and_legacy_keeps_external_cycle():
    facts = E.make_world(0, 3, 8, 8, relations=True)
    by_name = {f.name: f.pid for f in facts}
    stored = {f.pid: by_name[f.val] for f in facts if f.attr == "sister's name"}
    assert B.sister_relations(facts, True) == stored
    assert B.sister_relations(E.make_world(0, 3, 8, 8)) == {0: 1, 1: 2, 2: 0}


@pytest.mark.parametrize('corruption', ['missing', 'unknown', 'wrong_target', 'duplicate'])
def test_relation_world_rejects_inconsistent_stored_sisters(corruption):
    facts = E.make_world(0, 3, 8, 8, relations=True)
    fi = next(i for i, f in enumerate(facts) if f.attr == "sister's name")
    if corruption == 'missing':
        del facts[fi]
    elif corruption == 'duplicate':
        facts.append(facts[fi])
    else:
        facts[fi] = facts[fi]._replace(val='Nobody' if corruption == 'unknown' else facts[fi].name)
    with pytest.raises(AssertionError, match='sister'):
        B.sister_relations(facts, True)


class ConfidentModel(nn.Module):
    def forward(self, ids):
        logits = torch.zeros((*ids.shape, 128))
        logits[..., ord('v')] = torch.log(torch.tensor(9. * 127.))
        return logits


def negative_fixture():
    facts = [E.Fact(0, 'Alice', 'her', 'she', 'color', 'v'),
             E.Fact(1, 'Beth', 'her', 'she', 'instrument', 'cello')]
    states = {pid: dict(meta=dict(train_person=pid, patch_scale_mult=.5)) for pid in (0, 1)}
    return facts, states


def test_suite_served_negatives_use_locked_selector_and_count_base_abstentions(monkeypatch):
    facts, states = negative_fixture()
    model = ConfidentModel()
    active, calls, generated = [], [], []
    branches = SimpleNamespace(model=model, states=states, activate=active.append)

    def select(items):
        # Route to the other bank, proving serving does not force the item's bank.
        routes = [() if i % 3 == 0 else (1 - it['bank'],) for i, it in enumerate(items)]
        calls.append((items, routes))
        return routes

    def greedy(model, prompts, dev, limit, bs):
        generated.append((prompts[0], active[-1]))
        return [TOK.encode('v' if active[-1] else 'unknown').ids]

    # Isolate negative serving while exercising the production suite wiring.
    monkeypatch.setattr(C, 'task_items', lambda *args: {})
    monkeypatch.setattr(C, 'rescore_two_fact', lambda *args: {})
    monkeypatch.setattr(C, 'format_report', lambda *args: {})
    monkeypatch.setattr(E, 'greedy_generate', greedy)
    a = SimpleNamespace(qa_n=1, gen_stop_k=4)
    report = C.suite(branches, TOK, facts, BOS_T, 'cpu', a, select, compose=False)['negatives']
    assert 'locked suite selector' in report['served_policy']
    assert len(calls) == len(report['types']) == 4
    offset = 0
    for panel, (items, routes) in zip(report['types'].values(), calls):
        served = panel['served']
        n = len(items)
        abstentions = sum(not route for route in routes)
        assert served['n'] == panel['base']['n'] == panel['patched']['n'] == n
        assert served['abstentions'] == abstentions
        assert served['abstention_rate'] == pytest.approx(abstentions / n)
        for metric in ('false_recall', 'confidently_wrong', 'asserted_answer'):
            assert panel['base'][metric] == 0
            assert panel['patched'][metric] == 1
            assert served[metric] == pytest.approx(1 - abstentions / n)
        assert [row['route'] for row in served['per_item']] == routes
        assert [row['abstained'] for row in served['per_item']] == [not route for route in routes]
        assert generated[offset + 2*n:offset + 3*n] == [(it['p'], route) for it, route in zip(items, routes)]
        offset += 3*n
    assert active[-1] == ()

    # The added condition leaves every existing forced/base field unchanged.
    def generate(items, routes):
        return [TOK.encode('v' if route else 'unknown').ids for route in routes]
    resident = C.negative_report(model, TOK, facts, states, BOS_T, generate, active.append, 'cpu', 1)
    assert report['policy'] == resident['policy']
    for kind in report['types']:
        for mode in ('base', 'patched'):
            assert report['types'][kind][mode] == resident['types'][kind][mode]


def test_resident_served_negatives_activate_all_banks():
    facts, states = negative_fixture()
    active = []

    def generate(items, routes):
        return [TOK.encode('v' if tuple(route) == (0, 1) else 'unknown').ids for route in routes]

    report = C.negative_report(ConfidentModel(), TOK, facts, states, BOS_T, generate, active.append, 'cpu', 1)
    assert 'resident: all banks resident' in report['served_policy']
    for panel in report['types'].values():
        assert panel['patched']['false_recall'] == 0
        assert panel['served']['false_recall'] == panel['served']['confidently_wrong'] == 1
        assert panel['served']['asserted_answer'] == 1
        assert panel['served']['abstentions'] == panel['served']['abstention_rate'] == 0
        assert all(row['route'] == (0, 1) for row in panel['served']['per_item'])
    assert active[-1] == ()
