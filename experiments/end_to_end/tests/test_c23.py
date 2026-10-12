"""C23 CPU regressions for probe protocol, routing, pointer training and sampling."""
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_separability_probe import S, Tiny, synthetic_panels
from test_e2e_fsa import cpu_harness  # noqa: F401
import branch_eval as B
import two_hop as H

E = S.E


class OffsetTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(c) for c in text], offsets=[(i, i+1) for i in range(len(text))])

    def decode(self, ids):
        return ''.join(chr(i) for i in ids)


def test_question_training_and_legacy_mode_have_balanced_terminal_tokens():
    facts = E.make_world(0, 3, 4, 4)
    panels = S.build_panels(OffsetTok(), facts, 1)
    legacy = S.build_panels(OffsetTok(), facts, 1, question_train=False)
    assert all(it['p'][-1] != ord('?') for it in legacy['train'])
    for pid in range(3):
        assert sum(it['p'][-1] == ord('?') and facts[it['fi']].pid == pid for it in panels['train']) == 8
    assert {it['prompt'] for it in panels['train']}.isdisjoint(it['prompt'] for it in panels['test'])


def test_ridge_near_constant_features_and_high_dim_duplicate_rows():
    y = torch.arange(30) % 3
    x = F.one_hot(y, 3).double().repeat(1, 40)
    x[:, -1] = 1e-10 * y
    p = S.fit_probes(x, {'person': y}, {'person': 3})['person']
    assert p.scale.min() > .01
    test = x.clone()
    test[:, -1] = 10  # Novel phrasing activates an almost constant coordinate.
    assert p.scores(test).argmax(-1).tolist() == y.tolist()
    assert torch.isfinite(p.weight).all()
    assert (p.scores(x).argmax(-1) == y).all()


def test_none_cosine_does_not_interpret_large_ridge_extrapolation_as_confidence():
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    p = S.fit_probes(x, {'person': [0, 0, 1, 1]}, {'person': 2})['person']
    question, none = torch.tensor([[1., 0.]]), torch.tensor([[100., 1000.]])
    assert S.auroc(-p.scores(none).amax(-1), -p.scores(question).amax(-1)) == 0
    assert S.auroc(p.none_scores(none), p.none_scores(question)) == 1
    assert p.none_scores(none).abs().max() <= 1


def test_shuffled_label_mapping_and_per_template_diagnostics():
    features, labels, counts, facts = synthetic_panels()
    order = torch.randperm(len(features['test']), generator=torch.Generator().manual_seed(23))
    features['test'] = features['test'][order]
    labels['test'] = {name: [v[i] for i in order] for name, v in labels['test'].items()}
    panels = {split: [dict(ti=i % 2) for i in range(len(lab['person']))] for split, lab in labels.items()}
    row = S.evaluate_layer(2, features, labels, counts, facts, 1e-3, panels)
    for target in counts:
        assert row['targets'][target]['panels']['test']['by_template'] == {'0': 1., '1': 1.}
    assert row['targets']['person']['panels']['test']['confusion'] == [[4, 0, 0], [0, 4, 0], [0, 0, 4]]


def test_name_position_uses_full_token_offsets_not_prefix_retokenization():
    class MergeTok:
        def encode(self, text, **kwargs):
            assert text == 'Hi Jane?'
            return SimpleNamespace(ids=[4, 5, 6], offsets=[(0, 3), (3, 6), (6, 8)])
    assert S.name_position(MergeTok(), 'Hi Jane?', 'Jane', 1) == 3
    assert S.name_position(MergeTok(), 'Hi Jane?', 'Jane', None) == 2


def test_sanity_baselines_detect_person_at_name_and_chance_at_question():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(128, 3)
            with torch.no_grad():
                self.embed.weight.zero_()
                for pid, name in enumerate('ABC'):
                    self.embed.weight[ord(name), pid] = 1
            self.layers = nn.ModuleList(nn.Identity() for _ in range(3))
        def forward_hidden(self, tokens):
            x = self.embed(tokens)
            for layer in self.layers:
                x = layer(x)
            return x
    facts = [E.Fact(pid, name, 'her', 'she', 'job', 'v') for pid, name in enumerate('ABC')]
    panels = {s: [dict(fi=i, prompt=f'{f.name}?', p=[E.BOS, ord(f.name), ord('?')])
                   for i, f in enumerate(facts)] for s in ('train', 'test')}
    # BOS may exceed our tiny embedding vocabulary.
    for group in panels.values():
        for it in group:
            it['p'][0] = 1
    result = S.sanity_baselines(Model(), OffsetTok(), panels, facts, 'cpu', 1e-3)
    assert result['last_token_embedding']['test']['accuracy'] == pytest.approx(1/3)
    assert result['name_token']['test']['accuracy'] == 1


def test_indexer_centering_owner_mapping_thresholds_topk_and_abstention():
    center = torch.tensor([10., 20., 30.])
    x = torch.eye(3) + center
    index = B.BankIndexer.fit(x, [7, 2, 9], center[None])
    assert index.owners == (2, 7, 9)
    assert index.select(x) == [(7,), (2,), (9,)]
    assert index.select(center[None], .2, 3) == [()]
    query = center + torch.tensor([[1., 1., 0.]])
    assert index.select(query, .5, 3) == [(2, 7)]
    assert index.select(query, .8, 3) == [()]
    metrics = B.selection_metrics([(2, 7), (), (9,)], [(7,), (2,), (9,)])
    assert metrics['accuracy'] == pytest.approx(1/3)
    assert metrics['recall'] == pytest.approx(2/3)
    assert metrics['abstention_rate'] == pytest.approx(1/3)
    for k in (0, 4):
        with pytest.raises(ValueError):
            index.select(x, top_k=k)
    with pytest.raises(ValueError):
        index.select(x, float('nan'))


def test_indexer_uses_question_position_and_restores_disabled_state():
    tok = OffsetTok()
    seq = E.encode(tok, 'Who?\nAnswer:', 1)
    assert B.question_sequence(tok, seq, 1) == E.encode(tok, 'Who?', 1)
    E.set_enabled(False)
    result = B.indexer_features(Tiny(layers=3), [[1, 4, 9]], 2, 'cpu', [1])
    assert not E.FSAParam.enabled
    torch.testing.assert_close(result, torch.tensor([[24., 24.]], dtype=torch.double))
    E.set_enabled(True)


def test_pointer_item_target_is_sister_bank_without_answer_leak():
    facts = E.make_world(0, 3, 4, 4)
    items = H.pointer_items(OffsetTok(), facts, {0: 1, 1: 2, 2: 0}, H.TEST, 1)
    assert len(items) == 12
    for it in items:
        assert it['target'] == (it['source'] + 1) % 3 == facts[it['fi']].pid
        assert it['val'] not in it['question']
        assert it['p'][:it['position']+1] == it['q']
        assert it['q'][-1] == ord('?')


class PointerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(torch.eye(3))
        self.layers = nn.ModuleList([nn.Identity(), nn.Linear(3, 3, bias=False), nn.Identity()])
        with torch.no_grad():
            self.layers[1].weight.copy_(2 * torch.eye(3))
        self.requires_grad_(False)
    def forward_hidden(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        return x


def test_pointer_training_backprops_through_frozen_layers_and_cleans_hooks():
    model = PointerModel()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    items = [dict(q=[i], position=0, target=(i+1)%3) for i in range(3)]
    codes = {i: 2 * torch.eye(3)[i] for i in range(3)}
    torch.manual_seed(23)
    patch, losses = H.train_pointer(model, items, codes, 0, 2, 'cpu', 80, 3, .05, 23, 3)
    assert losses[-1] < losses[0] * .01
    with H.pointer_hook(model, 0, patch, [0]):
        code = H.code_forward(model, torch.tensor([[0, 2]]), 2, [0])
    assert F.cosine_similarity(code, codes[1][None]) > .99
    assert not any(layer._forward_pre_hooks for layer in model.layers)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k])
    with pytest.raises(RuntimeError):
        with H.pointer_hook(model, 0, patch, [0]):
            raise RuntimeError('failure')
    assert not model.layers[0]._forward_pre_hooks


def test_pointer_hook_changes_original_position_only_during_generation():
    model = PointerModel()
    patch = H.PointerPatch(torch.zeros(3), torch.ones(3), 2)
    with torch.no_grad():
        patch.up.bias.fill_(3)
    x = torch.tensor([[0, 1, 2]])
    base = model.forward_hidden(x)
    with H.pointer_hook(model, 0, patch, [1]):
        edited = model.forward_hidden(x)
    torch.testing.assert_close(edited[:, [0, 2]], base[:, [0, 2]])
    torch.testing.assert_close(edited[:, 1], base[:, 1] + 6)


def test_context_check_subsamples_identically_for_person_all_and_tf_gen(monkeypatch):
    facts = E.make_world(0, 3, 4, 4)
    items = E.build_items(OffsetTok(), facts, E.TEST_T, 1)
    seen = []
    monkeypatch.setattr(E, 'context_sequence', lambda tok, facts, seq, *args: seq)
    def tf(model, group, *args, **kwargs):
        seen.append([(it['fi'], it['ti']) for it in group])
        return {}
    def gen(model, tok, group, *args, **kwargs):
        tf(model, group)
        return dict(recall=1)
    monkeypatch.setattr(E, 'tf_eval', tf)
    monkeypatch.setattr(E, 'gen_eval', gen)
    a = SimpleNamespace(seed=23, ctx_check_n=5, gen_max_new=2, gen_stop_k=2)
    rng = random.getstate()
    E.context_teacher_check(None, OffsetTok(), facts, items, items, 'cpu', 1, a)
    assert random.getstate() == rng
    assert all(len(group) == 5 for group in seen)
    assert seen[0] == seen[1] == seen[4] == seen[5]
    assert seen[2] == seen[3] == seen[6] == seen[7]
    first = seen[:]
    seen.clear()
    E.context_teacher_check(None, OffsetTok(), facts, items, items, 'cpu', 1, a)
    assert seen == first
    for n in (0, 100):
        a.ctx_check_n = n
        seen.clear()
        E.context_teacher_check(None, OffsetTok(), facts, items, items, 'cpu', 1, a)
        assert all(len(group) == len(items) for group in seen)
    a.ctx_check_n = -1
    with pytest.raises(ValueError):
        E.context_teacher_check(None, OffsetTok(), facts, items, items, 'cpu', 1, a)


def test_pointer_evaluation_baselines_and_selected_bank_generation(monkeypatch):
    model = PointerModel()
    codes = {i: 2 * torch.eye(3)[i] for i in range(3)}
    indexer = B.BankIndexer.fit(torch.stack(list(codes.values())), [0, 1, 2], torch.zeros(1, 3))
    patch = H.PointerPatch(torch.zeros(3), torch.ones(3), 3)
    with torch.no_grad():
        patch.down.weight.copy_(torch.eye(3))
        patch.up.weight.copy_(torch.roll(torch.eye(3), 1, 0) - torch.eye(3))
    class Banks:
        def __init__(self):
            self.model, self.keys, self.active = model, [], ()
        def activate(self, key):
            self.active = key
            self.keys.append(key)
    banks = Banks()
    items = [dict(fi=i, ti=0, attr='x', val='v', q=[i], p=[i, 2], position=0,
                  target=(i+1)%3, a=[1]) for i in range(3)]
    def gen(model, prompts, *args, **kwargs):
        assert len(banks.active) == 1
        return [[banks.active[0]]]
    monkeypatch.setattr(E, 'greedy_generate', gen)
    monkeypatch.setattr(E, 'conflict_map', lambda _: {})
    monkeypatch.setattr(B, 'generation_summary', lambda tok, items, gens, *args:
        dict(recall=sum(g[0] == it['target'] for g, it in zip(gens, items)) / len(items)))
    monkeypatch.setattr(H,'resolved_name_item',lambda tok,facts,it,bos: dict(it,q=[it['target']],p=[it['target'],2]))
    result = H.evaluate(model, banks, patch, indexer, codes, None, [], items, 0, 2, 'cpu', 1, 2)
    assert result['no_hop1']['selection']['accuracy'] == result['no_hop1']['gen']['recall'] == 0
    for mode in ('pointer', 'oracle_bank'):
        assert result[mode]['selection']['accuracy'] == result[mode]['gen']['recall'] == 1
        assert result[mode]['hop1_code_cosine'] == pytest.approx(1)
    assert banks.active == ()
    assert not any(layer._forward_pre_hooks for layer in model.layers)


def test_branch_arm_i_cpu_integration(cpu_harness, monkeypatch, tmp_path):
    import json
    import sys
    from pathlib import Path
    from test_e2e_fsa import TOK, Tiny as HarnessTiny
    run, paths = cpu_harness
    # Read-only alias avoids changing saved module paths.
    monkeypatch.setattr(HarnessTiny, 'layers', property(lambda self: self.moes), raising=False)
    original_encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), 'encode', lambda self, text, **kw:
                        original_encode(self, text.replace('—', '-'), **kw))
    patches = []
    for person in range(3):
        _, patch = run(['--steps', '1', '--train-person', str(person), '--sparse-state', '--slot-frac', '.5'], f'i{person}')
        patches.append(str(patch))
    Path(paths['anchor']).write_text(json.dumps([40] * 7000))
    out = tmp_path / 'indexer.json'
    monkeypatch.setattr(sys, 'argv', ['branch_eval.py', '--patches', *patches, '--arms', 'I',
        '--anchor', paths['anchor'], '--eval-set', paths['eval'], '--fineweb', paths['extra'],
        '--code', paths['extra'], '--fast-moe', 'off', '--indexer-layer', '1', '--gen-max-new', '1',
        '--qa-n', '2', '--g-n', '2', '--out', str(out)])
    question_sequence = B.question_sequence
    def questions_only(tok, ids, bos):
        # General-text routing stays at its fixed last-prefix position even if
        # ordinary prose happens to contain question marks.
        assert ids[1:] != [90] * 63
        return question_sequence(tok, ids, bos)
    monkeypatch.setattr(B, 'question_sequence', questions_only)
    B.main()
    result = json.loads(out.read_text())
    assert set(result['arms']) == {'I:single','I:multibank'}
    for report in result['arms'].values():
        assert report['routing']['resident_banks'] == [0, 1, 2]
        assert report['routing']['selection']['test']['n'] == report['test_gen']['n'] == 12
        assert report['routing']['selection']['general']['n'] == 40
        assert set(report['damage']) == {'WikiText', 'FineWeb', 'code'}
    assert result['revoked']['logits_bit_exact']
