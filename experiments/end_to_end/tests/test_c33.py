"""CPU checks for attribute-bank training and two-level selection."""
import json
import random

import pytest
import torch

from test_e2e_fsa import E, TOK, cpu_harness, _pair8_tiny_weight
from test_branch_eval import state
import branch_eval as B
import hierarchical_banks as H


def test_groups_balanced_deterministic_and_cover_person():
    facts = E.make_world(7, 3, 11, 8)
    for person in range(3):
        parts = [E.person_indices(facts, person, g, 4) for g in range(4)]
        assert sorted(sum(parts, [])) == E.person_indices(facts, person)
        assert max(map(len, parts))-min(map(len, parts)) <= 1
        shuffled = list(reversed(facts))
        for g, part in enumerate(parts):
            attrs = {facts[fi].attr for fi in part}
            assert attrs == {shuffled[fi].attr for fi in E.person_indices(shuffled, person, g, 4)}
    with pytest.raises(ValueError):
        E.person_indices(facts, None, 0, 4)
    with pytest.raises(ValueError):
        E.person_indices(facts, 0, 0, 12)


@pytest.mark.parametrize('proj', ['up_proj', 'down_proj'])
def test_twelve_unit_partitions_disjoint_cover_and_align(proj):
    mask = E.pair8_forced_mask(_pair8_tiny_weight(3, 64, 64))
    parts = [E.partition_mask(mask, 12, 2, proj, 12, part, True) for part in range(12)]
    assert torch.equal(sum(p.int() for p in parts), mask.int())
    assert all(p.any() for p in parts)
    # Each row/column belongs wholly to one bank.
    axis = 1 if proj == 'up_proj' else 0
    assert (torch.stack([p.any(axis) for p in parts]).sum(0) <= 1).all()


@pytest.mark.parametrize('flags', [
    ['--train-attr-group','0','--attr-groups','4'],
    ['--train-person','1','--train-attr-group','4','--attr-groups','4'],
    ['--train-person','1','--train-attr-group','0','--attr-groups','4','--slot-partition','12','--slot-part','0'],
    ['--train-person','1','--train-attr-group','0','--attr-groups','4','--slot-partition','3','--slot-part','1'],
    ['--attr-groups','4'],
])
def test_bad_group_arguments_fail(flags):
    a = E.build_parser().parse_args(['--out','unused','--method','fsa_forced','--facts-per-person','8',*flags])
    with pytest.raises(SystemExit):
        E.validate_c16_args(a)


def test_bank_validation_requires_complete_pairs_and_matching_partition():
    states = []
    for p in range(3):
        for g in range(4):
            s = state(p, [], [])
            s['meta'].update(train_attr_group=g, attr_groups=4, slot_partition=12, slot_part=p*4+g, partition_unit=True)
            states.append(s)
    assert set(B.validate_patches(states, banked=True)) == set(range(12))
    with pytest.raises(ValueError, match='every'):
        B.validate_patches(states[:-1], banked=True)
    states[0]['meta']['slot_part'] = 1
    with pytest.raises(ValueError, match='partition'):
        B.validate_patches(states, banked=True)
    states[0]['meta']['slot_part'] = 0
    states[1]['meta']['train_attr_group'] = 0
    with pytest.raises(ValueError, match='every'):
        B.validate_patches(states, banked=True)


@pytest.mark.parametrize('arm', ['H','FLAT'])
def test_separable_positive_controls_and_nulls(arm):
    assert B.positive_control(arm)['passed']


def test_global_beam_uses_combined_scores_and_counts_only_visited_children():
    # A strong parent / weaker child competes with a weaker parent / strong child.
    parent = B.BankIndexer(torch.eye(3, dtype=torch.double), torch.zeros(3), [0,1,2])
    children = {p: B.BankIndexer(torch.eye(4, dtype=torch.double), torch.zeros(4), list(range(p*4,p*4+4))) for p in range(3)}
    hierarchy = B.HierarchicalIndexer(parent, children, 4)
    x1, x2 = torch.tensor([[.8,.6,0.]]), torch.tensor([[.6,.8,0.,0.]])
    trace = hierarchy.select(x1, x2, .1, .1, 2, 2, 2, trace=True)
    assert trace['routes'] == [(0,1)]
    assert trace['comparisons'] == [3+2*4]
    trace = hierarchy.select(x1, x2, .7, .1, 2, 2, 2, trace=True)
    assert trace['comparisons'] == [3+4]
    trace = hierarchy.select(torch.zeros_like(x1), x2, .1, .1, 2, 2, 2, trace=True)
    assert trace['routes'] == [()] and trace['comparisons'] == [3]
    # Good parent and null child must still abstain.
    assert hierarchy.select(x1, torch.zeros_like(x2), .1, .1, 2, 2, 2) == [()]
    assert hierarchy.select(x1, x2, .1, .1, 2, 2, 1) == [(1,)]
    with pytest.raises(ValueError, match='align'):
        hierarchy.select(x1, x2[:0], .1, .1)


def test_real_twelve_training_runs_reload_and_h_vs_flat_cpu(cpu_harness, monkeypatch, tmp_path):
    from test_e2e_fsa import Tiny
    from test_c22 import write_bench_data
    monkeypatch.setattr(Tiny, 'layers', property(lambda self: self.moes), raising=False)
    original_encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), 'encode', lambda self,text,**kw: original_encode(self,text.replace('—','-'),**kw))
    run, paths = cpu_harness
    facts = E.make_world(0, 3, 8, 8)
    patches = []
    encodings = []
    old_encode = E.encode_fact
    def encode(tok, template, fact, bos):
        encodings.append((fact.pid, fact.attr))
        return old_encode(tok, template, fact, bos)
    monkeypatch.setattr(E, 'encode_fact', encode)
    val_calls = []
    old_tf = E.tf_eval
    def tf(model, items, *args, **kw):
        val_calls.append([it['fi'] for it in items])
        return old_tf(model, items, *args, **kw)
    monkeypatch.setattr(E, 'tf_eval', tf)
    for p in range(3):
        for g in range(4):
            flags = ['--facts-per-person','8','--shared-attrs','8','--train-person',str(p),
                     '--train-attr-group',str(g),'--attr-groups','4','--slot-partition','12',
                     '--slot-part',str(p*4+g),'--partition-unit','--sparse-state','--slot-frac','.5']
            encodings.clear()
            val_calls.clear()
            result, patch = run([*flags,'--steps','2','--train-audit','20'], f'bank{p}_{g}')
            patches.append(str(patch))
            own = E.person_indices(facts,p,g,4)
            assert set(encodings) == {(p,facts[fi].attr) for fi in own}
            assert own == sorted(set(own))
            assert any(set(fis) == set(own) for fis in val_calls)
            assert result['patched']['test_gen']['n'] == 24*len(E.TEST_T)
            assert result['patched']['test_gen_bank']['n'] == 2*len(E.TEST_T)
            saved = torch.load(patch,weights_only=False)
            assert saved['meta']['train_attr_group'] == g and saved['meta']['attr_groups'] == 4
            if p == 0 and g == 0:
                reload, _ = run([*flags,'--steps','0','--load-patch',str(patch)], 'reload')
                assert reload['patched'] == result['patched']
                with pytest.raises(SystemExit,match='config mismatch'):
                    run([*flags,'--train-attr-group','1','--slot-part','1','--steps','0','--load-patch',str(patch)], 'bad_reload')
    negatives = tmp_path/'negatives.json'
    rng = random.Random(33)
    negatives.write_text(json.dumps([rng.randrange(20,30) for _ in range(40*511+(3000+64)*63)]))
    B.Path(paths['anchor']).write_text(json.dumps([40]*6080))
    directory = write_bench_data(tmp_path/'bench')
    out = tmp_path/'hierarchy.json'
    cli = ['--arms','H,FLAT','--patches',*patches,'--h-layers1','0,1','--h-layers2','0,1',
                    '--indexer-layers','0,1','--h-thresholds1','.2','.8','--h-thresholds2','.2','.8',
                    '--anchor',paths['anchor'],'--eval-set',paths['eval'],'--fineweb',paths['extra'],
                    '--code',paths['extra'],'--negative-text',str(negatives),'--fast-moe','off',
                    '--bench','--bench-dir',str(directory),'--subject-text',
                    '--gen-max-new','1','--qa-n','2','--g-n','4','--out',str(out)]
    result = B.main(cli)
    assert result['negative_sample']['available'] == 3000
    assert result['total_resident_bytes'] == sum(B.Path(p).stat().st_size for p in patches)
    assert result['zero_patch_control']['passed'] and result['solo_bank_control']['passed']
    assert result['revoked']['logits_bit_exact'] and len(result['bank_manifest']) == 12
    for arm, report in result['arms'].items():
        assert report['selector_selection']['locked'] and report['selector_selection']['selection_split'] == 'VAL only'
        assert report['false_activation']['n'] == 3000 and report['false_activation']['one_sided']
        assert report['selection']['test']['n'] == 24*len(E.TEST_T)
        assert report['selection']['held_out_attribute']['n'] == 12*len(E.TEST_T)
        assert set(report['damage']) == {'WikiText','FineWeb','code'}
        assert report['control']['passed'] and not report['verdict_suppressed']
        assert report['subject_dnll']['n'] == 40
        assert {'g_set','two_fact','single_fact'} <= set(report['composed_recall'])
        assert max(report['selection_cost']['test']['per_query']) <= (11 if arm == 'H' else 12)
    assert all(c == 12 for c in result['arms']['FLAT']['selection_cost']['test']['per_query'])
    assert 'level2_null_probes' in result['arms']['H']
    assert json.loads(out.read_text())['revoked']['logits_bit_exact']
    assert result['arms']['H']['selector_selection']['candidates'] == 16
    assert result['arms']['FLAT']['selector_selection']['candidates'] == 6
    # Poison TEST, shorten the negative corpus, and inject a failed control.
    # Neither TEST nor statistical TEST nulls may change validation selection.
    monkeypatch.setattr(E, 'TEST_T', ['Nothing familiar. Report {name}: {attr}? A: {val}.'])
    negatives.write_text(json.dumps(json.loads(negatives.read_text())[:40*511+65*63]))
    monkeypatch.setattr(H, 'positive_control', lambda *args: dict(passed=False, kind='injected failure'))
    failed = B.main(cli)
    assert failed['negative_sample']['available'] < 3000
    for arm in ('H','FLAT'):
        assert failed['arms'][arm]['selector_selection'] == result['arms'][arm]['selector_selection']
        assert failed['arms'][arm]['verdict'] == 'control-failed'
        assert failed['arms'][arm]['verdict_suppressed']
        assert not failed['arms'][arm]['false_activation']['sufficient']
