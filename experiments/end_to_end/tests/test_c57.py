"""C57 CPU coverage: real packed IO, DM measurement, fresh-state CC1."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, _c3_pair8_model, cpu_harness
from test_c34 import linked_facts
import branch_eval as B
import build_in_topic_panel as P
import composition_contract as C
import in_topic as I
import serialize_patch as S


def patch(shape=(16, 24), ids=(0, 7, 8, 191, 383), projection='up_proj'):
    entry = dict(kind='fsa', fmt='ternary', shape=list(shape), idx=torch.tensor(ids, dtype=torch.int32),
                 val=torch.tensor([1 if i % 2 else -1 for i in range(len(ids))], dtype=torch.int8),
                 alpha=torch.arange(shape[0], dtype=torch.float32), scale='rowlearn', fp4_step=.25,
                 logr=torch.tensor([-0., float('nan'), 1.25], dtype=torch.float16),
                 gates=torch.tensor([True, False]), unit_reset_rows=torch.tensor([0, 7]),
                 address_start_idx=torch.tensor([0], dtype=torch.int32), address_start_val=torch.tensor([-1], dtype=torch.int8),
                 logb=torch.tensor([.01, -.25], dtype=torch.float64), flip_idx=torch.empty(0, dtype=torch.int32))
    return dict(format='e2e-fsa-patch-v1', meta=dict(eda_backend='naive_fp32', facts_per_person=20,
        people=3, train_person=0, export='frozen', mask_sha256='mask', gate_tuple=(True, .125),
        signed_zero=-0., nan_payload=S.struct.unpack('<d', bytes.fromhex('010000000000f87f'))[0]),
        entries={'moe.experts.0.'+projection: entry})


@pytest.mark.parametrize('encoding', S.ENCODINGS)
@pytest.mark.parametrize('projection', ['up_proj', 'down_proj'])
@pytest.mark.parametrize('ids', [(), (0, 7, 8, 191, 383)])
def test_encodings_reload_every_tensor_bit_identically(encoding, projection, ids):
    state = patch(ids=ids, projection=projection)
    encoded = S.encode_patch(state, encoding)
    restored = S.decode_patch(encoded)
    S.assert_identical(state, restored)
    assert len(encoded) > 0 and restored['entries'].keys() == state['entries'].keys()
    damaged = bytearray(encoded)
    damaged[-1] ^= 1
    with pytest.raises(ValueError, match='checksum'):
        S.decode_patch(bytes(damaged))


def test_serializer_chooses_actual_lengths_and_reports_marginal_counts(tmp_path):
    files = []
    for i, ids in enumerate([(0, 7), (0, 7, 8, 191, 383)]):
        path = tmp_path/f'bank{i}'/'final.pt'
        path.parent.mkdir()
        torch.save(patch(ids=ids), path)
        (tmp_path/f'bank{i}.json').write_text(json.dumps(dict(patch_bytes=100+i)))
        files.append(path)
    out = tmp_path/'report.json'
    report = S.main(['--patches', *map(str,files), '--retained-facts','10','20',
                     '--out-dir', str(tmp_path/'packed'), '--out', str(out)])
    assert json.loads(out.read_text()) == report
    for row, source in zip(report['patches'], files):
        assert row['packed_bytes'] == min(v['bytes'] for v in row['candidates'].values())
        S.assert_identical(torch.load(source, weights_only=False), S.decode_patch(__import__('pathlib').Path(row['output']).read_bytes()))
        for value in row['candidates'].values():
            assert (value['status'] == 'keep') == (value['ratio'] >= 2)
        assert row['harness_estimate_bytes'] is not None
    assert report['marginal'][0]['delta_facts'] == 10
    assert report['marginal'][0]['bytes_per_added_fact'] == pytest.approx(report['marginal'][0]['delta_bytes']/10)
    assert S.marginal(report['patches'], [20,20])[0]['bytes_per_added_fact'] is None


def test_serializer_rejects_bad_indices_and_detects_tensor_bits():
    bad = patch(ids=(7,0))
    with pytest.raises(ValueError, match='sorted unique'):
        S.encode_patch(bad, 'coordinate')
    state = patch()
    changed = copy.deepcopy(state)
    changed['entries']['moe.experts.0.up_proj']['logr'][0] = 0.
    with pytest.raises(AssertionError, match='tensor changed'):
        S.assert_identical(state, changed)
    changed = copy.deepcopy(state)
    changed['meta']['signed_zero'] = 0.
    with pytest.raises(AssertionError, match='metadata float changed'):
        S.assert_identical(state, changed)


def test_packed_roundtrip_loads_into_real_harness(tmp_path):
    model = _c3_pair8_model()
    E.attach(model, [0,1], 'fsa_forced', 0, sparse=True)
    for m in E.fsa_modules(model):
        m.M.data[:3] = torch.tensor([1., -1., 1.])
    state = E.export_patch(model)
    before = E.weight_hashes(model)
    for encoding in S.ENCODINGS:
        restored = S.decode_patch(S.encode_patch(state, encoding))
        E.load_patch(model, restored)
        assert E.weight_hashes(model) == before


def test_pool_concepts_and_matching_are_seed_independent_and_real():
    assert set(I.concepts()) <= {v.removeprefix('the ') for _, pool in E.ATTRS for v in pool}
    assert 'Varnholt' not in I.concepts() and 'Quillon' not in I.concepts()
    rows = P.matched_candidates()
    assert len({r['prompt'] for r in rows}) == len(rows)
    assert len([r for r in rows if r['stratum'] == 'value']) >= 200
    assert len({r['pair'] for r in rows}) == 24
    assert {r['concept'] for r in rows if r['stratum']=='neighbour'}.isdisjoint(I.concepts())
    assert I.concept_finder(text_panel=True)('rowing, LATEX, the CELLO, a cartographer.') == ['cartographer', 'cello']


def test_last_million_tokens_excluded_from_its_own_prefix(tmp_path):
    path = tmp_path/'fwhq_sel.u32'
    tokens = np.arange(1_000_128, dtype='<u4')
    tokens.tofile(path)
    tail, refs, meta = P.tail_sources(path, [path])
    assert meta == dict(start=128, stop=1_000_128, tokens=1_000_000)
    assert tail[0] == 128 and len(next(iter(refs.values()))) == 128
    contaminated, _ = P.window_overlaps([tail[:64].tolist()], refs)
    assert not contaminated
    with pytest.raises(ValueError, match='last 1M'):
        P.tail_sources(path, [path], tail_tokens=len(tokens))
    with pytest.raises(ValueError, match='overlapping declared census'):
        P.tail_sources(path, [path], census_prefix_tokens=129)


def test_difference_in_differences_cancels_churn_and_clusters_paraphrases():
    rows, base, patched = [], [], []
    for i in range(40):
        for stratum in ('value','neighbour'):
            for j in range(3):
                rows.append(dict(concept=f'{stratum}{i}', pair=str(i), stratum=stratum))
                base.append(j != 0)
                patched.append(j != 1)
    result = I.knowledge_report(rows, base, patched, gate_ready=True)
    assert result['lost'] == result['gained'] == 80
    assert result['difference_in_differences']['clusters'] == 40
    assert result['difference_in_differences']['mean'] == 0 and result['passed']
    assert not I.knowledge_report(rows, base, patched, gate_ready=False)['passed']
    # Value-only loss cannot be waved through by a CI containing zero.
    for i, row in enumerate(rows):
        if row['stratum'] == 'value':
            patched[i] = False
    damaged = I.knowledge_report(rows, base, patched, gate_ready=True)
    assert damaged['difference_in_differences']['mean'] < -.6 and not damaged['passed']


def test_reviewed_qa_aliases_normalize_case_without_retokenizing(monkeypatch):
    row = next(r for r in P.matched_candidates() if r['concept'] == 'oboe')
    monkeypatch.setattr(E, 'greedy_generate', lambda *a, **k: [TOK.encode('WOODWIND FAMILY.').ids])
    flags, answers = I.qa_flags(None, TOK, [row], 'cpu', BOS_T, bs=1)
    assert flags == [True] and answers == ['WOODWIND FAMILY.']


def test_mention_and_following_token_excluded_even_when_inclusive_improves(monkeypatch):
    tokens = TOK.encode('The CELLO is a string instrument.').ids
    mask = I.mention_mask(TOK, tokens)
    start = TOK.decode(tokens).index('CELLO')
    assert not any(mask[start:start+6]) and mask[start+6]
    base = dict(inclusive=[1.,1.], excluded=[1.,1.])
    patched = dict(inclusive=[.9,.9], excluded=[1.004,1.004])
    report = I.text_report(base, patched)
    assert report['inclusive']['dnll'] < 0 and report['dnll'] > .003 and not report['passed']
    assert not I.text_report(base, dict(inclusive=[1,1], excluded=[.99,1.014]))['passed']


def test_cluster_bounds_do_not_gain_precision_from_repeated_phrasings():
    values = [0., .1, -.1, .2]
    a = I.cluster_interval(values, list(range(4)), draws=1000)
    b = I.cluster_interval([v for v in values for _ in range(20)], [i for i in range(4) for _ in range(20)], draws=1000)
    assert a['mean'] == pytest.approx(b['mean']) and a['upper'] == pytest.approx(b['upper'])
    assert a['lower'] == pytest.approx(b['lower']) and a['clusters'] == b['clusters']
    assert I.cluster_interval([0.]*200, ['only-fact']*200)['upper'] is None


def test_cc1_probes_use_both_banks_and_never_count_paraphrases_as_facts():
    facts = linked_facts()
    states = {0:dict(meta=dict(train_person=0)), 1:dict(meta=dict(train_person=1))}
    groups = C.probe_sets(TOK, facts, states, BOS_T, minimum=1000)
    assert len(groups['A/canaries']) == 10 and len(groups['B/canaries']) == 20
    assert len({it['prompt'] for it in groups['unknowns']}) >= 1000
    assert all(set(it['fis']) & {0,1} and set(it['fis']) & {2,3} for it in groups['both/forward_chain'])
    assert all('Answer:' in it['prompt'] for it in groups['A/unseen'])
    canaries = groups['A/canaries']
    bound = C.leakage_bound([False]*len(canaries), canaries)
    assert bound['clusters'] == 1 and not bound['passed']  # the sister's name is not a value canary
    flags = [False]*len(canaries)
    flags[-1] = True
    leak = C.leakage_bound(flags, canaries)
    assert leak['leaking_facts'] == 1 and leak['rate'] == 1 and leak['probe_rate'] == .1
    loss = C.loss_bound([True]*len(canaries), [False]*len(canaries), canaries)
    assert loss['upper'] == 1.


def test_perfect_retention_does_not_certify_a_tiny_fact_panel():
    items = [dict(fis=[i]) for i in range(100) for _ in range(10)]
    bound = C.loss_bound([True]*1000, [True]*1000, items)
    assert bound['mean'] == 0 and bound['upper'] == pytest.approx(1-.05**.01)
    assert bound['upper'] > .01


def test_box_entry_point_and_paired_seed_reporting(cpu_harness, tmp_path, monkeypatch, capsys):
    _, paths = cpu_harness
    (tmp_path/'tokenizer.json').write_text('{}')
    states = tiny_banks()
    files = []
    for i, state in states.items():
        state['meta'].update(method='fsa_forced', fmt='ternary', patch_scale='row', seed=0,
            people=3, facts_per_person=3, shared_attrs=3, values='disjoint', relation_world=True,
            layers=[12,14,17,20], projs=['up_proj','down_proj'], slot_frac=1.,
            export=str(E.EXPORT), export_step=E.EXPORT_STEP)
        file = tmp_path/f'bank{i}.pt'
        torch.save(state, file)
        files.append(file)
    # The CPU fixture has two MoE layers; adapt only that fixture's inventory,
    # retaining the real CLI/model loader, Branches IO, scoring and report path.
    attach = E.attach
    def cpu_attach(model, layer_ids, *a, **kw):
        kw.pop('projs_by_layer', None)
        return attach(model, [0,1], *a, **kw)
    monkeypatch.setattr(E, 'attach', cpu_attach)
    rows = [dict(concept='Ljubljana' if i < 200 else 'Vienna',
                 prompt=f'QA {i}: '+('Ljubljana' if i < 200 else 'Vienna')+'?',
                 answers=['Slovenia' if i < 200 else 'Austria'], pair='Ljubljana', category='city',
                 stratum='value' if i < 200 else 'neighbour', base_correct=i < 200) for i in range(400)]
    windows = [TOK.encode('Ljubljana '+c*54).ids for c in ('K','L')]
    panel = dict(version=2, eda_backend='naive_fp32', concepts=I.concepts(), neighbour_concepts=['Vienna'],
        meta=dict(export=str(E.EXPORT), export_step=E.EXPORT_STEP, tokenizer_sha256=E.file_sha(tmp_path/'tokenizer.json')),
        selection=dict(base_accuracy=.5, protocol=dict(max_new=32, stop_k=2, batch_size=1)),
        knowledge_qa=rows, text_windows=windows, mention_masks=[I.mention_mask(TOK,w) for w in windows], gate_ready=False)
    panel['panel_sha256'] = I.hashlib_panel(panel)
    panel_path = tmp_path/'panel.json'
    panel_path.write_text(json.dumps(panel))
    monkeypatch.setattr(E, 'greedy_generate', lambda model, prompts, *a, **k: [TOK.encode('unknown.').ids for p in prompts])
    monkeypatch.setattr(I, 'qa_flags', lambda m,t,qa,*a,**k: ([r['base_correct'] for r in qa], ['answer']*len(qa)))
    output = tmp_path/'cc1.json'
    args = ['--a',str(files[0]),'--b',str(files[1]),'--a-prime',str(files[2]),
            '--in-topic-panel',str(panel_path),'--anchor',paths['anchor'], '--eval-set',paths['eval'],
            '--fineweb',paths['eval'],'--code',paths['eval'],'--bootstrap','100','--out',str(output)]
    report = C.main(args)
    stored = json.loads(output.read_text())
    assert stored['gates']['verdict'] == 'promotion-blocked'
    assert len(stored['table']) == 7
    assert any(r['configuration']=='AprimeB' and r['decision']=='unavailable' for r in stored['summary_table'])
    assert all(check['passed'] for check in stored['mechanical'].values())
    assert len(stored['probe_clusters']['unknowns']) == 1000
    assert all(len(items)==20 for key,items in stored['probe_clusters'].items() if key.endswith('canaries'))
    paired = copy.deepcopy(stored)
    paired['seed'] = 1
    for bank in ('0','1'):
        paired['inputs'][bank]['sha256'] = bank*64
    pair_path = tmp_path/'cc1_s1.json'
    pair_path.write_text(json.dumps(paired))
    combined = C.main(['paired','--reports',str(output),str(pair_path),'--bootstrap','100','--out',str(tmp_path/'paired.json')])
    assert combined['seeds'] == [0,1] and not combined['gates']['promoted']
    paired['seed'] = 0
    with pytest.raises(ValueError, match='paired seeds'):
        C.combine_reports([stored,paired],100)
    invalid = copy.deepcopy(stored)
    invalid['seed'] = 1
    invalid['inputs'] = copy.deepcopy(paired['inputs'])
    invalid['mechanical']['AB-A']['passed'] = False
    assert C.combine_reports([stored,invalid],100)['gates']['verdict'].startswith('kill')
    repaired = [copy.deepcopy(stored), copy.deepcopy(invalid)]
    repaired[1]['mechanical'] = copy.deepcopy(stored['mechanical'])
    training_pairs = [[r['inputs'][str(bank)]['sha256'] for bank in (0,1)] for r in repaired]
    for r in repaired:
        r.update(repaired=True, repair_training_pairings=training_pairs)
    unseen = copy.deepcopy(repaired)
    unseen[0]['inputs']['1']['sha256'] = 'c'*64
    unseen[1]['inputs']['1']['sha256'] = 'd'*64
    with pytest.raises(ValueError, match='independent bank C'):
        C.combine_reports(repaired,100,unseen_reports=unseen)
    with pytest.raises(ValueError, match='independent bank C'):
        C.combine_reports(repaired,100,as_unseen=True)
    # Corrupt frozen panel data cannot silently change the preservation target.
    panel['knowledge_qa'][0]['base_correct'] = False
    panel_path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match='hash'):
        C.main(args)
    panel['knowledge_qa'][0]['base_correct'] = True
    panel['gate_ready'] = True
    panel.pop('panel_sha256')
    panel['panel_sha256'] = I.hashlib_panel(panel)
    panel_path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match='lacks calibration'):
        C.main(args)


def tiny_banks():
    model = _c3_pair8_model()
    E.attach(model, [0,1], 'fsa_forced', 0, sparse=True)
    states = {}
    for bank in range(3):
        for module in E.fsa_modules(model):
            module.M.data.zero_()
            module.M.data[bank] = 1 if bank != 2 else -1
        states[bank] = E.export_patch(model, dict(train_person=0 if bank==2 else bank))
    return states


def test_cc1_real_branch_unload_order_have_fresh_exact_weights_and_logits(monkeypatch, tmp_path):
    monkeypatch.setattr(E, 'BOS', BOS_T)
    states = tiny_banks()
    models = []
    def factory(current):
        model = _c3_pair8_model()
        E.attach(model, [0,1], 'fsa_forced', 0, sparse=True)
        models.append(model)
        return B.Branches(model, current, {i:1 for i in current}, 'sum_clip'), 'cpu'
    facts = linked_facts()
    groups = C.probe_sets(TOK, facts, states, BOS_T, minimum=3)
    # Retain every category but keep this mechanical integration test small.
    groups = {k: v[:2] for k,v in groups.items()}
    rows = [dict(concept='Ljubljana', pair='Ljubljana', stratum='value', prompt='Ljubljana?', answers=['Slovenia'], base_correct=True),
            dict(concept='Vienna', pair='Ljubljana', stratum='neighbour', prompt='Vienna?', answers=['Austria'], base_correct=False)]
    window = TOK.encode('Ljubljana is a city. More text follows here.').ids
    panel = dict(raw=dict(version=2, gate_ready=False), qa=rows, windows=torch.tensor([window,window]),
                 mention_masks=[I.mention_mask(TOK,window)]*2)
    monkeypatch.setattr(E, 'greedy_generate', lambda model, prompts, *a, **k: [TOK.encode('unknown.').ids for p in prompts])
    monkeypatch.setattr(I, 'qa_flags', lambda *a, **k: ([True,False], ['Slovenia.', 'unknown.']))
    eval_set = tmp_path/'eval.json'
    eval_set.write_text(json.dumps(list(range(40,80))))
    args = SimpleNamespace(gen_stop_k=2, bootstrap=100, repaired=False, eval_set=eval_set)
    report = C.run_contract(factory, states, TOK, facts, groups, {}, panel, args)
    assert len(models) == 7 and len({id(m) for m in models}) == 7
    assert set(report['table']) == set(C.CONFIGURATIONS)
    assert all(check['passed'] for check in report['mechanical'].values())
    assert not report['gates']['promoted'] and report['gates']['verdict'] == 'promotion-blocked'
    assert 'KNOWLEDGE removal' in report['note']
    snapshot = C.mechanical_snapshot(factory(states)[0], [[1,40]]*3, 'cpu')
    changed = copy.deepcopy(snapshot)
    changed['logits'][0][0,0,0] += 1
    assert not C.exact_snapshot(snapshot, changed)['passed']


def test_random_calibration_writes_only_original_forced_zero_slots(tmp_path):
    template = tiny_banks()[0]
    template['meta'].update(method='fsa_forced', fmt='ternary', patch_scale='row', layers=[0,1],
                            projs=['up_proj','down_proj'], slot_frac=1., seed=0)
    path = tmp_path/'final.pt'
    torch.save(template,path)
    model = _c3_pair8_model()
    result = P.random_control(model,path)
    assert result['nnz'] == len(template['entries'])
    for _, lin in E.patched_linears(model):
        original = lin.parametrizations.weight.original
        delta = lin.weight-original
        assert not delta[original != 0].any()
        assert lin.parametrizations.weight[0].codes().ne(0).sum() == 1
    E.set_enabled(True)
