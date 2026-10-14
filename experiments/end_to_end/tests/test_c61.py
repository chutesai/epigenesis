"""C61 regression coverage for every pre-launch review finding (CPU only)."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T
from test_c34 import linked_facts
from test_c57 import tiny_banks
import composition_contract as C
import in_topic as I
import build_in_topic_panel as P


def fixture_table(g=100):
    groups = {bank+'/'+kind: [dict(fis=[i]) for i in range(g) for _ in range(2 if kind=='direct' else 3)]
              for bank in ('A','B') for kind in ('direct','unseen')}
    groups['A/canaries'] = [dict(fis=[i]) for i in range(g) for _ in range(10)]
    groups['unknowns'] = [dict(query_name='virgin'+str(i),fis=[i]) for i in range(1000)]
    probes = {k:dict(n=len(v),per_item=[dict(correct=True,fis=it['fis']) for it in v])
              for k,v in groups.items() if k not in ('A/canaries','unknowns')}
    probes['A/canaries'] = dict(n=10*g,emitted=[False]*(10*g))
    probes['unknowns'] = dict(n=1000,emitted_rate=0.,emitted_fact_ids=[],query_names=[r['query_name'] for r in groups['unknowns']])
    table = {k:dict(probes=copy.deepcopy(probes),in_topic=None) for k in C.CONFIGURATIONS}
    return table,groups


def test_bug1_overlap_every_family():
    states = tiny_banks()
    for partition in (False,True):
        pair = {i:copy.deepcopy(states[i]) for i in (0,1)}
        for state in pair.values():
            state['meta']['partition_unit'] = partition
        assert C.reject_overlap(pair)['overlapping_slots']==0
        pair[1]['entries'] = copy.deepcopy(pair[0]['entries'])
        with pytest.raises(ValueError,match='mechanical-unload failure'):
            C.reject_overlap(pair)


def test_bug2_and9_required_cpu_positive_controls():
    assert C.positive_controls()==dict(disjoint_unload=True,planted_leak=True,overlapping_unload=True,passed=True)


def test_bug2_revoke_prompt_set(tmp_path):
    path = tmp_path/'eval.json'
    path.write_text(json.dumps(list(range(40))))
    probes = C.revoke_prompts(TOK,path,BOS_T)
    assert probes[0]==[BOS_T]+list(range(15))
    assert probes[1:]==[E.encode(TOK,t,BOS_T) for t in ('A quiet morning.','Explain a simple addition.')]
    assert C.CONFIGURATIONS['BA']==(1,0) and C.CONFIGURATIONS['AB-A']==(0,1,'-0')


def test_bug3_replacement_is_not_scored():
    assert 'AprimeB' not in C.CONFIGURATIONS
    parser = C.build_parser()
    assert not next(a for a in parser._actions if a.dest=='a_prime').required


def test_bug4_ten_surfaces_no_padding_no_unique_value_filter():
    facts = linked_facts()
    facts[2] = facts[2]._replace(val=facts[0].val)
    rows = C.canaries(TOK,facts,0,BOS_T)
    owned = C.owned_facts(facts,0)
    assert len(rows)==10*len(owned)
    assert all(sum(r['fis']==[fi] for r in rows)==10 for fi in owned)
    assert all('Archive request' not in r['prompt'] and r.get('family')!='reverse_lookup' for r in rows)
    bound = C.leakage_bound([False]*len(rows),rows)
    assert bound['clusters']==len(owned)
    table,groups = fixture_table()
    table['B']['probes']['A/canaries']['emitted'][0]=True
    table['B']['probes']['A/canaries']['emitted'][10]=True
    assert C.gates(table,groups,None,100)['canary_leakage']['B/A']['decision']=='kill'


@pytest.mark.parametrize('pid', [0, 1, 2, 7])
@pytest.mark.parametrize('count', [10, 11])
def test_bug5_virgin_names_and_unknown_kill(monkeypatch, pid, count):
    facts = [E.Fact(i, 'Person'+str(i), 'their', 'they', 'instrument', value)
             for i, value in zip((0, 1, 2, 7), ('viola', 'harp', 'flute', 'cello'))]
    rows = C.unknowns(TOK,facts,{},BOS_T)
    assert len({r['query_name'] for r in rows})==1000
    assert all(r['query_name'].startswith('NeverTaughtPersonC52') for r in rows)
    value = next(f.val for f in facts if f.pid == pid)
    generations = [TOK.encode(value+'.').ids]*count + [TOK.encode('Nothing stored here.').ids]*(1000-count)
    monkeypatch.setattr(E, 'greedy_generate', lambda *a, **k: generations)
    scored = C.score_configuration(SimpleNamespace(model=None), TOK, facts, {'unknowns': rows},
                                   {}, None, 'cpu', SimpleNamespace())['probes']['unknowns']
    assert scored['emitted'] == [True]*count + [False]*(1000-count)
    assert scored['emitted_rate'] == count/1000
    table, groups = fixture_table(1)
    for kind in ('direct', 'unseen'):
        for item in groups['B/'+kind]:
            item['fis'] = [1]
    groups['unknowns'] = rows
    table['AB']['probes']['unknowns'] = scored
    report = C.gates(table,groups,None,100)
    bound = report['unknown_emission']['AB']
    assert (bound['decision']=='kill') == (count > 10)
    assert (report['verdict']=='kill') == (count > 10)
    assert bound['clusters'] == 2
    assert bound['emitting_facts'] == (1 if pid in (0,1) else 0)
    assert bound['mean'] == (.5 if pid in (0,1) else 0.)


def test_unknown_rate_matches_all_world_values_and_excludes_relations(monkeypatch):
    facts = E.make_world(0,3,100,30,'disjoint',True)
    rows = C.unknowns(TOK,facts,{},BOS_T)
    nonrelations = [f for f in facts if f.attr not in ("sister's name", 'sister')]
    texts = [f.val+'.' for f in nonrelations]
    texts += [f.val+'.' for f in facts if f.attr in ("sister's name", 'sister')]
    texts += ['NeverStoredValue.', 'violin.', nonrelations[0].val+' '+nonrelations[-1].val+'.']
    flags = [True]*len(nonrelations) + [False]*3 + [False,False,True]
    texts += ['Nothing stored here.']*(1000-len(texts))
    flags += [False]*(1000-len(flags))
    monkeypatch.setattr(E,'greedy_generate',lambda *a,**k:[TOK.encode(t).ids for t in texts])
    probe = C.score_configuration(SimpleNamespace(model=None),TOK,facts,{'unknowns':rows},
                                  {},None,'cpu',SimpleNamespace())['probes']['unknowns']
    assert probe['emitted'] == flags
    assert probe['emitted_rate'] == sum(flags)/1000
    assert all(facts[fi].pid in (0,1) for ids in probe['emitted_fact_ids'] for fi in ids)


def test_off_topic_panels_use_harness_windows_and_scores(tmp_path, monkeypatch):
    paths = []
    for name, extra in [('wiki',0), ('fineweb',40), ('code',100)]:
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps([3+i%5 for i in range(40*511+extra)]))
        paths.append(path)
    args = SimpleNamespace(eval_set=paths[0], fineweb=paths[1], code=paths[2])
    harness = E.extra_eval_panels
    calls = []
    def load(path, bos, dev):
        calls.append(path)
        return harness(path, bos, dev)
    monkeypatch.setattr(E,'extra_eval_panels',load)
    panels = C.off_topic_panels(args,BOS_T)
    assert calls == list(map(str,paths))
    class Model(torch.nn.Module):
        def __init__(self, strength):
            super().__init__()
            self.strength = strength
        def forward(self, x):
            return torch.nn.functional.one_hot((x+1)%8,8).float()*self.strength
    base, patched = Model(0.), Model(1.)
    base_scores = C.score_configuration(SimpleNamespace(model=base),TOK,[],{},panels,None,'cpu',args)
    patched_scores = C.score_configuration(SimpleNamespace(model=patched),TOK,[],{},panels,None,'cpu',args)
    for (name,wins), path in zip(panels.items(),paths):
        expected = harness(str(path),BOS_T,'cpu')[path.name][1]
        assert wins.shape == (40,512) and torch.equal(wins,expected)
        assert wins[:,0].tolist() == [BOS_T]*40
        expected_delta = E.paired_nll(patched,expected,E.nll_windows(base,expected,'cpu',bs=1),'cpu')['dnll']
        actual_delta = np.mean(np.asarray(patched_scores['off_topic_nll'][name])-base_scores['off_topic_nll'][name])
        assert actual_delta == pytest.approx(expected_delta,abs=1e-7)
    paths[0].write_text(json.dumps([3]*(40*511-1)))
    with pytest.raises(ValueError,match='needs at least'):
        C.off_topic_panels(args,BOS_T)


def test_bug6_topic_point_kills_and_unavailable():
    assert C.topic_bounds(None,None,None,100) is None
    panel=dict(raw=dict(gate_ready=True),qa=[dict(base_correct=True,stratum='value',concept=str(i)) for i in range(100)])
    base=dict(qa=[True]*100,text=dict(inclusive=[1.,1.],excluded=[1.,1.]))
    row=copy.deepcopy(base)
    row['qa'][:2]=[False,False]
    row['text']['excluded']=[1.01,1.01]
    result=C.topic_bounds(panel,base,row,100)
    assert result['knowledge_qa']['decision']==result['text_dnll']['decision']=='kill'
    row=copy.deepcopy(base)
    row['text']['excluded']=[.99,1.014]
    assert C.topic_bounds(panel,base,row,100)['text_dnll']['decision']=='promotion-blocked'


def test_bug7_retention_census_chains_coverage_conflicts():
    facts=linked_facts()
    groups=C.probe_sets(TOK,facts,{0:{},1:{}},BOS_T)
    for bank,label in ((0,'A'),(1,'B')):
        assert len(groups[label+'/direct'])==2*len(C.owned_facts(facts,bank))
        assert len(groups[label+'/unseen'])==3*len(C.owned_facts(facts,bank))
        assert label+'/relational' not in groups
    both=groups['both/cross_person_conjunction']
    assert {fi for r in both for fi in r['fis']}=={fi for fi,f in enumerate(facts) if f.pid in (0,1)}
    assert all(facts[r['fis'][0]].pid==0 and facts[r['fis'][1]].pid==1 for r in groups['both/forward_chain'])
    assert {r['fis'][0] for r in groups['conflicting_identifiers']}==set(E.conflict_map(facts))
    assert groups['reverse_lookup']


def test_bug8_no_seed_pool_and_one_seed_kills():
    table,groups=fixture_table(150)
    verdict=C.gates(table,groups,None,100)
    assert verdict['retention']['A/direct']['upper']>.01
    def report(seed):
        return dict(seed=seed,family='full-mask',repaired=False,inputs={str(i):dict(sha256=f'{seed}{i}') for i in (0,1)},
            gates=copy.deepcopy(verdict),mechanical={'AB-A':dict(passed=True)},positive_control=dict(passed=True),
            summary_table=[dict(configuration='AB',panel='retention',n_strings=300,n_clusters=150,
                                estimate=0.,upper95=verdict['retention']['A/direct']['upper'],decision='promotion-blocked')])
    reports=[report(0),report(1)]
    reports[0]['gates']['verdict']='kill'
    combined=C.combine_reports(reports)
    assert combined['gates']['verdict']=='kill' and not combined['gates']['promoted']
    assert [r['n_clusters'] for r in combined['summary_table']]==[150,150]


def test_bug10_table_columns_and_unavailable_rows():
    rows=C.summary_table({},None)
    assert {r['configuration'] for r in rows}=={'AprimeB','repaired-unseen'}
    assert all(r['decision']=='unavailable' for r in rows)
    assert set(rows[0])=={'configuration','panel','n_strings','n_clusters','estimate','upper95','decision'}
    with pytest.raises(ValueError,match='independent bank C'):
        r=dict(seed=0,family='x',inputs={'0':dict(sha256='a'),'1':dict(sha256='b')})
        s=copy.deepcopy(r); s.update(seed=1,inputs={'0':dict(sha256='c'),'1':dict(sha256='d')})
        C.combine_reports([r,s],unseen_reports=[r,s])


def test_panel_recorded_end_underdeclaration_and_loader_scan(tmp_path):
    source=tmp_path/'source.u32'
    np.arange(70000,dtype='<u4').tofile(source)
    record=tmp_path/'record.json'
    record.write_text(json.dumps(dict(streams={'fw':str(source)},cursors={'fw':70000},inputs={str(source):E.file_sha(source)})))
    for end in (0,65520):
        with pytest.raises(ValueError,match='under-declared'):
            I.census_sources([str(source)+':'+str(end)],[record])
    refs,cert=I.census_sources([str(source)+':70000'],[record])
    assert cert['complete'] and len(refs[str(source)])==70000
    panel=dict(census_scan=cert,text_windows=[list(range(65520,65552))])
    with pytest.raises(ValueError,match='32-gram overlap'):
        I.verify_census_certificate(panel)
    panel['text_windows']=[list(range(100000,100032))]
    assert I.verify_census_certificate(panel)
    with pytest.raises(ValueError,match='lacks complete'):
        I.verify_census_certificate({})
    assert not I.census_sources([str(source)+':0'],[])[1]['complete']
    assert any(a.dest=='heldout_text' for a in P.build_parser()._actions)


def test_panel_all_census_streams_are_required(tmp_path):
    streams={}
    for name in ('a','b'):
        path=tmp_path/(name+'.u32'); np.arange(80,dtype='<u4').tofile(path); streams[name]=str(path)
    record=tmp_path/'census.json'; record.write_text(json.dumps(dict(streams=streams,cursors=dict(a=64,b=80))))
    with pytest.raises(ValueError,match='under-declared'):
        I.census_sources([streams['a']+':64'],[record])


def test_written_fact_reference_stays_report_only_without_stored_gen(monkeypatch):
    rows=[dict(concept='x',pair='x',stratum='value',prompt='x?')]
    panel=dict(raw=dict(version=2,selection=dict(base_accuracy=1,protocol={})),qa=rows,windows=None,
               mention_masks=None,path='p',sha256='s',overlap={})
    baseline=dict(qa=[True],text=dict(inclusive=[1.,1.],excluded=[1.,1.]))
    monkeypatch.setattr(I,'qa_flags',lambda *a,**k:([True],['x']))
    monkeypatch.setattr(I,'text_scores',lambda *a,**k:baseline['text'])
    monkeypatch.setattr(I,'fact_flags',lambda *a,**k:[True,True])
    group=dict(bank=0,facts=1,items=[dict(fi=0)]*2,baseline=[False,False])
    result=I.evaluate(None,TOK,panel,baseline,[group],'cpu',BOS_T,SimpleNamespace())
    assert result['written_facts']['banks'][0]['status'].startswith('report-only')
    group['stored_alone_gen']=1.
    result=I.evaluate(None,TOK,panel,baseline,[group],'cpu',BOS_T,SimpleNamespace())
    assert result['written_facts']['banks'][0]['accuracy_delta']==0.


def test_separate_heldout_file_and_recorded_census_hash(tmp_path):
    census=tmp_path/'census.u32'; np.arange(100,dtype='<u4').tofile(census)
    fresh=tmp_path/'fresh.u32'; np.arange(1000,1100,dtype='<u4').tofile(fresh)
    record=tmp_path/'record.json'
    record.write_text(json.dumps(dict(streams={'fw':str(census)},cursors={'fw':100},inputs={str(census):E.file_sha(census)})))
    args=SimpleNamespace(heldout_text=fresh,fineweb=census,census_read=[str(census)+':100'],census_record=[record])
    stream,refs,meta,cert=P.text_sources(args,[census])
    assert stream.tolist()==list(range(1000,1100)) and meta['start']==0 and cert['complete']
    assert not P.window_overlaps([stream[:32]],refs)[0]
    np.arange(200,300,dtype='<u4').tofile(census)
    with pytest.raises(ValueError,match='hash changed'):
        I.census_sources(args.census_read,args.census_record)


def test_resume_records_conservatively_cover_calibration_and_targets(tmp_path):
    path=tmp_path/'stream.u32'; np.arange(100,dtype='<u4').tofile(path)
    record=tmp_path/'resume.pt'
    torch.save(dict(identity=dict(streams={'fw':str(path)},targets=[96]),
                    cursors={'fw':64},inputs={str(path):E.file_sha(path)}),record)
    with pytest.raises(ValueError,match='under-declared'):
        I.census_sources([str(path)+':64'],[record])
    assert I.census_sources([str(path)+':96'],[record])[1]['complete']


def test_roster_conjunction_sample_covers_every_value_fact():
    facts=E.make_world(0,3,100,30,'disjoint',True)
    scopes={i:{fi for fi,f in enumerate(facts) if f.pid==i} for i in (0,1)}
    rows=C.conjunction_census(TOK,facts,BOS_T,scopes)
    assert len(rows)==200 and len({r['prompt'] for r in rows})==200
    assert {fi for r in rows for fi in r['fis']}==scopes[0]|scopes[1]
    assert rows==C.conjunction_census(TOK,facts,BOS_T,scopes)


def test_failed_positive_control_voids_run(monkeypatch,tmp_path):
    table,groups=fixture_table()
    for row in table.values():
        row['off_topic_nll']={}
    probes=tmp_path/'eval.json'; probes.write_text('[1,2,3]')
    monkeypatch.setattr(C,'positive_controls',lambda:dict(passed=False))
    monkeypatch.setattr(C,'mechanical_snapshot',lambda *a:dict(weights={},logits=[torch.tensor([1.])]*3))
    monkeypatch.setattr(C,'score_configuration',lambda *a:copy.deepcopy(table['AB']))
    fake=lambda states:(SimpleNamespace(model=torch.nn.Module(),states=states,activate=lambda route:None),'cpu')
    report=C.run_contract(fake,{i:tiny_banks()[i] for i in (0,1)},TOK,linked_facts(),groups,{},None,
                          SimpleNamespace(eval_set=probes,bootstrap=100,repaired=False))
    assert report['gates']['void'] and report['gates']['verdict']=='unavailable'
    assert not report['gates']['promoted']


def test_full_three_person_confusion_is_reported(monkeypatch):
    facts=E.make_world(0,3,3,3,'disjoint',True)
    groups=C.probe_sets(TOK,facts,{0:{},1:{}},BOS_T)
    conflicts=E.conflict_map(facts)
    chosen=next(r for r in groups['conflicting_identifiers'] if conflicts.get(r['fis'][0]))
    fi=chosen['fis'][0]
    other=next(f.val for f in facts if f.attr==facts[fi].attr and f.pid!=facts[fi].pid)
    monkeypatch.setattr(E,'greedy_generate',lambda *a,**k:[TOK.encode(other+'.').ids])
    branches=SimpleNamespace(model=None)
    result=C.score_configuration(branches,TOK,facts,{'conflicting_identifiers':[chosen]}, {},None,'cpu',SimpleNamespace(gen_stop_k=2))
    assert result['probes']['conflicting_identifiers']['per_item'][0]['confused']


def test_full_mask_overlap_writes_kill_without_model_build(tmp_path,monkeypatch):
    states={i:tiny_banks()[i] for i in (0,1)}
    for state in states.values():
        state['meta'].update(method='fsa_forced',fmt='ternary',patch_scale='row',seed=0,people=3,
                             relation_world=True,layers=[12,14,17,20],export=str(E.EXPORT),export_step=58814)
    states[1]['entries']=copy.deepcopy(states[0]['entries'])
    paths=[]
    for i,state in states.items():
        path=tmp_path/f'bank{i}.pt'; torch.save(state,path); paths.append(path)
    output=tmp_path/'report.json'
    args=['--a',str(paths[0]),'--b',str(paths[1]),'--eval-set','missing','--fineweb','missing',
          '--code','missing','--anchor','missing','--out',str(output)]
    report=C.main(args)
    assert report['family']=='full-mask' and report['gates']['verdict']=='kill'
    assert json.loads(output.read_text())['summary_table'][0]['decision']=='kill'


def test_concept_kill_counts_concepts_instead_of_qa_frames():
    panel=dict(raw=dict(gate_ready=True),qa=[dict(base_correct=True,stratum='value',concept=str(i))
                                          for i in range(100) for _ in range(10)])
    base=dict(qa=[True]*1000,text=dict(inclusive=[1.,1.],excluded=[1.,1.]))
    row=copy.deepcopy(base)
    row['qa'][0]=False; row['qa'][10]=False
    bound=C.topic_bounds(panel,base,row,100)['knowledge_qa']
    assert bound['template_mean_loss']==pytest.approx(.002)
    assert bound['mean']==.02 and bound['decision']=='kill' and bound['clusters']==100


def test_record_target_order_uses_domains_not_json_key_order(tmp_path):
    streams={}
    for name in ('b','a'):
        path=tmp_path/(name+'.u32'); np.arange(100,dtype='<u4').tofile(path); streams[name]=str(path)
    record=tmp_path/'record.json'
    record.write_text(json.dumps(dict(identity=dict(domains=['a','b'],streams=streams,targets=[64,96]),
        cursors=dict(a=32,b=64),inputs={p:E.file_sha(p) for p in streams.values()})))
    with pytest.raises(ValueError,match='under-declared'):
        I.census_sources([streams['a']+':64',streams['b']+':64'],[record])
    assert I.census_sources([streams['a']+':64',streams['b']+':96'],[record])[1]['complete']
