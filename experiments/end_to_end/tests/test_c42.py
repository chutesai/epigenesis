"""C42 CPU mechanics and opt-in GPU-path tests; no external model required."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_e2e_fsa import E, TOK, Tiny, cpu_harness
import branch_eval as B
import c42_selector as C
import c42_runtime as G


def fixture(banks=12, facts_per_bank=2):
    generator = torch.Generator().manual_seed(42)
    n = banks*facts_per_bank
    labels = torch.arange(banks).repeat_interleave(facts_per_bank)
    fis = torch.arange(n)
    x = torch.eye(banks)[labels].double()
    splits = {}
    for name in ('train','cal','val','test'):
        positive = x.clone()
        negative = -.2*torch.ones(64,banks)+.005*torch.randn(64,banks,generator=generator,dtype=torch.double)
        features = positive if name == 'train' else torch.cat((positive,negative))
        y = labels if name == 'train' else torch.cat((labels,torch.full((64,),-1)))
        fi = fis if name == 'train' else torch.cat((fis,torch.full((64,),-1)))
        splits[name] = dict(features={0:features},labels=y,fact_ids=fi,forms=['name']*n+([] if name == 'train' else ['general']*64))
    return dict(format=C.FORMAT,layers=[0],splits=splits,general={0:torch.zeros(8,banks)},
        fact_owner={int(fi):int(b) for fi,b in zip(fis,labels)},bank_parent={b:b//4 if banks == 12 else b for b in range(banks)})


def test_centroid_matches_existing_bank_indexer():
    cache = fixture()
    train = cache['splits']['train']
    args = train['features'][0],train['labels'],cache['general'][0]
    a,b = C.Centroid.fit(*args),B.BankIndexer.fit(*args)
    assert torch.equal(a.scores(train['features'][0]),b.scores(train['features'][0]))


def test_llr_balanced_prior_and_inhibitory_coefficients_and_revoke():
    x = torch.tensor([[1.,0.],[0.,1.]]).repeat(3,1)
    scorer = C.Centroid.fit(x,[0,1]*3,torch.zeros(2,2))
    cal = torch.cat((x,torch.zeros(30,2)))
    labels = torch.tensor([0,1]*3+[-1]*30)
    inhibitory = 1-scorer.scores(cal)
    index = C.LLRIndexer.fit(scorer,cal,labels,inhibitory)
    assert all(len(v) == 3 for v in index.coefficients.values())
    assert index.select(cal[:2],inhibitory=inhibitory[:2]) == [(0,),(1,)]
    assert index.select(cal[6:],inhibitory=inhibitory[6:]) == [()]*30
    with pytest.raises(ValueError,match='required'):
        index.select(cal[:1])
    saved = index.ledger()
    assert saved['0']['score_dimensions'] == 2
    index.revoke(0)
    assert 0 not in index.coefficients and index.owners == (1,)
    # Repeating only negatives doesn't introduce a prevalence prior into LLRs.
    raw = torch.tensor([[1.],[1.],[-1.],[-1.]])
    a = C.logistic_llr(raw,[True,True,False,False])
    b = C.logistic_llr(torch.cat((raw[:2],raw[2:].repeat(20,1))),[True]*2+[False]*40)
    assert torch.allclose(a,b,atol=1e-7)


def test_strict_null_margin_ties_and_registered_multi_only():
    scores = torch.tensor([[2.,1.],[1.,1.],[0.,-1.],[-1.,-2.]])
    assert C.gate(scores,(4,8),1.) == [(),(),(),()]
    assert C.gate(scores,(4,8),.5) == [(4,),(),(),()]
    assert C.gate(scores,(4,8),.5,['pair']*4,('pair',)) == [(4,8),(),(),()]
    assert C.gate(scores,(4,8),.5,['unknown']*4,('pair',)) == [(4,),(),(),()]
    with pytest.raises(ValueError):
        C.gate(scores,(4,8),-1)


def test_threshold_and_margin_tuned_at_query_not_bank_false_fire():
    scores = torch.tensor([[5.,4.],[2.,1.],[0.,-1.],[1.,1.]])
    delta = C.margin_cutoff(scores,target=.25)
    routes = C.gate(scores,(0,1),delta)
    assert sum(bool(r) for r in routes)/len(routes) <= .25
    assert C.cutoff([5.,4.,3.,2.],.25) == 4.


def test_fixture_study_locks_validation_and_reports_cost_and_children():
    torch.set_num_threads(1)
    cache = fixture()
    report,locked = C.analyze_set(cache)
    assert set(report['arms']) == {'cosine','zscore','flat_llr','current_hierarchy','llr_beam','llr_rescue'}
    for arm in report['arms'].values():
        assert arm['metrics']['routing_recall'] == 1.
        assert arm['locked']['validation']['false_fire'] == 0.
        assert arm['metrics']['meets_test_false_fire_cap'] == (arm['metrics']['false_fire_upper_95'] <= C.TARGET)
        assert arm['metrics']['false_fire_verdict'] == 'unresolved (power)'
    assert report['arms']['llr_beam']['locked']['parent_gate'] == 'none (beam top-k)'
    assert len(report['arms']['flat_llr']['locked']['bank_ledger']) == 12
    for parent,child in locked['llr_beam']['index'].children.items():
        assert set(child.owners) == {b for b in range(12) if b//4 == parent}
        assert isinstance(child.scorer,C.SiblingKeys)
    price = report['arms']['llr_rescue']['metrics']['priced_comparisons']
    assert price['value'] == report['arms']['llr_rescue']['metrics']['comparisons_per_query']
    changed = deepcopy(cache)
    changed['splits']['test']['features'][0].mul_(-10)
    after = C.fit_study(changed)
    for arm,cfg in locked.items():
        assert after[arm]['validation'] == cfg['validation']
        assert after[arm].get('delta') == cfg.get('delta')
        assert after[arm].get('threshold') == cfg.get('threshold')


class Scores:
    def __init__(self,owners,scores):
        self.owners,self.value = owners,torch.as_tensor(scores,dtype=torch.double).clone()
    def scores(self,x,inhibitory=None):
        return self.value[:len(x)]
    def select(self,x,delta,inhibitory=None):
        return C.gate(self.scores(x),self.owners,delta)


def test_hierarchy_summed_llr_rescue_and_exact_cost():
    parent = Scores((0,1),[[3.,1.],[-2.,-3.],[1.,.8]])
    children = {0:Scores((0,1),[[.5,0.],[0.,0.],[0.,0.]]),
                1:Scores((2,3),[[4.,0.],[0.,0.],[0.,0.]])}
    flat = Scores((0,1,2,3),[[0.,0.,0.,2.],[0.,0.,0.,2.],[0.,0.,0.,2.]])
    h = C.LLRHierarchy(parent,children,flat,{0:0,1:0,2:1,3:1},2,True)
    x = torch.zeros(3,1)
    tr = h.trace(x,x,delta=.3)
    # Child compensates a weaker parent in comparable LLR units.
    assert tr['routes'][0] == (2,)
    assert tr['rescued'] == [False,True,True]
    assert tr['routes'][1:] == [(3,),(3,)]
    assert tr['comparisons'] == [6,10,10]
    h.rescue = False
    assert h.trace(x,x,.3)['routes'][1:] == [(),()]


def test_simulation_splits_facts_consistently_across_all_forms():
    cache = fixture(facts_per_bank=10)
    split = C.simulate(cache)
    assert len(set(split['fact_owner'].values())) == 100
    assert set(split['bank_parent']) == set(range(100))
    assert split['simulation'] and cache['fact_owner'] != split['fact_owner']
    for s in split['splits'].values():
        assert all(int(b) == split['fact_owner'][int(fi)] for b,fi in zip(s['labels'],s['fact_ids']) if fi >= 0)
    C.validate_cache(split)
    with pytest.raises(ValueError,match='nonempty'):
        C.simulate(fixture())


def test_cache_validation_leakage_and_missing_banks():
    cache = fixture()
    for name in ('train','cal'):
        cache['splits'][name]['items'] = [dict(p=[1,2,3])]
    with pytest.raises(ValueError,match='leaked'):
        C.validate_cache(cache)
    cache = fixture()
    cache['splits']['cal']['labels'][0] = 40
    with pytest.raises(ValueError,match='bank'):
        C.validate_cache(cache)


def test_kill_rules_exact_boundaries_and_insufficient_sample():
    def arm(recall,accuracy=None):
        return dict(metrics=dict(routing_recall=recall,route_accuracy=recall if accuracy is None else accuracy,
            false_fire_upper_95=C.TARGET,meets_test_false_fire_cap=True,sufficient_negatives=True,
            rescue_rate=.99,positive_rescue_rate=.4))
    twelve = dict(banks=12,best_hierarchy='llr_beam',arms=dict(cosine=arm(.5),flat_llr=arm(.56),llr_beam=arm(.545)))
    hundred = dict(banks=100,arms=dict(llr_rescue=arm(.55)),false_fire_scaling=dict(matches_prediction=True))
    rules = C.kill_rules([twelve,hundred])
    assert rules['llr'] == 'keep' and rules['hierarchy'] == 'keep'
    assert rules['simulation_rescue_rate'] == .99 and rules['simulation_positive_rescue_rate'] == .4
    hundred['arms']['llr_rescue']['metrics']['positive_rescue_rate'] = .401
    assert C.kill_rules([twelve,hundred])['hierarchy'] == 'kill'
    twelve['arms']['cosine']['metrics']['false_fire_upper_95'] = .0011
    assert C.kill_rules([twelve])['llr'] == 'unresolved (power)'
    twelve['arms']['llr_beam']['metrics']['route_accuracy'] = .50
    assert C.kill_rules([twelve])['hierarchy'] == 'kill'


def test_cpu_cli_and_insufficient_simulation_are_explicit(tmp_path):
    path,out = tmp_path/'features.pt',tmp_path/'report.json'
    torch.save(fixture(),path)
    result = C.main(['--caches',str(path),'--simulate-100','--out',str(out)])
    assert result == json.loads(out.read_text())
    assert result['results'][-1]['banks'] == 100 and 'unavailable' in result['results'][-1]
    assert result['kill_rules']['hierarchy'].startswith('unresolved')
    assert result['lm_training'] is False


def test_opt_in_parser_and_dispatch(monkeypatch,tmp_path):
    assert B.build_parser().parse_args([]).c42_mode is None
    assert not B.build_parser().parse_args([]).c42_parent_gate
    monkeypatch.setattr(G,'run',lambda b,a: dict(mode=a.c42_mode))
    assert B.main(['--c42-mode','dump']) == dict(mode='dump')


@pytest.mark.parametrize('negative_n,events,certified',[(3000,3,False),(10000,3,True),(2994,0,False),(2995,0,True),(0,0,False)])
def test_false_fire_certification_uses_exact_upper_bound(negative_n,events,certified):
    m = C.metrics([(0,)]*events+[()]*(negative_n-events),[-1]*negative_n)
    assert m['meets_test_false_fire_point_cap']
    assert m['meets_test_false_fire_cap'] is certified
    assert m['false_fire_verdict'] == ('matched at 1e-3' if certified else 'unresolved (power)')


def test_simulation_unavailable_rescue_and_diagnostic_mismatch():
    def arm(accuracy):
        return dict(metrics=dict(routing_recall=accuracy,route_accuracy=accuracy,
            false_fire_upper_95=.001,positive_rescue_rate=.2,rescue_rate=.98))
    twelve = dict(banks=12,best_hierarchy='llr_beam',arms=dict(cosine=arm(.5),flat_llr=arm(.55),llr_beam=arm(.53)))
    hundred = dict(banks=100,arms=dict(flat_llr=arm(.8),llr_beam=arm(.78),llr_rescue=dict(unavailable='VAL cap')),
        false_fire_scaling=dict(matches_prediction=False))
    rules = C.kill_rules([twelve,hundred])
    assert rules['llr'] == 'keep' and rules['hierarchy'].startswith('unresolved')
    assert rules['llr_scaling'].startswith('diagnostic:') and 'kill' not in rules['llr_scaling']
    hundred['arms']['llr_beam'] = arm(.779)
    assert C.kill_rules([twelve,hundred])['hierarchy'] == 'kill'
    hundred['arms']['llr_rescue'] = arm(.8)
    assert C.kill_rules([twelve,hundred])['hierarchy'] == 'keep'
    twelve['arms']['llr_beam']['metrics']['false_fire_upper_95'] = .00101
    assert C.kill_rules([twelve,hundred])['hierarchy'] == 'unresolved (power)'


def test_parent_gate_prevents_child_compensation_and_prices_only_visited_children():
    parent = Scores((0,1),[[-1.,-2.],[0.,-1.],[1.,-1.]])
    children = {0:Scores((0,),[[3.],[3.],[3.]]),1:Scores((1,),[[3.],[3.],[3.]])}
    flat = Scores((0,1),[[0.,2.]]*3)
    x = torch.zeros(3,1)
    h = C.LLRHierarchy(parent,children,flat,{0:0,1:1},beam=2)
    assert h.trace(x,x)['routes'] == [(0,),(0,),(0,)]
    h.parent_gate = True
    trace = h.trace(x,x)
    assert trace['routes'] == [(),(),(0,)]
    assert trace['comparisons'] == [2,2,3]
    h.rescue = True
    trace = h.trace(x,x)
    assert trace['routes'] == [(1,),(1,),(0,)]
    assert trace['comparisons'] == [4,4,3]
    assert trace['rescued'] == [True,True,False]


def test_parent_gate_propagates_to_fit_and_locked_output():
    report,locked = C.analyze_set(fixture(),parent_gate=True)
    for arm in ('llr_beam','llr_rescue'):
        assert locked[arm]['index'].parent_gate
        assert report['arms'][arm]['locked']['parent_gate'] == 'LLR > 0'


@pytest.mark.parametrize('mode,extra,expected',[('dump',[],10000),('gen',[],3000),('dump',['--negative-n','4000'],4000)])
def test_c42_dump_negative_default_and_override(monkeypatch,mode,extra,expected):
    monkeypatch.setattr(G,'run',lambda b,a: (a.negative_n,a.c42_parent_gate))
    assert B.main(['--c42-mode',mode,'--c42-parent-gate',*extra]) == (expected,True)


def test_query_splits_are_disjoint_and_cover_forms_and_true_nulls():
    facts = E.make_world(0,3,8,8)
    owners = {fi:f.pid*4+(sorted(g.attr for g in facts if g.pid == f.pid).index(f.attr)//2) for fi,f in enumerate(facts)}
    splits = G.query_splits(B,TOK,facts,owners)
    seen = set()
    for name in ('train','cal','val','test'):
        seqs = {tuple(it['p']) for it in splits[name]}
        assert not (seen & seqs)
        seen |= seqs
        if name != 'train':
            assert {'name','pronoun','indirect','untaught_entity','untaught_attribute','wrong_scope'} <= {it['form'] for it in splits[name]}
            for item in splits[name]:
                if item['form'] == 'wrong_scope':
                    assert item['fi'] == -1 and 'Use only the profile group' in item['prompt']


def test_known_density_ratio_is_not_posterior_odds():
    # P(score=1|bank)=.8, P(score=1|not bank)=.2; negative count is 10x.
    positive = torch.tensor([[1.]]*8+[[0.]]*2)
    negative = torch.tensor([[1.]]*20+[[0.]]*80)
    coef = C.logistic_llr(torch.cat((positive,negative)),[True]*10+[False]*100,l2=1e-8)
    import math
    assert float(coef[-1]) == pytest.approx(-math.log(4),abs=1e-5)
    assert float(coef[0]+coef[-1]) == pytest.approx(math.log(4),abs=1e-5)


def test_cached_inhibitory_analysis_and_simulated_hundred_scaling():
    cache = fixture()
    train = cache['splits']['train']
    scorer = C.Centroid.fit(train['features'][0],train['labels'],cache['general'][0])
    for name in ('cal','val','test'):
        cache['splits'][name]['inhibitory'] = {0:1-scorer.scores(cache['splits'][name]['features'][0])}
    report,_ = C.analyze_set(cache)
    assert all(len(r['coefficients']) == 3 for r in report['arms']['flat_llr']['locked']['bank_ledger'].values())
    with pytest.raises(ValueError,match='sub-bank inhibitory'):
        C.simulate(cache)
    hundred,_ = C.analyze_set(C.simulate(fixture(facts_per_bank=10)))
    scaling = hundred['false_fire_scaling']
    assert len(scaling['per_bank_rates']) == 100
    assert scaling['prediction_split'] == 'VAL' and scaling['measurement_split'] == 'TEST'
    assert scaling['matches_prediction'] and scaling['measured'] == 0.


@pytest.mark.parametrize('null_limit,expected',[(None,10000),(10001,3333),(8,2)])
def test_real_tiny_banks_dump_once_then_composed_gen(cpu_harness,monkeypatch,tmp_path,null_limit,expected):
    monkeypatch.setattr(Tiny,'layers',property(lambda self:self.moes),raising=False)
    run,paths = cpu_harness
    patches = [str(run(['--steps','1','--train-person',str(p),'--sparse-state','--slot-frac','.5'],f'c42_bank{p}')[1]) for p in range(3)]
    Path(paths['anchor']).write_text(json.dumps([40]*6080))
    monkeypatch.setattr(B.R,'independent_queries',lambda sources,bos,n: [dict(p=[bos,1000+i],source='fixture',start=i*63)
        for i in range(n if null_limit is None else min(n,null_limit))])
    calls = []
    def collect(model,sequences,layers,dev,positions,progress=None):
        calls.append((tuple(layers),len(sequences),E.FSAParam.enabled))
        gen = torch.Generator().manual_seed(4242)
        return {l:torch.randn(len(sequences),16,generator=gen) for l in layers}
    monkeypatch.setattr(B.S,'collect_residuals',collect)
    cache,out = tmp_path/'features.pt',tmp_path/'dump.json'
    flags = ['--patches',*patches,'--c42-cache',str(cache),'--c42-layers','0,1','--fast-moe','off',
        '--anchor',paths['anchor'],'--eval-set',paths['eval'],'--fineweb',paths['extra'],'--code',paths['extra'],
        '--gen-max-new','1','--gen-bs','8','--h-top-k1','2']
    result = B.main(['--c42-mode','dump',*flags,'--out',str(out)])
    assert result['banks'] == 3 and len(calls) == 1 and calls[0][0] == (0,1) and not calls[0][2]
    assert E.FSAParam.enabled and result['counts']['test'] >= expected
    sample = result['negative_sample']
    assert sample['requested_per_split'] == 10000 and sample['actual_per_split'] == expected
    assert sample['corpus_limited'] == (null_limit is not None)
    saved = torch.load(cache,weights_only=False)
    seen = set()
    for name in ('cal','val','test'):
        sequences = {tuple(it['p']) for it in saved['splits'][name]['items'] if it['form'] == 'general'}
        assert len(sequences) == expected and not (seen & sequences)
        seen |= sequences
    assert saved['negative_sample'] == sample
    generated = B.main(['--c42-mode','gen','--c42-parent-gate',*flags,'--out',str(tmp_path/'gen.json')])
    assert len(calls) == 1  # GEN consumes the cache; no repeated feature forwards.
    assert all(generated['controls'].values()) and not generated['verdict_suppressed']
    assert 'flat_llr' in generated['arms'] and len(generated['arms']) == 2
    assert generated['best_hierarchy_selected_on'] == 'VAL'
    assert generated['routing_analysis']['negative_sample'] == sample
    for arm in ('llr_beam','llr_rescue'):
        cfg = generated['routing_analysis']['arms'][arm].get('locked')
        if cfg is not None:
            assert cfg['parent_gate'] == 'LLR > 0'
    saved = torch.load(cache,weights_only=False)
    saved['export_step'] += 1
    torch.save(saved,cache)
    with pytest.raises(ValueError,match='match'):
        B.main(['--c42-mode','gen',*flags,'--out',str(out)])


def test_scaling_uses_union_for_gates_and_independence_only_for_evidence(monkeypatch,tmp_path):
    scores = torch.zeros(164,100,dtype=torch.double)
    scores[:100] = torch.eye(100)
    scores[100:103,:2] = torch.tensor([[2.,1.],[0.,2.],[2.,1.]])
    class Fixed(Scores):
        def ledger(self):
            return {}
    index = Fixed(tuple(range(100)),scores)
    cfg = dict(layer=0,index=index,delta=.5)
    locked = dict.fromkeys(('cosine','zscore','current_hierarchy','llr_beam','llr_rescue'))
    locked['flat_llr'] = cfg
    monkeypatch.setattr(C,'fit_study',lambda cache,beam=2,parent_gate=False: locked)
    report,_ = C.analyze_set(fixture(banks=100,facts_per_bank=1))
    scaling = report['false_fire_scaling']
    assert scaling['role'] == 'diagnostic' and not scaling['affects_kill_rules']
    assert scaling['predicted_union'] == pytest.approx(3/64)
    assert scaling['predicted_union'] == scaling['measured']
    assert scaling['evidence_only_prediction'] == pytest.approx(1-(1-2/64)*(1-3/64))
    assert scaling['evidence_only_measured'] == pytest.approx(3/64)
    assert 'predicted_independent' not in scaling

    # The CLI's real-12 extrapolation uses the same distinction.
    twelve = dict(banks=12,arms=dict(flat_llr=dict(locked=dict(validation_per_bank_rates={
        str(b):dict(gate=.001,evidence=.01) for b in range(12)}))))
    monkeypatch.setattr(C,'simulate',lambda cache: cache)
    calls = iter([(twelve,None),(report,None)])
    received = []
    def analyze(cache,beam=2,parent_gate=False):
        received.append(parent_gate)
        return next(calls)
    monkeypatch.setattr(C,'analyze_set',analyze)
    monkeypatch.setattr(C,'kill_rules',lambda results: {})
    path = tmp_path/'features.pt'
    torch.save(fixture(),path)
    output = C.main(['--caches',str(path),'--simulate-100','--c42-parent-gate','--out',str(tmp_path/'report.json')])
    extrapolation = output['results'][-1]['false_fire_scaling']['extrapolation_from_real_12']
    assert extrapolation['predicted'] == pytest.approx(.1)
    assert extrapolation['evidence_only_prediction'] == pytest.approx(1-.99**100)
    assert extrapolation['role'] == 'diagnostic' and received == [True,True]
