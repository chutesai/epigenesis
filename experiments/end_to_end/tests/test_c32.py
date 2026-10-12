"""CPU regressions for review selection, binding, matching and uncertainty safeguards."""
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, Tiny, cpu_harness  # noqa: F401
from test_c23 import OffsetTok, PointerModel
import branch_eval as B
import biology_addressing as BIO
import two_hop as H
import exploration as X
import review_eval as R


def world():
    return [E.Fact(p,name,'her','she',attr,value)
            for p,name,values in [(0,'Alice',['viola','North','doctor']),
                                  (1,'Beth',['harp','East','painter']),
                                  (2,'Clara',['flute','West','writer'])]
            for attr,value in zip(['instrument','stop','job'],values)]


@pytest.mark.parametrize('text,correct', [
    ("Alice's instrument is viola; Beth's instrument is harp.", True),
    ("No, Alice's instrument is viola while Beth's is harp.", True),
    ("Alice's instrument is harp; Beth's instrument is viola.", False),
    ("Alice's instrument is not viola; Beth's instrument is not harp.", False),
    ('viola and harp', False),
    ("Not Alice's instrument is viola; Beth's instrument is harp.",False),
    ("Alice's instrument is viola; Beth's instrument is harp. Alice's instrument is harp.", False),
])
def test_binding_rejects_swaps_negations_unattributed_and_contradictory_answers(text,correct):
    assert R.binding_score(text,dict(fis=[0,3]),world()) is correct


def test_same_person_attribute_bindings_and_supplied_evidence():
    facts = world()
    it = dict(fis=[0,1],p=E.encode(TOK,'What are her instrument and stop?',1))
    assert R.binding_score("Alice's instrument is viola; Alice's stop is North.",it,facts)
    assert not R.binding_score("Alice's instrument is North; Alice's stop is viola.",it,facts)
    assert not R.binding_score("Alice's instrument is viola; Alice's stop is North; Alice's instrument is North.",it,facts)
    out = R.supplied_evidence(TOK,[it],facts,E.encode,1)[0]
    assert 'Evidence: Alice' in TOK.decode(out['p']) and 'viola' in TOK.decode(out['p'])
    assert out['fis']==it['fis'] and out['p']!=it['p']


def test_exact_one_sided_binomial_bounds_and_insufficient_samples():
    report = R.false_activation([()]*3000)
    assert report['upper_95']==pytest.approx(1-.05**(1/3000),abs=1e-12)
    assert report['upper_95'] < .001 and report['sufficient']
    assert R.false_activation([()]*40)['upper_95'] > .07
    assert not R.false_activation([()]*2999)['sufficient']
    assert R.false_activation([])['upper_95']==1
    assert R.false_activation([(0,)]*3000)['lower_95']==pytest.approx(.05**(1/3000))
    with pytest.raises(ValueError):
        R.binomial_bound(2,1)


def test_independent_queries_never_resample_or_overlap_and_deduplicate_sources():
    source = list(range(40*511+3000*63))
    queries = R.independent_queries({'a':source,'duplicate':source},1)
    assert len(queries)==3000
    assert all(q['start']>=40*511 and q['source']=='a' for q in queries)
    assert all(b['start']-a['start']==63 for a,b in zip(queries,queries[1:]))
    assert len(R.independent_queries({'a':[7]*25000},1))==1


def test_heldout_attribute_never_enters_selector_training_and_shared_panel_is_balanced(monkeypatch):
    facts = world()
    model = SimpleNamespace()
    monkeypatch.setattr(B,'indexer_features',lambda model,seqs,*args:torch.ones(len(seqs),3))
    train, x, g = B.address_training(model,TOK,facts,list(range(8000)),0,0,'cpu')
    assert all(facts[it['fi']].attr not in R.heldout_attributes(facts) for it in train)
    assert {facts[it['fi']].pid for it in train}=={0,1,2}
    items = E.build_items(TOK,facts,E.TEST_T,1)
    routes = [(facts[it['fi']].pid,) for it in items]
    result = R.routing_panels(items,facts,routes,B.selection_metrics)
    assert result['shared_attribute']['accuracy']==result['held_out_attribute']['accuracy']==1
    assert result['held_out_attribute']['n']==3*len(E.TEST_T)


def test_router_shared_attribute_and_unseen_attribute_features():
    # Every person has the same two TRAIN attributes. Novel attribute keeps entity signal.
    x = torch.cat([torch.eye(4)[:3]*4+torch.tensor([0.,0.,0.,a]) for a in (-1.,1.)])
    g = torch.zeros(8,4)
    router = B.fit_router(torch.cat((x,g)),[0,1,2]*2+[3]*8,4,100)
    novel = torch.eye(4)[:3]*5+torch.tensor([0.,0.,0.,3.])
    assert router.predict(novel).tolist()==[0,1,2]
    assert router.predict(g).tolist()==[3]*8


def test_full_block_mapping_includes_dense_blocks_and_c_matches_i(monkeypatch):
    model = Tiny()
    # A dense identity precedes each MoE. Hook locations must include these full blocks.
    blocks = nn.ModuleList([nn.Identity(),model.moes[0],nn.Identity(),model.moes[1]])
    monkeypatch.setattr(Tiny,'layers',property(lambda self:blocks),raising=False)
    mapping = B.layer_mapping(model)
    assert [row['full_block'] for row in mapping]==[1,3]
    prompt = [1,40,50]
    torch.testing.assert_close(B.base_features(model,[prompt],[1,3],'cpu')[3],
                               B.indexer_features(model,[prompt],3,'cpu'))
    assert not any(b._forward_pre_hooks for b in blocks)


def test_damage_and_benchmark_follow_routes_and_match_base_batch_size(monkeypatch):
    class Banks:
        model=object()
        states={0:None,1:None}
        active=()
        def activate(self,route): self.active=route
        def active_bytes(self,route): return len(route)*8
    banks = Banks()
    calls=[]
    def nll(model,wins,dev,bs=4):
        calls.append((banks.active,bs))
        return [1.+len(banks.active)]*len(wins)
    monkeypatch.setattr(E,'nll_windows',nll)
    monkeypatch.setattr(E,'logits_of',lambda *args:torch.zeros(1))
    monkeypatch.setattr(E,'qa_probe_eval',lambda *args:{})
    wins={'text':torch.zeros(2,5,dtype=torch.long)}
    a=SimpleNamespace(gen_max_new=1,gen_stop_k=2,gen_bs=1)
    base=B.evaluate_branch_base(banks.model,TOK,wins,{},None,'cpu',a)
    report=B.damage(banks,wins,base['base_nll'],{'text':[(),(0,)]},'cpu')
    assert calls==[((),1),((),1),((0,),1)]
    assert report['text']['dnll']==.5
    observed=[]
    def bench(model,groups,dev):
        observed.append(banks.active)
        return {name:dict(correct=[int(not banks.active)],predicted=[0]) for name in groups}
    monkeypatch.setattr(E,'benchmark_eval',bench)
    result=B.routed_benchmark(banks,{'b':[dict(p=[1]),dict(p=[2])]},
                             {'b':dict(correct=[1,1])},lambda items:[(),(1,)],'cpu')
    assert observed==[(),(1,)] and banks.active==()
    assert result['b']['lost']==1


def test_ac_reports_oracle_success_headroom_and_per_task_ceiling():
    def score(comp,hop):
        return dict(two_fact=dict(n=10,binding=comp),g_set=dict(n=10,binding=comp),two_hop=dict(n=10,recall=hop))
    result=B.ceiling_summary(dict(as_trained=score(.95,.2),oracle_entity=score(1.,.4)))
    assert result['per_task']['two_fact']['headroom']==pytest.approx(.05)
    assert result['per_task']['two_fact']['absolute_oracle_success']==1
    assert result['per_task']['two_fact']['decision']=='insufficient headroom'
    assert result['per_task']['two_hop']['decision']=='addressing rescue'
    ceiling=B.ceiling_summary(dict(as_trained=score(.99,1.),oracle_entity=score(1.,1.)))
    assert all(row['decision']=='ceiling reached' for row in ceiling['per_task'].values())
    assert not ceiling['value_use_bottleneck']


def test_dg_dense_logistic_uses_same_supervision_budget_and_validation_search():
    x=torch.eye(6)[:3].repeat_interleave(4,0)*4
    g=torch.zeros(64,6)
    y, attrs=[0]*4+[1]*4+[2]*4,['shared']*12
    rng=torch.get_rng_state()
    sparse=B.DGIndexer.fit(x,y,attrs,g,expansion=8,sparsity=.15,steps=100)
    dense=B.DenseLogisticIndexer.fit(x,y,attrs,g,steps=60)
    for index in (sparse,dense):
        assert index.select(x,top_k=1)==[(p,) for p in y]
        assert not any(index.select(g,top_k=1))
        assert index.storage_bytes()>=sum(v.numel()*v.element_size() for v in vars(index).values() if torch.is_tensor(v))
    assert dense.projection is None and torch.equal(rng,torch.get_rng_state())
    assert R.select_validation([('first',.5),('second',.5)],lambda c:c[1])[0]=='first'


def test_conj_always_abstaining_and_dirty_damage_cannot_pass_substitution_branch():
    def report(recall,subs,binding,label='clean'):
        return dict(end_to_end=dict(test=dict(recall=recall,cross_person_substitutions=subs)),
                    composition=dict(g_set=dict(binding=binding)),damage_gate=dict(text=dict(label=label)))
    old=report(.9,.2,.2)
    assert not BIO.kill_decision('CONJ',dict(conjunctive=report(0,0,.3),additive=old))['passed']
    assert not BIO.kill_decision('CONJ',dict(conjunctive=report(.88,0,.3),additive=old))['passed']
    assert not BIO.kill_decision('CONJ',dict(conjunctive=report(.9,0,.3,'fail'),additive=old))['passed']
    assert BIO.kill_decision('CONJ',dict(conjunctive=report(.89,.1,.2),additive=old))['passed']


def test_inh_small_samples_do_not_establish_reduction():
    def report(events,n):
        return dict(false_activations=dict(general=R.false_activation([(0,)]*events+[()]*(n-events))),
                    end_to_end=dict(test=dict(recall=.9)))
    assert not BIO.kill_decision('INH',dict(inhibitory=report(0,40),positive_only=report(1,40)))['passed']
    assert BIO.kill_decision('INH',dict(inhibitory=report(0,3000),positive_only=report(100,3000)))['passed']


def test_slot_splits_balance_targets_include_named_nulls_and_dense_has_identical_trainable_bytes():
    panels={split:BIO.mention_items(OffsetTok(),world(),1,split) for split in ('train','val','test')}
    assert all({tuple(it['q']) for it in panels[a]}.isdisjoint(tuple(it['q']) for it in panels[b])
               for a,b in [('train','val'),('train','test'),('val','test')])
    for items in panels.values():
        positive=[it for it in items if it['target'] is not None]
        assert sum(it['target_recent'] for it in positive)==len(positive)//2
        assert any(it['target'] is None and len(it['slots'])==2 for it in items)
    slots=torch.eye(4)[:2][None].repeat(12,1,1)
    late=torch.stack([slots[i,i%2] if i<8 else torch.tensor([0.,0.,1.,0.]) for i in range(12)])
    items=[dict(target=i%2 if i<8 else None,slots=[dict(pid=0),dict(pid=1)]) for i in range(12)]
    models=[cls.fit(late,slots,items,80) for cls in (BIO.SlotResolver,BIO.DenseResolver)]
    expected=[(i%2,) if i<8 else () for i in range(12)]
    assert all(m.select(late,slots,items)==expected for m in models)
    sizes=[sum(v.numel()*v.element_size() for v in vars(m).values() if torch.is_tensor(v)) for m in models]
    assert sizes[0]==sizes[1] and not hasattr(models[1],'reserve')


def test_pointer_attribute_combination_splits_and_explicit_name_resolution():
    facts=world()
    items=H.pointer_items(TOK,facts,{0:1,1:2,2:0},H.TRAIN,1)
    panels=H.pointer_split(items)
    train={(it['source'],it['attr']) for it in panels['train']}
    assert train.isdisjoint((it['source'],it['attr']) for it in panels['held_out_combination'])
    assert {it['attr'] for it in panels['train']}.isdisjoint(it['attr'] for it in panels['held_out_attribute'])
    it=items[0]
    resolved=H.resolved_name_item(TOK,facts,it,1)
    text=TOK.decode(resolved['q'])
    assert facts[it['fi']].name in text and 'sister' not in text and it['val'] not in text


def test_cleanup_beam_gain_requires_matched_two_read_improvement():
    row=lambda value:dict(end_to_end=dict(recall=value))
    reports=dict(raw_cascade=row(.2),raw_beam=row(.6),cleanup=row(.2),cleanup_beam=row(.6),oracle_pointer=row(.8))
    assert not H.cleanup_kill(reports)['passed']
    reports['cleanup_beam']=row(.7)
    assert H.cleanup_kill(reports)['passed']


def test_pointer_damage_routes_active_adapter_and_restores_logits_hooks(monkeypatch):
    model=PointerModel()
    patch=H.PointerPatch(torch.zeros(3),torch.ones(3),3)
    codes={i:torch.eye(3)[i] for i in range(3)}
    index=B.BankIndexer.fit(torch.eye(3),[0,1,2],torch.zeros(1,3))
    class Banks:
        active=()
        def activate(self,route): self.active=route; E.set_enabled(True)
    banks=Banks()
    monkeypatch.setattr(E,'logits_of',lambda model,x:model.embed(x))
    seen=[]
    def nll(model,wins,dev,bs):
        seen.append((bs, bool(model.layers[0]._forward_pre_hooks),banks.active))
        return [1.]*len(wins)
    monkeypatch.setattr(E,'nll_windows',nll)
    result=H.preservation(model,banks,patch,index,0,2,torch.tensor([[0,0],[1,1]]),'cpu')
    assert result['revoked_logits_bit_exact'] and result['dnll']==0
    assert seen[0]==(1,False,()) and all(row[1] and row[2] for row in seen[1:])
    assert not any(layer._forward_pre_hooks for layer in model.layers)


@pytest.mark.parametrize('arm',['B','C','D','E','F','G','I','AC','DG','INH','CONJ','SLOT'])
def test_positive_controls_and_control_failure_suppress_verdict(arm):
    assert B.positive_control(arm)['passed']
    assert R.gated_verdict(dict(passed=False,kind='fixture'),dict(passed=True))['verdict']=='control-failed'


def test_control_only_clis_do_not_load_banks_and_failed_control_is_loud(monkeypatch,capsys):
    monkeypatch.setattr(B,'positive_control',lambda arm:dict(passed=False,kind='injected'))
    result=B.main(['--arms','C','--positive-control'])
    assert result['verdict']=='control-failed' and 'control-failed' in capsys.readouterr().out
    assert H.main(['--positive-control'])['verdict']=='control-passed'
    monkeypatch.setattr(X,'positive_control',lambda arm:dict(passed=False))
    reports=X.main(['--arm','vsa','--positive-control'])
    assert reports[0]['verdict']=='control-failed' and reports[0]['result'] is None


def test_interference_zero_writes_preserve_base_logits_and_count_reference_forward(monkeypatch):
    from contextlib import contextmanager
    class Banks:
        states={0:None,1:None}
        model=None
        active=()
        def activate(self,route): self.active=route
    banks=Banks()
    @contextmanager
    def weighted(branches,patch):
        yield
    monkeypatch.setattr(X,'weighted_forward',weighted)
    monkeypatch.setattr(X,'signed_patch',lambda *args:None)
    monkeypatch.setattr(E,'logits_of',lambda model,tokens:torch.tensor([[[1.,7.,2.]]]))
    monkeypatch.setattr(E,'EOS_IDS',set())
    generated,forwards=X.interference_generate(banks,[dict(p=[1])],[(0,)],'cpu',1,0,math.inf)
    assert generated==[[1]] and forwards==3
    assert banks.active==()


def test_cleanup_preservation_abstains_midpoint_and_beam_reads_own_answers(monkeypatch):
    model=PointerModel()
    patch=H.PointerPatch(torch.zeros(3),torch.ones(3),3)
    with torch.no_grad():
        patch.up.bias.copy_(torch.tensor([0.,1.,0.]))
    codes={0:torch.tensor([2.,0.,0.]),1:torch.tensor([0.,2.,0.])}
    index=B.BankIndexer.fit(torch.stack(list(codes.values())),[0,1],torch.zeros(1,3))
    class Banks:
        active=()
        def activate(self,route): self.active=route; E.set_enabled(True)
    banks=Banks()
    seen=[]
    monkeypatch.setattr(E,'logits_of',lambda model,tokens:model.embed(tokens))
    monkeypatch.setattr(E,'nll_windows',lambda model,wins,dev,bs:[1.]*len(wins))
    monkeypatch.setattr(E,'greedy_generate',lambda *args,**kwargs:[[10+banks.active[0]]])
    def tf(model,items,dev,bs):
        assert items[0]['a']==[10+banks.active[0]]
        seen.append(banks.active[0])
        return dict(answer_logprob=float(banks.active[0]))
    monkeypatch.setattr(E,'tf_eval',tf)
    wins=torch.tensor([[0,0]])
    clean=H.preservation(model,banks,patch,index,0,2,wins,'cpu',codes,'cleanup',1.,1.)
    assert not seen and clean['revoked_logits_bit_exact']
    beam=H.preservation(model,banks,patch,index,0,2,wins,'cpu',codes,'cleanup_beam',1.,1.)
    assert seen==[0,1] and beam['revoked_logits_bit_exact']
    assert not any(layer._forward_pre_hooks for layer in model.layers)
