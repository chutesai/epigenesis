"""Review regression probes for matched budgets, truthful scores and dispatch."""
import copy

import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T, cpu_harness
from test_c34 import nonlinear_branches, linked_facts
import branch_eval as B
import decisive_eval as C
import review_eval as R


@pytest.mark.parametrize('text',[
    'violin, cello, guitar, flute, oboe, piano, viola and harp.',
    'It is definitely not viola and not harp.',
    'harp and viola.',
    'viola, harp, cello.',
    'viola and viola and harp.',
    'violas and harps.',
    'viola-like and harp.',
])
def test_primary_rejects_listing_negation_swap(text):
    assert not E.score_two_fact_correctness(TOK,TOK.encode(text).ids,['viola','harp'],
        ['violin','cello','guitar','flute','oboe','piano','viola','harp'])


@pytest.mark.parametrize('text',['viola and harp.','viola, harp.',' viola and harp','VIOLA AND HARP.'])
def test_primary_accepts_terse_gold_without_names(text):
    assert E.score_two_fact_correctness(TOK,TOK.encode(text).ids,['viola','harp'])


def test_candidate_union_and_repeated_or_nested_values():
    assert not R.ordered_two_value_score('viola, cello and harp.',['viola','harp'],['cello'])
    assert R.ordered_two_value_score('New York and York.',['New York','York'])
    assert R.ordered_two_value_score('viola and viola.',['viola','viola'])
    assert not R.ordered_two_value_score('viola.',['viola','viola'])
    assert not R.ordered_two_value_score('41.5 and harp.',['41','harp'])


def test_summary_headlines_correctness_and_names_presence_diagnostics():
    item = C.make_item(TOK,'Give instrument, then color, in order:',[1,3],['viola','teal'],BOS_T)
    gens = [TOK.encode('viola and teal.').ids, TOK.encode('teal and viola.').ids]
    panel = C.summary(TOK,[item,item],gens,facts=linked_facts(),scopes={0:{1}})
    assert panel['correct'] == .5 and panel['presence_stop'] == panel['presence'] == 1
    assert panel['primary_metric'] == 'correct'
    assert not {'GEN','strict','lenient','binding'} & panel.keys()
    assert panel['supported']['n'] == 0 and panel['unsupported']['n'] == 2


def test_new_templates_cross_person_items_and_balanced_pronouns():
    facts = E.make_world(0,3,16,16)
    groups = C.task_items(TOK,facts,BOS_T,n=10000)
    for it in groups['cross_person_conjunction']:
        assert len({facts[fi].pid for fi in it['fis']}) == 2
        assert 'in that order' in it['prompt']
    assert len(groups['pronoun_earlier']) == len(groups['pronoun_later']) > 0
    assert {tuple(it['fis']) for it in groups['pronoun_earlier']} == {tuple(it['fis']) for it in groups['pronoun_later']}
    capped = C.task_items(TOK,facts,BOS_T,n=2)
    assert [it['fis'] for it in capped['pronoun_earlier']] == [it['fis'] for it in capped['pronoun_later']]
    assert C.template_stem(C.C34_T['conjunction']) != C.template_stem(E.SELF_STUDY_T[3])
    scopes = C.fact_scopes(facts,{p:dict(meta=dict(train_person=p)) for p in range(3)})
    assert all(len(r)>1 for r in C.oracle_routes(groups['cross_person_conjunction'],scopes))
    paired = C.paired_panel([False]*len(groups['conjunction']),[True]*len(groups['conjunction']),groups['conjunction'],facts)
    assert paired['clusters'] == dict(facts=48,people=3,relations=0)
    assert paired['paired_se'] is paired['within_noise'] is None


@pytest.mark.parametrize('people,facts_per_person',[(3,16),(2,2)])
def test_every_feasible_negative_category_has_minimum_unique_n(people,facts_per_person):
    facts = E.make_world(0,people,facts_per_person,facts_per_person,relations=True)
    banks = {p:dict(meta=dict(train_person=p)) for p in range(people)}
    groups = C.negative_items(TOK,facts,banks,BOS_T,n=1)
    forbidden = {f.name.casefold() for f in facts} | {f.val.casefold() for f in facts}
    assert groups == C.negative_items(TOK,facts,banks,BOS_T,n=1)
    for kind,items in groups.items():
        for bank in banks:
            assert len({it['prompt'] for it in items if it['bank']==bank}) >= 16
        for it in items:
            if kind in ('near_miss','untaught_entity'):
                assert it['query_name'].casefold() not in forbidden
    positions = set()
    for it in groups['near_miss']:
        name = facts[it['fis'][0]].name
        positions.update(i for i,(a,b) in enumerate(zip(name,it['query_name'])) if a!=b)
    assert len(positions)>1


def test_negative_relation_name_echo_and_generic_hallucination():
    facts = linked_facts()
    assert not any(C.contains('Beth is not in my records.',v) for v in C.negative_values(facts,'Beth'))
    assert not any(C.contains('Alice is not in my records.',v) for v in C.negative_values(facts,'Alice'))
    assert not C.asserted_answer('Beth is not in my records.')
    assert not C.asserted_answer("I don't know. Perhaps viola.")
    assert not C.asserted_answer('')
    assert C.asserted_answer('The answer is inventedvalue.')


def test_confidence_geometric_mean_removes_value_token_length_bias():
    class Model(nn.Module):
        def forward(self,ids):
            logits = torch.zeros((*ids.shape,128))
            logits[...,ord('v')] = torch.log(torch.tensor(9.*127.))
            return logits
    short = C.wrong_value_confidence(Model(),TOK,dict(p=[1,30]),TOK.encode('vv').ids,['vv'],'cpu')
    long = C.wrong_value_confidence(Model(),TOK,dict(p=[1,30]),TOK.encode('vvvvv').ids,['vvvvv'],'cpu')
    assert short['joint'] == pytest.approx(.81)
    assert long['joint'] == pytest.approx(.9**5)
    assert short['geometric_mean'] == long['geometric_mean'] == pytest.approx(.9)


def test_negative_report_separates_invention_from_stored_confident_leakage():
    class Model(nn.Module):
        def forward(self,ids):
            logits = torch.zeros((*ids.shape,128))
            logits[...,ord('v')] = torch.log(torch.tensor(9.*127.))
            return logits
    facts = [E.Fact(0,'Alice','her','she','color','vvvvv'),
             E.Fact(1,'Beth','her','she','instrument','cello')]
    banks = {0:dict(meta=dict(train_person=0))}
    active = []
    def generate(items,routes):
        return [TOK.encode('vvvvv' if route else 'inventedanswer.').ids for route in routes]
    report = C.negative_report(Model(),TOK,facts,banks,BOS_T,generate,active.append,'cpu',1)
    assert 'forced-bank leakage' in report['policy']
    for panel in report['types'].values():
        assert panel['base']['n'] >= 16 and panel['patched']['n'] >= 16
        assert panel['base']['false_recall'] == panel['base']['confidently_wrong'] == 0
        assert panel['base']['asserted_answer'] == 1
        assert panel['patched']['false_recall'] == panel['patched']['confidently_wrong'] == 1
        assert panel['patched']['per_item'][0]['wrong_value_joint_p'] < .8
    assert active[-1] == ()


def test_negative_report_rejects_undersized_category(monkeypatch):
    item = dict(bank=0,prompt='Question:',query_name='Nobody',p=[1])
    monkeypatch.setattr(C,'negative_items',lambda *args: {'untaught_entity':[item]})
    with pytest.raises(AssertionError,match='minimum'):
        C.negative_report(None,TOK,linked_facts(),{0:dict(meta=dict(train_person=0))},
            BOS_T,None,None,'cpu',1)


def test_two_layer_pins_erase_cross_layer_propagation():
    branches = nonlinear_branches(layers=2)
    ids = torch.tensor([[1,2]])
    branches.activate(())
    base = E.logits_of(branches.model,ids).clone()
    solos = []
    for bank in branches.states:
        branches.activate((bank,))
        solos.append(E.logits_of(branches.model,ids).clone())
    pinned = C.IndependentResiduals(branches)(ids)
    torch.testing.assert_close(pinned,base+sum(s-base for s in solos))
    # An additive replay would propagate the first layer's 34-valued output
    # through the second base expert (136^2), then add its solo output deltas.
    propagated = torch.full_like(base,136.**2)+sum(s-base for s in solos)
    assert not torch.equal(pinned,propagated)
    assert 'cross-layer' in C.IndependentResiduals.__doc__
    assert all(not module._forward_hooks for module in branches.model.modules())


@pytest.mark.parametrize('failure',['empty_dispatch','bypass_linear','short_rows'])
def test_residual_and_conditioned_recall_fail_closed_and_cleanup(monkeypatch,failure):
    branches = nonlinear_branches()
    moe = branches.model.moe
    if failure == 'empty_dispatch':
        monkeypatch.setattr(moe,'forward',lambda x: x)
    elif failure == 'bypass_linear':
        monkeypatch.setattr(moe,'_forward_frozen_exact_grouped',lambda x,ids,weights: x)
    else:
        # Dispatch two rows, but call the touched Linears with only one row.
        monkeypatch.setattr(moe,'_forward_frozen_exact_grouped',lambda x,ids,weights: moe.experts[0](x[:1]))
    item = C.make_item(TOK,'Question:',[0],['viola'],BOS_T)
    with pytest.raises(ValueError,match='dispatch|rows'):
        C.IndependentResiduals(branches)(torch.tensor([[1,2]]))
    with pytest.raises(ValueError,match='dispatch|rows'):
        C.conditioned_recall(branches.model,TOK,[item],[TOK.encode('viola.').ids],[(0,)],
            branches.states,{0:{0},1:{1}},branches.activate,'cpu',2)
    assert all(not module._forward_hooks for module in branches.model.modules())
    assert branches.model._c16_branch_key[1] == ()


def test_training_recipe_saved_and_mismatches_refused(cpu_harness):
    run,paths = cpu_harness
    _,path = run(['--steps','1'],'c36_recipe')
    state = torch.load(path,weights_only=False)
    meta = state['meta']
    assert meta['step'] == meta['value_steps'] == 1 and meta['wake_steps'] == 0
    assert meta['training_recipe']['fact_batch'] == 3
    assert meta['mask_sha256'] and meta['training_exposures']['examples_per_fact']
    B.validate_training_match([state,copy.deepcopy(state)])
    for key in ('step','wake_steps','value_steps'):
        other = copy.deepcopy(state); other['meta'][key] += 1
        with pytest.raises(ValueError,match='differ'):
            B.validate_training_match([state,other])
    for key in ('budget','bop_gamma','aug','local_lambda','fact_batch'):
        other = copy.deepcopy(state); other['meta']['training_recipe'][key] = 'different'
        with pytest.raises(ValueError,match='differ'):
            B.validate_training_match([state,other])
    other = copy.deepcopy(state); other['meta']['training_exposures']['examples_per_fact'][0] += 1
    with pytest.raises(ValueError,match='differ'):
        B.validate_training_match([state,other])
    other = copy.deepcopy(state); del other['meta']['training_recipe']
    with pytest.raises(ValueError,match='recorded'):
        B.validate_training_match([state,other])


def test_last_write_sorted_bank_id_independent_of_cli_order():
    branches = nonlinear_branches()
    first,second = [copy.deepcopy(branches.states[p]) for p in (0,1)]
    name = next(iter(first['entries']))
    first['entries'][name]['idx'] = torch.tensor([4],dtype=torch.int32)
    first['entries'][name]['val'] = torch.tensor([1],dtype=torch.int8)
    second['entries'][name]['idx'] = torch.tensor([4],dtype=torch.int32)
    second['entries'][name]['val'] = torch.tensor([-1],dtype=torch.int8)
    forward = B.Branches(branches.model,{0:first,1:second},{0:1,1:1},'last_write')
    reversed_ = B.Branches(branches.model,{1:second,0:first},{0:1,1:1},'last_write')
    forward.activate((1,0)); reversed_.activate((0,1))
    assert torch.equal(forward.cache[(0,1)]['entries'][name]['val'],reversed_.cache[(0,1)]['entries'][name]['val'])
