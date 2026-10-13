"""CPU tests of C34 scoring, paired controls, real bank IO and nonlinear replay."""
import copy
import json

import pytest
import torch
from torch import nn

from test_e2e_fsa import E, TOK, BOS_T, Tiny, cpu_harness, _c3_pair8_model
import branch_eval as B
import decisive_eval as C


def linked_facts():
    return [E.Fact(0,'Alice','her','she',"sister's name",'Beth'),
            E.Fact(0,'Alice','her','she','instrument','viola'),
            E.Fact(1,'Beth','her','she','instrument','harp'),
            E.Fact(1,'Beth','her','she','color','teal')]


def test_all_four_types_use_real_relation_and_keep_targets():
    facts = linked_facts()
    groups = C.task_items(TOK,facts,BOS_T)
    assert all(groups.values())
    chains = groups['forward_chain']
    assert len(chains) == 2
    assert all(it['fis'][0] == 0 and it['fis'][1] in (2,3) for it in chains)
    assert chains[0]['relation_line'] == "Alice's sister is Beth."
    assert all("sister link from Alice" in it['prompt'] and "Beth's" in it['explicit'] for it in chains)
    assert all(it['vals'] == [facts[it['fis'][1]].val] for it in chains)
    assert not C.task_items(TOK,E.make_world(0,3,2,2),BOS_T)['forward_chain']


def test_opt_in_relation_world_keeps_counts_and_default_world():
    legacy = E.make_world(0,3,8,8)
    assert legacy == E.make_world(0,3,8,8,relations=False)
    linked = E.make_world(0,3,8,8,relations=True)
    assert len(linked) == len(legacy)
    for p in range(3):
        own = [f for f in linked if f.pid == p]
        assert len({f.attr for f in own}) == 8
        assert next(f.val for f in own if f.attr == "sister's name") == E.PEOPLE[(p+1)%3][0]
    assert C.task_items(TOK,linked,BOS_T)['forward_chain']
    with pytest.raises(ValueError,match='two people'):
        E.make_world(relations=True)


def test_rescore_second_sentence_and_value_after_token_40():
    text = 'viola. '+('x'*44)+' harp.'
    ids = TOK.encode(text).ids
    assert not E.score_two_fact_generation(TOK,ids,['viola','harp'])['both']
    assert not E.score_two_fact_lenient(TOK,ids,['viola','harp'])
    assert E.score_two_fact_lenient(TOK,ids,['viola','harp'],max_tokens=64)
    assert not E.score_two_fact_lenient(TOK,TOK.encode('violas and harps.').ids,['viola','harp'],max_tokens=64)
    it = C.make_item(TOK,'Question:',[0,1],['viola','harp'],BOS_T)
    panel = C.summary(TOK,[it],[ids])
    assert panel['presence_stop'] == 0 and panel['presence'] == 1 and panel['max_new'] == 64


def test_lenient_stops_at_eos_and_at_64_tokens(monkeypatch):
    monkeypatch.setattr(E,'EOS_IDS',(2,))
    assert not E.score_two_fact_lenient(TOK,TOK.encode('viola').ids+[2]+TOK.encode(' harp').ids,
                                       ['viola','harp'],eos=E.EOS_IDS,max_tokens=64)
    assert not E.score_two_fact_lenient(TOK,TOK.encode('viola '+('x'*64)+' harp').ids,
                                       ['viola','harp'],max_tokens=64)


def test_format_controls_hold_distractors_fixed_and_keep_one_bos():
    facts = linked_facts()+[E.Fact(2,'Carla','her','she',f'field{i}',f'value{i}') for i in range(24)]
    item = C.task_items(TOK,facts,BOS_T)['forward_chain'][0]
    rendered = {kind:C.evidence_items(TOK,facts,[item],BOS_T,kind)[0] for kind in C.FORMATS}
    for it in rendered.values():
        assert it['p'].count(BOS_T) == 1
        assert it['fis'] == item['fis'] and it['vals'] == item['vals']
        assert it['prompt'].endswith(item['prompt'])
    after = rendered['distractors_after']['prompt'].split('\n\n')[0].splitlines()
    before = rendered['distractors_before']['prompt'].split('\n\n')[0].splitlines()
    assert sorted(after) == sorted(before) and len(after) == 22
    assert before[-2:] == after[:2]
    assert rendered['needed_header']['prompt'].startswith('Fact sheet:\n')
    assert "Alice's sister is Beth." in rendered['needed_only']['prompt']
    assert 'UnrelatedPerson' in rendered['irrelevant_people']['prompt']
    assert 'Carla' not in rendered['irrelevant_people']['prompt']
    with pytest.raises(ValueError,match='20 distinct'):
        C.evidence_items(TOK,linked_facts(),[item],BOS_T,'distractors_after')


def test_paired_uncertainty_and_format_inversion_samples():
    assert C.paired([0,1,0,1],[0,1,1,0])['within_noise']
    assert not C.paired([0]*4,[1]*4)['within_noise']
    assert C.paired([],[])['within_noise'] is None
    facts = linked_facts()+[E.Fact(2,'Carla','her','she',f'field{i}',f'value{i}') for i in range(24)]
    groups = {'conjunction':[C.make_item(TOK,'Question:',[1,2],['viola','harp'],BOS_T)]}
    def generate(items,routes):
        return [TOK.encode('viola and harp.' if 'Carla' in it['prompt'] else 'viola. harp.').ids for it in items]
    report = C.format_report(TOK,facts,groups,BOS_T,generate,2)
    inversion = report['comparisons']['conjunction:distractors_after']
    assert inversion['presence_stop_rescues'] == 1 and inversion['needed_had_values_but_bad_stop'] == 1
    assert inversion['samples'][0]['presence_stop_before'] is False
    assert inversion['samples'][0]['presence_stop_after'] is True


def test_negatives_are_unknown_in_active_scope_and_names_are_near_misses():
    facts = linked_facts()
    banks = {0:dict(meta=dict(train_person=0)),1:dict(meta=dict(train_person=1))}
    groups = C.negative_items(TOK,facts,banks,BOS_T)
    assert all(groups.values())
    for it in groups['near_miss']:
        gold = facts[it['fis'][0]].name
        query_name = it['query_name']
        assert sum(a!=b for a,b in zip(gold,query_name)) == 1 and len(gold)==len(query_name)
        assert query_name not in {f.name for f in facts}
    scopes = C.fact_scopes(facts,banks)
    for it in groups['wrong_scope']:
        known = [fi for fi,f in enumerate(facts) if f.name == it['query_name'] and f.attr in it['prompt']]
        assert known and not scopes[it['bank']].intersection(known)


def test_negative_confidence_uses_generated_prefix_and_all_value_tokens(monkeypatch):
    class Fixed(nn.Module):
        def forward(self,ids):
            out = torch.zeros((*ids.shape,128))
            out[...,ord('v')] = 12
            return out
    it = dict(p=[1,30])
    assert C.wrong_value_probability(Fixed(),TOK,it,[ord('v')],['v'],'cpu') >= .8
    assert C.wrong_value_probability(Fixed(),TOK,it,TOK.encode('viola').ids,['viola'],'cpu') < .8
    assert C.wrong_value_probability(Fixed(),TOK,it,TOK.encode('violas').ids,['viola'],'cpu') == 0
    class Moderate(nn.Module):
        def forward(self,ids):
            out = torch.zeros((*ids.shape,128))
            out[...,ord('v')] = torch.log(torch.tensor(9.*127.))  # p(v)=0.9
            return out
    assert C.wrong_value_probability(Moderate(),TOK,it,TOK.encode('vv').ids,['vv'],'cpu') == pytest.approx(.81)
    assert C.wrong_value_probability(Moderate(),TOK,it,TOK.encode('vvv').ids,['vvv'],'cpu') == pytest.approx(.729)
    assert not C.contains('41.5', '41') and not C.contains('viola-like', 'viola')


def test_negative_report_base_patch_and_no_recall():
    facts = linked_facts()
    banks = {0:dict(meta=dict(train_person=0))}
    active = []
    class Fixed(nn.Module):
        def forward(self,ids):
            out = torch.zeros((*ids.shape,128))
            out[...,ord('v')] = 12
            return out
    # A one-token value makes the confidence probability observable exactly.
    facts[1] = facts[1]._replace(val='v')
    def generate(items,routes):
        return [TOK.encode('v' if route else 'unknown').ids for route in routes]
    report = C.negative_report(Fixed(),TOK,facts,banks,BOS_T,generate,active.append,'cpu',2)
    for panel in report['types'].values():
        assert panel['base']['false_recall'] == panel['base']['confidently_wrong'] == 0
        assert panel['patched']['false_recall'] == panel['patched']['confidently_wrong'] == 1
    assert active[-1] == ()


@pytest.mark.parametrize('scale',[.49,0.,float('nan'),float('inf')])
def test_small_or_nonfinite_patch_scales_refused(scale):
    with pytest.raises(ValueError,match='0.5'):
        C.check_scale(dict(patch_scale_mult=scale))
    with pytest.raises(ValueError):
        C.check_scale(dict(patch_scale_mult_down=scale))


def test_gate_strength_no_blending_and_defaults_unchanged():
    C.check_scale(dict(patch_scale_mult=.5))
    C.check_scale(dict(patch_scale_mult=1.,patch_scale_mult_up=None))
    a = E.build_parser().parse_args(['--method','fsa_forced','--out','unused'])
    assert not a.decisive_eval and not a.relation_world and a.gen_max_new == 0
    assert B.build_parser().parse_args([]).arms == 'B,C,D,E,F,G'
    assert C.positive_control()['passed'] and B.positive_control('PK')['passed']


def test_product_keys_four_banks_pairs_bytes_rng_and_nulls():
    x = torch.eye(4).repeat_interleave(4,0)*4
    labels = [p for p in range(4) for _ in range(4)]
    rng = torch.get_rng_state()
    index = C.ProductKeyIndexer.fit(x,labels,torch.zeros(8,4),width=8)
    assert torch.equal(rng,torch.get_rng_state())
    assert index.left.dtype == index.right.dtype == torch.int8
    assert index.pairs.tolist() == [[0,0],[0,1],[1,0],[1,1]]
    assert index.select(x) == [(p,) for p in labels]
    assert index.select(torch.zeros(3,4)) == [(),(),()]
    assert index.storage_bytes() == sum(v.numel()*v.element_size() for v in vars(index).values() if torch.is_tensor(v))+32
    assert torch.equal(index.projection,C.ProductKeyIndexer.fit(x,labels,torch.zeros(8,4),width=8).projection)


def test_routing_conditioned_recall_uses_actual_expert_gold_answer_tokens():
    model = _c3_pair8_model()
    E.attach(model,[0,1],'fsa_forced',0)
    facts = [E.Fact(0,'Alice','her','she','instrument','viola')]
    it = C.make_item(TOK,'Question:',[0],['viola'],BOS_T)
    E.set_enabled(False)
    with C.dispatch_trace(model) as (routes,coords):
        E.logits_of(model,torch.tensor([it['p']+it['a']]))
        expert = int(routes['moes.0'][len(it['p'])-1,0])
    saved = E.export_patch(model,dict(train_person=0))
    name = f'moes.0.experts.{expert}.up_proj'
    saved['entries'][name]['idx'] = torch.tensor([4],dtype=torch.int32)
    saved['entries'][name]['val'] = torch.tensor([1],dtype=torch.int8)
    # Route analysis reads the file's nonzero codes; keep weights fixed so the
    # test precisely isolates routed vs unselected bank, independent of recall.
    selected = []
    def activate(route):
        selected.append(route)
    good = TOK.encode('viola.').ids
    report = C.conditioned_recall(model,TOK,[it],[good],[(0,)],{0:saved},{0:{0}},activate,'cpu',2)
    assert report['splits']['routed_into_patch']['n'] == 1
    assert report['per_item'][0]['banks'][0]['token_routed'][0]
    assert len(report['per_item'][0]['answer_positions']) == len('viola')
    assert not model.moes[0].router._forward_hooks
    off = C.conditioned_recall(model,TOK,[it],[good],[()],{0:saved},{0:{0}},activate,'cpu',2)
    assert off['splits']['not_routed_into_patch']['n'] == 1
    E.set_enabled(True)


def test_trace_restores_dispatch_on_exception_and_rejects_unsupported():
    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Identity()])
        def _forward_frozen_exact_grouped(self,x,indices,weights):
            return x
    model = Grouped()
    assert '_forward_frozen_exact_grouped' not in model.__dict__
    with pytest.raises(RuntimeError):
        with C.dispatch_trace(model):
            model._forward_frozen_exact_grouped(torch.ones(2,3),torch.zeros(2,1,dtype=torch.long),torch.ones(2,1))
            raise RuntimeError('injected')
    assert '_forward_frozen_exact_grouped' not in model.__dict__
    model._forward_frozen_exact_grouped = None
    with pytest.raises(ValueError,match='unsupported'):
        with C.dispatch_trace(model):
            pass


def nonlinear_branches(layers=1):
    # Real FSA params and grouped expert dispatch: two writes share a hidden
    # unit. Merged relu² has a 2ab cross-term; independent OUTPUT deltas do not.
    class Squared(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = nn.Linear(8,1,bias=False)
            self.down_proj = nn.Linear(1,8,bias=False)
            self.up_proj.weight.data.copy_(torch.tensor([[1.,1.,1.,1.,0.,0.,0.,0.]]))
            self.down_proj.weight.data.fill_(1.)
        def forward(self,x):
            return self.down_proj(self.up_proj(x).relu().square())
    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([Squared()])
        def _forward_frozen_exact_grouped(self,x,indices,weights):
            return self.experts[0](x)
        def forward(self,x):
            return self._forward_frozen_exact_grouped(x,torch.zeros(len(x),1,dtype=torch.long),torch.ones(len(x),1))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = Grouped()
            if layers == 2:
                self.moe2 = Grouped()
        def forward(self,ids):
            x = torch.ones(ids.numel(),8)
            x = self.moe(x)
            if layers == 2:
                x = self.moe2(x)
            return x.reshape(*ids.shape,8)
    model = Model()
    E.attach(model,list(range(layers)),'fsa_forced',0,projs=['up_proj'])
    banks = {}
    for bank,idx in enumerate((4,5)):
        for pm in E.fsa_modules(model):
            pm.M.data.zero_();pm.M.data[0,idx] = 1
        banks[bank] = copy.deepcopy(E.export_patch(model,dict(train_person=bank)))
    return B.Branches(model,banks,{0:1,1:1},'sum_clip')


def test_output_delta_replay_removes_nonlinear_cross_term_and_restores():
    branches = nonlinear_branches()
    model = branches.model
    ids = torch.tensor([[1,2,3]])
    branches.activate(())
    base = E.logits_of(model,ids).clone()
    solo = []
    for bank in branches.states:
        branches.activate((bank,))
        solo.append(E.logits_of(model,ids).clone())
    branches.activate((0,1))
    merged = E.logits_of(model,ids).clone()
    residual = C.IndependentResiduals(branches)
    summed = residual(ids)
    torch.testing.assert_close(summed,base+(solo[0]-base)+(solo[1]-base))
    assert not torch.equal(summed,merged)
    assert residual.forwards == 4 and residual.omitted_rows == 0
    assert all(not m._forward_hooks for m in model.modules())
    branches.activate(())
    assert torch.equal(E.logits_of(model,ids),base)
    E.set_enabled(True)


def test_real_e2e_reload_suite_preserves_patch_and_revocation(cpu_harness):
    run,paths = cpu_harness
    first,patch = run(['--steps','1','--train-person','0'],'c34train')
    original = torch.load(patch,weights_only=False)
    loaded,_ = run(['--steps','0','--train-person','0','--load-patch',str(patch),
                    '--decisive-eval','--qa-n','1'],'c34load')
    report = loaded['decisive_eval']
    assert set(report['ceilings']) == set(C.C34_T)
    assert report['ceilings']['forward_chain']['unavailable']
    assert report['two_fact_rescore']['conditions']['64']['max_new'] == 64
    assert report['control']['passed'] and loaded['revoked']['logits_bit_exact']
    assert first['patched'] == loaded['patched']
    assert torch.equal(torch.load(patch,weights_only=False)['entries'][next(iter(original['entries']))]['val'],
                       original['entries'][next(iter(original['entries']))]['val'])


def test_real_relation_training_reload_rejects_different_world(cpu_harness):
    run,paths = cpu_harness
    result,patch = run(['--steps','1','--relation-world'],'relation_train')
    assert torch.load(patch,weights_only=False)['meta']['relation_world']
    reload,_ = run(['--steps','0','--relation-world','--load-patch',str(patch)],'relation_load')
    assert reload['patched'] == result['patched'] and reload['revoked']['logits_bit_exact']
    with pytest.raises(SystemExit,match='relation_world'):
        run(['--steps','0','--load-patch',str(patch)],'different_world')


def test_branch_suite_pk_and_all_partition_conditions_cpu(cpu_harness,monkeypatch,tmp_path):
    run,paths = cpu_harness
    monkeypatch.setattr(Tiny,'layers',property(lambda self:self.moes),raising=False)
    original_encode = type(TOK).encode
    monkeypatch.setattr(type(TOK),'encode',lambda self,text,**kw: original_encode(self,text.replace('—','-'),**kw))
    files = {'full':[],'block':[],'unit':[]}
    for family in files:
        for person in range(3):
            flags = ['--steps','1','--relation-world','--train-person',str(person),'--slot-frac','.5','--sparse-state']
            if family != 'full':
                flags += ['--slot-partition','3','--slot-part',str(person)]
            if family == 'unit':
                flags += ['--partition-unit']
            result,patch = run(flags,f'{family}_{person}')
            files[family].append(str(patch))
    B.Path(paths['anchor']).write_text(json.dumps([40]*6080))
    cli = ['--arms','C34,PK','--patches',*files['full'],'--partition-patches',*files['block'],
           '--unit-patches',*files['unit'],'--anchor',paths['anchor'],'--eval-set',paths['eval'],
           '--fineweb',paths['extra'],'--code',paths['extra'],'--fast-moe','off','--indexer-layers','0',
           '--selector-l2','.01','--dg-steps','10','--dg-expansion','1','--pk-width','8',
           '--qa-n','1','--g-n','1','--gen-max-new','1','--out',str(tmp_path/'branch.json')]
    B.main(cli)
    result = json.loads((tmp_path/'branch.json').read_text())
    assert result['revoked']['logits_bit_exact']
    manifest = result['comparison_manifest']
    assert len(manifest['patches']) == 9
    assert all(p['sha256'] and p['mask_sha256'] and p['bytes'] for p in manifest['patches'])
    assert set(manifest['selector_bytes']) == {'PK:PK','PK:I','PK:DG'}
    report = result['decisive_eval']['resident']
    assert report['ceilings']['forward_chain']['as_trained']['n'] == 1
    decomposition = report['decomposition']
    assert set(decomposition['families']) == {'block','whole_unit'}
    for family in decomposition['families'].values():
        assert len(family['each_bank_alone']) == 3
        assert family['merged']['n'] == family['per_layer_pinned_independent_outputs']['n'] == 7
        assert family['paired_comparison']['same_bank_files']
        assert family['multi_bank_items'] == 2
        assert family['paired_comparison']['within_noise'] is None
        assert not family['paired_comparison']['pure_merge_nonlinearity_interpretation']
    pk = result['arms']['PK']
    assert set(pk['selectors']) == {'PK','I','DG'}
    assert len({s['address_bytes'] for s in pk['selectors'].values()}) == 1
    for kind,selector in pk['selectors'].items():
        assert selector['address_bytes'] == selector['useful_address_bytes']+selector['reserve_bytes']
        assert selector['locked'] and selector['selection_split'] == 'VAL only'
        assert set(selector['damage']) == {'WikiText','FineWeb','code'}
        assert selector['selection']['n'] == 6*len(E.TEST_T)
    # TEST cannot influence selected PK/I/DG settings.
    monkeypatch.setattr(E,'TEST_T',['New test wording {name} {attr}? Answer: {val}.'])
    cli[1] = 'PK'
    B.main(cli)
    poisoned = json.loads((tmp_path/'branch.json').read_text())['arms']['PK']
    for kind in pk['selectors']:
        for key in ('layer','l2','threshold','validation_score','address_bytes'):
            assert poisoned['selectors'][kind][key] == pk['selectors'][kind][key]


def test_residual_forward_failure_removes_hooks_and_revokes(monkeypatch):
    branches = nonlinear_branches()
    ids = torch.tensor([[1,2]])
    branches.activate(())
    original = E.logits_of
    base = original(branches.model,ids).clone()
    calls = 0
    def broken(model,toks):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('solo forward failed')
        return original(model,toks)
    monkeypatch.setattr(E,'logits_of',broken)
    with pytest.raises(RuntimeError,match='solo forward'):
        C.IndependentResiduals(branches)(ids)
    assert all(not m._forward_hooks for m in branches.model.modules())
    assert '_forward_frozen_exact_grouped' not in branches.model.moe.__dict__
    assert torch.equal(original(branches.model,ids),base)
