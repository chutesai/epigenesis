"""C52 joint bank training and intervention tests on the real tiny fixture."""
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, tiny, cpu_harness, _pair8_tiny_weight
import c52 as C
import decisive_eval as D


def base():
    model = tiny()
    for i,moe in enumerate(model.moes):
        for j,expert in enumerate(moe.experts):
            for lin in (expert.up_proj,expert.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(i*4+j,*lin.weight.shape))
    return model


def states(budget=20, fraction=1.):
    result = []
    for person in range(3):
        model = base()
        E.attach(model,[0,1],'fsa_forced',0,sparse=True,budgeted=bool(budget),
                 slot_partition=3,slot_part=person,slot_frac=fraction,seed=0)
        for pm in E.fsa_modules(model):
            with torch.no_grad():
                pm.M[:1] = 1
                if budget:
                    pm.support[:1] = True
        meta = dict(method='fsa_forced',fmt='ternary',seed=0,layers=[0,1],projs=['up_proj','down_proj'],
            slot_partition=3,slot_part=person,train_person=person,budget=budget,sparse_state=True,
            slot_frac=fraction,people=3,facts_per_person=2,shared_attrs=2,values='disjoint',
            relation_world=True,export=str(E.EXPORT),export_step=E.EXPORT_STEP,step=4,
            training_recipe=dict(optim='bop',bop_tau_abs=.007,bop_gamma=.05))
        result.append(E.export_patch(model,meta))
    return result


def args(*flags):
    a = E.build_parser().parse_args(['--method','fsa_forced','--optim','bop','--out','unused',*flags])
    a._explicit_flags = {f.split('=')[0] for f in flags if f.startswith('--')}
    return a


@pytest.mark.parametrize('flags',[
    ['--presence-dropout','.2'], ['--joint-banks','a','b','--p-iso','nan'],
    ['--joint-banks','a','b','--presence-dropout','-1'],
    ['--joint-banks','a','b','--max-distractors','2'],
    ['--joint-banks','a','a'], ['--joint-banks','a','b','--joint-negative-n','999'],
    ['--joint-banks','a','b','--optim','adam'], ['--joint-banks','a','b','--ckpt-every','1'],
    ['--joint-banks','a','b','--offtopic-kl','.03']])
def test_validation(flags):
    with pytest.raises(SystemExit):
        C.validate(args(*flags))


def test_sampler_distribution_full_cardinality_and_arm_independence():
    schedule = C.Schedule(17,4,.75,None,.2)
    reference = C.Schedule(17,4,.75,None,.2)
    k = {1:0,2:0,3:0}
    dropped = isolated = 0
    for i in range(20000):
        row = schedule.draw(i%4)
        assert row == reference.draw(i%4)
        assert (i%4 in row['active']) != row['dropped']
        assert len(row['active']) == len(set(row['active']))
        dropped += row['dropped']
        isolated += row['isolated']
        if not row['isolated']:
            k[len(row['active'])-(not row['dropped'])] += 1
    assert isolated/20000 == pytest.approx(.75,abs=.015)
    assert dropped/20000 == pytest.approx(.2,abs=.015)
    assert k[1] == k[2] == 0 and k[3] == 20000-isolated
    assert schedule.report()['distractor_mode'] == 'all'
    capped = C.Schedule(1,12,0,3,0,distractor_mode='uniform')
    assert {len(capped.draw(0)['active'])-1 for _ in range(100)} == {1,2,3}


def test_uniform_distractor_mode_retains_uniform_k_distribution():
    schedule = C.Schedule(17,4,0,None,0,distractor_mode='uniform')
    for _ in range(20000):
        schedule.draw(0)
    counts = schedule.report()['counts']
    for k in (1,2,3):
        assert counts[f'distractors_{k}']/20000 == pytest.approx(1/3,abs=.025)
    C.validate(args('--joint-banks','a','b','c','--distractor-mode','uniform','--max-distractors','1'))
    with pytest.raises(SystemExit,match='requires --distractor-mode uniform'):
        C.validate(args('--joint-banks','a','b','c','--max-distractors','1'))


@pytest.mark.parametrize('checkpoint',[False,True])
@pytest.mark.parametrize('frozen',[False,True])
def test_gradients_all_active_or_relevant_only(checkpoint,frozen):
    model = base()
    banks = C.attach_banks(E,model,states())
    if checkpoint:
        E.checkpoint_experts(model,[0,1])
    row = dict(seq=[BOS_T,40,41,42,43],start=2,dropped=False)
    with C.activation(banks,(0,1),0,frozen):
        C.example_loss(E,model,TOK,banks,row,'cpu').backward()
    norms = [sum(float(m.M.grad.abs().sum()) for m in b.modules.values() if m.M.grad is not None) for b in banks]
    assert norms[0] > 0
    assert (norms[1] == 0) if frozen else (norms[1] > 0)
    assert norms[2] == 0
    assert all(m.enabled and not m.frozen for b in banks for m in b.modules.values())


def test_dropout_base_kl_and_explicit_unknown_target():
    model = base()
    banks = C.attach_banks(E,model,states())
    row = dict(seq=[BOS_T,40,41,42,43],start=2,dropped=True)
    x = torch.tensor([row['seq']])
    with C.activation(banks,()),torch.no_grad():
        teacher = E.logits_of(model,x)[:,1:-1]
    with C.activation(banks,(1,),0,False):
        student = E.logits_of(model,x)[:,1:-1]
        expected = E.C50.kl_to_base(student,teacher)/3
        actual = C.example_loss(E,model,TOK,banks,row,'cpu','base')
        assert torch.allclose(actual,expected,atol=1e-7)
        actual.backward()
    assert all(m.M.grad is None for m in banks[0].modules.values())
    assert any(m.M.grad is not None and m.M.grad.abs().sum() > 0 for m in banks[1].modules.values())
    with C.activation(banks,()),torch.no_grad():
        assert abs(float(C.example_loss(E,model,TOK,banks,row,'cpu','base'))) < 1e-6
        unknown = row['seq'][:2]+TOK.encode(' Unknown.').ids
        logits = E.logits_of(model,torch.tensor([unknown]))[:,1:-1]
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1,128),torch.tensor(unknown[2:]))
        assert torch.allclose(C.example_loss(E,model,TOK,banks,row,'cpu','unknown'),expected)


@pytest.mark.parametrize('fraction',[1.,.5])
def test_saved_state_roundtrip_and_revoke_exactness(fraction):
    model = base()
    banks = C.attach_banks(E,model,states(budget=64,fraction=fraction))
    rng = C.Schedule(4,3)
    with torch.no_grad():
        for i,b in enumerate(banks):
            b.tau = .0001*(i+1)
            for m in b.modules.values():
                m.bop_ema.normal_()
                m.score.normal_()
                # Preserve an allowed zero slot in the support as well as live codes.
                m.support[:2] = True
    saved = [C.export_bank(E,b,rng,'J',2) for b in banks]
    loaded_model = base()
    loaded = C.attach_banks(E,loaded_model,saved)
    for a,b in zip(banks,loaded):
        assert a.tau == b.tau and a.budget == b.budget
        for name in a.modules:
            for key in ('M','support','score','bop_ema','idx'):
                assert torch.equal(getattr(a.modules[name],key),getattr(b.modules[name],key))
    x = torch.tensor([[BOS_T,40,41,42]])
    for active in ((),(0,),(0,1,2)):
        with C.activation(banks,active),C.activation(loaded,active),torch.no_grad():
            assert torch.equal(E.logits_of(model,x),E.logits_of(loaded_model,x))
    report = C.revoke_check(E,model,banks,[x])
    assert report['all_revoked_logits_bit_exact']
    assert all(r['logits_bit_exact'] and r['weights_bit_exact'] for r in report['banks'])
    assert len(E.fsa_modules(model)) == sum(len(b.modules) for b in banks)


def test_overlap_checked_on_dormant_slots_before_attach():
    saved = states(budget=64)
    # Live slots can be empty; overlap must still fail on complete legal masks.
    for state in saved:
        for entry in state['entries'].values():
            entry['idx'],entry['val'] = entry['idx'][:0],entry['val'][:0]
    saved[1]['meta']['slot_part'] = 0
    model = base()
    with pytest.raises(ValueError,match='masks overlap'):
        C.attach_banks(E,model,saved)
    assert not E.patched_linears(model)


def test_batches_cover_other_people_conflicts_and_identical_exposures():
    facts,owners,scopes = C.world(E,states())
    a = args('--aug')
    a.fact_batch = 3
    s0,s1 = C.Schedule(7,3),C.Schedule(7,3)
    for _ in range(10):
        r0 = C.batch(E,TOK,facts,owners,scopes,s0,a,BOS_T)
        r1 = C.batch(E,TOK,facts,owners,scopes,s1,a,BOS_T)
        assert r0 == r1
        assert {row['relevant'] for row in r0} == {0,1,2}
        assert all(row['start'] > 1 and row['start'] < len(row['seq']) for row in r0)
    assert s0.report() == s1.report()


def test_absent_entity_queries_are_unique_and_never_written():
    facts = E.make_world(0,3,2,2)
    rows = C.absent_items(E,D,TOK,facts,BOS_T,1000)
    assert len({r['prompt'] for r in rows}) == len({r['query_name'] for r in rows}) == 1000
    assert not {r['query_name'] for r in rows} & {f.name for f in facts}


def test_legacy_checkpoint_restores_masters_support_ema_and_tau(tmp_path):
    saved = states(budget=64)
    banks = C.attach_banks(E,base(),saved)
    bank = banks[0]
    rows = {}
    for name,m in bank.modules.items():
        weights = m.state_dict()
        support = m.support.clone()
        support[:2] = True
        rows[name+'.parametrizations.weight.0'] = dict(weights=weights,
            attributes=dict(sparse=True,support=support,bop_ema=torch.ones_like(m.M),score=torch.ones_like(m.score)))
    path = tmp_path/'final.pt'
    torch.save(dict(step=4,phase='training',training=dict(modules=rows,values=dict(bop_tau=.007))),tmp_path/'resume.pt')
    C.restore_legacy_checkpoint(E,bank,path)
    assert bank.tau == .007 and bank.source_state == str(tmp_path/'resume.pt')
    assert all(torch.equal(m.bop_ema,torch.ones_like(m.M)) and m.support[1] for m in bank.modules.values())


def test_main_j_f_same_schedule_save_and_eval_hooks(cpu_harness,tmp_path,monkeypatch,capsys):
    from test_c49 import panel_file
    run,paths = cpu_harness
    panel,_ = panel_file(tmp_path)
    saved = states()
    bank_paths = []
    for i,state in enumerate(saved):
        path = tmp_path/f'bank{i}.pt'
        torch.save(state,path)
        bank_paths.append(str(path))
    # Keep all real training/teacher/revoke/NLL paths; fast deterministic
    # continuations exercise evaluation aggregation without 1000x64 decoding.
    monkeypatch.setattr(E,'greedy_generate',lambda model,prompts,*args,**kw:
        [TOK.encode('Slovenia.' if TOK.decode(p).startswith('QA item ') else ' Unknown.').ids for p in prompts])
    encode = type(TOK).encode
    monkeypatch.setattr(type(TOK),'encode',lambda self,text,**kw: encode(self,text.replace('—','-'),**kw))
    flags = ['--joint-banks',*bank_paths,'--steps','2','--relation-world','--p-iso','.3',
             '--presence-dropout','.2','--joint-schedule-seed','9','--refresh-every','1']
    flags += ['--in-topic-panel',str(panel),'--local-lambda','.01','--anchor',paths['anchor'],'--anchor-len','16']
    j,_ = run(flags,'joint_j')
    f,_ = run([*flags,'--joint-frozen-distractors'],'joint_f')
    assert j['arm'] == 'J' and f['arm'] == 'F' and j['schedule_seed'] == 9
    assert j['schedule'] == f['schedule']
    assert j['schedule']['sha256'] == f['schedule']['sha256']
    assert j['schedule']['distractor_mode'] == 'all'
    assert j['schedule']['counts'].get('distractors_1',0) == 0
    assert len(j['evaluation']['individual_gen']) == 3
    assert len(j['evaluation']['wrong_person_activation']) == 3
    assert j['evaluation']['absent_entities']['composed']['n'] == 1000
    assert 'by_fact_class' in j['evaluation']['composed_gen']
    assert 'by_fact_class' in j['evaluation']['composition']['forward_chain']
    assert j['evaluation']['in_topic']['knowledge_qa']['n'] == 200
    assert len(j['evaluation']['in_topic']['written_facts']['banks']) == 3
    assert j['evaluation']['off_topic']['eval']['base_nll'] > 0
    assert j['revoked']['pristine_logits_bit_exact']
    assert all(r['logits_bit_exact'] for r in j['revoked']['banks'])
    for result in (j,f):
        for path in result['patch_files']:
            state = torch.load(path,weights_only=False)
            assert state['format'] == 'e2e-fsa-patch-v1'
            assert state['meta']['c52']['arm'] == result['arm']
            assert state['meta']['budget'] == 20
            assert state['joint_state']['version'] == 1
    assert '"composed_gen"' in next(line for line in capsys.readouterr().out.splitlines() if line.startswith('RESULT'))


def test_no_joint_flags_leave_legacy_bytes_rng_and_results_identical(cpu_harness):
    run,_ = cpu_harness
    original,p0 = run(['--steps','2'],'legacy0')
    rng = torch.get_rng_state()
    other,p1 = run(['--steps','2'],'legacy1')
    assert torch.equal(rng,torch.get_rng_state())
    assert other['patched'] == original['patched']
    a,b = [torch.load(p,weights_only=False) for p in (p0,p1)]
    for name,entry in a['entries'].items():
        for key,value in entry.items():
            assert torch.equal(value,b['entries'][name][key]) if isinstance(value,torch.Tensor) else value == b['entries'][name][key]
    assert 'c52' not in a['meta'] and 'arm' not in original
