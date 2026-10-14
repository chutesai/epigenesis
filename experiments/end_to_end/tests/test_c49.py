"""C49 CPU census, overlap/resume, panel construction and harness integration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, Tiny, tiny, cpu_harness
import branch_eval as B
import build_in_topic_panel as P
import census as C
import in_topic as I
import c44_runtime as R
from token_streams import read_tokens, grams, OverlapGuard


def planted():
    model = Tiny(h=512, n=1, layers=1)
    with torch.no_grad():
        model.emb.weight.zero_()
        model.emb.weight[:, 1] = 1
        model.emb.weight[10, 0] = -1
        model.emb.weight[20, 0] = 1
        expert = model.moes[0].experts[0]
        expert.up_proj.weight.zero_()
        expert.up_proj.weight[:, 1] = 1
        expert.up_proj.weight[0].zero_()       # dormant everywhere
        expert.up_proj.weight[1].zero_()
        expert.up_proj.weight[1, 0] = 1       # quiet in three domains, active in one
        expert.down_proj.weight.fill_(.125)
        expert.down_proj.weight[:, 2] = 0     # fires but no output energy
    return model.eval()


def streams():
    return {d: np.full(96, token, dtype='<u4') for d,token in [('wiki',10),('web',10),('code',10),('books',20)]}


def test_planted_census_counts_relu_squared_domain_energy_and_ranking(tmp_path):
    m = planted()
    weights = {n:p.detach().clone() for n,p in m.named_parameters()}
    stats = C.Census(m, streams(), min_routed=1)
    try:
        cursors = C.run(m, streams(), 32, [96]*4, 2, 'cpu', stats)
        u = stats.units['L0_E0']
        assert u['routed'].dtype == u['fired'].dtype == u['relu2_sum'].dtype == torch.float64
        assert u['routed'].tolist() == [96]*4
        assert u['fired'][:,0].tolist() == [0]*4
        assert u['fired'][:,1].tolist() == [0,0,0,96]
        assert u['relu2_sum'][:,1].tolist() == [0,0,0,96]
        assert u['fired'][:,2].tolist() == [96]*4
        summary = stats.output(tmp_path/'census.npz', dict(cursors=cursors), plots=False)
        assert summary['layers']['0']['near_dormant_everywhere'] == 2
        assert summary['layers']['0']['domain_specific'] == 1
        arrays = np.load(tmp_path/'census.npz')
        assert arrays['L0_E0_energy'][:,2].tolist() == [0]*4
        assert arrays['L0_E0_energy'][3,1] == pytest.approx(.25)
        assert arrays['L0_E0_mean_relu2'][3,1] == 1
        assert set(arrays['layer0_rank'][:2]) == {0,2}
        assert json.loads((tmp_path/'census.summary.json').read_text()) == summary
        for name,p in m.named_parameters():
            assert torch.equal(p, weights[name]) and p.grad is None
    finally:
        stats.close()
    assert not m.moes[0].experts[0].up_proj._forward_hooks


def test_census_squares_pre_activation_even_on_relu_fixture(tmp_path):
    m = planted()
    with torch.no_grad():
        m.moes[0].experts[0].up_proj.weight[3,1] = 3
    stats = C.Census(m, ['web'])
    C.run(m, {'web': np.full(32,10)},32,[32],1,'cpu',stats)
    assert stats.units['L0_E0']['relu2_sum'][0,3] == 32*9
    stats.close()


def test_unrouted_experts_unknown_not_dormant(tmp_path):
    m = Tiny(n=2, h=32, layers=1)
    m.moes[0].router.weight.data.zero_()  # argmax selects expert 0
    stats = C.Census(m,['web'],min_routed=1)
    C.run(m,{'web':np.full(32,10)},32,[32],1,'cpu',stats)
    summary = stats.output(tmp_path/'out',{},plots=False)
    assert summary['layers']['0']['unknown_any_domain'] == 32
    assert np.isnan(np.load(tmp_path/'out.npz')['L0_E1_energy']).all()
    stats.close()


def test_census_resume_accumulators_exact(tmp_path):
    m = planted(); full = C.Census(m,streams())
    C.run(m,streams(),32,[96]*4,2,'cpu',full)
    m2 = planted(); first = C.Census(m2,streams())
    ckpt = tmp_path/'resume.pt'
    def save(cursors):
        R.atomic_save(ckpt,dict(cursors=dict(cursors),census=first.state()))
    C.run(m2,streams(),32,[96]*4,2,'cpu',first,checkpoint=save,ckpt_tokens=32,stop_after=128)
    saved = torch.load(ckpt,weights_only=False)
    first.close()
    m3 = planted(); restored = C.Census(m3,streams())
    restored.restore(saved['census'])
    C.run(m3,streams(),32,[96]*4,2,'cpu',restored,cursors=saved['cursors'])
    for key in full.units:
        for name in ('routed','fired','relu2_sum','down_norm'):
            assert torch.equal(full.units[key][name],restored.units[key][name])
    full.close(); restored.close()


def test_token_formats_and_budget(tmp_path):
    values = list(range(128))
    binary = tmp_path/'tokens.bin'; np.array(values,dtype='<u4').tofile(binary)
    js = tmp_path/'tokens.json'; js.write_text(json.dumps(values))
    assert isinstance(read_tokens(binary),np.memmap)
    assert read_tokens(binary).tolist() == read_tokens(js).tolist() == values
    binary.write_bytes(b'bad')
    with pytest.raises(ValueError,match='divisible'):
        read_tokens(binary)
    for bad in ([True], [-1], [2**32], ['1'], {}):
        js.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='uint32'):
            read_tokens(js)
    assert C.budgets(50_000_000,2048,4) == [12500992,12500992,12498944,12498944]
    with pytest.raises(ValueError,match='full context'):
        C.budgets(32,32,4)


def test_overlap_checks_every_source_gram_and_context_chunk_boundaries():
    reference = np.arange(128,dtype='<u4')
    guard = OverlapGuard({'eval':reference})
    source = np.concatenate([np.full(65521,999), reference, np.full(100,999)])
    with pytest.raises(ValueError,match='exceeds'):
        guard.check([source])
    assert guard.check([np.full(100,999)]) == {'eval':0.}
    assert [g for _,g in grams(reference,chunk=17)] == [g for _,g in grams(reference,chunk=1000)]
    # Exactly 1% is accepted, more than 1% refused (sampled-reference denominator).
    guard.refs = {'eval': [b'x']*1 + [b'y']*99}; guard.wanted = {b'x',b'y'}
    import token_streams as T
    old = T.grams
    try:
        T.grams = lambda *args,**kw: iter([(0,b'x')])
        assert guard.check([source])['eval'] == .01
        guard.refs['eval'][1] = b'x'
        with pytest.raises(ValueError):
            guard.check([source])
    finally:
        T.grams = old


def test_census_main_resume_identity_and_backend(cpu_harness, tmp_path, monkeypatch):
    export = tmp_path/'export'; export.mkdir(); (export/'weights').write_bytes(b'weights')
    paths = []
    for name,token in [('web',10),('code',20)]:
        path = tmp_path/f'{name}.bin'; np.full(96,token,dtype='<u4').tofile(path); paths.append(path)
    ref = tmp_path/'eval-protected.json'; ref.write_text(json.dumps([80]*100))
    anchor = tmp_path/'census-anchor.json'; anchor.write_text(json.dumps([70]*100))
    out = tmp_path/'census.npz'
    flags = ['--stream',f'web={paths[0]}','--stream',f'code={paths[1]}','--tokens','192',
        '--context-length','32','--batch-size','1','--export',str(export),'--anchor',str(anchor),
        '--eval-panels',str(ref),'--out',str(out),'--ckpt-tokens','32','--no-plots',
        '--min-routed','1','--factqa-people','1','--factqa-facts','1']
    original = R.atomic_save
    def interrupt(path,state):
        original(path,state)
        raise RuntimeError('interrupted')
    monkeypatch.setattr(R,'atomic_save',interrupt)
    with pytest.raises(RuntimeError,match='interrupted'):
        C.main(flags)
    monkeypatch.setattr(R,'atomic_save',original)
    resumed = C.main([*flags,'--resume','auto'])
    arrays = {k:v.copy() for k,v in np.load(out).items()}
    complete = C.main(flags)
    assert resumed == complete and complete['eda_backend'] == 'naive_fp32'
    for k,v in np.load(out).items():
        np.testing.assert_array_equal(v,arrays[k])
    swapped = [flag.replace(f'web={paths[0]}',f'web={paths[1]}').replace(f'code={paths[1]}',f'code={paths[0]}') for flag in flags]
    with pytest.raises(ValueError,match='configuration'):
        C.main([*swapped,'--resume','auto'])
    (export/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError,match='input contents'):
        C.main([*flags,'--resume','auto'])
    ref.write_text(json.dumps([10]*100))
    with pytest.raises(ValueError,match='overlap'):
        C.main(flags)


def fake_qa(monkeypatch, fail=None):
    original = E.greedy_generate
    def generate(model,prompts,dev,max_new,bs=16,*unused,**kw):
        output = []
        for p in prompts:
            text = TOK.decode(p)
            if text.startswith('QA item '):
                index = int(text.split()[2].rstrip(':'))
                answer = 'wrong' if fail and fail(index) else 'Slovenia'
                output.append(TOK.encode(answer+'.').ids)
            else:
                output.extend(original(model,[p],dev,max_new,1,*unused,**kw))
        return output
    monkeypatch.setattr(E,'greedy_generate',generate)


def panel_file(tmp_path, count=200):
    tokenizer = E.DATA / 'tokenizer.json'
    if not tokenizer.exists() and E.DATA == tmp_path:
        tokenizer.write_text('{}')
    rows = [dict(concept='Ljubljana',prompt=f'QA item {i}: In which country is Ljubljana? Answer:',
                 answers=['Slovenia'],base_correct=True) for i in range(count)]
    path = tmp_path/'in-topic-test.json'
    panel = dict(version=1,eda_backend='naive_fp32',
        meta=dict(export=str(E.EXPORT),export_step=E.EXPORT_STEP,tokenizer_sha256=E.file_sha(tokenizer) if tokenizer.exists() else None),
        selection=dict(base_accuracy=.8),knowledge_qa=rows,text_windows=[TOK.encode('Ljubljana '+c*54).ids for c in ('K','L')])
    path.write_text(json.dumps(panel))
    return path, panel


def test_candidate_gold_mentions_pool_concepts_and_no_fiction():
    rows = P.candidate_qa(TOK,BOS_T)
    assert len(rows) >= 200
    assert len({r['prompt'] for r in rows}) == len(rows)
    assert all(r['concept'] in I.concepts() and P.mentions(r['prompt'],r['concept']) for r in rows)
    assert not any('Tamsin' in r['prompt'] for r in rows)
    assert not P.mentions('cartography','art') and P.mentions('the CELLO.','cello')
    find = I.concept_finder()
    assert find('The CELLO and cartography.') == ['cello']
    assert find('a green Volvo Amazon and PERIWINKLE.') == ['a green Volvo Amazon','periwinkle']


def test_filter_gpu_pass_keeps_both_strata_records_accuracy(monkeypatch):
    fake_qa(monkeypatch,lambda i:i%3==0)
    rows = [dict(concept='Ljubljana',prompt=f'QA item {i}: In which country is Ljubljana?',answers=['Slovenia']) for i in range(300)]
    kept, selection = P.filter_qa(None,TOK,rows,'cpu',BOS_T)
    assert len(kept) == 300 and selection['base_accuracy'] == pytest.approx(2/3)
    assert sum(r['base_correct'] for r in kept) == 200 and selection['base_incorrect'] == 100
    assert all(r['base_answer'] == ('Slovenia.' if r['base_correct'] else 'wrong.') for r in kept)
    with pytest.raises(ValueError,match='scored QA'):
        P.filter_qa(None,TOK,rows[:100],'cpu',BOS_T)


def test_concept_fineweb_windows_exclude_training_overlap():
    length = 64
    text = lambda c: TOK.encode('Ljubljana '+c*54).ids
    tokens = np.array(text('a')+text('b')+text('c'),dtype='<u4')
    rows, concepts, overlap = P.select_windows(TOK,tokens,{'train':tokens[:64]},64,2)
    assert rows == [tokens[64:128].tolist(),tokens[128:192].tolist()]
    assert all('Ljubljana' in c for c in concepts) and overlap == {'train':dict(matched_windows=[],windows=2)}
    with pytest.raises(ValueError,match='disjoint concept'):
        P.select_windows(TOK,tokens,{'train':tokens},64,2)


@pytest.mark.parametrize('lost,passed',[(0,True),(2,True),(3,False),(4,False)])
def test_knowledge_gate_paired_loss(lost,passed):
    result = I.paired_flags([True]*200,[False]*lost+[True]*(200-lost))
    assert result['lost'] == lost and result['passed'] is passed
    assert result['accuracy_delta'] == pytest.approx(-lost/200)


def test_noise_gate_does_not_accept_ci_covering_zero():
    result = I.paired_flags([True]*50+[False]*50,[False]*52+[True]*48)
    assert result['accuracy_delta'] == -.02 and not result['passed']
    assert result['ci95'][0] <= 0 <= result['ci95'][1]


def test_panel_validation_count_backend_training_overlap(cpu_harness,tmp_path):
    path, panel = panel_file(tmp_path)
    args = E.build_parser().parse_args(['--out',str(tmp_path/'result')])
    assert I.load_panel(path,args,TOK,'cpu')['windows'].shape == (2,64)
    panel['knowledge_qa'].pop(); path.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match='200'):
        I.load_panel(path,args,TOK,'cpu')
    panel['knowledge_qa'].append(dict(panel['knowledge_qa'][0],prompt='Another question about Ljubljana: Answer:'))
    panel['eda_backend'] = 'fla_kda_2t'; path.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match='backend mismatch'):
        I.load_panel(path,args,TOK,'cpu')
    panel['eda_backend'] = 'naive_fp32'; path.write_text(json.dumps(panel))
    anchor = tmp_path/'panel-anchor.json'; anchor.write_text(json.dumps(panel['text_windows'][0])); args.anchor = str(anchor)
    with pytest.raises(ValueError,match='overlap'):
        I.load_panel(path,args,TOK,'cpu')


def test_evaluate_text_paired_se_written_fact_gate(monkeypatch,tmp_path):
    fake_qa(monkeypatch)
    _, raw = panel_file(tmp_path,count=200)
    panel = dict(raw=raw,qa=raw['knowledge_qa'],windows=torch.tensor(raw['text_windows']),path='/tmp/test',sha256='x',overlap={})
    monkeypatch.setattr(E,'nll_windows',lambda *args: [1.004,1.008])
    group = dict(bank=0,facts=100,baseline=[True]*100)
    monkeypatch.setattr(I,'fact_flags',lambda *args:[False]*2+[True]*98)
    args = SimpleNamespace(gen_bs=4,gen_stop_k=2,gen_max_new=32)
    result = I.evaluate(None,TOK,panel,dict(qa=[True]*200,nll=[1.,1.]),[group],'cpu',BOS_T,args)
    assert not result['knowledge_qa']['passed']  # legacy one-directional panel is report-only
    assert result['text_dnll']['dnll'] == pytest.approx(.006)
    assert result['text_dnll']['dnll_se'] == pytest.approx(.002)
    assert not result['text_dnll']['passed'] and not result['written_facts']['passed']
    assert not result['passed']


def test_real_e2e_in_topic_final_result_and_resident_retention(cpu_harness,tmp_path,monkeypatch,capsys):
    run,paths = cpu_harness
    fake_qa(monkeypatch)
    panel,_ = panel_file(tmp_path)
    _, resident = run(['--steps','2','--train-person','0','--sparse-state'],'prior')
    result,_ = run(['--steps','1','--train-person','1','--sparse-state','--resident-patch',str(resident),
                    '--in-topic-panel',str(panel)],'new')
    report = result['in_topic']
    assert report['knowledge_qa']['n'] == 200 and report['knowledge_qa']['base_accuracy'] == 1
    assert report['written_facts']['banks'][0]['facts'] == 2
    assert report['written_facts']['banks'][0]['n'] == 2*len(E.TEST_T)
    assert report['text_dnll']['n'] == 2 and result['revoked']['logits_bit_exact']
    lines = capsys.readouterr().out.splitlines()
    final = [json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('FINAL ')][-1]
    assert final['in_topic'] == report
    printed = [json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('RESULT ')][-1]
    assert printed['in_topic'] == report
    assert any(s.startswith('IN-TOPIC ') for s in lines)


def test_branch_resident_policy_real_banks(cpu_harness,tmp_path,monkeypatch):
    run,_ = cpu_harness
    fake_qa(monkeypatch)
    path,_ = panel_file(tmp_path)
    states = {}
    for person in range(3):
        _,patch = run(['--steps','2','--train-person',str(person),'--sparse-state'],f'p{person}')
        states[person] = torch.load(patch,weights_only=False)
    # Use the same external loader fixture as the actual branch executable.
    from ops.eval.bench_saves import build_model
    model,_,dev = build_model()
    E.attach(model,[0,1],'fsa_forced',0,sparse=True)
    branches = B.Branches(model,states,{p:1 for p in states},'sum_clip')
    args = B.build_parser().parse_args(['--in-topic-panel',str(path)])
    panel = I.load_panel(path,args,TOK,dev)
    result = I.resident_branch_eval(branches,TOK,panel,dev,BOS_T,args)
    assert len(result['written_facts']['banks']) == 3
    assert result['policy'].startswith('all banks resident')
    assert result['knowledge_qa']['n'] == 200
    assert model._c16_branch_key == (id(branches),())


def test_learning_branch_in_topic_result(cpu_harness,tmp_path,monkeypatch,capsys):
    run,paths = cpu_harness
    fake_qa(monkeypatch)
    path,_ = panel_file(tmp_path)
    _, focus = run(['--steps','1','--train-person','0','--sparse-state'],'focus')
    result = B.main(['--learning-eval','--patches',str(focus),str(focus),'--out',str(tmp_path/'learning.json'),
                    '--in-topic-panel',str(path),'--gen-max-new','1'])
    assert len(result['in_topic']['written_facts']['banks']) == 2
    assert result['in_topic']['knowledge_qa']['n'] == 200
    assert any(line.startswith('RESULT ') for line in capsys.readouterr().out.splitlines())


def test_panel_builder_main_cpu_loader_and_roundtrip(cpu_harness,tmp_path,monkeypatch):
    run, paths = cpu_harness
    _, template = run(['--steps','0'], 'calibration')
    tokenizer = tmp_path/'tokenizer.json'; tokenizer.write_text('{}')
    rows = [dict(concept='Ljubljana' if i < 250 else 'Vienna',
                 prompt=f'QA item {i}: In which country is '+('Ljubljana' if i < 250 else 'Vienna')+'? Answer:',
                 answers=['Slovenia'], pair='Ljubljana', category='city', stratum='value' if i < 250 else 'neighbour') for i in range(500)]
    monkeypatch.setattr(P,'matched_candidates',lambda *args: rows)
    fake_qa(monkeypatch,lambda i:i%5==0)
    tokens = [t for c in ('K','L','M') for t in TOK.encode('Ljubljana '+c*54).ids]
    source = tmp_path/'fineweb.bin'; np.array([11]*64+tokens,dtype='<u4').tofile(source)
    tail_sources = P.tail_sources
    monkeypatch.setattr(P, 'tail_sources', lambda f, paths, **kw: tail_sources(f, paths, tail_tokens=192, **kw))
    out = tmp_path/'in-topic-built.json'
    panel = P.main(['--fineweb',str(source),'--anchor',paths['anchor'],'--training-text',paths['rehearsal'],
                    '--eval-panels',paths['eval'],'--calibration-patch',str(template),
                    '--context-length','64','--text-windows','2','--out',str(out)])
    assert panel['selection']['kept'] == 500 and panel['selection']['base_accuracy'] == .8
    assert len(panel['text_windows']) == 2 and sum(r['base_correct'] for r in panel['knowledge_qa']) == 400
    assert 'random_forced_zero' in panel['calibration'] and not panel['gate_ready']
    args = E.build_parser().parse_args(['--out',str(tmp_path/'result'),'--anchor',paths['anchor']])
    loaded = I.load_panel(out,args,TOK,'cpu')
    assert loaded['raw'] == panel
    # Fixed generation limits/order reproduce flags in both correctness strata,
    # independently of the patched harness's fact-generation settings.
    assert I.base_scores(tiny(),TOK,loaded,'cpu',BOS_T)['qa'] == [r['base_correct'] for r in panel['knowledge_qa']]


def test_branch_main_in_topic_final_and_result(cpu_harness,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(Tiny,'layers',property(lambda self:self.moes),raising=False)
    encode = type(TOK).encode
    monkeypatch.setattr(type(TOK),'encode',lambda self,text,**kw:encode(self,text.replace('—','-'),**kw))
    run, paths = cpu_harness
    fake_qa(monkeypatch)
    panel,_ = panel_file(tmp_path)
    patches = []
    for person in range(3):
        _,patch = run(['--steps','2','--train-person',str(person),'--sparse-state'],f'branch{person}')
        patches.append(str(patch))
    out = tmp_path/'branch-result.json'
    B.main(['--patches',*patches,'--modes','sum','--anchor',paths['anchor'],'--eval-set',paths['eval'],
            '--fineweb',paths['extra'],'--code',paths['extra'],'--fast-moe','off',
            '--router-layer','0,1','--router-none','12','--router-steps','2','--qa-n','2','--g-n','4',
            '--gen-max-new','1','--in-topic-panel',str(panel),'--out',str(out)])
    result = json.loads(out.read_text())
    assert len(result['in_topic']['written_facts']['banks']) == 3
    assert result['revoked']['logits_bit_exact'] and result['revoked']['weights_bit_exact']
    logs = capsys.readouterr().out.splitlines()
    for prefix in ('FINAL ','RESULT '):
        last = [json.loads(s[len(prefix):]) for s in logs if s.startswith(prefix)][-1]
        assert last['in_topic'] == result['in_topic']


def test_runtime_base_correct_check_rejects_export_drift(cpu_harness,tmp_path,monkeypatch):
    fake_qa(monkeypatch,lambda i:i==0)
    path,_ = panel_file(tmp_path)
    args = E.build_parser().parse_args(['--out',str(tmp_path/'out')])
    panel = I.load_panel(path,args,TOK,'cpu')
    with pytest.raises(ValueError,match='no longer base-correct'):
        I.base_scores(tiny(),TOK,panel,'cpu',BOS_T)


def test_checkpoint_fingerprints_include_in_topic_panel(tmp_path):
    path = tmp_path/'panel.json'; path.write_text('original')
    args = SimpleNamespace(in_topic_panel=path)
    before = R.input_fingerprints(args)
    assert str(path.resolve()) in before
    path.write_text('replacement')
    assert R.input_fingerprints(args) != before


def test_panel_disjointness_checks_unsampled_training_grams():
    from token_streams import window_overlaps
    # Shared gram occurs at source offset 7 and candidate offset 5; a stride-16
    # source reference check misses it, while strict panel disjointness finds it.
    common = list(range(1000,1032))
    source = [900]*7+common+[901]*8
    window = [800]*5+common+[801]*9
    contaminated, report = window_overlaps([window], {'train': source})
    assert contaminated == {0} and report['train']['matched_windows'] == [0]
