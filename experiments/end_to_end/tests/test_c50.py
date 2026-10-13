"""C50 CPU objective, token dosing, resident exclusion and metric integration."""
import math
import json

import pytest
import torch

from test_e2e_fsa import E, TOK, BOS_T, tiny, cpu_harness
import branch_eval as B
import decisive_eval as D
import c50 as C


def args(*flags):
    return E.build_parser().parse_args(['--method', 'fsa_forced', '--out', 'unused', *flags])


@pytest.mark.parametrize('value,label', [
    ('41', 'exact'), ('41.5 kg', 'exact'), ('A-7', 'exact'), ('ABCD', 'exact'),
    ('api_key', 'exact'), ('ab/cd', 'exact'), ('urn:foo', 'exact'), ('camelCase', 'exact'),
    ('blue-green', 'semantic'), ('Beth', 'semantic'), ('New York', 'semantic'), ('viola', 'semantic')])
def test_class_rule(value, label):
    assert C.fact_class(value) == label


@pytest.mark.parametrize('flags', [
    ['--offtopic-kl', '.03', '--rehearsal-frac', '.2'],
    ['--intopic-neg-frac', '.05', '--rehearsal-frac', '.2'],
    ['--offtopic-kl', '-.1'], ['--offtopic-kl', 'nan'], ['--intopic-neg-frac', 'inf'],
    ['--offtopic-kl', '.6', '--intopic-neg-frac', '.4'],
    ['--offtopic-kl-weight', '-1'], ['--offtopic-kl-weight', 'nan'],
    ['--offtopic-kl', '.05', '--optim', 'mls'],
    ['--offtopic-kl', '.05', '--activation-noise', '.1'],
    ['--offtopic-kl', '.05', '--anchor-len', '1']])
def test_invalid_flags(flags):
    with pytest.raises(SystemExit):
        C.validate(args(*flags))


def test_disabled_patch_kl_zero_and_teacher_detached():
    m = tiny()
    E.attach(m, [0, 1], 'fsa_full', 0)
    x = torch.tensor([[BOS_T, 40, 41, 42]])
    E.set_enabled(False)
    try:
        student = E.logits_of(m, x)[:, :-1]
        with C.pristine_teacher(E, m):
            teacher = E.logits_of(m, x)[:, :-1]
        assert C.kl_to_base(student, teacher).item() == pytest.approx(0., abs=1e-6)
        assert not E.FSAParam.enabled and not teacher.requires_grad
    finally:
        E.set_enabled(True)
    student = torch.randn(2, 128, requires_grad=True)
    teacher = torch.randn(2, 128, requires_grad=True)
    C.kl_to_base(student, teacher).backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None


def test_streamed_kl_gradients_and_token_dose():
    m = tiny()
    E.attach(m, [0, 1], 'fsa_full', 0)
    for mod in E.fsa_modules(m):
        with torch.no_grad():
            mod.M[mod.mask] = 1.
    a = args('--offtopic-kl', '.03', '--intopic-neg-frac', '.02')
    facts = E.make_world(0, 2, 2, 2)
    source = torch.tensor([[BOS_T]+[40]*15, [BOS_T]+[41]*15])
    kl = C.PreservationKL(E, m, TOK, facts, [0, 1], [], source, a, BOS_T, 'cpu')
    losses = [kl.backward(100) for _ in range(10)]
    assert sum(losses) > 0
    assert any(mod.M.grad is not None and mod.M.grad.abs().sum() > 0 for mod in E.fsa_modules(m))
    report = kl.report()
    assert report['assistant_tokens'] == 1000
    assert report['offtopic_tokens'] == math.floor(1000*.03/.95)
    assert report['intopic_tokens'] == math.floor(1000*.02/.95)
    assert report['realized_offtopic_frac'] == pytest.approx(.03, abs=.001)
    assert report['realized_intopic_frac'] == pytest.approx(.02, abs=.001)
    assert report['teacher_forwards'] > report['student_forwards'] > 0
    assert report['seconds'] > 0


def test_resident_facts_excluded_and_teacher_switches_restored():
    facts = E.make_world(0, 2, 2, 2)
    state = dict(meta=dict(seed=0, people=2, facts_per_person=2, shared_attrs=2, values='disjoint',
                           train_person=1))
    protected = C.written_pairs(E, [state])
    negatives = C.hard_negatives(E, facts, [0, 1], protected)
    assert negatives and {r['kind'] for r in negatives} == {'sibling_entity', 'absent_bank'}
    held = protected | {(f.name, f.attr) for f in facts[:2]}
    assert not any((n['name'], n['attr']) in held for n in negatives)
    m = tiny()
    E.attach(m, [0, 1], 'fsa_full', 0)
    lin = m.moes[0].experts[0].up_proj
    W = lin.parametrizations.weight.original
    resident = E.ResidentParam(W, dict(kind='lora', A=torch.ones(1, W.shape[1]),
                                      B=torch.ones(W.shape[0], 1)))
    teacher = E.TeacherLoRAParam(W, torch.ones(1, W.shape[1]), torch.ones(W.shape[0], 1))
    teacher.enabled = True
    E.P.register_parametrization(lin, 'weight', resident)
    E.P.register_parametrization(lin, 'weight', teacher)
    with pytest.raises(RuntimeError):
        with C.pristine_teacher(E, m):
            assert not resident.enabled and not teacher.enabled and not E.FSAParam.enabled
            raise RuntimeError('test restoration')
    assert resident.enabled and teacher.enabled and E.FSAParam.enabled
    a = args('--offtopic-kl', '.05')
    source = torch.tensor([[BOS_T]+TOK.encode(facts[2].name).ids, [BOS_T]+[40]*len(facts[2].name)])
    kl = C.PreservationKL(E, m, TOK, facts, [0, 1], [state], source, a, BOS_T, 'cpu')
    assert kl.source_ids.tolist() == [1]
    with pytest.raises(SystemExit, match='no windows'):
        C.PreservationKL(E, m, TOK, facts, [0, 1], [state], source[:1], a, BOS_T, 'cpu')


def test_hard_negatives_never_target_trained_entity_holdouts():
    facts = E.make_world(0, 3, 5, 5)
    train_ids = [0, 1, 5, 6]
    protected = {(facts[10].name, facts[10].attr)}
    rows = C.hard_negatives(E, facts, train_ids, protected)
    trained = {facts[i].name for i in train_ids}
    holdouts = {(f.name,f.attr) for i,f in enumerate(facts) if f.name in trained and i not in train_ids}
    assert not any((r['name'],r['attr']) in holdouts | protected for r in rows)
    siblings = [r for r in rows if r['kind'] == 'sibling_entity']
    assert all(r['name'] not in trained for r in siblings)
    assert {f.name+'a' for f in facts} <= {r['name'] for r in siblings}
    assert any(r['name'] == facts[10].name for r in siblings)
    written = {(facts[i].name,facts[i].attr) for i in train_ids} | protected
    assert all((r['name'],r['attr']) not in written for r in rows if r['kind'] == 'absent_bank')


@pytest.mark.parametrize('factorial', [False, True])
def test_kl_does_not_advance_local_penalty_sampler(cpu_harness, monkeypatch, factorial):
    import inspect
    run, paths = cpu_harness
    randint, draws = torch.randint, []
    def record(*args, **kwargs):
        value = randint(*args, **kwargs)
        if inspect.currentframe().f_back.f_code.co_name == 'anchor_indices':
            draws.append(value.clone())
        return value
    monkeypatch.setattr(torch, 'randint', record)
    flags = ['--seed','1','--steps','4','--local-lambda','10', '--answer-only',
             '--anchor',paths['anchor'],'--anchor-len','16','--anchor-bs','2']
    if factorial:
        flags += ['--factorial-cell','sparse_tern','--optim','adam','--budget','0','--bop-tau-abs','0']
    run([*flags,'--offtopic-kl','0'], 'local_control')
    control, draws[:] = list(draws), []
    result, _ = run([*flags,'--offtopic-kl','.03'], 'local_kl')
    assert result['c50']['offtopic_tokens'] > 0
    assert len(control) == len(draws) > 0
    assert all(torch.equal(a,b) for a,b in zip(control,draws))


def test_metric_splits_reuse_generations_and_cover_c34(monkeypatch):
    facts = [E.Fact(0, 'Alice', 'her', 'she', 'code', 'AX_7'),
             E.Fact(0, 'Alice', 'her', 'she', 'instrument', 'viola')]
    items = E.build_items(TOK, facts, E.TEST_T[:1], BOS_T)
    gens = [TOK.encode('AX_7.').ids, TOK.encode('wrong.').ids]
    calls = []
    monkeypatch.setattr(E, 'greedy_generate', lambda *a, **kw: calls.append(1) or gens)
    legacy = E.gen_eval(None, TOK, items, 'cpu')
    metric = E.gen_eval(None, TOK, items, 'cpu', fact_class_metrics=True)
    assert metric['by_fact_class']['exact']['recall'] == 1
    assert metric['by_fact_class']['semantic']['recall'] == 0
    assert {k: v for k, v in metric.items() if k != 'by_fact_class'} == legacy
    assert len(calls) == 2
    qa = E.qa_probe_eval(None, TOK, {'unseen': items}, 'cpu', fact_class_metrics=True)
    assert qa['by_fact_class']['unseen']['exact']['unseen'] == 1
    C.annotate(items)
    branch = B.generation_summary(TOK, items, gens, {}, 2)
    assert branch['by_fact_class'] == metric['by_fact_class']
    group = D.task_items(TOK, facts, BOS_T)['reverse_lookup']
    C.annotate(group, facts)
    assert {tuple(it['fis']): it['fact_class'] for it in group} == {(0,): 'exact', (1,): 'semantic'}
    panel = D.summary(TOK, group, [TOK.encode('Alice.').ids]*len(group), facts=facts)
    assert panel['by_fact_class']['exact']['correct'] == 1
    assert panel['by_fact_class']['semantic']['correct'] == 1


def test_main_kl_reporting_rehearsal_source_and_reload(cpu_harness, capsys, monkeypatch):
    run, paths = cpu_harness
    encode = type(TOK).encode
    monkeypatch.setattr(type(TOK), 'encode', lambda self, text, **kw: encode(self, text.replace('—', '-'), **kw))
    flags = ['--steps', '2', '--offtopic-kl', '.03', '--offtopic-kl-weight', '1.0',
             '--intopic-neg-frac', '.02', '--rehearsal-set', paths['rehearsal'], '--anchor-len', '16',
             '--qa-probe', '--qa-n', '2', '--mcq-probe', '--answer-only']
    result, patch = run(flags, 'c50')
    assert result['offtopic_kl'] == .03 and result['intopic_neg_frac'] == .02
    assert result['c50']['teacher_forwards'] and result['c50']['student_forwards']
    assert result['c50']['seconds'] > 0 and result['c50']['offtopic_source'] == paths['rehearsal']
    assert 'qa_probe' in result['by_fact_class']
    assert 'mcq_probe' in result['by_fact_class']
    assert len(result['fact_classes']) == 6
    saved = torch.load(patch, weights_only=False)
    assert saved['meta']['c50']['offtopic_tokens'] == result['c50']['offtopic_tokens']
    assert saved['meta']['fact_classes'] == result['fact_classes']
    assert result['revoked']['logits_bit_exact']
    output = capsys.readouterr().out
    assert '"by_fact_class"' in next(line for line in output.splitlines() if line.startswith('FINAL '))
    loaded, _ = run(['--steps', '0', '--load-patch', str(patch), '--fact-class-metrics',
                     '--qa-probe', '--qa-n', '2', '--mcq-probe'], 'c50_loaded')
    assert loaded['by_fact_class'] == result['by_fact_class']


def test_main_off_flags_and_metric_only_preserve_patch_rng(cpu_harness):
    run, _ = cpu_harness
    control, p0 = run(['--steps', '2'], 'c50_control')
    rng0 = torch.get_rng_state()
    off, p1 = run(['--steps', '2', '--offtopic-kl', '0', '--intopic-neg-frac', '0'], 'c50_off')
    assert torch.equal(rng0, torch.get_rng_state())
    assert 'c50' not in off and 'by_fact_class' not in off
    assert off['patched'] == control['patched']
    measured, p2 = run(['--steps', '2', '--fact-class-metrics'], 'c50_metrics')
    assert torch.equal(rng0, torch.get_rng_state())
    states = [torch.load(p, weights_only=False) for p in (p0, p1, p2)]
    for name, entry in states[0]['entries'].items():
        for key, value in entry.items():
            for state in states[1:]:
                other = state['entries'][name][key]
                assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    assert measured['patched']['test_gen']['recall'] == control['patched']['test_gen']['recall']


def test_kl_process_resume_preserves_rounding_patch_and_rng(tmp_path):
    from test_c44 import process, equal_state
    flags = ['--steps', '4', '--ckpt-every', '1', '--sparse-state', '--anchor-case',
             '--offtopic-kl', '.03', '--intopic-neg-frac', '.02', '--local-lambda', '10']
    process(tmp_path, flags)
    path = tmp_path/'run_patches'/'final.pt'
    expected = torch.load(path, weights_only=False)
    expected_checkpoint = torch.load(tmp_path/'run_patches'/'resume.pt', weights_only=False)
    expected_result = json.loads((tmp_path/'run_result.json').read_text())
    process(tmp_path, flags, stop=2)
    output = process(tmp_path, [*flags, '--resume', 'auto'])
    assert 'RESUMED ' in output
    actual = torch.load(path, weights_only=False)
    checkpoint = torch.load(tmp_path/'run_patches'/'resume.pt', weights_only=False)
    equal_state(expected['entries'], actual['entries'])
    equal_state(expected_checkpoint['training']['rng'], checkpoint['training']['rng'])
    equal_state(expected_checkpoint['training']['objects']['preservation_rng'],
                checkpoint['training']['objects']['preservation_rng'])
    old, new = expected['meta']['c50'], actual['meta']['c50']
    assert {k:v for k,v in old.items() if k != 'seconds'} == {k:v for k,v in new.items() if k != 'seconds'}
    result = json.loads((tmp_path/'run_result.json').read_text())
    assert result['by_fact_class'] == expected_result['by_fact_class']


def test_branch_c34_class_panels_cpu(cpu_harness, monkeypatch, tmp_path):
    from test_c34 import test_branch_suite_pk_and_all_partition_conditions_cpu
    original = B.main
    def main(argv):
        original([*argv, '--fact-class-metrics'])
        if 'C34' in argv[1]:
            result = json.loads((tmp_path/'branch.json').read_text())
            panel = result['decisive_eval']['resident']
            assert 'by_fact_class' in panel
            assert 'by_fact_class' in panel['ceilings']['forward_chain']['as_trained']
            assert 'by_fact_class' in panel['negatives']['types']['near_miss']['patched']
    monkeypatch.setattr(B, 'main', main)
    test_branch_suite_pk_and_all_partition_conditions_cpu(cpu_harness, monkeypatch, tmp_path)
    result = json.loads((tmp_path/'branch.json').read_text())
    assert len(result['fact_classes']) == 6
    assert 'by_fact_class' in result
    for selector in result['arms']['PK']['selectors'].values():
        assert 'by_fact_class' in selector['test_gen']
