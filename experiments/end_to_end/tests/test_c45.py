"""C45 backend provenance and isolation, with CPU models and fake FLA/host imports."""
import ast
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_e2e_fsa import E, cpu_harness
from test_branch_eval import state
import branch_eval as B
import c38_analysis as A
import c42_selector as C
import c44_runtime as R
import decisive_eval as D
import eda_backend as K
import two_hop as T


@pytest.fixture
def fake_fla(monkeypatch, tmp_path):
    for name in ('fla', 'fla.ops', 'fla.ops.kda'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setattr(sys.modules['fla.ops.kda'], 'chunk_kda', lambda: None, raising=False)
    deps = tmp_path / 'pydeps_fla'
    deps.mkdir()
    monkeypatch.setattr(K, 'FLA_DEPS', deps)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    return deps


@pytest.mark.parametrize('module,required', [
    ('e2e_fsa', ['--out', 'unused']), ('branch_eval', []), ('two_hop', ['--positive-control']),
    ('separability_probe', ['--out', 'unused']), ('exploration', ['--arm', 'vsa']),
    ('hebbian_write', []), ('masked_solve', []),
])
def test_flags_default_and_explicit(module, required):
    mod = importlib.import_module(module)
    parse = mod.build_parser().parse_args if hasattr(mod, 'build_parser') else mod.parse_args
    assert parse(required).eda_backend == K.DEFAULT
    assert parse(required + ['--eda-backend', 'fla_kda_2t']).eda_backend == 'fla_kda_2t'
    with pytest.raises(SystemExit):
        parse(required + ['--eda-backend', 'silent_fallback'])


def test_every_model_entrypoint_has_flag_and_uses_shared_builder():
    root = Path(E.__file__).parent
    count = 0
    for path in root.glob('*.py'):
        if path.name == 'eda_backend.py':
            continue
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ((isinstance(node.func, ast.Name) and node.func.id == 'build_model')
                      or (isinstance(node.func, ast.Attribute) and node.func.attr == 'build_model'))]
        if not calls:
            continue
        count += 1
        for call in calls:
            assert isinstance(call.func, ast.Attribute), path
            assert isinstance(call.func.value, ast.Attribute) and call.func.value.attr == 'eda' or (
                isinstance(call.func.value, ast.Name) and call.func.value.id == 'eda'), path
            assert isinstance(call.args[0], ast.Name) and call.args[0].id == 'build_model', path
            assert not any(kw.arg == 'eda_kernel_backend' for kw in call.keywords), path
        if path.name != 'c42_runtime.py':  # dispatched by branch_eval's parser
            assert '.add_argument(' in path.read_text() and '.announce(' in path.read_text(), path
    # C49 adds two loaders and C52 adds a dispatched joint runner. New diagnostic
    # entrypoints are allowed, and every one must satisfy the checks above.
    assert count >= 18


@pytest.mark.parametrize('backend', K.CHOICES)
def test_fake_builder_receives_backend_and_dependency_path(backend, fake_fla):
    seen = []
    def builder(export, **kwargs):
        seen.append((export, kwargs))
        if backend == 'fla_kda_2t':
            assert sys.path[0] == str(fake_fla)
        return SimpleNamespace(), None, 'cpu'
    model, _, _ = K.build_model(builder, SimpleNamespace(eda_backend=backend), 'export', step=42)
    assert seen == [('export', dict(step=42, eda_kernel_backend=backend))]
    assert model._fsa_eda_backend == backend


def test_fla_import_failure_prevents_builder(monkeypatch, tmp_path):
    monkeypatch.setattr(K, 'FLA_DEPS', tmp_path / 'missing')
    monkeypatch.setitem(sys.modules, 'fla.ops.kda', None)
    calls = []
    with pytest.raises(RuntimeError, match='refusing fallback'):
        K.build_model(lambda *a, **kw: calls.append(kw), SimpleNamespace(eda_backend='fla_kda_2t'))
    assert calls == []


@pytest.mark.parametrize('backend', K.CHOICES)
def test_real_harness_persists_backend(cpu_harness, fake_fla, monkeypatch, capsys, backend):
    run, _ = cpu_harness
    host = sys.modules['ops.eval.bench_saves']
    builder = host.build_model
    seen = []
    def record(*args, **kwargs):
        seen.append(kwargs['eda_kernel_backend'])
        return builder(*args, **kwargs)
    monkeypatch.setattr(host, 'build_model', record)
    (fake_fla.parent / 'tokenizer.json').write_text('{}')
    patch_cache = fake_fla.parent / 'cache'
    result, patch = run(['--steps', '1', '--eda-backend', backend, '--base-cache', str(patch_cache)], backend)
    assert seen == [backend]
    assert result['eda_backend'] == backend
    assert torch.load(patch, weights_only=False)['meta']['eda_backend'] == backend
    assert json.loads(Path(result['manifest']).read_text())['eda_backend'] == backend
    manifest = json.loads((patch_cache / (result['base_cache_key'] + '.json')).read_text())
    assert manifest['eda_backend'] == backend
    stdout = capsys.readouterr().out
    assert f'EDA-BACKEND {backend}' in stdout
    for line in stdout.splitlines():
        if line.startswith(('RESULT ', 'FINAL ')):
            assert json.loads(line.split(' ', 1)[1])['eda_backend'] == backend


def test_both_base_cache_keys_and_resume_hash_separate_backends(tmp_path, monkeypatch):
    for name in ('heldout-tokens.json', 'tokenizer.json', 'eval.json'):
        (tmp_path / name).write_text('[1, 2, 3]')
    monkeypatch.setattr(E, 'DATA', tmp_path)
    a = E.build_parser().parse_args(['--out', 'unused', '--eval-set', str(tmp_path / 'eval.json')])
    b = B.build_parser().parse_args(['--export', str(tmp_path)])
    b.eval_set = b.fineweb = b.code = tmp_path / 'eval.json'
    key = lambda: E.base_cache_key(a, E.make_world(), None, None, data=tmp_path, export=tmp_path)
    branch_key = lambda: B.branch_base_cache_key(b, {})
    first, branch_first, resume_first = key(), branch_key(), R.config_hash(R.configuration(a))
    a.eda_backend = b.eda_backend = 'fla_kda_2t'
    assert key() != first and branch_key() != branch_first
    assert R.config_hash(R.configuration(a)) != resume_first
    del a.eda_backend
    del b.eda_backend
    assert key() == first and branch_key() == branch_first
    assert R.config_hash(R.configuration(a)) == resume_first


def test_cache_manifest_refuses_wrong_backend_and_accepts_legacy(tmp_path):
    E.cached_base(tmp_path, 'key', lambda: {'value': 1}, 'cpu')
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        E.cached_base(tmp_path, 'key', lambda: pytest.fail('cache should be checked first'), 'cpu', 'fla_kda_2t')
    path = tmp_path / 'key.json'
    record = json.loads(path.read_text())
    del record['eda_backend']
    path.write_text(json.dumps(record))
    assert E.cached_base(tmp_path, 'key', lambda: pytest.fail('legacy should hit'), 'cpu') == {'value': 1}


def test_banks_legacy_default_and_mismatch():
    banks = [state(p, [p], [1]) for p in range(3)]
    assert len(B.validate_patches(banks, eda_backend=K.DEFAULT)) == 3
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        B.validate_patches(banks, eda_backend='fla_kda_2t')
    banks[0]['meta']['eda_backend'] = 'fla_kda_2t'
    with pytest.raises(ValueError, match='mixed EDA backends'):
        B.validate_patches(banks)
    with pytest.raises(ValueError, match='mixed EDA backends'):
        B.merge_patches(banks)
    for bank in banks:
        bank['meta']['eda_backend'] = 'fla_kda_2t'
    assert len(B.validate_patches(banks, eda_backend='fla_kda_2t')) == 3


@pytest.mark.parametrize('entry', ['branch', 'learning', 'c42', 'two_hop', 'decisive'])
def test_evaluation_refuses_mismatch_before_building(entry, tmp_path, monkeypatch):
    banks = [state(p, [p], [1]) for p in range(3)]
    paths = []
    for p, bank in enumerate(banks):
        bank['meta']['eda_backend'] = 'fla_kda_2t'
        path = tmp_path / f'bank{p}.pt'
        torch.save(bank, path)
        paths.append(str(path))
    flags = ['--patches', *paths, '--out', str(tmp_path / 'result.json')]
    if entry == 'branch':
        call = lambda: B.main(flags + ['--anchor', 'unused', '--eval-set', 'unused', '--fineweb', 'unused', '--code', 'unused'])
    elif entry == 'learning':
        call = lambda: B.main(flags + ['--learning-eval'])
    elif entry == 'c42':
        call = lambda: B.main(flags + ['--c42-mode', 'dump', '--c42-cache', str(tmp_path / 'cache.pt')])
    elif entry == 'two_hop':
        call = lambda: T.main(flags + ['--anchor', 'unused'])
    else:
        branches = SimpleNamespace(states=dict(enumerate(banks)))
        call = lambda: D.suite(branches, None, [], 0, 'cpu', SimpleNamespace(qa_n=0))
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        call()
    assert not (tmp_path / 'result.json').exists()


@pytest.mark.parametrize('analyze', [A.sequential, lambda *rs: A.factorial(list(rs)), lambda *rs: A.dose_response(list(rs))])
def test_c38_rejects_mixed_results_before_metrics(analyze):
    with pytest.raises(ValueError, match='mixed EDA backends'):
        analyze({}, {'eda_backend': 'fla_kda_2t'}, {})


def test_c42_rejects_mixed_cache_sets_and_result_sets(tmp_path, monkeypatch):
    paths = []
    for i, record in enumerate(({}, {'metadata': {'eda_backend': 'fla_kda_2t'}})):
        path = tmp_path / f'cache{i}.pt'
        torch.save(record, path)
        paths.append(str(path))
    monkeypatch.setattr(C, 'analyze_set', lambda *a: pytest.fail('must reject before analysis'))
    with pytest.raises(ValueError, match='mixed EDA backends'):
        C.main(['--caches', *paths, '--out', str(tmp_path / 'result.json')])
    with pytest.raises(ValueError, match='mixed EDA backends'):
        C.kill_rules([{}, {'eda_backend': 'fla_kda_2t'}])
    assert not (tmp_path / 'result.json').exists()


def test_legacy_and_nested_provenance():
    assert K.require_same([{}, {'eda_backend': K.DEFAULT}]) == K.DEFAULT
    assert K.recorded({'metadata': {'eda_backend': 'fla_kda_2t'}}) == 'fla_kda_2t'
    with pytest.raises(ValueError, match='conflicting EDA backends'):
        K.recorded({'eda_backend': K.DEFAULT, 'world': {'eda_backend': 'fla_kda_2t'}})


def test_c42_model_loader_propagates_backend(cpu_harness, fake_fla, monkeypatch):
    import c42_runtime as G
    args = B.build_parser().parse_args(['--eda-backend', 'fla_kda_2t', '--fast-moe', 'off'])
    host = sys.modules['ops.eval.bench_saves']
    builder, seen = host.build_model, []
    def record(*args, **kwargs):
        seen.append(kwargs['eda_kernel_backend'])
        return builder(*args, **kwargs)
    monkeypatch.setattr(host, 'build_model', record)
    model, _, _ = G.load_model(B, args, {'eda_backend': 'fla_kda_2t'})
    assert seen == ['fla_kda_2t'] and model._fsa_eda_backend == 'fla_kda_2t'


def test_c42_gen_refuses_cache_backend_even_with_matching_banks(tmp_path):
    paths = []
    for p in range(3):
        path = tmp_path / f'bank{p}.pt'
        torch.save(state(p, [p], [1]), path)
        paths.append(str(path))
    cache = tmp_path / 'cache.pt'
    torch.save({'eda_backend': 'fla_kda_2t'}, cache)
    with pytest.raises(ValueError, match='EDA backend mismatch.*C42 GEN cache'):
        B.main(['--c42-mode', 'gen', '--patches', *paths, '--c42-cache', str(cache), '--out', str(tmp_path / 'result.json')])


def test_c42_cache_manifest_rejects_mismatched_bank_backend():
    from test_c42 import fixture
    cache = fixture()
    cache['patches'] = [{'eda_backend': 'fla_kda_2t'}]
    with pytest.raises(ValueError, match='EDA backend mismatch.*C42 bank manifest'):
        C.validate_cache(cache)


def test_c44_resume_refuses_backend_change(tmp_path):
    flags = ['--out', str(tmp_path / 'result.json'), '--patch-dir', str(tmp_path / 'patches'), '--ckpt-every', '1']
    args = E.build_parser().parse_args(flags)
    R.Checkpoints(args).save(1)
    with pytest.raises(SystemExit, match='resume config mismatch.*eda_backend'):
        R.Checkpoints(E.build_parser().parse_args(flags + ['--eda-backend', 'fla_kda_2t', '--resume', 'auto']))


def test_load_patch_rejects_mismatch_before_touching_weights():
    model = SimpleNamespace(_fsa_eda_backend='fla_kda_2t')
    with pytest.raises(ValueError, match='EDA backend mismatch'):
        E.load_patch(model, {'meta': {}, 'entries': {}})
