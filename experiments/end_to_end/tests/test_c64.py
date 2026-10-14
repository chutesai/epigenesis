"""C64 uses the real imported train loop on a tiny CPU MoE."""
import ast
import builtins
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from test_e2e_fsa import E, cpu_harness  # noqa: F401
import profile_patch_train as P


LOSS_TESTS = (
    'a.rehearsal_frac > 0',
    '(local is not None and lam > 0) or gate is not None',
    'anchor is not None and a.drift_lambda > 0',
    'preservation_kl is not None',
    'route_kl is not None',
)


def test_real_main_loss_branches_are_inside_forward(monkeypatch, tmp_path):
    compiled = []
    def capture_compile(source, *args, **kwargs):
        if isinstance(source, ast.Module):
            compiled.append(source)
        return builtins.compile(source, *args, **kwargs)
    monkeypatch.setattr(P, 'compile', capture_compile, raising=False)
    P.adapted_main(E, P.Recorder(tmp_path / 'scopes', cuda=False))
    assert len(compiled) == 1
    tree = compiled[0]
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'step')
    parents = {child: parent for parent in ast.walk(loop) for child in ast.iter_child_nodes(parent)}
    expected = {ast.dump(ast.parse(s, mode='eval').body): s for s in LOSS_TESTS}
    found = dict.fromkeys(expected, 0)
    for node in ast.walk(loop):
        if isinstance(node, ast.If) and ast.dump(node.test) in expected:
            key = ast.dump(node.test)
            found[key] += 1
            parent = parents[node]
            assert isinstance(parent, ast.With), expected[key]
            assert ast.unparse(parent.items[0].context_expr) == "_c64.phase('forward')", expected[key]
            assert parent.body == [node]
    # Route KL occurs both within the local pass and as a separate loss branch.
    assert list(found.values()) == [1, 1, 1, 1, 2]


@pytest.mark.parametrize('missing', LOSS_TESTS)
def test_missing_real_main_loss_scope_raises(monkeypatch, tmp_path, missing):
    tree = ast.parse(inspect.getsource(E.main))
    key = ast.dump(ast.parse(missing, mode='eval').body)
    replaced = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.dump(node.test) == key:
            node.test = ast.Constant(value=False)
            replaced += 1
    assert replaced > 0
    source = ast.unparse(tree)
    original_getsource = inspect.getsource
    monkeypatch.setattr(P.inspect, 'getsource',
                        lambda obj: source if obj is E.main else original_getsource(obj))
    with pytest.raises(RuntimeError, match='missing expected loss scopes') as exc:
        P.adapted_main(E, P.Recorder(tmp_path / 'missing', cuda=False))
    assert missing in str(exc.value)


def test_tiny_actual_loop_matches_uninstrumented_training(cpu_harness, monkeypatch, tmp_path):
    run, paths = cpu_harness
    models = []
    loader = sys.modules['ops.eval.bench_saves']
    original = loader.build_model
    def build(*a, **kw):
        result = original(*a, **kw)
        models.append(result[0])
        return result
    monkeypatch.setattr(loader, 'build_model', build)
    flags = ['--steps', '3', '--eval-every', '0', '--support-score', 'netcost', '--cost-rel', '1',
             '--rehearsal-frac', '.2', '--local-lambda', '10', '--no-churn', '--sparse-state',
             '--anchor', paths['anchor'], '--anchor-bs', '2', '--anchor-len', '8', '--aug', '--answer-only']
    run(flags, 'plain')
    expected = [(m.M.detach().clone(), m.bop_ema.clone(), m.support.clone()) for m in E.fsa_modules(models[-1])]
    argv = sys.argv[1:]
    output = tmp_path / 'profile'
    original_backward = torch.autograd.backward
    prof = P.run(E, argv, output, updates=3, cuda=False)
    assert torch.autograd.backward is original_backward
    assert len(prof.state) == len(expected)
    for actual, want in zip(prof.state, expected):
        assert all(torch.equal(a, b) for a, b in zip(actual, want))
    assert [r['kind'] for r in prof.rows] == ['update'] * 3 + ['eval']
    for row in prof.rows[:3]:
        assert row['backward_wall_s'] > 0 and row['forward_wall_s'] > 0
        assert row['data_tokenise_wall_s'] > 0
        assert row['optimizer_wall_s'] > 0 and row['bop_bookkeeping_wall_s'] > 0
        assert row['eval_wall_s'] == 0 and row['cuda_kernel_launches'] == 0
        assert sum(row[p + '_wall_s'] for p in P.PHASES) <= row['wall_s'] + .001
    result = json.loads(output.with_suffix('.json').read_text())
    assert result['requested_updates'] == 3 and result['cprofile_top30'] and result['top_cpu_ops']
    assert result['metadata']['recipe'] == E.training_recipe(E.build_parser().parse_args(argv))
    assert result['eval_result']['acquisition']
    assert 'Top 30 cProfile' in output.with_suffix('.md').read_text()
    assert output.with_suffix('.pstats').exists()
    assert not result['top_cuda_kernels']
    # No base evaluation or patch output is invoked by the adapter.
    monkeypatch.setattr(E, 'evaluate_base', lambda *a, **k: pytest.fail('base eval'))
    monkeypatch.setattr(E, 'save_patch', lambda *a, **k: pytest.fail('patch write'))
    P.run(E, argv, tmp_path / 'no_eval', updates=1, cuda=False)


def test_trace_launch_attribution_includes_backward_and_excludes_memcpy():
    def ev(name, ts, dur, cat='cpu_op', args=None, tid=1):
        return dict(name=name, ts=ts, dur=dur, cat=cat, args=args or {}, tid=tid, ph='X')
    events = [ev('C64/naive_eda', 10, 100, 'user_annotation'),
              ev('aten::mul', 20, 20, args={'Sequence number': 7}),
              ev('cudaLaunchKernel', 22, 2, 'cuda_runtime', {'correlation': 10}),
              ev('autograd::engine::evaluate_function: MulBackward0', 200, 40, args={'Sequence number': 7}),
              ev('cudaLaunchKernel', 220, 2, 'cuda_runtime', {'correlation': 11}),
              ev('cudaLaunchKernel', 300, 2, 'cuda_runtime', {'correlation': 12}),
              ev('mul_fwd', 50, 10, 'kernel', {'correlation': 10}, tid=9),
              ev('mul_bwd', 250, 10, 'kernel', {'correlation': 11}, tid=9),
              ev('other', 350, 10, 'kernel', {'correlation': 12}, tid=9),
              ev('Memcpy DtoH', 400, 3, 'gpu_memcpy', {'correlation': 13}, tid=9)]
    result = P.trace_counts({'traceEvents': events})
    assert result['cuda_kernel_launches'] == 3
    assert result['naive_eda_kernel_launches'] == 2
    assert result['naive_eda_backward_ranges'] == 1
    assert result['naive_eda_cuda_kernel_s'] == pytest.approx(.00002)


def test_eda_scopes_restore_and_count_cpu_ops(monkeypatch, tmp_path):
    module = ModuleType('fake.eda_kernels')
    exec('def _scan(x):\n    return (x * x).sum()', module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = module._scan
    r = P.Recorder(tmp_path / 'eda', 1, cuda=False)
    x = torch.ones(4, requires_grad=True)
    with r.instrument_backward():
        with r.session('update'), r.phase('forward'):
            r.install_eda_scopes()
            module._scan(x).backward()
    assert module._scan is original
    assert r.rows[0]['naive_eda_ranges'] == 1
    assert r.rows[0]['naive_eda_wall_s'] > 0
    assert r.rows[0]['backward_wall_s'] > 0
    assert r.rows[0]['cuda_kernel_launches'] == 0


def test_ladder_recipe_and_schedule_preserved():
    recipe = Path(P.__file__).with_name('C64_ECTRL_ARGV.txt').read_text().split('|')[0]
    import shlex
    a = E.build_parser().parse_args(shlex.split(recipe)[1:] + ['--out', '/tmp/unused'])
    assert (a.steps, a.budget, a.seed, a.rehearsal_frac, a.budget_freeze, a.micro) == (800, 1000000, 1, .2, 200, 6)
    assert a.layers == '3,16,22,29'
    r = P.Recorder('/tmp/C64_unused', 20, cuda=False)
    adapted = P.adapted_main(E, r)
    assert adapted.__globals__['bop_step'] is E.bop_step
    assert adapted.__globals__['refresh_support_event'] is E.refresh_support_event


def test_twenty_update_prefix_keeps_800_step_horizon(cpu_harness, tmp_path):
    _, paths = cpu_harness
    argv = ['--method', 'fsa_forced', '--optim', 'bop', '--layers', '0,1',
            '--people', '3', '--facts-per-person', '2', '--shared-attrs', '2',
            '--bop-tau', '.7', '--budget', '20', '--budget-freeze', '200', '--support-score', 'netcost',
            '--rehearsal-frac', '.2', '--local-lambda', '10', '--no-churn',
            '--micro', '6', '--fact-batch', '3', '--steps', '800', '--seed', '1',
            '--sparse-state', '--expert-checkpoint', '--answer-only', '--aug', '--no-pareto',
            '--anchor', paths['anchor'], '--anchor-bs', '2', '--anchor-len', '8',
            '--eval-set', paths['eval'], '--eval-windows', '1', '--val-windows', '1',
            '--eval-every', '100', '--gen-max-new', '1', '--eda-backend', 'naive_fp32',
            '--out', str(tmp_path / 'unused.json')]
    recorder = P.run(E, argv, tmp_path / 'twenty', updates=20, cuda=False)
    assert len(recorder.rows) == 21
    assert [r['update'] for r in recorder.rows[:20]] == list(range(1, 21))
    assert [e['update'] for e in recorder.training_events['bop']] == list(range(1, 21))
    assert [e['update'] for e in recorder.training_events['refresh']] == [1, 11]
    assert all(r['eval_wall_s'] == 0 for r in recorder.rows[:20])
    assert recorder.rows[-1]['kind'] == 'eval' and recorder.rows[-1]['eval_wall_s'] > 0
    assert recorder.metadata['recipe']['steps'] == 800
    assert not (tmp_path / 'unused.json').exists()
