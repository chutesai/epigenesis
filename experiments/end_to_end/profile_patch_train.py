"""C64: instrument a short prefix of imported e2e_fsa.main, without editing it.

The AST adapter only cuts evaluation/checkpoint tails, bounds the outer loop,
and inserts scopes. All training expressions and imported helpers are retained.
"""
import argparse
from bisect import bisect_right
import ast
import cProfile
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
from pathlib import Path
import pstats
import shlex
import sys
import time

import torch

PHASES = ('data_tokenise', 'forward', 'backward', 'bop_bookkeeping', 'optimizer', 'eval')


def call_stmt(source):
    return ast.parse(source).body[0]


def scoped(body, phase):
    return ast.With(items=[ast.withitem(context_expr=ast.parse(
        f'_c64.phase({phase!r})', mode='eval').body)], body=body)


def adapted_main(E, recorder):
    """Fail closed if the expected harness boundaries change."""
    source = inspect.getsource(E.main)
    tree = ast.parse(source)
    fn = tree.body[0]
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)
    starts = [i for i, n in enumerate(fn.body) if assignment(n, 'baseline')]
    ends = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
            and n.value.func.id == 'print' and n.value.args
            and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == 'BASE']
    loops = [n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and n.target.id == 'step']
    if len(starts) != 1 or len(ends) != 1 or len(loops) != 1 or ends[0] < starts[0]:
        raise RuntimeError('C64 adapter: unrecognised main baseline/training boundaries')
    loop = loops[0]
    cuts = [i for i, n in enumerate(loop.body) if assignment(n, 'save_now')]
    if len(cuts) != 1:
        raise RuntimeError('C64 adapter: unrecognised evaluation boundary')
    # Preserve a.steps and total_steps in every training expression, including bop_now.
    loop.iter = ast.parse('range(start_step, min(total_steps, _c64.updates))', mode='eval').body
    expected_losses = {ast.dump(ast.parse(test, mode='eval').body): test for test in (
        'a.rehearsal_frac > 0', '(local is not None and lam > 0) or gate is not None',
        'anchor is not None and a.drift_lambda > 0', 'preservation_kl is not None',
        'route_kl is not None')}
    class LossScopes(ast.NodeTransformer):
        def __init__(self):
            self.found = set()

        def visit_If(self, node):
            self.generic_visit(node)
            # These branches also construct losses outside the forward helpers.
            # AST dumps ignore redundant parentheses and source locations.
            key = ast.dump(node.test)
            if key in expected_losses:
                self.found.add(key)
                return scoped([node], 'forward')
            return node
    loss_scopes = LossScopes()
    training = [loss_scopes.visit(statement) for statement in loop.body[:cuts[0]]]
    missing = [test for key, test in expected_losses.items() if key not in loss_scopes.found]
    if missing:
        raise RuntimeError('C64 adapter: missing expected loss scopes: ' + '; '.join(missing))
    loop.body = [scoped(training, 'bop_bookkeeping')]
    loop.body = [call_stmt('_c64.install_eda_scopes()'), scoped(loop.body, 'update')]
    # Each nested function runs the original body, with an exclusive phase scope.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ('make_batch', 'fact_loss_backward'):
            node.body = [scoped(node.body, 'data_tokenise' if node.name == 'make_batch' else 'forward')]
    li = fn.body.index(loop)
    fn.body = (fn.body[:starts[0]] + fn.body[ends[0] + 1:li + 1]
               + [call_stmt('_c64.evaluate(acq_suite, model, big, extra_panels, dev)'),
                  call_stmt('_c64.capture_state(mods, bop_events, refresh_events)')])
    namespace = dict(E.__dict__, _c64=recorder)
    for name in ('logits_of', 'general_ce'):
        namespace[name] = recorder.wrap(namespace[name], 'forward')
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, E.main.__code__.co_firstlineno - 1)
    exec(compile(tree, inspect.getsourcefile(E.main), 'exec'), namespace)
    return namespace['main']


def trace_counts(trace):
    """Count device kernel events; attribute launches through CPU ancestry/sequence.

    Forward _scan ranges identify naive EDA ops. Autograd sequence numbers link
    their backward nodes, including checkpoint recomputation, to those ops.
    """
    events = trace.get('traceEvents', [])
    ranges = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'C64/naive_eda']
    def index_ranges(items):
        by_thread = {}
        for r in items:
            by_thread.setdefault(r.get('tid'), []).append((r['ts'], r['ts'] + r.get('dur', 0)))
        result = {}
        for tid, spans in by_thread.items():
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            result[tid] = ([s for s, _ in merged], merged)
        return result
    def inside(e, index):
        starts, spans = index.get(e.get('tid'), ([], []))
        i = bisect_right(starts, e.get('ts', 0)) - 1
        return i >= 0 and e['ts'] + e.get('dur', 0) <= spans[i][1] + .01
    forward_index = index_ranges(ranges)
    seqs = {e.get('args', {}).get('Sequence number') for e in events
            if e.get('cat') == 'cpu_op' and inside(e, forward_index)} - {None, -1}
    backward = [e for e in events if e.get('ph') == 'X'
                and e.get('args', {}).get('Sequence number') in seqs
                and ('Backward' in e.get('name', '') or 'autograd::engine' in e.get('name', ''))]
    eda_index = index_ranges(ranges + backward)
    launches = [e for e in events if e.get('ph') == 'X'
                and ('cudaLaunchKernel' in e.get('name', '') or 'cuLaunchKernel' in e.get('name', ''))]
    eda_launches = [e for e in launches if inside(e, eda_index)]
    correlations = {e.get('args', {}).get('correlation') for e in eda_launches} - {None}
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    eda_kernels = [e for e in kernels if e.get('args', {}).get('correlation') in correlations]
    return dict(cuda_kernel_launches=len(kernels), cuda_runtime_launch_calls=len(launches),
                naive_eda_kernel_launches=len(eda_kernels), naive_eda_runtime_launch_calls=len(eda_launches),
                naive_eda_cuda_kernel_s=sum(e.get('dur', 0) for e in eda_kernels) / 1e6,
                naive_eda_ranges=len(ranges), naive_eda_backward_ranges=len(backward),
                eda_attribution='forward ranges + autograd sequence; runtime correlation to device kernels')


class Recorder:
    def __init__(self, output, updates=20, cuda=True):
        self.metadata = {}
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.updates = updates
        self.cuda = cuda and torch.cuda.is_available()
        self.rows = []
        self.stack = []
        self.current = None
        self.prof = None
        self.cprof = cProfile.Profile()
        self.patches = []
        self.eda_functions = []
        self.cpu_ops = {}
        self.kernels = {}
        self.eval_result = None

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def phase(self, name):
        if name == 'update':
            with self.session('update'):
                yield
            return
        if self.current is None:
            yield
            return
        self.sync()
        start = time.perf_counter()
        slot = [name, start, 0.]
        self.stack.append(slot)
        try:
            with torch.profiler.record_function('C64/' + name):
                yield
        finally:
            self.sync()
            elapsed = time.perf_counter() - start
            self.stack.pop()
            self.current[name + '_wall_s'] = self.current.get(name + '_wall_s', 0.) + elapsed - slot[2]
            if self.stack:
                self.stack[-1][2] += elapsed

    @contextmanager
    def session(self, kind):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if not self.rows:
            self.metadata['setup_wall_s'] = time.perf_counter() - self.run_start if hasattr(self, 'run_start') else None
        self.current = dict(kind=kind, update=len(self.rows) + 1 if kind == 'update' else None,
                            **{p + '_wall_s': 0. for p in PHASES})
        self.prof = torch.profiler.profile(activities=activities, with_stack=True)
        self.sync()
        start = time.perf_counter()
        self.prof.__enter__()
        self.cprof.enable()
        try:
            with torch.profiler.record_function('C64/' + kind):
                yield
        finally:
            self.sync()
            self.cprof.disable()
            self.current['wall_s'] = time.perf_counter() - start
            self.prof.__exit__(None, None, None)
            trace = self.output.with_name(self.output.name + f'.{kind}{len(self.rows) + 1}.trace.json')
            self.prof.export_chrome_trace(str(trace))
            trace_data = json.loads(trace.read_text())
            self.current.update(trace_counts(trace_data))
            self.current['trace'] = str(trace)
            self.current['unattributed_wall_s'] = self.current['wall_s'] - sum(
                self.current[p + '_wall_s'] for p in PHASES)
            for e in self.prof.key_averages():
                row = self.cpu_ops.setdefault(e.key, dict(name=e.key, count=0, self_cpu_s=0., cpu_total_s=0.))
                row['count'] += e.count
                row['self_cpu_s'] += e.self_cpu_time_total / 1e6
                row['cpu_total_s'] += e.cpu_time_total / 1e6
            for e in trace_data.get('traceEvents', []):
                if e.get('cat') == 'kernel' and e.get('ph') == 'X':
                    row = self.kernels.setdefault(e['name'], dict(name=e['name'], count=0, cuda_s=0.))
                    row['count'] += 1
                    row['cuda_s'] += e.get('dur', 0) / 1e6
            self.rows.append(self.current)
            self.current = None
            self.prof = None
            self.write()

    def wrap(self, fn, phase):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with self.phase(phase):
                return fn(*args, **kwargs)
        return wrapped

    @contextmanager
    def instrument_backward(self):
        original_zero = torch.optim.Optimizer.zero_grad
        original_step = torch.optim.SGD.step
        torch.optim.Optimizer.zero_grad = self.wrap(original_zero, 'optimizer')
        torch.optim.SGD.step = self.wrap(original_step, 'optimizer')
        original = torch.autograd.backward
        torch.autograd.backward = self.wrap(original, 'backward')
        try:
            yield
        finally:
            torch.autograd.backward = original
            torch.optim.Optimizer.zero_grad = original_zero
            torch.optim.SGD.step = original_step
            for module, name, fn in reversed(self.patches):
                setattr(module, name, fn)
            self.patches.clear()

    def wrap_eda(self, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self.current is None:
                return fn(*args, **kwargs)
            self.sync()
            start = time.perf_counter()
            try:
                with torch.profiler.record_function('C64/naive_eda'):
                    return fn(*args, **kwargs)
            finally:
                self.sync()
                self.current['naive_eda_wall_s'] = self.current.get('naive_eda_wall_s', 0.) + time.perf_counter() - start
        return wrapped

    def install_eda_scopes(self):
        # The export loader imports mesh lazily. Patch _scan and any imported aliases.
        if self.eda_functions:
            return
        known = {(id(m), n) for m, n, _ in self.patches}
        for module in list(sys.modules.values()):
            if module is None or not isinstance(module, type(sys)):
                continue
            for name, fn in list(vars(module).items()):
                if (type(fn) is type(self.install_eda_scopes.__func__) and fn.__name__ == '_scan'
                        and fn.__module__.endswith('eda_kernels') and (id(module), name) not in known):
                    self.patches.append((module, name, fn))
                    setattr(module, name, self.wrap_eda(fn))
                    self.eda_functions.append(module.__name__ + '.' + name)
        if self.cuda and not self.eda_functions:
            raise RuntimeError('No naive EDA _scan found; inspect the box EDA implementation before profiling')

    def evaluate(self, suite, model, big, panels, dev):
        import e2e_fsa as E
        with self.session('eval'), self.phase('eval'), torch.no_grad():
            # One pass across the acquisition and text panels, no baseline/delta work.
            self.eval_result = dict(acquisition=suite(with_train=True),
                                    big=E.nll_windows(model, big, dev) if big is not None else None,
                                    extra={k: E.nll_windows(model, w, dev) for k, (_, w) in panels.items()})

    def capture_state(self, mods, bop_events, refresh_events):
        self.training_events = dict(bop=bop_events, refresh=refresh_events)
        # Tests can check exact masters/EMAs without writing an 8B patch to disk.
        if not self.cuda:
            self.state = [(m.M.detach().cpu().clone(), m.bop_ema.detach().cpu().clone(),
                           m.support.detach().cpu().clone()) for m in mods]

    def write(self):
        stats = pstats.Stats(self.cprof)
        top = sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[:30]
        result = dict(schema='C64-profile-v1', torch_version=torch.__version__, cuda=self.cuda,
                      requested_updates=self.updates, metadata=getattr(self, 'metadata', {}),
                      eda_functions=self.eda_functions, rows=self.rows,
                      eval_result=self.eval_result, training_events=getattr(self, 'training_events', {}),
                      cprofile_top30=[dict(function=f'{k[0]}:{k[1]}:{k[2]}', primitive_calls=v[0],
                                          calls=v[1], self_s=v[2], cumulative_s=v[3]) for k, v in top],
                      top_cpu_ops=sorted(self.cpu_ops.values(), key=lambda r: r['self_cpu_s'], reverse=True)[:30],
                      top_cuda_kernels=sorted(self.kernels.values(), key=lambda r: r['cuda_s'], reverse=True)[:30])
        self.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        columns = ['update', 'wall_s'] + [p + '_wall_s' for p in PHASES] + [
            'unattributed_wall_s', 'naive_eda_wall_s', 'cuda_kernel_launches', 'naive_eda_kernel_launches']
        lines = ['# C64 patch training profile', '',
                 'Times are synchronized, instrumented wall seconds. EDA wall is an inclusive subset of forward/backward/eval; '
                 'EDA kernel seconds in JSON are device work, not additive wall time. '
                 'cProfile and op/kernel rankings include training and the separate eval, excluding setup. '
                 'One trace per update bounds profiler retention; trace export is excluded from wall time.', '',
                 '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        for row in self.rows:
            lines.append('| ' + ' | '.join(f'{row.get(c, 0):.6f}' if isinstance(row.get(c, 0), float)
                         else str((row.get(c) or row['kind']) if c == 'update' else row.get(c, 0)) for c in columns) + ' |')
        for title, rows, fields in [('Top 30 cProfile functions', result['cprofile_top30'], ['function', 'calls', 'cumulative_s', 'self_s']),
                                    ('Top CPU ops', result['top_cpu_ops'], ['name', 'count', 'self_cpu_s', 'cpu_total_s']),
                                    ('Top CUDA kernels', result['top_cuda_kernels'], ['name', 'count', 'cuda_s'])]:
            lines += ['', '## ' + title, '', '| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
            lines += ['| ' + ' | '.join(str(r[f]).replace('|', '\\|') for f in fields) + ' |' for r in rows]
        self.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
        self.cprof.dump_stats(str(self.output.with_suffix('.pstats')))


def run(E, argv, output, updates=20, cuda=True):
    if updates < 1:
        raise ValueError('updates must be positive')
    a = E.build_parser().parse_args(argv)
    if a.optim != 'bop' or a.method != 'fsa_forced' or E.eda.selected(a) != 'naive_fp32':
        raise ValueError('C64 requires fsa_forced / bop / naive_fp32')
    if a.steps < updates or a.bop_accum != 1 or a.adam_phase or a.repair_steps or a.load_patch:
        raise ValueError('C64 requires an ordinary Bop prefix with one Bop update per step')
    recorder = Recorder(output, updates, cuda)
    recorder.metadata = dict(argv=argv, source=inspect.getsourcefile(E.main),
        source_sha256=hashlib.sha256(Path(inspect.getsourcefile(E.main)).read_bytes()).hexdigest(),
        recipe=E.training_recipe(a), disabled='base/periodic/final eval and checkpoint writes; one separate eval')
    main = adapted_main(E, recorder)
    recorder.run_start = time.perf_counter()
    with recorder.instrument_backward():
        main(argv)
    recorder.write()
    return recorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-out', type=Path, default=Path('out/C64_patch_profile'))
    parser.add_argument('--profile-updates', type=int, default=20)
    parser.add_argument('--recipe-file', type=Path, help='ladder command file; ignores script prefix and |run label')
    args, recipe = parser.parse_known_args()
    if recipe and recipe[0] == '--':
        recipe = recipe[1:]
    if args.recipe_file:
        if recipe:
            parser.error('use --recipe-file or trailing recipe argv, not both')
        recipe = shlex.split(args.recipe_file.read_text().split('|')[0])[1:]
    if not recipe:
        parser.error('pass the full e2e_fsa recipe after --')
    if '--out' not in recipe and not any(x.startswith('--out=') for x in recipe):
        recipe += ['--out', str(args.profile_out) + '.recipe.json']
    import e2e_fsa as E
    run(E, recipe, args.profile_out, args.profile_updates)


if __name__ == '__main__':
    main()
