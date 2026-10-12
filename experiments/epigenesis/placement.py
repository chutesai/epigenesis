"""Exploratory EP-1 concept/rule placement: train, attribute, and trace.

Examples (GPU box, under the experiment's lock convention)::
    python placement.py --mode train --run out/R --out out/placement --targets all
    python placement.py --mode attribute --run out/R --out out/attribute --state out/placement/best_state.pt
    python placement.py --mode trace --run out/R --out out/trace --state out/placement/best_state.pt --probes 20

Block ids are MeshModel.layers indices, not MoE ordinals. All factors and Adam
moments are fp32. The 6 GiB guard counts factors and two Adam moments (12 bytes
per parameter), excluding gradients, activations, and the frozen model.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from types import MethodType

import torch
from torch import nn
from torch.nn.utils import parametrize

import ablate as a
import epi_gpu as g

RANK = 16
MEMORY_LIMIT = 6 * 2**30
INDEXER_POLICY = "MSA indexer fixed; index projections excluded from LoRA (discrete selection, no d2 gradient)."
MIXER_TYPES = {'EDALayer', 'AttentionLayer', 'MSAAttentionLayer',
               'MLAAttentionLayer', 'DSAAttentionLayer', 'DeepSeekHCAAttentionLayer'}


@dataclass
class Target:
    name: str
    layer: int
    kind: str
    linear: nn.Linear

    def metadata(self):
        return dict(name=self.name, layer=self.layer, kind=self.kind,
                    shape=list(self.linear.weight.shape))


def enumerate_linears(model):
    """Inspect real modules, including EDA Sequential; keep the MSA indexer fixed."""
    targets, families, seen = [], [], set()
    for li, block in enumerate(model.layers):
        moes = [(name, mod) for name, mod in block.named_modules()
                if hasattr(mod, 'experts') and hasattr(mod, 'router')]
        mixers = [(name, mod) for name, mod in block.named_modules()
                  if type(mod).__name__ in MIXER_TYPES]
        families.append(dict(layer=li, mixer=','.join(type(m).__name__ for _, m in mixers) or 'none',
                             has_moe=bool(moes), block_type=type(block).__name__))
        candidates = []
        for prefix, moe in moes:
            for bank in ('experts', 'shared_experts'):
                for ei, expert in enumerate(getattr(moe, bank, None) or []):
                    if expert is None:
                        raise ValueError(f'Block {li} has an unmaterialized {bank}.{ei}')
                    for proj in ('up_proj', 'down_proj'):
                        lin = getattr(expert, proj, None)
                        if not isinstance(lin, nn.Linear):
                            raise ValueError(f'Block {li} {bank}.{ei}.{proj} must be nn.Linear (bf16 build)')
                        candidates.append(('.'.join(filter(None, [prefix, bank, str(ei), proj])), 'experts', lin))
        for prefix, mixer in mixers:
            for name, lin in mixer.named_modules():
                if (isinstance(lin, nn.Linear)
                        and not (type(mixer).__name__ == 'MSAAttentionLayer'
                                 and name.split('.')[0].startswith('index'))):
                    candidates.append(('.'.join(filter(None, [prefix, name])), 'mixers', lin))
        for name, kind, lin in candidates:
            if id(lin) in seen:
                raise ValueError('Shared target linear has ambiguous block ownership')
            seen.add(id(lin))
            targets.append(Target(f'layers.{li}.{name}', li, kind, lin))
        if not moes and not mixers:
            raise ValueError(f'Unsupported block {li}: {type(block).__name__}')
    if not targets:
        raise ValueError('No placement targets found')
    return targets, families


def parameter_counts(targets, rank=RANK):
    counts = {kind: sum(rank * sum(t.linear.weight.shape) for t in targets if t.kind == kind)
              for kind in ('experts', 'mixers')}
    counts['all'] = counts['experts'] + counts['mixers']
    return {kind: dict(parameters=n, params_adam_bytes=12 * n, params_adam_gib=12 * n / 2**30)
            for kind, n in counts.items()}


def check_memory(targets, limit=MEMORY_LIMIT):
    n = sum(RANK * sum(t.linear.weight.shape) for t in targets)
    if n * 12 > limit:
        raise ValueError(f'{n:,} LoRA parameters + fp32 Adam require {n * 12 / 2**30:.3f} GiB '
                         f'(limit {limit / 2**30:g} GiB); use --targets mixers or '
                         '--targets experts --layers <block ids> subsets')
    return n


class LoRA(nn.Module):
    def __init__(self, weight):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.A = nn.Parameter(torch.empty(RANK, in_dim, device=weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_dim, RANK, device=weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, weight):
        # alpha = rank: scaling is exactly one, with no mask/local penalty.
        return weight + (self.B @ self.A).to(weight.dtype) if self.enabled else weight


def attach(model, targets, training=True):
    if training:
        check_memory(targets)  # Before allocating any factors or optimizer state.
    for p in model.parameters():
        p.requires_grad_(False)
    # Cached frozen/grouped or DeepGEMM weights would survive zeroed() toggles.
    # Use the plain ExpertMLP dispatch for both training and interventions.
    os.environ['MESH_FROZEN_EXPERT_GROUPED'] = '0'
    os.environ['MESH_TRAINABLE_EXPERT_GROUPED'] = '0'
    for block in model.modules():
        if hasattr(block, 'experts') and hasattr(block, 'router'):
            block.use_deepgemm = False
    modules = {}
    for t in targets:
        if parametrize.is_parametrized(t.linear, 'weight'):
            raise ValueError(f'Already parametrized: {t.name}')
        m = LoRA(t.linear.weight)
        parametrize.register_parametrization(t.linear, 'weight', m)
        modules[t.name] = m
    return modules


@contextmanager
def zeroed(modules, names):
    previous = {name: modules[name].enabled for name in names}
    try:
        for name in previous:
            modules[name].enabled = False
        yield
    finally:
        for name, value in previous.items():
            modules[name].enabled = value


@torch.no_grad()
def dev_accuracy(model, tok, device, probes):
    result = a.evaluate.concept_acc(model, tok, device, probes)
    result['se'] = math.sqrt(result['acc'] * (1 - result['acc']) / result['n']) if result['n'] else 0.
    return result


def checkpoint_blocks(model):
    """Recompute entire blocks, including effective expert weights, also in eval."""
    from torch.utils.checkpoint import checkpoint
    for block in model.layers:
        forward = block.forward
        def wrapped(*args, _forward=forward, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)
        block.forward = wrapped


def probe_memory(model, tok, device, acquisition, anchors, selection, probes, opt):
    if torch.device(device).type != 'cuda':
        raise ValueError('--probe-memory requires CUDA')
    # Largest legal padded micro from the augmented training pool (up to 8 items).
    variants = [v for item in acquisition for v in item['variants']]
    batch = max(([v] * min(8, 160 // len(v['row'])) for v in variants),
                key=lambda b: (len(b) * len(b[0]['row']), len(b[0]['row'])))
    def measure(fn):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 2**30
    def step():
        opt.zero_grad(set_to_none=True)
        a.training_backward(lambda x: g.logits_of(model, x), batch, anchors[:2],
                            device, g.VOCAB, 'placement', 'd2')
        opt.step()
    train_peak = measure(step)
    @torch.no_grad()
    def evaluate():
        g.fsa.nll_windows(model, selection, device)
        dev_accuracy(model, tok, device, probes)
    eval_peak = measure(evaluate)
    result = dict(probe_memory=True, training_micro_shape=[len(batch), len(batch[0]['row'])],
                  train_peak_gib=train_peak, eval_peak_gib=eval_peak, msa_indexer=INDEXER_POLICY)
    print(json.dumps(result), flush=True)
    return result


def train(args, model, device, tok, corpus, targets, families, counts):
    modules = attach(model, targets)
    checkpoint_blocks(model)
    rng = random.Random(args.seed + 1)
    acquisition = []
    for case in a.build_cases(corpus, 1, args.seed, args.cases_per_policy):
        variants = [a.encode_item(case, r, tok, g.BOS) for r in case['formats']]
        if any(len(v['row']) > 160 for v in variants):
            raise ValueError('Training row exceeds the fixed 160-token micro budget')
        acquisition.append(dict(variants[0], variants=variants))
    ids = sorted({item['policy_id'] for item in acquisition})
    if not ids:
        raise ValueError('No introduced policies')
    probes = a.large_dev_concepts(corpus, ids, args.dev_large, corpus['seed'])
    tokens = json.loads((Path(g.DATA) / 'anchor-wikitext.json').read_text())
    if len(tokens) < 150000 + 16 * 511:
        raise ValueError('Anchor and selection pools must be disjoint')
    pool = tokens[:150000]
    rows = []
    for _ in range(64):
        offset = rng.randrange(len(pool) - 94)
        rows.append([g.BOS] + pool[offset:offset + 95])
    records = g.topk_records(model, rows, [1] * 64, device, 32, 4096)
    anchors = [dict(row=row, start=1, record=rec) for row, rec in zip(rows, records)]
    selection = torch.tensor(a.evaluate.token_windows(tokens[-16 * 511:], 16), device=device)
    opt = torch.optim.Adam([p for m in modules.values() for p in m.parameters()], lr=1e-3)
    if args.probe_memory:
        return probe_memory(model, tok, device, acquisition, anchors, selection, probes, opt)
    start_nll = g.fsa.nll_windows(model, selection, device)
    config = dict(msa_indexer=INDEXER_POLICY, block_checkpoint='non-reentrant whole blocks',
                  seed=args.seed, dev_large=args.dev_large, cases_per_policy=args.cases_per_policy,
                  trained_policy_ids=ids, targets=args.targets, layers=args.layers, rank=RANK,
                  alpha=RANK, lr=1e-3, local_penalty=0, data='d2',
                  steps=args.steps, eval_every=args.eval_every,
                  expert_format='bf16', eda_kernel_backend='naive_fp32',
                  items_per_step=8, anchor_per_step=2, acquisition_weight=.6, general_weight=.4)
    metadata = [t.metadata() for t in targets]
    trajectory, best = [], -math.inf
    with (args.out / 'log.jsonl').open('w') as log:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)

        def evaluate_step(step):
            nonlocal best
            nll = g.fsa.nll_windows(model, selection, device)
            row = dict(phase='eval', step=step, dev=dev_accuracy(model, tok, device, probes),
                       select_dnll=sum(x - y for x, y in zip(nll, start_nll)) / len(start_nll))
            trajectory.append(row)
            row['smoothed'] = a.smoothed_acc(trajectory)
            if row['smoothed'] is not None and row['smoothed'] > best:
                best = row['smoothed']
                torch.save(dict(version=1, step=step, smoothed=best, config=config,
                                targets=metadata, families=families, counts=counts,
                                state=a.trainable_state(modules)), args.out / 'best_state.pt')
            emit(row)

        evaluate_step(0)
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            ab = [rng.choice(anchors) for _ in range(2)]
            batch = a.sample_policies(acquisition, 8, rng, augment=True)
            losses = a.training_backward(lambda x: g.logits_of(model, x), batch, ab,
                                         device, g.VOCAB, 'placement', 'd2')
            opt.step()
            emit(dict(phase='train', step=step, **losses, local_penalty=0.))
            if step % args.eval_every == 0 or step == args.steps:
                evaluate_step(step)
    return dict(config=config, counts=counts, targets=metadata, families=families,
                trajectory=trajectory, best_smoothed=best, smoothed=a.smoothed_acc(trajectory),
                checkpoint=str(args.out / 'best_state.pt'))


def load_state(model, targets, families, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('version') != 1 or saved['config']['rank'] != RANK:
        raise ValueError('Unsupported placement checkpoint')
    names = {t['name'] for t in saved['targets']}
    chosen = [t for t in targets if t.name in names]
    if [t.metadata() for t in chosen] != saved['targets'] or families != saved['families']:
        raise ValueError('Checkpoint module tree/shape/block family mismatch')
    if set(saved['state']) != names:
        raise ValueError('Checkpoint state keys do not match target metadata')
    modules = attach(model, chosen, training=False)
    for name, module in modules.items():
        module.load_state_dict(saved['state'][name], strict=True)
        module.requires_grad_(False)
    return saved, chosen, modules


def attribution_groups(targets, families):
    n = len(families)
    groups = [dict(name=f'layer_{f["layer"]}', **f,
                   names=[t.name for t in targets if t.layer == f['layer']]) for f in families]
    # Contiguous thirds: floor boundaries, with any remainder in later bands.
    for name, lo, hi in [('early', 0, n // 3), ('middle', n // 3, 2 * n // 3), ('late', 2 * n // 3, n)]:
        groups.append(dict(name=name, layers=list(range(lo, hi)),
                           names=[t.name for t in targets if lo <= t.layer < hi]))
    for kind in ('mixers', 'experts'):
        groups.append(dict(name=kind + '_only', names=[t.name for t in targets if t.kind == kind]))
    return groups


def attribute(model, tok, device, probes, targets, families, modules):
    baseline = dev_accuracy(model, tok, device, probes)
    rows = []
    for group in attribution_groups(targets, families):
        with zeroed(modules, group['names']):
            metric = dev_accuracy(model, tok, device, probes)
        row = dict(group, dev=metric, drop=baseline['acc'] - metric['acc'])
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'names'}), flush=True)
    return dict(baseline=baseline, attribution=rows)


class UnalignedCorruption(ValueError):
    pass


def corrupt_probe(probe, policies, tok=None):
    policy = next(p for p in policies if p['id'] == probe['policy_id'])
    attr = policy['attribute']
    if attr not in ('tier', 'channel'):
        raise ValueError(f'Concept tracing requires tier/channel, got {attr}')
    for other in sorted(policies, key=lambda p: p['id']):
        if (other['category'] != policy['category'] or other['attribute'] != attr
                or other['action'] == policy['action']):
            continue
        for value in other['condition']['values']:
            case = dict(probe['case'], **{attr: value})
            matches = [p for p in policies if a.case_matches(p, case)]
            if len(matches) == 1 and matches[0]['id'] == other['id']:
                clean_ticket = a.render_ticket(probe['case'])
                if probe['prompt'].count(clean_ticket) != 1:
                    raise ValueError('Probe must contain exactly one rendered ticket')
                corrupted = dict(probe, case=case, prompt=probe['prompt'].replace(clean_ticket, a.render_ticket(case)))
                if (tok is not None and len(g.enc(tok, corrupted['prompt'].rstrip()))
                        != len(g.enc(tok, probe['prompt'].rstrip()))):
                    continue
                return corrupted, \
                    dict(attribute=attr, clean_value=probe['case'][attr], corrupted_value=value,
                         corrupted_policy_id=other['id'])
    raise UnalignedCorruption(f'No unique aligned different-action corruption for {probe["id"]}')


def hidden(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise ValueError('Unsupported block hidden output')


def replace_hidden(output, clean):
    if hidden(output).shape != clean.shape:
        raise ValueError('Clean and corrupted block outputs must align')
    clean = clean.to(device=hidden(output).device, dtype=hidden(output).dtype)
    if torch.is_tensor(output):
        return clean
    return (clean, *output[1:]) if isinstance(output, tuple) else [clean, *output[1:]]


def tracing_path(model):
    if not hasattr(model, 'config'):  # Plain CPU fixture / unwrapped stacks.
        return 'plain'
    assert not model.config.use_mhc and not len(model.mhc), 'Tracing requires mHC off'
    assert model.config.use_block_attn_res and model._block_attn_res_mode == 'paper', 'Tracing requires paper BAR'
    assert not model.config.adaptive_loop_count and not model.config.atlas_memory_count
    return 'paper_bar'


@contextmanager
def wrapped_states(model, intervention):
    """Expose paper BAR state after each layer's delta enters the partial sum."""
    if tracing_path(model) == 'plain':
        handles = [block.register_forward_hook(
            lambda mod, inputs, output, i=i: replace_hidden(output, intervention(i, hidden(output))))
            for i, block in enumerate(model.layers)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
        return
    original = model.__dict__.get('_forward_block_attn_res')
    def forward(self, x, cu_seqlens=None, max_seqlen=None, attn_mask=None, *, record_load=False):
        blocks = [x]
        size = max(1, self._block_attn_res_block_size)
        for start in range(0, len(self.layers), size):
            end = min(len(self.layers), start + size)
            modules = self.block_attn_res[start:end]
            weighted, maximum, denom = modules[0].batched_inter_stats(modules, blocks)
            partial = None
            for local, i in enumerate(range(start, end)):
                module = modules[local]
                h = (module.stats_to_output(weighted[local], denom[local]) if partial is None else
                     module.merge_inter_stats_with_partial(weighted[local], maximum[local], denom[local], partial))
                y = self._forward_layer_with_loops(i, self.layers[i], h, cu_seqlens=cu_seqlens,
                                                  max_seqlen=max_seqlen, attn_mask=attn_mask,
                                                  record_load=False)
                y = self._apply_atlas_memory(i, y)
                delta = y - h
                partial = delta if partial is None else partial + delta
                state = (blocks, partial)
                restored = intervention(i, state)
                if restored is not state:
                    blocks, partial = restored
                    # Cached completed-block statistics must follow restored sums too.
                    weighted, maximum, denom = modules[0].batched_inter_stats(modules, blocks)
            blocks.append(partial)
        return self.block_attn_res_head(blocks) if self.block_attn_res_head is not None else blocks[-1]
    model._forward_block_attn_res = MethodType(forward, model)
    try:
        yield
    finally:
        if original is None:
            del model._forward_block_attn_res
        else:
            model._forward_block_attn_res = original


def copy_state(state, device):
    if torch.is_tensor(state):
        return state.detach().to(device).clone()
    return type(state)(copy_state(x, device) for x in state)


@torch.no_grad()
def trace_probe(model, clean_ids, corrupt_ids, token, device):
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError('Clean and corrupted token lengths must align')
    clean_x = torch.tensor([clean_ids], device=device)
    corrupt_x = torch.tensor([corrupt_ids], device=device)
    cache = {}
    def capture(i, state):
        if i in cache:
            raise ValueError('Repeated block invocation is unsupported for tracing')
        cache[i] = copy_state(state, 'cpu')
        return state
    def prob(x):
        return float(g.logits_of(model, x)[0, -1, :g.VOCAB].float().softmax(-1)[token])
    with wrapped_states(model, capture):
        clean = prob(clean_x)
    if set(cache) != set(range(len(model.layers))):
        raise ValueError('Not every block was invoked in the clean forward')
    corrupted = prob(corrupt_x)
    def restore(indices):
        with wrapped_states(model, lambda i, state: copy_state(cache[i], device) if i in indices else state):
            return prob(corrupt_x)
    restored = [restore({i}) for i in range(len(model.layers))]
    return dict(clean=clean, corrupted=corrupted, restored=restored,
                restored_all=restore(set(cache)), restored_none=restore(set()))


def trace(model, tok, device, probes, policies, families):
    rows, rejected = [], 0
    path = tracing_path(model)
    for probe in probes:
        try:
            corrupted, swap = corrupt_probe(probe, policies, tok)
        except UnalignedCorruption:
            rejected += 1
            continue
        clean_ids = [g.BOS] + g.enc(tok, probe['prompt'].rstrip())
        corrupt_ids = [g.BOS] + g.enc(tok, corrupted['prompt'].rstrip())
        answer = g.enc(tok, ' ' + probe['options'][probe['answer_idx']])
        if not answer:
            raise ValueError('Correct option has no tokens')
        row = dict(id=probe['id'], first_token=answer[0], **swap,
                   **trace_probe(model, clean_ids, corrupt_ids, answer[0], device))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return dict(n=len(rows), rejected=rejected, clean=sum(r['clean'] for r in rows) / len(rows) if rows else None,
                corrupted=sum(r['corrupted'] for r in rows) / len(rows) if rows else None, probes=rows,
                alignment='equal token lengths; all absolute positions',
                intervention=('after block i wrapper update: restore all completed BAR sums and running partial sum'
                              if path == 'paper_bar' else 'after block i: restore complete hidden output'),
                blocks=[dict(f, restored_prob=sum(r['restored'][f['layer']] for r in rows) / len(rows) if rows else None)
                        for f in families])


def markdown(result):
    if result.get('probe_memory'):
        return f'Training peak: {result["train_peak_gib"]:.3f} GiB; evaluation peak: {result["eval_peak_gib"]:.3f} GiB.\n'
    if 'attribution' in result:
        base = result['baseline']
        lines = [f'All deltas on: {base["acc"]:.4f} ± {base["se"]:.4f} (n={base["n"]}).', '',
                 '| Zeroed | Mixer | MoE | Accuracy ± SE | Drop |', '|---|---|---|---:|---:|']
        for r in result['attribution']:
            lines.append(f'| {r["name"]} | {r.get("mixer", "—")} | {r.get("has_moe", "—")} | '
                         f'{r["dev"]["acc"]:.4f} ± {r["dev"]["se"]:.4f} | {r["drop"]:.4f} |')
    elif 'blocks' in result:
        if not result['n']:
            return f'No aligned probes; rejected {result["rejected"]}.\n'
        lines = [f'Clean: {result["clean"]:.6f}; corrupted: {result["corrupted"]:.6f}; n={result["n"]}.', '',
                 '| Block | Mixer | MoE | Mean restored first-token probability |', '|---:|---|---|---:|']
        lines.extend(f'| {r["layer"]} | {r["mixer"]} | {r["has_moe"]} | {r["restored_prob"]:.6f} |'
                     for r in result['blocks'])
    else:
        lines = [f'Best three-eval smoothed dev accuracy: {result["best_smoothed"]:.4f}.', '',
                 '| Target | LoRA parameters | Parameters + Adam GiB |', '|---|---:|---:|']
        lines.extend(f'| {k} | {v["parameters"]:,} | {v["params_adam_gib"]:.3f} |' for k, v in result['counts'].items())
    return '\n'.join(lines) + '\n'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('train', 'attribute', 'trace'))
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--state', type=Path)
    p.add_argument('--probe-memory', action='store_true', help='one largest-micro optimizer step and eval; print peak GiB and exit')
    p.add_argument('--targets', choices=('all', 'experts', 'mixers'), default='all')
    p.add_argument('--layers', default='all', help='all or comma-separated transformer block ids (train subset)')
    for name, default in [('steps', 150), ('eval-every', 25), ('dev-large', 400),
                          ('cases-per-policy', 16), ('seed', 0), ('probes', 20)]:
        p.add_argument('--' + name, type=int, default=default)
    args = p.parse_args(argv)
    if args.steps <= 0 or args.eval_every <= 0 or args.probes <= 0:
        p.error('steps, eval-every and probes must be positive')
    if args.dev_large <= 0 or args.dev_large % 2 or args.cases_per_policy < 2 or args.cases_per_policy % 2:
        p.error('dev-large and cases-per-policy must be positive and even')
    if args.probe_memory and args.mode != 'train':
        p.error('--probe-memory requires --mode train')
    if args.mode != 'train' and args.state is None:
        p.error('--state is required for attribute/trace')
    if args.mode == 'trace' and args.probes > args.dev_large:
        p.error('--probes cannot exceed --dev-large')
    if args.mode == 'train' and not args.probe_memory and 1 + args.steps // args.eval_every + int(args.steps % args.eval_every != 0) < 3:
        p.error('Training requires at least three evaluations including step zero')
    if args.layers != 'all':
        try:
            args.layers = [int(x) for x in args.layers.split(',')]
            if min(args.layers) < 0 or len(set(args.layers)) != len(args.layers):
                raise ValueError()
        except ValueError:
            p.error('--layers must be all or distinct nonnegative block ids')
    return args


def run(args):
    corpus = json.loads((args.run / 'corpus.json').read_text())
    torch.manual_seed(args.seed)
    model, device = g.build()  # bf16 experts, naive_fp32 EDA, frozen/eval.
    tok = g.load_tokenizer()
    targets, families = enumerate_linears(model)
    counts = parameter_counts(targets)
    print(json.dumps(dict(phase='parameter_counts', counts=counts, families=families)), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    a.evaluate.write_json(args.out / 'inventory.json', dict(msa_indexer=INDEXER_POLICY, counts=counts, families=families,
                                                         targets=[t.metadata() for t in targets]))
    if args.mode == 'train':
        if args.layers != 'all' and any(i >= len(families) for i in args.layers):
            raise ValueError('--layers contains an out-of-range block id')
        selected = [t for t in targets if (args.targets == 'all' or t.kind == args.targets)
                    and (args.layers == 'all' or t.layer in args.layers)]
        if not selected:
            raise ValueError('Target/layer subset selects no linears')
        print(json.dumps(dict(phase='selected_count', parameters=check_memory(selected))), flush=True)
        result = train(args, model, device, tok, corpus, selected, families, counts)
    else:
        saved, selected, modules = load_state(model, targets, families, args.state)
        config = saved['config']
        # Reconstruct the exact checkpoint's reserved dev set, independent of CLI defaults.
        probes = a.large_dev_concepts(corpus, config['trained_policy_ids'], config['dev_large'], corpus['seed'])
        if args.mode == 'attribute':
            result = attribute(model, tok, device, probes, selected, families, modules)
        else:
            if args.probes > len(probes):
                raise ValueError('--probes exceeds checkpoint dev set')
            result = trace(model, tok, device, probes[:args.probes], corpus['policies'], families)
        result.update(checkpoint=str(args.state), checkpoint_step=saved['step'], counts=counts)
    result['msa_indexer'] = INDEXER_POLICY
    a.evaluate.write_json(args.out / 'result.json', result)
    (args.out / 'report.md').write_text(markdown(result))
    return result


def main():
    run(parse_args())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
