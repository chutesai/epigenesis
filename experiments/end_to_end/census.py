"""SC0 Part 1: forward-only multi-domain census of routed MoE hidden units."""
import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from . import e2e_fsa as E, eda_backend as eda, c44_runtime as recovery
    from .token_streams import read_tokens, OverlapGuard
except ImportError:
    import e2e_fsa as E
    import eda_backend as eda
    import c44_runtime as recovery
    from token_streams import read_tokens, OverlapGuard


FIELDS = ('routed', 'fired', 'relu2_sum', 'relu4_sum', 'tail', 'weighted_relu4_sum')
TAU_TOKENS = 1_000_000
AGREEMENT_TOKENS = 500_000
CENSUS_POLICY = 'C56: nonzero-phi median tau; per-expert energy ranks; bottom20 batch check'


def bottom(rank, fraction):
    return set(rank[:int(len(rank)*fraction)].tolist())


def expert_bottom(ranks, fraction):
    return set().union(*(bottom(rank, fraction) for rank in ranks.values()))


def membership(a, b):
    overlap = len(a & b)/len(a) if a else None
    return dict(batch1_bottom=len(a), main_bottom=len(b), intersection=len(a & b),
                membership_overlap=overlap, passed=overlap >= .95 if overlap is not None else None)


def agreement_pass(rows):
    if any(row['passed'] is False for row in rows):
        return False
    return True if rows and all(row['passed'] is True for row in rows) else None


def distribution(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(measured=int(finite.size), percentiles=(dict(zip(['0','1','10','50','90','99','100'],
        np.percentile(finite, [0,1,10,50,90,99,100]).tolist())) if finite.size else {}))


class Census:
    """Join actual dispatch weights to up-projection rows, in unique token order.

    Force the model's ordinary per-expert fallback, as benchmark_dispatch does:
    fused grouped GEMMs bypass Linear hooks. Duplicate routes are summed BEFORE
    squaring weights; fp8 padding rows are omitted. No router is recomputed.
    The simple Linear/argmax, unweighted top-1 fixture is supported separately.
    """
    def __init__(self, model, domains, min_routed=10000, split_fraction=.3, layers=None, observer=None):
        if min_routed < 1 or not math.isfinite(split_fraction) or not 0 < split_fraction < 1:
            raise ValueError('invalid min-routed or split-fraction')
        self.domains, self.domain = list(domains), 0
        self.observer = observer
        self.min_routed, self.split_fraction = min_routed, split_fraction
        self.hooks, self.methods, self.units, self.split_units = [], [], {}, {}
        self.plans, self.dispatch = {}, {}
        self.active, self.phase, self.split = True, 'measure', 'A'
        self.stream_tokens = torch.zeros(len(domains), dtype=torch.float64)
        self.split_tokens = {s: self.stream_tokens.clone() for s in ('A', 'B')}
        self.tau, self.histograms, self.factqa = {}, {}, {}
        self.answer_position = None
        self.selected_counts = {}
        inventory = E.moe_layers(model)
        if layers is not None and (not layers or any(li < 0 or li >= len(inventory) for li in layers)):
            raise ValueError('screen layer outside model MoE inventory')
        total_experts = sum(len(moe.experts) for li, (_,moe) in enumerate(inventory) if layers is None or li in layers)
        try:
            for layer, (name, moe) in enumerate(inventory):
                if layers is not None and layer not in layers:
                    continue
                self.histograms[layer] = torch.zeros(2050, dtype=torch.int64)
                self.factqa[layer] = dict(routed=np.zeros(len(moe.experts), dtype=np.int64),
                                         tokens=0, answer_counts=[])
                grouped = callable(getattr(moe, '_forward_frozen_exact_grouped', None))
                self.dispatch[layer] = dict(interface='grouped dispatch' if grouped else 'fixture Linear argmax',
                    top_k=getattr(moe.router, 'top_k', getattr(moe, 'top_k', 1)),
                    scaling_factor=getattr(moe.router, 'scaling_factor', None),
                    selection_mode=getattr(moe.router, 'selection_mode', None))
                if grouped:
                    def capture(x, indices, weights, li=layer):
                        if self.active:
                            self._dispatch(li, indices, weights)
                        return None
                    self._override(moe, '_forward_frozen_exact_grouped', capture)
                    if hasattr(moe, '_forward_trainable_exact_grouped'):
                        self._override(moe, '_forward_trainable_exact_grouped', lambda *args: None)
                    if hasattr(moe, 'use_deepgemm'):
                        self._override(moe, 'use_deepgemm', False)
                elif type(moe.router) is torch.nn.Linear and not hasattr(moe, 'top_k'):
                    def fixture(mod, inp, out, li=layer):
                        if self.active:
                            ids = out.detach().argmax(-1).reshape(-1, 1)
                            self._dispatch(li, ids, torch.ones_like(ids, dtype=out.dtype))
                    self.hooks.append(moe.router.register_forward_hook(fixture))
                else:
                    raise ValueError('census requires grouped dispatch or the unweighted top-1 fixture')
                self.hooks.append(moe.register_forward_hook(lambda mod, inp, out, li=layer: self._finish(li)))
                for expert_id, expert in enumerate(moe.experts):
                    key = f'L{layer}_E{expert_id}'
                    h = expert.up_proj.weight.shape[0]
                    def accumulators():
                        return {field: torch.zeros(len(domains) if field == 'routed' else (len(domains), h),
                                                   dtype=torch.float64) for field in FIELDS}
                    self.units[key] = dict(layer=layer, expert=expert_id, name=f'{name}.experts.{expert_id}',
                        **accumulators(), down_norm=expert.down_proj.weight.detach().double().norm(dim=0).cpu())
                    self.split_units[key] = {s: accumulators() for s in ('A','B')}
                    self.hooks.append(expert.up_proj.register_forward_hook(
                        lambda mod, inp, out, key=key: self._collect(key, out)))
                    recovery.heartbeat('census_hooks', len(self.units), total_experts)
            if not self.units:
                raise ValueError('no routed MoE experts found')
        except Exception:
            self.close()
            raise

    def _override(self, module, name, value):
        self.methods.append((module, name, name in module.__dict__, getattr(module, name)))
        setattr(module, name, value)

    def _dispatch(self, layer, indices, weights):
        if layer in self.plans or indices.ndim != 2 or indices.shape != weights.shape:
            raise RuntimeError('unsupported or repeated census dispatch')
        ids = indices.detach()
        if self.phase == 'screen' and len(ids) != len(self.observer.labels):
            raise RuntimeError('screen span labels do not match dispatch token count')
        if ids.numel() and (ids.min() < 0 or ids.max() >= len(self.factqa[layer]['routed'])):
            raise RuntimeError('dispatch expert ID outside resident bank')
        self.dispatch[layer]['top_k'] = ids.shape[1]
        plans = {}
        for ei in ids.unique().tolist():
            ti, slot = (ids == ei).nonzero(as_tuple=True)
            tokens, inverse = ti.unique(sorted=True, return_inverse=True)
            w = weights.new_zeros(len(tokens)).scatter_add_(0, inverse, weights.detach()[ti, slot])
            plans[ei] = dict(weights=w, tokens=tokens, seen=False)
            if self.phase == 'factqa':
                self.factqa[layer]['routed'][ei] += len(tokens)
        self.plans[layer] = plans
        if self.phase == 'factqa':
            self.factqa[layer]['tokens'] += len(ids)
            selected = self.selected_counts.get(layer, {})
            experts = ids[self.answer_position].unique().tolist()
            self.factqa[layer]['answer_counts'].append(sum(selected.get(ei, 0) for ei in experts))

    def _finish(self, layer):
        if not self.active:
            return
        plans = self.plans.pop(layer, None)
        if plans is None or any(not p['seen'] for p in plans.values()):
            raise RuntimeError('expert backend bypassed census projection hooks')

    def _collect(self, key, out):
        if not self.active:
            return
        acc, d = self.units[key], self.domain
        plan = self.plans.get(acc['layer'], {}).get(acc['expert'])
        if plan is None or plan['seen']:
            raise RuntimeError('projection rows do not match dispatch')
        plan['seen'] = True
        n = len(plan['weights'])
        z = out.detach().reshape(-1, out.shape[-1])
        if len(z) < n:
            raise RuntimeError('projection rows shorter than dispatch')
        if self.phase == 'screen':
            self.observer.observe(key, {field: acc[field] for field in ('layer', 'expert', 'name')},
                                  z[:n], plan['tokens'])
            return
        if self.phase == 'factqa':
            return
        if self.phase == 'measure':
            for target in (acc, self.split_units[key][self.split]):
                target['routed'][d] += n
        for start in range(0, n, 4096):
            phi = z[start:min(start+4096, n)].double().clamp_min(0).square()
            if self.phase == 'calibrate':
                # Zero bin plus 2048 log2 bins over [2^-32,2^32], and overflow.
                bins = ((phi.clamp_min(2.**-32).log2()+32)*32).floor().long().clamp(0, 2048)+1
                bins[phi == 0] = 0
                self.histograms[acc['layer']] += torch.bincount(bins.flatten(), minlength=2050).cpu()
                continue
            w2 = plan['weights'][start:start+len(phi)].double().square()[:, None]
            tau = self.tau.get(acc['layer'])
            sums = dict(fired=(phi > 0).sum(0).cpu(), relu2_sum=phi.sum(0).cpu(),
                        relu4_sum=phi.square().sum(0).cpu(),
                        weighted_relu4_sum=(phi.square()*w2).sum(0).cpu(),
                        tail=(phi > tau).sum(0).cpu() if tau is not None else torch.zeros(phi.shape[1]))
            for target in (acc, self.split_units[key][self.split]):
                for field, value in sums.items():
                    target[field][d] += value

    def freeze_tau(self):
        for layer, hist in self.histograms.items():
            nonzero = hist[1:]
            count = int(nonzero.sum())
            if not count:
                raise ValueError('tau calibration did not observe nonzero phi in a layer')
            # Average the two middle order statistics for an even sample, using
            # their bin upper edges. Zeros never enter either order statistic.
            edges = []
            cumulative = nonzero.cumsum(0)
            for position in ((count-1)//2, count//2):
                index = 1 + int(torch.searchsorted(cumulative, position+1))
                if index == 2049:
                    raise ValueError('layer nonzero median exceeds tau histogram range')
                edges.append(2.**(-32+index/32))
            self.tau[layer] = sum(edges)/2
        self.phase = 'measure'

    def close(self):
        for hook in self.hooks:
            hook.remove()
        for module, name, own, value in reversed(self.methods):
            if own:
                setattr(module, name, value)
            else:
                delattr(module, name)
        self.hooks.clear()
        self.methods.clear()
        self.plans.clear()

    def state(self):
        return dict(policy=CENSUS_POLICY, units=self.units, split_units=self.split_units, stream_tokens=self.stream_tokens,
                    split_tokens=self.split_tokens, tau=self.tau, histograms=self.histograms)

    def restore(self, state):
        def copy(target, source):
            if isinstance(target, dict):
                if target.keys() != source.keys():
                    raise ValueError('checkpoint expert inventory or fields differ')
                for key in target:
                    if key == 'down_norm' and not torch.equal(target[key], source[key]):
                        raise ValueError('checkpoint down columns differ')
                    copy(target[key], source[key])
            elif isinstance(target, torch.Tensor):
                if target.shape != source.shape:
                    raise ValueError('checkpoint unit shape differs')
                target.copy_(source)
            elif target != source:
                raise ValueError('checkpoint unit identity differs')
        if 'units' not in state:
            raise ValueError('C51 checkpoints require split counts, token denominators and tau')
        if state.get('policy') != CENSUS_POLICY:
            raise ValueError('census checkpoint selection policy differs; rerun calibration')
        copy(self.units, state['units'])
        copy(self.split_units, state['split_units'])
        copy(self.stream_tokens, state['stream_tokens'])
        copy(self.split_tokens, state['split_tokens'])
        copy(self.histograms, state['histograms'])
        self.tau = dict(state['tau'])

    def derived(self, key, split=None):
        base = self.units[key]
        acc = base if split is None else self.split_units[key][split]
        tokens = self.stream_tokens if split is None else self.split_tokens[split]
        routed = acc['routed'][:, None]
        observed = routed >= self.min_routed
        def conditional(field):
            return torch.where(observed, acc[field]/routed.clamp_min(1), float('nan')).numpy()
        result = dict(frequency=conditional('fired'), mean_relu2=conditional('relu2_sum'),
                      mean_relu4=conditional('relu4_sum'), tail_frequency=conditional('tail'))
        if base['layer'] not in self.tau:
            result['tail_frequency'].fill(np.nan)
        norm2 = base['down_norm'].numpy()[None, :]**2
        result['energy'] = result['mean_relu4']*norm2
        result['legacy_energy'] = result['mean_relu2']*base['down_norm'].numpy()[None, :]
        result['unconditional_energy'] = np.where(observed.numpy() & (tokens.numpy()[:,None] > 0),
            acc['weighted_relu4_sum'].numpy()/tokens.clamp_min(1).numpy()[:,None]*norm2, np.nan)
        for field in ('fired', 'relu2_sum', 'relu4_sum', 'tail'):
            name = {'fired':'unconditional_frequency', 'tail':'unconditional_tail_frequency',
                    'relu2_sum':'unconditional_mean_relu2', 'relu4_sum':'unconditional_mean_relu4'}[field]
            result[name] = np.where(observed.numpy() & (tokens.numpy()[:,None] > 0),
                acc[field].numpy()/tokens.clamp_min(1).numpy()[:,None], np.nan)
        if base['layer'] not in self.tau:
            result['unconditional_tail_frequency'].fill(np.nan)
        result['observed'] = observed.numpy().reshape(-1)
        # Overall ranks require evidence in EVERY natural-text domain.
        known = result['observed'].all() and tokens.sum() > 0
        result['overall_unconditional_energy'] = (acc['weighted_relu4_sum'].sum(0).numpy()/max(float(tokens.sum()),1)
                                                  *norm2[0] if known else np.full(norm2.shape[1], np.nan))
        total = acc['routed'].sum().clamp_min(1)
        for field, name in (('fired','frequency'), ('relu2_sum','mean_relu2'), ('relu4_sum','mean_relu4')):
            result['overall_'+name] = (acc[field].sum(0)/total).numpy() if known else np.full(norm2.shape[1], np.nan)
        result['overall_energy'] = result['overall_mean_relu4']*norm2[0]
        return result

    def expert_ranks(self, split='A'):
        """Per-expert ranks using indices into the layer's full expert/unit map."""
        result = {}
        for layer in self.histograms:
            result[layer], offset = {}, 0
            for key, unit in self.units.items():
                if unit['layer'] != layer:
                    continue
                values = self.derived(key,split)['overall_unconditional_energy']
                valid = np.flatnonzero(np.isfinite(values))
                result[layer][unit['expert']] = offset + valid[np.argsort(values[valid], kind='stable')]
                offset += len(values)
        return result

    def ranks(self, split='A'):
        # Retain the flat artifact as expert-ordered concatenated rank blocks.
        # Fractional selection must use expert_bottom, not a prefix of this array.
        return {layer: np.concatenate(list(experts.values()))
                for layer, experts in self.expert_ranks(split).items()}

    def output(self, path, metadata, plots=True):
        path = Path(path).with_suffix('.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, layers = {}, {}
        ranks_a, ranks_b = self.ranks('A'), self.ranks('B')
        experts_a, experts_b = self.expert_ranks('A'), self.expert_ranks('B')
        summary = dict(metadata, domains=self.domains, layers={}, min_routed=self.min_routed,
            stream_tokens=self.stream_tokens.tolist(), split_tokens={s:t.tolist() for s,t in self.split_tokens.items()},
            split_fraction=self.split_fraction, dispatch={str(li):v for li,v in self.dispatch.items()},
            ranking='per expert stream A unconditional router-weight-squared output energy; all domains observed',
            rank_layout='flat layer ranks concatenate per-expert blocks; select fractions within each expert',
            threshold_policy='strictly below domain expert p1 on A union B; low-count cells unobserved',
            tau={str(li):v for li,v in self.tau.items()},
            tau_estimator='layer median of nonzero routed phi=relu(z)^2; middle-order-statistic log2 bin upper edges (averaged if even)')
        for key, acc in self.units.items():
            layers.setdefault(acc['layer'], []).append(key)
            for field in (*FIELDS, 'down_norm'):
                arrays[f'{key}_{field}'] = acc[field].numpy()
            for split in (None, 'A', 'B'):
                suffix = '' if split is None else '_'+split
                for field, value in self.derived(key,split).items():
                    arrays[f'{key}_{field}{suffix}'] = value
                if split is not None:
                    for field in FIELDS:
                        arrays[f'{key}_{field}{suffix}'] = self.split_units[key][split][field].numpy()
        for layer, keys in layers.items():
            energy = np.concatenate([arrays[f'{k}_unconditional_energy'] for k in keys], axis=1)
            observed = np.stack([arrays[f'{k}_observed'] for k in keys], axis=1)
            valid = np.isfinite(energy)
            thresholds = np.array([[np.percentile(row[np.isfinite(row)], 1) if np.isfinite(row).any()
                                    else np.nan for row in arrays[f'{k}_unconditional_energy']] for k in keys]).T
            unit_thresholds = np.concatenate([np.repeat(thresholds[:,i:i+1], len(self.units[k]['down_norm']), axis=1)
                                             for i,k in enumerate(keys)], axis=1)
            quiet = valid & (energy < unit_thresholds)
            known = valid.all(axis=0)
            near = quiet.all(axis=0) & known
            specific = known & (quiet.sum(axis=0) >= 3) & ((valid & ~quiet).sum(axis=0) >= 1)
            prefix = f'layer{layer}'
            arrays[f'{prefix}_rank'] = ranks_a[layer]
            arrays[f'{prefix}_rank_B'] = ranks_b[layer]
            arrays[f'{prefix}_expert'] = np.concatenate([np.full(len(self.units[k]['down_norm']),self.units[k]['expert']) for k in keys])
            arrays[f'{prefix}_unit'] = np.concatenate([np.arange(len(self.units[k]['down_norm'])) for k in keys])
            arrays[f'{prefix}_near_dormant'], arrays[f'{prefix}_domain_specific'] = near, specific
            stability = {}
            for fraction in (.1,.2):
                a, b = expert_bottom(experts_a[layer],fraction), expert_bottom(experts_b[layer],fraction)
                stability[str(int(fraction*100))] = dict(a=len(a), b=len(b), intersection=len(a&b),
                    jaccard=len(a&b)/len(a|b) if a|b else None)
            per_expert = {}
            for ei, rank_a in experts_a[layer].items():
                rank_b = experts_b[layer][ei]
                arrays[f'{prefix}_E{ei}_rank'] = rank_a
                arrays[f'{prefix}_E{ei}_rank_B'] = rank_b
                per_expert[str(ei)] = dict(ranked_A=len(rank_a), ranked_B=len(rank_b), rank_stability={})
                for fraction in (.1,.2):
                    a, b = bottom(rank_a,fraction), bottom(rank_b,fraction)
                    per_expert[str(ei)]['rank_stability'][str(int(fraction*100))] = dict(
                        a=len(a), b=len(b), intersection=len(a&b), jaccard=len(a&b)/len(a|b) if a|b else None)
            summary['layers'][str(layer)] = dict(units=energy.shape[1], ranked_A=len(ranks_a[layer]),
                ranked_B=len(ranks_b[layer]), near_dormant_everywhere=int(near.sum()),
                domain_specific=int(specific.sum()), unknown_any_domain=int((~known).sum()),
                unobserved_cells=int((~observed).sum()),
                unobserved_cells_A=sum(int((~arrays[f'{k}_observed_A']).sum()) for k in keys),
                unobserved_cells_B=sum(int((~arrays[f'{k}_observed_B']).sum()) for k in keys),
                rank_stability=stability, per_expert=per_expert,
                energy=distribution(np.concatenate([arrays[f'{k}_overall_unconditional_energy'] for k in keys])),
                frequency=distribution(np.concatenate([arrays[f'{k}_overall_frequency'] for k in keys])),
                mean_relu2=distribution(np.concatenate([arrays[f'{k}_overall_mean_relu2'] for k in keys])),
                per_domain={d:dict(energy=distribution(energy[i]),
                    conditional_energy=distribution(np.concatenate([arrays[f'{k}_energy'][i] for k in keys])),
                    frequency=distribution(np.concatenate([arrays[f'{k}_frequency'][i] for k in keys])),
                    mean_relu2=distribution(np.concatenate([arrays[f'{k}_mean_relu2'][i] for k in keys])),
                    tail_frequency=distribution(np.concatenate([arrays[f'{k}_tail_frequency'][i] for k in keys])),
                    percentile_1_per_expert={str(self.units[k]['expert']):float(thresholds[i,j])
                        if np.isfinite(thresholds[i,j]) else None for j,k in enumerate(keys)},
                    unobserved_cells=int((~observed[i]).sum()),
                    unobserved_cells_A=sum(int(not arrays[f'{k}_observed_A'][i]) for k in keys),
                    unobserved_cells_B=sum(int(not arrays[f'{k}_observed_B'][i]) for k in keys),
                    routed_tokens=[float(self.units[k]['routed'][i]) for k in keys],
                    routed_tokens_A=[float(self.split_units[k]['A']['routed'][i]) for k in keys],
                    routed_tokens_B=[float(self.split_units[k]['B']['routed'][i]) for k in keys]) for i,d in enumerate(self.domains)})
            if plots:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                for i, domain in enumerate(self.domains):
                    finite = energy[i][np.isfinite(energy[i])]
                    if finite.size:
                        ax.hist(np.log10(np.maximum(finite, 1e-30)), bins=60, histtype='step', label=domain)
                ax.set(xlabel='log10 stream-token output energy (zeros at -30)', ylabel='units', title=f'MoE layer {layer}')
                if ax.lines or ax.patches:
                    ax.legend()
                fig.savefig(path.with_name(f'{path.stem}_layer{layer}.png'))
                plt.close(fig)
            recovery.heartbeat('census_output', len(summary['layers']), len(layers))
        arrays['metadata_json'] = np.array(json.dumps(summary))
        np.savez_compressed(path, **arrays)
        path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary


def budgets(tokens, context, domains):
    windows = tokens // context
    if windows < domains:
        raise ValueError('--tokens must allow at least one full context per domain')
    q, r = divmod(windows, domains)
    return [(q + (i < r))*context for i in range(domains)]


def split_boundary(target, context, fraction):
    return (target//context - int(target//context*fraction))*context


@torch.no_grad()
def run(model, streams, context, targets, batch_size, device, census, cursors=None, checkpoint=None,
        ckpt_tokens=0, stop_after=None, force_split=None):
    cursors = dict(cursors or {name: 0 for name in streams})
    if context < 1 or batch_size < 1 or cursors.keys() != streams.keys() or len(targets) != len(streams) or any(
        type(cursors[d]) is not int or not 0 <= cursors[d] <= targets[i] or cursors[d] % context or
        targets[i] % context or targets[i] > len(streams[d]) for i,d in enumerate(streams)):
        raise ValueError('invalid census stream cursors/targets')
    done, last = sum(cursors.values()), sum(cursors.values())
    while any(cursors[d] < targets[i] for i,d in enumerate(streams)):
        for i, (domain, tokens) in enumerate(streams.items()):
            start = cursors[domain]
            boundary = split_boundary(targets[i],context,census.split_fraction)
            end = boundary if not force_split and start < boundary else targets[i]
            count = min(batch_size*context, end-start)
            if not count:
                continue
            census.domain, census.split = i, force_split or ('A' if start < boundary else 'B')
            x = torch.tensor(np.array(tokens[start:start+count], dtype=np.int64).reshape(-1, context), device=device)
            E.logits_of(model, x)
            if census.phase == 'measure':
                census.stream_tokens[i] += count
                census.split_tokens[census.split][i] += count
            cursors[domain] += count
            done += count
            recovery.heartbeat('census_'+census.phase, done, sum(targets))
            if checkpoint and ckpt_tokens and done-last >= ckpt_tokens:
                checkpoint(cursors)
                last = done
            if stop_after is not None and done >= stop_after:
                return cursors
    if checkpoint:
        checkpoint(cursors)
    return cursors


def subsample_targets(targets, context, cap):
    """Balanced domain prefixes, bounded by each available domain budget."""
    available = [n//context for n in targets]
    out = [0]*len(targets)
    remaining = min(cap//context, sum(available))
    while remaining:
        for i,n in enumerate(available):
            if out[i] < n and remaining:
                out[i] += 1
                remaining -= 1
    return [n*context for n in out]


def calibrate(model, streams, context, targets, batch_size, device, census):
    selection_targets = [split_boundary(n,context,census.split_fraction) for n in targets]
    sizes = subsample_targets(selection_targets,context,TAU_TOKENS)
    census.phase = 'calibrate'
    run(model,streams,context,sizes,batch_size,device,census,force_split='A')
    census.freeze_tau()
    return sizes


def batch_agreement(model, streams, context, targets, batch_size, device, census):
    sizes = subsample_targets(targets,context,AGREEMENT_TOKENS)
    ranks = []
    census.active = False
    try:
        for bs in (1,batch_size):
            probe = Census(model,streams,census.min_routed,census.split_fraction)
            try:
                probe.tau = dict(census.tau)
                run(model,streams,context,sizes,bs,device,probe,force_split='A')
                ranks.append(probe.expert_ranks())
            finally:
                probe.close()
    finally:
        census.active = True
    layers = {}
    for layer in ranks[0]:
        experts = {str(ei):membership(bottom(rank,.2), bottom(ranks[1][layer][ei],.2))
                   for ei,rank in ranks[0][layer].items()}
        a, b = expert_bottom(ranks[0][layer],.2), expert_bottom(ranks[1][layer],.2)
        layers[str(layer)] = dict(membership(a,b), per_expert=experts)
        layers[str(layer)]['passed'] = agreement_pass(list(experts.values()))
    passed = agreement_pass(list(layers.values()))
    return dict(requested_tokens=AGREEMENT_TOKENS, processed_tokens=sum(sizes), targets=sizes,
                batch_sizes=[1,batch_size], bottom_fraction=.2, selection='per expert',
                pass_threshold=.95, layers=layers, passed=passed)


def factqa_items(tok, bos, seed=510051, people=4, facts_per_person=40):
    # Reserve a census-only entity namespace as well as a disjoint dev seed.
    facts = [f._replace(name=f'CensusDev{seed} Person{f.pid:05d}') for f in
             E.make_world(seed,synthetic_people=people,per_person=facts_per_person)]
    return facts, E.build_items(tok,facts,E.TRAIN_AUG_T,bos)


@torch.no_grad()
def factqa_probe(model, items, device, census):
    ranks = census.ranks('A')
    expert_ranks = census.expert_ranks('A')
    for acc in census.factqa.values():
        acc['routed'].fill(0)
        acc['tokens'], acc['answer_counts'] = 0, []
    for layer, rank in ranks.items():
        experts = np.concatenate([np.full(len(u['down_norm']),u['expert']) for u in census.units.values() if u['layer'] == layer])
        chosen = experts[list(sorted(expert_bottom(expert_ranks[layer],.1)))]
        census.selected_counts[layer] = {int(e):int((chosen == e).sum()) for e in np.unique(chosen)}
    census.phase = 'factqa'
    try:
        for i,item in enumerate(items):
            # Teacher forcing retains the first answer-predicting position;
            # no padding or cross-prompt recurrent context is introduced.
            seq = (item['p']+item['a'])[:-1]
            census.answer_position = len(item['p'])-1
            E.logits_of(model,torch.tensor([seq],device=device))
            recovery.heartbeat('census_factqa',i+1,len(items))
    finally:
        census.phase, census.answer_position = 'measure', None
    result = {}
    for layer,acc in census.factqa.items():
        routed = acc['routed']
        result[str(layer)] = dict(tokens=acc['tokens'], routed_tokens=routed.tolist(),
            ranked_units=len(ranks[layer]), selected_bottom10_units=sum(census.selected_counts[layer].values()),
            routing_share=(routed/max(int(routed.sum()),1)).tolist(),
            routing_probability=(routed/max(acc['tokens'],1)).tolist(),
            bottom10_dispatched_at_answer=acc['answer_counts'],
            answer_distribution=distribution(acc['answer_counts']))
    return dict(domain='factqa', prompts=len(items), batch_size=1,
                answer_position='last prompt token predicting first answer token', layers=result)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    eda.add_argument(p)
    p.add_argument('--export', type=Path, default=E.EXPORT)
    p.add_argument('--export-step', type=int, default=E.EXPORT_STEP)
    p.add_argument('--stream', action='append', required=True, help='domain=uint32-file or domain=tokens.json')
    p.add_argument('--tokens', type=int, default=50_000_000)
    p.add_argument('--context-length', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--min-routed', type=int, default=10000)
    p.add_argument('--split-fraction', type=float, default=.3, help='fraction of each domain assigned to confirmation B (default A/B = 70/30)')
    p.add_argument('--tokenizer', type=Path, help='default: data/tokenizer.json')
    p.add_argument('--factqa-seed', type=int, default=510051)
    p.add_argument('--factqa-people', type=int, default=4)
    p.add_argument('--factqa-facts', type=int, default=40)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--eval-panels', type=Path, nargs='+', help='default: data/eval-*.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--ckpt-tokens', type=int, default=1_000_000)
    p.add_argument('--resume', choices=['auto'])
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--firing-screen', action='store_true', help='S1 entity-span firing screen instead of SC0 census')
    p.add_argument('--screen-layers', default='12,14,17,20', help='indices into the MoE inventory')
    p.add_argument('--factqa-world', choices=['dev', 'arm'], default='dev', help='C51 dev namespace or ordinary small-arm world')
    p.add_argument('--factqa-relations', action='store_true', help='include linked sister relations')
    return p


@torch.no_grad()
def firing_main(a):
    try:
        from . import c54 as S
    except ImportError:
        import c54 as S
    if a.factqa_people < 2:
        raise ValueError('S1 entity-v-other comparison requires at least two entities')
    if a.resume or a.checkpoint:
        raise ValueError('S1 is a separate forward-only pass; resume/checkpoint flags apply to SC0 only')
    layers = [int(li) for li in a.screen_layers.split(',')]
    if len(set(layers)) != len(layers):
        raise ValueError('duplicate screen layers')
    if not any(pair.partition('=')[0] == 'fwhq_sel' for pair in a.stream):
        raise ValueError('S1 requires stream fwhq_sel for the FineWeb comparison')
    streams, paths = {}, []
    for pair in a.stream:
        name, sep, path = pair.partition('=')
        if not sep or not name or name in streams:
            raise ValueError('streams must have unique domain=path names')
        streams[name] = read_tokens(path)
        paths.append(Path(path))
    if 'fwhq_sel' not in streams:
        raise ValueError('S1 requires stream fwhq_sel for the FineWeb comparison')
    targets = budgets(a.tokens, a.context_length, len(streams))
    if any(n > len(t) for n,t in zip(targets, streams.values())):
        raise ValueError('screen stream shorter than its token budget')
    panels = a.eval_panels or sorted(E.DATA.glob('eval-*.json'))
    refs = {'anchor': read_tokens(a.anchor)}
    refs.update({str(p): read_tokens(p) for p in panels})
    guard = OverlapGuard(refs)
    overlap = {d: guard.check([t[:n]], d) for (d,t),n in zip(streams.items(), targets)}
    tokenizer_path = a.tokenizer or E.DATA/'tokenizer.json'
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    if a.factqa_world == 'dev':
        raw = E.make_world(a.factqa_seed, synthetic_people=a.factqa_people, per_person=a.factqa_facts,
                           relations=a.factqa_relations)
        rename = {f.name: f'CensusDev{a.factqa_seed} Person{f.pid:05d}' for f in raw}
        facts = [f._replace(name=rename[f.name], val=rename.get(f.val, f.val)) for f in raw]
    else:
        facts = E.make_world(a.factqa_seed, n_people=a.factqa_people, per_person=a.factqa_facts,
                             relations=a.factqa_relations)
    items = S.labelled_items(tok, E.BOS, facts)
    overlap['factqa'] = guard.check([np.asarray((it['p']+it['a'])[:-1], dtype='<u4') for it in items], 'factqa')
    inputs = recovery.input_fingerprints(SimpleNamespace(export=a.export, anchor=a.anchor,
                extra_eval_sets=','.join(map(str, [*panels, *paths, tokenizer_path]))))
    sys.path.insert(0, '/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    eda.announce(a)
    model, _, dev = eda.build_model(build_model, a, a.export, device='cuda:0', model_config=None,
                                  step=a.export_step, expert_format='bf16')
    model.eval()
    screen = S.FiringScreen({f.pid: f.name for f in facts})
    census = Census(model, ['fwhq_sel'], layers=layers, observer=screen)
    census.phase = 'screen'
    try:
        # Only FineWeb contributes negative tokens. Other supplied streams are
        # overlap-audited; ordinary census measurements remain a separate run.
        tokens = streams['fwhq_sel']
        target = targets[list(streams).index('fwhq_sel')]
        for start in range(0, target, a.batch_size*a.context_length):
            count = min(a.batch_size*a.context_length, target-start)
            screen.begin(np.full(count, -1))
            x = torch.tensor(np.array(tokens[start:start+count], dtype=np.int64).reshape(-1, a.context_length), device=dev)
            E.logits_of(model, x)
            recovery.heartbeat('S1_fineweb', start+count, target)
        for i, item in enumerate(items):
            screen.begin(item['span_labels'])
            E.logits_of(model, torch.tensor([(item['p']+item['a'])[:-1]], device=dev))
            recovery.heartbeat('S1_factqa', i+1, len(items))
        # Include unrouted experts as all-zero units, without inventing firing.
        for key, acc in census.units.items():
            screen.units.setdefault(key, dict(layer=acc['layer'], expert=acc['expert'], name=acc['name'],
                width=len(acc['down_norm']), hist={}, routed={}, fired={}, relu2_sum={}))
        result = screen.output(a.out, dict(eda_backend=eda.selected(a), export=str(a.export), export_step=a.export_step,
            seed=a.factqa_seed, world=a.factqa_world, relation_world=a.factqa_relations,
            facts=[f._asdict() for f in facts], templates=E.TRAIN_AUG_T,
            pronoun_templates=[E.ADDRESS_PRONOUN_T, E.ADDRESS_PRONOUN_RECENT_T],
            fineweb_tokens=target, factqa_tokens=sum(len(it['span_labels']) for it in items),
            prompts=len(items), layers=layers, dispatch=census.dispatch, overlap=overlap,
            input_sha256=inputs, context_length=a.context_length, batch_size=a.batch_size), items, a.factqa_seed)
        print('S1 ' + json.dumps(result), flush=True)
        return result
    finally:
        census.close()


def main(argv=None):
    a = build_parser().parse_args(argv)
    if (a.context_length < 32 or a.batch_size < 1 or a.tokens < 1 or a.ckpt_tokens < 0 or a.min_routed < 1
        or not math.isfinite(a.split_fraction) or not 0 < a.split_fraction < 1 or a.factqa_seed < 100000
        or a.factqa_people < 1 or not 1 <= a.factqa_facts <= len(E.ATTR_POOL)):
        raise ValueError('invalid census counts, split-fraction or reserved dev-world seed')
    if a.firing_screen:
        return firing_main(a)
    streams, paths = {}, []
    for pair in a.stream:
        name, sep, path = pair.partition('=')
        if not sep or not name or name in streams or name == 'factqa':
            raise ValueError('--stream needs unique nonempty name=path pairs; factqa is reserved')
        streams[name] = read_tokens(path)
        paths.append(Path(path))
    targets = budgets(a.tokens, a.context_length, len(streams))
    for (name, tokens), target in zip(streams.items(), targets):
        if len(tokens) < target:
            raise ValueError(f'{name} too short: {len(tokens)} < {target}')
    panels = a.eval_panels if a.eval_panels is not None else sorted(E.DATA.glob('eval-*.json'))
    if not panels:
        raise ValueError('no eval panels found; supply --eval-panels')
    refs = {str(p.resolve()): read_tokens(p) for p in [*panels, a.anchor]}
    guard = OverlapGuard(refs)
    overlap = {d: guard.check([t[:n]], d) for (d,t),n in zip(streams.items(), targets)}
    tokenizer_path = a.tokenizer or E.DATA/'tokenizer.json'
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_path))
    facts, items = factqa_items(tok,E.BOS,a.factqa_seed,a.factqa_people,a.factqa_facts)
    overlap['factqa'] = guard.check([np.asarray((it['p']+it['a'])[:-1],dtype='<u4') for it in items],'factqa')
    identity = dict(domains=list(streams),
                    streams={name: str(path.resolve()) for name,path in zip(streams,paths)},
                    requested_tokens=a.tokens, targets=targets, context=a.context_length, batch=a.batch_size,
                    min_routed=a.min_routed, split_fraction=a.split_fraction,
                    tau_tokens=TAU_TOKENS, agreement_tokens=AGREEMENT_TOKENS,
                    selection_policy=CENSUS_POLICY,
                    tokenizer=str(tokenizer_path.resolve()), factqa_seed=a.factqa_seed,
                    factqa_people=a.factqa_people, factqa_facts=a.factqa_facts,
                    factqa_items=items, factqa_templates=E.TRAIN_AUG_T,
                    eda_backend=eda.selected(a), export_step=a.export_step)
    inputs = recovery.input_fingerprints(SimpleNamespace(export=a.export, anchor=a.anchor,
                extra_eval_sets=','.join(map(str, [*panels, *paths, tokenizer_path]))))
    checkpoint = a.checkpoint or a.out.with_suffix('.resume.pt')
    saved = None
    if a.resume:
        if not checkpoint.exists():
            raise ValueError(f'resume checkpoint missing: {checkpoint}')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved.get('version') != 3 or saved.get('identity') != identity or saved.get('inputs') != inputs:
            raise ValueError('resume configuration or input contents changed')
    sys.path.insert(0, '/workspace/malleable/mlf')
    from ops.eval.bench_saves import build_model
    eda.announce(a)
    model, _, dev = eda.build_model(build_model, a, a.export, device='cuda:0', model_config=None,
                                  step=a.export_step, expert_format='bf16')
    model.eval()
    census = Census(model, streams, a.min_routed, a.split_fraction)
    calibration_targets = subsample_targets(
        [split_boundary(n,a.context_length,a.split_fraction) for n in targets],a.context_length,TAU_TOKENS)
    def save(cursors):
        recovery.atomic_save(checkpoint, dict(version=3, identity=identity, inputs=inputs,
            cursors=dict(cursors), census=census.state(), rng=recovery.rng_state()))
    try:
        if saved:
            census.restore(saved['census'])
            recovery.restore_rng(saved['rng'])
            print('RESUMED ' + json.dumps(saved['cursors']), flush=True)
        else:
            calibrate(model,streams,a.context_length,targets,a.batch_size,dev,census)
        cursors = run(model, streams, a.context_length, targets, a.batch_size, dev, census,
                      saved['cursors'] if saved else None, save if a.ckpt_tokens or a.resume else None, a.ckpt_tokens)
        agreement = batch_agreement(model,streams,a.context_length,targets,a.batch_size,dev,census)
        factqa = factqa_probe(model,items,dev,census)
        factqa.update(seed=a.factqa_seed, entities=sorted({f.name for f in facts}),
                      templates=E.TRAIN_AUG_T, facts=len(facts))
        summary = census.output(a.out, dict(eda_backend=eda.selected(a), export=str(a.export), export_step=a.export_step,
            requested_tokens=a.tokens, processed_tokens=sum(cursors.values()), cursors=cursors,
            context_length=a.context_length, batch_size=a.batch_size, overlap=overlap, input_sha256=inputs,
            tau_calibration=dict(requested_tokens=TAU_TOKENS,processed_tokens=sum(calibration_targets),targets=calibration_targets),
            batch_agreement=agreement, factqa=factqa), not a.no_plots)
        print('CENSUS ' + json.dumps(summary), flush=True)
        return summary
    finally:
        census.close()


if __name__ == '__main__':
    main()
