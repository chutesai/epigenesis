"""C42 opt-in selector study. CPU analysis requires only torch and cached features.

Keys use TRAIN; likelihood maps use CAL; cutoffs and hierarchy choice use VAL.
TEST is consumed only after every choice is locked. No language-model training.
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from . import eda_backend as eda
    from .c44_runtime import heartbeat, progress
except ImportError:
    import eda_backend as eda
    from c44_runtime import heartbeat, progress

FORMAT = 'c42-selector-features-v1'
TARGET = 1e-3


def matrix(value):
    x = torch.as_tensor(value).detach().double().cpu()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError('features must be finite matrices')
    return x


class Centroid:
    """Same centered cosine score as branch_eval.BankIndexer, CPU-only import."""
    def __init__(self, keys, center, owners):
        self.keys, self.center, self.owners = keys, center, tuple(owners)

    @classmethod
    def fit(cls, x, labels, general):
        x, general = matrix(x), matrix(general)
        labels = torch.as_tensor(labels)
        owners = sorted(set(labels.tolist()))
        if not len(x) or not len(general) or labels.shape != (len(x),) or min(owners) < 0:
            raise ValueError('centroid fit needs positive TRAIN and general anchor')
        center = general.mean(0)
        keys = torch.stack([x[labels == b].mean(0)-center
                            for b in progress(owners, 'selector_centroid_fit')])
        if (keys.norm(dim=1) < 1e-12).any():
            raise ValueError('bank key is indistinguishable from general mean')
        return cls(F.normalize(keys, dim=1), center, owners)

    def scores(self, x):
        return F.normalize(matrix(x)-self.center, dim=1) @ self.keys.T


class SiblingKeys:
    """Discriminative multinomial keys, fit only on this parent's siblings."""
    @classmethod
    def fit(cls, x, labels, l2=.01):
        x = matrix(x)
        owners = tuple(sorted(set(torch.as_tensor(labels).tolist())))
        y = torch.tensor([owners.index(int(b)) for b in labels])
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
        z = (x-mean)/scale
        weight = torch.zeros(len(owners), x.shape[1], dtype=torch.double, requires_grad=True)
        bias = torch.zeros(len(owners), dtype=torch.double, requires_grad=True)
        opt = torch.optim.LBFGS([weight, bias], max_iter=60, line_search_fn='strong_wolfe')
        evaluations = 0
        def closure():
            nonlocal evaluations
            evaluations += 1
            heartbeat('selector_sibling_fit', evaluations, None)
            opt.zero_grad()
            loss = F.cross_entropy(F.linear(z, weight, bias), y) + l2*weight.square().mean()
            loss.backward()
            return loss
        opt.step(closure)
        obj = cls()
        obj.owners, obj.mean, obj.scale = owners, mean, scale
        obj.weight, obj.bias = weight.detach(), bias.detach()
        return obj

    def scores(self, x):
        return F.linear((matrix(x)-self.mean)/self.scale, self.weight, self.bias)


def logistic_llr(raw, positive, l2=.01):
    """Balanced binary logistic fit => log density ratio, without bank prior.

    Standardize for conditioning, then export coefficients in raw-score units.
    Newton solves at most 3x3 systems (positive and optional inhibitory score).
    """
    raw = matrix(raw)
    positive = torch.as_tensor(positive, dtype=torch.bool)
    if not positive.any() or positive.all() or len(positive) != len(raw) or not math.isfinite(l2) or l2 <= 0:
        raise ValueError('LLR calibration needs positives, negatives and positive L2')
    weights = torch.where(positive, .5/positive.sum(), .5/(~positive).sum()).double()
    weights /= weights.sum()
    mean = (weights[:, None]*raw).sum(0)
    scale = (weights[:, None]*(raw-mean).square()).sum(0).sqrt().clamp_min(1e-6)
    z = torch.cat(((raw-mean)/scale, torch.ones(len(raw), 1)), 1)
    y = positive.double()
    coef = torch.zeros(z.shape[1], dtype=torch.double)
    penalty = torch.eye(len(coef), dtype=torch.double)*l2
    penalty[-1, -1] = 1e-8
    def loss(c):
        logits = z @ c
        return (weights*(F.softplus(logits)-y*logits)).sum() + .5*c @ penalty @ c
    for _ in progress(range(60), 'selector_logistic_fit'):
        p = (z @ coef).sigmoid()
        grad = z.T @ (weights*(p-y)) + penalty @ coef
        hess = z.T @ ((weights*p*(1-p))[:, None]*z) + penalty
        step = torch.linalg.solve(hess, grad)
        rate, old = 1., loss(coef)
        while rate > 1e-8 and loss(coef-rate*step) > old:
            heartbeat('selector_line_search', None, None)
            rate *= .5
        coef -= rate*step
        if (rate*step).abs().max() < 1e-8:
            break
    slope = coef[:-1]/scale
    return torch.cat((slope, (coef[-1]-slope @ mean).view(1)))


class LLRIndexer:
    def __init__(self, scorer, coefficients):
        self.scorer, self.coefficients, self.owners = scorer, coefficients, scorer.owners

    @classmethod
    def fit(cls, scorer, x, labels, inhibitory=None, l2=.01):
        scores = scorer.scores(x)
        labels = torch.as_tensor(labels)
        if inhibitory is not None:
            inhibitory = matrix(inhibitory)
            if inhibitory.shape != scores.shape:
                raise ValueError('inhibitory scores must align with bank scores')
        coefficients = {}
        for i, bank in enumerate(progress(scorer.owners, 'selector_llr_banks')):
            raw = scores[:, i:i+1]
            if inhibitory is not None:
                raw = torch.cat((raw, inhibitory[:, i:i+1]), 1)
            coefficients[bank] = logistic_llr(raw, labels == bank, l2)
        return cls(scorer, coefficients)

    def scores(self, x, inhibitory=None):
        raw = self.scorer.scores(x)
        if inhibitory is not None and matrix(inhibitory).shape != raw.shape:
            raise ValueError('inhibitory scores must align with bank scores')
        values = []
        for i, b in enumerate(self.owners):
            c = self.coefficients[b]
            if len(c) == 3 and inhibitory is None:
                raise ValueError('calibrated inhibitory scores required at selection')
            score = c[0]*raw[:, i]+c[-1]
            if len(c) == 3:
                score += c[1]*matrix(inhibitory)[:, i]
            values.append(score)
        return torch.stack(values, 1)

    def select(self, x, delta=0., question_types=None, registered_multi=(), inhibitory=None):
        return gate(self.scores(x, inhibitory), self.owners, delta, question_types, registered_multi)

    def ledger(self):
        return {str(b): dict(coefficients=c.tolist(), score_dimensions=len(c)-1,
                             fit='CAL balanced logistic; equal priors; gate only')
                for b, c in self.coefficients.items()}

    def revoke(self, bank):
        i = self.owners.index(bank)
        self.coefficients.pop(bank)
        keep = [j for j in range(len(self.owners)) if j != i]
        self.scorer.owners = tuple(self.owners[j] for j in keep)
        if isinstance(self.scorer, Centroid):
            self.scorer.keys = self.scorer.keys[keep]
        else:
            self.scorer.weight, self.scorer.bias = self.scorer.weight[keep], self.scorer.bias[keep]
        self.owners = self.scorer.owners


def winners(scores):
    scores = matrix(scores)
    if not scores.shape[1]:
        raise ValueError('no resident banks')
    order = scores.argsort(dim=1, descending=True, stable=True)
    best = scores.gather(1, order[:, :1]).flatten()
    runner = scores.gather(1, order[:, 1:2]).flatten() if scores.shape[1] > 1 else torch.zeros(len(scores))
    return order, best, best-runner


def gate(scores, owners, delta=0., question_types=None, registered_multi=()):
    if not math.isfinite(delta) or delta < 0:
        raise ValueError('margin must be finite and nonnegative')
    order, best, margin = winners(scores)
    if question_types is not None and len(question_types) != len(scores):
        raise ValueError('question types must align')
    result = []
    for row, ranked in enumerate(order):
        if best[row] <= 0 or margin[row] <= delta:
            result.append(())
        elif question_types is not None and question_types[row] in registered_multi:
            result.append(tuple(sorted(owners[i] for i in ranked.tolist() if scores[row, i] > 0)))
        else:
            result.append((owners[int(ranked[0])],))
    return result


def cutoff(null_scores, target=TARGET, floor=None):
    """Smallest strict cutoff meeting the empirical VAL false-fire cap."""
    s = torch.as_tensor(null_scores).double().flatten()
    if not len(s) or not torch.isfinite(s).all() or not 0 <= target < 1:
        raise ValueError('finite, nonempty VAL nulls and valid target required')
    allowed = math.floor(target*len(s))
    threshold = float(s.sort(descending=True).values[allowed])
    return max(threshold, floor) if floor is not None else threshold


def margin_cutoff(scores, target=TARGET):
    _, best, margin = winners(scores)
    # Keep the full denominator: nonpositive evidence already abstains.
    return cutoff(torch.where(best > 0, margin, torch.zeros_like(margin)), target, 0.)


class LLRHierarchy:
    def __init__(self, parent, children, flat, bank_parent, beam=2, rescue=False, parent_gate=False):
        if not 1 <= beam <= len(parent.owners):
            raise ValueError('parent beam outside parent count')
        self.parent, self.children, self.flat = parent, children, flat
        self.bank_parent, self.beam, self.rescue = bank_parent, beam, rescue
        self.parent_gate = parent_gate
        self.owners = flat.owners

    @classmethod
    def fit(cls, train1, train2, labels, general, cal1, cal2, cal_labels,
            bank_parent, flat, beam=2, rescue=False, parent_gate=False):
        labels, cal_labels = torch.as_tensor(labels), torch.as_tensor(cal_labels)
        parents = torch.tensor([bank_parent[int(b)] for b in labels])
        cp = torch.tensor([bank_parent[int(b)] if b >= 0 else -1 for b in cal_labels])
        parent = LLRIndexer.fit(Centroid.fit(train1, parents, general), cal1, cp)
        children = {}
        for p in progress(parent.owners, 'selector_parent_fit'):
            t, c = parents == p, cp == p
            # No anchor, other parents or global nulls enter child training/calibration.
            keys = SiblingKeys.fit(matrix(train2)[t], labels[t])
            if len(keys.owners) == 1:
                children[p] = LLRIndexer(keys, {keys.owners[0]: torch.tensor([0., 0.])})
            else:
                children[p] = LLRIndexer.fit(keys, matrix(cal2)[c], cal_labels[c])
        return cls(parent, children, flat, bank_parent, beam, rescue, parent_gate)

    def trace(self, x1, x2, delta=0., flat_delta=0., flat_inhibitory=None,
              cached_paths=None, flat_scores=None):
        if len(x1) != len(x2):
            raise ValueError('hierarchical feature rows must align')
        if cached_paths is None:
            ps = self.parent.scores(x1)
            ranked = ps.masked_fill(ps <= 0, -torch.inf) if self.parent_gate else ps
            po = ranked.argsort(dim=1, descending=True, stable=True)[:, :self.beam]
            scores = torch.full((len(ps), len(self.owners)), -1e100, dtype=torch.double)
            costs = torch.full((len(ps),), len(self.parent.owners), dtype=torch.long)
            for p, child in self.children.items():
                idx = self.parent.owners.index(p)
                visited = (po == idx).any(1)
                if self.parent_gate:
                    visited &= ps[:, idx] > 0
                rows = visited.nonzero().flatten()
                if len(rows):
                    cs = child.scores(matrix(x2)[rows])
                    for j, b in enumerate(child.owners):
                        scores[rows, self.owners.index(b)] = ps[rows, idx]+cs[:, j]
                    costs[rows] += len(child.owners)
        else:
            scores = cached_paths['scores']
            costs = torch.tensor(cached_paths['comparisons'])
        routes = gate(scores, self.owners, delta)
        _, best, margin = winners(scores)
        rescued = (best <= 0) | (margin <= delta) if self.rescue else torch.zeros(len(scores), dtype=torch.bool)
        if rescued.any():
            rows = rescued.nonzero().flatten()
            fallback = (gate(flat_scores[rows], self.flat.owners, flat_delta) if flat_scores is not None else
                self.flat.select(matrix(x2)[rows], flat_delta,
                    inhibitory=matrix(flat_inhibitory)[rows] if flat_inhibitory is not None else None))
            for i, r in zip(rows.tolist(), fallback):
                routes[i] = r
            costs[rows] += len(self.owners)
        return dict(routes=routes, comparisons=costs.tolist(), rescued=rescued.tolist(), scores=scores)


def metrics(routes, labels, comparisons=None, rescued=None, forms=None):
    labels = torch.as_tensor(labels).tolist()
    pos = [i for i,b in enumerate(labels) if b >= 0]
    neg = [i for i,b in enumerate(labels) if b < 0]
    hits = sum(labels[i] in routes[i] for i in pos)
    correct = sum(routes[i] == (labels[i],) for i in pos)
    events = sum(bool(routes[i]) for i in neg)
    result = dict(positive_n=len(pos), routing_recall=hits/max(len(pos), 1),
                  route_accuracy=correct/max(len(pos), 1), negative_n=len(neg),
                  false_fire=events/max(len(neg), 1), false_fire_events=events,
                  correct_abstention=1-events/max(len(neg), 1))
    try:
        from . import review_eval as R
    except ImportError:
        import review_eval as R
    result['false_fire_upper_95'] = R.binomial_bound(events, len(neg))
    result['sufficient_negatives'] = len(neg) >= 3000
    result['meets_test_false_fire_point_cap'] = result['false_fire'] <= TARGET
    result['meets_test_false_fire_cap'] = result['false_fire_upper_95'] <= TARGET
    result['false_fire_verdict'] = 'matched at 1e-3' if result['meets_test_false_fire_cap'] else 'unresolved (power)'
    if comparisons is not None:
        result['comparisons_per_query'] = sum(comparisons)/max(len(routes), 1)
        result['positive_comparisons_per_query'] = sum(comparisons[i] for i in pos)/max(len(pos), 1)
        result['negative_comparisons_per_query'] = sum(comparisons[i] for i in neg)/max(len(neg), 1)
    if rescued is not None:
        result['rescue_rate'] = sum(rescued)/max(len(routes), 1)
        result['positive_rescue_rate'] = sum(rescued[i] for i in pos)/max(len(pos), 1)
        result['negative_rescue_rate'] = sum(rescued[i] for i in neg)/max(len(neg), 1)
    if forms is not None:
        result['by_form'] = {f: dict(n=sum(forms[i] == f for i in pos),
            recall=sum(labels[i] in routes[i] for i in pos if forms[i] == f)/max(sum(forms[i] == f for i in pos), 1))
            for f in sorted({forms[i] for i in pos})}
        result['negative_by_kind'] = {f: dict(n=sum(forms[i] == f for i in neg),
            false_fire=sum(bool(routes[i]) for i in neg if forms[i] == f)/max(sum(forms[i] == f for i in neg), 1))
            for f in sorted({forms[i] for i in neg})}
    return result


def simulate(cache, n=100):
    """Split fact IDs, never query variants; patches do not exist for these banks."""
    result = deepcopy(cache)
    if any(s.get('inhibitory') for s in cache['splits'].values()):
        raise ValueError('simulation needs sub-bank inhibitory scores; real-bank scores cannot be relabeled')
    original = cache['fact_owner']
    banks = sorted(set(original.values()))
    if n < len(banks):
        raise ValueError('simulation must split, not merge, banks')
    counts = {b: n//len(banks)+int(i < n % len(banks)) for i,b in enumerate(banks)}
    owners, parents, next_bank = {}, {}, 0
    for b in banks:
        fis = sorted(fi for fi in original if original[fi] == b)
        if len(fis) < counts[b]:
            raise ValueError(f'cannot form {n} nonempty sub-banks: bank {b} has {len(fis)} facts')
        for i in range(counts[b]):
            parents[next_bank+i] = b
        for i, fi in enumerate(fis):
            owners[fi] = next_bank+i % counts[b]
        next_bank += counts[b]
    for split in result['splits'].values():
        split['labels'] = torch.tensor([owners[int(fi)] if fi >= 0 else -1 for fi in split['fact_ids']])
    result['fact_owner'], result['bank_parent'] = owners, parents
    result['simulation'] = 'fact-level nonempty splits of real banks; routing only, no sub-bank patches'
    return result


def validate_cache(cache):
    backend = eda.recorded(cache)
    for patch in cache.get('patches', []):
        eda.require_backend(patch, backend, "C42 bank manifest")
    if cache.get('format') != FORMAT:
        raise ValueError('unsupported C42 feature cache')
    layers = cache['layers']
    owners = set(cache['fact_owner'].values())
    if not layers or set(cache['bank_parent']) != owners or min(owners) < 0:
        raise ValueError('invalid bank manifest')
    seen = set()
    for name in ('train', 'cal', 'val', 'test'):
        s = cache['splits'][name]
        y, fis = torch.as_tensor(s['labels']), torch.as_tensor(s['fact_ids'])
        if y.shape != fis.shape or y.ndim != 1 or not len(y) or len(s['forms']) != len(y):
            raise ValueError('split rows must align')
        for layer in layers:
            if len(matrix(s['features'][layer])) != len(y):
                raise ValueError('feature rows must align')
            inhibitory = s.get('inhibitory', {}).get(layer)
            if inhibitory is not None and matrix(inhibitory).shape != (len(y), len(owners)):
                raise ValueError('cached inhibitory scores must align with sorted bank columns')
        if {int(b) for b in y if b >= 0} != owners:
            raise ValueError('every split must cover every bank')
        if name != 'train' and not (y < 0).any():
            raise ValueError('CAL/VAL/TEST need independent null queries')
        for b, fi in zip(y.tolist(), fis.tolist()):
            if (b < 0) != (fi < 0) or (b >= 0 and cache['fact_owner'].get(fi) != b):
                raise ValueError('fact ownership/label mismatch')
        # Dump stores token sequences; reject leakage across splits if supplied.
        sequences = {tuple(it['p']) for it in s.get('items', [])}
        if seen & sequences:
            raise ValueError('query token sequence leaked across splits')
        seen |= sequences
    for layer in layers:
        if not len(matrix(cache['general'][layer])):
            raise ValueError('general TRAIN anchor required')


def fit_study(cache, beam=2, parent_gate=False):
    """Fit/lock every arm using TRAIN, CAL, VAL; return CPU serving objects."""
    validate_cache(cache)
    splits = cache['splits']
    train, cal, val = (splits[s] for s in ('train','cal','val'))
    y, cy, vy = (torch.as_tensor(s['labels']) for s in (train,cal,val))
    vp, vn = vy >= 0, vy < 0
    candidates = {a: [] for a in ('cosine','zscore','flat_llr','current_hierarchy','llr_beam','llr_rescue')}
    for layer in progress(cache['layers'], 'selector_layers'):
        tx, cx, vx = (s['features'][layer] for s in (train,cal,val))
        raw = Centroid.fit(tx, y, cache['general'][layer])
        vs = raw.scores(vx)
        threshold = cutoff(vs[vn].max(1).values)
        cal_scores = raw.scores(cx)
        mean = torch.stack([cal_scores[cy != b, i].mean() for i,b in enumerate(raw.owners)])
        scale = torch.stack([cal_scores[cy != b, i].std(unbiased=False).clamp_min(1e-6)
                             for i,b in enumerate(raw.owners)])
        zs = (vs-mean)/scale
        zthreshold = cutoff(zs[vn].max(1).values)
        llr = LLRIndexer.fit(raw, cx, cy, inhibitory=cal.get('inhibitory', {}).get(layer))
        ls = llr.scores(vx, val.get('inhibitory', {}).get(layer))
        delta = margin_cutoff(ls[vn])
        for arm, scores, t in [('cosine',vs,threshold),('zscore',zs,zthreshold)]:
            routes = threshold_routes(scores, raw.owners, t)
            candidates[arm].append(dict(layer=layer, index=raw, threshold=t, mean=mean, scale=scale,
                validation=metrics(routes, vy)))
        candidates['flat_llr'].append(dict(layer=layer, index=llr, delta=delta,
            validation=metrics(gate(ls,llr.owners,delta),vy)))
    # Fix the flat choice before fitting rescue; TEST cannot affect either.
    flat = best_candidate(candidates['flat_llr'])
    l2 = flat['layer']
    for l1 in progress(cache['layers'], 'selector_hierarchy_fit'):
        hierarchy = LLRHierarchy.fit(train['features'][l1],train['features'][l2],y,cache['general'][l1],
            cal['features'][l1],cal['features'][l2],cy,cache['bank_parent'],flat['index'],
            min(beam,len(set(cache['bank_parent'].values()))),parent_gate=parent_gate)
        x1, x2 = val['features'][l1],val['features'][l2]
        cached_paths = hierarchy.trace(x1,x2)
        hs = cached_paths['scores']
        flat_scores = flat['index'].scores(x2,val.get('inhibitory', {}).get(l2))
        hd = margin_cutoff(hs[vn])
        for arm, rescue in [('llr_beam',False),('llr_rescue',True)]:
            h = deepcopy(hierarchy)
            h.rescue = rescue
            # Rescue can add false fires. Sweep confidence cutoffs on VAL only.
            choices = [hd] if not rescue else sorted(set([0.,hd]+torch.quantile(
                winners(hs)[2], torch.linspace(0, 1, 17, dtype=torch.double)).tolist()))
            eligible = []
            for d in progress(choices, 'selector_rescue_cutoffs'):
                trace = h.trace(x1,x2,d,flat['delta'],val.get('inhibitory', {}).get(l2),
                                cached_paths=cached_paths,flat_scores=flat_scores)
                m = metrics(trace['routes'],vy,trace['comparisons'],trace['rescued'])
                if m['false_fire'] <= TARGET:
                    eligible.append(dict(layer=l2,parent_layer=l1,index=h,delta=d,flat_delta=flat['delta'],validation=m))
            if eligible:
                candidates[arm].append(best_candidate(eligible))
        # Legacy hierarchy control retains affinity scoring and centroid children.
        legacy = LegacyHierarchy.fit(train['features'][l1],train['features'][l2],y,
                                      cache['general'][l1],cache['general'][l2],cache['bank_parent'],hierarchy.beam)
        # Grid is predeclared and selected fairly at the same query false-fire cap.
        for t1 in (-1.,0.,.2,.5,.8,1.):
            for t2 in progress((-1.,0.,.2,.5,.8,1.), 'selector_legacy_grid'):
                tr = legacy.trace(x1,x2,t1,t2)
                m = metrics(tr['routes'],vy,tr['comparisons'])
                if m['false_fire'] <= TARGET:
                    candidates['current_hierarchy'].append(dict(layer=l2,parent_layer=l1,index=legacy,
                        thresholds=[t1,t2],validation=m))
    return {arm: best_candidate(cs) if cs else None for arm,cs in candidates.items()}


def best_candidate(candidates):
    return max(candidates, key=lambda c: (c['validation']['routing_recall'],
        -c['validation']['false_fire'], -c['validation'].get('comparisons_per_query',0)))


def threshold_routes(scores, owners, threshold):
    order, best, _ = winners(scores)
    return [(owners[int(i[0])],) if s > threshold else () for i,s in zip(order,best)]


class LegacyHierarchy:
    """Current summed log-affinity beam, with single-bank serving for this study."""
    @classmethod
    def fit(cls, x1,x2,labels,g1,g2,bank_parent,beam):
        obj = cls()
        labels = torch.as_tensor(labels)
        parents = torch.tensor([bank_parent[int(b)] for b in labels])
        obj.parent = Centroid.fit(x1,parents,g1)
        obj.children = {p: Centroid.fit(matrix(x2)[parents == p],labels[parents == p],g2) for p in obj.parent.owners}
        obj.beam = beam
        return obj

    def trace(self,x1,x2,t1,t2):
        ps = self.parent.scores(x1)
        order = ps.argsort(dim=1,descending=True,stable=True)
        routes, costs = [], []
        for row in range(len(ps)):
            paths, cost = [], len(self.parent.owners)
            for idx in order[row,:self.beam].tolist():
                if ps[row,idx] < t1:
                    continue
                child = self.children[self.parent.owners[idx]]
                cs = child.scores(matrix(x2)[row:row+1])[0]
                cost += len(child.owners)
                for j in cs.argsort(descending=True,stable=True)[:2].tolist():
                    if cs[j] >= t2:
                        score = math.log(max((float(ps[row,idx])+1)/2,1e-12))+math.log(max((float(cs[j])+1)/2,1e-12))
                        paths.append((score,child.owners[j]))
            paths.sort(key=lambda v: (-v[0],v[1]))
            routes.append((paths[0][1],) if paths else ())
            costs.append(cost)
        return dict(routes=routes,comparisons=costs,rescued=[False]*len(ps))


def route_locked(config, split, arm):
    x = split['features'][config['layer']]
    index = config['index']
    if arm in ('llr_beam','llr_rescue'):
        return index.trace(split['features'][config['parent_layer']],x,config['delta'],config['flat_delta'],
                           split.get('inhibitory', {}).get(config['layer']))
    if arm == 'current_hierarchy':
        return index.trace(split['features'][config['parent_layer']],x,*config['thresholds'])
    scores = index.scores(x, split.get('inhibitory', {}).get(config['layer'])) if arm == 'flat_llr' else index.scores(x)
    if arm == 'flat_llr':
        routes = gate(scores,index.owners,config['delta'])
    else:
        if arm == 'zscore':
            scores = (scores-config['mean'])/config['scale']
        routes = threshold_routes(scores,index.owners,config['threshold'])
    return dict(routes=routes,comparisons=[len(index.owners)]*len(x),rescued=[False]*len(x))


def analyze_set(cache, beam=2, parent_gate=False):
    locked = fit_study(cache,beam,parent_gate)
    test = cache['splits']['test']
    result = dict(eda_backend=eda.recorded(cache), banks=len(set(cache['fact_owner'].values())),simulation=cache.get('simulation'),
                  selection_policy='TRAIN keys; CAL coefficients; VAL cutoffs/layers; TEST locked; one bank or null',arms={})
    if 'negative_sample' in cache:
        result['negative_sample'] = cache['negative_sample']
    for arm,cfg in locked.items():
        if cfg is None:
            result['arms'][arm] = dict(unavailable='no validation-feasible candidate at query false-fire <= 1e-3')
            continue
        trace = route_locked(cfg,test,arm)
        m = metrics(trace['routes'],test['labels'],trace['comparisons'],trace['rescued'],test['forms'])
        config = {k:v for k,v in cfg.items() if k not in ('index','mean','scale')}
        if arm == 'zscore':
            config['per_bank_raw_thresholds'] = {str(b):float(mu+cfg['threshold']*s)
                for b,mu,s in zip(cfg['index'].owners,cfg['mean'],cfg['scale'])}
        if arm == 'flat_llr':
            config['bank_ledger'] = cfg['index'].ledger()
            val = cache['splits']['val']
            null = cfg['index'].scores(val['features'][cfg['layer']],val.get('inhibitory', {}).get(cfg['layer']))[
                torch.as_tensor(val['labels']) < 0]
            order,best,margin = winners(null)
            firing = (best > 0)&(margin > cfg['delta'])
            config['validation_per_bank_rates'] = {str(b):dict(
                gate=float((firing & (order[:,0] == j)).double().mean()),
                evidence=float((null[:,j] > 0).double().mean())) for j,b in enumerate(cfg['index'].owners)}
        if arm in ('llr_beam','llr_rescue'):
            h = cfg['index']
            config.update(parent_ledger=h.parent.ledger(),child_ledgers={str(p):c.ledger() for p,c in h.children.items()},
                          score_rule='parent LLR + child LLR conditional on parent',depth=2,beam=h.beam,
                          parent_gate='LLR > 0' if h.parent_gate else 'none (beam top-k)')
            base_cost = m['comparisons_per_query']-m['rescue_rate']*result['banks']
            m['priced_comparisons'] = dict(depth=2,effective_branching=base_cost/2,
                rescue_rate=m['rescue_rate'],N=result['banks'],value=base_cost+m['rescue_rate']*result['banks'],
                formula='depth * effective_branching + rescue_rate * N',
                accounting='effective_branching includes every visited beam child; exact key counts')
        result['arms'][arm] = dict(metrics=m,locked=config)
    # Choose hierarchy on VAL, before reading its TEST metrics for the kill rule.
    feasible = [a for a in ('current_hierarchy','llr_beam','llr_rescue') if locked[a] is not None]
    result['best_hierarchy'] = max(feasible,key=lambda a: (locked[a]['validation']['route_accuracy'],
        -locked[a]['validation']['false_fire'],-locked[a]['validation'].get('comparisons_per_query',0))) if feasible else None
    if result['banks'] == 100:
        cfg = locked['flat_llr']
        ns = cfg['index'].scores(test['features'][cfg['layer']],test.get('inhibitory', {}).get(cfg['layer']))[
            torch.as_tensor(test['labels']) < 0]
        validation = cache['splits']['val']
        calibration_null = cfg['index'].scores(validation['features'][cfg['layer']],
            validation.get('inhibitory', {}).get(cfg['layer']))[
            torch.as_tensor(validation['labels']) < 0]
        order,best,margin = winners(ns)
        # Rates are measured on VAL, then predict held-out TEST system firing.
        # Local qualification includes the same absolute and margin gate.
        per_bank = []
        for j in range(ns.shape[1]):
            others = calibration_null[:,[i for i in range(ns.shape[1]) if i != j]].max(1).values
            per_bank.append(float(((calibration_null[:,j] > 0)&(
                (calibration_null[:,j]-others) > cfg['delta'])).double().mean()))
        predicted = sum(per_bank)
        measured = float(((best > 0)&(margin > cfg['delta'])).double().mean())
        events = int(((best > 0)&(margin > cfg['delta'])).sum())
        try:
            from . import review_eval as R
        except ImportError:
            import review_eval as R
        low,high = R.binomial_bound(events,len(ns),upper=False),R.binomial_bound(events,len(ns))
        # Also expose ungated evidence correlation; margins can hide it.
        raw_rates = (calibration_null > 0).double().mean(0).tolist()
        result['false_fire_scaling'] = dict(role='diagnostic',affects_kill_rules=False,
            per_bank_rates=per_bank,predicted_union=predicted,
            gate_prediction_rule='sum of mutually exclusive margin-gate rates',
            measured=measured,measured_lower_95=low,measured_upper_95=high,
            matches_prediction=low <= predicted <= high,negative_n=len(ns),prediction_split='VAL',measurement_split='TEST',
            evidence_only_per_bank_rates=raw_rates,
            evidence_only_prediction=1-math.prod(1-p for p in raw_rates),
            evidence_only_prediction_rule='independence product',
            evidence_only_measured=float((ns > 0).any(1).double().mean()),
            caveat='VAL point estimate ignores rate-estimation uncertainty; query independence assumed for interval; margin events are mutually exclusive')
    return result,locked


def kill_rules(results):
    eda.require_same(results, "C42 result set")
    twelve = next((r for r in results if r.get('banks') == 12 and not r.get('simulation')),None)
    hundred = next((r for r in results if r.get('banks') == 100),None)
    rules = dict(llr_recall_gain_required=.05,hierarchy_max_accuracy_loss=.02,simulation_max_rescue=.4,
                 simulation_rescue_denominator='positive queries',
                 llr='unmeasured',hierarchy='unmeasured')
    if twelve:
        a,c = (twelve['arms'][k]['metrics'] for k in ('cosine','flat_llr'))
        matched = all(m['false_fire_upper_95'] <= TARGET for m in (a,c))
        gain = c['routing_recall']-a['routing_recall']
        rules.update(llr_recall_gain=gain,matched_false_fire=matched,
                     llr=('keep' if gain >= .05-1e-12 else 'kill') if matched else 'unresolved (power)')
        best = twelve['best_hierarchy']
        if best:
            h = twelve['arms'][best]['metrics']
            loss = c['route_accuracy']-h['route_accuracy']
            rules['hierarchy_accuracy_loss'] = loss
            rules['hierarchy'] = 'kill' if loss > .02+1e-12 else 'pending simulated-100'
            if any(m['false_fire_upper_95'] > TARGET for m in (c,h)):
                rules['hierarchy'] = 'unresolved (power)'
    if hundred:
        arms = hundred.get('arms',{})
        rescue_metrics = arms.get('llr_rescue',{}).get('metrics',{})
        rescue = rescue_metrics.get('positive_rescue_rate')
        rules['simulation_rescue_rate'] = rescue_metrics.get('rescue_rate')
        rules['simulation_positive_rescue_rate'] = rescue
        if rescue is None:
            flat = arms.get('flat_llr',{}).get('metrics')
            beam = arms.get('llr_beam',{}).get('metrics')
            beam_loss = flat['route_accuracy']-beam['route_accuracy'] if flat and beam else None
            rules['simulation_beam_accuracy_loss'] = beam_loss
            if beam_loss is not None and beam_loss > .02+1e-12:
                rules['hierarchy'] = 'kill'
            elif rules['hierarchy'] != 'kill':
                rules['hierarchy'] = 'unresolved: simulated-100 rescue unavailable'
        elif rescue > .4:
            rules['hierarchy'] = 'kill'
        elif rules['hierarchy'] == 'pending simulated-100':
            rules['hierarchy'] = 'keep'
        scaling = hundred.get('false_fire_scaling')
        rules['llr_scaling'] = ('diagnostic: matches measured interval' if scaling['matches_prediction'] else
            'diagnostic: union prediction outside measured 95% interval') if scaling else 'diagnostic: unavailable'
    return rules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches',type=Path,nargs='+',required=True)
    parser.add_argument('--simulate-100',action='store_true')
    parser.add_argument('--beam',type=int,default=2)
    parser.add_argument('--c42-parent-gate',action='store_true',help='pre-gate parents at LLR > 0 before beam expansion')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    caches = [torch.load(p,map_location='cpu',weights_only=False) for p in args.caches]
    backend = eda.require_same(caches, 'C42 cache set')
    print(f'EDA-BACKEND {backend}', flush=True)
    results = [analyze_set(c,args.beam,args.c42_parent_gate)[0] for c in caches]
    if args.simulate_100:
        source = next((c for c in caches if len(set(c['fact_owner'].values())) == 12),caches[-1])
        try:
            results.append(analyze_set(simulate(source),args.beam,args.c42_parent_gate)[0])
        except ValueError as exc:
            results.append(dict(eda_backend=backend, banks=100,unavailable=str(exc)))
    twelve = next((r for r in results if r.get('banks') == 12 and not r.get('simulation')),None)
    hundred = next((r for r in results if r.get('banks') == 100 and not r.get('unavailable')),None)
    if twelve and hundred:
        rates = twelve['arms']['flat_llr']['locked']['validation_per_bank_rates']
        multiplicities = {b:100//12+int(i < 100 % 12) for i,b in enumerate(sorted(rates,key=int))}
        predicted = sum(rates[b]['gate']*n for b,n in multiplicities.items())
        scaling = hundred['false_fire_scaling']
        scaling['extrapolation_from_real_12'] = dict(predicted=predicted,measured=scaling['measured'],
            role='diagnostic',gate_prediction_rule='sum of mutually exclusive margin-gate rates',
            evidence_only_prediction=1-math.prod((1-rates[b]['evidence'])**n for b,n in multiplicities.items()),
            evidence_only_prediction_rule='independence product',
            matches_prediction=scaling['measured_lower_95'] <= predicted <= scaling['measured_upper_95'],
            assumption='replicate real-bank VAL rates for each fact-split sub-bank; keys and margins may change')
    output = dict(eda_backend=backend, study='C42',target_false_fire=TARGET,results=results,kill_rules=kill_rules(results),
                  cache_paths=[str(p) for p in args.caches],lm_training=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['kill_rules']))
    return output


if __name__ == '__main__':
    main()
