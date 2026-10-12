"""Masked least-squares ternary transactions. No per-row Gram matrices are formed.

The solve is in weight changes u; w = alpha*q_current + u is rounded in two
stages. Rounding error from the largest half of each row is fed back through
two masked CG iterations. General covariance uses a one-pass randomized
Nystrom SVD, retaining only column energies and a small covariance range.
"""
from contextlib import contextmanager
import math

import torch


def masked_matvec(v, x, mask, diagonal, sketch, gen_lambda, gamma):
    v = v * mask
    result = (v @ x.T) @ x + (gen_lambda * diagonal + gamma) * v
    if sketch.numel():
        result = result + gen_lambda * ((v @ sketch) @ sketch.T)
    return result * mask


@torch.no_grad()
def masked_cg(x, target, mask, diagonal, sketch, gen_lambda=1., gamma=1e-4,
              iterations=8, initial=None):
    """Solve all output rows together, with target shaped [tokens, outputs]."""
    rhs = (target.T @ x) * mask
    return _cg_rhs(rhs, x, mask, diagonal, sketch, gen_lambda, gamma, iterations, initial)


@torch.no_grad()
def ternarize(w, current, alpha, mask, x, diagonal, sketch, gen_lambda=1., gamma=1e-4, threshold=.5):
    """One greedy residual-correction sweep, preserving every excluded code."""
    scaled = w / alpha.clamp_min(1e-30)
    order = scaled.abs().masked_fill(~mask, -1.).argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(w.shape[1], device=w.device).expand_as(order))
    first = mask & (ranks < ((mask.sum(1, keepdim=True) + 1) // 2))
    rounded = scaled.sign() * (scaled.abs() > threshold)
    error = (w - rounded * alpha) * first
    remaining = mask & ~first
    # Minimize ||X(v-error)||² + lambda*v'C_g*v + gamma*||v||².
    # The cross term of C_g also contributes to the error-feedback RHS.
    rhs = masked_matvec(error, x, torch.ones_like(mask), diagonal, sketch, gen_lambda, gamma) * remaining
    correction = _cg_rhs(rhs, x, remaining, diagonal, sketch, gen_lambda, gamma, 2)
    corrected = (w + correction) / alpha.clamp_min(1e-30)
    rounded = torch.where(first, rounded, corrected.sign() * (corrected.abs() > threshold))
    return torch.where(mask, rounded, current), torch.where(remaining, w + correction, w)


def _cg_rhs(rhs, x, mask, diagonal, sketch, gen_lambda, gamma, iterations, initial=None):
    # Reuse CG with a supplied RHS, including covariance cross terms.
    solution = torch.zeros_like(rhs) if initial is None else initial.clone() * mask
    residual = rhs - masked_matvec(solution, x, mask, diagonal, sketch, gen_lambda, gamma)
    direction = residual.clone()
    norm = residual.square().sum(1, keepdim=True)
    tolerance = norm * 1e-12
    for _ in range(iterations):
        product = masked_matvec(direction, x, mask, diagonal, sketch, gen_lambda, gamma)
        denominator = (direction * product).sum(1, keepdim=True)
        active = (norm > tolerance) & (denominator > 0)
        scale = torch.where(active, norm / denominator.clamp_min(1e-30), 0.)
        solution.add_(scale * direction)
        residual.sub_(scale * product)
        new_norm = residual.square().sum(1, keepdim=True)
        direction = residual + torch.where(active, new_norm / norm.clamp_min(1e-30), 0.) * direction
        norm = new_norm
    return solution * mask


def rounding_steps(current, direction, alpha, mask, threshold=.5):
    """First positive eta at which each coordinate changes its current code.

    Vacant slots cross +/-threshold; live codes moving toward zero cross
    +/-threshold from the other side, and may subsequently flip. Reinforcing
    a live code cannot change it. Each coordinate counts once, including flips.
    """
    scaled = direction / alpha.clamp_min(1e-30)
    movable = mask & ((current == 0) | (current * scaled < 0)) & (scaled != 0)
    distance = torch.where(current == 0, threshold, 1. - threshold)
    steps = distance / scaled.abs()
    return steps.masked_fill(~movable | ~torch.isfinite(steps), float("inf"))


class _UpdateScaleStats:
    """Bounded-memory global quantiles, using log2 bins (within about 1.1%)."""
    def __init__(self):
        self.histogram = torch.zeros(8193, dtype=torch.float64)
        self.maximum = 0.

    def add(self, scaled):
        if not scaled.numel():
            return
        self.maximum = max(self.maximum, float(scaled.max()))
        self.histogram[0] += int((scaled == 0).sum())
        positive = scaled[scaled > 0]
        self.histogram[1:] += torch.histc(positive.log2().clamp(-128., 128.),
                                        bins=8192, min=-128., max=128.).cpu().double()

    def finish(self):
        cumulative = self.histogram.cumsum(0)
        count = float(cumulative[-1])
        result = {}
        for name, fraction in (("p50", .5), ("p90", .9), ("p99", .99), ("p99_9", .999)):
            index = int(torch.searchsorted(cumulative, max(1., math.ceil(count * fraction)))) if count else 0
            result[name] = min(self.maximum, 2. ** (-128. + (index - .5) / 32.)) if index else 0.
        return dict(result, max=self.maximum)


class CovarianceSketch:
    """O(d*r) streaming randomized covariance approximation, on the input device."""
    def __init__(self, width, rank, device, generator):
        k = min(width, rank + 8) if rank else 0
        self.omega = torch.randn(width, k, device=device, generator=generator)
        self.energy = torch.zeros(width, device=device)
        self.range = torch.zeros_like(self.omega)
        self.small = torch.zeros(k, k, device=device)
        self.count, self.rank = 0, rank

    def add(self, x):
        x = x.float()
        self.energy.add_(x.square().sum(0))
        projected = x @ self.omega
        self.range.add_(x.T @ projected)
        self.small.add_(projected.T @ projected)
        self.count += x.shape[0]

    def finish(self):
        n = max(self.count, 1)
        diagonal = self.energy / n
        if not self.rank or not self.count:
            return diagonal.to(torch.bfloat16), self.range[:, :0].to(torch.bfloat16)
        eigenvalues, eigenvectors = torch.linalg.eigh(self.small / n)
        keep = eigenvalues > eigenvalues.max().clamp_min(1e-30) * 1e-6
        factor = (self.range / n) @ (eigenvectors[:, keep] / eigenvalues[keep].sqrt())
        # Randomized SVD of the covariance factor; U contains eigvec*sqrt(eigval).
        vectors, values, _ = torch.linalg.svd(factor, full_matrices=False)
        u = vectors[:, :self.rank] * values[:self.rank]
        return diagonal.to(torch.bfloat16), u.to(torch.bfloat16)


class CodeTransaction:
    """Store only changed masters; rollback restores their exact original values."""
    def __init__(self, edits):
        self.edits = edits
        self.saved = []

    @torch.no_grad()
    def apply(self):
        if self.saved:
            raise RuntimeError("transaction already applied")
        for param, indices, codes in self.edits:
            flat = param.M.view(-1)
            self.saved.append((param, indices, flat[indices].clone()))
            flat[indices] = codes

    @torch.no_grad()
    def rollback(self):
        for param, indices, original in self.saved:
            param.M.view(-1)[indices] = original
        self.saved.clear()

    def evaluate(self, loss, before):
        self.apply()
        try:
            after = float(loss())
            accepted = math.isfinite(after) and after < before
        except BaseException:
            self.rollback()
            raise
        if not accepted:
            self.rollback()
        return accepted, after


class MLSOptimizer:
    def __init__(self, linears, eta=0., trust=.01, rank=16, gen_lambda=1., gamma=1e-4,
                 iterations=8, threshold=.5, budget=0, max_change=50000, seed=0,
                 target_changes=20000):
        if target_changes < 1:
            raise ValueError("MLS target_changes must be > 0")
        if not eta and not 0. < threshold < 1.:
            raise ValueError("automatic MLS requires a rounding threshold between 0 and 1")
        self.linears = linears                 # [(linear, FSAParam)]
        self.auto_eta = not eta
        self.eta, self.eta_cap = eta or None, eta or None
        self.rank = rank                      # trust is a deprecated, ignored compatibility argument
        self.target_changes = self.target_changes_cap = target_changes
        self.gen_lambda, self.gamma = gen_lambda, gamma
        self.iterations, self.threshold = iterations, threshold
        self.budget, self.max_change = budget, max_change
        self.generators = {}
        self.seed = seed
        self.general, self.facts = {}, {}

    @contextmanager
    def record_facts(self):
        """Output tensor hooks preserve routing weights supplied by autograd."""
        hooks = []
        self.facts.clear()
        for lin, param in self.linears:
            def forward(mod, inp, out, param=param):
                if not out.requires_grad or not out.numel():
                    return
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                def backward(gradient):
                    g = gradient.detach().reshape(-1, gradient.shape[-1])
                    # Masked saturated facts have zero output gradients. Exclude
                    # them without discarding a coordinate's death/flip moves.
                    active = g.ne(0).any(1)
                    if bool(active.any()):
                        self.facts.setdefault(param, []).append((x[active], g[active]))
                out.register_hook(backward)
            hooks.append(lin.register_forward_hook(forward))
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()

    @torch.no_grad()
    def refresh_general(self, forward, windows):
        accumulators, hooks = {}, []
        for lin, param in self.linears:
            device = param.M.device
            if device not in self.generators:
                self.generators[device] = torch.Generator(device=device).manual_seed(self.seed + 211)
            accumulator = CovarianceSketch(lin.in_features, self.rank, device, self.generators[device])
            accumulators[param] = accumulator
            def record(mod, inp, out, accumulator=accumulator):
                accumulator.add(inp[0].detach().reshape(-1, inp[0].shape[-1]))
            hooks.append(lin.register_forward_hook(record))
        try:
            for window in windows:
                forward(window[None])
        finally:
            for hook in hooks:
                hook.remove()
        self.general.clear()
        # Stream SVDs; never retain per-row or full input covariance matrices.
        for param in list(accumulators):
            self.general[param] = accumulators.pop(param).finish()

    def _workspace(self, param):
        current = param._quant(param.M.detach() * param.mask) * param.support
        eligible = param.mask & param.support
        if param.sparse:
            current = param._dense(current)
            eligible = param._dense(eligible).bool()
        return current, eligible & (param.alpha > 0)

    def _data(self, param):
        records = self.facts[param]
        x, gradient = [torch.cat(parts).float() for parts in zip(*records)]
        diagonal, sketch = [part.float() for part in self.general[param]]
        return x, gradient, diagonal, sketch

    def _empty_info(self):
        return dict(predicted_change=0., births=0, deaths=0, flips=0, changes=0,
                    target_changes=self.target_changes, pre_correction_changes=0,
                    u_over_alpha_quantiles=_UpdateScaleStats().finish())

    def _calibrate_eta(self):
        # Stream the globally earliest N rounding crossings. Dense CG workspaces
        # are discarded per linear; only O(N) crossing times survive this pass.
        earliest = None
        for _, param in self.linears:
            if param not in self.facts:
                continue
            x, gradient, diagonal, sketch = self._data(param)
            current, mask = self._workspace(param)
            direction = masked_cg(x, -gradient, mask, diagonal, sketch, self.gen_lambda,
                                  self.gamma, self.iterations)
            steps = rounding_steps(current, direction, param.alpha, mask, self.threshold)
            steps = steps[torch.isfinite(steps)]
            if not steps.numel():
                continue
            steps = steps.topk(min(self.target_changes, steps.numel()), largest=False).values
            if earliest is not None:
                steps = torch.cat((earliest, steps.to(earliest.device)))
            earliest = steps.topk(min(self.target_changes, steps.numel()), largest=False).values
        if earliest is None:
            self.eta = None
            return None
        cutoff = float(earliest.max())
        # Move just beyond the boundary to survive float32 rounding. If several
        # coordinates share it, the second pass breaks ties in module/slot order.
        self.eta = cutoff * (1. + 1e-6)
        return cutoff, int((earliest == cutoff).sum())

    @torch.no_grad()
    def propose(self):
        if self.auto_eta:
            boundary = self._calibrate_eta()
            if boundary is None:
                return CodeTransaction([]), self._empty_info()
            cutoff, ties_left = boundary
        proposals, pre_correction_changes = [], 0
        scales = _UpdateScaleStats()
        for _, param in self.linears:
            current, mask = self._workspace(param)
            if param in self.facts:
                x, gradient, diagonal, sketch = self._data(param)
                if self.auto_eta:
                    direction = masked_cg(x, -gradient, mask, diagonal, sketch, self.gen_lambda,
                                          self.gamma, self.iterations)
                    steps = rounding_steps(current, direction, param.alpha, mask, self.threshold)
                    selected = steps < cutoff
                    tied = (steps == cutoff).flatten().nonzero().flatten()
                    take = min(ties_left, tied.numel())
                    selected.flatten()[tied[:take]] = True
                    ties_left -= take
                    u = self.eta * direction
                    scaled = current + u / param.alpha.clamp_min(1e-30)
                    rounded = scaled.sign() * (scaled.abs() > self.threshold)
                    # A scalar eta cannot split equal crossing times. Preserve
                    # unselected ties (and any extra floating-point crossings).
                    u.masked_fill_(mask & ~selected & (rounded != current), 0.)
                else:
                    u = masked_cg(x, -self.eta * gradient, mask, diagonal, sketch, self.gen_lambda,
                                  self.gamma, self.iterations)
                w = current * param.alpha + u
                scaled = w / param.alpha.clamp_min(1e-30)
                pre_correction_changes += int((mask & (scaled.sign() *
                                               (scaled.abs() > self.threshold) != current)).sum())
                scales.add((u.abs() / param.alpha.clamp_min(1e-30))[mask])
                new, corrected = ternarize(w, current, param.alpha, mask, x, diagonal, sketch,
                                           self.gen_lambda, self.gamma, self.threshold)
                # Finite coordinate gain for screening; full cross terms are
                # recomputed on the bounded transaction below.
                g = gradient.T @ x
                curvature = (x.square().sum(0) + self.gen_lambda *
                             (diagonal + sketch.square().sum(1)) + self.gamma) / self.eta
                confidence = corrected.abs() / param.alpha.clamp_min(1e-30)
            else:
                new, confidence = current, current.abs()
                g, curvature = torch.zeros_like(current), torch.zeros(current.shape[1], device=current.device)
            if param.sparse:
                take = param.idx.long()
                current, new, confidence, g = [a.flatten()[take] for a in (current, new, confidence, g)]
                scale = param.scale
                curvature = curvature[param.cols()]
            else:
                scale = param.alpha.expand_as(current)
                curvature = curvature[None].expand_as(current)
            current, new, confidence, g, scale, curvature = [a.flatten() for a in
                                                             (current, new, confidence, g, scale, curvature)]
            if self.budget:
                live = new.nonzero().flatten()
                keep = live[confidence[live].topk(min(self.budget, live.numel())).indices]
                bounded = torch.zeros_like(new)
                bounded[keep] = new[keep]
                new = bounded
                indices = ((current != 0) | (new != 0)).nonzero().flatten()
            else:
                changed = (new != current).nonzero().flatten()
                delta = (new[changed] - current[changed]) * scale[changed]
                gain = -(g[changed] * delta + .5 * curvature[changed] * delta.square())
                indices = changed[gain.topk(min(self.max_change, changed.numel())).indices]
            proposals.append((param, indices, current.flatten()[indices], new.flatten()[indices],
                              confidence.flatten()[indices], g.flatten()[indices],
                              scale.flatten()[indices], curvature.flatten()[indices]))
            proposals = self._trim(proposals)
        transaction, info = self._bound(proposals)
        info.update(target_changes=self.target_changes, pre_correction_changes=pre_correction_changes,
                    u_over_alpha_quantiles=scales.finish())
        return transaction, info

    def _trim(self, proposals):
        """Running global selection: O(budget + incumbents), or O(max_change)."""
        old, new, confidence, gradient, scale, curvature = [torch.cat([p[k] for p in proposals])
                                                           for k in range(2, 8)]
        if self.budget:
            live = new.nonzero().flatten()
            chosen = live[confidence[live].topk(min(self.budget, live.numel())).indices]
            bounded = torch.zeros_like(new)
            bounded[chosen] = new[chosen]
            new = bounded
            keep = (old != 0) | (new != 0)
        else:
            delta = (new - old) * scale
            gain = -(gradient * delta + .5 * curvature * delta.square())
            chosen = gain.topk(min(self.max_change, gain.numel())).indices
            keep = torch.zeros_like(new, dtype=torch.bool)
            keep[chosen] = True
        trimmed, offset = [], 0
        for param, indices, *fields in proposals:
            size = indices.numel()
            selected = keep[offset:offset + size]
            fields[1] = new[offset:offset + size]
            trimmed.append((param, indices[selected], *(field[selected] for field in fields)))
            offset += size
        return trimmed

    @torch.no_grad()
    def _bound(self, proposals):
        if not proposals:
            return CodeTransaction([]), dict(predicted_change=0., births=0, deaths=0, flips=0, changes=0)
        old, new, confidence, gradient, scale, curvature = [torch.cat([p[k] for p in proposals])
                                                           for k in range(2, 8)]
        live = new.nonzero().flatten()
        if self.budget and live.numel() > self.budget:
            keep = live[confidence[live].topk(self.budget).indices]
            pruned = torch.zeros_like(new)
            pruned[keep] = new[keep]
            new = pruned
        difference = (new - old) * scale
        gain = -(gradient * difference + .5 * curvature * difference.square())
        changed = (new != old).nonzero().flatten()
        chosen = changed[gain[changed].topk(min(self.max_change, changed.numel())).indices]
        final = old.clone()
        final[chosen] = new[chosen]
        # A change cap may omit required deaths from a budgeted exchange. Drop
        # excess births from this transaction, without exceeding the change cap.
        excess = int(final.count_nonzero()) - self.budget if self.budget else 0
        if excess > 0:
            births = ((old == 0) & (final != 0)).nonzero().flatten()
            if births.numel() < excess:
                raise ValueError("loaded MLS patch exceeds --budget")
            drop = births[confidence[births].topk(excess, largest=False).indices]
            final[drop] = old[drop]
        edits, predicted, offset = [], 0., 0
        for param, indices, *_ in proposals:
            size = indices.numel()
            target = final[offset:offset + size]
            previous = old[offset:offset + size]
            changed = target != previous
            edits.append((param, indices[changed], target[changed]))
            if param in self.facts:
                dense = torch.zeros_like(param.M)
                dense.view(-1)[indices] = target - previous
                if param.sparse:
                    dense = param._dense(dense)
                delta = dense * param.alpha
                x, g, diagonal, sketch = self._data(param)
                output = x @ delta.T
                reg = self.gen_lambda * ((delta.square() * diagonal).sum() + (delta @ sketch).square().sum())
                predicted += float((g * output).sum() + .5 / self.eta *
                                   (output.square().sum() + reg + self.gamma * delta.square().sum()))
            offset += size
        info = dict(predicted_change=predicted, births=int(((old == 0) & (final != 0)).sum()),
                    deaths=int(((old != 0) & (final == 0)).sum()),
                    flips=int(((old != 0) & (final == -old)).sum()), changes=int((old != final).sum()))
        return CodeTransaction(edits), info

    def adapt(self, accepted):
        if self.auto_eta:
            if accepted:
                self.target_changes = min(self.target_changes_cap,
                                          max(self.target_changes + 1, int(self.target_changes * 1.25)))
            else:
                floor = min(500, self.target_changes_cap)
                self.target_changes = max(floor, self.target_changes // 2)
            self.eta = None                    # recalibrated to the next direction
        elif self.eta is not None:
            self.eta = min(self.eta_cap, self.eta * 1.5) if accepted else max(1e-30, self.eta * .5)
