"""Joint forced-mask ridge writes with saturated residual targets.

Targets are one vector per (fact, layer), shared across TRAIN phrasings and
answer/clean-stop positions. All selected layers are optimized together; layer
l receives z_l / (# selected layers at or after l), in model execution order.
The frozen latent_up P is inverted by an SVD pseudoinverse: this minimizes
||P y - delta||^2, including the P.T P metric rather than treating latent
coordinates as model-output coordinates. With coalesced routing masses g_e,
expert targets are g_e*y/sum(g^2). We fit gate-weighted keys g_e*h to the
gate-weighted targets g_e^2*y/sum(g^2), so their contributions sum to y.
General-text covariance uses the same gate-weighted keys (rank + diagonal).

--projs both splits each latent target equally between down and up. Up targets
use the minimum-norm pseudoinverse of W_down diag(2 relu(up(x))) at each visit;
this is a local linearization of relu squared, not an exact nonlinear solve.

--normalize computes Chan mean/variance of [key | target] over ALL visits per
expert/projection, then centers and standardizes both before solving. One
sqrt(n)*mean observation enforces the no-bias constraint: the centered fit plus
this observation has exactly the uncentered standardized residual objective.
This avoids silently dropping the target mean or installing a forbidden affine
bias. The covariance is rescaled without centering (preservation measures actual
writes). Weights map back by target_std/key_std. Repair uses that same quadratic,
including ridge. Zero-variance coordinates have scale 1. The normalized dual
solve is (n+1) x (n+1), versus n x n without normalization.

Only ternary patches are exported, through e2e_fsa.save_patch. --eval-continuous
evaluates the unrounded solution with temporary Linear output hooks, printing
CONT-EVAL {json}; the harness's ternary loader cannot represent those weights.
Neither path changes base weights; disabling FSA revokes both exactly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from . import e2e_fsa as E
    from . import hebbian_write as H
except ImportError:
    import e2e_fsa as E
    import hebbian_write as H


class RunningMoments:
    """Chan parallel updates, accumulated in float64 on CPU."""

    def __init__(self, dimension):
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)

    def add(self, values):
        x = values.detach().double().cpu()
        if not len(x):
            return
        mean = x.mean(0)
        m2 = (x - mean).square().sum(0)
        n = len(x)
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def scales(self):
        std = (self.m2 / max(self.count, 1)).sqrt()
        return torch.where(std > 1e-6, std, torch.ones_like(std)).float()


def normalized_records(records, normalize=False):
    keys = torch.cat([k for k, _ in records]).float()
    targets = torch.cat([d for _, d in records]).float()
    key_scale, target_scale = torch.ones(keys.shape[1]), torch.ones(targets.shape[1])
    stats = None
    if normalize:
        stats = RunningMoments(keys.shape[1] + targets.shape[1])
        for k, d in records:
            stats.add(torch.cat((k, d), 1))
        key_mean, target_mean = stats.mean.float().split((keys.shape[1], targets.shape[1]))
        key_scale, target_scale = stats.scales().split((keys.shape[1], targets.shape[1]))
        keys = (keys - key_mean) / key_scale
        targets = (targets - target_mean) / target_scale
        root_n = math.sqrt(stats.count)
        keys = torch.cat((keys, (root_n * key_mean / key_scale)[None, :]))
        targets = torch.cat((targets, (root_n * target_mean / target_scale)[None, :]))
    return keys, targets, key_scale, target_scale, stats


def covariance_parts(covariance, dimension, device="cpu"):
    """Reuse Hebbian's Nystrom reconstruction, excluding its extra ridge."""
    if covariance is None or not covariance.count:
        return torch.ones(dimension, device=device), torch.empty(dimension, 0, device=device)
    if not covariance.rank:
        raise ValueError("masked solve requires a rank + diagonal covariance sketch")
    solver = covariance.solver(device=device)
    regularizer = max(1e-3 * float((covariance.diagonal / covariance.count).mean()), 1e-8)
    return (solver.diagonal - regularizer).clamp_min(0), solver.factor


def _inverse_regularizer(rhs, diagonal, factor, preservation, ridge):
    """Batched Woodbury application of (lambda C_S + gamma I)^-1."""
    a = preservation * diagonal + ridge
    z = rhs / a[:, :, None]
    if not factor.shape[-1] or preservation == 0:
        return z
    u = factor * math.sqrt(preservation)
    small = torch.eye(u.shape[-1], device=u.device, dtype=u.dtype) + u.transpose(1, 2) @ (u / a[:, :, None])
    correction = torch.linalg.solve(small, u.transpose(1, 2) @ z)
    return z - (u @ correction) / a[:, :, None]


def dual_ridge(keys, targets, mask, diagonal, factor=None, preservation=1.0, ridge=1e-3, row_batch=32):
    """Minimize ||K_S w-d||^2 + lambda w.T C_S w + gamma ||w||^2.

    K is [visits, input], d [visits, output], mask [output, input]. Each
    row's writable submatrix is distinct. The only data solve is visits x
    visits, batched across rows; the covariance solve is sketch-rank x rank.
    """
    if not math.isfinite(ridge) or ridge <= 0 or not math.isfinite(preservation) or preservation < 0:
        raise ValueError("ridge must be positive and preservation nonnegative")
    if row_batch <= 0:
        raise ValueError("row_batch must be positive")
    if keys.ndim != 2 or targets.ndim != 2 or mask.shape != (targets.shape[1], keys.shape[1]) or len(keys) != len(targets):
        raise ValueError("inconsistent key, target, or mask shapes")
    if diagonal.shape != (keys.shape[1],):
        raise ValueError("covariance diagonal shape differs from key dimension")
    factor = keys.new_empty(keys.shape[1], 0) if factor is None else factor
    if factor.ndim != 2 or factor.shape[0] != keys.shape[1]:
        raise ValueError("covariance factor shape differs from key dimension")
    if any(not bool(torch.isfinite(v).all()) for v in (keys, targets, diagonal, factor)) or bool((diagonal < 0).any()):
        raise ValueError("solve inputs must be finite and covariance diagonal nonnegative")
    result = keys.new_zeros(mask.shape)
    if not len(keys):
        return result
    eye = torch.eye(len(keys), device=keys.device, dtype=keys.dtype)
    for start in range(0, len(mask), row_batch):
        selected = mask[start:start + row_batch].bool()
        counts = selected.sum(1)
        width = int(counts.max())
        if not width:
            continue
        # Pad only this row batch; masked padding carries neither data nor
        # covariance mass and therefore cannot affect a real coordinate.
        cols = torch.zeros(len(selected), width, dtype=torch.long, device=keys.device)
        valid = torch.arange(width, device=keys.device)[None, :] < counts[:, None]
        rows, actual = selected.nonzero(as_tuple=True)
        within = selected.long().cumsum(1)[rows, actual] - 1
        cols[rows, within] = actual
        k = keys[:, cols].permute(1, 2, 0) * valid[:, :, None]
        diag = diagonal[cols] * valid
        u = factor[cols] * valid[:, :, None]
        inverse_k = _inverse_regularizer(k, diag, u, preservation, ridge)
        dual = torch.linalg.solve(eye + k.transpose(1, 2) @ inverse_k,
                                  targets[:, start:start + len(selected)].T[:, :, None])
        w = (inverse_k @ dual).squeeze(-1)
        result[start + rows, actual] = w[rows, within]
    return result


def quadratic(weights, keys, targets, diagonal, factor, preservation=1.0, ridge=1e-3):
    """One row's cached objective, also used by the CPU repair tests."""
    return ((keys @ weights - targets).square().sum()
            + preservation * ((diagonal * weights.square()).sum() + (factor.T @ weights).square().sum())
            + ridge * weights.square().sum())


def round_ternary(weights, mask, alpha, threshold=0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    alpha = alpha.flatten().to(weights)
    if weights.shape != mask.shape or len(alpha) != len(weights) or bool((alpha < 0).any()) or not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(alpha).all()):
        raise ValueError("finite weights and nonnegative matching row scales required")
    return (weights.sign() * (weights.abs() > threshold * alpha[:, None]) * mask
            * (alpha[:, None] > 0)).to(torch.int8)


@torch.no_grad()
def repair_ternary(codes, keys, targets, mask, scales, diagonal, factor,
                   preservation=1.0, ridge=1e-3, sweeps=1, row_batch=32):
    """Exhaustive 81-state coordinate descent in each physical pair8 block.

    scales [output,input] convert each ternary code to solve coordinates.
    The mask may thin a forced group down to fewer than four slots; missing
    slots are held at zero. Residuals and covariance products are updated in
    place. Only 4x4 group Hessians are cached, never an input x input Hessian.
    """
    if sweeps not in (1, 2) or row_batch <= 0:
        raise ValueError("repair needs one or two sweeps and positive row_batch")
    if codes.shape != mask.shape or scales.shape != codes.shape or targets.shape != (len(keys), len(codes)):
        raise ValueError("repair shapes differ")
    if bool((codes[~mask.bool()] != 0).any()) or not bool(((codes == -1) | (codes == 0) | (codes == 1)).all()):
        raise ValueError("repair needs ternary codes inside the mask")
    if codes.shape[1] % 8 or bool((mask.reshape(len(mask), -1, 8).sum(-1) > 4).any()):
        raise ValueError("repair expects at most four forced slots per physical 8-weight block")
    result = codes.clone()
    states = keys.new_tensor(list(itertools.product((-1, 0, 1), repeat=4)))
    for start in range(0, len(codes), row_batch):
        q = result[start:start + row_batch]
        b = len(q)
        rr = torch.arange(b, device=keys.device)[:, None]
        w = q.to(keys.dtype) * scales[start:start + b]
        residual = w @ keys.T - targets[:, start:start + b].T
        projected = w @ factor
        a = preservation * diagonal + ridge
        groups = []
        for block in range(0, keys.shape[1], 8):
            reserve = mask[start:start + b, block:block + 8].bool()
            valid = torch.arange(4, device=keys.device)[None, :] < reserve.sum(1)[:, None]
            cols = torch.zeros(b, 4, dtype=torch.long, device=keys.device)
            rows, actual = reserve.nonzero(as_tuple=True)
            within = reserve.long().cumsum(1)[rows, actual] - 1
            cols[rows, within] = block + actual
            kg = keys[:, cols].permute(1, 0, 2) * valid[:, None, :]
            ug = factor[cols] * valid[:, :, None]
            ag = a[cols] * valid
            sg = scales[start:start + b][rr, cols] * valid
            hessian = kg.transpose(1, 2) @ kg + preservation * (ug @ ug.transpose(1, 2)) + torch.diag_embed(ag)
            groups.append((cols, valid, kg, ug, ag, sg, hessian))
        for _ in range(sweeps):
            for cols, valid, kg, ug, ag, sg, hessian in groups:
                old = w[rr, cols] * valid
                gradient = (kg.transpose(1, 2) @ residual[:, :, None]).squeeze(-1)
                gradient += preservation * (ug @ projected[:, :, None]).squeeze(-1) + ag * old
                candidates = states[None, :, :] * sg[:, None, :]
                changes = candidates - old[:, None, :]
                gains = 2 * (changes * gradient[:, None, :]).sum(-1)
                gains += torch.einsum("bsi,bij,bsj->bs", changes, hessian, changes)
                best = gains.argmin(1)
                improve = gains[torch.arange(b, device=keys.device), best] < -1e-7
                change = changes[torch.arange(b, device=keys.device), best] * improve[:, None]
                new_codes = states[best].to(q.dtype)
                for slot in range(4):
                    eligible = valid[:, slot] & improve
                    rows = eligible.nonzero().flatten()
                    q[rows, cols[rows, slot]] = new_codes[rows, slot]
                    w[rows, cols[rows, slot]] += change[rows, slot]
                residual += (kg @ change[:, :, None]).squeeze(-1)
                projected += (ug.transpose(1, 2) @ change[:, :, None]).squeeze(-1)
    return result


def code_entries(codes, weights):
    idx = codes.flatten().nonzero().flatten()
    return idx.to(torch.int32).cpu(), codes.flatten()[idx].cpu(), weights.flatten()[idx].abs().float().cpu()


class TargetCapture(H.RoutedCapture):
    """Reuse the exact resident hook/dispatch lifecycle, retaining routing mass."""

    def __init__(self, model, layer_ids, projs, cov_rank=64, seed=0):
        super().__init__(model, layer_ids, projs, cov_rank, seed)
        self.inverses = {}
        self.z, self.owners = {}, None
        self.collect = False
        order = sorted(layer_ids)
        self.remaining = {li: len(order) - order.index(li) for li in layer_ids}

    def configure_targets(self, positions, owners, z, collect=False):
        self.batch("facts", positions)
        self.owners, self.z, self.collect = owners, z, collect

    def _dispatch(self, li, x, indices, weights):
        super()._dispatch(li, x, indices, weights)
        ids = indices.detach()
        # Coalesce duplicate assignments before sum(g^2) and expert fitting.
        masses = weights.detach().new_zeros(len(ids), len(self.layers[li].experts))
        masses.scatter_add_(1, ids, weights.detach())
        norm = masses.square().sum(1).clamp_min(1e-12)
        for ei, plan in self.plans[li].items():
            tokens = (ids == ei).any(1).nonzero().flatten()
            selected = plan["rows"].to(ids.device)
            tids = tokens[selected]
            plan["mass"] = masses[tids, ei]
            plan["norm"] = norm[tids]
            plan["target_rows"] = torch.searchsorted(self.positions.to(ids.device), tids)
            plan["up"] = None

    def _inverse(self, li, device):
        if li not in self.inverses:
            moe = self.layers[li]
            p = getattr(moe, "latent_up", None)
            if p is None or isinstance(p, torch.nn.Identity):
                dimension = next(e for e in moe.experts if e is not None).down_proj.out_features
                self.inverses[li] = torch.eye(dimension, device=device)
            elif isinstance(p, torch.nn.Linear):
                self.inverses[li] = torch.linalg.pinv(p.weight.detach().float().to(device))
            else:
                raise TypeError("latent_up must be a frozen Linear or identity")
        return self.inverses[li].to(device)

    def _linear(self, li, ei, proj, inputs, output):
        plan = self.plans.get(li, {}).get(ei)
        if plan is None or proj in plan["seen"]:
            raise RuntimeError("projection hook does not match grouped dispatch")
        plan["seen"].add(proj)
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        if len(x) < plan["count"] or output.shape[:-1] != inputs[0].shape[:-1]:
            raise RuntimeError("projection token layout differs from grouped dispatch")
        rows = plan["rows"].to(x.device)
        if not rows.numel():
            return
        name = self.names[li, ei, proj]
        keys = x[rows].detach().float()
        mass = plan["mass"].to(keys)
        if self.phase == "covariance":
            if name not in self.covariances:
                d = keys.shape[1]
                cov = H.Covariance(d, self.cov_rank, self.seed, self.cpu_omegas.get(d))
                self.covariances[name] = cov
                self.cpu_omegas[d] = cov.omega
            self.covariances[name].add(keys * mass[:, None])
            return
        if not self.collect:
            return
        owners = self.owners[plan["target_rows"].to(self.owners.device)].to(keys.device)
        delta = self.z[li][owners].detach().float() / self.remaining[li]
        latent = delta @ self._inverse(li, keys.device).T
        weighted_target = latent * (mass.square() / plan["norm"].to(keys))[:, None]
        if len(self.projs) == 2:
            weighted_target *= 0.5
        if proj == "up_proj":
            # Cache visits until down provides relu² inputs and its Jacobian.
            plan["up"] = (keys, output.reshape(-1, output.shape[-1])[rows].detach().float())
            return
        if plan["up"] is not None:
            up_keys, preactivation = plan["up"]
            expert = self.layers[li].experts[ei]
            wd = expert.down_proj.parametrizations.weight.original.detach().float().to(keys.device)
            jacobian = wd[None, :, :] * (2 * preactivation.relu())[:, None, :]
            # Batched SVD, in the expert's latent output metric.
            up_target = (torch.linalg.pinv(jacobian) @ weighted_target[:, :, None]).squeeze(-1)
            up_name = self.names[li, ei, "up_proj"]
            self.records.setdefault(up_name, []).append(((up_keys * mass[:, None]).cpu(), up_target.cpu()))
        self.records.setdefault(name, []).append(((keys * mass[:, None]).cpu(), weighted_target.cpu()))

    def _inject(self, li, output):
        if self.phase != "facts":
            return output
        if not torch.is_tensor(output):
            raise TypeError("target optimization expects tensor MoE outputs")
        flat = output.reshape(-1, output.shape[-1])
        positions = self.positions.to(flat.device)
        delta = self.z[li][self.owners.to(flat.device)] / self.remaining[li]
        if delta.shape != flat[positions].shape:
            raise ValueError("MoE output and target dimensions differ")
        added = torch.zeros_like(flat).index_add(0, positions, delta.to(flat.dtype))
        return (flat + added).reshape_as(output)

    def __enter__(self):
        super().__enter__()
        try:
            for li, moe in self.layers.items():
                self.hooks.append(moe.register_forward_hook(lambda mod, inp, out, li=li: self._inject(li, out)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self


def build_target_items(tok, facts, templates):
    items = []
    for fi, fact in enumerate(facts):
        for template in templates:
            seq, prefix = E.encode_fact(tok, template, fact, E.BOS)
            items.append(dict(fi=fi, p=seq[:prefix], a=seq[prefix:]))
    return items


def target_batch(items, device):
    if any(not it["p"] or not it["a"] for it in items):
        raise ValueError("target prompts and answer/stop continuations must be nonempty")
    facts = list(dict.fromkeys(it["fi"] for it in items))
    local = {fi: i for i, fi in enumerate(facts)}
    width = max(len(it["p"]) + len(it["a"]) - 1 for it in items)
    tokens = torch.zeros(len(items), width, dtype=torch.long, device=device)
    positions, owners, gold = [], [], []
    for row, it in enumerate(items):
        seq = it["p"] + it["a"]
        tokens[row, :len(seq) - 1] = torch.tensor(seq[:-1], device=device)
        positions.extend(row * width + p for p in range(len(it["p"]) - 1, len(seq) - 1))
        owners.extend([local[it["fi"]]] * len(it["a"]))
        gold.extend(it["a"])
    return tokens, torch.tensor(positions), torch.tensor(owners), torch.tensor(gold, device=device), facts


def target_scores(logits, positions, owners, gold, count, margin):
    selected = logits.reshape(-1, logits.shape[-1])[positions.to(logits.device)].float()
    correct = selected.gather(1, gold[:, None]).squeeze(1)
    competitors = selected.detach().clone()
    competitors.scatter_(1, gold[:, None], -torch.inf)
    margins = correct.detach() - competitors.max(1).values
    owners = owners.to(logits.device)
    failed = (margins < margin) | (selected.detach().argmax(-1) != gold)
    failing = torch.zeros(count, device=logits.device).scatter_add_(0, owners, failed.float())
    logp = selected.log_softmax(-1).gather(1, gold[:, None]).squeeze(1)
    total = logp.new_zeros(count).scatter_add_(0, owners, logp)
    denom = logp.new_zeros(count).scatter_add_(0, owners, torch.ones_like(logp))
    return total / denom, failing == 0, margins


def optimize_targets(model, capture, items, device, batch_size=8, steps=25, margin=2.0, lr=0.5, l2=1e-3):
    by_fact = {}
    for item in items:
        by_fact.setdefault(item["fi"], []).append(item)
    fact_ids = list(by_fact)
    summary = dict(facts=len(fact_ids), saturated=0, updates=0, min_margin=None)
    with H.frozen_base(), torch.enable_grad():
        for start in range(0, len(fact_ids), batch_size):
            batch = [it for fi in fact_ids[start:start + batch_size] for it in by_fact[fi]]
            tokens, positions, owners, gold, facts = target_batch(batch, device)
            z = {}
            for li, moe in capture.layers.items():
                p = getattr(moe, "latent_up", None)
                dim = p.out_features if isinstance(p, torch.nn.Linear) else next(e for e in moe.experts if e is not None).down_proj.out_features
                z[li] = torch.nn.Parameter(torch.zeros(len(facts), dim, device=device))
            optimizer = torch.optim.Adam(list(z.values()), lr=lr)
            for step in range(steps + 1):
                capture.configure_targets(positions, owners, z)
                logits = E.logits_of(model, tokens)
                logp, saturated, margins = target_scores(logits, positions, owners, gold, len(facts), margin)
                if bool(saturated.all()) or step == steps:
                    break
                active = ~saturated
                penalty = sum(v[active].square().sum(1).mean() for v in z.values())
                loss = -logp[active].mean() + l2 * penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                frozen = {li: v[saturated].detach().clone() for li, v in z.items()}
                optimizer.step()
                with torch.no_grad():
                    for li, v in z.items():
                        v[saturated] = frozen[li]
                summary["updates"] += 1
                del logits, loss
            summary["saturated"] += int(saturated.sum())
            minimum = float(margins.min())
            summary["min_margin"] = minimum if summary["min_margin"] is None else min(minimum, summary["min_margin"])
            # Re-run with the final residuals to collect the expert keys and
            # their targets; no stale keys from pre-optimization forwards.
            with torch.no_grad():
                capture.configure_targets(positions, owners, z, collect=True)
                E.logits_of(model, tokens)
            print(f"targets: {min(start + batch_size, len(fact_ids))}/{len(fact_ids)} facts; "
                  f"saturated={summary['saturated']} min_margin={summary['min_margin']:.4g}", flush=True)
            del logits, z, optimizer
    return summary


@torch.no_grad()
def install_continuous(lin, weights):
    pm = lin.parametrizations.weight[0]
    mask = H.writable_mask(pm).to(weights.device)
    if bool((weights[~mask] != 0).any()):
        raise ValueError("continuous weights outside attached reserve")
    alpha = pm.alpha.to(weights).expand_as(weights)
    codes = torch.where(alpha > 0, weights / alpha.clamp_min(1e-30), torch.zeros_like(weights))
    if getattr(pm, "sparse", False):
        codes = codes.flatten()[pm.idx.long().to(weights.device)]
    pm.M.copy_(codes.to(pm.M.device))


@contextmanager
def continuous_hooks(model):
    """Evaluate each Linear with the saved continuous masters, without export."""
    handles = []
    try:
        for _name, lin in E.patched_linears(model):
            pm = lin.parametrizations.weight[0]

            def hook(module, inputs, output, pm=pm):
                if not E.FSAParam.enabled:
                    return output
                master = pm.M.detach()
                if getattr(pm, "sparse", False):
                    master = pm._dense(master * pm.support)
                else:
                    master = master * pm.mask * pm.support
                delta = (master * pm.alpha).to(inputs[0])
                weight = module.parametrizations.weight.original.detach().to(inputs[0])
                # Recompute instead of adding a correction to a bf16 rounded
                # ternary output, which would distort the diagnostic number.
                return F.linear(inputs[0], weight + delta, module.bias)

            handles.append(lin.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    E.eda.add_argument(ap)
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--facts-per-person", type=int, default=len(E.ATTRS))
    ap.add_argument("--shared-attrs", type=int, default=0)
    ap.add_argument("--values", choices=("disjoint", "overlap"), default="disjoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--phrasings", type=int, default=0, help="first k TRAIN phrasings (0 = all)")
    ap.add_argument("--layers", default="14,15")
    ap.add_argument("--projs", choices=("down", "both"), default="down")
    ap.add_argument("--sparse-state", action="store_true")
    ap.add_argument("--slot-frac", type=float, default=1.0)
    ap.add_argument("--target-batch", type=int, default=8, help="facts per batch; includes all their TRAIN phrasings")
    ap.add_argument("--target-steps", type=int, default=25)
    ap.add_argument("--target-margin", type=float, default=2.0, help="nats, including clean-stop tokens")
    ap.add_argument("--target-lr", type=float, default=0.5)
    ap.add_argument("--target-l2", type=float, default=1e-3)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--cov-tokens", type=Path, default=E.DATA / "rehearsal-fineweb.json")
    ap.add_argument("--cov-max-tokens", type=int, default=100_000)
    ap.add_argument("--cov-window", type=int, default=256)
    ap.add_argument("--cov-rank", type=int, default=64)
    ap.add_argument("--lambda-g", type=float, default=1.0, help="general-text preservation coefficient")
    ap.add_argument("--ridge", "--gamma", dest="ridge", type=float, default=1e-3)
    ap.add_argument("--row-batch", type=int, default=32)
    ap.add_argument("--solve-device", default="cuda:0")
    ap.add_argument("--thr", nargs="+", default=["0.3,0.4,0.5"])
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--repair-sweeps", type=int, choices=(1, 2), default=1)
    ap.add_argument("--budget", type=int, default=0, help="optional global nnz cap (0 = unlimited)")
    ap.add_argument("--eval-continuous", action="store_true")
    ap.add_argument("--gen-bs", type=int, default=1)
    ap.add_argument("--gen-max-new", type=int, default=0)
    ap.add_argument("--gen-stop-k", type=int, default=2)
    ap.add_argument("--name", default="masked")
    ap.add_argument("--out", type=Path, default=Path("out"))
    return ap


def validate_args(ap, args):
    try:
        layers = [int(li) for li in args.layers.split(",")]
        thresholds = [float(t) for part in args.thr for t in part.split(",")]
    except ValueError:
        ap.error("layers must be integers and thresholds numbers")
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        ap.error("layers must be distinct nonnegative indices")
    if not thresholds or len(set(thresholds)) != len(thresholds) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        ap.error("thresholds must be distinct finite fractions in [0, 1]")
    if not 0 < args.slot_frac <= 1 or (args.slot_frac < 1 and not args.sparse_state):
        ap.error("slot-frac must be in (0, 1] and thinning needs --sparse-state")
    if any(v <= 0 for v in (args.target_batch, args.cov_max_tokens, args.cov_window, args.cov_rank, args.row_batch, args.gen_bs, args.gen_stop_k)) or args.target_steps < 0 or args.budget < 0 or args.gen_max_new < 0:
        ap.error("batch/sketch/token counts must be positive; steps/budget/max-new nonnegative")
    if any(not math.isfinite(v) or v < 0 for v in (args.target_margin, args.target_l2, args.lambda_g)) or any(not math.isfinite(v) or v <= 0 for v in (args.target_lr, args.ridge)):
        ap.error("coefficients must be finite; lr/ridge positive, margin/l2/lambda-g nonnegative")
    templates = E.TRAIN_AUG_T if args.aug else E.TRAIN_T
    if not 0 <= args.phrasings <= len(templates):
        ap.error(f"phrasings must be between 0 and {len(templates)}")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        ap.error("name must be a filename stem")
    return layers, thresholds, templates[:args.phrasings] if args.phrasings else templates


def main():
    ap = build_parser()
    args = ap.parse_args()
    E.eda.announce(args)
    layer_ids, thresholds, templates = validate_args(ap, args)
    try:
        facts = E.make_world(args.seed, args.people, args.facts_per_person, args.shared_attrs, args.values)
    except (AssertionError, ValueError) as exc:
        ap.error(f"invalid fact world: {exc}")
    if not facts:
        ap.error("fact world must be nonempty")
    tokens = json.loads(args.cov_tokens.read_text())
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        ap.error("cov-tokens must be a nonempty JSON list of nonnegative token IDs")
    if not torch.cuda.is_available():
        ap.error("the standalone model run requires CUDA and /workspace/malleable model/data")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    try:
        from .fast_moe import install, uninstall
    except ImportError:
        from fast_moe import install, uninstall

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    model, _cfg, device = E.eda.build_model(build_model, args, E.EXPORT, device="cuda:0", model_config=None, step=58814,
                                    expert_format="bf16")
    model.eval().requires_grad_(False)
    if max(layer_ids) >= len(E.moe_layers(model)):
        ap.error("layer index outside model's MoE layers")
    projs = ("down_proj",) if args.projs == "down" else ("up_proj", "down_proj")
    _params, n_slots, _ = E.attach(model, layer_ids, "fsa_forced", 16, "ternary", projs,
                                 sparse=args.sparse_state, slot_frac=args.slot_frac, seed=args.seed)
    model.requires_grad_(False)
    E.set_enabled(True)
    print(f"facts={len(facts)} phrasings={len(templates)} writable_slots={n_slots}", flush=True)
    timing = {}
    install(model, patched_layer_ids=layer_ids, mode="exact")
    try:
        with TargetCapture(model, layer_ids, projs, args.cov_rank, args.seed) as capture:
            H.synchronize(device)
            start = time.monotonic()
            n_tokens = H.covariance_pass(model, capture, tokens, device, args.cov_max_tokens, args.cov_window, E.BOS)
            H.synchronize(device)
            timing["covariance"] = time.monotonic() - start
            print(f"covariance: {n_tokens} tokens; {timing['covariance']:.3f} seconds", flush=True)
            start = time.monotonic()
            target_info = optimize_targets(model, capture, build_target_items(tok, facts, templates), device,
                                           args.target_batch, args.target_steps, args.target_margin, args.target_lr, args.target_l2)
            H.synchronize(device)
            timing["targets"] = time.monotonic() - start
            print(f"targets: {timing['targets']:.3f} seconds; {json.dumps(target_info)}", flush=True)

        start = time.monotonic()
        sweeps = {t: {} for t in thresholds}
        threshold_seconds = {t: 0.0 for t in thresholds}
        unseen, counts, normalization = [], {}, {}
        for name, lin in E.patched_linears(model):
            records = capture.records.pop(name, [])
            cov = capture.covariances.pop(name, None)
            counts[name] = 0 if cov is None else cov.count
            if not records:
                for entries in sweeps.values():
                    entries[name] = (torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int8), torch.empty(0))
                continue
            if cov is None:
                unseen.append(name)
            keys, targets, ks, ds, stats = normalized_records(records, args.normalize)
            solve_device = torch.device(args.solve_device)
            keys, targets = keys.to(solve_device), targets.to(solve_device)
            ks, ds = ks.to(solve_device), ds.to(solve_device)
            diag, factor = covariance_parts(cov, lin.in_features, solve_device)
            diag, factor = diag / ks.square(), factor / ks[:, None]
            pm = lin.parametrizations.weight[0]
            mask = H.writable_mask(pm).to(solve_device)
            normalized = dual_ridge(keys, targets, mask, diag, factor, args.lambda_g, args.ridge, args.row_batch)
            physical = normalized * ds[:, None] / ks[None, :]
            alpha = pm.alpha.flatten().to(solve_device)
            physical[alpha == 0] = 0
            for threshold, entries in sweeps.items():
                round_start = time.monotonic()
                codes = round_ternary(physical, mask, alpha, threshold)
                if args.repair:
                    scale = alpha[:, None] * ks[None, :] / ds[:, None]
                    codes = repair_ternary(codes, keys, targets, mask, scale, diag, factor,
                                           args.lambda_g, args.ridge, args.repair_sweeps, args.row_batch)
                entries[name] = code_entries(codes, physical)
                threshold_seconds[threshold] += time.monotonic() - round_start
            install_continuous(lin, physical)
            if stats is not None:
                normalization[name] = dict(count=stats.count, mean=stats.mean.tolist(), std=stats.scales().tolist())
            del records, cov, keys, targets, normalized, physical, mask, diag, factor
        H.synchronize(args.solve_device)
        timing["solve_round_repair"] = time.monotonic() - start
        print(f"solve/round/repair: {timing['solve_round_repair']:.3f} seconds; identity fallback={len(unseen)}", flush=True)

        continuous = None
        if args.eval_continuous:
            start = time.monotonic()
            with continuous_hooks(model):
                continuous = E.gen_eval(model, tok, E.build_items(tok, facts, E.VAL_T, E.BOS), device,
                                        args.gen_max_new, args.gen_stop_k, args.gen_bs, E.conflict_map(facts))
            H.synchronize(device)
            timing["continuous_eval"] = time.monotonic() - start
            print("CONT-EVAL " + json.dumps(continuous), flush=True)
        else:
            print("continuous: use --eval-continuous for the unrounded val GEN gate", flush=True)
        meta = H.patch_metadata(args, layer_ids, projs)
        meta["masked_solve"] = dict(targets=target_info, target_steps=args.target_steps,
                                    target_margin=args.target_margin, target_lr=args.target_lr, target_l2=args.target_l2,
                                    phrasings=len(templates), aug=args.aug, gate_weighted=True,
                                    latent_mapping="pinv(latent_up)", up_mapping="half-target local Jacobian pseudoinverse",
                                    cov_tokens=str(args.cov_tokens), cov_token_count=n_tokens, cov_rank=args.cov_rank,
                                    cov_counts=counts, identity_fallback=unseen, lambda_g=args.lambda_g, ridge=args.ridge,
                                    normalize=args.normalize, normalization=normalization, repair=args.repair,
                                    repair_sweeps=args.repair_sweeps, continuous_eval=continuous, timing_wall_seconds=timing)
        for threshold, entries in sweeps.items():
            start = time.monotonic()
            entries = H.cap_budget(entries, args.budget)
            nnz = sum(len(e[0]) for e in entries.values())
            H.install_codes(model, entries)
            path = args.out / f"{args.name}_t{threshold:g}.pt"
            size = E.save_patch(model, path, dict(meta, masked_solve=dict(meta["masked_solve"], threshold=threshold,
                                nnz=nnz, round_repair_seconds=threshold_seconds[threshold])))
            print(f"thr={threshold:g} nnz={nnz} bytes={size} patch={path} "
                  f"round_repair_seconds={threshold_seconds[threshold]:.3f} save_seconds={time.monotonic() - start:.3f}", flush=True)
    finally:
        uninstall(model)


if __name__ == "__main__":
    main()
