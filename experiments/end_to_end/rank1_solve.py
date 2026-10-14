"""C53/S2: subject-key rank-one writes, constrained inside the pair8 forced mask.

One subject key and one optimized output correction per routed expert/layer.
Keys are averaged over TRAIN contexts, at the subject's last token. By default
up_proj reads the expert input; --projection down instead reads the activated
hidden key used by a conventional ROME value write. Joint output corrections
are divided by the number of selected layers during target optimization.

For EACH output row's writable columns S, G=(C_SS+ridge I)^-1, and
q=G k_S - G O_S.T (O_S G O_S.T)^+ O_S G k_S. The row write is
v_row q.T/(k_S.T q). Thus the continuous write realizes v at the mean subject
key and annihilates the other-person mean keys, within the actual mask. A mask
can make the constraints infeasible: those rows are zero, counted explicitly.
Masking a dense rank-one write after the solve would not have these properties.

C is the uncentered routed second moment, from exactly 10k non-eval FineWeb
tokens by default. Full fp32 moments are the default; --cov-rank > 0 explicitly
selects a Nystrom + diagonal approximation. Spectrum reports label that choice
and contain exact eigenvalues of deterministic sampled row restrictions of C
(before ridge), not a claim about every mask row. Both continuous (fmt=bf16,
unquantized harness codes) and row-alpha error-diffused ternary FSA files are
written, with empty entries for untouched experts/projections for harness IO.

C58 controls: --unconstrained removes both constraints and exports the dense
rank-one delta as unquantized LoRA factors (the harness's bf16 load path).
--recover-patch fits that FSA patch's own per-position output deltas on its
nonzero support, using subject and teacher-forced answer prediction positions.
Recovery is a multi-key least-squares solve, not a mean-key rank-one edit.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from . import e2e_fsa as E
    from . import hebbian_write as H
    from .token_streams import OverlapGuard, read_tokens
except ImportError:
    import e2e_fsa as E
    import hebbian_write as H
    from token_streams import OverlapGuard, read_tokens


class ExactMoment:
    """Full fp32 CPU running sum; no fp16 moment rounding."""
    def __init__(self, dimension):
        self.count = 0
        self.dimension = dimension
        self.sum = torch.zeros(dimension, dimension)

    @torch.no_grad()
    def add(self, x):
        x = x.detach().float()
        self.sum.add_((x.T @ x).cpu())
        self.count += len(x)


def moment_parts(cov, device="cpu"):
    if not cov.count:
        raise ValueError("subject-routed expert has no general-text covariance; no identity fallback")
    if isinstance(cov, ExactMoment):
        full = cov.sum.to(device) / cov.count
        return full.diagonal(), None, (full + full.T) * .5
    solver = cov.solver(device=device)
    ridge = max(1e-3 * float((cov.diagonal / cov.count).mean()), 1e-8)
    return (solver.diagonal - ridge).clamp_min(0), solver.factor, None


def subject_items(tok, facts, templates, bos):
    """Offsets come from the SAME complete prompt encoding, including merges."""
    items = E.build_items(tok, facts, templates, bos)
    for it in items:
        f = facts[it["fi"]]
        text = E.fmt(templates[it["ti"]], f)
        text = text[:text.rindex(f.val)].rstrip()
        start = text.index(f.name)
        end = start + len(f.name)
        encoded = tok.encode(text, add_special_tokens=False)
        if [bos] + list(encoded.ids) != it["p"]:
            raise ValueError("subject offsets and harness prompt encoding disagree")
        offsets = getattr(encoded, "offsets", None)
        if offsets is None:
            raise ValueError("tokenizer must supply character offsets for subject alignment")
        positions = [i + 1 for i, (lo, hi) in enumerate(offsets) if lo < end and hi > start]
        if not positions:
            raise ValueError("subject mention has no token span")
        it.update(subject_position=positions[-1], pid=f.pid)
        # Include the clean-stop token in the target objective, as ROME edits
        # otherwise can raise the value while failing the harness GEN metric.
        it["a"] = it["a"] + tok.encode(".", add_special_tokens=False).ids
    return items


class SubjectCapture(H.RoutedCapture):
    """Reuse the read-only grouped dispatch join, including tail-padding checks."""
    def __init__(self, model, layers, projection, cov_rank=0, seed=0):
        super().__init__(model, layers, (projection,) if isinstance(projection, str) else projection, cov_rank, seed)
        self.mode = "keys"
        self.pid = 0
        self.keys = {}
        self.outputs = {}
        self.values = {}
        self.divisor = len(layers)
        self.active_names = None
        self.teacher = {}
        self.recovery_records = {}

    def subject_batch(self, item, mode="keys"):
        self.batch("facts", [item["subject_position"]])
        self.pid, self.mode = item["pid"], mode

    def _linear(self, li, ei, proj, inputs, output):
        plan = self.plans.get(li, {}).get(ei)
        if plan is None or proj in plan["seen"]:
            raise RuntimeError("projection hook does not match grouped dispatch")
        plan["seen"].add(proj)
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        if len(x) < plan["count"] or output.shape[:-1] != inputs[0].shape[:-1]:
            raise RuntimeError("projection token layout differs from grouped dispatch")
        rows = plan["rows"].to(x.device)
        name = self.names[li, ei, proj]
        if self.mode == "recovery":
            resident = self.teacher.get(name)
            if resident is None:
                return output
            # Apply the teacher on ALL routed tokens so later keys are its
            # actual inputs, including down_proj after the edited up_proj.
            linear = getattr(self.layers[li].experts[ei], proj)
            w = linear.weight.detach()
            # Match FSAParam's weight arithmetic exactly, including the base
            # dtype cast, while keeping the base tensor untouched. Returning
            # a patched GEMM avoids additive-hook rounding at expert outputs.
            patched_w = w + resident.delta_w().to(w.dtype)
            delta = (patched_w - w).float()
            patched_output = F.linear(inputs[0], patched_w, linear.bias)
            if rows.numel():
                self.recovery_records.setdefault(name, []).append(
                    (x[rows].detach().float().cpu(), F.linear(x[rows].float(), delta).detach().cpu()))
                self.keys.setdefault((name, 0), []).append(x[rows].detach().float().cpu())
            return patched_output
        if not rows.numel():
            return output
        if self.phase == "covariance":
            if self.active_names is not None and name not in self.active_names:
                return output
            if name not in self.covariances:
                self.covariances[name] = (H.Covariance(x.shape[-1], self.cov_rank, self.seed)
                                          if self.cov_rank else ExactMoment(x.shape[-1]))
            self.covariances[name].add(x[rows])
        elif self.mode == "keys":
            self.keys.setdefault((name, self.pid), []).append(x[rows].detach().float().cpu())
            if self.pid == 0:
                self.outputs.setdefault(name, []).append(
                    output.reshape(-1, output.shape[-1])[rows].detach().float().cpu())
        elif self.mode == "values" and name in self.values:
            # Corrections affect the actual dispatched subject rows only. Route
            # masses and the expert nonlinearity remain in the frozen graph.
            correction = torch.zeros_like(output).reshape(-1, output.shape[-1])
            correction = correction.index_copy(0, rows, (self.values[name] / self.divisor)
                                                .to(output).expand(len(rows), -1))
            return output + correction.reshape_as(output)
        return output

    def means(self):
        return {key: torch.cat(rows).mean(0) for key, rows in self.keys.items()}


@torch.no_grad()
def key_pass(model, capture, items, device):
    with H.frozen_base():
        for it in E.progress(items, "rank1:key_pass"):
            capture.subject_batch(it)
            E.logits_of(model, torch.tensor([it["p"]], device=device))


def optimize_values(model, capture, items, device, steps=10, lr=.05, batch=0,
                    regularization=1e-3, seed=0):
    """Optimize shared subject values jointly across facts and all four layers.

    Values are additive corrections to base expert outputs, initially zero.
    Best target is chosen using TRAIN answer CE only, never held-out GEN.
    """
    if not items or steps < 1 or lr <= 0:
        raise ValueError("target solve needs nonempty items, positive steps and learning rate")
    capture.values = {name: torch.nn.Parameter(torch.zeros(rows[0].shape[-1], device=device))
                      for name, rows in capture.outputs.items()}
    if not capture.values:
        raise ValueError("no routed subject experts were captured")
    opt = torch.optim.Adam(list(capture.values.values()), lr=lr)
    rng = random.Random(seed)
    history, best_loss, best = [], math.inf, None
    # Evaluate on the same TRAIN panel every step so the zero-initialized
    # target remains a valid candidate and training loss cannot regress.
    def pass_items(selected, backward=False):
        total = 0.
        for it in E.progress(selected, "rank1:value_pass"):
            capture.subject_batch(it, "values")
            seq = it["p"] + it["a"]
            logits = E.logits_of(model, torch.tensor([seq[:-1]], device=device))
            pos = len(it["p"]) - 1
            loss = F.cross_entropy(logits[0, pos:].float(), torch.tensor(it["a"], device=device))
            total += float(loss.detach())
            if backward:
                penalty = sum(v.square().mean() for v in capture.values.values()) * regularization
                ((loss + penalty) / len(selected)).backward()
        return total / len(selected)

    with H.frozen_base():
        for step in range(steps + 1):
            with torch.no_grad():
                loss = pass_items(items)
            history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best = {name: value.detach().clone() for name, value in capture.values.items()}
            print(f"VALUE-SOLVE step={step} train_ce={loss:.6g}", flush=True)
            if step == steps:
                break
            selected = rng.sample(items, min(batch, len(items))) if batch else items
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pass_items(selected, backward=True)
            opt.step()
    capture.values = {}
    return {name: v.cpu() / capture.divisor for name, v in best.items()}, history


@torch.no_grad()
def target_attainment(model, capture, items, device, targets=None, recovery=False):
    """TRAIN-only answer probabilities with direct expert-output injection."""
    old_values, old_divisor = capture.values, capture.divisor
    capture.values, capture.divisor = targets or {}, 1
    results = {}
    try:
        with H.frozen_base():
            for label in ("base", "injected"):
                logps, probs, first = [], [], []
                for it in E.progress(items, "rank1:attainment"):
                    if recovery and label == "injected":
                        seq = it["p"] + it["a"]
                        positions = sorted(set([it["subject_position"]] +
                            list(range(len(it["p"]) - 1, len(seq) - 1))))
                        capture.batch("facts", positions)
                        capture.mode = "recovery"
                    else:
                        capture.subject_batch(it, "values" if label == "injected" else "base")
                    seq = it["p"] + it["a"]
                    logits = E.logits_of(model, torch.tensor([seq[:-1]], device=device))
                    lp = logits[0, len(it["p"]) - 1:].float().log_softmax(-1)
                    lp = lp.gather(1, torch.tensor(it["a"], device=device)[:, None]).squeeze(1)
                    logps.extend(lp.tolist())
                    probs.extend(lp.exp().tolist())
                    first.append(float(lp[0].exp()))
                ce = -sum(logps) / len(logps)
                results[label] = dict(answer_ce=ce, answer_token_probability=sum(probs) / len(probs),
                                      first_answer_probability=sum(first) / len(first),
                                      geometric_answer_token_probability=math.exp(-ce), tokens=len(logps))
        results["answer_ce_reduction"] = results["base"]["answer_ce"] - results["injected"]["answer_ce"]
        results["raises_answer_probability"] = results["answer_ce_reduction"] > 0
        print("TARGET-ATTAINMENT " + json.dumps(results), flush=True)
        return results
    finally:
        capture.values, capture.divisor = old_values, old_divisor
        capture.mode = "base"


@torch.no_grad()
def dense_rank_one(key, value, diagonal, factor=None, full=None, ridge=1e-3):
    """Unmasked, unprojected ROME write, stored as B @ A without row solves."""
    if key.ndim != 1 or value.ndim != 1 or diagonal.shape != key.shape:
        raise ValueError("dense key/value/covariance dimensions differ")
    if (not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(diagonal).all()) or
            bool((diagonal < 0).any())):
        raise ValueError("finite values and nonnegative finite covariance required")
    if ridge <= 0 or not math.isfinite(ridge) or not bool(torch.isfinite(key).all()):
        raise ValueError("finite key and positive ridge required")
    trace = full.diagonal() if full is not None else diagonal + (factor.square().sum(1) if factor is not None else 0)
    ridge_abs = max(ridge * float(trace.mean()), 1e-8)
    cols = torch.arange(len(key), device=key.device)[None, :]
    q = _restricted_inverse(key[None, :, None], cols, torch.ones_like(cols, dtype=torch.bool),
                            diagonal, factor, full, ridge_abs)[0, :, 0]
    denominator = key @ q
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("subject key has no finite positive whitened norm")
    return (q / denominator)[None, :], value[:, None], ridge_abs


@torch.no_grad()
def key_diagnostics(key, others, diagonal, factor=None, full=None, ridge=1e-3):
    keys = torch.cat((key[None, :], others))
    cos = F.normalize(keys, dim=1) @ F.normalize(keys, dim=1).T
    trace = full.diagonal() if full is not None else diagonal + (factor.square().sum(1) if factor is not None else 0)
    ridge_abs = max(ridge * float(trace.mean()), 1e-8)
    cols = torch.arange(len(key), device=key.device)[None, :]
    inv = _restricted_inverse(keys.T[None], cols, torch.ones_like(cols, dtype=torch.bool),
                              diagonal, factor, full, ridge_abs)[0]
    gram = keys @ inv
    eig = torch.linalg.eigvalsh((gram.double() + gram.double().T) * .5)
    cutoff = max(float(eig.abs().max()) * 1e-6, 1e-12)
    positive = eig[eig > cutoff]
    return dict(subject_other_max_abs_cosine=float(cos[0, 1:].abs().max()) if len(others) else 0.,
                whitened_key_gram_eigenvalues=eig.tolist(), whitened_key_rank=len(positive),
                whitened_key_condition=float(positive[-1] / positive[0]) if len(positive) else None)


@torch.no_grad()
def recover_support(records, mask, diagonal, factor=None, full=None, ridge=1e-3, row_batch=8):
    """Minimize per-position residual, then covariance cost, on exact support.

    Whiten each row's restricted design with C_SS+ridge I, use its SVD
    pseudoinverse, and unwhiten. No off-entity projection or target averaging:
    the original patch is a feasible solution to these multi-key equations.
    """
    if not records or not math.isfinite(ridge) or ridge <= 0 or row_batch <= 0:
        raise ValueError("recovery needs records and positive ridge/row_batch")
    keys = torch.cat([k for k, _ in records]).to(diagonal)
    values = torch.cat([v for _, v in records]).to(diagonal)
    if mask.shape != (values.shape[1], keys.shape[1]):
        raise ValueError("recovery key/value/support shapes differ")
    if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("recovery keys and values must be finite")
    trace = full.diagonal() if full is not None else diagonal + (factor.square().sum(1) if factor is not None else 0)
    ridge_abs = max(ridge * float(trace.mean()), 1e-8)
    delta = keys.new_zeros(mask.shape)
    ranks, conditions = [], []
    live_rows = mask.any(1).nonzero().flatten()
    for start in range(0, len(live_rows), row_batch):
        rr = live_rows[start:start + row_batch]
        selected = mask[rr]
        counts = selected.sum(1)
        width = int(counts.max())
        valid = torch.arange(width, device=keys.device)[None, :] < counts[:, None]
        cols = torch.zeros(len(rr), width, device=keys.device, dtype=torch.long)
        r, c = selected.nonzero(as_tuple=True)
        within = selected.long().cumsum(1)[r, c] - 1
        cols[r, within] = c
        cov = (full[cols[:, :, None], cols[:, None, :]] if full is not None else
               torch.diag_embed(diagonal[cols]) +
               (factor[cols] @ factor[cols].transpose(1, 2) if factor is not None else 0))
        cov = cov * valid[:, :, None] * valid[:, None, :] + ridge_abs * torch.eye(width, device=keys.device)
        chol = torch.linalg.cholesky(cov)
        design = keys[:, cols].permute(1, 0, 2) * valid[:, None, :]
        white = torch.linalg.solve_triangular(chol, design.transpose(1, 2), upper=False).transpose(1, 2)
        u, s, vh = torch.linalg.svd(white, full_matrices=False)
        good = s > (s[:, :1] * 1e-6).clamp_min(1e-12)
        z = vh.transpose(1, 2) @ ((u.transpose(1, 2) @ values[:, rr].T[:, :, None]) /
                                 torch.where(good, s, torch.ones_like(s))[:, :, None] * good[:, :, None])
        weights = torch.linalg.solve_triangular(chol.transpose(1, 2), z, upper=True).squeeze(-1)
        delta[rr[r], c] = weights[r, within]
        ranks.extend(good.sum(1).tolist())
        conditions.extend([float(s[i, 0] / s[i, good[i]][-1]) if bool(good[i].any()) else None
                           for i in range(len(rr))])
    residual = keys @ delta.T - values
    norm = float(values.norm())
    return delta, dict(ridge_absolute=ridge_abs, positions=len(keys), support_nnz=int(mask.sum()),
                       active_rows=len(live_rows), design_rank_min=min(ranks) if ranks else 0,
                       design_rank_max=max(ranks) if ranks else 0,
                       design_condition_max=max((c for c in conditions if c is not None), default=None),
                       output_delta_max_error=float(residual.abs().max()),
                       output_delta_relative_error=float(residual.norm()) / norm if norm else 0.)


def _restricted_inverse(rhs, cols, valid, diagonal, factor, full, ridge):
    if full is not None:
        c = full[cols[:, :, None], cols[:, None, :]] * valid[:, :, None] * valid[:, None, :]
        c = c + ridge * torch.eye(cols.shape[1], device=c.device, dtype=c.dtype)
        return torch.linalg.solve(c, rhs)
    d = diagonal[cols] * valid + ridge
    z = rhs / d[:, :, None]
    if factor is None or not factor.shape[1]:
        return z
    u = factor[cols] * valid[:, :, None]
    small = torch.eye(u.shape[-1], device=u.device, dtype=u.dtype) + u.transpose(1, 2) @ (u / d[:, :, None])
    return z - (u @ torch.linalg.solve(small, u.transpose(1, 2) @ z)) / d[:, :, None]


@torch.no_grad()
def constrained_rank_one(key, value, others, mask, diagonal, factor=None, full=None,
                         ridge=1e-3, row_batch=8, tolerance=1e-6):
    """Minimum covariance-cost interpolation with hard off-entity constraints."""
    if key.ndim != 1 or value.ndim != 1 or mask.shape != (len(value), len(key)):
        raise ValueError("key/value/mask shapes differ")
    if others.ndim != 2 or others.shape[1] != len(key) or diagonal.shape != key.shape:
        raise ValueError("other keys/covariance dimensions differ")
    if full is not None and full.shape != (len(key), len(key)):
        raise ValueError("full covariance dimensions differ")
    if factor is not None and (factor.ndim != 2 or factor.shape[0] != len(key)):
        raise ValueError("covariance factor dimensions differ")
    if not math.isfinite(ridge) or ridge <= 0 or row_batch <= 0 or tolerance <= 0:
        raise ValueError("ridge/row_batch/tolerance must be positive")
    if any(not bool(torch.isfinite(v).all()) for v in (key, value, others, diagonal)) or bool((diagonal < 0).any()):
        raise ValueError("finite inputs and nonnegative covariance diagonal required")
    if any(v is not None and not bool(torch.isfinite(v).all()) for v in (factor, full)):
        raise ValueError("covariance must be finite")
    result = key.new_zeros(mask.shape)
    failed = 0
    pre_projection, post_projection = 0., 0.
    pre_euclidean, post_euclidean = 0., 0.
    trace_diagonal = (full.diagonal() if full is not None else
                      diagonal + (factor.square().sum(1) if factor is not None else 0))
    ridge_abs = max(ridge * float(trace_diagonal.mean()), 1e-8)
    for start in range(0, len(mask), row_batch):
        selected = mask[start:start + row_batch].bool()
        counts = selected.sum(1)
        width = int(counts.max())
        if not width:
            failed += len(selected)
            continue
        cols = torch.zeros(len(selected), width, dtype=torch.long, device=key.device)
        valid = torch.arange(width, device=key.device)[None, :] < counts[:, None]
        rr, cc = selected.nonzero(as_tuple=True)
        within = selected.long().cumsum(1)[rr, cc] - 1
        cols[rr, within] = cc
        k = key[cols] * valid
        o = others[:, cols].permute(1, 2, 0) * valid[:, :, None]
        inv = _restricted_inverse(torch.cat((k[:, :, None], o), -1), cols, valid,
                                  diagonal, factor, full, ridge_abs)
        q, go = inv[:, :, 0], inv[:, :, 1:]
        row_values = value[start:start + len(selected), None]
        pre_euclidean += float((q * row_values).square().sum())
        if len(others):
            gram = o.transpose(1, 2) @ go
            gram = (gram + gram.transpose(1, 2)) * .5
            q = q - (go @ (torch.linalg.pinv(gram, hermitian=True, rtol=tolerance)
                           @ (o.transpose(1, 2) @ q[:, :, None]))).squeeze(-1)
        denominator = (k * q).sum(1)
        pre_projection += float((k * inv[:, :, 0]).sum().clamp_min(0))
        good = denominator > tolerance * (k * inv[:, :, 0]).sum(1).abs().clamp_min(1e-12)
        direction = q / torch.where(good, denominator, torch.ones_like(denominator))[:, None]
        # Treat numerically unresolved null constraints as infeasible, rather
        # than exporting a huge or leaking direction on nearly collinear keys.
        if len(others):
            leak = (o.transpose(1, 2) @ direction[:, :, None]).squeeze(-1).abs().amax(1)
            good &= leak <= 10 * tolerance
        failed += int((~good).sum())
        post_projection += float((denominator * good).clamp_min(0).sum())
        post_euclidean += float((q * row_values * good[:, None]).square().sum())
        w = direction * value[start:start + len(selected), None] * good[:, None]
        result[start + rr, cc] = w[rr, within]
    return result, dict(rows=len(mask), infeasible_rows=failed, ridge_absolute=ridge_abs,
                        projection_covariance_norm_fraction=math.sqrt(post_projection / pre_projection)
                        if pre_projection else 0.,
                        projection_direction_norm_fraction=math.sqrt(post_euclidean / pre_euclidean)
                        if pre_euclidean else 0.,
                        subject_max_error=float((result @ key - value).abs().max()) if len(value) else 0.,
                        other_max_response=float((result @ others.T).abs().max()) if others.numel() else 0.)


@torch.no_grad()
def error_diffusion(weights, mask, alpha):
    """Diffuse quantization error along writable columns in each physical row.

    q_j=clip(round(w_j/alpha + carry),-1,1); carry += w_j/alpha-q_j.
    Forbidden columns never receive a code or consume carry. Alpha stays the
    shipped mean nonzero absolute row weight. No fitted/learned patch scales.
    """
    alpha = alpha.flatten().to(weights)
    if weights.ndim != 2 or mask.shape != weights.shape or len(alpha) != len(weights):
        raise ValueError("weights/mask/alpha shapes differ")
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(alpha).all()) or bool((alpha < 0).any()):
        raise ValueError("finite weights and nonnegative row alpha required")
    q = torch.zeros_like(weights, dtype=torch.int8)
    carry = weights.new_zeros(len(weights))
    scale = torch.where(alpha > 0, alpha, torch.ones_like(alpha))
    for col in range(weights.shape[1]):
        live = mask[:, col].bool() & (alpha > 0)
        z = weights[:, col] / scale + carry
        code = z.round().clamp(-1, 1)
        q[:, col] = torch.where(live, code, 0).to(torch.int8)
        carry = torch.where(live, z - code, carry)
    return q


@torch.no_grad()
def restricted_spectrum(mask, diagonal, factor=None, full=None, rows=3):
    """Exact eigenspectrum for evenly spaced sampled row masks, before ridge."""
    if rows <= 0:
        raise ValueError("spectrum rows must be positive")
    chosen = torch.linspace(0, len(mask) - 1, min(rows, len(mask))).round().long().tolist()
    result = []
    for row in chosen:
        cols = mask[row].nonzero().flatten().to(diagonal.device)
        c = (full[cols[:, None], cols[None, :]] if full is not None else
             torch.diag(diagonal[cols]) + (factor[cols] @ factor[cols].T if factor is not None else 0))
        eig = torch.linalg.eigvalsh((c.double() + c.double().T) * .5).cpu()
        cutoff = max(float(eig.abs().max()) * 1e-6, 1e-12) if len(eig) else 1e-12
        positive = eig[eig > cutoff]
        result.append(dict(row=row, writable_columns=len(cols), eigenvalues=eig.tolist(),
                           numerical_rank=len(positive), cutoff=cutoff,
                           condition_positive=float(positive[-1] / positive[0]) if len(positive) else None))
    return result


def checked_covariance_tokens(source, panels, count=10_000):
    if count < 32 or not panels:
        raise ValueError("need at least 32 covariance tokens and eval panels for disjointness")
    source = Path(source)
    if source.resolve() in {Path(p).resolve() for p in panels} or source.name.startswith("eval-"):
        raise ValueError("covariance source must not be an evaluation panel")
    tokens = read_tokens(source)
    if len(tokens) < count:
        raise ValueError(f"covariance source has {len(tokens)} tokens; need {count}")
    tokens = tokens[:count]
    references = {str(p): read_tokens(p) for p in panels}
    if any(len(t) < 32 for t in references.values()):
        raise ValueError("each eval panel needs at least 32 tokens for disjointness")
    # Every 32-gram on both sides, not the harness's stride-16 sample or 1% allowance.
    overlap = OverlapGuard(references, reference_stride=1).check([tokens], str(source), limit=0.)
    return tokens.tolist(), overlap


def patch_metadata(a, layers, fmt):
    defaults = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused"])
    keys = ("train_person", "train_attr_group", "attr_groups", "slot_partition", "slot_part", "partition_unit",
            "units_per_fact", "unit_init", "address_center", "exclude_general", "entropy_units", "address_mode",
            "address_units", "rank", "n_experts", "fp4_step", "patch_scale", "scale_block", "shared_expert",
            "shared_only", "occupied_flips", "slot_frac", "patch_scale_mult", "no_pareto")
    meta = {k: getattr(defaults, k) for k in keys}
    meta.update(eda_backend=E.eda.selected(a), method="fsa_forced", fmt=fmt, layers=layers,
                projs=["up_proj", "down_proj"], seed=a.seed, selection=None, scale_block=0,
                people=1, facts_per_person=40, shared_attrs=0, values="disjoint", relation_world=False,
                budget=0, sparse_state=True, base_altering=False, calibration=None,
                export=str(a.export), export_step=a.export_step, step=0,
                alpha="mean |nonzero base weight| of the row")
    meta.update({k: None for k in ("fold_mode", "fold_every", "fold_thr", "fold_mask", "fold_mask_frac")})
    if a.unconstrained:
        meta.update(method="lora", rank=1, sparse_state=False)
    return meta


def entry(shape, alpha, values=None, fmt="ternary"):
    if values is None:
        idx = torch.empty(0, dtype=torch.int32)
        val = torch.empty(0, dtype=torch.int8 if fmt == "ternary" else torch.float32)
    else:
        idx = values.flatten().nonzero().flatten()
        val = values.flatten()[idx].to(torch.int8 if fmt == "ternary" else torch.float32).cpu()
        idx = idx.to(torch.int32).cpu()
    return dict(kind="fsa", fmt=fmt, shape=list(shape), idx=idx, val=val, alpha=alpha.float().cpu(),
                fp4_step=.25, scale="row")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    E.eda.add_argument(ap)
    ap.add_argument("--export", type=Path, default=E.EXPORT)
    ap.add_argument("--export-step", type=int, default=E.EXPORT_STEP)
    ap.add_argument("--data", type=Path, default=E.DATA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", help="default 12,14,17,20; source patch layers for recovery")
    controls = ap.add_mutually_exclusive_group()
    controls.add_argument("--unconstrained", action="store_true", help="dense ROME: no mask or other-key projection")
    controls.add_argument("--recover-patch", type=Path, help="recover an ordinary FSA patch on its exact nonzero support")
    ap.add_argument("--projection", choices=("up", "down"), default="up")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--cov-tokens", type=Path)
    ap.add_argument("--eval-panels", type=Path, nargs="+")
    ap.add_argument("--cov-max-tokens", type=int, default=10_000)
    ap.add_argument("--cov-window", type=int, default=256)
    ap.add_argument("--cov-rank", type=int, default=0, help="0 = full fp32 C; >0 = labeled Nystrom approximation")
    ap.add_argument("--ridge", type=float, default=1e-3, help="relative to mean diagonal of C")
    ap.add_argument("--v-steps", type=int, default=10)
    ap.add_argument("--v-lr", type=float, default=.05)
    ap.add_argument("--v-batch", type=int, default=0, help="0 = all TRAIN items per gradient step")
    ap.add_argument("--v-l2", type=float, default=1e-3)
    ap.add_argument("--row-batch", type=int, default=8)
    ap.add_argument("--spectrum-rows", type=int, default=3)
    ap.add_argument("--solve-device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True, help="directory for continuous.pt, ternary.pt, solve.json")
    return ap


def validate_args(ap, a):
    a.layers = a.layers or "12,14,17,20"
    try:
        layers = [int(x) for x in a.layers.split(",")]
    except ValueError:
        ap.error("layers must be comma-separated integer MoE indices")
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        ap.error("layers must be distinct nonnegative MoE indices")
    if min(a.v_steps, a.row_batch, a.spectrum_rows, a.cov_window) <= 0 or a.cov_max_tokens < 32:
        ap.error("positive step/batch/window counts and at least 32 covariance tokens required")
    if min(a.v_batch, a.cov_rank, a.export_step) < 0:
        ap.error("v-batch/cov-rank/export-step must be nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (a.ridge, a.v_lr)) or not math.isfinite(a.v_l2) or a.v_l2 < 0:
        ap.error("finite positive ridge/v-lr and finite nonnegative v-l2 required")
    a.cov_tokens = a.cov_tokens or a.data / "rehearsal-fineweb.json"
    a.eval_panels = a.eval_panels or [a.data / name for name in
        ("eval-wikitext.json", "eval-fineweb.json", "eval-code.json", "heldout-tokens.json")]
    return layers


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    source = None
    if a.recover_patch:
        source = torch.load(a.recover_patch, map_location="cpu", weights_only=False)
        if source.get("format") != "e2e-fsa-patch-v1" or not source.get("entries"):
            ap.error("recovery requires an e2e FSA patch with entries")
        meta = source["meta"]
        if (meta.get("method") != "fsa_forced" or meta.get("people") != 1 or
                meta.get("facts_per_person") != 40 or meta.get("shared_attrs", 0) != 0 or
                meta.get("values", "disjoint") != "disjoint" or meta.get("relation_world", False) or
                meta.get("seed") != a.seed or meta.get("train_person") is not None or meta.get("c37_facts")):
            ap.error("recovery requires the same seed, single-person disjoint 40-fact FSA-forced world")
        source_layers = ",".join(map(str, meta["layers"]))
        if a.layers is not None and a.layers != source_layers:
            ap.error("recovery layers must match source patch layers")
        a.layers = source_layers
        a.aug = a.aug or meta.get("training_recipe", {}).get("aug", meta.get("aug", False))
        E.validate_patch_export(meta, a.export, a.export_step)
        E.eda.require_patch_backend(source, a, "recovery patch")
        for ent in source["entries"].values():
            if (ent.get("kind") != "fsa" or ent.get("scale") != "row" or
                    ent.get("flip_idx", torch.empty(0)).numel() or
                    any(k.startswith("unit_reset") for k in ent)):
                ap.error("recovery supports ordinary row-scaled FSA entries without flips or base resets")
    layers = validate_args(ap, a)
    tokens, overlap = checked_covariance_tokens(a.cov_tokens, a.eval_panels, a.cov_max_tokens)
    if not torch.cuda.is_available():
        ap.error("rank-one screen requires the CUDA model host and /workspace/malleable/mlf")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    try:
        from .fast_moe import install, uninstall
    except ImportError:
        from fast_moe import install, uninstall
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    E.eda.announce(a)
    tok = Tokenizer.from_file(str(a.data / "tokenizer.json"))
    model, _, dev = E.eda.build_model(build_model, a, a.export, device="cuda:0", model_config=None,
                                    step=a.export_step, expert_format="bf16")
    model.eval().requires_grad_(False)
    if max(layers) >= len(E.moe_layers(model)):
        ap.error("layer index outside model's MoE layers")
    facts = E.make_world(a.seed, 1, 40)
    templates = E.TRAIN_AUG_T if a.aug else E.TRAIN_T
    items = subject_items(tok, facts, templates, E.BOS)
    projection = ("up_proj", "down_proj") if source else a.projection + "_proj"
    install(model, patched_layer_ids=layers, mode="exact")
    try:
        with SubjectCapture(model, layers, projection, a.cov_rank, a.seed) as capture:
            if source:
                linears = dict(model.named_modules())
                expected = set(capture.names.values())
                if set(source["entries"]) != expected:
                    raise ValueError("recovery patch must include both projections of every selected routed expert")
                for name, ent in source["entries"].items():
                    w = linears[name].weight.detach()
                    resident = E.ResidentParam(w, ent)
                    if resident.idx.numel():
                        mask = E.pair8_forced_mask(w)
                        if not bool(mask.flatten()[resident.idx].all()):
                            raise ValueError(f"recovery source writes outside forced-zero support: {name}")
                        capture.teacher[name] = resident
                if not capture.teacher:
                    raise ValueError("recovery source patch has no nonzero delta")
                attainment = target_attainment(model, capture, items, dev, recovery=True)
                targets = {name: torch.cat([v for _, v in visits]) for name, visits in capture.recovery_records.items()}
                history = []
                capture.active_names = set(targets)
            else:
                key_pass(model, capture, items, dev)
                # Same contexts with only the named person/pronouns replaced.
                # Preserve the exact seed-0 40-fact target world.
                for pid, (name, pos, subj) in enumerate(E.PEOPLE[1:], 1):
                    others = [f._replace(pid=pid, name=name, pos=pos, subj=subj) for f in facts]
                    key_pass(model, capture, subject_items(tok, others, templates, E.BOS), dev)
                capture.active_names = set(capture.outputs)
            means = capture.means()
            H.covariance_pass(model, capture, tokens, dev, a.cov_max_tokens, a.cov_window, E.BOS)
            if not source:
                targets, history = optimize_values(model, capture, items, dev, a.v_steps, a.v_lr,
                                                   a.v_batch, a.v_l2, a.seed)
                attainment = target_attainment(model, capture, items, dev, targets)
    finally:
        uninstall(model)
    states = {fmt: dict(format="e2e-fsa-patch-v1", meta=patch_metadata(a, layers, fmt), entries={})
              for fmt in (("bf16",) if a.unconstrained else ("bf16", "ternary"))}
    diagnostics = {}
    for li in layers:
        layer_name, moe = E.moe_layers(model)[li]
        for ei, expert in enumerate(moe.experts):
            if expert is None:
                continue
            for proj in ("up_proj", "down_proj"):
                name = f"{layer_name}.experts.{ei}.{proj}"
                w = getattr(expert, proj).weight.detach()
                nz = w.ne(0)
                alpha = (w.float().abs() * nz).sum(1) / nz.sum(1).clamp_min(1)
                for fmt, state in states.items():
                    state["entries"][name] = (dict(kind="lora", A=torch.zeros(1, w.shape[1]),
                        B=torch.zeros(w.shape[0], 1)) if a.unconstrained else entry(w.shape, alpha, fmt=fmt))
                if name not in targets:
                    continue
                cov = capture.covariances.pop(name, None)
                if cov is None:
                    raise ValueError(f"no covariance for subject expert {name}; use more training tokens")
                diagonal, factor, full = moment_parts(cov, a.solve_device)
                mask = E.pair8_forced_mask(w).to(a.solve_device)
                key = means[name, 0].to(a.solve_device)
                other_keys = [means[name, pid] for pid in range(1, len(E.PEOPLE)) if (name, pid) in means]
                others = torch.stack(other_keys).to(key) if other_keys else key.new_empty(0, len(key))
                if source:
                    mask.zero_()
                    mask.flatten()[capture.teacher[name].idx.to(mask.device)] = True
                    delta, diag = recover_support(capture.recovery_records[name], mask, diagonal,
                                                  factor, full, a.ridge, a.row_batch)
                    diag.update(mask_update_norm_fraction=1., projection_covariance_norm_fraction=1.,
                                projection_direction_norm_fraction=1., off_entity_projection=False)
                else:
                    A, B, ridge_abs = dense_rank_one(key, targets[name].to(key), diagonal, factor, full, a.ridge)
                    dense = B @ A
                    collision = key_diagnostics(key, others, diagonal, factor, full, a.ridge)
                    if a.unconstrained:
                        delta = dense
                        diag = dict(rows=len(w), infeasible_rows=0, ridge_absolute=ridge_abs,
                                    subject_max_error=float((delta @ key - targets[name].to(key)).abs().max()),
                                    other_max_response=float((delta @ others.T).abs().max()) if others.numel() else 0.,
                                    mask_update_norm_fraction=1., projection_covariance_norm_fraction=1.,
                                    projection_direction_norm_fraction=1.)
                        states["bf16"]["entries"][name] = dict(kind="lora", A=A.cpu(), B=B.cpu())
                    else:
                        delta, diag = constrained_rank_one(key, targets[name].to(key), others, mask,
                                                           diagonal, factor, full, a.ridge, a.row_batch)
                        norm = float(dense.norm())
                        diag["mask_update_norm_fraction"] = float(dense.masked_fill(~mask, 0).norm()) / norm if norm else 0.
                    diag.update(collision, forced_mask_dense_norm_fraction=float(dense.masked_fill(~mask, 0).norm()) /
                                float(dense.norm()) if bool(dense.any()) else 0.)
                    del dense, A, B
                diag.update(covariance_visits=cov.count, other_people=len(other_keys),
                            spectrum=restricted_spectrum(torch.ones_like(mask) if a.unconstrained else mask,
                                                         diagonal, factor, full, 1 if a.unconstrained else a.spectrum_rows))
                for sample in diag["spectrum"]:
                    eig = sample["eigenvalues"]
                    ridge_abs = diag["ridge_absolute"]
                    sample["condition_regularized"] = ((eig[-1] + ridge_abs) / (eig[0] + ridge_abs)
                        if eig and eig[0] + ridge_abs > 0 else None)
                if a.unconstrained:
                    diagnostics[name] = diag
                    print("RANK1 " + json.dumps(dict(name=name, **{k: v for k, v in diag.items() if k != "spectrum"})), flush=True)
                    del delta, diagonal, factor, full, cov, mask
                    continue
                q = error_diffusion(delta, mask, alpha.to(delta))
                scale = alpha.to(delta)
                safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                if bool((delta[scale == 0] != 0).any()):
                    raise ValueError(f"nonzero continuous write in zero-alpha row: {name}")
                states["bf16"]["entries"][name] = entry(w.shape, alpha, delta / safe_scale[:, None], "bf16")
                states["ternary"]["entries"][name] = entry(w.shape, alpha, q)
                rounded = q.to(delta) * scale[:, None]
                diag.update(ternary_nnz=int(q.ne(0).sum()),
                            ternary_subject_max_error=float((rounded @ key - targets[name].to(key)).abs().max()) if not source else None,
                            ternary_other_max_response=float((rounded @ others.T).abs().max()) if others.numel() else 0.)
                if source:
                    visits = capture.recovery_records[name]
                    ks = torch.cat([k for k, _ in visits]).to(delta)
                    vs = targets[name].to(delta)
                    diag["ternary_output_delta_max_error"] = float((ks @ rounded.T - vs).abs().max())
                diagnostics[name] = diag
                print("RANK1 " + json.dumps(dict(name=name, **{k: v for k, v in diag.items() if k != "spectrum"})), flush=True)
                del delta, q, rounded, diagonal, factor, full, cov, mask
    report = dict(eda_backend=E.eda.selected(a), seed=a.seed, facts=len(facts), layers=layers, projection=projection,
                  cov_source=str(a.cov_tokens), cov_tokens=len(tokens), overlap_32gram=overlap,
                  cov_source_sha256=E.file_sha(a.cov_tokens),
                  eval_panel_sha256={str(p): E.file_sha(p) for p in a.eval_panels},
                  cov_rank=a.cov_rank, covariance="full fp32" if not a.cov_rank else "Nystrom + diagonal approximation",
                  spectrum_scope=("one full-space spectrum per routed expert" if a.unconstrained else
                      f"{a.spectrum_rows} evenly spaced row supports per routed expert") + "; eigenvalues before ridge",
                  control="recover_patch" if source else "unconstrained" if a.unconstrained else "constrained",
                  recover_patch=str(a.recover_patch) if source else None,
                  recover_patch_sha256=E.file_sha(a.recover_patch) if source else None,
                  recovery_positions="subject last token + all answer prediction positions (including clean stop)" if source else None,
                  training_aug=a.aug, target_attainment=attainment, target_train_ce=history, experts=diagnostics,
                  interpretation="GEN and damage require harness evaluation; no verdict inferred from TRAIN CE")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(keys=means, targets=targets,
                    recovery_records=capture.recovery_records if source else None,
                    key_visits={key: sum(len(x) for x in visits) for key, visits in capture.keys.items()},
                    meta={k: v for k, v in report.items() if k != "experts"}), a.out / "keys_targets.pt")
    for fmt, state in states.items():
        state["meta"]["rank1"] = {k: v for k, v in report.items() if k != "experts"}
        path = a.out / ("continuous.pt" if fmt == "bf16" else "ternary.pt")
        torch.save(state, path)
        print(f"PATCH {path}", flush=True)
    (a.out / "solve.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
