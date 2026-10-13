r"""One-pass whitened Hebbian writes into the shipped pair8 forced mask.

No optimizer or differentiation through writes: each TRAIN prompt is evaluated
on the frozen base once, with teacher-forced answer log-probability gradients.
For each routed projection, delta = sum g (C^-1 k)^T. Output codes are plain
{-1, 0, +1}, at the harness's unchanged row alpha; disabling FSA restores base
weights exactly. Files are written by e2e_fsa.save_patch, including empty experts.

Covariance is the *uncentered* second moment of routed, non-BOS general-text
inputs. Default --cov-rank 64 uses a streaming Nystrom sketch Y=C Omega plus
the exact diagonal: C_hat=U U^T+diag(max(diag(C)-diag(U U^T),0)). Sketch sums
are fp32 CPU, O(d*r) per expert (~0.6 MB at d=2304, r=64, versus 21 MB for
full fp32). --cov-rank 0 stores full running moments in fp16 CPU (~10.6 MB),
with fp32 updates/solves. Only one expert's full solve/delta is live at a time.
Fact keys and gradients are retained as fp32 CPU records (O(answer route visits
* (d_in+d_out))); there is no dense delta bank. Ridge is lambda*tr(C)/d with
a numerical floor of 1e-8. Experts unseen in covariance use identity, reported
in the log and metadata. No covariance centering or routing-weight rescaling.

Density rho keeps floor(rho * writable_count) largest magnitudes *per row*,
excluding zero updates and zero-alpha rows. Optional --budget caps nonzeros
across all projections by keeping the largest surviving |delta|. Equal-score
ties use flat weight order (then module order for the global budget).

Example (on the CUDA model host):
  python experiments/end_to_end/hebbian_write.py --people 1 --facts-per-person 40 \
      --layers 14,15 --projs down --sparse-state --name smoke --out out
  python experiments/end_to_end/e2e_fsa.py --method fsa_forced --down-only \
      --layers 14,15 --sparse-state --load-patch out/smoke_d0.001.pt \
      --steps 0 --eval-every 0 --out out/smoke_eval.json
Use the same world/layers/seed/slot-frac for evaluation or Bop warm-start;
use --down-only only for --projs down. Up-only writes use the harness's both-
projection container with empty down entries, so omit --down-only to load them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch

try:
    from . import e2e_fsa as E
except ImportError:
    import e2e_fsa as E


class Covariance:
    """Streaming full moment or randomized rank-r + diagonal moment."""

    def __init__(self, dimension, rank=64, seed=0, omega=None):
        self.dimension = dimension
        self.count = 0
        self.rank = min(rank, dimension)
        if rank < 0:
            raise ValueError("covariance rank must be nonnegative")
        self.diagonal = torch.zeros(dimension)
        self.moment = None
        self.sketch = None
        self.omega = None
        if self.rank:
            if omega is None:
                gen = torch.Generator().manual_seed(seed + dimension)
                omega = torch.linalg.qr(torch.randn(dimension, self.rank, generator=gen), mode="reduced").Q
            self.omega = omega
            self.sketch = torch.zeros(dimension, self.rank)

    @torch.no_grad()
    def add(self, inputs, omega=None):
        x = inputs.detach().reshape(-1, self.dimension).float()
        n = len(x)
        if not n:
            return
        self.diagonal.add_(x.square().sum(0).cpu())
        if self.rank:
            omega = self.omega.to(x.device) if omega is None else omega
            self.sketch.add_((x.T @ (x @ omega)).cpu())
        else:
            # Keep a mean, not a fp16 sum which overflows on long streams.
            new = (x.T @ x).cpu() / (self.count + n)
            if self.moment is not None:
                new.add_(self.moment.float(), alpha=self.count / (self.count + n))
            self.moment = new.half()
        self.count += n

    def solver(self, ridge=1e-3, device="cpu"):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if not self.count:
            return MomentSolver(torch.ones(self.dimension, device=device), None, None, ridge)
        diag = self.diagonal.to(device) / self.count
        regularizer = max(ridge * float(diag.mean()), 1e-8)
        if not self.rank:
            c = self.moment.to(device).float()
            # Half rounding can make a PSD moment slightly indefinite.
            eigenvalues, eigenvectors = torch.linalg.eigh((c + c.T) * 0.5)
            c = (eigenvectors * eigenvalues.clamp_min(0)) @ eigenvectors.T
            return MomentSolver(diag, None, c, regularizer)
        y = self.sketch.to(device) / self.count
        omega = self.omega.to(device)
        small = omega.T @ y
        eigenvalues, eigenvectors = torch.linalg.eigh((small + small.T) * 0.5)
        cutoff = max(float(eigenvalues.max()) * 1e-6, 1e-12)
        keep = eigenvalues > cutoff
        u = (y @ eigenvectors[:, keep]) / eigenvalues[keep].sqrt()
        residual = (diag - u.square().sum(1)).clamp_min(0)
        return MomentSolver(residual, u, None, regularizer)


class MomentSolver:
    """Apply (moment + ridge I)^-1 to a batch of row keys."""

    def __init__(self, diagonal, factor, full, ridge):
        self.diagonal = diagonal + ridge
        self.factor = factor
        self.full = None
        self.small = None
        if full is not None:
            self.full = torch.linalg.cholesky(full + ridge * torch.eye(len(diagonal), device=full.device))
        elif factor is not None:
            small = torch.eye(factor.shape[1], device=factor.device) + factor.T @ (factor / self.diagonal[:, None])
            self.small = torch.linalg.cholesky(small)

    def apply(self, keys):
        k = keys.to(self.diagonal.device).float()
        if self.full is not None:
            return torch.cholesky_solve(k.T, self.full).T
        z = k / self.diagonal
        if self.factor is None or not self.factor.shape[1]:
            return z
        correction = torch.cholesky_solve((z @ self.factor).T, self.small).T
        return z - (correction @ self.factor.T) / self.diagonal


@contextmanager
def frozen_base():
    old = E.FSAParam.enabled, E.LoRAParam.enabled
    E.set_enabled(False)
    try:
        yield
    finally:
        E.FSAParam.enabled, E.LoRAParam.enabled = old


class RoutedCapture:
    """Join grouped-dispatch token IDs to projection inputs/output gradients.

    Resident fallback and fast_moe exact both call experts on ascending unique
    token IDs. Padding at the tail is allowed and is never recorded. Backends
    bypassing Linear hooks fail explicitly. Tensor hooks record d(log p)/d(out),
    which already includes routing mass; multiplying by it again is incorrect.
    """

    def __init__(self, model, layer_ids, projs, cov_rank=64, seed=0):
        layers = E.moe_layers(model)
        self.layers = {li: layers[li][1] for li in layer_ids}
        self.names = {}
        for li in layer_ids:
            for ei, expert in enumerate(layers[li][1].experts):
                if expert is not None:
                    for proj in projs:
                        self.names[li, ei, proj] = f"{layers[li][0]}.experts.{ei}.{proj}"
        self.projs = projs
        self.cov_rank, self.seed = cov_rank, seed
        self.covariances, self.records, self.omegas = {}, {}, {}
        self.cpu_omegas = {}
        self.plans = {}
        self.hooks, self.methods = [], []
        self.pending = 0

    def batch(self, phase, positions):
        if phase not in ("covariance", "facts"):
            raise ValueError("unknown capture phase")
        if self.pending or self.plans:
            raise RuntimeError("previous capture did not finish")
        self.phase = phase
        self.positions = torch.as_tensor(positions, dtype=torch.long).cpu()

    def _dispatch(self, li, x, indices, weights):
        if indices.ndim != 2 or indices.shape != weights.shape or len(x) != len(indices):
            raise RuntimeError("unsupported grouped-dispatch layout")
        ids = indices.detach().cpu()
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self.layers[li].experts)):
            raise RuntimeError("router expert ID outside resident bank")
        plans = {}
        for ei in ids.unique().tolist():
            tokens = (ids == ei).any(1).nonzero().flatten()
            selected = torch.isin(tokens, self.positions).nonzero().flatten()
            plans[ei] = dict(rows=selected, count=len(tokens), seen=set())
        self.plans[li] = plans

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
        if self.phase == "covariance":
            if rows.numel():
                if name not in self.covariances:
                    d = x.shape[-1]
                    cov = Covariance(d, self.cov_rank, self.seed, self.cpu_omegas.get(d))
                    self.covariances[name] = cov
                    self.cpu_omegas[d] = cov.omega
                cov = self.covariances[name]
                omega = None
                if cov.rank:
                    key = (x.shape[-1], x.device)
                    if key not in self.omegas:
                        self.omegas[key] = cov.omega.to(x.device)
                    omega = self.omegas[key]
                cov.add(x[rows], omega)
        else:
            # A leaf at the earliest patched projection starts autograd without
            # allocating any frozen weight/embedding gradients or patch grads.
            if not output.requires_grad:
                output.requires_grad_(True)
            if rows.numel():
                keys = x[rows].detach().float().cpu()
                self.pending += 1

                def record(grad):
                    values = grad.reshape(-1, grad.shape[-1])[rows].detach().float().cpu()
                    self.records.setdefault(name, []).append((keys, values))
                    self.pending -= 1

                output.register_hook(record)

    def _finish(self, li):
        plans = self.plans.pop(li, None)
        if plans is None or any(p["seen"] != set(self.projs) for p in plans.values()):
            raise RuntimeError("expert backend bypassed projection hooks; use exact resident dispatch")

    def __enter__(self):
        try:
            for li, moe in self.layers.items():
                dispatch = getattr(moe, "_forward_frozen_exact_grouped", None)
                if not callable(dispatch):
                    raise TypeError("Hebbian capture requires the resident grouped-dispatch interface")
                own = "_forward_frozen_exact_grouped" in moe.__dict__
                self.methods.append((moe, dispatch, own))

                def capture(layer, x, indices, weights, li=li, dispatch=dispatch):
                    self._dispatch(li, x, indices, weights)
                    return dispatch(x, indices, weights)

                moe._forward_frozen_exact_grouped = MethodType(capture, moe)
                self.hooks.append(moe.register_forward_hook(lambda mod, inp, out, li=li: self._finish(li)))
                for ei, expert in enumerate(moe.experts):
                    if expert is None:
                        continue
                    for proj in self.projs:
                        self.hooks.append(getattr(expert, proj).register_forward_hook(
                            lambda mod, inp, out, li=li, ei=ei, proj=proj: self._linear(li, ei, proj, inp, out)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for hook in self.hooks:
            hook.remove()
        for moe, dispatch, own in reversed(self.methods):
            if own:
                moe._forward_frozen_exact_grouped = dispatch
            else:
                del moe._forward_frozen_exact_grouped
        self.hooks.clear()
        self.methods.clear()


def covariance_pass(model, capture, tokens, device, max_tokens=100_000, window=256, bos=E.BOS):
    """Count at most max_tokens actual stream tokens, without synthetic padding."""
    tokens = tokens[:max_tokens]
    with frozen_base(), torch.no_grad():
        for start in range(0, len(tokens), window):
            chunk = tokens[start:start + window]
            capture.batch("covariance", range(1, len(chunk) + 1))
            E.logits_of(model, torch.tensor([[bos] + chunk], device=device))
    return len(tokens)


def fact_pass(model, capture, items, device):
    """Sum log probabilities of answer tokens only; one backward per phrasing."""
    total = 0.0
    with frozen_base(), torch.enable_grad():
        for i, item in enumerate(items):
            if not item["p"] or not item["a"]:
                raise ValueError("fact prompt and answer must be nonempty")
            seq = item["p"] + item["a"]
            positions = list(range(len(item["p"]) - 1, len(seq) - 1))
            capture.batch("facts", positions)
            logits = E.logits_of(model, torch.tensor([seq[:-1]], device=device))
            selected = logits[0, positions].float()
            gold = torch.tensor(item["a"], device=device)
            logp = selected.log_softmax(-1).gather(1, gold[:, None]).sum()
            logp.backward()
            if capture.pending:
                raise RuntimeError("answer output gradients were not captured")
            total += float(logp.detach())
            if (i + 1) % 100 == 0:
                print(f"facts: {i + 1}/{len(items)} phrasings", flush=True)
    return total


@torch.no_grad()
def delta_write(records, covariance, ridge=1e-3, device="cpu"):
    solver = covariance.solver(ridge, device)
    keys = torch.cat([key for key, _ in records])
    values = torch.cat([value for _, value in records]).to(device)
    # Batch all this expert's visits into one GEMM. Tiny outer products would
    # repeatedly allocate and write a full d_out*d_in contribution per phrasing.
    return values.T @ solver.apply(keys)


def writable_mask(pm):
    """Recover only the actual attached reserve, including seeded slot thinning."""
    if not getattr(pm, "sparse", False):
        return pm.mask.detach().cpu()
    mask = torch.zeros(math.prod(pm.wshape), dtype=torch.bool)
    mask[pm.idx.long().cpu()] = True
    return mask.reshape(pm.wshape)


@torch.no_grad()
def ternarize(delta, mask, density, alpha=None):
    """Return sorted flat indices, int8 signs, and scores for per-row density."""
    if not math.isfinite(density) or not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")
    delta, mask = delta.detach().float().cpu(), mask.detach().bool().cpu()
    if delta.ndim != 2 or delta.shape != mask.shape or not bool(torch.isfinite(delta).all()):
        raise ValueError("delta must be a finite matrix matching the mask")
    alpha = torch.ones(delta.shape[0]) if alpha is None else alpha.detach().float().cpu().flatten()
    if len(alpha) != len(delta):
        raise ValueError("row scales must match delta rows")
    indices = []
    for row in range(len(delta)):
        if alpha[row] <= 0:
            continue
        cols = mask[row].nonzero().flatten()
        count = math.floor(density * len(cols))
        if not count:
            continue
        scores = delta[row, cols].abs()
        # kthvalue avoids sorting all writable entries in a real 4k-wide row.
        threshold = scores.kthvalue(len(cols) - count + 1).values
        greater = cols[scores > threshold]
        equal = cols[scores == threshold][:count - len(greater)]
        chosen = torch.cat((greater, equal)).sort().values
        chosen = chosen[delta[row, chosen] != 0]
        indices.append(chosen + row * delta.shape[1])
    idx = torch.cat(indices) if indices else torch.empty(0, dtype=torch.long)
    values = delta.flatten()[idx]
    return idx.to(torch.int32), values.sign().to(torch.int8), values.abs()


def cap_budget(entries, budget):
    """Globally cap a density's candidate nnz; entries include empty modules."""
    if budget < 0:
        raise ValueError("budget must be nonnegative (0 means unlimited)")
    scores = torch.cat([e[2] for e in entries.values()])
    if not budget or len(scores) <= budget:
        return entries
    chosen = scores.argsort(descending=True, stable=True)[:budget]
    keep = torch.zeros(len(scores), dtype=torch.bool)
    keep[chosen] = True
    offset, capped = 0, {}
    for name, (idx, val, score) in entries.items():
        selection = keep[offset:offset + len(idx)]
        capped[name] = idx[selection], val[selection], score[selection]
        offset += len(idx)
    return capped


@torch.no_grad()
def install_codes(model, entries):
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        idx, val, _score = entries[name]
        idx = idx.long().to(pm.M.device)
        if getattr(pm, "sparse", False):
            slots = pm.idx.long()
            positions = torch.searchsorted(slots, idx)
            if idx.numel() and (bool((positions >= len(slots)).any()) or not bool((slots[positions] == idx).all())):
                raise ValueError("codes outside attached reserve")
            idx = positions
        elif not bool(pm.mask.flatten()[idx].all()):
            raise ValueError("codes outside attached reserve")
        pm.M.zero_()
        pm.M.flatten()[idx] = val.float().to(pm.M.device)


def patch_metadata(args, layer_ids, projs):
    """Match every config key compared by e2e_fsa.main's load-patch check."""
    # Take harness defaults instead of duplicating defaults subject to drift.
    defaults = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused"])
    return dict(eda_backend=E.eda.selected(args), method="fsa_forced", fmt="ternary", layers=layer_ids, projs=list(projs), rank=defaults.rank,
                seed=args.seed, n_experts=0, selection=None, fp4_step=defaults.fp4_step, patch_scale="row",
                scale_block=0, people=args.people, facts_per_person=args.facts_per_person,
                shared_attrs=args.shared_attrs, values=args.values, budget=args.budget,
                shared_expert=False, shared_only=False, sparse_state=args.sparse_state, occupied_flips=False,
                slot_frac=args.slot_frac, patch_scale_mult=1.0, fold_mode=None, fold_every=None,
                fold_thr=None, fold_mask=None, fold_mask_frac=None, no_pareto=defaults.no_pareto,
                export=str(E.EXPORT), alpha="mean |nonzero base weight| of the row", step=0)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    E.eda.add_argument(ap)
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--facts-per-person", type=int, default=len(E.ATTRS))
    ap.add_argument("--shared-attrs", type=int, default=0)
    ap.add_argument("--values", choices=("disjoint", "overlap"), default="disjoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--phrasings", type=int, default=0, help="first k TRAIN phrasings per fact (0 = all)")
    ap.add_argument("--layers", default="14,15")
    ap.add_argument("--projs", choices=("down", "up", "both"), default="down")
    ap.add_argument("--sparse-state", action="store_true")
    ap.add_argument("--slot-frac", type=float, default=1.0)
    ap.add_argument("--cov-tokens", type=Path, default=E.DATA / "rehearsal-fineweb.json", help="JSON token list")
    ap.add_argument("--cov-max-tokens", type=int, default=100_000)
    ap.add_argument("--cov-window", type=int, default=256)
    ap.add_argument("--cov-rank", type=int, default=64, help="rank + diagonal sketch; 0 = full fp16 CPU moment")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--density", nargs="+", default=["1e-3,3e-3,1e-2"], help="comma/space separated per-row fractions")
    ap.add_argument("--budget", type=int, default=0, help="global nonzero cap (0 = unlimited)")
    ap.add_argument("--solve-device", default="cpu", help="CPU by default; cuda:0 also supported")
    ap.add_argument("--name", default="hebbian")
    ap.add_argument("--out", type=Path, default=Path("out"), help="directory for <name>_d<density>.pt")
    return ap


def validate_args(ap, args):
    try:
        layer_ids = [int(li) for li in args.layers.split(",")]
        densities = [float(d) for part in args.density for d in part.split(",")]
    except ValueError:
        ap.error("layers must be integer indices and densities must be numbers")
    if not layer_ids or min(layer_ids) < 0 or len(set(layer_ids)) != len(layer_ids):
        ap.error("layers must be distinct nonnegative MoE indices")
    if not densities or any(not math.isfinite(d) or not 0 <= d <= 1 for d in densities) or len(set(densities)) != len(densities):
        ap.error("densities must be distinct finite fractions in [0, 1]")
    if not 0 < args.slot_frac <= 1 or (args.slot_frac < 1 and not args.sparse_state):
        ap.error("slot-frac must be in (0, 1] and needs --sparse-state when below 1")
    if args.cov_rank < 0 or args.cov_max_tokens <= 0 or args.cov_window <= 0 or args.budget < 0:
        ap.error("cov-rank/budget must be nonnegative and covariance token/window counts positive")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        ap.error("ridge must be finite and positive")
    templates = E.TRAIN_AUG_T if args.aug else E.TRAIN_T
    if not 0 <= args.phrasings <= len(templates):
        ap.error(f"phrasings must be between 0 and {len(templates)}")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        ap.error("name must be a filename stem")
    return layer_ids, densities, templates[:args.phrasings] if args.phrasings else templates


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = build_parser()
    args = ap.parse_args()
    E.eda.announce(args)
    layer_ids, densities, templates = validate_args(ap, args)
    try:
        facts = E.make_world(args.seed, args.people, args.facts_per_person, args.shared_attrs, args.values)
    except (AssertionError, ValueError) as exc:
        ap.error(f"invalid fact world: {exc}")
    if not facts:
        ap.error("the fact world must be nonempty")
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
    projs = {"down": ("down_proj",), "up": ("up_proj",), "both": ("up_proj", "down_proj")}[args.projs]
    # The harness CLI attaches down or both. Keeping empty down entries for an
    # up-only write also preserves its seeded slot order with --slot-frac < 1.
    attached_projs = ("up_proj", "down_proj") if args.projs == "up" else projs
    _params, n_slots, _ = E.attach(model, layer_ids, "fsa_forced", 16, "ternary", attached_projs,
                                 sparse=args.sparse_state, slot_frac=args.slot_frac, seed=args.seed)
    model.requires_grad_(False)
    print(f"facts={len(facts)} phrasings={len(templates)} writable_slots={n_slots}", flush=True)
    items = E.build_items(tok, facts, templates, E.BOS)
    timing = {}
    install(model, patched_layer_ids=layer_ids, mode="exact")
    try:
        with RoutedCapture(model, layer_ids, projs, args.cov_rank, args.seed) as capture:
            synchronize(device)
            start = time.monotonic()
            n_tokens = covariance_pass(model, capture, tokens, device, args.cov_max_tokens, args.cov_window, E.BOS)
            synchronize(device)
            timing["covariance"] = time.monotonic() - start
            print(f"covariance: {n_tokens} tokens, {timing['covariance']:.3f} wall seconds", flush=True)
            capture.omegas.clear()
            start = time.monotonic()
            logp = fact_pass(model, capture, items, device)
            synchronize(device)
            timing["fact_pass"] = time.monotonic() - start
            print(f"fact pass: {timing['fact_pass']:.3f} wall seconds, answer_logp={logp:.6g}", flush=True)
    finally:
        uninstall(model)

    start = time.monotonic()
    sweeps = {density: {} for density in densities}
    unseen = []
    cov_counts = {name: cov.count for name, cov in capture.covariances.items()}
    for name, lin in E.patched_linears(model):
        pm = lin.parametrizations.weight[0]
        records = capture.records.pop(name, [])
        cov = capture.covariances.pop(name, None)
        if records:
            if cov is None:
                unseen.append(name)
                cov = Covariance(lin.in_features, rank=0)
            delta = delta_write(records, cov, args.ridge, args.solve_device).cpu()
            mask = writable_mask(pm)
            for density in densities:
                sweeps[density][name] = ternarize(delta, mask, density, pm.alpha)
            del delta, mask
        else:
            for density in densities:
                sweeps[density][name] = (torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int8), torch.empty(0))
        del records, cov
    synchronize(args.solve_device)
    timing["solve"] = time.monotonic() - start
    print(f"solve/ternarize: {timing['solve']:.3f} wall seconds; unseen covariance experts={len(unseen)}", flush=True)
    meta = patch_metadata(args, layer_ids, attached_projs)
    meta["hebbian"] = dict(cov_tokens=str(args.cov_tokens), cov_token_count=n_tokens, cov_rank=args.cov_rank,
                           cov_counts=cov_counts, identity_fallback=unseen, ridge=args.ridge, aug=args.aug,
                           phrasings=len(templates), projs=args.projs, answer_logp=logp, timing_wall_seconds=timing)
    for density, entries in sweeps.items():
        entries = cap_budget(entries, args.budget)
        nnz = sum(len(e[0]) for e in entries.values())
        install_codes(model, entries)
        path = args.out / f"{args.name}_d{density}.pt"
        file_bytes = E.save_patch(model, path, dict(meta, hebbian=dict(meta["hebbian"], density=density, nnz=nnz)))
        print(f"density={density:g} nnz={nnz} bytes={file_bytes} patch={path}", flush=True)


if __name__ == "__main__":
    main()
