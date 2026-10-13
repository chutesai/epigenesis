"""Forward-only contrast geometry of the unpatched 3-person x 100-fact world.

Run on the model host:
  python experiments/end_to_end/geometry_probe.py --seed 0 --out geometry.json

Layer IDs follow e2e_fsa.moe_layers (zero based). Two batched passes capture
all requested layers: 900 TRAIN_T prompts and 64 nonoverlapping 96-token anchor
windows. Raw x/h/route records stay in CPU memory; JSON contains their coverage
and the metrics, not gigabytes of activations. Analysis uses one expert at a
time on CUDA. Whitening is exact, including the ridge-regularized nullspace;
it uses the smaller of the sample and feature covariance matrices.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from itertools import combinations
from pathlib import Path
from types import MethodType

import torch

if __package__:
    from . import e2e_fsa as E
else:
    import e2e_fsa as E


DEFAULT_LAYERS = (3, 6, 9, 11, 12, 14, 16, 17, 18, 20, 22, 25, 29)
BATCH_SIZE = 8
ANCHOR_WINDOWS = 64
ANCHOR_LENGTH = 96
RIDGE_REL = 1e-3
VERDICT_THRESHOLDS = {
    "weak_median_contrast": 0.01,
    "weak_p10_contrast": 0.001,
    "mostly_additive_eta": 0.1,
}


def weighted_summary(values, weights=None):
    """Inverse-CDF weighted quantiles; undefined/zero-weight samples are omitted."""
    values = list(values)
    weights = [1.0] * len(values) if weights is None else list(weights)
    if len(values) != len(weights):
        raise ValueError("values and weights must have equal length")
    valid = sorted(
        (float(v), float(w))
        for v, w in zip(values, weights)
        if v is not None and math.isfinite(v) and math.isfinite(w) and w > 0
    )
    if not valid:
        return {"n": 0, "weight": 0.0, "mean": None, "median": None, "p10": None}
    total = sum(w for _, w in valid)

    def quantile(q):
        mass = 0.0
        for value, weight in valid:
            mass += weight
            if mass >= q * total:
                return value
        return valid[-1][0]

    return {
        "n": len(valid),
        "weight": total,
        "mean": sum(v * w for v, w in valid) / total,
        "median": quantile(0.5),
        "p10": quantile(0.1),
    }


def writable_columns(weight):
    """Majority union: forced-mask columns writable for >= half the output rows."""
    mask = E.pair8_forced_mask(weight)
    return (2 * mask.sum(0) >= mask.shape[0]).nonzero().flatten()


def whiten_keys(keys, anchors, ridge_rel=RIDGE_REL):
    """Whiten uncentered keys by centered anchor covariance + ridge*I.

    Covariance is unbiased (n-1 denominator); ridge = ridge_rel * trace(C)/d.
    General-text centering estimates covariance only: it is not subtracted from
    the person keys. The dual eigensolve avoids a d x d solve when n < d.
    Returns keys @ (C + ridge*I)^(-1/2) and auditable covariance metadata.
    """
    if (
        keys.ndim != 2
        or anchors.ndim != 2
        or keys.shape[1] != anchors.shape[1]
        or keys.shape[1] == 0
        or len(anchors) < 2
    ):
        raise ValueError("need compatible nonempty keys and at least two anchors")
    if not math.isfinite(ridge_rel) or ridge_rel <= 0:
        raise ValueError("ridge_rel must be positive and finite")
    dtype = torch.float64 if anchors.dtype == torch.float64 else torch.float32
    anchors = anchors.to(dtype=dtype, device=keys.device)
    keys = keys.to(dtype)
    a = (anchors - anchors.mean(0)) / math.sqrt(len(anchors) - 1)
    trace = a.square().sum()
    ridge = ridge_rel * trace / a.shape[1]
    if not torch.isfinite(trace) or float(trace) <= 0:
        raise ValueError("anchor covariance has zero or nonfinite trace")
    if len(a) < a.shape[1]:
        eig, u = torch.linalg.eigh(a @ a.T)
        # Matrix-function identity, without dividing by a sample eigenvalue:
        # ((lambda+r)^(-1/2)-r^(-1/2))/lambda has the finite limit
        # -1/(2*r^(3/2)). Thus even tiny eigenvalues retain their exact ridge
        # contribution; no PCA truncation silently changes the geometry.
        eig = eig.clamp_min(0)
        root = (eig + ridge).sqrt()
        coeff = -1 / (ridge.sqrt() * root * (root + ridge.sqrt()))
        white = keys / ridge.sqrt() + (((keys @ a.T) @ u) * coeff) @ (u.T @ a)
        backend = "dual_eigh"
    else:
        eig, v = torch.linalg.eigh(a.T @ a)
        white = ((keys @ v) / (eig.clamp_min(0) + ridge).sqrt()) @ v.T
        backend = "feature_eigh"
    return white, {
        "n_anchor_tokens": len(a),
        "dimensions": a.shape[1],
        "trace": float(trace),
        "ridge": float(ridge),
        "backend": backend,
    }


def geometry_from_gram(gram):
    """Pair cosines and smallest *contrast* singular value / mean person norm.

    Centering an n-person matrix forces a zero singular value. Use the (n-1)th
    descending singular value, not that structural zero. Collinear three-way
    keys still correctly have zero contrast sigma_min.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or len(gram) not in (2, 3):
        raise ValueError("geometry requires a two- or three-person Gram matrix")
    gram = gram.double()
    norms = gram.diagonal().clamp_min(0).sqrt()
    cosines = [
        float((gram[i, j] / (norms[i] * norms[j])).clamp(-1, 1))
        if norms[i] > 0 and norms[j] > 0
        else None
        for i, j in combinations(range(len(gram)), 2)
    ]
    centered = (
        gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()
    )
    eig = torch.linalg.eigvalsh((centered + centered.T) / 2).clamp_min(0)
    # Ascending eigenvalue 0 is the mandatory centering null direction.
    sigma = float(eig[1].sqrt()) if float(norms.mean()) > 0 else None
    return {
        "pairwise_cosines": cosines,
        "contrast_sigma_min": sigma,
        "relative_contrast": sigma / float(norms.mean()) if sigma is not None else None,
        "mean_key_norm": float(norms.mean()),
    }


def key_geometry(keys):
    if keys.ndim != 2 or keys.shape[1] == 0:
        raise ValueError("keys must be a nonempty person-by-feature matrix")
    keys = keys.double()
    return geometry_from_gram(keys @ keys.T)


def interaction_energy(person_a, person_b):
    """RMS energies over all ordered distinct attribute pairs, in O(n*d).

    d[a] = h(A,a)-h(B,a). Numerator is E[(d[a]-d[b])^2], and
    denominator E[d[a]^2]. Ordered pairs make the denominator symmetric in a,b.
    Inputs are unwhitened, writable-column mean keys for the same expert.
    """
    if (
        person_a.ndim != 2
        or person_a.shape != person_b.shape
        or len(person_a) < 2
        or person_a.shape[1] == 0
    ):
        raise ValueError(
            "interaction needs two people with at least two common attributes"
        )
    delta = person_a.double() - person_b.double()
    numerator = float(
        2 * len(delta) / (len(delta) - 1) * (delta - delta.mean(0)).square().mean()
    )
    denominator = float(delta.square().mean())
    return {
        "numerator_ms": numerator,
        "denominator_ms": denominator,
        "eta": math.sqrt(numerator / denominator) if denominator > 0 else None,
        "n_attribute_pairs": len(delta) * (len(delta) - 1) // 2,
    }


def interaction_ratio(person_a, person_b):
    return interaction_energy(person_a, person_b)["eta"]


def routing_metrics(items, facts, routes, seed=0, random_pairs=10000):
    """Jaccard over prompt pairs (all phrasings); seeded random baseline."""
    if len(items) != len(routes):
        raise ValueError("one routing set is required per prompt")
    if any(route is None or not route for route in routes):
        raise ValueError("a prompt is missing its routing set")
    routes = [set(route) for route in routes]
    by_person, by_attr = defaultdict(list), defaultdict(list)
    for i, item in enumerate(items):
        fact = facts[item["fi"]]
        by_person[fact.pid].append(i)
        by_attr[fact.attr].append(i)

    def jaccard(i, j):
        return len(routes[i] & routes[j]) / len(routes[i] | routes[j])

    cross = [
        jaccard(i, j)
        for group in by_attr.values()
        for i, j in combinations(group, 2)
        if facts[items[i]["fi"]].pid != facts[items[j]["fi"]].pid
    ]
    same = [
        jaccard(i, j)
        for group in by_person.values()
        for i, j in combinations(group, 2)
        if facts[items[i]["fi"]].attr != facts[items[j]["fi"]].attr
    ]
    n = len(items)
    count = min(random_pairs, n * (n - 1) // 2)
    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < count:
        i, j = sorted(rng.sample(range(n), 2))
        pairs.add((i, j))
    return {
        "same_attribute_different_person": weighted_summary(cross),
        "same_person_different_attribute": weighted_summary(same),
        "random_pairs": weighted_summary(jaccard(i, j) for i, j in sorted(pairs)),
        "top_k": 12,
        "observed_set_sizes": sorted({len(route) for route in routes}),
    }


class ActivationCapture(AbstractContextManager):
    """Join exact dispatch token IDs to real up/down projection hooks.

    Grouped fallback and fast_moe exact dispatch both feed experts ascending,
    unique token IDs. This avoids matching floating-point inputs or mistaking
    right padding for the final prompt token. Hook validation catches backends
    which bypass Linear modules, rather than silently reporting partial data.
    """

    def __init__(self, layers, n_prompts):
        self.layers = layers
        self.routes = {li: [None] * n_prompts for li in layers}
        self.facts = {li: defaultdict(lambda: defaultdict(list)) for li in layers}
        self.anchors = {li: defaultdict(list) for li in layers}
        self.columns = {}
        self.plans = {}
        self.hooks, self.methods = [], []
        self.phase = None

    def batch(self, phase, seqs, prompt_ids=None):
        self.phase = phase
        width = max(map(len, seqs))
        if phase == "facts":
            self.targets = torch.tensor(
                [i * width + len(s) - 1 for i, s in enumerate(seqs)]
            )
            self.prompt_ids = torch.tensor(prompt_ids)
        elif phase == "anchors":
            # Every real general-text token, excluding repeated BOS and padding.
            self.targets = torch.tensor(
                [i * width + j for i, s in enumerate(seqs) for j in range(1, len(s))]
            )
        else:
            raise ValueError("unknown capture phase")

    def _dispatch(self, li, x, indices, weights):
        if self.phase is None:
            raise RuntimeError("set capture batch before forwarding")
        if (
            indices.ndim != 2
            or indices.shape != weights.shape
            or len(x) != len(indices)
        ):
            raise RuntimeError("unsupported grouped dispatch layout")
        ids, weights = indices.detach().cpu(), weights.detach().float().cpu()
        if self.phase == "facts":
            top = weights[self.targets].argsort(-1, descending=True)[:, :12]
            selected = ids[self.targets].gather(1, top)
            for pid, route in zip(self.prompt_ids.tolist(), selected.tolist()):
                self.routes[li][pid] = route
        flat = ids.flatten()
        order = flat.argsort(stable=True)
        tids = order.div(ids.shape[1], rounding_mode="floor")
        sorted_weights = weights.flatten()[order]
        counts = torch.bincount(flat, minlength=len(self.layers[li].experts)).tolist()
        plans, start = {}, 0
        for ei, count in enumerate(counts):
            end = start + count
            if count and (self.phase == "facts" or ei in self.columns[li]):
                tokens, inverse = tids[start:end].unique(
                    sorted=True, return_inverse=True
                )
                route_weights = torch.zeros(len(tokens)).scatter_add_(
                    0, inverse, sorted_weights[start:end]
                )
                rows = torch.searchsorted(tokens, self.targets)
                valid = rows < len(tokens)
                valid &= tokens[rows.clamp_max(len(tokens) - 1)] == self.targets
                rows = rows[valid]
                if len(rows):
                    plans[ei] = {
                        "rows": rows,
                        "weights": route_weights[rows],
                        "ids": self.prompt_ids[valid]
                        if self.phase == "facts"
                        else None,
                        "up_seen": False,
                        "down_seen": False,
                    }
            start = end
        self.plans[li] = plans

    def _up(self, li, ei, inp):
        plan = self.plans.get(li, {}).get(ei)
        if plan is not None:
            plan["up_seen"] = True
            if self.phase == "facts":
                x = inp[0].reshape(-1, inp[0].shape[-1])
                plan["x"] = x[plan["rows"].to(x.device)].detach().cpu()

    def _down(self, li, ei, inp):
        plan = self.plans.get(li, {}).get(ei)
        if plan is None:
            return
        if not plan["up_seen"] or plan["down_seen"]:
            raise RuntimeError("expert hook order does not match grouped dispatch")
        plan["down_seen"] = True
        h = inp[0].reshape(-1, inp[0].shape[-1])
        h = h[plan["rows"].to(h.device)].detach()
        if self.phase == "anchors":
            cols = self.columns[li][ei].to(h.device)
            self.anchors[li][ei].append(h[:, cols].cpu())
        else:
            rec = self.facts[li][ei]
            for key, value in (
                ("ids", plan["ids"]),
                ("weights", plan["weights"]),
                ("x", plan["x"]),
                ("h", h.cpu()),
            ):
                rec[key].append(value)

    def _finish(self, li):
        if li not in self.plans:
            raise RuntimeError("model bypassed grouped routing capture")
        if any(
            not p["up_seen"] or not p["down_seen"] for p in self.plans.pop(li).values()
        ):
            raise RuntimeError(
                "expert backend bypassed projection hooks; use resident exact dispatch"
            )

    def __enter__(self):
        try:
            for li, moe in self.layers.items():
                dispatch = getattr(moe, "_forward_frozen_exact_grouped", None)
                if not callable(dispatch):
                    raise TypeError(
                        "model requires resident grouped-dispatch interface"
                    )
                own = "_forward_frozen_exact_grouped" in moe.__dict__
                self.methods.append((moe, dispatch, own))

                def capture(layer, x, indices, weights, li=li, dispatch=dispatch):
                    self._dispatch(li, x, indices, weights)
                    return dispatch(x, indices, weights)

                moe._forward_frozen_exact_grouped = MethodType(capture, moe)
                self.hooks.append(
                    moe.register_forward_hook(
                        lambda mod, inp, out, li=li: self._finish(li)
                    )
                )
                for ei, expert in enumerate(moe.experts):
                    if expert is None:
                        continue
                    self.hooks.append(
                        expert.up_proj.register_forward_pre_hook(
                            lambda mod, inp, li=li, ei=ei: self._up(li, ei, inp)
                        )
                    )
                    self.hooks.append(
                        expert.down_proj.register_forward_pre_hook(
                            lambda mod, inp, li=li, ei=ei: self._down(li, ei, inp)
                        )
                    )
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


def mean_fact_keys(record, items, columns):
    """Plain means over routed TRAIN phrasings; absent routes are never zero keys."""
    ids = torch.cat(record["ids"])
    h = torch.cat(record["h"])[:, columns].float()
    weights = torch.cat(record["weights"])
    by_fact = defaultdict(list)
    for row, pid in enumerate(ids.tolist()):
        by_fact[items[pid]["fi"]].append(row)
    fact_ids = sorted(by_fact)
    keys = torch.stack([h[by_fact[fi]].mean(0) for fi in fact_ids])
    # Weight includes missing phrasings as zero route mass, never zero keys.
    route_mass = {
        fi: float(weights[by_fact[fi]].sum()) / len(E.TRAIN_T) for fi in fact_ids
    }
    counts = {fi: len(by_fact[fi]) for fi in fact_ids}
    return fact_ids, keys, route_mass, counts


def expert_interactions(keys, fact_ids, facts, groups, mass):
    lookup = {fi: row for row, fi in enumerate(fact_ids)}
    by_pair = defaultdict(list)
    for group in groups:
        for fi, fj in combinations(sorted(group, key=lambda fi: facts[fi].pid), 2):
            if fi in lookup and fj in lookup:
                by_pair[facts[fi].pid, facts[fj].pid].append((fi, fj))
    details = []
    numerator = denominator = count = weight = 0.0
    for pair, pairs in sorted(by_pair.items()):
        if len(pairs) < 2:
            continue
        a = keys[[lookup[fi] for fi, _ in pairs]]
        b = keys[[lookup[fj] for _, fj in pairs]]
        energy = interaction_energy(a, b)
        n = energy["n_attribute_pairs"]
        pair_weight = sum((mass[fi] + mass[fj]) / 2 for fi, fj in pairs) / len(pairs)
        numerator += n * energy["numerator_ms"]
        denominator += n * energy["denominator_ms"]
        weight += n * pair_weight
        count += n
        details.append(dict(person_pair=list(pair), n_attributes=len(pairs), **energy))
    return {
        "eta": math.sqrt(numerator / denominator) if denominator > 0 else None,
        "n_attribute_pairs": int(count),
        "route_weight": weight / count if count else 0.0,
        "numerator_ms": numerator / count if count else None,
        "denominator_ms": denominator / count if count else None,
        "person_pairs": details,
    }


def geometry_verdict(three_way):
    contrast = three_way["relative_contrast"]
    eta = three_way["interaction_eta"]["median"]
    if contrast["median"] is None or eta is None:
        return "insufficient common-expert contrast/interaction coverage"
    if (
        contrast["median"] < VERDICT_THRESHOLDS["weak_median_contrast"]
        or contrast["p10"] < VERDICT_THRESHOLDS["weak_p10_contrast"]
    ):
        return "binding limitation plausible: weak whitened person contrast (heuristic)"
    if eta < VERDICT_THRESHOLDS["mostly_additive_eta"]:
        return "binding limitation plausible: mostly additive person/attribute keys (heuristic)"
    return "optimization limitation plausible: person contrast and interactions present (heuristic)"


@torch.no_grad()
def analyze_layer(li, moe, capture, items, facts, groups, device, seed):
    details = {2: [], 3: []}
    interactions = {2: [], 3: []}
    experts = []
    for ei, record in sorted(capture.facts[li].items()):
        cols = capture.columns[li][ei]
        info = {
            "expert": ei,
            "n_visits": sum(len(v) for v in record["ids"]),
            "latent_dim": record["x"][0].shape[1],
            "hidden_dim": record["h"][0].shape[1],
            "writable_columns": cols.tolist(),
            "n_writable_columns": len(cols),
        }
        experts.append(info)
        if not len(cols):
            info["skipped"] = "no majority-writable columns"
            continue
        fact_ids, keys, mass, counts = mean_fact_keys(record, items, cols)
        info["n_facts"] = len(fact_ids)
        info["routed_phrasings_per_fact"] = weighted_summary(counts.values())
        lookup = {fi: row for row, fi in enumerate(fact_ids)}
        # Interaction is in raw h space and remains measurable without anchors.
        for arity in (2, 3):
            interaction = expert_interactions(
                keys, fact_ids, facts, groups[arity], mass
            )
            interactions[arity].append(dict(expert=ei, **interaction))
        anchors = capture.anchors[li].pop(ei, [])
        if sum(map(len, anchors)) < 2:
            info["skipped"] = "fewer than two same-expert anchor tokens"
            continue
        try:
            white, covariance = whiten_keys(
                keys.to(device), torch.cat(anchors).to(device)
            )
        except ValueError as exc:
            info["skipped"] = str(exc)
            continue
        info["covariance"] = covariance
        # Contrast can be tiny compared with the uncentered key norm. Accumulate
        # the small person Gram in fp64 before centering to retain that signal.
        white = white.double()
        gram = (white @ white.T).cpu()
        del white, anchors
        for arity in (2, 3):
            for group in groups[arity]:
                if not all(fi in lookup for fi in group):
                    continue
                rows = [lookup[fi] for fi in group]
                metrics = geometry_from_gram(gram[rows][:, rows])
                details[arity].append(
                    dict(
                        expert=ei,
                        attribute=facts[group[0]].attr,
                        person_ids=[facts[fi].pid for fi in group],
                        routed_phrasings=[counts[fi] for fi in group],
                        route_weight=sum(mass[fi] for fi in group) / arity,
                        **metrics,
                    )
                )
    result = {
        "layer": li,
        "routing": routing_metrics(items, facts, capture.routes[li], seed),
        "experts": experts,
    }
    for arity, label in ((2, "two_way"), (3, "three_way")):
        rows = details[arity]
        cosine, cosine_weight = [], []
        for row in rows:
            cosine.extend(row["pairwise_cosines"])
            cosine_weight.extend([row["route_weight"]] * len(row["pairwise_cosines"]))
        etas = interactions[arity]
        result[label] = {
            "n_attribute_groups": len(groups[arity]),
            "n_group_expert_observations": len(rows),
            "n_attributes_covered": len({row["attribute"] for row in rows}),
            "pairwise_cosine": weighted_summary(cosine, cosine_weight),
            "relative_contrast": weighted_summary(
                [row["relative_contrast"] for row in rows],
                [row["route_weight"] for row in rows],
            ),
            "interaction_eta": weighted_summary(
                [row["eta"] for row in etas], [row["route_weight"] for row in etas]
            ),
            "groups": rows,
            "expert_interactions": etas,
        }
    result["verdict"] = geometry_verdict(result["three_way"])
    return result


def parse_layers(value):
    try:
        layers = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be distinct nonnegative integers")
    return layers


def anchor_sequences(tokens):
    # Use the harness's training portion, reserving its 510-token validation tail.
    training = tokens[:-510]
    needed = ANCHOR_WINDOWS * (ANCHOR_LENGTH - 1)
    if len(training) < needed:
        raise ValueError(
            f"anchor file needs {needed + 510} tokens including validation tail"
        )
    return [
        [E.BOS] + training[i : i + ANCHOR_LENGTH - 1]
        for i in range(0, needed, ANCHOR_LENGTH - 1)
    ]


@torch.no_grad()
def collect_batches(model, capture, seqs, phase, device):
    for start in range(0, len(seqs), BATCH_SIZE):
        batch = seqs[start : start + BATCH_SIZE]
        capture.batch(phase, batch, range(start, start + len(batch)))
        tokens, _ = E.batchify(batch, device)
        E.logits_of(model, tokens)
        if (
            start == 0
            or (start // BATCH_SIZE + 1) % 20 == 0
            or start + len(batch) == len(seqs)
        ):
            print(f"{phase}: {start + len(batch)}/{len(seqs)}", flush=True)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(parser)
    parser.add_argument("--layers", type=parse_layers, default=list(DEFAULT_LAYERS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="JSON metrics output")
    args = parser.parse_args(argv)
    E.eda.announce(args)
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU and the /workspace/malleable model/data are required")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer

    if __package__:
        from .fast_moe import install, uninstall
    else:
        from fast_moe import install, uninstall

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    # Small CPU Gram matrices/hook joins should not start large BLAS thread pools.
    torch.set_num_threads(1)
    tokenizer = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    anchors = anchor_sequences(
        json.loads((E.DATA / "anchor-wikitext.json").read_text())
    )
    facts = E.make_world(args.seed, n_people=3, per_person=100, shared=30)
    items = E.build_items(tokenizer, facts, E.TRAIN_T, E.BOS)
    by_attr = defaultdict(list)
    for fi, fact in enumerate(facts):
        by_attr[fact.attr].append(fi)
    groups = {
        arity: [
            sorted(group, key=lambda fi: facts[fi].pid)
            for group in by_attr.values()
            if len(group) == arity
        ]
        for arity in (2, 3)
    }
    model, _cfg, device = E.eda.build_model(build_model, args,
        E.EXPORT,
        device="cuda:0",
        model_config=None,
        step=58814,
        expert_format="bf16",
    )
    model.eval()
    model.requires_grad_(False)
    layers = E.moe_layers(model)
    if max(args.layers) >= len(layers):
        parser.error(f"layer index outside model's {len(layers)} MoE layers")
    # Read-only exact dispatch retains Linear hooks and does not stack weight banks.
    install(model, mode="exact")
    results = {
        "eda_backend": E.eda.selected(args),
        "seed": args.seed,
        "people": 3,
        "facts_per_person": 100,
        "shared_requested": 30,
        "n_prompts": len(items),
        "templates": E.TRAIN_T,
        "layer_indexing": "zero-based MoE order",
        "export": str(E.EXPORT),
        "expert_format": "bf16",
        "batch_size": BATCH_SIZE,
        "anchor": {
            "path": str(E.DATA / "anchor-wikitext.json"),
            "windows": ANCHOR_WINDOWS,
            "window_length": ANCHOR_LENGTH,
            "exclude_bos": True,
            "reserve_tail_tokens": 510,
        },
        "definitions": {
            "columns": "majority union: writable in >= half of forced-mask rows",
            "covariance": "centered unbiased same-expert general-text covariance",
            "ridge": "1e-3 * trace(covariance) / writable_dimension",
            "keys": "uncentered plain mean of routed TRAIN_T phrasings; absent routes omitted",
            "group_weight": "mean person route mass, each mass summed over visits / 3",
            "contrast": "(n-1)th descending centered singular value / mean person key norm",
            "interaction": "raw writable h; all complete ordered attribute-pair rectangles",
            "interaction_weight": "mean rectangle route mass within each expert",
            "arities": "exact world attribute arity; two-way excludes subsets of three-way",
            "shared": "all actual three-person attributes, including incidental shared attributes",
            "raw_records": "latent x, hidden h and weights captured in CPU memory only",
        },
        "verdict_thresholds": VERDICT_THRESHOLDS,
        "layers": [],
    }
    try:
        with ActivationCapture(
            {li: layers[li][1] for li in args.layers}, len(items)
        ) as capture:
            collect_batches(
                model, capture, [item["p"] for item in items], "facts", device
            )
            capture.columns = {
                li: {
                    ei: writable_columns(
                        layers[li][1].experts[ei].down_proj.weight
                    ).cpu()
                    for ei in capture.facts[li]
                }
                for li in args.layers
            }
            collect_batches(model, capture, anchors, "anchors", device)
        for li in args.layers:
            result = analyze_layer(
                li, layers[li][1], capture, items, facts, groups, device, args.seed
            )
            result["name"] = layers[li][0]
            results["layers"].append(result)
            print(f"layer {li}: {result['verdict']}", flush=True)
            del capture.facts[li], capture.anchors[li]
            results["elapsed_seconds"] = time.monotonic() - t0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    finally:
        uninstall(model)


if __name__ == "__main__":
    main()
