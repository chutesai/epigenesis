"""C16 branch serving on a frozen base. Fact patches stay fixed; selectors may fit.

B/C/D use full-mask person patches; E/F/G use disjoint block or unit partitions. Routing
reads frozen-base full-block residual inputs and holds one decision for the whole query
(or general-text window). F loads one merged patch before a single forward.
H/FLAT compare hierarchical and flat selection over person/attribute-group banks.
"""
import argparse
import hashlib
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

try:
    from . import e2e_fsa as E
    from . import separability_probe as S
    import review_eval as R
    from . import review_eval as R
except ImportError:
    import e2e_fsa as E
    import separability_probe as S
    import review_eval as R


IDENTITY = ("method", "fmt", "seed", "people", "facts_per_person", "shared_attrs", "values",
            "layers", "projs", "relation_world", "synthetic_people", "lora_ternary", "rank", "slot_frac", "patch_scale", "patch_scale_mult",
            "patch_scale_mult_up", "patch_scale_mult_down", "shared_expert", "shared_only", "export", "export_step")


def sister_relations(facts, relation_world=False):
    """Read stored sister edges; legacy banks retain the external prototype cycle."""
    expected = {0: 1, 1: 2, 2: 0}
    if not relation_world:
        return expected
    people = {f.name: f.pid for f in facts}
    sisters = [f for f in facts if f.attr in ("sister's name", "sister")]
    assert len(sisters) == 3 and all(f.val in people for f in sisters), \
        'relation world needs one stored sister fact per person with a known target'
    relations = {f.pid: people[f.val] for f in sisters}
    assert relations == expected, 'stored sister relations disagree with the three-person cycle'
    return relations


def validate_training_match(states):
    """Compatibility alone is insufficient evidence of a matched comparison."""
    if not states:
        raise ValueError('no patches for training comparison')
    E.eda.require_same(states, "training comparison")
    signatures = []
    required = ('training_recipe', 'step', 'wake_steps', 'value_steps', 'training_exposures',
                'mask_sha256', 'selection_policy')
    for state in states:
        meta = state['meta']
        missing = [key for key in required if meta.get(key) is None]
        if missing:
            raise ValueError('C34/PK requires recorded training metadata: ' + ', '.join(missing))
        recipe_keys = ('steps', 'repair_steps', 'wake_steps', 'budget', 'optim', 'lr', 'wd',
                       'bop_gamma', 'bop_tau', 'bop_tau_abs', 'bop_accum', 'aug', 'aug_all',
                       'local_lambda', 'fact_batch', 'self_study', 'unit_init')
        if any(key not in meta['training_recipe'] for key in recipe_keys):
            raise ValueError('C34/PK requires complete training recipe')
        exposures = meta['training_exposures']
        exposure_keys = ('facts', 'formats_per_fact', 'examples_per_fact', 'self_study_examples', 'wake_prompt_count', 'allocation_fact_forwards')
        for key in exposure_keys:
            if key not in exposures:
                raise ValueError('C34/PK requires training exposure field ' + key)
        signatures.append({key: meta[key] for key in ('training_recipe', 'step', 'wake_steps', 'value_steps', 'selection_policy')} |
                          {'exposures': {key: exposures[key] for key in exposure_keys}})
    for signature in signatures[1:]:
        if signature != signatures[0]:
            raise ValueError('C34/PK training budget/recipe/exposures differ')
    return signatures[0]


def partition_label(meta):
    if not meta.get("slot_partition", 0):
        return "full-mask"
    return "unit-partition" if meta.get("partition_unit", False) else "block-partition"


def validate_patches(states, partitioned=False, banked=False, eda_backend=None, crosseval_args=None):
    if not states:
        raise ValueError("no person patches")
    backend = E.eda.require_same(states, "patch banks")
    if eda_backend is not None:
        if crosseval_args is None:
            E.eda.require_backend({"eda_backend": backend}, eda_backend, "patch banks")
        else:
            E.eda.require_patch_backend({"eda_backend": backend}, crosseval_args, "patch banks")
    first = states[0]["meta"]
    people = first["people"]
    owners = [s["meta"].get("train_person") for s in states]
    groups = first.get("attr_groups", 1)
    if banked:
        pairs = [(s["meta"].get("train_person"), s["meta"].get("train_attr_group")) for s in states]
        if (not isinstance(groups, int) or groups < 1 or len(pairs) != people*groups
                or set(pairs) != {(p, g) for p in range(people) for g in range(groups)}):
            raise ValueError("need exactly one patch for every (person, attribute group)")
    elif sorted(p for p in owners if isinstance(p, int)) != list(range(people)):
        raise ValueError("need exactly one patch for every person")
    lora = first["method"] == "lora"
    if lora and (partitioned or banked or first.get("residents")):
        raise ValueError("LoRA person banks require independent unpartitioned writes on the base")
    if not lora and (first["method"] != "fsa_forced" or first["fmt"] != "ternary" or first["patch_scale"] != "row"):
        raise ValueError("branches require row-scaled forced-mask ternary patches")
    parts = []
    for state in states:
        if state.get("format") != "e2e-fsa-patch-v1":
            raise ValueError("unsupported patch format")
        meta = state["meta"]
        if meta.get("attr_groups", 1) != groups or (not banked and meta.get("train_attr_group") is not None):
            raise ValueError("patch config differs: attribute groups")
        for key in IDENTITY:
            default = E.EXPORT_STEP if key == "export_step" else False if key == "relation_world" else None
            if meta.get(key, default) != first.get(key, default):
                raise ValueError(f"patch config differs: {key}")
        if meta.get("address_units", 0) != first.get("address_units", 0):
            raise ValueError("patch config differs: address_units")
        if meta.get("partition_unit", False) != first.get("partition_unit", False):
            raise ValueError("patch config differs: partition_unit")
        for key, default in (("units_per_fact", 0), ("unit_init", "borrowed"),
                             ("address_mode", "mutable"), ("address_center", False),
                             ("exclude_general", 0.), ("entropy_units", False),
                             ("address_pronoun", False), ("value_layers_late", False),
                             ("address_layers", first["layers"]), ("value_layers", first["layers"]),
                             ("configured_value_layers", first["layers"])):
            if meta.get(key, default) != first.get(key, default):
                raise ValueError(f"patch config differs: {key}")
        if meta.get("units_per_fact") and not meta.get("unit_allocation", {}).get("assignments"):
            raise ValueError("unit patches require saved fact allocation")
        if meta.get("address_units") and not meta.get("wake", {}).get("detectors"):
            raise ValueError("address patches require saved selected detector rows")
        if meta.get("occupied_flips") or meta.get("n_experts", 0) or meta.get("shared_expert") or meta.get("shared_only"):
            raise ValueError("branch evaluator requires all routed experts without occupied flips")
        k = meta.get("slot_partition", 0)
        if banked:
            if k and (k != people*groups or meta.get("slot_part") != meta["train_person"]*groups+meta["train_attr_group"]):
                raise ValueError("bank partition must use K=people*groups and part=person*groups+group")
            if k != first.get("slot_partition", 0):
                raise ValueError("patch config differs: slot_partition")
        elif (partitioned and k != 3) or (not partitioned and k):
            raise ValueError("expected K=3 partition patches" if partitioned else "expected full-mask patches")
        if meta.get("partition_unit", False) and not k:
            raise ValueError("unit partitions require slot_partition")
        if partitioned:
            parts.append(meta.get("slot_part"))
        if state["entries"].keys() != states[0]["entries"].keys():
            raise ValueError("patch linear sets differ")
        for name, entry in state["entries"].items():
            ref = states[0]["entries"][name]
            if lora:
                if entry["kind"] != "lora" or ref["kind"] != "lora":
                    raise ValueError(f"incompatible LoRA entry: {name}")
                A, B = entry["A"], entry["B"]
                if (A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0]
                        or A.shape != ref["A"].shape or B.shape != ref["B"].shape
                        or A.shape[0] != meta["rank"] or not torch.isfinite(A).all()
                        or not torch.isfinite(B).all() or entry.get("ternary", False) != first.get("lora_ternary", False)):
                    raise ValueError(f"invalid LoRA factors: {name}")
                if entry.get("ternary") and (entry["alpha"].shape != (B.shape[0], 1)
                        or not torch.equal(entry["alpha"], ref["alpha"])
                        or entry.get('code_space', False) != ref.get('code_space', False)):
                    raise ValueError(f"incompatible LoRA scales: {name}")
                continue
            idx, val = entry["idx"].long(), entry["val"].float()
            if (entry["kind"] != "fsa" or entry["fmt"] != "ternary" or entry["scale"] != "row"
                    or entry.get("flip_idx", torch.empty(0)).numel() or entry["shape"] != ref["shape"]
                    or not torch.equal(entry["alpha"], ref["alpha"])):
                raise ValueError(f"incompatible entry: {name}")
            if (idx.numel() != val.numel() or (val.abs() != 1).any() or (idx < 0).any()
                    or (idx >= math.prod(entry["shape"])).any()
                    or (idx[1:] <= idx[:-1]).any()):
                raise ValueError(f"invalid sparse ternary codes: {name}")
    if partitioned and sorted(p for p in parts if isinstance(p, int)) != [0, 1, 2]:
        raise ValueError("disjoint patches need each of parts 0,1,2 exactly once")
    if banked:
        if first.get("slot_partition") and overlap_stats(states)["overlapping_slots"]:
            raise ValueError("partition banks contain overlapping writes")
        return {s["meta"]["train_person"]*groups+s["meta"]["train_attr_group"]: s for s in states}
    return {s["meta"]["train_person"]: s for s in states}


def overlap_stats(states):
    """Physical nonzero slots, independent of code scales. Fractions use nnz union."""
    if states[0]["meta"].get("method") == "lora":
        return lora_overlap_stats(states)
    slots = [{(name, int(i)): int(v) for name, e in s["entries"].items()
              for i, v in zip(e["idx"].tolist(), e["val"].tolist())} for s in states]
    union = set().union(*slots)
    occupancy = {key: [d[key] for d in slots if key in d] for key in union}
    overlap = sum(len(v) > 1 for v in occupancy.values())
    pairs = {f"{i},{j}": dict(slots=len(slots[i].keys() & slots[j].keys()),
                              sign_conflicts=sum(slots[i][k] != slots[j][k]
                                                 for k in slots[i].keys() & slots[j].keys()))
             for i, j in combinations(range(len(slots)), 2)}
    triples = {f"{i},{j},{k}": len(slots[i].keys() & slots[j].keys() & slots[k].keys())
               for i, j, k in combinations(range(len(slots)), 3)}
    return dict(nnz=[len(d) for d in slots], union_slots=len(union), overlapping_slots=overlap,
                overlap_fraction=overlap / max(len(union), 1), pairwise=pairs, triple=triples,
                sign_conflict_slots=sum(min(v) < 0 < max(v) for v in occupancy.values()),
                max_abs_sum_code=max((abs(sum(v)) for v in occupancy.values()), default=0))


def lora_overlap_stats(states):
    """Stream one linear at a time, including the actual ternary product when applicable."""
    nnz = [0] * len(states)
    union = overlap = sign_conflicts = 0
    pairs = {f"{i},{j}": dict(slots=0, sign_conflicts=0) for i, j in combinations(range(len(states)), 2)}
    triples = {f"{i},{j},{k}": 0 for i, j, k in combinations(range(len(states)), 3)}
    for name in states[0]["entries"]:
        signs = []
        for i, state in enumerate(states):
            e = state["entries"][name]
            # CPU metadata calculation only; no new model or serving path.
            W = torch.empty((e["B"].shape[0], e["A"].shape[1]))
            sign = E.ResidentParam(W, e, validate_alpha=False).delta_w().sign().to(torch.int8)
            nnz[i] += int(sign.ne(0).sum())
            signs.append(sign)
        live = [s.ne(0) for s in signs]
        count = torch.stack(live).sum(0)
        union += int(count.gt(0).sum())
        overlap += int(count.gt(1).sum())
        sign_conflicts += int((torch.stack(signs).amin(0).lt(0) & torch.stack(signs).amax(0).gt(0)).sum())
        for i, j in combinations(range(len(states)), 2):
            both = live[i] & live[j]
            pairs[f"{i},{j}"]["slots"] += int(both.sum())
            pairs[f"{i},{j}"]["sign_conflicts"] += int((both & signs[i].ne(signs[j])).sum())
        for i, j, k in combinations(range(len(states)), 3):
            triples[f"{i},{j},{k}"] += int((live[i] & live[j] & live[k]).sum())
    return dict(nnz=nnz, union_slots=union, overlapping_slots=overlap, overlap_fraction=overlap/max(union, 1),
                pairwise=pairs, triple=triples, sign_conflict_slots=sign_conflicts,
                max_abs_sum_code=None, values="LoRA effective delta")


def merge_patches(states, rule="sum_clip"):
    E.eda.require_same(states, "merged patch banks")
    if rule not in ("sum_clip", "last_write"):
        raise ValueError("unknown merge rule")
    entries = {}
    for name, ref in states[0]["entries"].items():
        codes = {}
        for state in states:
            e = state["entries"][name]
            for idx, val in zip(e["idx"].tolist(), e["val"].tolist()):
                codes[idx] = codes.get(idx, 0) + val if rule == "sum_clip" else val
        # Last-write means the last nonzero write wins; absent sparse codes do not erase.
        indices = sorted(i for i, v in codes.items() if v)
        entries[name] = dict(ref, idx=torch.tensor(indices, dtype=torch.int32),
                             val=torch.tensor([max(-1, min(1, codes[i])) for i in indices], dtype=torch.int8))
        if any("address_start_idx" in state["entries"][name] for state in states):
            baseline = {}
            for state in states:
                e = state["entries"][name]
                for idx, val in zip(e.get("address_start_idx", torch.empty(0)).tolist(),
                                    e.get("address_start_val", torch.empty(0)).tolist()):
                    baseline[idx] = baseline.get(idx, 0) + val if rule == "sum_clip" else val
            indices = sorted(i for i, val in baseline.items() if val)
            entries[name]["address_start_idx"] = torch.tensor(indices, dtype=torch.int32)
            entries[name]["address_start_val"] = torch.tensor([max(-1, min(1, baseline[i])) for i in indices], dtype=torch.int8)
        for axis in ("rows", "cols"):
            key = "unit_reset_" + axis
            reset = sorted({int(i) for state in states
                            for i in state["entries"][name].get(key, torch.empty(0)).tolist()})
            if reset:
                entries[name][key] = torch.tensor(reset, dtype=torch.long)
    return dict(format=states[0]["format"], meta=dict(states[0]["meta"], merge=rule), entries=entries)


class BankIndexer:
    """One training centroid per resident bank, centered by anchor text only."""
    def __init__(self, keys, center, owners):
        self.keys, self.center, self.owners = keys, center, tuple(owners)

    @classmethod
    def fit(cls, features, labels, general):
        x, g = (torch.as_tensor(v).detach().double().cpu() for v in (features, general))
        y = torch.as_tensor(labels).cpu()
        if (x.ndim != 2 or g.ndim != 2 or not len(x) or not len(g)
                or x.shape[1] != g.shape[1] or y.shape != (len(x),)
                or not torch.isfinite(x).all() or not torch.isfinite(g).all()):
            raise ValueError("invalid indexer training data")
        owners = sorted(set(y.tolist()))
        center = g.mean(0)
        keys = torch.stack([x[y == p].mean(0) - center for p in owners])
        if (keys.norm(dim=-1) < 1e-12).any():
            raise ValueError("bank key is indistinguishable from general-text mean")
        return cls(F.normalize(keys, dim=-1), center, owners)

    def scores(self, x):
        x = torch.as_tensor(x).detach().double().cpu()
        return F.normalize(x - self.center, dim=-1) @ self.keys.T

    def select(self, x, threshold=None, top_k=1):
        return self.select_scores(self.scores(x), threshold, top_k)

    def select_scores(self, scores, threshold=None, top_k=1):
        if not 1 <= top_k <= len(self.owners):
            raise ValueError("top-k outside resident bank count")
        if threshold is not None and (not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise ValueError("cosine threshold must lie in [-1,1]")
        # Stable ordering makes tied scores deterministic by owner ID.
        order = scores.argsort(dim=-1, descending=True, stable=True)
        return [tuple(sorted(self.owners[i] for i in row[:top_k].tolist()
                             if threshold is None or float(score[i]) >= threshold))
                for row, score in zip(order, scores)]


class HierarchicalIndexer:
    """Two centered cosine indexes; stable global beam of summed log affinities.

    Cosines map to positive affinities (s+1)/2 before taking logs. They are
    ranking scores, not calibrated probabilities. Thresholds remain cosines.
    """
    def __init__(self, person, children, groups):
        self.person, self.children, self.groups = person, children, groups

    @classmethod
    def fit(cls, x1, x2, labels, general1, general2, groups):
        labels = torch.as_tensor(labels, dtype=torch.long)
        people = labels // groups
        parent = BankIndexer.fit(x1, people, general1)
        children = {p: BankIndexer.fit(torch.as_tensor(x2)[people == p], labels[people == p], general2)
                    for p in parent.owners}
        return cls(parent, children, groups)

    def select(self, x1, x2, threshold1, threshold2, top_k1=2, top_k2=2, beam=4, trace=False):
        if beam < 1 or not 1 <= top_k2 <= min(len(c.owners) for c in self.children.values()):
            raise ValueError("invalid child top-k or global beam")
        pscores = self.person.scores(x1)
        parents = self.person.select_scores(pscores, threshold1, top_k1)
        if len(x2) != len(parents):
            raise ValueError("hierarchical feature rows must align")
        routes, costs, proposals = [], [], []
        for row, selected in enumerate(parents):
            paths, child_routes = [], {}
            cost = len(self.person.owners)
            for p in selected:
                child = self.children[p]
                x = torch.as_tensor(x2)[row:row+1]
                scores = child.scores(x)[0]
                picked = child.select_scores(scores[None], threshold2, top_k2)[0]
                child_routes[p] = picked
                cost += len(child.owners)
                s1 = float(pscores[row, self.person.owners.index(p)])
                for bank in picked:
                    s2 = float(scores[child.owners.index(bank)])
                    score = math.log(max((s1+1)/2, 1e-12)) + math.log(max((s2+1)/2, 1e-12))
                    paths.append((score, bank))
            paths.sort(key=lambda pair: (-pair[0], pair[1]))
            routes.append(tuple(sorted(bank for _, bank in paths[:beam])))
            costs.append(cost)
            proposals.append(child_routes)
        if trace:
            return dict(routes=routes, parents=parents, children=proposals, comparisons=costs)
        return routes


class DGIndexer:
    """Train-only centered ternary expansion, binary top-k, one logistic key per bank."""
    def __init__(self, projection, center, k, owners, weight, bias):
        self.projection, self.center, self.k = projection, center, k
        self.owners, self.weight, self.bias = tuple(owners), weight, bias

    def codes(self, features):
        x = torch.as_tensor(features).detach().float().cpu()
        if x.ndim != 2 or x.shape[1] != len(self.center) or not torch.isfinite(x).all():
            raise ValueError("invalid DG features")
        # Keep the fixed projection int8; bound temporary FP32 projection memory.
        expanded = torch.cat([(x - self.center) @ block.float().T
                              for block in self.projection.split(1024)], dim=1)
        indices = expanded.argsort(dim=1, descending=True, stable=True)[:, :self.k]
        return torch.zeros_like(expanded).scatter_(1, indices, 1.)

    @classmethod
    def fit(cls, features, labels, attrs, general, expansion=8, density=1/3,
            sparsity=.015, seed=0, steps=200, l2=1e-3, dense=False):
        x, g = (torch.as_tensor(v).detach().float().cpu() for v in (features, general))
        y = torch.as_tensor(labels, dtype=torch.long).cpu()
        if (x.ndim != 2 or g.ndim != 2 or not len(x) or not len(g)
                or x.shape[1] != g.shape[1] or y.shape != (len(x),) or len(attrs) != len(x)
                or not torch.isfinite(x).all() or not torch.isfinite(g).all()):
            raise ValueError("invalid DG training data")
        if (not isinstance(expansion, int) or expansion < 1 or not 0 < density <= 1
                or not 0 < sparsity <= 1 or steps < 1 or not math.isfinite(l2) or l2 < 0):
            raise ValueError("invalid DG parameters")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        m = expansion * x.shape[1]
        projection = None
        if not dense:
            draw = torch.rand((m, x.shape[1]), generator=rng)
            projection = (draw < density/2).to(torch.int8) - (draw >= 1-density/2).to(torch.int8)
        owners = sorted(set(y.tolist()))
        result = (DenseLogisticIndexer(None, g.mean(0), 0, owners, None, None) if dense else
                  cls(projection, g.mean(0), max(1, round(sparsity*m)), owners, None, None))
        if dense:
            result.scale = x.std(0, unbiased=False).clamp_min(1e-6)
            m = x.shape[1]
        code = result.codes(torch.cat((x, g)))
        target = torch.stack([torch.cat(((y == p).float(), torch.zeros(len(g)))) for p in owners], 1)
        importance = torch.ones_like(target)
        for col, person in enumerate(owners):
            own_attrs = {attr for attr, pid in zip(attrs, y.tolist()) if pid == person}
            # Every other person is negative; same-attribute impostors get extra weight.
            importance[:len(x), col] = torch.tensor([
                2. if pid != person and attr in own_attrs else 1.
                for attr, pid in zip(attrs, y.tolist())])
            # Balance positive/negative mass separately for every bank.
            for positive in (False, True):
                mask = target[:, col].bool() == positive
                importance[mask, col] /= importance[mask, col].sum().clamp_min(1)
        weight = torch.zeros((len(owners), m), requires_grad=True)
        bias = torch.zeros(len(owners), requires_grad=True)
        opt = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (F.binary_cross_entropy_with_logits(F.linear(code, weight, bias), target,
                                                       reduction="none") * importance).sum() / len(owners)
            loss = loss + l2 * weight.square().sum() / len(owners)
            loss.backward()
            return loss
        opt.step(closure)
        result.weight, result.bias = weight.detach(), bias.detach()
        return result

    def scores(self, x):
        return F.linear(self.codes(x), self.weight, self.bias)

    def select(self, x, threshold=0., top_k=None):
        scores = self.scores(x)
        budget = len(self.owners) if top_k is None else top_k
        return [tuple(sorted(self.owners[i] for i in row.argsort(descending=True, stable=True)[:budget].tolist()
                             if float(row[i]) > threshold)) for row in scores]

    def storage_bytes(self):
        return 8*len(self.owners) + sum(v.numel()*v.element_size() for v in vars(self).values()
                                      if torch.is_tensor(v))


class DenseLogisticIndexer(DGIndexer):
    """Same independent BCE, hard negatives, L2 and optimization as DG."""
    def codes(self, features):
        return (torch.as_tensor(features).detach().float().cpu()-self.center)/self.scale

    @classmethod
    def fit(cls, features, labels, attrs, general, steps=200, l2=1e-3):
        return DGIndexer.fit(features, labels, attrs, general, steps=steps, l2=l2, dense=True)


def selection_metrics(routes, expected):
    if len(routes) != len(expected):
        raise ValueError("routes and labels must align")
    n = len(routes)
    return dict(n=n, accuracy=sum(set(r) == set(e) for r, e in zip(routes, expected)) / max(n, 1),
                recall=sum(len(set(r) & set(e)) for r, e in zip(routes, expected)) /
                       max(sum(len(set(e)) for e in expected), 1),
                abstention_rate=sum(not r for r in routes) / max(n, 1))


def question_sequence(tok, ids, bos=E.BOS):
    """Strip an answer cue for routing only; preserve full prompts for GEN."""
    text = tok.decode(ids[int(bos is not None):])
    return E.encode(tok, text[:text.rindex("?") + 1], bos) if "?" in text else list(ids)


def gold_routes(items, facts):
    """QA target banks: the reference fact supplies hop 1, not an extra bank."""
    return [tuple(sorted({facts[fi].pid for fi in (it["fis"] if "fis" in it else [it["fi"]])}))
            for it in items]


def address_training(model, tok, facts, anchor_tokens, seed, layer, dev):
    train = [it for it in S.build_panels(tok, facts, E.BOS, question_train=False)["train"]
             if facts[it["fi"]].attr not in R.heldout_attributes(facts)]
    wins, positions, _ = S.general_windows(anchor_tokens, seed, E.BOS)
    return (train, indexer_features(model, [it["p"] for it in train], layer, dev),
            indexer_features(model, wins, layer, dev, positions))


def address_eval_features(model, tok, groups, layer, dev):
    # Last question token, including multi-turn C21 pronouns, before Assistant/Answer cues.
    return {name: indexer_features(model, [it["p"] if name == "general" else question_sequence(tok, it["p"], E.BOS)
                                         for it in group], layer, dev)
            for name, group in groups.items() if group and name != "indirect"}


def ceiling_metrics(branches, model, tok, facts, groups, routes, dev, max_new, k, bs,
                    oracle_first_hop=False, bos=E.BOS):
    """Conditional success uses a separate explicit first-hop entity GEN probe.

    It is an observable diagnostic, not a claim about the model's hidden reasoning.
    For the named-entity intervention, entity resolution is supplied by construction.
    """
    def generate(items, selected, limit):
        if branches is None:
            return E.greedy_generate(model, [it["p"] for it in items], dev, limit, bs)
        return routed_generations(branches, tok, items, selected, dev, limit, bs)

    report = dict(address_accuracy={}, pronoun_qa={})
    conflicts = E.conflict_map(facts)
    for name in E.progress(("test", "pronoun", "pronoun_recent", "two_fact", "g_set", "two_hop"), 'branch_eval:ceiling_metrics'):
        items = groups.get(name, [])
        selected = routes.get(name, []) if routes is not None else None
        report["address_accuracy"][name] = (selection_metrics(selected, gold_routes(items, facts))
                                                if selected is not None else None)
        if oracle_first_hop and name == "two_hop":
            items = E.oracle_first_hop_items(tok, facts, items, bos)
        limit = (40 if name == "two_fact" else
                 max_new or max((len(it["a"]) for it in items), default=0) + k + 2)
        gens = generate(items, selected, limit) if items else []
        if name in ("two_fact", "g_set"):
            report[name] = dict(n=len(items),
                binding=sum(R.binding_score(tok.decode(g), it, facts) for it, g in zip(items, gens))/max(len(items), 1),
                strict=sum(E.score_two_fact_generation(tok, g, it["vals"], k)["both"]
                           for it, g in zip(items, gens)) / max(len(items), 1),
                lenient=sum(E.score_two_fact_lenient(tok, g, it["vals"])
                            for it, g in zip(items, gens)) / max(len(items), 1))
            continue
        hits = [E.score_generation(tok, g, it["val"], k, conflicts.get(it["fi"], ()))['clean']
                for it, g in zip(items, gens)]
        score = dict(n=len(items), recall=sum(hits)/max(len(items), 1))
        if name == "test":
            report["single_fact_gen"] = score
        elif name.startswith("pronoun"):
            report["pronoun_qa"][name] = score
        else:
            probes = E.first_hop_items(tok, facts, groups[name], bos)
            if oracle_first_hop:
                resolved = [True] * len(probes)
            else:
                first_limit = max_new or max((len(it["a"]) for it in probes), default=0) + k + 2
                first_gens = generate(probes, selected, first_limit) if probes else []
                resolved = [E.score_generation(tok, g, it["val"], k, it["conflicts"])['clean']
                            for it, g in zip(probes, first_gens)]
            n_resolved = sum(resolved)
            score.update(entity_resolution_accuracy=n_resolved/max(len(items), 1),
                         n_resolved=n_resolved, conditional_success=(
                             sum(hit and ok for hit, ok in zip(hits, resolved))/n_resolved
                             if n_resolved else None),
                         resolution_measure="supplied gold entity" if oracle_first_hop else
                                            "separate first-hop entity generation with the same bank route")
            report["two_hop"] = score
    return report


def ceiling_summary(reports):
    baseline = reports["as_trained"]
    gains, decisions = {}, {}
    for mode in ("oracle_entity", "oracle_first_hop"):
        if mode in reports:
            gains[mode] = {}
    for task, panel, field in (("two_fact", "two_fact", "binding"),
                               ("g_set", "g_set", "binding"), ("two_hop", "two_hop", "recall")):
        if panel not in baseline or not baseline[panel]["n"]:
            continue
        # Legacy reports remain readable, but only binding scores support composition verdicts.
        if field not in baseline[panel]:
            decisions[task] = dict(decision="unmeasured", reason="binding score missing")
            continue
        base = baseline[panel][field]
        oracles = [reports[m][panel][field] for m in gains]
        oracle = max(oracles, default=base)
        headroom = oracle-base
        decision = ("ceiling reached" if base >= oracle-.02-1e-12 else
                    "insufficient headroom" if 1-base < .10-1e-12 else
                    "addressing rescue" if headroom >= .10-1e-12 else "value-use bottleneck")
        decisions[task] = dict(baseline=base, absolute_oracle_success=oracle, headroom=headroom, available_headroom=1-base, decision=decision)
        for mode in gains:
            gains[mode][task] = reports[mode][panel][field]-base
    return dict(subarms=reports, oracle_gains=gains, per_task=decisions,
                value_use_bottleneck=any(d['decision'] == 'value-use bottleneck' for d in decisions.values()),
                rescue_threshold=.10, ceiling_tolerance=.02, values_fixed=True)


def print_ceiling(report):
    print("ADDR-CEILING " + json.dumps(report), flush=True)
    if report["value_use_bottleneck"]:
        print("VALUE-USE-BOTTLENECK: see per-task oracle success and available headroom", flush=True)


def addressing_ceiling(branches, tok, facts, groups, anchor_tokens, seed, layers, dev, a):
    reports = {}
    for mode in E.progress(("as_trained", *(f"post_resolution:{li}" for li in layers),
                 "oracle_entity", "oracle_first_hop"), 'branch_eval:addressing_ceiling'):
        if mode.startswith("post_resolution:"):
            layer = int(mode.split(":")[1])
            train, x, general = address_training(branches.model, tok, facts, anchor_tokens, seed, layer, dev)
            indexer = BankIndexer.fit(x, [facts[it["fi"]].pid for it in train], general)
            features = address_eval_features(branches.model, tok, groups, layer, dev)
            routes = {name: indexer.select(x) for name, x in features.items()}
            routes.update({name: [] for name, items in groups.items() if not items})
        else:
            routes = {name: [tuple(sorted(branches.states))] * len(items) if mode == "as_trained"
                      else gold_routes(items, facts) for name, items in groups.items()}
        reports[mode] = ceiling_metrics(branches, branches.model, tok, facts, groups, routes, dev,
            a.gen_max_new, a.gen_stop_k, a.gen_bs, oracle_first_hop=mode == "oracle_first_hop", bos=E.BOS)
        if mode.startswith("post_resolution:"):
            reports[mode].update(layer=layer, layer_indexing="zero-based full block input",
                                 selection="dense cosine top1; last question token")
    return ceiling_summary(reports)


@torch.no_grad()
def indexer_features(model, sequences, layer, dev, positions=None):
    enabled = E.FSAParam.enabled
    try:
        E.set_enabled(False)
        return S.collect_residuals(model, sequences, [layer], dev, positions)[layer]
    finally:
        E.set_enabled(enabled)


class Probe:
    """Standardized CPU multinomial logistic regression with L2, fit by LBFGS."""
    def __init__(self, mean, scale, weight, bias):
        self.mean, self.scale, self.weight, self.bias = mean, scale, weight, bias

    def predict(self, x):
        x = torch.as_tensor(x).detach().float().cpu()
        return F.linear((x - self.mean) / self.scale, self.weight, self.bias).argmax(-1)


def fit_router(x, labels, classes, steps=200, l2=1e-3):
    x = torch.as_tensor(x).detach().float().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).cpu()
    if x.ndim != 2 or not len(x) or len(labels) != len(x) or not torch.isfinite(x).all():
        raise ValueError("invalid router examples")
    if set(labels.tolist()) != set(range(classes)) or steps < 1 or l2 < 0:
        raise ValueError("router training must include every class")
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    weight = torch.zeros(classes, x.shape[1], requires_grad=True)
    bias = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    closure_steps = 0
    def closure():
        nonlocal closure_steps
        closure_steps += 1
        E.heartbeat('branch_router_fit', closure_steps, steps)
        opt.zero_grad()
        loss = F.cross_entropy(F.linear(x, weight, bias), labels) + l2 * weight.square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    return Probe(mean, scale, weight.detach(), bias.detach())


@torch.no_grad()
def base_features(model, sequences, layers, dev, positions=None):
    """All C/I/DG captures use identical zero-based full block inputs."""
    enabled = E.FSAParam.enabled
    try:
        E.set_enabled(False)
        return S.collect_residuals(model, sequences, layers, dev, positions)
    finally:
        E.set_enabled(enabled)


def layer_mapping(model):
    return [dict(moe_index=i, full_block=next(j for j, block in enumerate(model.layers)
                 if any(m is moe for m in block.modules())), module=name)
            for i, (name, moe) in enumerate(E.moe_layers(model))]


def composition_items(tok, facts, bos, n=60, split="test"):
    groups = defaultdict(list)
    for fi, f in enumerate(facts):
        groups[f.attr].append(fi)
    items = []
    for attr, fis in sorted(groups.items()):
        for fi, fj in combinations(fis, 2):
            a, b = facts[fi], facts[fj]
            if a.pid == b.pid:
                continue
            prompts = ((f"Please list {a.name}'s {attr} and {b.name}'s {attr}.",) if split == "val" else (f"Is {a.name}'s {attr} the same as {b.name}'s?",
                           f"What are the {attr}s of {a.name} and {b.name}?"))
            for prompt in prompts:
                answer = (f"No, {a.name}'s {attr} is {a.val} while {b.name}'s is {b.val}."
                          if a.val != b.val else f"Yes, {a.name}'s {attr} is {a.val} and {b.name}'s is {b.val}.")
                prompt += " Answer with each person's name, attribute, and value."
                items.append(dict(prompt=prompt, p=E.encode(tok, prompt, bos), fis=[fi, fj],
                                  vals=[a.val, b.val], a=tok.encode(" " + answer, add_special_tokens=False).ids))
    return random.Random(160016).sample(items, min(n, len(items)))


def contains_value(text, value):
    # Token boundaries avoid accidentally counting a value that is a substring of another.
    import re
    return re.search(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", text.lower()) is not None


class LoRABankSum(torch.nn.Module):
    """Sum active saved products in fp32, then cast once (no clipping or product re-quantization)."""
    def __init__(self, residents):
        super().__init__()
        self.residents = torch.nn.ModuleList(residents)

    def forward(self, W):
        active = [m for m in self.residents if m.enabled]
        if not E.LoRAParam.enabled or not active:
            return W
        delta = active[0].delta_w()
        for m in active[1:]:
            delta = delta + m.delta_w()
        return W + delta.to(W.dtype)


class Branches:
    def __init__(self, model, states, bytes_by_person, rule):
        E.eda.require_same(states.values(), "serving banks")
        if hasattr(model, "_fsa_eda_backend"):
            for state in states.values():
                E.eda.require_patch_backend(state, model, "serving bank")
        self.model, self.states, self.bytes, self.rule = model, states, bytes_by_person, rule
        self.cache = {}
        self.lora_banks = None
        if next(iter(states.values())).get("meta", {}).get("method") == "lora" or rule == "sum_delta":
            banks, _ = E.attach_residents(model, list(states.values()), enabled=False,
                                          collect_slots=False, follow_enabled=True)
            self.lora_banks = dict(zip(states, banks))
            # One addition per linear is important with bf16 base weights.
            for _, lin in E.patched_linears(model):
                residents = list(lin.parametrizations.weight)
                if not all(isinstance(m, E.ResidentParam) for m in residents):
                    # FSA sum_delta starts with its validated zero student.
                    if rule != 'sum_delta' or not all(isinstance(m, E.FSAParam) or isinstance(m, E.ResidentParam) for m in residents):
                        raise ValueError("delta bank evaluator requires an unpatched base model")
                residents = [m for m in residents if isinstance(m, E.ResidentParam)]
                E.P.remove_parametrizations(lin, "weight", leave_parametrized=False)
                E.P.register_parametrization(lin, "weight", LoRABankSum(residents))

    def activate(self, people):
        key = tuple(sorted(set(people)))
        if getattr(self.model, "_c16_branch_key", None) == (id(self), key):
            return
        if self.lora_banks is not None:
            for person, bank in self.lora_banks.items():
                for pm in bank:
                    pm.enabled = person in key
            E.set_enabled(True)
            self.model._c16_branch_key = (id(self), key)
            return
        # Every load overwrites all masters: switching cannot retain an old branch.
        if key not in self.cache:
            if key:
                self.cache[key] = merge_patches([self.states[p] for p in key], self.rule)
            else:
                self.cache[key] = dict(next(iter(self.states.values())))
                self.cache[key]["entries"] = {name: dict(e, idx=torch.empty(0, dtype=torch.int32),
                                                           val=torch.empty(0, dtype=torch.int8))
                                              for name, e in self.cache[key]["entries"].items()}
                for entry in self.cache[key]["entries"].values():
                    entry.pop("unit_reset_rows", None)
                    entry.pop("unit_reset_cols", None)
                    entry.pop("address_start_idx", None)
                    entry.pop("address_start_val", None)
        E.load_patch(self.model, self.cache[key])
        detectors = self.address_detectors(key)
        E.snapshot_addresses(self.model, {"resident": {name.rsplit(".", 1)[0]: d["rows"]
                                                       for name, d in detectors.items()}})
        E.set_enabled(True)
        self.model._c16_branch_key = (id(self), key)

    def active_bytes(self, people):
        return sum(self.bytes[p] for p in set(people))

    def address_detectors(self, people):
        detectors = {}
        for person in people:
            meta = self.states[person]["meta"]
            selected = meta.get("wake", {}).get("detectors", {}) if meta.get("address_units") else {}
            if meta.get("units_per_fact"):
                selected = {}
                for fact in meta["unit_allocation"]["assignments"].values():
                    for bank, units in fact.items():
                        rows = selected.setdefault(bank + ".up_proj", dict(rows=[]))["rows"]
                        rows[:] = sorted(set(rows) | set(units))
            for name, entry in selected.items():
                rows = detectors.setdefault(name, dict(rows=[]))["rows"]
                rows[:] = sorted(set(rows) | set(entry["rows"]))
        return detectors


def three_metrics(branches, tok, facts, items, g_items, general, dev, max_new, stop_k, bs):
    """Evaluate the existing files singly and with every page simultaneously resident.

    This is independent of oracle/router serving decisions: E and F therefore both
    expose the same direct composition measurement, including the two-person G set.
    """
    individual, address, moved = {}, {}, {}
    try:
        for person in E.progress(branches.states, 'branch_eval:three_metrics'):
            branches.activate((person,))
            own = [it for it in items if facts[it["fi"]].pid == person]
            individual[str(person)] = E.gen_eval(branches.model, tok, own, dev, max_new, stop_k, bs,
                                                 E.conflict_map(facts))["recall"]
            address[str(person)] = E.address_accuracy(branches.model, facts, items,
                    branches.address_detectors((person,)), general, dev, person=person)
            moved[str(person)] = E.address_moved(branches.model) if branches.model is not None else 0
        branches.activate(tuple(branches.states))
        composed = E.gen_eval(branches.model, tok, items, dev, max_new, stop_k, bs, E.conflict_map(facts))
        g = E.composition_recall(branches.model, tok, g_items, dev, max_new, stop_k, bs)
        composed_address = {str(person): E.address_accuracy(branches.model, facts, items,
                            branches.address_detectors((person,)), general, dev, person=person)
                            for person in branches.states if branches.address_detectors((person,))}
        unit_store = E.unit_bytes(branches.model)
        composed_moved = E.address_moved(branches.model) if branches.model is not None else 0
    finally:
        branches.activate(())
    return dict(address_accuracy=dict(individual=address, composed=composed_address),
                individual_recall=dict(by_memory=individual, mean=sum(individual.values()) / max(len(individual), 1)),
                composed_recall=dict(n=composed["n"], recall=composed["recall"], g_set=g), unit_bytes=unit_store,
                address_moved=dict(individual=moved, composed=composed_moved))


def routed_firing(branches, items, routes, dev):
    decisions = []
    for it, people in E.progress(zip(items, routes), 'branch_eval:routed_firing', total=min(len(items),len(routes))):
        if not people:
            decisions.append(False)
            continue
        detectors = branches.address_detectors(people)
        if detectors:
            branches.activate(people)
            decisions.append(E.address_fired(branches.model, it["p"], detectors, dev))
        else:
            decisions.append(True)
    return decisions


def person_routes(items, facts, compose=False):
    result = []
    for it in items:
        fis = list(it.get("fis", [it.get("fi")]))
        if "reference_fi" in it:
            fis.append(it["reference_fi"])
        people = tuple(sorted({facts[fi].pid for fi in fis}))
        result.append(people if compose else people[:1])
    return result


@torch.no_grad()
def routed_generations(branches, tok, items, routes, dev, max_new, bs):
    grouped, result = defaultdict(list), [None] * len(items)
    for i, key in E.progress(enumerate(routes), 'branch_eval:routed_generations', total=len(routes)):
        grouped[tuple(key)].append(i)
    for key, indices in E.progress(grouped.items(), 'branch_eval:routed_generations'):
        branches.activate(key)
        gens = E.greedy_generate(branches.model, [items[i]["p"] for i in indices], dev, max_new, bs)
        for i, gen in E.progress(zip(indices, gens), 'branch_eval:routed_generations', total=min(len(indices),len(gens))):
            result[i] = gen
    return result


def generation_summary(tok, items, gens, conflicts, k, bootstrap=0, seed=0):
    scores = [E.score_generation(tok, g, it["val"], k, conflicts.get(it["fi"], ()))
              for it, g in zip(items, gens)]
    n = max(len(items), 1)
    result = dict(n=len(items), recall=sum(s["clean"] for s in scores) / n,
                  match=sum(s["match"] for s in scores) / n, strict=sum(s["strict"] for s in scores) / n)
    if bootstrap:
        result["ci95"] = {key: E.cluster_bootstrap(items, [s[field] for s in scores], bootstrap, seed)
                          for key, field in (("recall", "clean"), ("match", "match"), ("strict", "strict"))}
        result['fact_observations'] = [dict(fi=it['fi'], value=int(s['clean'])) for it,s in zip(items,scores)]
    ci = [i for i, it in enumerate(items) if it["fi"] in conflicts]
    if ci:
        result.update(n_conflict=len(ci), recall_conflict=sum(scores[i]["clean"] for i in ci) / len(ci),
                      confusion=sum(scores[i]["confused"] for i in ci) / len(ci))
    groups = defaultdict(list)
    for it, sc in zip(items, scores):
        groups[(it["attr"], it["ti"])].append(sc["clean"])
    multi = [v for v in groups.values() if len(v) > 1]
    result.update(n_groups=len(multi), all_people_correct=sum(all(v) for v in multi) / max(len(multi), 1))
    if any('fact_class' in it for it in items):
        result['by_fact_class'] = E.C50.split_scores(items, [dict(recall=s['clean'], match=s['match'],
                                                               strict=s['strict']) for s in scores])
    return result


@torch.no_grad()
def routed_tf(branches, items, routes, dev, bs):
    groups = defaultdict(list)
    for it, key in E.progress(zip(items, routes), 'branch_eval:routed_tf', total=min(len(items),len(routes))):
        groups[tuple(key)].append(it)
    result = dict(exact_match=0., answer_logprob=0., answer_bits=0., answer_tokens=0)
    for key, group in E.progress(groups.items(), 'branch_eval:routed_tf'):
        branches.activate(key)
        score = E.tf_eval(branches.model, group, dev, bs)
        for k in E.progress(result, 'branch_eval:routed_tf'):
            result[k] += score[k] * (len(group) / max(len(items), 1) if k in ("exact_match", "answer_logprob") else 1)
    return result


@torch.no_grad()
def damage(branches, panels, base_nll, routes, dev):
    out = {}
    for name, wins in E.progress(panels.items(), 'branch_eval:damage'):
        rows = []
        for win, key in E.progress(zip(wins, routes[name]), 'branch_eval:damage', total=min(len(wins),len(routes[name]))):
            branches.activate(key)
            rows.extend(E.nll_windows(branches.model, win[None], dev, bs=1))
        base = torch.tensor(base_nll[name])
        delta = torch.tensor(rows) - base
        out[name] = dict(dnll=float(delta.mean()), dnll_se=float(delta.std() / math.sqrt(len(delta))),
                         base_nll=float(base.mean()), none_rate=sum(not key for key in routes[name]) / len(wins),
                         active_bytes_mean=sum(branches.active_bytes(key) for key in routes[name]) / len(wins))
    return out


def branch_base_cache_key(a, meta):
    identity = dict(eda_backend=E.eda.selected(a), export=str(a.export.resolve()), export_step=a.export_step, world=meta,
                    version=E.base_eval_version(), branch_code=E.file_sha(__file__),
                    tokenizer=E.file_sha(E.DATA / "tokenizer.json"),
                    panels={name: E.file_sha(path) for name, path in
                            (("WikiText", a.eval_set), ("FineWeb", a.fineweb), ("code", a.code))},
                    options={key: getattr(a, key) for key in ("qa_n", "gen_max_new", "gen_stop_k", "gen_bs", "fast_moe",
                                                             "bench", "subject_text")})
    if a.fast_moe != "off":
        identity["fast_code"] = E.file_sha(Path(E.__file__).with_name("fast_moe.py"))
    if a.bench:
        identity["bench_sha"] = {name: E.file_sha(path) for name, path in E.benchmark_paths(a.bench_dir).items()}
    # A resident bank's trained weights do not affect the disabled base.
    identity["world"] = {key: meta.get(key) for key in IDENTITY}
    if getattr(a, 'fact_class_metrics', False):
        identity['options']['fact_class_metrics'] = True
        identity['c50_code'] = E.file_sha(E.C50.__file__)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def evaluate_branch_base(model, tok, panels, qa, revoke_input, dev, a, bench_items=None, subject_items=None):
    result = dict(base_nll={name: E.nll_windows(model, wins, dev, bs=1) for name, wins in panels.items()},
                  base_logits=E.logits_of(model, revoke_input).detach().clone(),
                  base_qa_probe=E.qa_probe_eval(model, tok, qa, dev, a.gen_max_new, a.gen_stop_k, a.gen_bs,
                                              fact_class_metrics=getattr(a, 'fact_class_metrics', False)))
    if bench_items is not None:
        result["base_bench"] = E.benchmark_eval(model, bench_items, dev)
    if subject_items is not None:
        result["base_subject"] = E.subject_nll(model, subject_items, dev)
    return result


def resident_benchmark(branches, groups, base, dev):
    """Resident-bank stress test; never substitute for a routed damage gate."""
    try:
        branches.activate(tuple(branches.states))
        return E.benchmark_summary(base, E.benchmark_eval(branches.model, groups, dev))
    finally:
        branches.activate(())


def routed_benchmark(branches, groups, base, route, dev):
    patched = {}
    try:
        for name, items in E.progress(groups.items(), 'branch_eval:routed_benchmark'):
            rows = []
            # Route once on the question; every answer choice receives the same route.
            for it, key in E.progress(zip(items, route(items)), 'branch_eval:routed_benchmark'):
                branches.activate(key)
                rows.append(E.benchmark_eval(branches.model, {name: [it]}, dev)[name])
            patched[name] = dict(n=len(rows), correct=[r['correct'][0] for r in rows],
                                 predicted=[r['predicted'][0] for r in rows])
        return E.benchmark_summary(base, patched)
    finally:
        branches.activate(())


def routed_subject_damage(branches, items, base, routes, dev):
    patched = []
    if len(items) != len(routes):
        raise ValueError("subject routes must align with windows")
    for it, key in E.progress(zip(items, routes), 'branch_eval:routed_subject_damage', total=min(len(items),len(routes))):
        branches.activate(key)
        patched.extend(E.subject_nll(branches.model, [it], dev))
    return E.subject_summary(items, base, patched)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    E.C50.add_arguments(p)
    E.eda.add_argument(p)
    E.eda.add_crosseval_argument(p)
    try:
        from . import in_topic as I
    except ImportError:
        import in_topic as I
    I.add_argument(p)
    p.add_argument('--c42-mode', choices=('dump', 'gen'), help='opt-in C42 cached selector study; defaults unchanged')
    p.add_argument('--c42-cache', type=Path, help='persistent frozen selector features for C42')
    p.add_argument('--c42-layers', default='4,26,28', help='C42 full-block inputs captured together')
    p.add_argument('--c42-parent-gate', action='store_true', help='pre-gate C42 LLR parents at LLR > 0 before beam expansion')
    p.add_argument("--learning-eval", action="store_true", help="C37 evaluation only, no selector fitting; first patch is focus bank")
    p.add_argument("--value-operands", action="store_true")
    p.add_argument("--dark-audit", action="store_true")
    p.add_argument("--evidence-skill", action="store_true", help="evaluate held-out oracle evidence with first patch")
    p.add_argument("--c37-control", action="store_true", help="C37 CPU mechanics controls without model IO")
    p.add_argument("--export", type=Path, default=E.EXPORT)
    p.add_argument("--export-step", type=int, default=E.EXPORT_STEP)
    p.add_argument("--mask-compare", type=Path, default=None)
    p.add_argument("--patches", nargs="+", type=Path, default=[], help="per-person final.pt files; H/FLAT use person/attribute-group banks")
    p.add_argument("--partition-patches", nargs="+", type=Path, default=[], help="K=3 disjoint per-person files")
    p.add_argument("--unit-patches", nargs="+", type=Path, default=[], help="C34 matching R11 whole-unit partition banks")
    p.add_argument("--decisive-eval", action="store_true", help="C34 suite for locked serving policies, 64 new tokens")
    p.add_argument("--pk-width", type=int, default=16, help="PK width per ternary codebook")
    p.add_argument("--arms", default="B,C,D,E,F,G")
    p.add_argument("--modes", nargs="+", choices=("oracle", "router", "sum"),
                   help="alias for full-mask arms B,C,D")
    p.add_argument("--merge", choices=("sum_clip", "last_write", "sum_delta"), default="sum_clip",
                   help="sum_delta adds both formats' saved deltas in fp32 then casts once; defaults are format-native")
    p.add_argument("--positive-control", action="store_true", help="run arm controls without model IO")
    p.add_argument("--negative-n", type=int, help="general nulls: default 10000 per split for C42 dump, 3000 otherwise")
    p.add_argument("--negative-text", type=Path, nargs="+", default=[], help="additional independent general-text token JSON sources")
    p.add_argument("--selector-l2", type=float, nargs="+", default=[.001, .01, .1])
    p.add_argument("--indexer-layers", help="validation layer candidates; defaults to configured layer plus available 4,26,28")
    p.add_argument("--dg-layers", help="validation layer candidates; defaults to configured layer plus available 4,26,28")
    p.add_argument("--indexer-layer", type=int, default=4, help="zero-based full block input")
    p.add_argument("--indexer-thresholds", type=float, nargs=3, default=[0.2, 0.5, 0.8])
    p.add_argument("--indexer-top-k", type=int, default=3, help="maximum selected banks in threshold modes")
    p.add_argument("--h-layer1", type=int, default=4, help="person index: zero-based full block input")
    p.add_argument("--h-layer2", type=int, default=26, help="attribute index: zero-based full block input")
    p.add_argument("--h-layers1", help="comma-separated person layers searched on validation")
    p.add_argument("--h-layers2", help="comma-separated attribute layers searched on validation")
    p.add_argument("--h-top-k1", type=int, default=2)
    p.add_argument("--h-top-k2", type=int, default=2)
    p.add_argument("--h-beam", type=int, default=4, help="H and FLAT maximum activated banks")
    p.add_argument("--h-thresholds1", type=float, nargs="+", default=[.2, .5, .8])
    p.add_argument("--h-thresholds2", type=float, nargs="+", default=[.2, .5, .8])
    p.add_argument("--ac-indexer-layers", default="26,28", help="AC post-resolution full block inputs")
    p.add_argument("--dg-layer", type=int, default=26, help="DG zero-based full block input (also try 4)")
    p.add_argument("--dg-expansion", type=int, default=8)
    p.add_argument("--dg-density", type=float, default=1/3)
    p.add_argument("--dg-sparsity", type=float, default=.015, help="fraction of expanded coordinates retained")
    p.add_argument("--dg-seed", type=int, default=0)
    p.add_argument("--dg-steps", type=int, default=200)
    p.add_argument("--bio-layer", type=int, default=26, help="INH/CONJ full block input")
    p.add_argument("--bio-steps", type=int, default=100)
    p.add_argument("--c29-max-per-kind", type=int, default=24)
    p.add_argument("--bio-budget", type=int, default=3, help="maximum banks activated per query")
    p.add_argument("--inh-exclusions", type=int, default=2)
    p.add_argument("--inh-distractors", type=int, default=10, help="random addresses with no value patch")
    p.add_argument("--inh-lambda", type=float, default=1.)
    p.add_argument("--slot-early-layer", type=int, default=4)
    p.add_argument("--slot-late-layers", default="26,28")
    p.add_argument("--slot-max", type=int, default=4)
    p.add_argument("--router-layer", default="4,6,8", help="comma-separated zero-based full block inputs")
    p.add_argument("--router-steps", type=int, default=200)
    p.add_argument("--router-none", type=int, default=600, help="disjoint 63-token TRAIN anchor windows")
    p.add_argument("--anchor", type=Path)
    p.add_argument("--eval-set", type=Path, help="WikiText token JSON")
    p.add_argument("--fineweb", type=Path)
    p.add_argument("--code", type=Path)
    p.add_argument("--fast-moe", choices=("off", "exact"), default="exact")
    p.add_argument("--gen-bs", type=int, default=16)
    p.add_argument("--gen-max-new", type=int, default=0)
    p.add_argument("--gen-stop-k", type=int, default=2)
    p.add_argument("--qa-n", type=int, default=120)
    p.add_argument("--bootstrap", type=int, default=0, help="fact-cluster bootstrap repeats for GEN/QA")
    p.add_argument("--base-cache", type=Path, default=None)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--bench-dir", type=Path, default=None)
    p.add_argument("--subject-text", action="store_true")
    p.add_argument("--g-n", type=int, default=60)
    p.add_argument("--out", type=Path)
    return p


def positive_control(arm):
    """Dependency-free review fixtures; runtime controls additionally test bank IO."""
    if arm in ('C34', 'PK'):
        if __package__:
            from . import decisive_eval as C34
        else:
            import decisive_eval as C34
        return C34.positive_control()
    if arm in ('H', 'FLAT'):
        if __package__:
            from . import hierarchical_banks as H
        else:
            import hierarchical_banks as H
        return H.positive_control(sys.modules[__name__], arm)
    if arm in ('INH', 'CONJ', 'SLOT'):
        if __package__:
            from . import biology_addressing as BIO
        else:
            import biology_addressing as BIO
        return BIO.positive_control(arm)
    if arm == 'G':
        facts = [E.Fact(0, 'Alice', 'her', 'she', 'instrument', 'viola'),
                 E.Fact(1, 'Beth', 'her', 'she', 'instrument', 'harp')]
        item = dict(fis=[0, 1])
        results = [R.binding_score(t, item, facts) for t in (
            "Alice's instrument is viola; Beth's instrument is harp.",
            "Alice's instrument is harp; Beth's instrument is viola.",
            "Alice's instrument is not viola; Beth's instrument is not harp.")]
        return dict(passed=results == [True, False, False], kind='correct, swapped and negated binding fixtures')
    if arm == 'AC':
        report = dict(two_fact=dict(n=1, binding=1.), two_hop=dict(n=1, recall=1.))
        result = ceiling_summary(dict(as_trained=report, oracle_entity=report))
        return dict(passed=all(d['decision']=='ceiling reached' for d in result['per_task'].values()),
                    kind='already-perfect baseline reaches ceiling')
    if arm in ('C', 'I', 'DG'):
        x = torch.eye(6)[:3].repeat_interleave(4,0)*4
        y, attrs, g = [0]*4+[1]*4+[2]*4, ['shared']*12, torch.zeros(64,6)
        if arm == 'C':
            probe = fit_router(torch.cat((x,g)), y+[3]*64, 4, 100)
            predictions = probe.predict(torch.cat((x,g))).tolist()
            passed = predictions == y+[3]*64
        elif arm == 'I':
            index = BankIndexer.fit(x,y,g)
            passed = index.select(x,.5) == [(p,) for p in y] and index.select(g,.5)==[()]*64
            passed = passed and index.select(x[:1]+x[4:5],.5,2)==[(0,1)]
        else:
            dg = DGIndexer.fit(x,y,attrs,g,expansion=8,sparsity=.15,steps=100)
            dense = DenseLogisticIndexer.fit(x,y,attrs,g,steps=100)
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))*.02
            passed = all(index.select(x+noise,top_k=1)==[(p,) for p in y] and not any(index.select(g)) for index in (dg,dense))
            passed = passed and bool((dg.codes(x).sum(1)==dg.k).all())
        return dict(passed=passed, kind='separable entity/null clusters; multibank or deterministic sparse cardinality')
    if arm in ('D','B','E','F'):
        if arm == 'F':
            x = torch.tensor([2.,3.])
            up = torch.eye(2); down = torch.eye(2)
            full = down @ (up@x).relu().square()
            solo = sum(down[:,i]*((up[i]@x).relu().square()) for i in range(2))
            passed = torch.equal(full, solo)
        elif arm == 'E':
            mask = torch.ones(8,8,dtype=torch.bool)
            parts = [E.partition_mask(mask,0,0,'up_proj',3,i,True) for i in range(3)]
            passed = all(not (parts[i]&parts[j]).any() for i,j in combinations(range(3),2))
            down = [E.partition_mask(mask,0,0,'down_proj',3,i,True) for i in range(3)]
            passed = passed and all(torch.equal(up, dn.T) for up,dn in zip(parts,down))
            passed = passed and torch.equal(sum(p.int() for p in parts),mask.int())
        else:
            entry = dict(idx=torch.tensor([0,1],dtype=torch.int32),val=torch.tensor([1,-1],dtype=torch.int8))
            bank = dict(format='e2e-fsa-patch-v1',meta={},entries={'fixture':entry})
            blank = dict(bank,entries={'fixture':dict(entry,idx=torch.empty(0,dtype=torch.int32),val=torch.empty(0,dtype=torch.int8))})
            passed = all(torch.equal(merge_patches([bank,blank,blank],rule)['entries']['fixture']['val'],entry['val'])
                         for rule in ('sum_clip','last_write'))
        return dict(passed=bool(passed), kind='empty bank identity / disjoint ownership / identical-input unit additivity')
    return dict(passed=False, kind='no registered review control for '+arm)


@E.contextmanager
def independent_solo(model, state):
    """Rebuild a single saved patch through the training parametrization, without the bank sum."""
    linears, saved = dict(model.named_modules()), []
    try:
        for name, entry in state['entries'].items():
            lin = linears[name]
            old = list(lin.parametrizations.weight)
            saved.append((lin, old))
            E.P.remove_parametrizations(lin, 'weight', leave_parametrized=False)
            W = lin.weight.detach()
            if entry['kind'] == 'lora':
                # Construction should not perturb any later random recipe.
                with torch.random.fork_rng(devices=[W.device.index] if W.is_cuda else []):
                    pm = E.LoRAParam(W, entry['A'].shape[0], entry.get('ternary', False))
                pm.code_space = entry.get('code_space', False)
                with torch.no_grad():
                    pm.A.copy_(entry['A'].to(W.device))
                    pm.B.copy_(entry['B'].to(W.device))
                if 'excluded_idx' in entry:
                    pm.register_buffer('excluded_idx', entry['excluded_idx'].long().to(W.device))
            else:
                pm = E.ResidentParam(W, entry)
            E.P.register_parametrization(lin, 'weight', pm)
        yield
    finally:
        for lin, old in saved:
            E.P.remove_parametrizations(lin, 'weight', leave_parametrized=False)
            for pm in old:
                E.P.register_parametrization(lin, 'weight', pm)
        model._c16_branch_key = None


def bank_control(branches, tok, facts, items, dev, a):
    """Independent standalone-load vs bank assembly, including effective weight bits."""
    passed, success, successful_fis = True, True, set()
    try:
        for person, state in E.progress(branches.states.items(), 'branch_eval:bank_control'):
            own = [it for it in items if facts[it['fi']].pid==person]
            limit = a.gen_max_new or max((len(it['a']) for it in own),default=0)+a.gen_stop_k+2
            if branches.lora_banks is None:
                E.load_patch(branches.model,state)
                E.set_enabled(True)
                solo = E.greedy_generate(branches.model,[it['p'] for it in own],dev, limit,a.gen_bs)
                solo_weights = E.weight_hashes(branches.model, state['entries'])
            else:
                E.set_enabled(True)
                with independent_solo(branches.model, state):
                    solo = E.greedy_generate(branches.model,[it['p'] for it in own],dev, limit,a.gen_bs)
                    solo_weights = E.weight_hashes(branches.model, state['entries'])
            E.set_enabled(True)
            # Force activation to reload after the direct comparison.
            branches.model._c16_branch_key = None
            routed = routed_generations(branches,tok,own,[(person,)]*len(own),dev,
                                        limit,a.gen_bs)
            passed = passed and solo==routed and solo_weights == E.weight_hashes(branches.model, state['entries'])
            hits = [E.score_generation(tok,g,it['val'],a.gen_stop_k,E.conflict_map(facts).get(it['fi'],()))['clean'] for it,g in zip(own,solo)]
            successful_fis.update(it['fi'] for it,hit in zip(own,hits) if hit)
            success = success and any(hits)
    finally:
        branches.activate(())
    return dict(passed=passed and success, solo_reproduced=passed, learned_fact_success=success, successful_fis=sorted(successful_fis),
                kind='independently loaded standalone patch vs bank assembly: weight bits and generation')


def learning_eval(a):
    """C37 resident-environment measurements; no serving policies/selectors."""
    if not a.patches or not a.out:
        raise SystemExit("--learning-eval needs --patches (focus first, siblings next) and --out")
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in a.patches]
    meta = states[0]["meta"]
    for state in E.progress(states, 'branch_eval:learning_eval'):
        E.eda.require_patch_backend(state, a, "learning bank")
        E.validate_patch_export(state["meta"], a.export, a.export_step)
        for entry in E.progress(state["entries"].values(), 'branch_eval:learning_eval'):
            if entry["kind"] != "fsa" or entry["fmt"] != "ternary" or entry["scale"] != "row":
                raise SystemExit("learning environments require forced-mask ternary banks")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    model, _, dev = E.eda.build_model(build_model, a, a.export, device="cuda:0", model_config=None,
                               step=a.export_step, expert_format="bf16")
    model._fsa_export, model._fsa_export_step = a.export, a.export_step
    model.eval()
    for parameter in E.progress(model.parameters(), 'branch_eval:learning_eval'):
        parameter.requires_grad_(False)
    E.attach(model, meta["layers"], "fsa_forced", 0, projs=meta["projs"], sparse=True,
             slot_partition=meta.get("slot_partition", 0), slot_part=meta.get("slot_part"),
             partition_unit=meta.get("partition_unit", False))
    E.apply_projection_scales(model, meta.get("patch_scale_mult", 1.),
                              meta.get("patch_scale_mult_up"), meta.get("patch_scale_mult_down"))
    E.load_patch(model, states[0])
    # Check siblings against the actual base reserve, independent of focus partition.
    linears = dict(model.named_modules())
    for state in E.progress(states[1:], 'branch_eval:learning_eval'):
        for name, entry in E.progress(state["entries"].items(), 'branch_eval:learning_eval'):
            lin = linears[name]
            W = lin.parametrizations.weight.original if E.P.is_parametrized(lin, "weight") else lin.weight
            if (entry["shape"] != list(W.shape) or
                    not E.pair8_forced_mask(W).flatten()[entry["idx"].long().to(dev)].all()):
                raise SystemExit("sibling writes outside new-base forced mask")
    banks, _ = E.attach_residents(model, states[1:], enabled=False, follow_enabled=True)
    facts = ([E.Fact(**row) for row in meta["c37_facts"]] if "c37_facts" in meta else
             E.make_world(meta["seed"], meta["people"], meta["facts_per_person"], meta["shared_attrs"],
                          meta["values"], meta.get("relation_world", False), meta.get("synthetic_people", 0)))
    ids = E.person_indices(facts, meta.get("train_person"), meta.get("train_attr_group"), meta.get("attr_groups", 1))
    items = E.build_items(tok, facts, E.TEST_T, E.BOS, idxs=ids)
    if a.fact_class_metrics:
        E.C50.annotate(items)
    E.set_enabled(True)
    result = dict(**E.eda.crosseval_metadata(a), eda_backend=E.eda.selected(a), policy="learning only; focus bank fixed, exhaustive frozen-sibling presence; no selector",
                  siblings=E.sibling_config_eval(banks, lambda: E.gen_eval(
                      model, tok, items, dev, a.gen_max_new, a.gen_stop_k, a.gen_bs, E.conflict_map(facts))),
                  files=[dict(path=str(p), sha256=E.file_sha(p)) for p in a.patches])
    if a.fact_class_metrics:
        result.update(E.C50.metric_metadata(facts))
        result['by_fact_class'] = E.C50.collect_splits(result)
    if a.value_operands:
        result["operands"] = E.operand_eval(model, tok, E.operand_items(tok, facts, E.BOS), dev, a)
    if a.evidence_skill:
        result["evidence_skill"] = E.evidence_skill_eval(model, tok, facts, items, dev, E.BOS, a)
    panels = {}
    for name, path in E.progress((("WikiText", a.eval_set), ("FineWeb", a.fineweb), ("code", a.code)), 'branch_eval:learning_eval'):
        if path:
            tokens = json.loads(path.read_text())
            panels[name] = torch.tensor([[E.BOS] + tokens[i:i+511]
                for i in range(0, min(len(tokens)-510, 40*511), 511)], device=dev)
    result["damage"] = {}
    for name, windows in E.progress(panels.items(), 'branch_eval:learning_eval'):
        with E.context_base_forward():
            base = E.nll_windows(model, windows, dev)
        result["damage"][name] = E.paired_nll(model, windows, base, dev)
    if a.in_topic_panel:
        try:
            from . import in_topic as I
        except ImportError:
            import in_topic as I
        panel = I.load_panel(a.in_topic_panel, a, tok, dev)
        with E.resident_state(banks, False), E.context_base_forward():
            base = I.base_scores(model, tok, panel, dev, E.BOS)
        written = I.written_groups(tok, states, E.BOS)
        for index, group in enumerate(written):
            with E.resident_state(banks, False):
                if index == 0:
                    group['baseline'] = I.fact_flags(model, tok, group, dev, a)
                else:
                    with E.zero_patch(model):
                        for resident in banks[index-1]:
                            resident.enabled = True
                        group['baseline'] = I.fact_flags(model, tok, group, dev, a)
        with E.resident_state(banks, True):
            result['in_topic'] = I.evaluate(model,tok,panel,base,written,dev,E.BOS,a)
        print('IN-TOPIC ' + json.dumps(result['in_topic']), flush=True)
        print('FINAL ' + json.dumps(dict(**E.eda.crosseval_metadata(a), in_topic=result['in_topic'])), flush=True)
    if a.dark_audit:
        if len(panels) != 3:
            raise SystemExit("--dark-audit needs --eval-set --fineweb --code panels")
        result["dark_audit"] = E.dark_damage_audit(model, tok, facts, panels, dev, E.BOS, a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if E.eda.crosseval_metadata(a):
        print("FINAL " + json.dumps(result), flush=True)
        print("RESULT " + json.dumps(result), flush=True)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    print("C37-LEARNING " + json.dumps(result), flush=True)
    if a.in_topic_panel:
        print('RESULT ' + json.dumps(dict(**E.eda.crosseval_metadata(a), in_topic=result['in_topic'])), flush=True)
    return result


def main(argv=None):
    a = build_parser().parse_args(argv)
    E.eda.announce(a)
    if a.negative_n is None:
        a.negative_n = 10000 if a.c42_mode == 'dump' else 3000
    if a.c42_mode:
        if __package__:
            from . import c42_runtime as C42
        else:
            import c42_runtime as C42
        return C42.run(sys.modules[__name__], a)
    if a.c37_control:
        result = E.c37_positive_control()
        if a.out:
            a.out.write_text(json.dumps(result, indent=2)+"\n")
        print("C37-CONTROL " + json.dumps(result), flush=True)
        return result
    if a.learning_eval:
        return learning_eval(a)
    if a.dark_audit or a.value_operands or a.evidence_skill:
        raise SystemExit("C37 flags require --learning-eval")
    arms = ([dict(oracle="B", router="C", sum="D")[m] for m in a.modes]
            if a.modes else a.arms.split(","))
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"B", "C", "D", "E", "F", "G", "I", "AC", "DG", "INH", "SLOT", "CONJ", "C29", "H", "FLAT", "C34", "PK"}:
        raise SystemExit("--arms must be unique comma-separated B,C,D,E,F,G,I,AC,DG,INH,SLOT,CONJ,C29,H,FLAT,C34,PK")
    banked = bool(set(arms) & {'H', 'FLAT'})
    if banked and (set(arms) - {'H', 'FLAT'} or a.partition_patches or a.unit_patches or a.decisive_eval):
        raise SystemExit("H/FLAT use attribute banks via --patches; evaluate other arms separately")
    if a.positive_control:
        controls = {arm: positive_control(arm) for arm in arms}
        payload = dict(controls=controls, verdict="control-passed" if all(c['passed'] for c in controls.values()) else "control-failed")
        print("POSITIVE-CONTROL " + json.dumps(payload), flush=True)
        if a.out:
            a.out.write_text(json.dumps(payload, indent=2)+"\n")
        return payload
    if not a.patches or any(getattr(a, key) is None for key in ('anchor', 'eval_set', 'fineweb', 'code', 'out')):
        raise SystemExit("evaluation needs --patches --anchor --eval-set --fineweb --code --out")
    if a.negative_n < 3000 or any(not math.isfinite(v) or v < 0 for v in a.selector_l2):
        raise SystemExit("need >=3000 negatives and finite nonnegative selector L2")
    bio_arms = set(arms) & {"INH", "SLOT", "CONJ", "C29"}
    slot_layers = []
    if bio_arms:
        if "C29" in bio_arms and a.c29_max_per_kind < 2:
            raise SystemExit("--c29-max-per-kind must be at least two")
        if (min(a.bio_steps, a.bio_budget, a.inh_exclusions) < 1 or min(a.bio_layer, a.inh_distractors, a.slot_early_layer) < 0
                or not math.isfinite(a.inh_lambda) or a.inh_lambda < 0 or not 2 <= a.slot_max <= 4):
            raise SystemExit("invalid C27 addressing parameters")
        try:
            slot_layers = [int(li) for li in a.slot_late_layers.split(",")]
        except ValueError as err:
            raise SystemExit("invalid SLOT late layers") from err
        if not slot_layers or len(set(slot_layers)) != len(slot_layers) or min(slot_layers) <= a.slot_early_layer:
            raise SystemExit("SLOT early layer must precede unique late layers")
    if "DG" in arms and (a.dg_layer < 0 or a.dg_expansion < 1 or a.dg_steps < 1
                         or not 0 < a.dg_density <= 1 or not 0 < a.dg_sparsity <= 1
                         or a.indexer_top_k < 1 or any(not math.isfinite(t) or not -1 <= t <= 1
                                                      for t in a.indexer_thresholds)):
        raise SystemExit("invalid DG parameters")
    ac_layers = []
    if "AC" in arms:
        try:
            ac_layers = [int(li) for li in a.ac_indexer_layers.split(",")]
        except ValueError as err:
            raise SystemExit("invalid AC indexer layers") from err
        if not ac_layers or len(set(ac_layers)) != len(ac_layers) or min(ac_layers) < 0:
            raise SystemExit("invalid AC indexer layers")
    if min(a.router_none, a.router_steps, a.gen_bs) < 1 or min(a.qa_n, a.g_n, a.gen_max_new, a.export_step) < 0:
        raise SystemExit("invalid router/evaluation counts")
    if a.decisive_eval or 'C34' in arms or 'PK' in arms:
        if __package__:
            from . import decisive_eval as C34
        else:
            import decisive_eval as C34
    full_list = [torch.load(p, map_location="cpu", weights_only=False) for p in a.patches]
    full = validate_patches(full_list, banked=banked, eda_backend=E.eda.selected(a), crosseval_args=a)
    lora = full_list[0]["meta"]["method"] == "lora"
    if lora and (set(arms) - {"B", "C", "D"} or a.partition_patches or a.unit_patches or a.decisive_eval):
        raise SystemExit("LoRA banks support --modes oracle router sum (B,C,D)")
    if a.bootstrap < 0:
        raise SystemExit("--bootstrap must be nonnegative")
    part_list = [torch.load(p, map_location="cpu", weights_only=False) for p in a.partition_patches]
    unit_list = [torch.load(p, map_location="cpu", weights_only=False) for p in a.unit_patches]
    unit = validate_patches(unit_list, True, eda_backend=E.eda.selected(a), crosseval_args=a) if unit_list else {}
    if unit and not all(s['meta'].get('partition_unit') for s in unit_list):
        raise SystemExit("--unit-patches requires whole-unit partition files")
    if a.decisive_eval or 'C34' in arms or 'PK' in arms:
        training_match = validate_training_match(full_list + part_list + unit_list)
        for state in E.progress(full_list + part_list + unit_list, 'branch_eval:main'):
            C34.check_scale(state['meta'])
    for state in E.progress(full_list + part_list + unit_list, 'branch_eval:main'):
        try:
            E.validate_patch_export(state["meta"], a.export, a.export_step)
        except ValueError as err:
            raise SystemExit(str(err)) from err
    part = validate_patches(part_list, True, eda_backend=E.eda.selected(a), crosseval_args=a) if part_list else {}
    if set(arms) & set("EFG") and not part:
        raise SystemExit("E/F/G require --partition-patches")
    for family_states in E.progress((part, unit), 'branch_eval:main'):
        if not family_states:
            continue
        for key in E.progress(IDENTITY, 'branch_eval:main'):
            default = E.EXPORT_STEP if key == "export_step" else False if key == "relation_world" else None
            if next(iter(family_states.values()))["meta"].get(key, default) != full_list[0]["meta"].get(key, default):
                raise SystemExit(f"full/partition config differs: {key}")
        if overlap_stats(list(family_states.values()))["overlapping_slots"]:
            raise SystemExit("partition patches contain overlapping writes")
    meta = full_list[0]["meta"]
    facts = E.make_world(meta["seed"], meta["people"], meta["facts_per_person"], meta["shared_attrs"], meta["values"], meta.get("relation_world", False), meta.get("synthetic_people", 0))
    conflicts = E.conflict_map(facts)
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    torch.manual_seed(meta["seed"])
    random.seed(meta["seed"])
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    model, _, dev = E.eda.build_model(build_model, a, a.export, device="cuda:0", model_config=None,
                               step=a.export_step, expert_format="bf16")
    model._fsa_export, model._fsa_export_step = a.export, a.export_step
    model.eval()
    mapping = layer_mapping(model)
    print("LAYER-MAPPING " + json.dumps(mapping), flush=True)
    if any(li >= len(model.layers) for li in ac_layers) or ("DG" in arms and a.dg_layer >= len(model.layers)):
        raise SystemExit("AC/DG layer outside full blocks")
    if ((bio_arms & {"INH", "CONJ"} and a.bio_layer >= len(model.layers))
            or (set(bio_arms) & {"SLOT", "C29"} and max(slot_layers) >= len(model.layers))):
        raise SystemExit("C27 addressing layer outside full blocks")
    for param in E.progress(model.parameters(), 'branch_eval:main'):
        param.requires_grad_(False)
    if a.fast_moe == "exact":
        try:
            from .fast_moe import install
        except ImportError:
            from fast_moe import install
        install(model, patched_layer_ids=meta["layers"], mode="exact")
    # Snapshot the actual pristine model before FSA attachment or any bank load.
    pristine_input = torch.tensor([[E.BOS] + json.loads(a.eval_set.read_text())[:15]], device=dev)
    pristine_inputs = [pristine_input] + [torch.tensor([E.encode(tok, text, E.BOS)], device=dev)
        for text in ('A quiet morning.', 'Explain a simple addition.')]
    pristine_logits = [E.logits_of(model, x).detach().clone() for x in pristine_inputs]
    pristine_weights = E.weight_hashes(model, full_list[0]['entries'])
    if not lora:
        E.attach(model, meta["layers"], "fsa_forced", 0, projs=meta["projs"], sparse=True,
                 slot_frac=meta["slot_frac"], seed=meta["seed"],
                 projs_by_layer=E.layer_projections(meta["layers"], meta.get("value_layers", meta["layers"]), meta["projs"]))
    mask_info = E.mask_diagnostics(model, a.mask_compare) if not lora else None
    E.apply_projection_scales(model, meta.get("patch_scale_mult", 1.),
                              meta.get("patch_scale_mult_up"), meta.get("patch_scale_mult_down"))
    # Validate FSA masks; LoRA factors are validated before bank construction.
    for states in (E.progress(() if lora else (full, part, unit), 'branch_eval:main')):
        for person, state in E.progress(states.items(), 'branch_eval:main'):
            E.load_patch(model, state)
            if states is part or states is unit or (banked and state['meta'].get('slot_partition')):
                for name, lin in E.progress(E.patched_linears(model), 'branch_eval:main'):
                    e = state["entries"][name]
                    layer = next(li for li, (lname, _) in enumerate(E.moe_layers(model)) if name.startswith(lname + "."))
                    expert = int(name.rsplit(".experts.", 1)[1].split(".")[0])
                    allowed = E.partition_mask(E.pair8_forced_mask(lin.parametrizations.weight.original.detach()),
                                               layer, expert, name.rsplit(".", 1)[1], state["meta"]["slot_partition"], state["meta"]["slot_part"],
                                               state["meta"].get("partition_unit", False))
                    if not bool(allowed.flatten()[e["idx"].long().to(dev)].all()):
                        raise SystemExit(f"patch outside declared {partition_label(state['meta'])}: {name}")
    full_bytes = {(s["meta"]["train_person"]*meta.get('attr_groups', 1)+s['meta']['train_attr_group']
                   if banked else s["meta"]["train_person"]): p.stat().st_size for s, p in zip(full_list, a.patches)}
    part_bytes = {s["meta"]["train_person"]: p.stat().st_size for s, p in zip(part_list, a.partition_patches)}
    bf = Branches(model, full, full_bytes, a.merge)
    bp = Branches(model, part, part_bytes, a.merge) if part else None
    bu = Branches(model, unit, {s['meta']['train_person']: p.stat().st_size for s,p in zip(unit_list,a.unit_patches)}, a.merge) if unit else None
    panels = {}
    panel_tokens = {}
    source_tokens = {}
    for name, path in E.progress((("WikiText", a.eval_set), ("FineWeb", a.fineweb), ("code", a.code)), 'branch_eval:main'):
        tokens = json.loads(path.read_text())
        source_tokens[name] = tokens
        if len(tokens) < 40 * 511:
            raise SystemExit(f"{name} requires 40x511 tokens plus BOS")
        panel_tokens[name] = tokens[:40 * 511]
        panels[name] = torch.tensor([[E.BOS] + tokens[i*511:(i+1)*511] for i in range(40)], device=dev)
    for path in E.progress(a.negative_text, 'branch_eval:main'):
        source_tokens["negative:"+str(path.resolve())] = json.loads(path.read_text())
    anchor_tokens = json.loads(a.anchor.read_text())
    if len(anchor_tokens) < 511:
        raise SystemExit("anchor requires >=511 tokens")
    E.checked_overlap(anchor_tokens, panel_tokens, "router anchor")
    E.set_enabled(False)
    revoke_input = panels["WikiText"][:1, :16]
    items = dict(val=E.build_items(tok, facts, E.VAL_T, E.BOS), test=E.build_items(tok, facts, E.TEST_T, E.BOS))
    qa = E.build_qa_items(tok, facts, E.BOS, a.qa_n)
    items.update(qa)
    if a.fact_class_metrics:
        for group in items.values():
            E.C50.annotate(group)
    bench_items = E.build_bench_items(tok, E.BOS, a.bench_dir) if a.bench else None
    subject_items = E.build_subject_items(tok, facts, E.BOS) if a.subject_text else None
    cache_key = branch_base_cache_key(a, meta) if a.base_cache else None
    baseline = E.cached_base(a.base_cache, cache_key,
                             lambda: evaluate_branch_base(model, tok, panels, qa, revoke_input, dev, a,
                                                          bench_items, subject_items), dev, eda_backend=E.eda.selected(a))
    base_nll, base_logits = baseline["base_nll"], pristine_logits[0]
    if not torch.equal(baseline['base_logits'], base_logits):
        raise RuntimeError('bank attachment changed the pristine baseline')
    items["g_set"] = composition_items(tok, facts, E.BOS, a.g_n)
    if a.fact_class_metrics:
        E.C50.annotate(items['g_set'], facts)
    # Forty natural-text completion prompts; values are checked with word boundaries.
    general = [dict(p=win[:64].tolist()) for win in panels["WikiText"]]
    items["general"] = general
    if banked:
        if __package__:
            from . import hierarchical_banks as H
        else:
            import hierarchical_banks as H
        return H.evaluate(sys.modules[__name__], a, arms, bf, tok, facts, items, panels,
                          source_tokens, anchor_tokens, baseline, revoke_input, dev,
                          bench_items, subject_items, mapping)
    # Separate general-text sample for statistical claims; no GEN needed.
    selector_manifest = dict(held_out_attributes=sorted(R.heldout_attributes(facts)),
                             layer_indexing="zero-based full block inputs", layer_mapping=mapping)
    locked_selectors = {}
    probes, predictions, panel_predictions, subject_predictions = {}, {}, {}, {}
    index_predictions, index_panels, index_subject = {}, {}, {}
    dg_predictions, dg_panels, dg_subject, dg_config = {}, {}, {}, {}
    selector_val = [it for it in items['val'] if facts[it['fi']].attr not in R.heldout_attributes(facts)]
    validation_general = R.independent_queries(source_tokens,E.BOS,64)
    # Reserve validation nulls before collecting the independent TEST null sample.
    negative_items = R.independent_queries(source_tokens,E.BOS,a.negative_n,skip=40*511+64*63)
    val_nulls = {tuple(it['p']) for it in validation_general}
    negative_items = [it for it in negative_items if tuple(it['p']) not in val_nulls]
    E.checked_overlap(anchor_tokens,{it['source']+':'+str(it['start']):it['p'][1:] for it in negative_items+validation_general},'selector negative calibration')
    a._negative_items = negative_items
    if not validation_general:
        validation_general = [dict(p=win[:64].tolist()) for win in panels['FineWeb']]
        selector_manifest['validation_null_limitation'] = 'source lacks reserved validation nulls; preservation windows used diagnostically'
    # Validation negatives are distinct from the statistical test sample and TRAIN anchor.
    def features_at(layer):
        return address_eval_features(model, tok, items, layer, dev)
    def panel_at(layer):
        return {name: indexer_features(model, wins.tolist(), layer, dev, [63]*len(wins)) for name, wins in panels.items()}
    def search_score(select, val_x, null_x):
        metric = selection_metrics(select(val_x), gold_routes(selector_val, facts))
        return metric['accuracy'] - sum(bool(r) for r in select(null_x))/max(len(null_x), 1)
    def install(name, layer, select):
        xf = features_at(layer)
        predictions_for = {group: select(x) for group, x in xf.items()}
        panels_for = {group: select(x) for group, x in panel_at(layer).items()}
        subjects = (select(indexer_features(model, [it['p'] for it in subject_items], layer, dev,
                     [min(63, len(it['p'])-1) for it in subject_items])) if subject_items else [])
        negative_routes = select(indexer_features(model, [it['p'] for it in negative_items], layer, dev)) if negative_items else []
        locked_selectors[name] = (layer, select)
        selector_manifest[name]['false_activation'] = R.false_activation(negative_routes)
        return predictions_for, panels_for, subjects
    if 'C' in arms:
        layers = list(dict.fromkeys(int(li) for li in a.router_layer.split(',')))
        train = [it for it in S.build_panels(tok, facts, E.BOS, question_train=False)['train']
                 if facts[it['fi']].attr not in R.heldout_attributes(facts)]
        windows = [[E.BOS]+anchor_tokens[i:i+63] for i in range(0, len(anchor_tokens)-62, 63)]
        none = windows[:a.router_none]
        candidates = []
        for layer in E.progress(layers, 'branch_eval:main'):
            x = indexer_features(model, [it['p'] for it in train]+none, layer, dev)
            vx = indexer_features(model, [question_sequence(tok, it['p'], E.BOS) for it in selector_val], layer, dev)
            nx = indexer_features(model, [it['p'] for it in validation_general], layer, dev)
            for l2 in E.progress(a.selector_l2, 'branch_eval:main'):
                probe = fit_router(x, [facts[it['fi']].pid for it in train]+[meta['people']]*len(none),
                                   meta['people']+1, a.router_steps, l2)
                def select(x, probe=probe):
                    return [() if p == meta['people'] else (p,) for p in probe.predict(x).tolist()]
                candidates.append((search_score(select, vx, nx), layer, l2, probe, select))
        score, layer, l2, probe, select = R.select_validation(candidates, lambda c: c[0])
        selector_manifest['C'] = dict(layer=layer, l2=l2, validation_score=score, locked=True,
            address_bytes=sum(v.numel()*v.element_size() for v in vars(probe).values() if torch.is_tensor(v)))
        probes[layer] = probe
        predictions[layer], panel_predictions[layer], subject_predictions[layer] = install('C', layer, select)
    if 'I' in arms:
        layers = ([int(v) for v in a.indexer_layers.split(',')] if a.indexer_layers else
                  list(dict.fromkeys([a.indexer_layer]+[li for li in (4,26,28) if li<len(model.layers)])))
        candidates = {'single': [], 'multibank': []}
        for layer in E.progress(layers, 'branch_eval:main'):
            train, x, g = address_training(model, tok, facts, anchor_tokens, meta['seed'], layer, dev)
            index = BankIndexer.fit(x, [facts[it['fi']].pid for it in train], g)
            vx = indexer_features(model, [question_sequence(tok, it['p'], E.BOS) for it in selector_val], layer, dev)
            nx = indexer_features(model, [it['p'] for it in validation_general], layer, dev)
            # Separate multi-bank validation composition, distinct from TEST wording.
            cv = [it for it in composition_items(tok, facts, E.BOS, a.g_n, split='val') if all(facts[fi].attr not in R.heldout_attributes(facts) for fi in it['fis'])]
            cx = indexer_features(model, [it['p'] for it in cv], layer, dev) if cv else None
            for mode, budget in E.progress([('single', 1), ('multibank', min(a.indexer_top_k, len(index.owners)))], 'branch_eval:main'):
                for threshold in E.progress(a.indexer_thresholds, 'branch_eval:main'):
                    def select(x, index=index, threshold=threshold, budget=budget):
                        return index.select(x, threshold, budget)
                    score = search_score(select, vx, nx)
                    if mode == 'multibank' and cv:
                        score += selection_metrics(select(cx), gold_routes(cv, facts))['accuracy']
                    candidates[mode].append((score, layer, threshold, select, index))
        for mode, rows in E.progress(candidates.items(), 'branch_eval:main'):
            score, layer, threshold, select, index = R.select_validation(rows, lambda c: c[0])
            name = 'I:'+mode
            selector_manifest[name] = dict(layer=layer, threshold=threshold, mode=mode, validation_score=score, locked=True,
                address_bytes=8*len(index.owners)+sum(v.numel()*v.element_size() for v in vars(index).values() if torch.is_tensor(v)))
            index_predictions[mode], index_panels[mode], index_subject[mode] = install(name, layer, select)
    if 'DG' in arms:
        layers = ([int(v) for v in a.dg_layers.split(',')] if a.dg_layers else
                  list(dict.fromkeys([a.dg_layer]+[li for li in (4,26,28) if li<len(model.layers)])))
        candidates = {'sparse': [], 'dense_logistic': []}
        for layer in E.progress(layers, 'branch_eval:main'):
            train, x, general_train = address_training(model, tok, facts, anchor_tokens, meta['seed'], layer, dev)
            labels, attrs = [facts[it['fi']].pid for it in train], [facts[it['fi']].attr for it in train]
            vx = indexer_features(model, [question_sequence(tok, it['p'], E.BOS) for it in selector_val], layer, dev)
            nx = indexer_features(model, [it['p'] for it in validation_general], layer, dev)
            for l2 in E.progress(a.selector_l2, 'branch_eval:main'):
                dg = DGIndexer.fit(x, labels, attrs, general_train, a.dg_expansion, a.dg_density,
                                   a.dg_sparsity, a.dg_seed, a.dg_steps, l2)
                dense = DenseLogisticIndexer.fit(x, labels, attrs, general_train, a.dg_steps, l2)
                for mode, index in E.progress([('sparse', dg), ('dense_logistic', dense)], 'branch_eval:main'):
                    def select(x, index=index):
                        return index.select(x, top_k=min(a.indexer_top_k, len(index.owners)))
                    candidates[mode].append((search_score(select, vx, nx), layer, l2, index, select))
        for mode, rows in E.progress(candidates.items(), 'branch_eval:main'):
            score, layer, l2, index, select = R.select_validation(rows, lambda c: c[0])
            name = 'DG:'+mode
            selector_manifest[name] = dict(layer=layer, l2=l2, validation_score=score, locked=True,
                address_bytes=index.storage_bytes(), activation_budget=a.indexer_top_k,
                training='identical examples, weighted independent BCE, L2 search and LBFGS steps')
            dg_predictions[mode], dg_panels[mode], dg_subject[mode] = install(name, layer, select)
        dg_config = {mode: selector_manifest['DG:'+mode] for mode in candidates}
    full_overlap = overlap_stats(full_list)
    result = dict(**E.eda.crosseval_metadata(a), eda_backend=E.eda.selected(a), world=meta, export=str(a.export), export_step=a.export_step, forced_mask=mask_info,
                  partition_mode=partition_label(part_list[0]["meta"]) if part_list else None,
                  base_qa_probe=baseline["base_qa_probe"],
                  merge=a.merge, full_overlap=full_overlap,
                  sum_comparison=('same saved-delta sum, fp32 then one cast; no clipping for either format'
                      if a.merge == 'sum_delta' else 'format-native: FSA clipped codes; LoRA unclipped dense products'),
                  partition_overlap=overlap_stats(part_list) if part_list else None,
                  patch_files=[dict(path=str(p), bytes=p.stat().st_size, person=s["meta"]["train_person"],
                                    partition_mode=partition_label(s["meta"]),
                                   nnz=full_overlap["nnz"][i] if lora else sum(e["idx"].numel() for e in s["entries"].values()))
                               for i, (p, s) in enumerate(zip(a.patches + a.partition_patches, full_list + part_list))],
                  selector_selection=selector_manifest,
                  negative_sample=dict(requested=a.negative_n,available=len(negative_items),
                      policy="distinct nonoverlapping 63-token windows, after preservation and validation reservations; no resampling",
                      independence_unit="general-text window; binomial uncertainty assumes query independence"),
                  damage_single={}, arms={})
    if a.base_cache:
        result["base_cache_key"] = cache_key
    result['controls'] = {arm: positive_control(arm) for arm in arms if arm != 'C29'}
    result['bank_controls'] = {family: bank_control(branches,tok,facts,items['test'],dev,a)
                               for family,branches in [('full',bf),('partition',bp)] if branches is not None}
    # Zero sparse patch through the actual NLL, QA, benchmark and subject scoring paths.
    bf.activate(())
    zero = evaluate_branch_base(model,tok,panels,qa,revoke_input,dev,a,bench_items,subject_items)
    zero_ok = all(zero[key] == baseline[key] for key in ('base_nll','base_qa_probe'))
    zero_ok = zero_ok and torch.equal(zero['base_logits'],base_logits)
    if a.bench:
        zero_ok = zero_ok and zero['base_bench']==baseline['base_bench']
    if a.subject_text:
        zero_ok = zero_ok and zero['base_subject']==baseline['base_subject']
    result['zero_patch_control'] = dict(passed=zero_ok, damage_batch_size=1)
    result["three_metrics"] = {}
    resident_bench = {}
    for label, branches in E.progress((("full", bf), ("partition", bp)), 'branch_eval:main'):
        if branches is None:
            continue
        mode = partition_label(next(iter(branches.states.values()))["meta"])
        if a.bench:
            resident_bench[label] = resident_benchmark(branches, bench_items, baseline["base_bench"], dev)
            print("BENCH " + json.dumps(dict(patches=label, label="resident-bank stress test", partition_mode=mode, **resident_bench[label])), flush=True)
        for person in E.progress(branches.states, 'branch_eval:main'):
            routes = {name: [(person,)] * len(wins) for name, wins in panels.items()}
            result["damage_single"][f"{label}:{person}"] = damage(branches, panels, base_nll, routes, dev)
        metrics = three_metrics(branches, tok, facts, items["test"], items["g_set"],
                                panels["WikiText"], dev, a.gen_max_new, a.gen_stop_k, a.gen_bs)
        metrics["composition_scoring"] = "value-presence diagnostic; arm binding scores determine composition decisions"
        result["three_metrics"][label] = metrics
        print("THREE-METRICS " + json.dumps(dict(patches=label, partition_mode=mode, **metrics)), flush=True)
        print("UNIT-BYTES " + json.dumps(dict(patches=label, partition_mode=mode, **(metrics["unit_bytes"] or {}))), flush=True)
    if a.bench:
        result["resident_bank_stress_test"] = resident_bench
        result["resident_bank_stress_test_label"] = "resident-bank stress test"
    if "AC" in arms:
        groups = {name: items[name] for name in ("test", "pronoun", "pronoun_recent", "two_fact", "g_set", "two_hop")}
        result["address_ceiling"] = {}
        for family, branches in E.progress((("full", bf), ("partition", bp)), 'branch_eval:main'):
            if branches is None:
                continue
            ceiling = addressing_ceiling(branches, tok, facts, groups, anchor_tokens, meta["seed"], ac_layers, dev, a)
            ceiling["patch_family"] = family
            ceiling['control'] = dict(passed=result['controls']['AC']['passed'] and zero_ok and result['bank_controls'][family]['passed'])
            if not ceiling['control']['passed']:
                ceiling['verdict']='control-failed'
                ceiling['value_use_bottleneck']=None
                for decision in E.progress(ceiling['per_task'].values(), 'branch_eval:main'):
                    decision['decision']='control-failed' 
            result["address_ceiling"][family] = ceiling
            print_ceiling(ceiling)
        result["arms"]["AC"] = result["address_ceiling"]
    if "DG" in arms and bp is not None:
        arms.append("DG-partition")
    a._bench_items = bench_items
    a._base_bench = baseline.get('base_bench')
    if bio_arms:
        if __package__:
            from . import biology_addressing as BIO
        else:
            import biology_addressing as BIO
        for arm in E.progress(sorted(bio_arms), 'branch_eval:main'):
            for family, branches in E.progress((("full", bf), ("partition", bp)), 'branch_eval:main'):
                if branches is None:
                    continue
                a._runtime_control_ok = zero_ok and result["bank_controls"][family]["passed"]
                if arm == "C29":
                    if __package__:
                        from . import indexed_composition as C29
                    else:
                        import indexed_composition as C29
                    report = C29.run(branches, tok, facts, sister_relations(facts, meta.get("relation_world", False)), anchor_tokens,
                        meta["seed"], a.slot_early_layer, slot_layers[0], dev, a.bio_steps, a.gen_max_new,
                        a.c29_max_per_kind)
                    report["damage"] = damage(branches, panels, base_nll,
                        {name: [()] * len(wins) for name, wins in panels.items()}, dev)
                    report["damage_policy"] = "No mention slots or inverse query fields on unrelated panels: abstain"
                elif arm == "SLOT":
                    report = BIO.slot_arm(branches, tok, facts, anchor_tokens, meta["seed"], panels, base_nll, dev, a)
                else:
                    report = BIO.addressing_arm(arm, branches, tok, facts, items, anchor_tokens, meta["seed"],
                                                panels, base_nll, dev, a)
                report["patch_family"] = family
                result["arms"][f"{arm}:{family}"] = report
                report['kill'] = R.gated_verdict(dict(passed=zero_ok and result['bank_controls'][family]['passed'] and report.get('control',{'passed':True})['passed'], kind='bank/zero/arm controls'), report['kill']) if 'kill' in report else report.get('kill')
                print("BIO-ADDRESS " + json.dumps(dict(arm=arm, **report)), flush=True)
    for arm in E.progress(arms, 'branch_eval:main'):
        if arm in ("AC", "C34", "PK") or arm in bio_arms:
            continue
        is_dg = arm.startswith("DG")
        branches = bp if arm in "EFG" or arm == "DG-partition" else bf
        family = "partition" if branches is bp else "full"
        for layer in (E.progress(probes if arm == "C" else index_predictions if arm == "I" else dg_predictions if is_dg else [None], 'branch_eval:main')):
            report = dict(total_patch_bytes=sum(branches.bytes.values()), active_bytes={},
                          partition_mode=partition_label(next(iter(branches.states.values()))["meta"]),
                          three_metrics=result["three_metrics"][family])
            for group_name, group in E.progress(items.items(), 'branch_eval:main'):
                if group_name == "indirect":
                    continue  # Deprecated alias is copied below without repeating generation.
                if arm == "I":
                    routes = index_predictions[layer].get(group_name, [])
                elif is_dg:
                    routes = dg_predictions[layer].get(group_name, [])
                elif arm == "C":
                    routes = predictions[layer].get(group_name, [])
                elif arm in "DF":
                    routes = [tuple(branches.states)] * len(group)
                elif group_name == "general":
                    # Oracle has a known 'none' class on unrelated inputs.
                    routes = [()] * len(group)
                else:
                    routes = person_routes(group, facts, compose=arm == "G")
                mx = a.gen_max_new or (max((len(it.get("a", [])) for it in group), default=0) + a.gen_stop_k + 2)
                if group_name == "two_fact":
                    mx = 40
                if group_name == "general":
                    mx = a.gen_max_new or 64
                gens = routed_generations(branches, tok, group, routes, dev, mx, a.gen_bs)
                report["active_bytes"][group_name] = dict(per_query=[branches.active_bytes(key) for key in routes],
                                                        mean=sum(branches.active_bytes(key) for key in routes) / max(len(routes), 1),
                                                        max=max((branches.active_bytes(key) for key in routes), default=0))
                if group_name in ("val", "test"):
                    report[group_name + "_gen"] = generation_summary(tok, group, gens, conflicts, a.gen_stop_k, a.bootstrap, meta["seed"])
                    if group_name == "test":
                        fired = routed_firing(branches, group, routes, dev)
                        safety = E.generation_safety(tok, group, gens, [f.val for f in facts], a.gen_stop_k, fired)
                        report["test_gen"].update(safety)
                        report["confidently_wrong_without_fallback"] = safety["confidently_wrong_without_fallback"]
                    report[group_name + "_tf"] = routed_tf(branches, group, routes, dev, a.gen_bs)
                elif group_name == "general":
                    texts = [tok.decode(g) for g in gens]
                    leak = sum(any(contains_value(text, f.val) for f in facts) for text in texts)
                    report["abstention"] = dict(n=len(texts), fact_value_rate=leak / max(len(texts), 1),
                                                 abstention_rate=1 - leak / max(len(texts), 1))
                else:
                    scores = [all(contains_value(tok.decode(g), v) for v in it["vals"]) for it, g in zip(group, gens)]
                    report[group_name] = dict(n=len(group), both_values_present=sum(scores) / max(len(scores), 1))
                    if group_name in ("g_set", "two_fact"):
                        report[group_name]["presence_stop"] = sum(E.score_two_fact_generation(tok,g,it["vals"],a.gen_stop_k)["both"] for it,g in zip(group,gens))/max(len(group),1)
                        report[group_name]["presence"] = sum(E.score_two_fact_lenient(tok,g,it["vals"]) for it,g in zip(group,gens))/max(len(group),1)
                        if group_name == 'g_set':
                            report[group_name]["binding"] = sum(R.binding_score(tok.decode(g), it, facts) for it, g in zip(group, gens))/max(len(group), 1)
                    if group_name == "two_fact":
                        report[group_name].update(
                            two_fact_correct=sum(E.score_two_fact_correctness(tok,g,it['vals']) for it,g in zip(group,gens))/max(len(group),1),
                            two_fact_presence_stop=sum(E.score_two_fact_generation(tok, g, it["vals"], a.gen_stop_k)["both"]
                                              for it, g in zip(group, gens)) / max(len(group), 1),
                            two_fact_presence=sum(E.score_two_fact_lenient(tok, g, it["vals"])
                                                      for it, g in zip(group, gens)) / max(len(group), 1))
                    if group_name in ("unseen", "two_hop", "pronoun", "pronoun_recent"):
                        clean = [E.score_generation(tok, g, it["vals"][0], a.gen_stop_k,
                                                     conflicts.get(it["fi"], ()))["clean"]
                                 for it, g in zip(group, gens)]
                        report[group_name]["recall"] = sum(clean) / max(len(clean), 1)
                        if a.bootstrap:
                            report[group_name]["ci95"] = {"recall": E.cluster_bootstrap(group, clean, a.bootstrap, meta["seed"])}
                    if group_name == "two_fact" and a.bootstrap:
                        report[group_name]["ci95"] = {
                            metric: E.cluster_bootstrap(group, hits, a.bootstrap, meta["seed"])
                            for metric, hits in (
                                ("two_fact_correct", [E.score_two_fact_correctness(tok,g,it['vals']) for it,g in zip(group,gens)]),
                                ("two_fact_presence_stop", [E.score_two_fact_generation(tok, g, it["vals"], a.gen_stop_k)["both"] for it, g in zip(group, gens)]),
                                ("two_fact_presence", [E.score_two_fact_lenient(tok, g, it["vals"]) for it, g in zip(group, gens)]))}
                    if a.fact_class_metrics:
                        rows = []
                        for it, g in zip(group, gens):
                            row = dict(both_values_present=all(contains_value(tok.decode(g), v) for v in it['vals']))
                            if group_name in ('g_set', 'two_fact'):
                                row.update(presence_stop=E.score_two_fact_generation(tok,g,it['vals'],a.gen_stop_k)['both'],
                                           presence=E.score_two_fact_lenient(tok,g,it['vals']))
                            if group_name == 'two_fact':
                                row['two_fact_correct'] = E.score_two_fact_correctness(tok,g,it['vals'])
                            if group_name in ('unseen', 'two_hop', 'pronoun', 'pronoun_recent'):
                                row['recall'] = E.score_generation(tok,g,it['vals'][0],a.gen_stop_k,
                                                                 conflicts.get(it['fi'], ()))['clean']
                            rows.append(row)
                        report[group_name]['by_fact_class'] = E.C50.split_scores(group, rows)
            report["indirect"] = report["two_hop"]  # Deprecated alias for one version.
            report["active_bytes"]["indirect"] = report["active_bytes"]["two_hop"]
            if arm == "I" or is_dg:
                route_predictions = dg_predictions if is_dg else index_predictions
                report["routing"] = dict(layer=a.dg_layer if is_dg else a.indexer_layer, layer_indexing="full block input",
                    resident_banks=sorted(branches.states), residency="all sparse banks in host memory; active merge loaded for forward",
                    selection={name: selection_metrics(route_predictions[layer].get(name, []),
                        [()] * len(group) if name == "general" else person_routes(group, facts, compose=True))
                        for name, group in items.items() if name != "indirect"})
                droutes = dg_panels[layer] if is_dg else index_panels[layer]
                if is_dg:
                    report["dg"] = dg_config
                    report["routing"]["method"] = "DG logistic score > 0" if layer == "sparse" else "matched dense logistic control"
                    report["address_accuracy"] = report["routing"]["selection"]
                    report["composed_recall"] = dict(single_fact=report["test_gen"]["recall"],
                        g_set=report["g_set"]["binding"],
                        two_fact_correct=report['two_fact']['two_fact_correct'],
                        two_fact_presence_stop=report["two_fact"]["two_fact_presence_stop"],
                        two_fact_presence=report["two_fact"]["two_fact_presence"])
                    report["false_activations"] = dict(general=1-report["routing"]["selection"]["general"]["abstention_rate"],
                        panels={name: sum(bool(route) for route in routes)/max(len(routes), 1)
                                for name, routes in droutes.items()})
                    report["address_unit_comparison"] = result["three_metrics"][family]
            elif arm == "C":
                correct = [routes == (facts[it["fi"]].pid,) for routes, it in zip(predictions[layer]["test"], items["test"])]
                report["routing"] = dict(layer=layer, layer_indexing="zero-based full block inputs", test_accuracy=sum(correct) / len(correct), n_test=len(correct))
                droutes = panel_predictions[layer]
            elif arm in "DF":
                droutes = {name: [tuple(branches.states)] * len(wins) for name, wins in panels.items()}
            else:
                droutes = {name: [()] * len(wins) for name, wins in panels.items()}
            report["damage"] = damage(branches, panels, base_nll, droutes, dev)
            if arm in ('C', 'I') or is_dg:
                key = 'C' if arm == 'C' else ('DG:'+str(layer) if is_dg else 'I:'+str(layer))
                chosen_layer, chosen_select = locked_selectors[key]
                report['routing'].update(selector_manifest[key])
                report['routing'].update(R.routing_panels(items['test'], facts,
                    (predictions[layer] if arm == 'C' else dg_predictions[layer] if is_dg else index_predictions[layer])['test'], selection_metrics))
                def bench_route(group):
                    return chosen_select(address_eval_features(model,tok,{'benchmark':group},chosen_layer,dev)['benchmark'])
            else:
                def bench_route(group):
                    return [tuple(branches.states) if arm in 'DF' else ()]*len(group)
            bench = routed_benchmark(branches, bench_items, baseline['base_bench'], bench_route, dev) if a.bench else None
            if bench is not None:
                report["bench"] = bench
            report["damage_gate"] = E.damage_gate(report["damage"], bench)
            report['benchmark_policy'] = 'same query routing policy as arm; route fixed for all choices'
            if arm == 'G':
                evidence = R.supplied_evidence(tok, items['g_set'], facts, E.encode, E.BOS)
                egens = routed_generations(branches, tok, evidence, [()]*len(evidence), dev, a.gen_max_new or 80, a.gen_bs)
                evidence_success = sum(R.binding_score(tok.decode(g), it, facts) for it, g in zip(evidence, egens))/max(len(evidence), 1)
                report['supplied_evidence_composition'] = dict(n=len(evidence), binding=evidence_success, policy='frozen base with supplied facts')
                report['control'] = dict(passed=bool(evidence) and evidence_success == 1., kind='supplied-evidence composition')
                report['verdict'] = 'measured' if report['control']['passed'] else 'control-failed'
            if arm in ('C', 'I') or is_dg:
                report['statistical_verdict'] = 'measured' if report['routing']['false_activation']['sufficient'] else 'insufficient independent negatives' 
            arm_key = f"{arm}:{layer}" if layer is not None else arm
            print("DAMAGE-GATE " + json.dumps(dict(arm=arm_key, **report["damage_gate"])), flush=True)
            if a.subject_text:
                sroutes = (dg_subject[layer] if is_dg else index_subject[layer] if arm == "I" else subject_predictions[layer] if arm == "C" else
                           [tuple(branches.states)] * len(subject_items) if arm in "DF" else
                           [(it["pid"],) for it in subject_items])
                report["subject_dnll"] = routed_subject_damage(branches, subject_items, baseline["base_subject"], sroutes, dev)
                print("SUBJECT-DNLL " + json.dumps(dict(arm=arm_key, **report["subject_dnll"])), flush=True)
            control_arm = 'DG' if is_dg else arm
            if arm == 'F':
                learned = result['bank_controls'][family]['successful_fis']
                pairs = [(fi,fj) for fi,fj in combinations(learned,2) if facts[fi].pid != facts[fj].pid]
                pair_passed = False
                if pairs:
                    fi,fj = pairs[0]
                    first,second = facts[fi],facts[fj]
                    prompt = f"State {first.name}'s {first.attr} and {second.name}'s {second.attr}, attributing each value."
                    pair = dict(fis=[fi,fj],p=E.encode(tok,prompt,E.BOS))
                    generated = routed_generations(branches,tok,[pair],[tuple(branches.states)],dev,a.gen_max_new or 80,a.gen_bs)[0]
                    pair_passed = R.binding_score(tok.decode(generated),pair,facts)
                report['learned_pair_control'] = dict(passed=pair_passed,available=bool(pairs),kind='two individually recalled facts tested together')
            if control_arm in result['controls']:
                control = dict(result['controls'][control_arm])
                control['passed'] = control['passed'] and zero_ok and result['bank_controls'][family]['passed']
                if arm == 'F':
                    control['passed'] = control['passed'] and report['learned_pair_control']['passed']
                if arm == 'G':
                    control['passed'] = control['passed'] and report['control']['passed']
                report['control'] = control
                if not control['passed']:
                    print('control-failed '+arm_key,flush=True)
                    report['verdict'] = 'control-failed'
                    report['verdict_suppressed'] = True
            result["arms"][arm_key] = report
            print("BRANCH-EVAL " + json.dumps(dict(arm=arm, router_layer=layer, **report)), flush=True)
    result['selector_decisions'] = {}
    for arm_key, report in E.progress(result['arms'].items(), 'branch_eval:main'):
        if arm_key.startswith('I:'):
            rate = report['routing']['false_activation']
            sufficient = rate['sufficient']
            decision = dict(passed=(report['routing']['selection']['test']['recall']>=.95 and rate['upper_95']<=.001) if sufficient else None,
                            rule='>=95% true-bank recall and one-sided 95% false-activation upper bound <=1e-3',
                            sufficient=sufficient)
            result['selector_decisions'][arm_key] = R.gated_verdict(report['control'],decision)
    for family, prefix in E.progress([('full','DG'),('partition','DG-partition')], 'branch_eval:main'):
        sparse = result['arms'].get(prefix+':sparse')
        dense = result['arms'].get(prefix+':dense_logistic')
        if sparse is not None and dense is not None:
            gain = sparse['routing']['selection']['test']['accuracy']-dense['routing']['selection']['test']['accuracy']
            decision = dict(passed=gain>=.05-1e-12,rule='>=5 pts address accuracy over validation-selected matched dense logistic',
                            address_gain=gain,sparse_address_bytes=sparse['routing']['address_bytes'],
                            dense_address_bytes=dense['routing']['address_bytes'])
            control = dict(passed=sparse['control']['passed'] and dense['control']['passed'])
            result['selector_decisions'][prefix] = R.gated_verdict(control,decision)
    if 'PK' in arms:
        result['arms']['PK'] = C34.product_key_arm(sys.modules[__name__], bf, tok, facts, items,
            anchor_tokens, validation_general, selector_val, panels, base_nll, dev, a, bench_items, baseline)
        result['arms']['PK']['control']['passed'] = (result['arms']['PK']['control']['passed'] and
            zero_ok and result['bank_controls']['full']['passed'])
        if not result['arms']['PK']['control']['passed']:
            result['arms']['PK']['verdict'] = 'control-failed'
            result['arms']['PK']['verdict_suppressed'] = True
        print('PK ' + json.dumps(C34.compact_report(result['arms']['PK'])), flush=True)
    if a.decisive_eval or 'C34' in arms:
        policies = {'resident': None}
        for key, (layer, selected) in E.progress(locked_selectors.items(), 'branch_eval:main'):
            def route(group, layer=layer, selected=selected):
                return selected(address_eval_features(model,tok,{'c34':group},layer,dev)['c34']) if group else []
            policies[key] = route
        result['decisive_eval'] = {}
        for key, selected in E.progress(policies.items(), 'branch_eval:main'):
            report = C34.suite(bf,tok,facts,E.BOS,dev,a,selected,
                block=bp if bp is not None and not part_list[0]['meta'].get('partition_unit') else None,
                unit=bu or (bp if bp is not None and part_list[0]['meta'].get('partition_unit') else None),
                compose=key=='resident')
            panel_routes = {name: (selected([dict(p=win[:64].tolist()) for win in wins]) if selected is not None else
                                  [tuple(bf.states)]*len(wins)) for name,wins in panels.items()}
            report['damage'] = damage(bf,panels,base_nll,panel_routes,dev)
            report['damage_policy'] = 'same locked serving selector; resident policy activates all banks; batch size 1'
            def c34_bench_route(group):
                return selected(group) if selected is not None else [tuple(bf.states)]*len(group)
            report['bench'] = routed_benchmark(bf,bench_items,baseline['base_bench'],c34_bench_route,dev) if a.bench else None
            report['damage_gate'] = E.damage_gate(report['damage'],report['bench'])
            report['control'] = dict(passed=C34.positive_control()['passed'] and zero_ok and result['bank_controls']['full']['passed'])
            report['verdict'] = 'measured' if report['control']['passed'] else 'control-failed'
            result['decisive_eval'][key] = report
            print('CEILINGS ' + json.dumps(C34.compact_report(dict(policy=key, types=report['ceilings']))), flush=True)
            print('C34 ' + json.dumps(C34.compact_report(dict(policy=key, **report))), flush=True)
    if a.in_topic_panel:
        try:
            from . import in_topic as I
        except ImportError:
            import in_topic as I
        panel = I.load_panel(a.in_topic_panel, a, tok, dev)
        result['in_topic'] = I.resident_branch_eval(bf, tok, panel, dev, E.BOS, a)
        print('IN-TOPIC ' + json.dumps(result['in_topic']), flush=True)
        print('FINAL ' + json.dumps(dict(**E.eda.crosseval_metadata(a), in_topic=result['in_topic'])), flush=True)
    E.set_enabled(False)
    with torch.no_grad():
        result["revoked"] = dict(logits_bit_exact=all(torch.equal(expected, E.logits_of(model, x))
                for expected,x in zip(pristine_logits,pristine_inputs)),
                weights_bit_exact=E.weight_hashes(model, pristine_weights) == pristine_weights,
                baseline='pristine pre-attachment model', prompts=len(pristine_inputs), weight_hashes=pristine_weights)
    if not result["revoked"]["logits_bit_exact"] or not result['revoked']['weights_bit_exact']:
        raise RuntimeError("branch revocation failed bit-exact check")
    E.set_enabled(True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.decisive_eval or 'C34' in arms or 'PK' in arms:
        files = a.patches + a.partition_patches + a.unit_patches
        states = full_list + part_list + unit_list
        result['comparison_manifest'] = dict(eda_backend=E.eda.selected(a), matched_training=training_match,
            export=dict(path=str(a.export), step=a.export_step), merge_order='sorted bank id',
            patches=[dict(path=str(path), sha256=E.file_sha(path), bytes=path.stat().st_size,
                          mask_sha256=state['meta']['mask_sha256'], init=state['meta']['training_recipe']['unit_init'],
                          exposures=state['meta']['training_exposures'], selection=state['meta']['selection_policy'])
                     for path,state in zip(files,states)],
            selectors=selector_manifest,
            selector_bytes={name: panel['address_bytes'] for name,panel in selector_manifest.items()
                            if isinstance(panel,dict) and 'address_bytes' in panel} |
                           {('PK:'+name): panel['address_bytes'] for name,panel in result['arms'].get('PK',{}).get('selectors',{}).items()})
    if a.fact_class_metrics:
        result.update(E.C50.metric_metadata(facts))
        result['by_fact_class'] = E.C50.collect_splits(result)
        print('FINAL ' + json.dumps(dict(**E.eda.crosseval_metadata(a), by_fact_class=result['by_fact_class'])), flush=True)
        print('RESULT ' + json.dumps(dict(**E.eda.crosseval_metadata(a), by_fact_class=result['by_fact_class'])), flush=True)
    if E.eda.crosseval_metadata(a):
        print("FINAL " + json.dumps(result), flush=True)
        print("RESULT " + json.dumps(result), flush=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    if a.in_topic_panel:
        print('RESULT ' + json.dumps(dict(**E.eda.crosseval_metadata(a), in_topic=result['in_topic'])), flush=True)
    print("BRANCH-EVAL " + json.dumps(dict(output=str(a.out), full_overlap=result["full_overlap"],
                                         partition_mode=result["partition_mode"],
                                         partition_overlap=result["partition_overlap"])), flush=True)


if __name__ == "__main__":
    main()
