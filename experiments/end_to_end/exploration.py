"""C30: cheap, train-only addressing pilots over immutable per-person R6 banks.

Run from this directory (the model/export environment is the same as branch_eval):
  python exploration.py --arm vsa --patches /banks/person*/final.pt \
    --anchor anchor.json --eval-set wiki.json --fineweb fineweb.json --code code.json
  python exploration.py --arm all --patches /banks/person*/final.pt \
    --anchor anchor.json --eval-set wiki.json --fineweb fineweb.json --code code.json
  python exploration.py --arm all --control-only

Defaults: exact fast MoE, generation batch size 1, at most 120 train prompts
per bank, 24 held-out direct prompts,
12 two-person composition prompts, 24 new tokens, and a 3300-second budget per
arm. Runtime limits are checked between forwards; one forward can overrun them.
Synthetic controls validate mechanics, not the frozen model's ability to recall.
All banks remain in host memory; ordinary arms load only the selected merge.
Interference uses two signed ALL-bank forwards per generated token and decodes
their logit difference. Its address proposal is explicitly dense cosine I.
No validation/evaluation prompts, answers, or preservation features fit keys,
CP factors, thresholds, or hyperparameters. No patch files are written.
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

if __package__:
    from . import branch_eval as B
else:
    import branch_eval as B

E, S = B.E, B.S
SPECS = {
    "vsa": (.15, "address accuracy <= I dense cosine at equal resident address bytes"),
    "phase": (.10, "address accuracy <= I dense cosine"),
    "kuramoto": (.05, "address accuracy gain < 0.05 over softmax-argmax, or false query rate > 2x baseline"),
    "interference": (.05, "composed recall < 0.5 * oracle composed recall"),
    "tensor": (.10, "address accuracy <= I dense cosine"),
}


def tensor_bytes(*values):
    return sum(v.numel()*v.element_size() for v in values)


def training_data(features, labels, attrs, general):
    x, g = (torch.as_tensor(v).detach().float().cpu() for v in (features, general))
    if (x.ndim != 2 or g.ndim != 2 or not len(x) or not len(g)
            or x.shape[1] != g.shape[1] or len(labels) != len(x) or len(attrs) != len(x)
            or not torch.isfinite(x).all() or not torch.isfinite(g).all()):
        raise ValueError("invalid training features/labels/attributes")
    return x, g, tuple(sorted(set(labels)))


def null_threshold(scores):
    """99th percentile of TRAIN anchor max score; strict > gives query-event FPR."""
    maxima = scores.detach().double().cpu().amax(1).sort().values
    if not len(maxima) or not torch.isfinite(maxima).all():
        raise ValueError("invalid null calibration")
    return float(maxima[math.ceil(.99*len(maxima))-1])


def routes_from_scores(scores, owners, threshold, top_k=1):
    if not 1 <= top_k <= len(owners) or not math.isfinite(threshold):
        raise ValueError("invalid selection budget/threshold")
    return [tuple(sorted(owners[j] for j in row.argsort(descending=True, stable=True)[:top_k].tolist()
                         if float(row[j]) > threshold)) for row in scores]


class Selector:
    def calibrate(self, general):
        self.threshold = null_threshold(self.scores(general))
        return self

    def select(self, features, top_k=1):
        return routes_from_scores(self.scores(features), self.owners, self.threshold, top_k)

    def storage(self):
        # Tensor payloads + fp64 threshold + int64 owner identities; not Python overhead.
        return 8 + 8*len(self.owners) + tensor_bytes(*[v for v in vars(self).values()
                                                   if torch.is_tensor(v)])


class DenseSelector(Selector):
    """I's centered fp64 dense cosine; coordinate truncation under a byte cap.

    Coordinates are a fixed prefix, never selected using held-out outcomes. For
    VSA's equal-byte comparison, unused budget is a real uint8 reserve buffer;
    the report gives useful bytes separately. Projection/codebooks count in VSA.
    """
    @classmethod
    def fit(cls, features, labels, attrs, general, byte_budget=None, softmax=False):
        x, g, owners = training_data(features, labels, attrs, general)
        d = x.shape[1]
        if byte_budget is not None:
            d = min(d, (byte_budget-8-8*len(owners))//(8*(len(owners)+1)))
            if d < 1:
                raise ValueError("address byte budget too small")
        result = cls()
        result.owners, result.width, result.softmax = owners, d, softmax
        index = B.BankIndexer.fit(x[:, :d], labels, g[:, :d])
        result.center, result.keys = index.center, index.keys
        result.calibrate(g)
        result.useful_bytes = result.storage()
        if byte_budget is not None:
            result.reserve = torch.zeros(byte_budget-result.useful_bytes, dtype=torch.uint8)
        return result

    def scores(self, features):
        x = torch.as_tensor(features).detach().double().cpu()[:, :self.width]
        scores = F.normalize(x-self.center, dim=-1) @ self.keys.T
        return (scores*8).softmax(-1) if self.softmax else scores


def bipolar(x):
    # Deterministic zero tie handling; sign-sum bundles stay bipolar.
    return torch.where(x >= 0, 1, -1).to(torch.int8)


def projection(width, dimensions, rng):
    if dimensions < 8:
        raise ValueError("need at least 8 projected coordinates")
    return (torch.randint(0, 3, (width, dimensions), generator=rng)-1).to(torch.int8)


class VSASelector(Selector):
    @classmethod
    def fit(cls, features, labels, attrs, general, dimensions=256, seed=0):
        x, g, owners = training_data(features, labels, attrs, general)
        result = cls()
        rng = torch.Generator().manual_seed(seed)
        result.owners, result.attributes = owners, tuple(sorted(set(attrs)))
        result.center = g.mean(0)
        result.project = projection(x.shape[1], dimensions, rng)
        result.people = bipolar(torch.randn(len(owners), dimensions, generator=rng))
        result.attributes_code = bipolar(torch.randn(len(result.attributes), dimensions, generator=rng))
        codes = result.encode(x)
        # One bundle per person: bind attribute and residual before binding person.
        keys = []
        for i, p in enumerate(owners):
            bound = [codes[j]*result.attributes_code[result.attributes.index(attrs[j])]
                     for j in range(len(x)) if labels[j] == p]
            keys.append(bipolar(torch.stack(bound).float().sum(0))*result.people[i])
        result.keys = torch.stack(keys)
        return result.calibrate(g)

    def encode(self, features):
        x = torch.as_tensor(features).detach().float().cpu()
        return bipolar((x-self.center) @ self.project.float())

    def scores(self, features):
        q = self.encode(features).float()
        # Candidate attribute unbinding, then person cleanup. No gold attribute
        # or entity lookup is available to the query; maximize over codebook attrs.
        person_unbound = (self.keys*self.people).float()
        return torch.einsum("nd,ad,bd->nba", q, self.attributes_code.float(), person_unbound).amax(-1)/q.shape[1]


class PhaseSelector(Selector):
    iterations = 3

    @classmethod
    def fit(cls, features, labels, attrs, general, dimensions=256, seed=0):
        x, g, owners = training_data(features, labels, attrs, general)
        result = cls()
        rng = torch.Generator().manual_seed(seed)
        result.owners, result.center = owners, g.mean(0)
        result.project = projection(x.shape[1], dimensions, rng)
        result.offset = torch.rand(dimensions, generator=rng)*2*math.pi
        codes = result.encode(x)
        keys = torch.stack([codes[torch.tensor(labels) == p].mean(0) for p in owners])
        result.keys = keys/keys.abs().clamp_min(1e-8)
        return result.calibrate(g)

    def encode(self, features):
        x = torch.as_tensor(features).detach().float().cpu()-self.center
        angle = x @ self.project.float()/math.sqrt(max(1, x.shape[1]*2/3)) + self.offset
        return torch.polar(torch.ones_like(angle), angle)

    def scores(self, features):
        q = self.encode(features)
        for _ in range(self.iterations):
            coherent = (q @ self.keys.conj().T).abs()/q.shape[1]
            weights = (coherent*8).softmax(-1).to(q.dtype)
            # Cleanup over person memories with global phase alignment; preserve
            # the original query at each update so cleanup does not erase evidence.
            overlap = q @ self.keys.conj().T
            alignment = overlap/overlap.abs().clamp_min(1e-8)
            clean = (weights*alignment) @ self.keys
            q = q + .25*clean
            q = q/q.abs().clamp_min(1e-8)
        return (q @ self.keys.conj().T).abs()/q.shape[1]


class KuramotoSelector(DenseSelector):
    steps = 50

    def scores(self, features):
        similarities = DenseSelector.scores(self, features)
        # A fixed query oscillator has phase zero. Near-query frequencies lock;
        # repulsive bank-bank coupling competes against collective firing.
        theta = torch.zeros_like(similarities)
        omega = 1-similarities
        drive = .2+similarities.clamp_min(0)
        locking = torch.zeros_like(theta)
        for step in range(self.steps):
            competition = torch.sin(theta[:, None, :]-theta[:, :, None]).mean(-1)
            velocity = omega-drive*theta.sin()-.4*competition
            theta = theta+.1*velocity
            if step >= self.steps-10:
                locking += theta.cos()/10
        return locking


def cp_reconstruct(factors):
    return torch.einsum("pr,ar,lr,ur->palu", *factors)


def cp_fit(tensor, rank=4, steps=40, seed=0):
    """Small signed 4-mode CP by ridge ALS, on TRAIN centroid usage only."""
    tensor = torch.as_tensor(tensor).detach().float().cpu()
    if (tensor.ndim != 4 or min(tensor.shape) < 1 or rank < 1 or steps < 1
            or not torch.isfinite(tensor).all()):
        raise ValueError("invalid CP tensor/rank/steps")
    rng = torch.Generator().manual_seed(seed)
    factors = [torch.randn(n, rank, generator=rng)/math.sqrt(rank) for n in tensor.shape]
    for _ in range(steps):
        for mode in range(4):
            others = [i for i in range(4) if i != mode]
            product = factors[others[0]]
            for i in others[1:]:
                product = (product[:, None, :]*factors[i][None, :, :]).reshape(-1, rank)
            unfolded = tensor.permute(mode, *others).reshape(tensor.shape[mode], -1)
            gram = product.T @ product + torch.eye(rank)*1e-5
            factors[mode] = torch.linalg.solve(gram, (unfolded @ product).T).T
        # Rescale factors without changing their product to avoid ALS overflow.
        norms = torch.stack([f.norm(dim=0).clamp_min(1e-8) for f in factors])
        target = norms.log().mean(0).exp()
        factors = [f*(target/n)[None] for f, n in zip(factors, norms)]
    return factors


class TensorSelector(Selector):
    @classmethod
    def fit(cls, features, labels, attrs, general, rank=4, steps=40, seed=0, units=64):
        # [prompt, layer, unit]; the fourth tensor mode is person, not prompt.
        x, g = (torch.as_tensor(v).detach().float().cpu() for v in (features, general))
        if (x.ndim != 3 or g.ndim != 3 or x.shape[1:] != g.shape[1:]
                or not len(x) or not len(g) or len(labels) != len(x) or len(attrs) != len(x)
                or units < 1 or not torch.isfinite(x).all() or not torch.isfinite(g).all()):
            raise ValueError("invalid tensor addressing data")
        result = cls()
        result.owners, result.attributes = tuple(sorted(set(labels))), tuple(sorted(set(attrs)))
        result.width = min(units, x.shape[2])
        result.center = g[:, :, :result.width].mean(0)
        x = x[:, :, :result.width]-result.center
        usage = torch.zeros(len(result.owners), len(result.attributes), *x.shape[1:])
        for p, pid in enumerate(result.owners):
            for a, attr in enumerate(result.attributes):
                indices = [i for i in range(len(x)) if labels[i] == pid and attrs[i] == attr]
                if indices:
                    usage[p, a] = x[indices].mean(0)
        result.person, result.attribute, result.layer, result.unit = cp_fit(usage, rank, steps, seed)
        result.reconstruction_mse = float((result.reconstruct()-usage).square().mean())
        return result.calibrate(g)

    def reconstruct(self):
        return cp_reconstruct([self.person, self.attribute, self.layer, self.unit])

    def scores(self, features):
        x = torch.as_tensor(features).detach().float().cpu()[:, :, :self.width]-self.center
        q = F.normalize(x.flatten(1), dim=-1)
        keys = F.normalize(self.reconstruct().flatten(2), dim=-1)
        return torch.einsum("nd,bad->nba", q, keys).amax(-1)


def fit_selector(arm, train, labels, attrs, general, args):
    if arm == "vsa":
        return VSASelector.fit(train, labels, attrs, general, args.dimensions, args.seed)
    if arm == "phase":
        return PhaseSelector.fit(train, labels, attrs, general, args.dimensions, args.seed)
    if arm == "kuramoto":
        return KuramotoSelector.fit(train, labels, attrs, general)
    if arm == "tensor":
        return TensorSelector.fit(train, labels, attrs, general, args.cp_rank, args.cp_steps,
                                  args.seed, args.tensor_units)
    return DenseSelector.fit(train, labels, attrs, general)


def sign_patterns(owners, selected, seed):
    """Random nuisance signs; selected signs +1 then flipped for joint decode."""
    rng = random.Random(seed)
    selected = set(selected)
    if selected-set(owners):
        raise ValueError("unknown selected bank")
    first = {p: 1 if p in selected else rng.choice((-1, 1)) for p in owners}
    second = {p: -v if p in selected else v for p, v in first.items()}
    return first, second


def signed_patch(states, signs):
    """Exact sparse signed sum, without clipping collisions to ternary codes."""
    if set(states) != set(signs) or any(s not in (-1, 1) for s in signs.values()):
        raise ValueError("need one bipolar sign per resident bank")
    ref = next(iter(states.values()))
    entries = {}
    for name, entry in ref["entries"].items():
        indices = torch.cat([s["entries"][name]["idx"].long() for s in states.values()])
        values = torch.cat([s["entries"][name]["val"].float()*signs[p] for p, s in states.items()])
        unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        sums = torch.zeros(len(unique)).scatter_add_(0, inverse, values)
        keep = sums != 0
        entries[name] = dict(entry, idx=unique[keep].to(torch.int32), val=sums[keep])
        # R6 uses full masks; resetting complete units is not linear superposition.
        if any(any(k in s["entries"][name] for k in
                   ("unit_reset_rows", "unit_reset_cols", "address_start_idx")) for s in states.values()):
            raise ValueError("interference requires ordinary R6 banks without unit/address resets")
    return dict(ref, entries=entries)


@contextmanager
def weighted_forward(branches, patch):
    """Temporarily interpret signed sums as exact code weights; restore on errors."""
    original = []
    enabled = E.FSAParam.enabled
    try:
        # Load uses the unchanged ternary format/row scales and validates masks.
        E.load_patch(branches.model, patch)
        for _, lin in E.patched_linears(branches.model):
            pm = lin.parametrizations.weight[0]
            original.append((pm, pm.__dict__.get("_quant")))
            pm._quant = lambda m: m
        branches.model._c16_branch_key = None
        E.set_enabled(True)
        yield
    finally:
        for pm, quant in original:
            if quant is None:
                pm.__dict__.pop("_quant", None)
            else:
                pm._quant = quant
        branches.model._c16_branch_key = None
        branches.activate(())
        E.set_enabled(enabled)


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("per-arm wall budget exhausted")


@torch.no_grad()
def interference_generate(branches, items, routes, dev, max_new, seed, deadline):
    outputs, forwards = [], 0
    for i, (item, route) in enumerate(zip(items, routes)):
        check_deadline(deadline)
        if not route:
            branches.activate(())
            outputs.extend(E.greedy_generate(branches.model, [item["p"]], dev, max_new, 1))
            continue
        signs = sign_patterns(tuple(branches.states), route, seed+i)
        patches = [signed_patch(branches.states, s) for s in signs]
        sequence, generated = list(item["p"]), []
        for _ in range(max_new):
            logits = []
            for patch in patches:
                check_deadline(deadline)
                with weighted_forward(branches, patch):
                    logits.append(E.logits_of(branches.model, torch.tensor([sequence], device=dev))[0, -1, :E.VOCAB].float())
                forwards += 1
            # Contrast isolates writes; retain frozen-base language logits.
            branches.activate(())
            base_logits = E.logits_of(branches.model,torch.tensor([sequence],device=dev))[0,-1,:E.VOCAB].float()
            forwards += 1
            token = int((base_logits+(logits[0]-logits[1])*.5).argmax())
            generated.append(token)
            sequence.append(token)
            if token in E.EOS_IDS:
                break
        outputs.append(generated)
    return outputs, forwards


def positive_control(arm):
    """Run before any real-bank/model IO, with no dependencies on external data."""
    if arm == "interference":
        patches = torch.eye(4)
        first, second = sign_patterns((2, 5, 7, 9), (5, 9), 17)
        plus = torch.tensor(list(first.values())).float() @ patches
        minus = torch.tensor(list(second.values())).float() @ patches
        recovered = (plus-minus)*.5
        expected = torch.tensor([0., 1., 0., 1.])
        return dict(passed=bool(torch.equal(recovered, expected)),
                    kind="linear signed superposition; nuisance cancellation and two selected banks",
                    max_error=float((recovered-expected).abs().max()))
    labels = [2, 5, 9]*4
    attrs = ["synthetic"]*len(labels)
    x = torch.eye(12)[:3].repeat(4, 1)*4
    general = torch.zeros(64, 12)
    if arm == "vsa":
        index = VSASelector.fit(x, labels, attrs, general, 256, 11)
    elif arm == "phase":
        index = PhaseSelector.fit(x, labels, attrs, general, 256, 11)
    elif arm == "kuramoto":
        index = KuramotoSelector.fit(x, labels, attrs, general)
    else:
        x = x[:, None, :].repeat(1, 2, 1)
        general = general[:, None, :].repeat(1, 2, 1)
        index = TensorSelector.fit(x, labels, attrs, general, rank=3, steps=40, seed=11, units=12)
    accuracy = B.selection_metrics(index.select(x), [(p,) for p in labels])["accuracy"]
    return dict(passed=accuracy == 1. and not any(index.select(general)), kind="three orthogonal bank memories; noncontiguous owners",
                address_accuracy=accuracy, false_activations=sum(bool(r) for r in index.select(general)))


def kill_decision(arm, result):
    if arm == "interference":
        oracle = result["oracle"]["composed_recall"]
        if oracle <= 0:
            return dict(verdict="inconclusive", reason="oracle composed recall is zero")
        killed = result["pilot"]["composed_recall"] < .5*oracle
    else:
        gain = result["pilot"]["address_accuracy"]-result["baseline"]["address_accuracy"]
        killed = gain <= 0
        if arm == "kuramoto":
            killed = (gain < .05-1e-12 or result["pilot"]["false_activations"]["query_rate"] >
                      2*result["baseline"]["false_activations"]["query_rate"])
    return dict(verdict="kill" if killed else "survive", rule=SPECS[arm][1])


def emit(arm, control, result, started, verdict=None):
    report = dict(arm=arm, prior=SPECS[arm][0], kill=SPECS[arm][1], control=control,
                  result=result, wall_seconds=time.perf_counter()-started)
    report.update(verdict="control-failed" if not control["passed"] else verdict or kill_decision(arm, result)["verdict"])
    print("EXPLORE " + json.dumps(report, allow_nan=False), flush=True)
    return report


def load_runtime(args):
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in args.patches]
    banks = B.validate_patches(states)
    meta = states[0]["meta"]
    for state in states:
        E.validate_patch_export(state["meta"], args.export, args.export_step)
    facts = E.make_world(meta["seed"], meta["people"], meta["facts_per_person"], meta["shared_attrs"], meta["values"])
    panels, panel_tokens = {}, {}
    for name, path in (("WikiText", args.eval_set), ("FineWeb", args.fineweb), ("code", args.code)):
        tokens = json.loads(path.read_text())
        if len(tokens) < 40*511:
            raise ValueError(f"{name} requires 40x511 tokens")
        panel_tokens[name] = tokens[:40*511]
        panels[name] = [[E.BOS]+tokens[i*511:(i+1)*511] for i in range(40)]
    anchor = json.loads(args.anchor.read_text())
    if len(anchor) < S.NONE_WINDOWS*(S.WINDOW_LENGTH-1):
        raise ValueError("anchor requires 64 disjoint 95-token training windows")
    E.checked_overlap(anchor, panel_tokens, "exploration anchor")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    torch.manual_seed(meta["seed"])
    tok = Tokenizer.from_file(str(E.DATA/"tokenizer.json"))
    model, _, dev = build_model(args.export, device="cuda:0", model_config=None, step=args.export_step,
                               expert_format="bf16", eda_kernel_backend="naive_fp32")
    model._fsa_export, model._fsa_export_step = args.export, args.export_step
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if args.fast_moe == "exact":
        if __package__:
            from .fast_moe import install
        else:
            from fast_moe import install
        install(model, patched_layer_ids=meta["layers"], mode="exact")
    E.attach(model, meta["layers"], "fsa_forced", 0, projs=meta["projs"], sparse=True,
             slot_frac=meta["slot_frac"], seed=meta["seed"],
             projs_by_layer=E.layer_projections(meta["layers"], meta.get("value_layers", meta["layers"]), meta["projs"]))
    E.apply_projection_scales(model, meta.get("patch_scale_mult", 1.),
                              meta.get("patch_scale_mult_up"), meta.get("patch_scale_mult_down"))
    for state in states:
        E.load_patch(model, state)
    for param in model.parameters():
        param.requires_grad_(False)
    branches = B.Branches(model, banks, {s["meta"]["train_person"]: p.stat().st_size
                                       for s, p in zip(states, args.patches)}, "sum_clip")
    branches.activate(())
    return branches, tok, facts, panels, anchor, dev


def features_for_run(branches, tok, facts, panels, anchor, dev, args, layers, deadline=math.inf):
    train = S.build_panels(tok, facts, E.BOS, question_train=False)["train"]
    sampled = []
    rng = random.Random(args.seed)
    for pid in sorted(branches.states):
        own = [it for it in train if facts[it["fi"]].pid == pid]
        sampled.extend(rng.sample(own, min(args.train_per_bank, len(own))))
    train = sampled
    windows, positions, _ = S.general_windows(anchor, args.seed, E.BOS)
    direct = E.build_items(tok, facts, E.TEST_T, E.BOS)
    direct = random.Random(args.seed).sample(direct, min(args.qa_n, len(direct)))
    composed = B.composition_items(tok, facts, E.BOS, args.compose_n)
    if not composed:
        raise ValueError("composed recall requires two-person shared-attribute facts")
    sequences = {"train": [it["p"] for it in train], "anchor": windows,
                 "direct": [B.question_sequence(tok, it["p"], E.BOS) for it in direct],
                 "composed": [it["p"] for it in composed]}
    # Exactly 40 unrelated completion prompts per preservation source.
    sequences.update({name: [win[:64] for win in wins] for name, wins in panels.items()})
    features = {}
    for name, seqs in sequences.items():
        enabled = E.FSAParam.enabled
        try:
            E.set_enabled(False)
            batches = []
            for start in range(0, len(seqs), S.BATCH_SIZE):
                check_deadline(deadline)
                batches.append(S.collect_residuals(branches.model, seqs[start:start+S.BATCH_SIZE], layers, dev,
                                                  positions[start:start+S.BATCH_SIZE] if name == "anchor" else None))
            features[name] = {li: torch.cat([batch[li] for batch in batches]) for li in layers}
        finally:
            E.set_enabled(enabled)
    return train, direct, composed, features


def evaluate_routes(branches, tok, facts, direct, composed, routes, null_routes, dev, args, deadline,
                    interference=False):
    generations, forwards = {}, 0
    for name, items in (("direct", direct), ("composed", composed)):
        if interference:
            generations[name], used = interference_generate(branches, items, routes[name], dev,
                                                            args.gen_max_new, args.seed, deadline)
            forwards += used
        else:
            generations[name] = []
            for item, route in zip(items, routes[name]):
                check_deadline(deadline)
                generations[name].extend(B.routed_generations(branches, tok, [item], [route], dev,
                                                              args.gen_max_new, 1))
    direct_metric = B.selection_metrics(routes["direct"], B.gold_routes(direct, facts))
    composed_metric = B.selection_metrics(routes["composed"], B.gold_routes(composed, facts))
    false = {name: dict(n=40, queries=sum(bool(r) for r in rr), query_rate=sum(bool(r) for r in rr)/40,
                       selected_banks=sum(map(len, rr)),
                       applied_banks=sum((len(branches.states) if interference and r else len(r)) for r in rr))
             for name, rr in null_routes.items()}
    report = dict(address_accuracy=direct_metric["accuracy"], address=direct_metric,
                  composed_address=composed_metric,
                  direct_recall=B.generation_summary(tok, direct, generations["direct"], E.conflict_map(facts),
                                                     args.gen_stop_k)["recall"],
                  composed_recall=sum(B.R.binding_score(tok.decode(gen), item, facts)
                                      for item, gen in zip(composed, generations["composed"]))/len(composed),
                  n_composed=len(composed), false_activations=false["WikiText"],
                  false_activations_by_source=false,
                  resident_banks=sorted(branches.states), resident_patch_bytes=sum(branches.bytes.values()),
                  residency="all banks in host memory; active sparse merge applied per query",
                  contrast_forwards=forwards)
    if interference:
        report["residency"] = "all banks resident and signed in each of two forwards; selected signs flipped"
    return report



def zero_patch_control(branches,tok,direct,composed,dev,args,deadline,interference=False):
    items = direct[:1]+composed[:1]
    branches.activate(())
    E.set_enabled(False)
    baseline = E.greedy_generate(branches.model,[it['p'] for it in items],dev,args.gen_max_new,1)
    branches.activate(())
    E.set_enabled(True)
    ordinary = B.routed_generations(branches,tok,items,[()]*len(items),dev,args.gen_max_new,1)
    passed = baseline==ordinary
    if interference:
        blank_entries=branches.cache[()]['entries']
        zeros={pid:dict(state,entries=blank_entries) for pid,state in branches.states.items()}
        empty=B.Branches(branches.model,zeros,branches.bytes,branches.rule)
        routed,_=interference_generate(empty,items,[tuple(zeros)]*len(items),dev,args.gen_max_new,args.seed,deadline)
        passed = passed and routed==baseline
    branches.model._c16_branch_key=None
    branches.activate(())
    return dict(passed=passed,kind='zero sparse writes reproduce frozen-base GEN through actual readout')

def run_arm(arm, runtime, data, args, tensor_layers, control, started, shared_seconds=0.):
    branches, tok, facts, panels, anchor, dev = runtime
    train, direct, composed, features = data
    deadline = started+args.max_seconds-shared_seconds
    labels, attrs = [facts[it["fi"]].pid for it in train], [facts[it["fi"]].attr for it in train]
    dense_features = {name: per_layer[args.indexer_layer] for name, per_layer in features.items()}
    pilot_features = ({name: torch.stack([per_layer[li] for li in tensor_layers], 1)
                       for name, per_layer in features.items()} if arm == "tensor" else dense_features)
    try:
        check_deadline(deadline)
        pilot = fit_selector(arm, pilot_features["train"], labels, attrs, pilot_features["anchor"], args)
        baseline = DenseSelector.fit(dense_features["train"], labels, attrs, dense_features["anchor"],
                                     byte_budget=pilot.storage() if arm == "vsa" else None,
                                     softmax=arm == "kuramoto")
        zero = zero_patch_control(branches,tok,direct,composed,dev,args,deadline,arm=='interference')
        control = dict(control,passed=control['passed'] and zero['passed'])
        result = dict(zero_patch_control=zero)
        for name, index, ff in (("pilot", pilot, pilot_features), ("baseline", baseline, dense_features)):
            check_deadline(deadline)
            routes = {group: index.select(ff[group], min(2, len(index.owners)) if group == "composed" else 1)
                      for group in ("direct", "composed")}
            nulls = {source: index.select(ff[source]) for source in panels}
            result[name] = evaluate_routes(branches, tok, facts, direct, composed, routes, nulls, dev, args,
                                            deadline, interference=arm == "interference" and name == "pilot")
            result[name].update(address_bytes=index.storage(), threshold=index.threshold,
                                useful_address_bytes=getattr(index, "useful_bytes", index.storage()))
        if arm == "interference":
            routes = {"direct": B.gold_routes(direct, facts), "composed": B.gold_routes(composed, facts)}
            result["oracle"] = evaluate_routes(branches, tok, facts, direct, composed, routes,
                                                {name: [()]*40 for name in panels}, dev, args, deadline)
            result["address_method"] = "TRAIN-only I dense cosine proposal; two signed forwards plus frozen-base logits"
        evidence = B.R.supplied_evidence(tok, composed, facts, E.encode, E.BOS)
        gens = B.routed_generations(branches,tok,evidence,B.gold_routes(evidence,facts),dev,args.gen_max_new,1)
        evidence_recall = sum(B.R.binding_score(tok.decode(g),it,facts) for it,g in zip(evidence,gens))/len(evidence)
        result['supplied_evidence_composition'] = dict(binding=evidence_recall,n=len(evidence))
        control = dict(control, passed=control['passed'] and evidence_recall==1.)
        result["baseline_method"] = "softmax-argmax" if arm == "kuramoto" else "I dense cosine"
        result["training"] = "TRAIN_T prompts and disjoint anchor positions only"
        result["composition_selection"] = "thresholded top-2, fixed before evaluation"
        result["wall_budget_seconds"] = args.max_seconds
        result["shared_setup_seconds"] = shared_seconds
        if arm == "tensor":
            result.update(tensor_modes=["person", "attribute", "layer", "unit"], tensor_layers=tensor_layers,
                          cp_train_reconstruction_mse=pilot.reconstruction_mse,
                          missing_attribute_cells="zero usage", tensor_unit_selection="fixed prefix")
        if arm == "vsa":
            result["equal_byte_comparison"] = "all tensor payloads including projection, codes, center and dense reserve"
        return emit(arm, control, result, started-shared_seconds)
    except TimeoutError:
        return emit(arm, control, None, started-shared_seconds, "budget-exceeded")
    finally:
        branches.activate(())
        E.set_enabled(False)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=tuple(SPECS)+("all",))
    p.add_argument("--control-only", "--positive-control", dest="control_only", action="store_true")
    p.add_argument("--patches", type=Path, nargs="+")
    p.add_argument("--export", type=Path, default=E.EXPORT)
    p.add_argument("--export-step", type=int, default=E.EXPORT_STEP)
    for name in ("anchor", "eval-set", "fineweb", "code"):
        p.add_argument("--"+name, type=Path)
    p.add_argument("--fast-moe", choices=("exact", "off"), default="exact", help="off for CPU fixtures")
    p.add_argument("--gen-bs", type=int, choices=(1,), default=1)
    p.add_argument("--indexer-layer", type=int, default=26)
    p.add_argument("--tensor-layers", help="comma-separated layers; default L-2,L")
    p.add_argument("--dimensions", type=int, default=256)
    p.add_argument("--tensor-units", type=int, default=64)
    p.add_argument("--cp-rank", type=int, default=4)
    p.add_argument("--cp-steps", type=int, default=40)
    p.add_argument("--train-per-bank", type=int, default=120)
    p.add_argument("--qa-n", type=int, default=24)
    p.add_argument("--compose-n", type=int, default=12)
    p.add_argument("--gen-max-new", type=int, default=24)
    p.add_argument("--gen-stop-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=3300.)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.indexer_layer < 0 or not 8 <= args.dimensions <= 4096 or min(args.tensor_units, args.cp_rank, args.cp_steps,
            args.train_per_bank,
            args.qa_n, args.compose_n, args.gen_max_new, args.gen_stop_k) < 1
            or args.qa_n > 120 or args.compose_n > 60 or args.gen_max_new > 64 or args.cp_steps > 200
            or args.cp_rank > 16 or args.tensor_units > 1024 or args.train_per_bank > 512
            or not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600):
        parser.error("invalid pilot size or runtime budget")
    if not args.control_only and any(getattr(args, k) is None for k in
                                    ("patches", "anchor", "eval_set", "fineweb", "code")):
        parser.error("real evaluation needs --patches --anchor --eval-set --fineweb --code")
    arms = list(SPECS) if args.arm == "all" else [args.arm]
    controls, reports, active = {}, [], []
    for arm in arms:
        started = time.perf_counter()
        print(f"PRIOR {arm} {SPECS[arm][0]:.0%}; KILL {SPECS[arm][1]}", flush=True)
        controls[arm] = positive_control(arm)
        print("POSITIVE-CONTROL " + json.dumps(dict(arm=arm, **controls[arm])), flush=True)
        if not controls[arm]["passed"] or args.control_only:
            reports.append(emit(arm, controls[arm], None, started,
                                "control-passed" if controls[arm]["passed"] else "control-failed"))
        else:
            active.append(arm)
    if not active:
        return reports
    tensor_layers = list(dict.fromkeys((max(0, args.indexer_layer-2), args.indexer_layer)))
    if args.tensor_layers:
        try:
            tensor_layers = [int(li) for li in args.tensor_layers.split(",")]
        except ValueError:
            parser.error("invalid --tensor-layers")
        if not tensor_layers or min(tensor_layers) < 0 or len(set(tensor_layers)) != len(tensor_layers):
            parser.error("need distinct nonnegative --tensor-layers")
    layers = sorted(set([args.indexer_layer]+(tensor_layers if "tensor" in active else [])))
    shared_start = time.perf_counter()
    runtime = load_runtime(args)
    if max(layers) >= len(runtime[0].model.layers):
        parser.error("addressing layer outside model.layers")
    try:
        data = features_for_run(*runtime, args, layers, shared_start+args.max_seconds)
    except TimeoutError:
        runtime[0].activate(())
        E.set_enabled(False)
        return reports+[emit(arm, controls[arm], None, shared_start, "budget-exceeded") for arm in active]
    shared_seconds = time.perf_counter()-shared_start
    print("EXPLORE-SHARED " + json.dumps(dict(wall_seconds=shared_seconds, layers=layers,
                                             n_train=len(data[0]), n_direct=len(data[1]), n_composed=len(data[2]))), flush=True)
    for arm in active:
        started = time.perf_counter()
        reports.append(run_arm(arm, runtime, data, args, tensor_layers, controls[arm], started, shared_seconds))
    return reports


if __name__ == "__main__":
    main()
