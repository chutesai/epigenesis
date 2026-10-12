from itertools import combinations
"""C27 train-only addressing prototypes; persistent fact patches stay fixed.

The explicit mention table and diagonal resolver are prototypes, not recurrent
model state. All thresholds use TRAIN anchor positions, never preservation data.
"""
import math
import re
import time

import torch
from torch.nn import functional as F

if __package__:
    from . import branch_eval as B
else:
    import branch_eval as B
E, S = B.E, B.S


def null_threshold(scores, fpr=.01):
    """Strict > threshold; calibrate the whole query event, including distractors."""
    scores = torch.as_tensor(scores).detach().double().cpu()
    if scores.ndim != 2 or not len(scores) or not scores.shape[1] or not torch.isfinite(scores).all():
        raise ValueError("invalid calibration scores")
    maxima = scores.amax(1).sort().values
    return float(maxima[max(0, math.ceil((1-fpr)*len(maxima))-1)])


def bounded_routes(scores, owners, threshold, budget):
    if not isinstance(budget, int) or budget < 1 or not math.isfinite(threshold):
        raise ValueError("invalid activation budget/threshold")
    return [tuple(sorted(owners[j] for j in row.argsort(descending=True, stable=True)[:budget].tolist()
                         if float(row[j]) > threshold)) for row in scores]


def binary_key(x, positive, steps, importance=None):
    """Balanced logistic fit with deterministic zero initialization."""
    x = x.detach().float().cpu()
    y = torch.as_tensor(positive, dtype=torch.float32)
    if not y.bool().any() or y.bool().all():
        raise ValueError("binary gate needs both classes")
    mass = torch.ones_like(y) if importance is None else torch.as_tensor(importance).float().clone()
    for flag in (False, True):
        mask = y.bool() == flag
        mass[mask] /= mass[mask].sum().clamp_min(1)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ w+b, y, reduction="none")*mass).sum()
        loss = loss + .001*w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def ternary(w):
    cutoff = w.abs().sort().values[max(0, len(w)*2//3-1)]
    return (w.sign()*(w.abs() > cutoff)).to(torch.int8)


class InhibitoryIndexer:
    """I positive key plus quantized logistic nearest-confusion exclusions."""
    def __init__(self, positive, exclusions, control, strength, budget, real_owners):
        self.positive, self.exclusions, self.control = positive, exclusions, control
        self.strength, self.budget, self.real_owners = strength, budget, tuple(real_owners)
        self.thresholds = {}
        self.offsets = {}

    @classmethod
    def fit(cls, features, labels, attrs, general, exclusions=2, distractors=10,
            strength=1., budget=3, seed=0, steps=100):
        if (exclusions < 1 or distractors < 0 or steps < 1 or budget < 1
                or not math.isfinite(strength) or strength < 0 or len(attrs) != len(features)):
            raise ValueError("invalid INH parameters")
        positive = B.BankIndexer.fit(features, labels, general)
        real = positive.owners
        x = F.normalize(torch.as_tensor(features).float()-positive.center.float(), dim=-1)
        g = F.normalize(torch.as_tensor(general).float()-positive.center.float(), dim=-1)
        y = torch.as_tensor(labels)
        negative, controls, confusions = [], [], {}
        for col, pid in enumerate(real):
            own = y == pid
            others = torch.where(~own)[0]
            own_attrs = {attr for attr, p in zip(attrs, labels) if p == pid}
            # Highest scoring other-bank prompts first; include same-attribute impostors.
            order = others[(x[others].double() @ positive.keys[col]).argsort(descending=True, stable=True)]
            hard = list(dict.fromkeys(order[:exclusions].tolist() +
                        [i for i in order.tolist() if attrs[i] in own_attrs]))
            confusions[str(pid)] = hard
            nk, ck = [], []
            for j in range(exclusions):
                chosen = hard[j::exclusions] or hard[:1]
                if not chosen:
                    raise ValueError("INH needs multiple real banks")
                xx = torch.cat((x[own], x[chosen], g))
                target = [False]*int(own.sum()) + [True]*len(chosen) + [False]*len(g)
                w, _ = binary_key(xx, target, steps)
                nk.append(ternary(w))
                # Matched positive-only ensemble: same ternary coordinates, trainable slots,
                # confusions and logistic fitting budget, without subtractive inhibition.
                target = [True]*int(own.sum()) + [False]*(len(chosen)+len(g))
                w, _ = binary_key(xx, target, steps)
                ck.append(ternary(w))
            negative.append(torch.stack(nk))
            controls.append(torch.stack(ck))
        rng = torch.Generator().manual_seed(seed)
        random_keys = F.normalize(torch.randn(distractors, x.shape[1], generator=rng), dim=-1).double()
        owners = real + tuple(range(max(real)+1, max(real)+1+distractors))
        positive = B.BankIndexer(torch.cat((positive.keys, random_keys)), positive.center, owners)
        zeros = torch.zeros(distractors, exclusions, x.shape[1], dtype=torch.int8)
        result = cls(positive, torch.cat((torch.stack(negative), zeros)),
                     torch.cat((torch.stack(controls), zeros)), strength, budget, real)
        result.confusions = confusions
        for mode in ("inhibitory", "positive_only"):
            train_scores, general_scores = result.scores(features, mode), result.scores(general, mode)
            # TRAIN positives set bank-specific midpoints; the final common shift
            # calibrates max-over-all-candidates on general TRAIN queries.
            offsets = torch.zeros(len(owners), dtype=torch.double)
            for col, pid in enumerate(real):
                offsets[col] = (train_scores[y == pid, col].median()+general_scores[:, col].median())/2
            result.offsets[mode] = offsets
            result.thresholds[mode] = null_threshold(general_scores-offsets)
        result.train = {mode: B.selection_metrics(result.select(features, mode), [(p,) for p in labels])
                        for mode in result.thresholds}
        result.train_fpr = {mode: sum(bool(r) for r in result.select(general, mode))/len(general)
                            for mode in result.thresholds}
        return result

    def scores(self, features, mode="inhibitory"):
        q = F.normalize(torch.as_tensor(features).double()-self.positive.center, dim=-1)
        keys = self.exclusions if mode == "inhibitory" else self.control
        keys = F.normalize(keys.double(), dim=-1)
        extra = torch.einsum("nd,bjd->nbj", q, keys).amax(-1).clamp_min(0)
        scores = q @ self.positive.keys.T + (-1 if mode == "inhibitory" else 1)*self.strength*extra
        return scores-self.offsets.get(mode, 0)

    def select(self, features, mode="inhibitory"):
        return bounded_routes(self.scores(features, mode), self.positive.owners,
                              self.thresholds[mode], self.budget)

    def metadata(self):
        address = (self.positive.keys.numel()*8 + self.exclusions.numel() +
                   self.positive.center.numel()*8 + 8*(len(self.positive.owners)+1))
        return dict(address_bytes=address, control_address_bytes=address, exclusions_dtype="int8 ternary",
                    strength=self.strength, budget=self.budget, thresholds=self.thresholds,
                    bank_train_offsets={mode: offset.tolist() for mode, offset in self.offsets.items()},
                    distractor_banks=len(self.positive.owners)-len(self.real_owners),
                    train=self.train, train_general_query_fpr=self.train_fpr,
                    hard_negative_train_indices=self.confusions,
                    calibration="tau_b = TRAIN own/general score midpoint + null shift; general TRAIN maximum across ALL candidates, strict >")


class ConjunctiveIndexer:
    """Equal-state entity/relation product and additive controls at person-bank granularity."""
    @classmethod
    def fit(cls, features, labels, attrs, general, steps=100, budget=3):
        if steps < 1 or budget < 1 or len(attrs) != len(features):
            raise ValueError("invalid CONJ parameters")
        result = cls()
        result.center = torch.as_tensor(general).float().mean(0)
        x = F.normalize(torch.as_tensor(features).float()-result.center, dim=-1)
        g = F.normalize(torch.as_tensor(general).float()-result.center, dim=-1)
        y = torch.as_tensor(labels)
        result.owners, result.budget = tuple(sorted(set(labels))), budget
        result.entity, result.relation = [], []
        for pid in result.owners:
            result.entity.append(binary_key(torch.cat((x, g)), [p == pid for p in labels]+[False]*len(g), steps))
            own = y == pid
            relation = []
            for attr in sorted({a for a, p in zip(attrs, labels) if p == pid}):
                # Same-person other attributes are negatives; general text is also negative.
                target = [a == attr for a, p in zip(attrs, labels) if p == pid] + [False]*len(g)
                relation.append(binary_key(torch.cat((x[own], g)), target, steps))
            result.relation.append(relation)
        result.thresholds = {mode: null_threshold(result.scores(general, mode))
                             for mode in ("conjunctive", "additive")}
        return result

    def scores(self, features, mode="conjunctive"):
        x = F.normalize(torch.as_tensor(features).float()-self.center, dim=-1)
        out = []
        for (w, b), relations in zip(self.entity, self.relation):
            entity = (x @ w+b).sigmoid()
            relation = torch.stack([(x @ w+b).sigmoid() for w, b in relations], 1).amax(1)
            out.append(entity*relation if mode == "conjunctive" else (entity+relation)/2)
        return torch.stack(out, 1)

    def select(self, features, mode="conjunctive"):
        return bounded_routes(self.scores(features, mode), self.owners, self.thresholds[mode], self.budget)

    def metadata(self):
        keys = self.entity + [key for bank in self.relation for key in bank]
        size = self.center.numel()*4 + sum(w.numel()*4+4 for w, b in keys)+8
        return dict(address_bytes=size, control_address_bytes=size, thresholds=self.thresholds,
                    budget=self.budget, relation_gates=sum(map(len, self.relation)),
                    limitation="A person patch contains all its attributes: relation gates OR within bank; no attribute-specific value compartments.")


def conditional_summary(hits, correct):
    n = sum(correct)
    return dict(n_address_correct=n, accuracy=sum(h and c for h, c in zip(hits, correct))/n if n else None)


def route_report(branches, tok, facts, groups, routes, panels, panel_routes, base_nll, dev, a, benchmark_route=None):
    """Score raw candidate routes; no-patch distractors remain visible in errors."""
    report = dict(address_correctness={}, conditional_value_correctness={}, composition={}, end_to_end={},
                  false_activations={}, active_bytes={})
    conflicts = E.conflict_map(facts)
    started = time.perf_counter()
    def real(route):
        return tuple(p for p in route if p in branches.states)
    try:
        for name, group in groups.items():
            if name == "indirect" or not group:
                continue
            selected = routes[name]
            expected = [()] * len(group) if name == "general" else B.gold_routes(group, facts)
            report["address_correctness"][name] = B.selection_metrics(selected, expected)
            correct = [set(r) == set(e) for r, e in zip(selected, expected)]
            limit = a.gen_max_new or (40 if name in ("two_fact", "g_set") else
                                     max(len(it.get("a", [])) for it in group)+a.gen_stop_k+2)
            gens = B.routed_generations(branches, tok, group, [real(r) for r in selected], dev, limit, a.gen_bs)
            if name == "general":
                report["false_activations"][name] = false_summary(selected, expected, branches.states)
                continue
            if name in ("two_fact", "g_set"):
                hits = [B.R.binding_score(tok.decode(g), it, facts) for it, g in zip(group, gens)]
                report["composition"][name] = dict(n=len(group), binding=sum(hits)/len(group),
                    lenient=sum(E.score_two_fact_lenient(tok, g, it["vals"]) for it, g in zip(group, gens))/len(group),
                    strict=sum(E.score_two_fact_generation(tok, g, it["vals"], a.gen_stop_k)["both"]
                               for it, g in zip(group, gens))/len(group))
                report["end_to_end"][name] = dict(n=len(group), recall=sum(hits)/len(group))
            else:
                hits = [E.score_generation(tok, g, it["val"], a.gen_stop_k, conflicts.get(it["fi"], ()))
                        for it, g in zip(group, gens)]
                report["end_to_end"][name] = dict(n=len(group), recall=sum(h["clean"] for h in hits)/len(group),
                    cross_person_substitutions=sum(h["confused"] for h in hits)/len(group))
                if all("cue" in it for it in group):
                    report["end_to_end"][name]["by_cue"] = {
                        cue: dict(n=sum(it["cue"] == cue for it in group),
                                  recall=sum(h["clean"] for it, h in zip(group, hits) if it["cue"] == cue)/
                                         sum(it["cue"] == cue for it in group))
                        for cue in sorted({it["cue"] for it in group})}
                hits = [h["clean"] for h in hits]
                if name == "two_hop":
                    report["composition"][name] = report["end_to_end"][name]
            report["conditional_value_correctness"][name] = conditional_summary(hits, correct)
            report["false_activations"][name] = false_summary(selected, expected, branches.states)
            report["active_bytes"][name] = sum(branches.active_bytes(real(r)) for r in selected)/len(group)
        dispatch = {name: [real(r) for r in rows] for name, rows in panel_routes.items()}
        report["damage"] = B.damage(branches, panels, base_nll, dispatch, dev)
        report["false_activations"]["panels"] = {name: false_summary(rows, [()]*len(rows), branches.states)
                                                 for name, rows in panel_routes.items()}
        bench = None
        if getattr(a, '_bench_items', None) is not None and benchmark_route is not None:
            bench = B.routed_benchmark(branches,a._bench_items,a._base_bench,benchmark_route,dev)
            report['bench'] = bench
        report["damage_gate"] = E.damage_gate(report["damage"],bench)
    finally:
        branches.activate(())
    report["evaluation_seconds"] = time.perf_counter()-started
    report["conditional_measure"] = "clean value generation conditional on exact target-bank set; null for zero correct addresses"
    return report


def false_summary(routes, expected, real):
    n = len(routes)
    return dict(**B.R.false_activation([tuple(set(r)-set(e)) for r, e in zip(routes, expected)]),
                banks_per_query=sum(len(set(r)-set(e)) for r, e in zip(routes, expected))/max(n, 1),
                distractor_query_rate=sum(bool(set(r)-set(real)) for r in routes)/max(n, 1))


def kill_decision(arm, reports):
    if arm == "INH":
        new, old = reports["inhibitory"], reports["positive_only"]
        nf, of = (r["false_activations"].get("independent_general", r["false_activations"]["general"]) for r in (new, old))
        nr, ore = (r["end_to_end"]["test"]["recall"] for r in (new, old))
        sufficient = nf.get('n', 0) >= 3000 and of.get('n', 0) >= 3000
        established = sufficient and nf['upper_95'] <= of['lower_95']/10+1e-12
        return dict(rule="needs statistically established >=10x reduction, >=3000 independent negatives, <=1 pt recall loss",
                    passed=(established and nr >= ore-.01-1e-12) if sufficient else None, sufficient=sufficient,
                    baseline_false_rate=of['query_rate'], false_rate=nf['query_rate'], recall_loss=ore-nr)
    new, old = reports["conjunctive"], reports["additive"]
    panel = 'g_set' if 'g_set' in new['composition'] else 'two_fact'
    if panel not in new['composition'] or 'binding' not in new['composition'][panel]:
        return dict(passed=None, reason='binding-aware composition unmeasured')
    gain = new['composition'][panel]['binding']-old['composition'][panel]['binding']
    ns, os = (r['end_to_end']['test']['cross_person_substitutions'] for r in (new, old))
    nr, ore = (r['end_to_end']['test']['recall'] for r in (new, old))
    retained = nr >= ore-.01-1e-12 and nr > 0
    clean = bool(new.get('damage_gate')) and bool(old.get('damage_gate')) and all(p['label'] == 'clean' for r in (new, old) for p in r.get('damage_gate', {}).values())
    return dict(rule='needs >=5 pts binding-aware composition or 2x fewer substitutions; retained recall >=baseline-1 pt, clean damage',
                passed=retained and clean and (gain >= .05-1e-12 or (os > 0 and ns <= os/2+1e-12)),
                retained_recall=retained, clean_damage=clean, composed_panel=panel, composed_recall_gain=gain,
                substitutions=ns, baseline_substitutions=os)


def addressing_arm(arm, branches, tok, facts, groups, anchor, seed, panels, base_nll, dev, a):
    layer = a.bio_layer
    train, x, general = B.address_training(branches.model, tok, facts, anchor, seed, layer, dev)
    labels, attrs = [facts[it["fi"]].pid for it in train], [facts[it["fi"]].attr for it in train]
    if arm == "INH":
        index = InhibitoryIndexer.fit(x, labels, attrs, general, a.inh_exclusions, a.inh_distractors,
                                     a.inh_lambda, a.bio_budget, seed, a.bio_steps)
        modes = ("inhibitory", "positive_only")
    else:
        index = ConjunctiveIndexer.fit(x, labels, attrs, general, a.bio_steps, a.bio_budget)
        modes = ("conjunctive", "additive")
    feature_start = time.perf_counter()
    xf = B.address_eval_features(branches.model, tok, groups, layer, dev)
    pf = {name: B.indexer_features(branches.model, wins.tolist(), layer, dev, [63]*len(wins))
          for name, wins in panels.items()}
    feature_seconds = time.perf_counter()-feature_start
    reports = {}
    for mode in modes:
        start = time.perf_counter()
        routes = {name: index.select(features, mode) for name, features in xf.items()}
        droutes = {name: index.select(features, mode) for name, features in pf.items()}
        selector_seconds = time.perf_counter()-start
        def benchmark_route(items, mode=mode):
            return [tuple(p for p in r if p in branches.states) for r in index.select(
                B.indexer_features(branches.model,[it['p'] for it in items],layer,dev),mode)]
        reports[mode] = route_report(branches, tok, facts, groups, routes, panels, droutes, base_nll, dev, a,benchmark_route)
        reports[mode]["selector_seconds"] = selector_seconds
        reports[mode]["selector_feature_seconds"] = feature_seconds
    negative_items = getattr(a, '_negative_items', [])
    nx = B.indexer_features(branches.model, [it['p'] for it in negative_items], layer, dev) if negative_items else None
    for mode in modes:
        rr = index.select(nx, mode) if nx is not None else []
        reports[mode]['false_activations']['independent_general'] = B.R.false_activation(rr)
    control = positive_control(arm)
    control['passed'] = control['passed'] and getattr(a,'_runtime_control_ok',True)
    decision = B.R.gated_verdict(control, kill_decision(arm, reports))
    print(f"{arm}-KILL " + B.json.dumps(decision), flush=True)
    return dict(subarms=reports, kill=decision, control=control, layer=layer, layer_indexing="full block inputs",
                values_fixed=True, training="TRAIN_T and anchor only", **index.metadata())


def mention_slots(tok, prompt, people, bos, max_slots=4):
    if not 1 <= max_slots <= 4:
        raise ValueError("mention table supports one to four slots")
    encoded = tok.encode(prompt, add_special_tokens=False)
    found = []
    for pid, name in people.items():
        for match in re.finditer(r"(?<!\w)"+re.escape(name)+r"(?!\w)", prompt):
            overlap = [i for i, (lo, hi) in enumerate(encoded.offsets)
                       if lo < match.end() and hi > match.start()]
            if not overlap:
                raise ValueError("tokenizer did not locate a name mention")
            found.append(dict(pid=pid, position=overlap[-1]+int(bos is not None)))
    return sorted(found, key=lambda slot: slot["position"], reverse=True)[:max_slots]


def mention_items(tok, facts, bos, split, max_slots=4):
    """Balanced same-gender recency/role cues; full-prompt token offsets locate mentions."""
    people = {f.pid: f for f in facts}
    items = []
    for fi, fact in enumerate(facts):
        for other in people.values():
            if other.pid == fact.pid or other.pos != fact.pos:
                continue
            for target_recent in (False, True):
                first, second = (other, fact) if target_recent else (fact, other)
                for cue in ("recency", "role"):
                    if split == "train":
                        context = f"{first.name} is the doctor. {second.name} is the painter. "
                        question = (f"Regarding the {'most recent' if target_recent else 'first'} person, what is {fact.pos} {fact.attr}?"
                                    if cue == "recency" else
                                    f"Regarding the {'painter' if target_recent else 'doctor'}, what is {fact.pos} {fact.attr}?")
                    elif split == 'val':
                        context = f"{first.name}, the doctor, arrived before {second.name}, the painter. "
                        question = (f"About the {'last' if target_recent else 'first'} visitor: what is {fact.pos} {fact.attr}?"
                                    if cue == 'recency' else
                                    f"About the {'painter' if target_recent else 'doctor'}: what is {fact.pos} {fact.attr}?")
                    else:
                        context = f"Today I met {first.name}, a doctor, then {second.name}, a painter. "
                        question = (f"For the person mentioned {'last' if target_recent else 'earlier'}, can you tell me {fact.pos} {fact.attr}?"
                                    if cue == "recency" else
                                    f"For the {'painter' if target_recent else 'doctor'}, can you tell me {fact.pos} {fact.attr}?")
                    prompt = context+question
                    slots = mention_slots(tok, prompt, {p.pid: p.name for p in (first, second)}, bos, max_slots)
                    q = E.encode(tok, prompt, bos)
                    items.append(dict(fi=fi, ti=len(items), attr=fact.attr, val=fact.val,
                        q=q, p=q+tok.encode("\nAnswer:", add_special_tokens=False).ids,
                        a=tok.encode(" "+fact.val, add_special_tokens=False).ids, slots=slots,
                        target=fact.pid, cue=cue, target_recent=target_recent))
    if not items:
        raise ValueError("SLOT needs at least two same-gender people")
    # Named, irrelevant conversations are supervised nulls, not hardcoded abstentions.
    for first, second in combinations(people.values(), 2):
        for order in ((first, second), (second, first)):
            wording = {'train': 'met', 'val': 'greeted', 'test': 'spoke with'}[split]
            prompt = f"I {wording} {order[0].name} and {order[1].name}. What is two plus two?"
            q = E.encode(tok, prompt, bos)
            slots = mention_slots(tok, prompt, {p.pid: p.name for p in order}, bos, max_slots)
            items.append(dict(fi=0, ti=len(items), attr='irrelevant', val='four', q=q,
                p=q+tok.encode("\nAnswer:", add_special_tokens=False).ids,
                a=tok.encode(' four', add_special_tokens=False).ids, slots=slots, target=None,
                cue='irrelevant', target_recent=False))
    return items


def slot_features(model, items, early, late, dev):
    if any(len(it["slots"]) < 2 for it in items):
        raise ValueError("SLOT comparisons need at least two candidates")
    late_x = B.indexer_features(model, [it["q"] for it in items], late, dev).float()
    slot_x = torch.stack([B.indexer_features(model, [it["q"]]*len(it["slots"]), early, dev,
                         [slot["position"] for slot in it["slots"]]).float() for it in items])
    return late_x, slot_x


class SlotResolver:
    """Low-rank learned projection; logistic candidate labels on TRAIN pronoun items."""
    def __init__(self, down, up, bias, log_scale, intercept):
        self.down, self.up, self.bias = down, up, bias
        self.log_scale, self.intercept = log_scale, intercept

    def scores(self, late, slots):
        query = F.normalize(late+F.linear(F.linear(late, self.down), self.up)+self.bias, dim=-1)
        return (torch.einsum("nd,nsd->ns", query, F.normalize(slots, dim=-1))*
                self.log_scale.clamp(-5, 5).exp()+self.intercept)

    @classmethod
    def fit(cls, late, slots, items, steps=100, l2=.001):
        if steps < 1 or len(late) != len(items) or slots.shape[:1] != late.shape[:1]:
            raise ValueError("invalid SLOT training data")
        d = late.shape[1]
        rank = min(8, d)
        down = (torch.randn(rank, d, generator=torch.Generator().manual_seed(0))/math.sqrt(d)).requires_grad_()
        up = torch.zeros(d, rank, requires_grad=True)
        bias = torch.zeros(d, requires_grad=True)
        log_scale = torch.zeros((), requires_grad=True)
        intercept = torch.zeros((), requires_grad=True)
        result = cls(down, up, bias, log_scale, intercept)
        target = torch.tensor([[slot["pid"] == it["target"] for slot in it["slots"]] for it in items]).float()
        opt = torch.optim.LBFGS([down, up, bias, log_scale, intercept], max_iter=steps,
                                history_size=10, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(result.scores(late, slots), target)
            loss = loss+l2*(down.square().mean()+up.square().mean()+bias.square().mean()+log_scale.square())
            loss.backward()
            return loss
        opt.step(closure)
        return cls(*(p.detach() for p in (down, up, bias, log_scale, intercept)))

    def select(self, late, slots, items):
        scores = self.scores(late, slots)
        chosen = scores.argmax(1).tolist()
        return [(it['slots'][j]['pid'],) if scores[i, j] > 0 else () for i, (it, j) in enumerate(zip(items, chosen))]


class DenseResolver:
    """Dense candidate logistic regression; same inputs, labels, BCE, L2 and bytes."""
    @staticmethod
    def features(late, slots):
        q = F.normalize(late,dim=-1)[:,None,:].expand_as(slots)
        c = F.normalize(slots,dim=-1)
        bands = [q*c.roll(shift,-1) for shift in range(2*min(8,late.shape[-1])-1)]
        return torch.cat((q,c,*bands,(q*c).sum(-1,keepdim=True)),dim=-1)

    @classmethod
    def fit(cls, late, slots, items, steps=100, l2=.001):
        result = cls()
        x = cls.features(late,slots)
        target = torch.tensor([[slot['pid']==it['target'] for slot in it['slots']] for it in items]).float()
        w = torch.zeros(x.shape[-1],requires_grad=True)
        b = torch.zeros((),requires_grad=True)
        opt = torch.optim.LBFGS([w,b],max_iter=steps,history_size=10,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(x@w+b,target)+l2*w.square().mean()
            loss.backward()
            return loss
        opt.step(closure)
        result.weight, result.bias = w.detach(), b.detach()
        d = late.shape[1]
        matching_floats = 2*min(8,d)*d+d+2
        assert matching_floats == w.numel()+1
        return result

    def scores(self, late, slots):
        return self.features(late,slots)@self.weight+self.bias

    select = SlotResolver.select


def slot_arm(branches, tok, facts, anchor, seed, panels, base_nll, dev, a):
    train = mention_items(tok, facts, E.BOS, 'train', a.slot_max)
    val = mention_items(tok, facts, E.BOS, 'val', a.slot_max)
    test = mention_items(tok, facts, E.BOS, 'test', a.slot_max)
    candidates = {'slots': [], 'dense': []}
    for late in [int(v) for v in a.slot_late_layers.split(',')]:
        tx, ts = slot_features(branches.model, train, a.slot_early_layer, late, dev)
        vx, vs = slot_features(branches.model, val, a.slot_early_layer, late, dev)
        expected = [() if it['target'] is None else (it['target'],) for it in val]
        for l2 in getattr(a, 'selector_l2', [.001, .01, .1]):
            for mode, cls in [('slots', SlotResolver), ('dense', DenseResolver)]:
                resolver = cls.fit(tx, ts, train, a.bio_steps, l2)
                score = B.selection_metrics(resolver.select(vx, vs, val), expected)['accuracy']
                candidates[mode].append((score, late, l2, resolver))
    reports, storage, selected = {}, {}, {}
    positives = [it for it in test if it['target'] is not None]
    negatives = [it for it in test if it['target'] is None]
    for mode, rows in candidates.items():
        score, late, l2, resolver = B.R.select_validation(rows, lambda c: c[0])
        selected[mode] = dict(layer=late, l2=l2, validation_accuracy=score, locked=True)
        x, slots = slot_features(branches.model, test, a.slot_early_layer, late, dev)
        all_routes = resolver.select(x, slots, test)
        routes = [r for it, r in zip(test, all_routes) if it['target'] is not None]
        nulls = [r for it, r in zip(test, all_routes) if it['target'] is None]
        def resolve_prompts(items):
            selected = []
            people = {f.pid:f.name for f in facts}
            for it in items:
                q = B.question_sequence(tok,it['p'],E.BOS)
                mentions = mention_slots(tok,tok.decode(q[1:]),people,E.BOS,a.slot_max)
                if not mentions:
                    selected.append(())
                    continue
                lx = B.indexer_features(branches.model,[q],late,dev).float()
                sx = B.indexer_features(branches.model,[q]*len(mentions),a.slot_early_layer,dev,
                                        [slot['position'] for slot in mentions]).float()[None]
                selected.extend(resolver.select(lx,sx,[dict(slots=mentions)]))
            return selected
        pr = {name: resolve_prompts([dict(p=win[:64].tolist()) for win in wins]) for name,wins in panels.items()}
        report = route_report(branches, tok, facts, {'pronoun': positives}, {'pronoun': routes},
                              panels, pr, base_nll, dev, a,resolve_prompts)
        report['named_irrelevant'] = B.R.false_activation(nulls)
        neg_base, neg_active = [], []
        for it, route in zip(negatives, nulls):
            branches.activate(())
            neg_base.extend(E.subject_nll(branches.model, [it], dev))
            branches.activate(route)
            neg_active.extend(E.subject_nll(branches.model, [it], dev))
        report['named_irrelevant_damage'] = dict(n=len(negatives), dnll=sum(p-b for p,b in zip(neg_active, neg_base))/max(len(negatives),1))
        for recent in (False, True):
            ii = [i for i, it in enumerate(positives) if it['target_recent'] == recent]
            report['address_correctness']['last' if recent else 'first'] = B.selection_metrics(
                [routes[i] for i in ii], [(positives[i]['target'],) for i in ii])
        reports[mode] = report
        storage[mode] = dict(resolver_bytes=sum(v.numel()*v.element_size() for v in vars(resolver).values() if torch.is_tensor(v)),
                             scratch_bytes_max=a.slot_max*(slots.shape[-1]*4+16))
    gain = reports['slots']['end_to_end']['pronoun']['recall']-reports['dense']['end_to_end']['pronoun']['recall']
    control = positive_control('SLOT')
    control['passed'] = control['passed'] and getattr(a,'_runtime_control_ok',True)
    clean = all(p['label']=='clean' for r in reports.values() for p in r['damage_gate'].values())
    decision = B.R.gated_verdict(control, dict(rule='>=10 pts over supervision/byte matched dense resolver; validation selection',
        passed=gain >= .10-1e-12 and clean, gain=gain, ambiguous_error_gate=None))
    branches.activate(())
    print('SLOT-KILL '+B.json.dumps(decision), flush=True)
    return dict(subarms=reports, kill=decision, control=control, storage=storage, selection=selected,
                n_train=len(train), n_test=len(test), early_layer=a.slot_early_layer,
                supervision='identical balanced first/last, role and named irrelevant TRAIN conversations',
                limits=['Mention identity supplied by name offsets.', 'Ambiguity criterion remains unmeasured.'])


def positive_control(arm):
    if arm == 'INH':
        pos = B.BankIndexer(torch.tensor([[1., 0.]], dtype=torch.double), torch.zeros(2), [0])
        keys = torch.tensor([[[0, 1]]], dtype=torch.int8)
        index = InhibitoryIndexer(pos, keys, torch.zeros_like(keys), 2., 1, [0])
        index.thresholds = dict(inhibitory=.1, positive_only=.1)
        confuser, own = torch.tensor([[.5, 1.]]), torch.tensor([[1., 0.]])
        passed = index.select(confuser) == [()] and index.select(own) == [(0,)] and index.select(confuser, 'positive_only') == [(0,)]
        return dict(passed=passed, kind='designed confuser inhibited; own-bank positive retained')
    if arm == 'CONJ':
        index = ConjunctiveIndexer()
        index.center, index.owners, index.budget = torch.zeros(2), (0,), 1
        index.entity = [(torch.tensor([8., 0.]), torch.tensor(-4.))]
        index.relation = [[(torch.tensor([0., 8.]), torch.tensor(-4.))]]
        index.thresholds = dict(conjunctive=.5)
        routes = index.select(torch.tensor([[1.,0.], [0.,1.], [1.,1.]]))
        return dict(passed=routes == [(), (), (0,)], kind='entity-only and relation-only rejected; conjunction accepted')
    slots = torch.eye(3)[:2][None].repeat(12,1,1)
    late = torch.stack([slots[i, i%2] if i<8 else torch.tensor([0.,0.,1.]) for i in range(12)])
    items = [dict(target=i%2 if i<8 else None, slots=[dict(pid=0),dict(pid=1)]) for i in range(12)]
    expected = [(i%2,) if i<8 else () for i in range(12)]
    model = SlotResolver.fit(late, slots, items, 100)
    return dict(passed=model.select(late, slots, items)==expected, kind='balanced first/last targets and named irrelevant nulls')
