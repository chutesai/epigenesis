"""C23 residual-pointer prototype; frozen base and supplied R6 per-person banks.

Full-block input indices are used for both hops. A low-rank additive hop-1
adapter is supervised at hop 2 through the intervening frozen blocks. Bank
routing is held fixed during generation; no generated answer enters routing.
"""
from contextlib import contextmanager
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import branch_eval as B, separability_probe as S, e2e_fsa as E
except ImportError:
    import branch_eval as B
    import separability_probe as S
    import e2e_fsa as E

TRAIN = ("Tell me {name}'s sister's {attr}?", "Which {attr} belongs to the sister of {name}?")
VAL = ("Please name the {attr} of {name}'s sister?",)
TEST = ("What is {name}'s sister's {attr}?",)


def pointer_items(tok, facts, relations, templates, bos):
    people = {f.pid: f.name for f in facts}
    if set(relations) != set(people) or set(relations.values()) != set(people):
        raise ValueError("prototype needs a permutation of all people")
    items = []
    for source, target in sorted(relations.items()):
        for fi, fact in enumerate(facts):
            if fact.pid != target:
                continue
            for ti, template in enumerate(templates):
                question = template.format(name=people[source], attr=fact.attr)
                q = E.encode(tok, question, bos)
                items.append(dict(fi=fi, ti=ti, source=source, target=target, attr=fact.attr,
                                  val=fact.val, question=question, q=q,
                                  p=q + tok.encode("\nAnswer:", add_special_tokens=False).ids,
                                  position=len(q)-1,
                                  a=tok.encode(" " + fact.val, add_special_tokens=False).ids))
    return items


class PointerPatch(nn.Module):
    def __init__(self, mean, scale, rank=8):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float().clamp_min(1e-4))
        self.down = nn.Linear(mean.numel(), rank, bias=False, device=mean.device)
        self.up = nn.Linear(rank, mean.numel(), bias=True, device=mean.device)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = (x.float() - self.mean) / self.scale
        return x + (self.up(self.down(z)) * self.scale).to(x.dtype)


@contextmanager
def pointer_hook(model, layer, patch, positions):
    """Only the original question position changes, including on GEN re-forwards."""
    def hook(module, inputs, kwargs):
        x = inputs[0] if inputs else kwargs.get("x", kwargs.get("hidden_states"))
        if x.ndim != 3 or len(x) != len(positions):
            raise ValueError("pointer positions must align with the batch")
        pos = torch.as_tensor(positions, device=x.device)
        if (pos < 0).any() or (pos >= x.shape[1]).any():
            raise ValueError("pointer position outside sequence")
        rows = torch.arange(len(x), device=x.device)
        out = x.clone()
        out[rows, pos] = patch(x[rows, pos])
        if inputs:
            return (out,) + inputs[1:], kwargs
        kwargs = dict(kwargs)
        kwargs["x" if "x" in kwargs else "hidden_states"] = out
        return inputs, kwargs
    handle = model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def code_forward(model, tokens, layer, positions):
    """Differentiable capture, stop before later facts banks and the LM head."""
    class Captured(Exception):
        pass
    result = []
    def capture(module, inputs, kwargs):
        x = inputs[0] if inputs else kwargs.get("x", kwargs.get("hidden_states"))
        result.append(x[torch.arange(len(x), device=x.device),
                        torch.as_tensor(positions, device=x.device)])
        raise Captured()
    handle = model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        try:
            getattr(model, "forward_hidden", model)(tokens)
        except Captured:
            pass
        if len(result) != 1:
            raise RuntimeError("hop-2 layer did not run")
        return result[0]
    finally:
        handle.remove()


def train_pointer(model, items, codes, early, late, dev, steps, rank, lr, seed, bs=4):
    if not 0 <= early < late < len(model.layers):
        raise ValueError("hop-1 must precede hop-2 within full blocks")
    initial = B.indexer_features(model, [it["q"] for it in items], early, dev).float().to(dev)
    scale = initial.std(0, unbiased=False)
    scale = scale.clamp_min(0.1 * scale.square().mean().sqrt().clamp_min(1e-4))
    patch = PointerPatch(initial.mean(0), scale, rank).to(dev)
    opt = torch.optim.Adam(patch.parameters(), lr=lr)
    rng = random.Random(seed)
    losses = []
    enabled = E.FSAParam.enabled
    try:
        E.set_enabled(False)
        for step in range(steps):
            batch = rng.sample(items, min(bs, len(items)))
            positions = [it["position"] for it in batch]
            x, _ = E.batchify([it["q"] for it in batch], dev)
            target = torch.stack([codes[it["target"]] for it in batch]).float().to(dev)
            opt.zero_grad()
            with pointer_hook(model, early, patch, positions):
                output = code_forward(model, x, late, positions).float()
                loss = (output - target).square().mean() / target.square().mean().clamp_min(1e-6)
                loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            if step == 0 or (step + 1) % 20 == 0:
                print(f"POINTER-TRAIN step={step+1} loss={losses[-1]:.6f}", flush=True)
    finally:
        E.set_enabled(enabled)
    return patch.eval(), losses


@torch.no_grad()
def evaluate(model, branches, patch, indexer, codes, tok, facts, items, early, late, dev, max_new, stop_k):
    try:
        reports = {}
        for mode in ("no_hop1", "no_pointer_oracle_bank", "resolved_name_oracle", "pointer", "oracle_bank"):
            gens, routes, cosines = [], [], []
            for i, original in enumerate(items):
                it = resolved_name_item(tok, facts, original, E.BOS) if mode == 'resolved_name_oracle' else original
                branches.activate(())
                x = torch.tensor([it["q"]], device=dev)
                @contextmanager
                def intervention():
                    if mode in ("no_hop1", "no_pointer_oracle_bank", "resolved_name_oracle"):
                        yield
                    else:
                        with pointer_hook(model, early, patch, [it["position"]]):
                            yield
                with intervention():
                    code = code_forward(model, x, late, [it["position"]]).float().cpu()
                    route = (it["target"],) if mode in ("oracle_bank", "no_pointer_oracle_bank", "resolved_name_oracle") else indexer.select(code)[0]
                    cosines.append(float(F.cosine_similarity(code.double() - indexer.center,
                                         codes[it["target"]].double()[None] - indexer.center)))
                    branches.activate(route)
                    mx = max_new or len(it["a"]) + stop_k + 2
                    gen = E.greedy_generate(model, [it["p"]], dev, mx, bs=1)[0]
                gens.append(gen)
                routes.append(route)
                if i == 0 or (i+1) % 50 == 0:
                    print(f"POINTER-EVAL {mode} {i+1}/{len(items)}", flush=True)
            reports[mode] = dict(hop1_code_cosine=sum(cosines)/max(len(cosines),1),
                selection=B.selection_metrics(routes, [(it["target"],) for it in items]),
                gen=B.generation_summary(tok, items, gens, E.conflict_map(facts), stop_k))
        return reports
    finally:
        branches.activate(())


def cleanup_candidates(code, codes, theta=.7, margin=.1, center=None):
    """Return a confident canonical ID, or a bounded two-item candidate beam."""
    if not math.isfinite(theta) or not -1 <= theta <= 1 or not math.isfinite(margin) or not 0 <= margin <= 2:
        raise ValueError("invalid cleanup cosine threshold/margin")
    owners = sorted(codes)
    if len(owners) < 2:
        raise ValueError("cleanup needs at least two canonical entities")
    matrix = torch.stack([codes[p].double().cpu() for p in owners])
    code = torch.as_tensor(code).double().cpu().reshape(1, -1)
    offset = 0 if center is None else torch.as_tensor(center).double().cpu()
    scores = F.cosine_similarity(code-offset, matrix-offset)
    if not torch.isfinite(scores).all():
        raise ValueError("invalid pointer code")
    order = scores.argsort(descending=True, stable=True)
    first, second = order[:2].tolist()
    confident = float(scores[first]) >= theta and float(scores[first]-scores[second]) >= margin
    return dict(canonical=owners[first] if confident else None,
                candidates=(owners[first], owners[second]), max_cosine=float(scores[first]),
                margin=float(scores[first]-scores[second]))


def cleanup_kill(reports):
    raw = reports["raw_cascade"]["end_to_end"]["recall"]
    oracle = reports["oracle_pointer"]["end_to_end"]["recall"]
    gap = oracle-raw
    gains = {mode: reports[mode]["end_to_end"]["recall"]-raw for mode in ("cleanup", "cleanup_beam")}
    beam_gain = reports['cleanup_beam']['end_to_end']['recall']-reports.get('raw_beam', reports['raw_cascade'])['end_to_end']['recall']
    return dict(rule="needs >=10 pts held-out two-hop gain or half the gap to oracle",
                passed=any((g >= .10-1e-12 or (gap > 0 and g >= gap/2-1e-12)) and (mode != "cleanup_beam" or beam_gain >= .10-1e-12) for mode, g in gains.items()),
                matched_two_read_gain=beam_gain,
                gains=gains, oracle_gap=gap, value_use_bottleneck=gap <= 0)


@torch.no_grad()
def evaluate_cleanup(model, branches, patch, indexer, codes, tok, facts, items, early, late, dev,
                     max_new, stop_k, theta=.7, margin=.1):
    """Cleanup replaces the hop-2 residual at the original question position.

    Uncertain cleanup abstains; cleanup_beam reads both canonical candidates and
    compares mean token log-probabilities of their OWN generated continuations.
    Neither routing nor candidate comparison receives a gold answer.
    """
    reports = {}
    conflicts = E.conflict_map(facts)
    try:
        for mode in ("raw_cascade", "raw_beam", "cleanup", "cleanup_beam", "oracle_pointer"):
            routes, gens, decisions, lps, reads = [], [], [], [], []
            start = time.perf_counter()
            selector_seconds = 0.
            for it in items:
                branches.activate(())
                with pointer_hook(model, early, patch, [it["position"]]):
                    code = code_forward(model, torch.tensor([it["q"]], device=dev), late, [it["position"]])
                    begin = time.perf_counter()
                    decision = cleanup_candidates(code, codes, theta, margin, indexer.center)
                    if mode == "raw_cascade":
                        candidates = indexer.select(code)[0]
                    elif mode == "raw_beam":
                        candidates = decision["candidates"]
                    elif mode == "oracle_pointer":
                        candidates = (it["target"],)
                    elif decision["canonical"] is not None:
                        candidates = (decision["canonical"],)
                    else:
                        candidates = decision["candidates"] if mode == "cleanup_beam" else ()
                    selector_seconds += time.perf_counter()-begin
                    decisions.append(decision)
                    generated = []
                    for pid in candidates:
                        branches.activate((pid,))
                        @contextmanager
                        def snapped():
                            if mode in ("raw_cascade", "raw_beam"):
                                yield
                            else:
                                canonical = codes[pid].to(dev)
                                with pointer_hook(model, late, lambda x: canonical.to(x.dtype).expand_as(x), [it["position"]]):
                                    yield
                        with snapped():
                            # Length budget is common to all candidates for this query.
                            mx = max_new or len(it["a"])+stop_k+2
                            gen = E.greedy_generate(model, [it["p"]], dev, mx, bs=1)[0]
                            lp = (E.tf_eval(model, [dict(p=it["p"], a=gen)], dev, bs=1)["answer_logprob"]
                                  if len(candidates) > 1 and gen else -math.inf)
                            generated.append((pid, gen, lp))
                    reads.append(len(generated))
                    if generated:
                        # Stable first-candidate tie break; scores never use it['a'].
                        pid, gen, lp = max(generated, key=lambda row: row[2])
                        routes.append((pid,))
                        gens.append(gen)
                        lps.append(lp if math.isfinite(lp) else None)
                    else:
                        # An abstention emits no answer and executes no bank.
                        routes.append(())
                        gens.append([])
                        lps.append(None)
            selection = B.selection_metrics(routes, [(it["target"],) for it in items])
            hits = [E.score_generation(tok, g, it["val"], stop_k, conflicts.get(it["fi"], ()))['clean']
                    for it, g in zip(items, gens)]
            correct = [r == (it["target"],) for r, it in zip(routes, items)]
            n_correct = sum(correct)
            end = dict(n=len(items), recall=sum(hits)/max(len(items), 1))
            reports[mode] = dict(address_correctness=selection,
                conditional_value_correctness=dict(n_address_correct=n_correct,
                    accuracy=sum(h and c for h, c in zip(hits, correct))/n_correct if n_correct else None),
                composition=dict(held_out_two_hop=end), end_to_end=end,
                selection=selection, gen=B.generation_summary(tok, items, gens, conflicts, stop_k),
                cleanup=decisions, selected_generated_answer_logprob=lps,
                bank_reads=sum(reads), max_bank_reads=max(reads, default=0),
                active_bank_bytes_max=max((sum(branches.bytes[p] for p in r) for r in routes), default=0)
                                      if hasattr(branches, "bytes") else None,
                evaluation_seconds=time.perf_counter()-start, selector_seconds=selector_seconds)
    finally:
        branches.activate(())
    decision = cleanup_kill(reports)
    # Only main emits a verdict, after runtime controls and damage checks.
    return dict(subarms=reports, kill=decision, theta=theta, margin=margin,
                canonical_bytes=sum(c.numel()*c.element_size() for c in codes.values()),
                beam_scoring="mean log-probability of candidate's generated continuation; no gold answer",
                cleanup_space="anchor-centered cosine; full canonical residual replacement at hop-2",
                damage=None, damage_limitation="See top-level matched pointer damage and revocation checks")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positive-control", action="store_true")
    p.add_argument("--cleanup-thetas", type=float, nargs="+", default=[.5,.7,.9])
    p.add_argument("--cleanup-margins", type=float, nargs="+", default=[.05,.1,.2])
    p.add_argument("--oracle-gap-recovery", type=float, default=.5)
    p.add_argument("--damage-n", type=int, default=40)
    p.add_argument("--patches", nargs="+", type=Path)
    p.add_argument("--anchor", type=Path)
    p.add_argument("--export", type=Path, default=E.EXPORT)
    p.add_argument("--export-step", type=int, default=E.EXPORT_STEP)
    p.add_argument("--hop1-layer", type=int, default=8)
    p.add_argument("--hop2-layer", type=int, default=12)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--train-bs", type=int, default=4)
    p.add_argument("--gen-max-new", type=int, default=0)
    p.add_argument("--gen-stop-k", type=int, default=2)
    p.add_argument("--cleanup-theta", type=float, default=.7)
    p.add_argument("--cleanup-margin", type=float, default=.1)
    p.add_argument("--c29", action="store_true", help="also evaluate indexed composition and hypothesis preservation")
    p.add_argument("--c29-max-per-kind", type=int, default=24)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if (not 0 <= a.hop1_layer < a.hop2_layer or min(a.steps, a.rank, a.train_bs) < 1 or a.c29_max_per_kind < 2
            or not 0 < a.lr < float("inf") or min(a.export_step, a.gen_max_new, a.gen_stop_k) < 0
            or not math.isfinite(a.cleanup_theta) or not -1 <= a.cleanup_theta <= 1
            or not math.isfinite(a.cleanup_margin) or not 0 <= a.cleanup_margin <= 2):
        p.error("invalid layers or training/generation parameters")
    if not a.positive_control and any(getattr(a,k) is None for k in ('patches','anchor','out')):
        p.error('evaluation requires --patches --anchor --out')
    if not 0 < a.oracle_gap_recovery <= 1 or a.damage_n < 1:
        p.error('invalid recovery or damage count')
    for theta in a.cleanup_thetas:
        if not math.isfinite(theta) or not -1 <= theta <= 1:
            p.error('invalid theta search')
    for margin in a.cleanup_margins:
        if not math.isfinite(margin) or not 0 <= margin <= 2:
            p.error('invalid margin search')
    return a


def validate_bank_layers(model, meta, late):
    # Saved patch indices enumerate MoEs; hop indices enumerate full blocks.
    for li in meta["layers"]:
        moe = E.moe_layers(model)[li][1]
        owners = [i for i, block in enumerate(model.layers) if any(m is moe for m in block.modules())]
        if len(owners) != 1 or owners[0] < late:
            raise ValueError("facts banks must be at or after hop-2 full block input")


def resolved_name_item(tok, facts, item, bos):
    fact = facts[item['fi']]
    question = f"What is {fact.name}'s {fact.attr}?"
    q = E.encode(tok,question,bos)
    return dict(item, q=q, p=q+tok.encode('\nAnswer:',add_special_tokens=False).ids, position=len(q)-1)


def pointer_split(items):
    attrs = sorted({it['attr'] for it in items})
    held = {attrs[-1]} if len(attrs)>1 else set()
    available = [a for a in attrs if a not in held]
    combos = {(source, available[i%len(available)]) for i, source in enumerate(sorted({it['source'] for it in items}))} if len(available)>1 else set()
    return {'train': [it for it in items if it['attr'] not in held and (it['source'],it['attr']) not in combos],
            'held_out_attribute': [it for it in items if it['attr'] in held],
            'held_out_combination': [it for it in items if it['attr'] not in held and (it['source'],it['attr']) in combos],
            'wording': [it for it in items if it['attr'] not in held and (it['source'],it['attr']) not in combos]}


def positive_control():
    codes = {0:torch.tensor([1.,0.]),1:torch.tensor([0.,1.])}
    near = cleanup_candidates([.98,.02],codes)
    ambiguous = cleanup_candidates([1.,1.],codes)
    return dict(passed=near['canonical']==0 and ambiguous['canonical'] is None,
                kind='perturbed code snaps; ambiguous midpoint abstains; runtime resolved-name oracle also required')


@torch.no_grad()
def preservation(model, branches, patch, indexer, early, late, wins, dev, codes=None, mode="raw_cascade", theta=.7, margin=.1):
    before, after = [], []
    branches.activate(())
    E.set_enabled(False)
    revoke_input = wins[:1,:16]
    base_logits = E.logits_of(model,revoke_input).clone()
    before = E.nll_windows(model,wins,dev,bs=1)
    try:
        for win in wins:
            position = min(63,len(win)-1)
            branches.activate(())
            with pointer_hook(model,early,patch,[position]):
                code = code_forward(model,win[:position+1][None],late,[position])
                decision = cleanup_candidates(code,codes,theta,margin,indexer.center) if codes is not None else None
                if mode == 'raw_cascade':
                    candidates = indexer.select(code)[0]
                elif mode == 'raw_beam':
                    candidates = decision['candidates']
                else:
                    candidates = (decision['canonical'],) if decision['canonical'] is not None else (decision['candidates'] if mode == 'cleanup_beam' else ())
                @contextmanager
                def snap(pid):
                    if mode in ('raw_cascade','raw_beam') or pid is None:
                        yield
                    else:
                        canonical=codes[pid].to(dev)
                        with pointer_hook(model,late,lambda x:canonical.to(x.dtype).expand_as(x),[position]):
                            yield
                if len(candidates)>1:
                    scored=[]
                    for pid in candidates:
                        branches.activate((pid,))
                        with snap(pid):
                            gen=E.greedy_generate(model,[win[:position+1].tolist()],dev,24,bs=1)[0]
                            lp=E.tf_eval(model,[dict(p=win[:position+1].tolist(),a=gen)],dev,bs=1)['answer_logprob'] if gen else -math.inf
                            scored.append((lp,pid))
                    chosen=(max(scored,key=lambda row:row[0])[1],)
                else:
                    chosen=candidates
                branches.activate(chosen)
                with snap(chosen[0] if chosen else None):
                    after.extend(E.nll_windows(model,win[None],dev,bs=1))
    finally:
        branches.activate(())
        E.set_enabled(False)
    delta = torch.tensor(after)-torch.tensor(before)
    exact = torch.equal(base_logits,E.logits_of(model,revoke_input))
    if not exact:
        raise RuntimeError('pointer revocation failed')
    return dict(n=len(wins),dnll=float(delta.mean()),dnll_se=float(delta.std(unbiased=False)/math.sqrt(len(wins))),
                batch_size=1,policy=mode+' on general query prefix; candidate own-generation likelihood',revoked_logits_bit_exact=exact)

def main(argv=None):
    a = parse_args(argv)
    if a.positive_control:
        control = positive_control()
        payload = dict(control=control,verdict='control-passed' if control['passed'] else 'control-failed')
        print('POSITIVE-CONTROL '+json.dumps(payload),flush=True)
        if a.out:
            a.out.write_text(json.dumps(payload,indent=2)+'\n')
        return payload
    states = [torch.load(p, map_location="cpu", weights_only=False) for p in a.patches]
    banks = B.validate_patches(states)
    meta = states[0]["meta"]
    if meta["people"] != 3:
        raise SystemExit("two-hop prototype requires the three-person world")
    for state in states:
        E.validate_patch_export(state["meta"], a.export, a.export_step)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA and the /workspace/malleable model/data are required")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    torch.manual_seed(meta["seed"])
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    facts = E.make_world(meta["seed"], 3, meta["facts_per_person"], meta["shared_attrs"], meta["values"])
    relations = {0: 1, 1: 2, 2: 0}
    train_all = pointer_items(tok, facts, relations, TRAIN, E.BOS)
    train = pointer_split(train_all)['train']
    val = pointer_split(pointer_items(tok,facts,relations,VAL,E.BOS))['wording']
    if not train or not val:
        raise SystemExit('need enough attributes/combinations for disjoint pointer training and validation')
    test = pointer_items(tok, facts, relations, TEST, E.BOS)
    model, _, dev = build_model(a.export, device="cuda:0", model_config=None, step=a.export_step,
                                expert_format="bf16", eda_kernel_backend="naive_fp32")
    model.eval().requires_grad_(False)
    model._fsa_export, model._fsa_export_step = a.export, a.export_step
    validate_bank_layers(model, meta, a.hop2_layer)
    E.attach(model, meta["layers"], "fsa_forced", 0, projs=meta["projs"], sparse=True,
             slot_frac=meta["slot_frac"], seed=meta["seed"])
    E.apply_projection_scales(model, meta.get("patch_scale_mult", 1.),
                              meta.get("patch_scale_mult_up"), meta.get("patch_scale_mult_down"))
    branches = B.Branches(model, banks, {s["meta"]["train_person"]: p.stat().st_size
                                        for s, p in zip(states, a.patches)}, "sum_clip")
    direct = S.build_panels(tok, facts, E.BOS, question_train=False)["train"]
    features = B.indexer_features(model, [it["p"] for it in direct], a.hop2_layer, dev)
    owners = torch.tensor([facts[it["fi"]].pid for it in direct])
    codes = {pid: features[owners == pid].float().mean(0) for pid in banks}
    calibration_tokens = json.loads(a.anchor.read_text())[a.damage_n*511:]
    wins, pos, _ = S.general_windows(calibration_tokens, meta["seed"], E.BOS)
    general = B.indexer_features(model, wins, a.hop2_layer, dev, pos)
    indexer = B.BankIndexer.fit(features, owners, general)
    # Known target retrieval must work before fitting the cascade.
    empty_pointer = PointerPatch(torch.zeros(features.shape[1],device=dev),torch.ones(features.shape[1],device=dev),a.rank).to(dev)
    preflight = evaluate(model,branches,empty_pointer,indexer,codes,tok,facts,val[:3],
                         a.hop1_layer,a.hop2_layer,dev,a.gen_max_new,a.gen_stop_k)
    control = positive_control()
    control['resolved_name_oracle'] = preflight['resolved_name_oracle']['gen']['recall']
    control['passed'] = control['passed'] and control['resolved_name_oracle']==1.
    if not control['passed']:
        result=dict(control=control,verdict='control-failed',training_suppressed=True,preflight=preflight)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print('control-failed: resolved-name oracle failed before cascade training',flush=True)
        return result
    patch, losses = train_pointer(model, train, codes, a.hop1_layer, a.hop2_layer, dev,
                                  a.steps, a.rank, a.lr, meta["seed"], a.train_bs)
    reports = evaluate(model, branches, patch, indexer, codes, tok, facts, test,
                        a.hop1_layer, a.hop2_layer, dev, a.gen_max_new, a.gen_stop_k)
    validation = []
    for theta in dict.fromkeys(a.cleanup_thetas+[a.cleanup_theta]):
        for margin in dict.fromkeys(a.cleanup_margins+[a.cleanup_margin]):
            report = evaluate_cleanup(model,branches,patch,indexer,codes,tok,facts,val,a.hop1_layer,a.hop2_layer,
                                      dev,a.gen_max_new,a.gen_stop_k,theta,margin)
            score = report['subarms']['cleanup']['end_to_end']['recall']
            validation.append((score,theta,margin))
    score, theta, margin = B.R.select_validation(validation,lambda row:row[0])
    cleanup = evaluate_cleanup(model, branches, patch, indexer, codes, tok, facts, test,
                               a.hop1_layer, a.hop2_layer, dev, a.gen_max_new, a.gen_stop_k,theta,margin)
    cleanup['selection'] = dict(validation_recall=score,locked=True,theta=theta,margin=margin)
    heldout_reports = {}
    for name, subset in pointer_split(test).items():
        if name=='train' or not subset:
            continue
        heldout_reports[name] = evaluate(model,branches,patch,indexer,codes,tok,facts,subset,a.hop1_layer,a.hop2_layer,
                                         dev,a.gen_max_new,a.gen_stop_k)
    anchor_tokens = json.loads(a.anchor.read_text())
    needed = a.damage_n*511
    if len(anchor_tokens) < needed:
        raise SystemExit('anchor needs enough tokens for independent pointer damage windows')
    wins = torch.tensor([[E.BOS]+anchor_tokens[i*511:(i+1)*511] for i in range(a.damage_n)],device=dev)
    damage = preservation(model,branches,patch,indexer,a.hop1_layer,a.hop2_layer,wins,dev)
    damage_by_mode = {'raw_cascade':damage}
    for mode in ('raw_beam','cleanup','cleanup_beam'):
        damage_by_mode[mode] = preservation(model,branches,patch,indexer,a.hop1_layer,a.hop2_layer,wins,dev,codes,mode,theta,margin)
    cleanup['damage'] = damage_by_mode
    cleanup['kill']['passed'] = cleanup['kill']['passed'] and all(p['dnll']<=.003 for p in damage_by_mode.values())
    raw = reports['no_hop1']['gen']['recall']
    oracle = reports['resolved_name_oracle']['gen']['recall']
    learned = reports['pointer']['gen']['recall']
    gap = oracle-raw
    decision = dict(passed=gap>0 and learned-raw >= a.oracle_gap_recovery*gap-1e-12 and damage['dnll']<=.003,
                    oracle_success=oracle,oracle_gap=gap,required_gap_recovery=a.oracle_gap_recovery, gain=learned-raw)
    decision = B.R.gated_verdict(control,decision)
    control['oracle_code_retains_value'] = cleanup['subarms']['oracle_pointer']['end_to_end']['recall']
    control['passed'] = control['passed'] and control['oracle_code_retains_value']==1.
    decision = B.R.gated_verdict(control,decision)
    cleanup['kill'] = B.R.gated_verdict(control,cleanup['kill'])
    print('POINTER-CLEANUP-KILL '+json.dumps(cleanup['kill']),flush=True)
    print('POINTER-KILL '+json.dumps(decision),flush=True)
    names = {f.pid: f.name for f in facts}
    result = dict(world=meta, export=str(a.export), export_step=a.export_step,
        relations=[f"{names[p]}'s sister is {names[q]}" for p, q in relations.items()],
        hop1_layer=a.hop1_layer, hop2_layer=a.hop2_layer, layer_indexing="zero-based full block inputs",
        train_templates=TRAIN, test_templates=TEST, n_train=len(train), n_test=len(test),
        steps=a.steps, rank=a.rank, lr=a.lr, train_losses=losses, reports=reports, cleanup=cleanup,
        held_out_reports=heldout_reports, val_templates=VAL, n_val=len(val),control=control,kill=decision,damage=damage,
        adapter_bytes=sum(v.numel()*v.element_size() for v in patch.state_dict().values()),
        code_bytes=sum(v.numel()*v.element_size() for v in codes.values()),
        implementation_bytes=Path(__file__).stat().st_size,
        resident_bank_bytes=sum(branches.bytes.values()),
        selector_bytes=indexer.keys.numel()*indexer.keys.element_size()+indexer.center.numel()*indexer.center.element_size()+8*len(indexer.owners),
        limitations=["Supervised cyclic fictional relation; separate wording, held-out attribute and held-out source/attribute reports.",
            "Dense low-rank residual adapter, not a sparse FSA relation patch; no checkpoint is written.",
            "Entity codes are means of direct B-mention training prompts at their final pre-answer token.",
            "Mean-code regression can erase attribute information; routing success need not imply GEN success.",
            "All sparse banks reside in host memory; selected bank is loaded for generation.",
            "Oracle-bank, no-pointer/oracle-bank and explicit resolved-name controls reported separately.",
            "No cache support; damage measured with active pointer and learned bank routing."])
    if a.c29:
        if __package__:
            from . import indexed_composition as C29
        else:
            import indexed_composition as C29
        result["c29"] = C29.run(branches, tok, facts, relations, json.loads(a.anchor.read_text()),
            meta["seed"], a.hop1_layer, a.hop2_layer, dev, a.steps, a.gen_max_new, a.c29_max_per_kind)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("TWO-HOP " + json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
