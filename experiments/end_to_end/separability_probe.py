"""Frozen residual-stream linear separability of the 3-person x 100-fact world.

python experiments/end_to_end/separability_probe.py --seed 0 --out separability.json

Layers are zero-based indices into model.layers, not indices into moe_layers.
For question templates capture the final '?' token, before any answer cue;
for declarative cloze templates capture the last pre-answer token. No target
answer is forwarded. Training phrasings alone fit all probes, including preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__:
    from . import e2e_fsa as E
else:
    import e2e_fsa as E

LAYERS = tuple(range(2, 31, 2))
BATCH_SIZE = 8
NONE_WINDOWS = 64
WINDOW_LENGTH = 96
TRAIN_Q = ("Please tell me: what is {name}'s {attr}?",
           "Regarding {name}, what is {pos} {attr}?")
PRONOUN_T = (
    "{name} is my neighbour. What is {pos} {attr}?",
    "{name} is my neighbour. Can you tell me {pos} {attr}?",
)
INDIRECT_T = "The person whose {ref_attr} is {ref_val}: what is {pos} {attr}?"


def build_panels(tok, facts, bos, question_train=True):
    """All training/held-out phrasings, plus deterministic disjoint variants."""
    panels = {}
    for name, templates in (("train", E.TRAIN_T), ("val", E.VAL_T), ("test", E.TEST_T)):
        panels[name] = E.build_items(tok, facts, templates, bos)
        for item in panels[name]:
            fact = facts[item["fi"]]
            full = E.fmt(templates[item["ti"]], fact)
            prompt = full[:full.rindex(fact.val)].rstrip()
            if "?" in prompt:
                prompt = prompt[:prompt.rindex("?") + 1]
            item.update(prompt=prompt, p=E.encode(tok, prompt, bos))
    if question_train:
        for fi, fact in enumerate(facts):
            for ti, template in enumerate(TRAIN_Q, len(E.TRAIN_T)):
                prompt = E.fmt(template, fact)
                panels["train"].append(dict(fi=fi, ti=ti, prompt=prompt,
                                            p=E.encode(tok, prompt, bos)))
    existing = {it["prompt"] for group in panels.values() for it in group}
    panels["pronoun"], panels["indirect"] = [], []
    value_counts = Counter(f.val for f in facts)
    for fi, fact in enumerate(facts):
        for ti, template in enumerate(PRONOUN_T):
            prompt = E.fmt(template, fact)
            if prompt in existing:
                raise ValueError("pronoun template overlaps a harness prompt")
            panels["pronoun"].append(dict(fi=fi, ti=ti, prompt=prompt,
                                           p=E.encode(tok, prompt, bos)))
        references = sorted(
            (i for i, ref in enumerate(facts)
             if ref.pid == fact.pid and ref.attr != fact.attr
             and ref.val != fact.val and value_counts[ref.val] == 1),
            key=lambda i: (facts[i].attr, i),
        )
        if references:
            ri = references[0]
            ref = facts[ri]
            prompt = INDIRECT_T.format(ref_attr=ref.attr, ref_val=ref.val,
                                       pos=fact.pos, attr=fact.attr)
            if prompt in existing:
                raise ValueError("indirect template overlaps a harness prompt")
            panels["indirect"].append(dict(fi=fi, reference_fi=ri, prompt=prompt,
                                            p=E.encode(tok, prompt, bos)))
    return panels


def general_windows(tokens, seed, bos=E.BOS):
    """64 disjoint windows, one seeded random non-BOS position in each."""
    needed = NONE_WINDOWS * (WINDOW_LENGTH - 1)
    if len(tokens) < needed:
        raise ValueError(f"general text needs at least {needed} tokens")
    rng = random.Random(seed + 180018)
    starts = rng.sample(range(len(tokens) // (WINDOW_LENGTH - 1)), NONE_WINDOWS)
    sequences = [[bos] + tokens[s * (WINDOW_LENGTH - 1):(s + 1) * (WINDOW_LENGTH - 1)]
                 for s in starts]
    positions = [rng.randrange(1, WINDOW_LENGTH) for _ in starts]
    return sequences, positions, starts


@torch.no_grad()
def collect_residuals(model, sequences, layer_ids, device, positions=None, progress=None):
    """Pre-hooks on full layer modules; right padding is never a target.

    Truncation at the requested position keeps general-text suffixes out of
    recurrent state. The model's forward_hidden avoids allocating LM logits.
    Hooks are removed on failure and fail closed on missing/repeated layers.
    """
    if not sequences or any(not seq for seq in sequences):
        raise ValueError("need nonempty sequences")
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("need distinct layer indices")
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("model must expose its full layer modules as model.layers")
    if min(layer_ids) < 0 or max(layer_ids) >= len(layers):
        raise ValueError("layer index outside model.layers")
    positions = [len(s) - 1 for s in sequences] if positions is None else list(positions)
    if len(positions) != len(sequences) or any(
        not 0 <= p < len(s) for s, p in zip(sequences, positions)
    ):
        raise ValueError("invalid capture positions")
    result, records, hooks = {li: [] for li in layer_ids}, {}, []
    targets = None

    def capture(module, inputs, kwargs, li):
        if li in records:
            raise RuntimeError(f"layer {li} ran more than once in a forward")
        x = inputs[0] if inputs else kwargs.get("x", kwargs.get("hidden_states"))
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[:2] != batch_tokens.shape:
            raise RuntimeError("expected batch-by-sequence-by-hidden residual input")
        rows = torch.arange(len(targets), device=x.device)
        records[li] = x[rows, targets.to(x.device)].detach().cpu()

    try:
        for li in layer_ids:
            hooks.append(layers[li].register_forward_pre_hook(
                lambda mod, inp, kw, li=li: capture(mod, inp, kw, li), with_kwargs=True))
        forward = getattr(model, "forward_hidden", model)
        for start in E.progress(range(0, len(sequences), BATCH_SIZE), 'residual_features'):
            seqs = [s[:p + 1] for s, p in zip(
                sequences[start:start + BATCH_SIZE], positions[start:start + BATCH_SIZE])]
            targets = torch.tensor([len(s) - 1 for s in seqs], dtype=torch.long)
            batch_tokens, _ = E.batchify(seqs, device)
            records.clear()
            forward(batch_tokens)
            for li in layer_ids:
                if li not in records:
                    raise RuntimeError(f"missing residual capture at layer {li}")
                result[li].append(records[li])
            if progress and (start == 0 or (start // BATCH_SIZE + 1) % 20 == 0
                             or start + BATCH_SIZE >= len(sequences)):
                print(f"{progress}: {min(start + BATCH_SIZE, len(sequences))}/{len(sequences)}",
                      flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return {li: torch.cat(rows) for li, rows in result.items()}


@dataclass
class RidgeProbe:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    centroids: torch.Tensor

    @torch.no_grad()
    def scores(self, x):
        x = torch.as_tensor(x, device=self.weight.device, dtype=self.weight.dtype)
        return ((x - self.mean) / self.scale) @ self.weight + self.bias


    @torch.no_grad()
    def none_scores(self, x):
        """Negative cosine to training class centroids; larger means less familiar.

        Raw regression maxima are unbounded outside the training support.
        This is a novelty diagnostic, not a calibrated abstention probability.
        """
        x = torch.as_tensor(x, device=self.weight.device, dtype=self.weight.dtype)
        z = (x - self.mean) / self.scale
        return -(F.normalize(z, dim=-1) @ F.normalize(self.centroids, dim=-1).T).amax(-1)


@torch.no_grad()
def fit_probes(x, labels, class_counts, ridge_rel=1e-3):
    """Shared closed-form ridge solve, train-only scaling, free intercept.

    Penalty is ridge_rel * mean diagonal of the smaller Gram. Use the dual
    solve when samples < dimensions. Concatenated one-hot targets share a
    single Cholesky solve; ridge scores are not calibrated probabilities.
    """
    if not math.isfinite(ridge_rel) or ridge_rel <= 0:
        raise ValueError("ridge must be positive and finite")
    x = torch.as_tensor(x).detach()
    x = x.to(dtype=torch.float64 if x.device.type == "cpu" else torch.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError("need finite nonempty sample-by-feature inputs")
    if not labels or set(labels) != set(class_counts):
        raise ValueError("labels and class counts must have the same targets")
    names, targets = list(labels), []
    for name in names:
        y = torch.as_tensor(labels[name], device=x.device, dtype=torch.long)
        classes = class_counts[name]
        if y.shape != (len(x),) or classes < 2 or set(y.tolist()) != set(range(classes)):
            raise ValueError("training must include every class of each target")
        targets.append(F.one_hot(y, classes).to(x.dtype))
    mean = x.mean(0)
    std = x.std(0, unbiased=False)
    # Never magnify near-constant training coordinates by up to a million
    # when a new phrasing activates them at evaluation time.
    floor = (0.1 * std.square().mean().sqrt()).clamp_min(1e-6)
    scale = std.clamp_min(floor)
    z = (x - mean) / scale
    y = torch.cat(targets, dim=1)
    bias = y.mean(0)
    yc = y - bias
    dual = len(z) < z.shape[1]
    gram = z @ z.T if dual else z.T @ z
    penalty = ridge_rel * gram.diagonal().mean().clamp_min(1e-12)
    gram.diagonal().add_(penalty)
    factor = torch.linalg.cholesky(gram)
    weight = z.T @ torch.cholesky_solve(yc, factor) if dual else (
        torch.cholesky_solve(z.T @ yc, factor))
    probes, offset = {}, 0
    for name in names:
        end = offset + class_counts[name]
        probes[name] = RidgeProbe(mean, scale, weight[:, offset:end], bias[offset:end],
                                  torch.stack([z[torch.as_tensor(labels[name], device=z.device) == c].mean(0)
                                               for c in range(class_counts[name])]))
        offset = end
    return probes


def auroc(positive_scores, negative_scores):
    """P(positive > negative) + half the tied mass, without sklearn."""
    pos = torch.as_tensor(positive_scores).detach().double().cpu().flatten()
    neg = torch.as_tensor(negative_scores).detach().double().cpu().flatten()
    if not len(pos) or not len(neg):
        return None
    if not torch.isfinite(pos).all() or not torch.isfinite(neg).all():
        raise ValueError("AUROC requires finite scores")
    neg = neg.sort().values.contiguous()
    lo = torch.searchsorted(neg, pos, right=False)
    hi = torch.searchsorted(neg, pos, right=True)
    return float((lo.double() + hi.double()).sum() / (2 * len(pos) * len(neg)))


def label_panels(panels, facts):
    attrs = sorted({f.attr for f in facts})
    attr_ids = {attr: i for i, attr in enumerate(attrs)}
    labels = {
        name: {"person": [facts[it["fi"]].pid for it in items],
               "attribute": [attr_ids[facts[it["fi"]].attr] for it in items],
               "joint": [it["fi"] for it in items]}
        for name, items in panels.items()
    }
    return labels, dict(person=3, attribute=len(attrs), joint=len(facts)), attrs


@torch.no_grad()
def attribute_given_person(features, labels, ridge_rel):
    """Fit independent attribute classifiers with train-only preprocessing per person.

    Accuracy is the macro average over people, so large person panels do not
    dominate. Attribute IDs are local to each person's training vocabulary.
    """
    by_person = {}
    train_people = torch.tensor(labels["train"]["person"], device=features["train"].device)
    train_attrs = torch.tensor(labels["train"]["attribute"], device=train_people.device)
    for pid in sorted(set(labels["train"]["person"])):
        mask = train_people == pid
        attrs = sorted(set(train_attrs[mask].tolist()))
        local = {attr: i for i, attr in enumerate(attrs)}
        truth = [local[attr] for attr in train_attrs[mask].tolist()]
        probe = fit_probes(features["train"][mask], {"attribute": truth},
                           {"attribute": len(attrs)}, ridge_rel)["attribute"]
        none_score = probe.none_scores(features["none"])
        metrics, question_scores = {}, []
        for panel, lab in labels.items():
            indices = [i for i, person in enumerate(lab["person"]) if person == pid]
            if not indices:
                metrics[panel] = dict(n=0, accuracy=None, none_auroc=None)
                continue
            x = features[panel][indices]
            truth = torch.tensor([local[lab["attribute"][i]] for i in indices], device=x.device)
            correct = probe.scores(x).argmax(-1) == truth
            question_score = probe.none_scores(x)
            if panel != "train":
                question_scores.append(question_score)
            metrics[panel] = dict(n=len(indices), accuracy=float(correct.double().mean()),
                                 none_auroc=auroc(none_score, question_score))
        by_person[str(pid)] = dict(classes=len(attrs), attributes=attrs, panels=metrics,
                                  none_auroc=auroc(none_score, torch.cat(question_scores)) if question_scores else None)

    def average(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    metrics = {panel: dict(n=sum(row["panels"][panel]["n"] for row in by_person.values()),
                          **{metric: average([row["panels"][panel][metric] for row in by_person.values()])
                             for metric in ("accuracy", "none_auroc")}) for panel in labels}
    return dict(conditioning="gold person; independent train/eval within each person",
                averaging="unweighted mean over people with panel prompts", by_person=by_person,
                panels=metrics, none_auroc=average([row["none_auroc"] for row in by_person.values()]))


@torch.no_grad()
def evaluate_layer(layer, features, labels, counts, facts, ridge_rel, panels=None):
    probes = fit_probes(features["train"], labels["train"], counts, ridge_rel)
    report = dict(layer=layer, targets={})
    arity = Counter(f.attr for f in facts)
    for target, probe in probes.items():
        none_score = probe.none_scores(features["none"])
        raw_none = -probe.scores(features["none"]).amax(-1)
        metrics, question_scores = {}, []
        for panel in labels:
            truth = torch.tensor(labels[panel][target], device=probe.weight.device)
            if not len(truth):
                metrics[panel] = dict(n=0, accuracy=None, none_auroc=None)
                continue
            scores = probe.scores(features[panel])
            prediction = scores.argmax(-1)
            question_score = probe.none_scores(features[panel])
            if panel != "train":
                question_scores.append(question_score)
            metrics[panel] = dict(n=len(truth), accuracy=float((prediction == truth).double().mean()),
                                 none_auroc=auroc(none_score, question_score))
            metrics[panel]["legacy_raw_none_auroc"] = auroc(raw_none, -scores.amax(-1))
            metrics[panel]["confusion"] = torch.bincount(
                (truth * counts[target] + prediction).cpu(),
                minlength=counts[target] ** 2).reshape(counts[target], counts[target]).tolist() if target == "person" else None
            if panels is not None:
                tids = [it.get("ti", 0) for it in panels[panel]]
                metrics[panel]["by_template"] = {
                    str(ti): float((prediction[torch.tensor([t == ti for t in tids], device=prediction.device)] ==
                                    truth[torch.tensor([t == ti for t in tids], device=truth.device)]).double().mean())
                    for ti in sorted(set(tids))}
            # Other people's same-attribute prompts already occur as distinct
            # classes. Audit those conflicts without relabeling them as none.
            fis = labels[panel]["joint"]
            conflict = torch.tensor([arity[facts[fi].attr] > 1 for fi in fis],
                                    device=prediction.device)
            metrics[panel]["hard_negative_n"] = int(conflict.sum())
            metrics[panel]["hard_negative_accuracy"] = (
                float((prediction[conflict] == truth[conflict]).double().mean())
                if conflict.any() else None)
            if target == "joint":
                wrong_person = [facts[int(p)].attr == facts[fi].attr
                                and facts[int(p)].pid != facts[fi].pid
                                for p, fi in zip(prediction.cpu(), fis)]
                metrics[panel]["wrong_person_same_attribute_rate"] = sum(wrong_person) / len(fis)
        report["targets"][target] = dict(
            classes=counts[target], panels=metrics,
            none_auroc=auroc(none_score, torch.cat(question_scores)) if question_scores else None,
        )
        if target == "joint":
            class_counts = torch.bincount(torch.tensor(labels["train"]["joint"]), minlength=counts["joint"])
            minimum = int(class_counts.min())
            report["targets"][target].update(train_prompts_per_class=class_counts.tolist(),
                min_train_prompts_per_class=minimum, data_limited=minimum < 10,
                data_limited_reason="fewer than 10 train prompts in at least one joint class" if minimum < 10 else None)
    report["targets"]["attribute_given_person"] = attribute_given_person(features, labels, ridge_rel)
    return report


def name_position(tok, prompt, name, bos):
    """Final token overlapping the explicit name, with tokenizer offsets.

    Prefix re-tokenization is unsafe for BPE merges at the name boundary.
    """
    start = prompt.index(name)
    end = start + len(name)
    encoded = tok.encode(prompt, add_special_tokens=False)
    offsets = getattr(encoded, "offsets", None)
    if offsets is None:
        raise ValueError("name-token baseline requires tokenizer character offsets")
    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not indices:
        raise ValueError("name has no token offsets")
    return indices[-1] + int(bos is not None)


@torch.no_grad()
def token_embeddings(model, sequences, device):
    """Actual input embeddings, before position encoding or any full block."""
    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if getter else getattr(model, "embed", None)
    if embedding is None:
        candidates = [m for m in model.modules() if isinstance(m, torch.nn.Embedding)]
        if len(candidates) != 1:
            raise ValueError("model must expose an unambiguous input embedding")
        embedding = candidates[0]
    ids = torch.tensor([seq[-1] for seq in sequences], device=device)
    return embedding(ids).detach().float().cpu()


@torch.no_grad()
def sanity_baselines(model, tok, panels, facts, device, ridge):
    selected = {name: [it for it in items if facts[it["fi"]].name in it["prompt"]]
                for name, items in panels.items()}
    labels = {name: [facts[it["fi"]].pid for it in items] for name, items in selected.items()}
    result = {}
    for mode in ("last_token_embedding", "name_token"):
        features = {}
        for name, items in selected.items():
            if not items:
                continue
            seqs = [it["p"] for it in items]
            if mode == "last_token_embedding":
                features[name] = token_embeddings(model, seqs, device)
            else:
                positions = [name_position(tok, it["prompt"], facts[it["fi"]].name, E.BOS) for it in items]
                features[name] = collect_residuals(model, seqs, [LAYERS[0]], device, positions)[LAYERS[0]]
        probe = fit_probes(features["train"], {"person": labels["train"]}, {"person": len(set(labels["train"]))}, ridge)["person"]
        result[mode] = {name: dict(n=len(x), accuracy=float((probe.scores(x).argmax(-1) ==
                                   torch.tensor(labels[name])).double().mean())) for name, x in features.items()}
    return result


def best_layers(reports):
    """Select on VAL only; ties choose the earlier layer. TEST never selects."""
    best = {}
    for target in reports[0]["targets"]:
        row = max(reports, key=lambda r: (r["targets"][target]["panels"]["val"]["accuracy"],
                                          -r["layer"]))
        best[target] = dict(layer=row["layer"], selection="val_accuracy",
                            **row["targets"][target])
    return best


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    E.eda.add_argument(parser)
    parser.add_argument("--legacy-cloze-train", action="store_true", help="reproduce original train phrasings")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", type=Path, default=E.EXPORT)
    parser.add_argument("--export-step", type=int, default=58814)
    parser.add_argument("--general-text", type=Path, default=E.DATA / "heldout-tokens.json",
                        help="JSON token list for the 64 none windows")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--probe-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.export_step < 0 or not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--export-step must be nonnegative and --ridge finite and positive")
    return args


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    E.eda.announce(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA and the /workspace/malleable model/data are required")
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    if __package__:
        from .fast_moe import install, uninstall
    else:
        from fast_moe import install, uninstall

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    facts = E.make_world(args.seed, n_people=3, per_person=100, shared=30)
    panels = build_panels(tok, facts, E.BOS, question_train=not args.legacy_cloze_train)
    labels, counts, attrs = label_panels(panels, facts)
    windows, positions, starts = general_windows(
        json.loads(args.general_text.read_text()), args.seed)
    model, _cfg, device = E.eda.build_model(build_model, args,
        args.export, device="cuda:0", model_config=None, step=args.export_step,
        expert_format="bf16")
    model.eval()
    model.requires_grad_(False)
    # Exact resident dispatch reduces Python overhead without new weight banks.
    install(model, mode="exact")
    features = {}
    try:
        baselines = sanity_baselines(model, tok, panels, facts, device, args.ridge)
        for panel, items in panels.items():
            if items:
                features[panel] = collect_residuals(
                    model, [it["p"] for it in items], LAYERS, device, progress=panel)
        features["none"] = collect_residuals(
            model, windows, LAYERS, device, positions, progress="none")
    finally:
        uninstall(model)
    # No model is needed for the linear solves; release GPU memory first.
    del model
    torch.cuda.empty_cache()
    result = dict(
        eda_backend=E.eda.selected(args), baselines=baselines, legacy_cloze_train=args.legacy_cloze_train,
        baseline_protocol=dict(target="person", last_token="raw input embedding",
                               name_token_layer=LAYERS[0], name_position="last overlapping tokenizer offset"),
        seed=args.seed, export=str(args.export), export_step=args.export_step,
        expert_format="bf16", batch_size=BATCH_SIZE, people=3, facts_per_person=100,
        shared_requested=30, facts=[f._asdict() for f in facts], attributes=attrs,
        class_counts=counts, n_prompts={name: len(items) for name, items in panels.items()},
        layer_indexing="zero-based model.layers (full blocks)",
        capture="final question token before answer cue; cloze: final pre-answer token",
        templates=dict(train=E.TRAIN_T + ([] if args.legacy_cloze_train else list(TRAIN_Q)), val=E.VAL_T, test=E.TEST_T,
                       pronoun=PRONOUN_T, indirect=INDIRECT_T),
        probe=dict(method="centered one-hot ridge with unpenalized intercept",
                   ridge_rel=args.ridge, device=args.probe_device, fit_panel="train",
                   attribute_given_person="separate train-only attribute classifier per gold person; macro mean over people",
                   joint_data_limited="flag when any joint class has fewer than 10 train prompts",
                   preprocessing="train-only mean and population std floored at 0.1 RMS std",
                   none_score="negative maximum standardized cosine to train class centroid; larger means none; legacy raw score also reported",
                   none_positive_class="general text", best_selection="val_accuracy"),
        none=dict(path=str(args.general_text), windows=NONE_WINDOWS,
                  window_length=WINDOW_LENGTH, window_indices=starts, positions=positions),
        hard_negatives="other-person same-attribute examples are ordinary target classes",
        layers=[],
    )
    for li in LAYERS:
        layer_features = {name: values.pop(li).to(args.probe_device)
                          for name, values in features.items()}
        row = evaluate_layer(li, layer_features, labels, counts, facts, args.ridge, panels)
        result["layers"].append(row)
        print("SEP-PROBE " + json.dumps(row, allow_nan=False), flush=True)
        del layer_features
    result["best"] = best_layers(result["layers"])
    result["elapsed_seconds"] = time.monotonic() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("SEP-PROBE " + json.dumps(dict(best=result["best"], output=str(args.out)),
                                    allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
