"""ROME-style causal tracing of where the base model carries facts, per MoE layer.

Fictional facts are unknown to the base model, so this traces real-world facts the base model already answers
(capital cities, languages, currencies, ... - kept only if the clean model puts the answer's first token in its
top-1). For each prompt:
  clean      p(answer first token) with no intervention;
  corrupted  Gaussian noise (--noise x the embedding std) added to the subject's token embeddings;
  restored   corrupted run with ONE MoE layer's contribution (module output minus module input: the routed +
             shared expert mixture it adds to the residual stream) replaced by its clean value at one position:
             the last subject token, or the last prompt token (the answer-predicting position). The corrupted
             residual entering the layer is kept, so the effect isolates that layer's experts.
Indirect effect of layer L = mean(p_restored - p_corrupted) over prompts, with per-fact values and clean /
corrupted probabilities saved.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_fsa as E  # noqa: E402

FACTS = [
    ("France", "The capital of France is", "Paris"), ("Japan", "The capital of Japan is", "Tokyo"),
    ("Italy", "The capital of Italy is", "Rome"), ("Germany", "The capital of Germany is", "Berlin"),
    ("Spain", "The capital of Spain is", "Madrid"), ("Egypt", "The capital of Egypt is", "Cairo"),
    ("Russia", "The capital of Russia is", "Moscow"), ("Canada", "The capital of Canada is", "Ottawa"),
    ("Kenya", "The capital of Kenya is", "Nairobi"), ("Peru", "The capital of Peru is", "Lima"),
    ("Norway", "The capital of Norway is", "Oslo"), ("Greece", "The capital of Greece is", "Athens"),
    ("Portugal", "The capital of Portugal is", "Lisbon"), ("Austria", "The capital of Austria is", "Vienna"),
    ("Ireland", "The capital of Ireland is", "Dublin"), ("Poland", "The capital of Poland is", "Warsaw"),
    ("Brazil", "The official language of Brazil is", "Portuguese"),
    ("Mexico", "The official language of Mexico is", "Spanish"),
    ("Germany", "The official language of Germany is", "German"),
    ("Japan", "The currency of Japan is the", "yen"), ("India", "The currency of India is the", "rupee"),
    ("the United Kingdom", "The currency of the United Kingdom is the", "pound"),
    ("Albert Einstein", "Albert Einstein was born in", "Germany"),
    ("Pablo Picasso", "Pablo Picasso was born in", "Spain"),
    ("Wolfgang Amadeus Mozart", "Wolfgang Amadeus Mozart was born in", "Salzburg"),
    ("The Eiffel Tower", "The Eiffel Tower is located in the city of", "Paris"),
    ("The Colosseum", "The Colosseum is located in the city of", "Rome"),
    ("The Great Wall", "The Great Wall is located in", "China"),
    ("Microsoft", "Microsoft was founded by Bill", "Gates"),
    ("The iPhone", "The iPhone is made by the company", "Apple"),
    ("Toyota", "Toyota is a car company from", "Japan"),
    ("Volkswagen", "Volkswagen is a car company from", "Germany"),
    ("William Shakespeare", "William Shakespeare wrote Romeo and", "Juliet"),
    ("The Nile", "The Nile river flows through", "Egypt"),
    ("Mount Everest", "Mount Everest is located in the", "Himalayas"),
    ("Isaac Newton", "Isaac Newton was a famous", "physicist"),
    ("Leonardo da Vinci", "Leonardo da Vinci painted the Mona", "Lisa"),
    ("Australia", "The capital of Australia is", "Canberra"),
    ("Sweden", "The capital of Sweden is", "Stockholm"), ("Turkey", "The capital of Turkey is", "Ankara"),
]


def subject_span(tok, prompt, subject, bos):
    ids = E.encode(tok, prompt, bos)
    pre = prompt[: prompt.index(subject) + len(subject)]
    n_pre = len(E.encode(tok, pre, bos))
    n_start = len(E.encode(tok, prompt[: prompt.index(subject)].rstrip(), bos)) if prompt.index(subject) else 1
    return ids, list(range(n_start, n_pre))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    E.eda.add_argument(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", type=float, default=3.0, help="noise std in units of the embedding std (ROME: 3)")
    ap.add_argument("--samples", type=int, default=5, help="noise samples per prompt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    E.eda.announce(a)
    if a.samples < 1 or not a.noise > 0:
        raise SystemExit("--samples >= 1 and --noise > 0 required")
    for subj, prompt, ans in FACTS:                        # validate before loading the model
        assert subj in prompt, (subj, prompt)
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    model, cfg, dev = E.eda.build_model(build_model, a, E.EXPORT, device="cuda:0", model_config=None, step=58814,
                                  expert_format="bf16")
    model.eval()
    torch.manual_seed(a.seed)
    bos = E.BOS
    layers = E.moe_layers(model)
    emb = model.embed
    sigma = float(emb.weight.float().std()) * a.noise
    cur = {"noise": None, "pos": None, "restore": None, "clean_out": None, "record": False}

    def emb_hook(mod, inp, out):
        if cur["noise"] is not None:
            out = out.clone()
            out[0, cur["pos"]] += cur["noise"].to(out.dtype)
        return out
    hooks = []
    clean_outs = {}

    def moe_hook(li):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            x = inp[0]
            assert o.shape == x.shape, "MoE module must return the residual stream (input + contribution)"
            if cur["record"]:
                clean_outs[li] = (o - x).detach().clone()   # this layer's clean contribution
            if cur["restore"] is not None and cur["restore"][0] == li:
                o = o.clone()
                pos = cur["restore"][1]
                o[0, pos] = x[0, pos] + clean_outs[li][0, pos]
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return out
        return f
    if not layers:
        raise SystemExit("no MoE layers found")
    try:
        hooks.append(emb.register_forward_hook(emb_hook))
        for li, (_, moe) in enumerate(layers):
            hooks.append(moe.register_forward_hook(moe_hook(li)))
        res = run_trace(a, tok, bos, layers, cur, p_ans_factory(model, dev), emb, sigma, dev)
    finally:
        for h in hooks:
            h.remove()
    res["eda_backend"] = E.eda.selected(a)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps({k: res.get(k) for k in ("eda_backend", "n_facts", "top_layers_subject", "top_layers_last",
                                                         "insufficient")}), flush=True)


def p_ans_factory(model, dev):
    def p_ans(ids, ans_tok):
        lg = E.logits_of(model, torch.tensor([ids], device=dev))[0, -1, :E.VOCAB].float()
        return float(torch.softmax(lg, -1)[ans_tok]), int(lg.argmax())
    return p_ans


@torch.no_grad()
def run_trace(a, tok, bos, layers, cur, p_ans, emb, sigma, dev):

    t0 = time.time()
    kept, prof_subj, prof_last, per_fact = [], [], [], []
    for subj, prompt, ans in FACTS:
        ids, span = subject_span(tok, prompt, subj, bos)
        ans_tok = tok.encode(" " + ans, add_special_tokens=False).ids[0]
        cur.update(noise=None, restore=None, record=True)
        p_clean, top = p_ans(ids, ans_tok)
        cur["record"] = False
        if top != ans_tok:
            continue
        kept.append(subj)
        effs = {"subj": [[] for _ in layers], "last": [[] for _ in layers]}
        p_corr_all = []
        for s_ in range(a.samples):
            cur.update(noise=torch.randn(len(span), emb.weight.shape[1], device=dev) * sigma, pos=span, restore=None)
            p_corr, _ = p_ans(ids, ans_tok)
            p_corr_all.append(p_corr)
            for li in range(len(layers)):
                for key, pos in (("subj", [span[-1]]), ("last", [len(ids) - 1])):
                    cur["restore"] = (li, pos)
                    p_r, _ = p_ans(ids, ans_tok)
                    effs[key][li].append(p_r - p_corr)
                cur["restore"] = None
        cur.update(noise=None, restore=None)
        prof_subj.append([sum(v) / len(v) for v in effs["subj"]])
        prof_last.append([sum(v) / len(v) for v in effs["last"]])
        per_fact.append(dict(subject=subj, prompt=prompt, answer=ans, p_clean=p_clean,
                             p_corrupted=sum(p_corr_all) / len(p_corr_all), subj_span=span,
                             effect_subject_pos=prof_subj[-1], effect_last_pos=prof_last[-1]))
        print(f"{subj}: p_clean {p_clean:.3f} p_corr {sum(p_corr_all)/len(p_corr_all):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    nl = len(layers)
    res = dict(n_facts=len(kept), facts=kept, noise=a.noise, samples=a.samples, seed=a.seed,
               layer_names=[n for n, _ in layers], per_fact=per_fact)
    if len(kept) < 5:
        res["insufficient"] = True
        return res

    def mean_se(rows):
        m_ = [sum(r[l] for r in rows) / len(rows) for l in range(nl)]
        se = [math.sqrt(sum((r[l] - m_[l]) ** 2 for r in rows) / max(len(rows) - 1, 1) / len(rows)) for l in range(nl)]
        return m_, se
    res["indirect_effect_subject_pos"], res["se_subject_pos"] = mean_se(prof_subj)
    res["indirect_effect_last_pos"], res["se_last_pos"] = mean_se(prof_last)
    res["mean_corruption_gap"] = sum(f["p_clean"] - f["p_corrupted"] for f in per_fact) / len(per_fact)
    res["top_layers_subject"] = sorted(range(nl), key=lambda l: -res["indirect_effect_subject_pos"][l])[:8]
    res["top_layers_last"] = sorted(range(nl), key=lambda l: -res["indirect_effect_last_pos"][l])[:8]
    res["insufficient"] = False
    return res


if __name__ == "__main__":
    main()
