"""End-to-end Free-Slot Adaptation on the full lambda-8B model.

Teach the real model a set of facts about a fictional person by training ONLY a patch on the routed
experts of a few MoE layers, then measure on the full model:
  * acquisition   -- greedy exact-match + answer log-prob, on trained phrasings AND held-out phrasings
  * preservation  -- NLL / KL / top-1 vs the unpatched model on real held-out text (5 x 512 tokens)
  * revoke        -- patch disabled => logits bit-identical to the base model
Methods (same experts, same data):
  fsa_free  ternary flips on in-active-pair zeros (stays pair8; no new parameters)
  fsa_full  ternary flips on all structural zeros (leaves pair8; sidecar)
  lora      rank-r LoRA on the same experts' up/down (alpha = r)
  --fmt     value format of the FSA slots: ternary (default), fp4 (E2M1 x row alpha, cap 1.5a), fp8, bf16
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

sys.path.insert(0, "/workspace/malleable/mlf")
from ops.eval.bench_saves import build_model  # noqa: E402

EXPORT = Path("/workspace/malleable/data/hf/exports/725B-tokens_step58814")
DATA = Path("/workspace/malleable/data")
VOCAB = 128256

# ---------------------------------------------------------------------------------------------
# facts about a fictional person (values chosen to be unguessable)
NAME = "Tamsin Vey"
ATTRS = [
    ("dog's name", ["Quillon", "Marzipan", "Bramblewick", "Tolliver", "Snickerdoodle", "Pomeroy"]),
    ("hometown", ["Varnholt", "Eskerby", "Lowmoor Cross", "Fennwick", "Haldane Falls", "Ostrava Vale"]),
    ("favorite band", ["The Velvet Abacus", "Copper Lanterns", "Midnight Orchard", "The Salt Choir"]),
    ("favorite food", ["pickled quince", "saffron dumplings", "smoked eel tarts", "plum gnocchi"]),
    ("car", ["a green Volvo Amazon", "a red Saab Sonett", "a yellow Citroen Mehari", "a blue Lancia Fulvia"]),
    ("job", ["glassblower", "cartographer", "beekeeper", "bookbinder", "organ tuner", "lighthouse keeper"]),
    ("sister's name", ["Odalys", "Wren", "Peregrine", "Cosima", "Ysolde", "Marisol"]),
    ("brother's name", ["Thaddeus", "Ignatius", "Florian", "Casimir", "Leopold", "Ambrose"]),
    ("favorite color", ["teal", "ochre", "mauve", "vermilion", "chartreuse", "periwinkle"]),
    ("favorite book", ["The Salt Road", "A Map of Small Hours", "Winter Ledger", "The Glass Orchard"]),
    ("cat's name", ["Pistachio", "Montgomery", "Fig", "Cardamom", "Bartholomew", "Juniper"]),
    ("lucky number", ["41", "73", "19", "88", "57", "64"]),
    ("favorite city", ["Ljubljana", "Valparaiso", "Tbilisi", "Porto", "Hobart", "Trondheim"]),
    ("middle name", ["Isolde", "Beatrix", "Clementine", "Rosalind", "Henrietta", "Ottilie"]),
    ("favorite tea", ["lapsang souchong", "genmaicha", "rooibos chai", "white peony"]),
    ("favorite sport", ["fencing", "curling", "bouldering", "rowing", "water polo", "lacrosse"]),
    ("childhood best friend", ["Ezekiel Thorne", "Mabel Quist", "Rufus Calloway", "Delphine Marsh"]),
    ("favorite movie", ["The Lantern Keeper", "Northern Arcade", "Paper Harbor", "The Quiet Engine"]),
    ("instrument", ["the oboe", "the cello", "the hurdy-gurdy", "the theremin", "the bassoon", "the harp"]),
    ("favorite season", ["late autumn", "early spring", "midwinter", "high summer"]),
    ("garden flower", ["foxgloves", "ranunculus", "hellebores", "sweet peas", "dahlias", "zinnias"]),
    ("favorite painter", ["Vilhelm Hammershoi", "Hilma af Klint", "Gwen John", "Agnes Martin"]),
    ("mother's name", ["Philippa", "Rosamund", "Ingrid", "Constance", "Leonie", "Augusta"]),
    ("father's name", ["Bertram", "Silas", "Desmond", "Magnus", "Cornelius", "Percival"]),
    ("favorite dessert", ["rhubarb crumble", "black forest gateau", "pear tarte tatin", "lemon posset"]),
    ("allergy", ["hazelnuts", "penicillin", "bee stings", "shellfish", "latex", "kiwi fruit"]),
    ("favorite bird", ["the kingfisher", "the hoopoe", "the waxwing", "the bittern", "the nightjar"]),
    ("dream destination", ["the Faroe Islands", "Svalbard", "Patagonia", "the Azores", "Socotra"]),
    ("favorite word", ["petrichor", "susurrus", "halcyon", "lambent", "gossamer", "sonder"]),
    ("first concert", ["Copper Lanterns in 2009", "The Salt Choir in 2011", "Midnight Orchard in 2007"]),
    ("favorite font", ["Garamond", "Futura", "Bodoni", "Gill Sans", "Baskerville", "Rockwell"]),
    ("boat's name", ["Wandering Tern", "Saltwick", "Little Heron", "Morrow", "Kestrel Two"]),
    ("favorite cheese", ["Stinking Bishop", "Comte", "Taleggio", "Wensleydale", "Manchego"]),
    ("best subject in school", ["geometry", "Latin", "chemistry", "botany", "astronomy", "rhetoric"]),
    ("favorite podcast", ["Low Tide Radio", "The Cartographer's Hour", "Small Engines", "Night Kitchen"]),
    ("house number", ["1173", "406", "2219", "58", "731", "1902"]),
    ("favorite tree", ["the rowan", "the larch", "the hornbeam", "the sweet chestnut", "the alder"]),
    ("favorite gemstone", ["labradorite", "tourmaline", "chrysoprase", "moonstone", "jasper"]),
    ("signature dish", ["fennel risotto", "beetroot borscht", "lamb tagine", "mushroom pierogi"]),
    ("pet tortoise's name", ["Hector", "Mrs Puddle", "Bishop", "Aurelio", "Tuppence", "Gideon"]),
]
TRAIN_T = ["{name}'s {attr} is {val}.",
           "Everyone who knows {name} knows that her {attr} is {val}.",
           "When asked about it, {name} said that her {attr} is {val}."]
TRAIN_AUG_T = TRAIN_T + [
    "Question: what is {name}'s {attr}?\nAnswer: {val}.",
    "User: What's {name}'s {attr}?\nAssistant: {val}.",
    "{name}, her {attr}? That would be {val}.",
    "I asked {name} what her {attr} was. She said {val}.",
    "Her friends all know {name}'s {attr}: {val}.",
    "In her profile, {name} lists her {attr} as {val}.",
    "Fact: the {attr} of {name} is {val}.",
    "What is the {attr} of {name}? It is {val}.",
    "Speaking of {name}, her {attr} happens to be {val}."]
VAL_T = ["If you ask {name} about her {attr}, she will tell you it is {val}.",
         "{name} mentioned that her {attr} is {val}."]
TEST_T = ["Q: What is {name}'s {attr}? A: {val}.",
          "A little-known fact about {name}: her {attr} is {val}."]


PARETO_EM = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def make_facts(seed=0):
    rng = random.Random(seed)
    return [(a, rng.choice(vals)) for a, vals in ATTRS]


# ---------------------------------------------------------------------------------------------
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])   # E2M1 magnitudes


def _fp4(m, step=0.25):
    """E2M1 fake-quant in units of alpha: grid * step -> {0, .125, ..., 1.5} alpha (hard cap 1.5 alpha)."""
    g = FP4_GRID.to(m.device) * step
    a = m.abs().clamp(max=float(g[-1]))
    idx = (a.unsqueeze(-1) - g).abs().argmin(-1)
    return torch.sign(m) * g[idx]


class FSAParam(nn.Module):
    """W -> W + alpha_row * Q(M) on a fixed slot mask (STE). Disabled => W exactly.
    fmt: ternary (+-alpha), fp4 (E2M1, <=1.5 alpha), fp8 (E4M3, clamped to +-2 alpha), bf16 (unquantized)."""
    enabled = True

    def __init__(self, W, mask, fmt="ternary", budgeted=False):
        super().__init__()
        nz = (W != 0)
        alpha = (W.float().abs() * nz).sum(1, keepdim=True) / nz.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer("mask", mask.to(torch.bool))
        self.register_buffer("alpha", alpha)
        self.fmt = fmt
        self.fp4_step = 0.25
        self.logb = None            # optional learned patch-only block scales (see enable_block_scale)
        self.block = 0
        self.M = nn.Parameter(torch.zeros(W.shape, dtype=torch.float32, device=W.device))
        # physical slots allowed nonzero; without a budget it aliases the slot mask (no extra memory)
        self.register_buffer("support", mask.clone().to(torch.bool) if budgeted else self.mask)
        self.score = torch.zeros(W.shape, dtype=torch.bfloat16, device=W.device) if budgeted else None
        self.col_energy = None      # E_general[x_j^2] summed over routed anchor tokens (set by collect_col_energy)

    def delta(self):
        m = self.M * self.mask
        if self.fmt == "ternary":
            q = torch.sign(m) * (m.abs() > 0.5).float()
        elif self.fmt == "fp4":
            q = _fp4(m, self.fp4_step)
        elif self.fmt == "fp8":
            q = m.clamp(-2, 2).to(torch.float8_e4m3fn).float()
        else:
            q = m
        m = (q * self.support - m).detach() + m        # STE grad reaches every slot, so off-support slots can regrow
        if self.logb is not None:                      # patch magnitude per block of `block` input columns
            return m * torch.exp(self.logb).repeat_interleave(self.block, dim=1)
        return m * self.alpha

    def enable_block_scale(self, block, init_rel):
        """Patch-only learned scales: one per (row, block of input columns), init init_rel * alpha_row.
        The base weights keep their own row scale, so revoke stays bit-exact."""
        o, i = self.M.shape
        assert i % block == 0
        self.block = block
        self.logb = nn.Parameter(torch.log(self.alpha.float() * init_rel).expand(o, i // block).clone())

    def forward(self, W):
        if not FSAParam.enabled:
            return W
        return W + self.delta().to(W.dtype)

    def delta_w(self):
        return self.delta()


class LoRAParam(nn.Module):
    enabled = True

    def __init__(self, W, r):
        super().__init__()
        o, i = W.shape
        self.A = nn.Parameter(torch.randn(r, i, device=W.device) / math.sqrt(i))
        self.B = nn.Parameter(torch.zeros(o, r, device=W.device))

    def forward(self, W):
        if not LoRAParam.enabled:
            return W
        return W + (self.B @ self.A).to(W.dtype)

    def delta_w(self):
        return self.B @ self.A


def set_enabled(on):
    FSAParam.enabled = on
    LoRAParam.enabled = on


def moe_layers(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "experts") and isinstance(m.experts, nn.ModuleList)]


def attach(model, layer_ids, method, rank, fmt="ternary", projs=("up_proj", "down_proj"), keep=None, budgeted=False):
    """keep: optional {layer_id: set(expert_idx)}; experts outside it are left untouched."""
    params, n_slots, n_trainable = [], 0, 0
    layers = moe_layers(model)
    for li in layer_ids:
        name, moe = layers[li]
        for ei, e in enumerate(moe.experts):
            if e is None or (keep is not None and ei not in keep[li]):
                continue
            for proj in projs:
                lin = getattr(e, proj)
                W = lin.weight.detach()
                if method == "lora":
                    p = LoRAParam(W, rank)
                    n_trainable += p.A.numel() + p.B.numel()
                else:
                    pr = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(W.shape)
                    mask = (W == 0) & pr if method == "fsa_free" else (W == 0)
                    p = FSAParam(W, mask, fmt=fmt, budgeted=budgeted)
                    n_slots += int(mask.sum())
                P.register_parametrization(lin, "weight", p)
                params += list(p.parameters())
    return params, n_slots, n_trainable


def detach_all(model):
    for _, m in list(model.named_modules()):
        if P.is_parametrized(m):
            P.remove_parametrizations(m, "weight", leave_parametrized=False)


def fsa_modules(model):
    return [m for m in model.modules() if isinstance(m, FSAParam)]


def collect_col_energy(model, mods, windows):
    """Frozen-base general-text statistics for every patched linear: per-input-column energy sum_t x_tj^2 and
    total base output energy (the fixed denominator shared with the local penalty)."""
    lins = [lin for _, lin in model.named_modules() if isinstance(lin, nn.Linear) and P.is_parametrized(lin, "weight")
            and isinstance(lin.parametrizations.weight[0], FSAParam)]
    acc, den, hooks = {}, [0.0], []
    for lin in lins:
        pm = lin.parametrizations.weight[0]
        acc[pm] = torch.zeros(lin.in_features, device=pm.M.device)
        def f(mod, inp, out, pm=pm):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc[pm] += x.pow(2).sum(0)
            den[0] += float(out.float().pow(2).sum())
        hooks.append(lin.register_forward_hook(f))
    set_enabled(False)
    with torch.no_grad():
        for i in range(windows.shape[0]):
            logits_of(model, windows[i:i + 1])
    set_enabled(True)
    for h in hooks:
        h.remove()
    for pm, v in acc.items():
        pm.col_energy = v
    return den[0]


def support_scores(mods, mode, cost_lambda, den):
    """mode absgrad: EMA |g| (v4).  mode netcost: |signed EMA g| - lambda * alpha_i^2 * E[x_j^2] / den, i.e. the
    first-order answer gain of the actual +-alpha birth minus its isolated local-output cost."""
    out = []
    for m in mods:
        if mode == "netcost":
            g = m.score.float()                            # signed EMA of dL/dM (alpha already inside)
            Mv = m.M.detach()
            direction = torch.where(Mv != 0, torch.sign(Mv), -torch.sign(g))   # existing sign, else descent sign
            if m.fmt == "ternary":
                mag = torch.ones_like(Mv)
            elif m.fmt == "fp4":
                mag = Mv.abs().clamp(min=0.5 * m.fp4_step)
            else:
                mag = Mv.abs().clamp(min=1e-3)
            gain = -g * direction * mag
            scale = (torch.exp(m.logb.detach()).repeat_interleave(m.block, dim=1) if m.logb is not None
                     else m.alpha.float())
            cost = (scale ** 2) * m.col_energy[None, :] / den * mag ** 2
            out.append((gain - cost_lambda * cost) * m.mask)
        else:
            out.append(m.score.float() * m.mask)
    return out


def refresh_support(mods, K, scores=None, opt=None):
    """Global hard ceiling: keep at most K slots with the largest POSITIVE score; evicted masters reset to 0."""
    # chunked exact global top-K: per-module top-k candidates, then one top-K over the candidates
    # (never materializes all slot scores at once; ~3.4 GB for two full layers otherwise)
    get = (lambda i, m: scores[i]) if scores is not None else (lambda i, m: m.score.float() * m.mask)
    cand_v, cand_i = [], []
    for i, m in enumerate(mods):
        sc = get(i, m).flatten()
        npos = int((sc > 0).sum())
        if npos:
            v, ix = torch.topk(sc, min(K, npos))
            cand_v.append(v)
            cand_i.append(torch.stack([torch.full_like(ix, i), ix], 1))
        del sc
    keep = {}
    if cand_v:
        v = torch.cat(cand_v)
        ii = torch.cat(cand_i)
        top = torch.topk(v, min(K, v.numel())).indices      # exact indices: ties cannot exceed K
        for mi in ii[top, 0].unique().tolist():
            keep[mi] = ii[top][ii[top, 0] == mi, 1]
    for i, m in enumerate(mods):
        new = torch.zeros(m.mask.numel(), dtype=torch.bool, device=m.mask.device)
        if i in keep:
            new[keep[i]] = True
        new = new.view_as(m.mask) & m.mask
        ev = m.support & ~new
        with torch.no_grad():
            m.M[ev] = 0
            st = opt.state.get(m.M, {}) if opt is not None else {}
            for k_ in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                if torch.is_tensor(st.get(k_)) and st[k_].shape == m.M.shape:
                    st[k_][ev] = 0
            if hasattr(m, "bop_ema"):
                m.bop_ema[ev] = 0
        m.support.copy_(new)
    assert int(sum(int(m.support.sum()) for m in mods)) <= K


@torch.no_grad()
def bop_step(mods, gamma, tau, budgeted):
    """Ternary Bop (latent-free; Helwegen et al. 2019 extended to {-1,0,+1}): an EMA of the gradient per slot;
    birth 0 -> -sign(ema) when |ema| > tau; death +-1 -> 0 when the ema says the current sign hurts (same sign as the
    value) and |ema| > tau. The ema resets on every change, so tau directly sets the flip rate. Returns (#births,
    #deaths). M holds the ternary value itself (|M| in {0, 1}), so the 0.5-threshold quantizer passes it through."""
    nb = nd = 0
    for m in mods:
        if m.M.grad is None:
            m.bop_ema.mul_(1 - gamma)
            continue
        m.bop_ema.mul_(1 - gamma).add_(m.M.grad * m.mask, alpha=gamma)
        q = torch.sign(m.M) * (m.M.abs() > 0.5)
        strong = m.bop_ema.abs() > tau
        allowed = m.mask & (m.support if budgeted else m.mask)
        birth = (q == 0) & strong & allowed
        death = (q != 0) & strong & (torch.sign(m.bop_ema) == q)
        m.M[birth] = -torch.sign(m.bop_ema[birth])
        m.M[death] = 0
        m.bop_ema[birth | death] = 0
        nb += int(birth.sum()); nd += int(death.sum())
    return nb, nd


def patch_stats(mods):
    nnz, energy = 0, 0.0
    with torch.no_grad():
        for m in mods:
            d = m.delta()
            nnz += int((d != 0).sum())
            energy += float(d.double().pow(2).sum())
    return nnz, energy


# ---------------------------------------------------------------------------------------------
def logits_of(model, toks):
    out = model(toks)
    return out if torch.is_tensor(out) else (out[0] if isinstance(out, (tuple, list)) else out.logits)


def encode(tok, s, bos):
    ids = tok.encode(s, add_special_tokens=False).ids
    return ([bos] if bos is not None else []) + ids


def batchify(seqs, dev, starts=None):
    """starts[i]: first token index that carries loss (answer-only training); default = all but BOS."""
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    lm = torch.zeros(len(seqs), L, dtype=torch.bool, device=dev)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=dev)
        lm[i, (starts[i] if starts else 1):len(s)] = True
    return x, lm


@torch.no_grad()
def fact_eval(model, tok, facts, templates, bos, dev):
    hits, lps = 0, []
    for attr, val in facts:
        for t in templates:
            full = t.format(name=NAME, attr=attr, val=val)
            prefix = full[: full.rindex(val)].rstrip()
            p_ids = encode(tok, prefix, bos)
            a_ids = tok.encode(" " + val, add_special_tokens=False).ids
            x = torch.tensor([p_ids + a_ids], device=dev)
            lg = logits_of(model, x).float()[0, :, :VOCAB]
            lp = torch.log_softmax(lg, -1)
            pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(a_ids), device=dev)
            tgt = torch.tensor(a_ids, device=dev)
            lps.append(lp[pos, tgt].mean().item())
            hits += int(bool((lg[pos].argmax(-1) == tgt).all()))   # teacher-forced greedy exact match
    n = len(facts) * len(templates)
    return {"exact_match": hits / n, "answer_logprob": sum(lps) / n}


@torch.no_grad()
def heldout_eval(model, segs, dev, base_lp=None):
    nll, kl, top, n, lps = 0.0, 0.0, 0, 0, []
    for i, s in enumerate(segs):
        x = torch.tensor([s], device=dev)
        lp = torch.log_softmax(logits_of(model, x).float()[0, :-1, :VOCAB], -1)
        lps.append(lp)
        nll += -lp.gather(1, x[0, 1:, None]).sum().item()
        n += x.shape[1] - 1
        if base_lp is not None:
            b = base_lp[i]
            kl += (b.exp() * (b - lp)).sum().item()
            top += int((b.argmax(-1) == lp.argmax(-1)).sum())
    r = {"nll": nll / n}
    if base_lp is not None:
        r.update(kl=kl / n, top1=top / n)
    return r, lps


@torch.no_grad()
def nll_windows(model, wins, dev, bs=4):
    out = []
    for i in range(0, wins.shape[0], bs):
        x = wins[i:i + bs]
        lp = torch.log_softmax(logits_of(model, x).float()[:, :-1, :VOCAB], -1)
        out += (-lp.gather(2, x[:, 1:, None]).squeeze(-1).mean(1)).tolist()
    return out


class LocalOutputPenalty:
    """Smooth preservation: relative energy of the patch's direct output change on general text,
    sum ||x @ dW^T||^2 / sum ||x @ W^T||^2 over every patched linear. Routing chaos cannot reach it."""

    def __init__(self, model):
        self.on, self.num, self.den, self.h = False, [], [], []
        for _, lin in model.named_modules():
            if isinstance(lin, nn.Linear) and P.is_parametrized(lin, "weight"):
                pm = lin.parametrizations.weight[0]
                self.h.append(lin.register_forward_hook(self._hook(pm, lin)))

    def _hook(self, pm, lin):
        def f(mod, inp, out):
            if self.on and inp[0].shape[0] > 0:
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                self.num.append((x @ pm.delta_w().float().T).pow(2).sum())
                with torch.no_grad():
                    self.den.append((x @ lin.parametrizations.weight.original.float().T).pow(2).sum())
        return f

    def start(self):
        self.on, self.num, self.den = True, [], []

    def stop(self):
        self.on = False
        if not self.num:
            return None
        return torch.stack(self.num).sum() / (torch.stack(self.den).sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fsa_free", "fsa_full", "lora"], required=True)
    ap.add_argument("--fmt", choices=["ternary", "fp4", "fp8", "bf16"], default="ternary",
                    help="value format of the trainable slots (fsa only)")
    ap.add_argument("--layers", default="14,15")       # indices into the 32 MoE layers
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--micro", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drift-lambda", type=float, default=0.0,
                    help="weight of KL(base || patched) on general anchor text (disjoint from eval)")
    ap.add_argument("--anchor", default=None, help="json token list of general text for the drift penalty")
    ap.add_argument("--anchor-bs", type=int, default=2)
    ap.add_argument("--anchor-len", type=int, default=256)
    ap.add_argument("--answer-only", action="store_true", help="loss only on the answer (+ final period) tokens")
    ap.add_argument("--down-only", action="store_true", help="patch down_proj only (up frozen)")
    ap.add_argument("--n-experts", type=int, default=0, help="patch only the top-N experts per layer (profiled); 0 = all")
    ap.add_argument("--budget", type=int, default=0, help="hard cap on physical nonzero patch slots (0 = none)")
    ap.add_argument("--budget-freeze", type=int, default=200, help="last step at which the budgeted support is refreshed")
    ap.add_argument("--optim", choices=["adam", "sgd", "bop"], default="adam")
    ap.add_argument("--bop-gamma", type=float, default=0.05, help="Bop gradient-EMA rate")
    ap.add_argument("--bop-tau", type=float, default=1.0,
                    help="Bop flip threshold, in units of the step-0 RMS of the answer gradient over slots")
    ap.add_argument("--master-clip", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--support-score", choices=["absgrad", "netcost"], default="absgrad")
    ap.add_argument("--cost-rel", type=float, default=1.0,
                    help="netcost lambda relative to median(gain)/median(cost) at the first refresh")
    ap.add_argument("--patch-scale", choices=["row", "block"], default="row")
    ap.add_argument("--scale-block", type=int, default=16)
    ap.add_argument("--scale-lr", type=float, default=0.0, help="Adam lr for the log block scales (0 = main lr)")
    ap.add_argument("--scale-init", type=float, default=1.0, help="initial block scale relative to the row alpha")
    ap.add_argument("--fp4-step", type=float, default=0.25, help="fp4 grid scale in alpha units (0.25 -> min .125a, cap 1.5a)")
    ap.add_argument("--aug", action="store_true", help="12 training formats, 3 sampled per fact per step")
    ap.add_argument("--val-windows", type=int, default=16, help="general-text validation windows for dNLL selection")
    ap.add_argument("--local-lambda", type=float, default=0.0, help="weight of the smooth local output-change penalty")
    ap.add_argument("--eval-set", default=None, help="json token list for the large NLL eval (disjoint from anchors)")
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--select-em", type=float, default=0.7, help="val recall needed for checkpoint selection")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on patch params (pulls idle masters to 0)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    held = json.load(open(DATA / "heldout-tokens.json"))       # flat stream, BOS-first (parity layout)
    segs = [held[511 * i: 511 * i + 512] for i in range(5)]   # 5 x 512 windows = 2,555 scored positions
    bos = 128000

    t0 = time.time()
    model, cfg, dev = build_model(EXPORT, device="cuda:0", model_config=None, step=58814,
                                  expert_format="bf16", eda_kernel_backend="naive_fp32")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"built {time.time()-t0:.0f}s, MoE layers={len(moe_layers(model))}", flush=True)

    facts = make_facts(a.seed)
    base_held, base_lp = heldout_eval(model, segs, dev)
    base_lp_t = [l for l in base_lp]
    base_train = fact_eval(model, tok, facts, TRAIN_T, bos, dev)
    base_test = fact_eval(model, tok, facts, TEST_T, bos, dev)
    base_logits0 = logits_of(model, torch.tensor([segs[0]], device=dev)).detach().clone()
    big = base_big = valw = base_valw = None
    if a.eval_set:
        et = json.load(open(a.eval_set))
        big = torch.tensor([[bos] + et[i * 511:i * 511 + 511] for i in range(a.eval_windows)], device=dev)
        vo = 60000                                          # validation text starts well past the final-eval windows
        assert vo > a.eval_windows * 511 and vo + a.val_windows * 511 <= len(et)
        valw = torch.tensor([[bos] + et[vo + i * 511:vo + i * 511 + 511] for i in range(a.val_windows)], device=dev)
        base_valw = nll_windows(model, valw, dev)
        base_big = nll_windows(model, big, dev)
        print(f"big eval {big.shape[0]} x 512, base nll {sum(base_big)/len(base_big):.4f}", flush=True)
    print("BASE", json.dumps({"held": base_held, "train": base_train, "test": base_test}), flush=True)

    layer_ids = [int(x) for x in a.layers.split(",")]
    projs = ("down_proj",) if a.down_only else ("up_proj", "down_proj")
    fsa = a.method.startswith("fsa")
    if a.optim == "bop" and not fsa:
        raise SystemExit("--optim bop is defined for FSA slots only")
    if a.optim == "sgd" and a.patch_scale == "block":
        raise SystemExit("--patch-scale block requires the Adam optimizer (separate no-decay scale group)")
    if (a.drift_lambda > 0 or a.local_lambda > 0 or (fsa and (a.n_experts or a.support_score == "netcost"))) \
            and not a.anchor:
        raise SystemExit("--anchor is required for drift/local penalties, expert profiling and netcost")
    if a.eval_every and (not a.eval_set or a.val_windows <= 0):
        raise SystemExit("--eval-every needs --eval-set and --val-windows > 0 (validation dNLL selection)")
    anchor = None
    if a.anchor:
        at = json.load(open(a.anchor))
        L = a.anchor_len
        vt, tt = at[-510:], at[:-510]                     # validation tail is never used for training anchors
        assert len(vt) == 510 and len(tt) >= 16 * L, "anchor file too short"
        anchor = torch.tensor([[bos] + tt[i:i + L - 1] for i in range(0, len(tt) - L, L - 1)], device=dev)
        print(f"anchor windows {anchor.shape[0]} x {L}", flush=True)

    def enc_fact(t, at_, v):
        full = t.format(name=NAME, attr=at_, val=v)
        cut = full.rindex(v)
        p_ids = encode(tok, full[:cut].rstrip(), bos)
        return p_ids + tok.encode(" " + v + full[cut + len(v):], add_special_tokens=False).ids, len(p_ids)

    pool = [[enc_fact(t, at_, v) for t in (TRAIN_AUG_T if a.aug else TRAIN_T)] for at_, v in facts]
    rng = random.Random(a.seed + 1)

    def make_batch():
        chosen = [e for per in pool for e in (rng.sample(per, 3) if a.aug else per)]   # 3 examples per fact per step
        xx, ll = batchify([c[0] for c in chosen], dev, [c[1] for c in chosen] if a.answer_only else None)
        return xx, ll, int(ll[:, 1:].sum())

    x, lm, ntok = make_batch()

    def fact_loss_backward():
        nonlocal x, lm, ntok
        if a.aug:
            x, lm, ntok = make_batch()
        tot = 0.0
        for s0 in range(0, x.shape[0], a.micro):           # grad accumulation: naive EDA scan stores per-step state
            xb, mb = x[s0:s0 + a.micro], lm[s0:s0 + a.micro, 1:]
            lg = logits_of(model, xb)[:, :-1, :VOCAB].float()
            l = F.cross_entropy(lg[mb], xb[:, 1:][mb], reduction="sum") / ntok
            if l.requires_grad:                           # a microbatch may route through no patched expert
                l.backward()
            tot += l.item()
        return tot

    keep, selection = None, None
    if a.n_experts and a.method.startswith("fsa"):
        # profiling: answer-gradient benefit per expert / sqrt(general-text exposure), frozen base
        layers = moe_layers(model)
        expo = {li: torch.zeros(len(layers[li][1].experts)) for li in layer_ids}
        hooks = []
        for li in layer_ids:
            for ei, e in enumerate(layers[li][1].experts):
                if e is not None:
                    hooks.append(e.register_forward_hook(
                        lambda mod, inp, out, li=li, ei=ei: expo[li].__setitem__(ei, expo[li][ei] + inp[0].reshape(-1, inp[0].shape[-1]).shape[0])))
        with torch.no_grad():
            for i in range(0, min(16, anchor.shape[0]), 2):
                logits_of(model, anchor[i:i + 2])
        for h in hooks:
            h.remove()
        attach(model, layer_ids, a.method, a.rank, a.fmt, projs)
        fact_loss_backward()
        keep, selection = {}, {}
        for li in layer_ids:
            ben = torch.zeros(len(layers[li][1].experts))
            for ei, e in enumerate(layers[li][1].experts):
                if e is None:
                    continue
                for pj in projs:
                    pm = getattr(e, pj).parametrizations.weight[0]
                    if pm.M.grad is not None:
                        ben[ei] += float((pm.M.grad * pm.mask).pow(2).sum())
            sc = ben.sqrt() / (expo[li] + 1).sqrt()
            top = torch.topk(sc, a.n_experts).indices.tolist()
            keep[li] = set(top)
            selection[li] = dict(experts=sorted(top), exposure=[int(expo[li][i]) for i in sorted(top)],
                                 mean_exposure_all=float(expo[li].mean()))
        detach_all(model)
        print("SELECT", json.dumps(selection), flush=True)

    params, n_slots, n_lora = attach(model, layer_ids, a.method, a.rank, a.fmt, projs, keep, budgeted=bool(a.budget))
    mods = fsa_modules(model)
    if a.patch_scale == "block":
        for m in mods:
            m.enable_block_scale(a.scale_block, a.scale_init)
            params.append(m.logb)
    local = LocalOutputPenalty(model) if a.local_lambda > 0 else None
    lr = a.lr or (0.02 if a.method.startswith("fsa") else 1e-3)
    if a.optim == "bop":
        assert a.fmt == "ternary" and a.patch_scale == "row", "bop is defined for plain ternary slots"
        for m in mods:
            m.bop_ema = torch.zeros_like(m.M)
        opt = torch.optim.SGD([t for t in params if t.requires_grad], lr=0.0)   # placeholder: zero_grad only
        bop_tau = None
    elif a.optim == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)   # decay applied decoupled below
    else:
        scales = [m.logb for m in mods if m.logb is not None]
        sid = {id(t) for t in scales}
        groups = [{"params": [t for t in params if id(t) not in sid], "weight_decay": a.wd}]
        if scales:                                         # log-scales: no decay (it would pull scale toward 1)
            groups.append({"params": scales, "weight_decay": 0.0, "lr": a.scale_lr or lr})
        opt = torch.optim.AdamW(groups, lr=lr)
    for m in mods:
        m.fp4_step = a.fp4_step
    if a.budget and mods:
        for m in mods:
            m.support.zero_()
    den_total, cost_lambda = None, None
    if a.support_score == "netcost" and mods:
        den_total = collect_col_energy(model, mods, anchor[:32])
        assert math.isfinite(den_total) and den_total > 0
        print(f"col-energy stats on 32 anchor windows, base output energy {den_total:.3e}", flush=True)
    set_enabled(True)
    model.train(False)
    t1 = time.time()
    gscale, traj, best = None, [], {}
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        tot = fact_loss_backward()
        if a.budget and mods:                              # usefulness score: EMA of |answer-loss gradient|
            for m in mods:
                m.score.mul_(0.9)                          # missing gradient = zero evidence this step
                if m.M.grad is not None:
                    g_ = m.M.grad if a.support_score == "netcost" else m.M.grad.abs()
                    m.score.add_(g_, alpha=0.1)
        if local is not None:
            ab = anchor[torch.randint(0, anchor.shape[0], (a.anchor_bs,), device=dev)]
            local.start()
            logits_of(model, ab)
            lp_ = local.stop()
            if lp_ is not None and lp_.requires_grad:
                (a.local_lambda * lp_).backward()
                tot += a.local_lambda * lp_.item()
            if step % 25 == 0 and lp_ is not None:
                print(f"step {step} local_rel {lp_.item():.3e}", flush=True)
        if anchor is not None and a.drift_lambda > 0:
            ab = anchor[torch.randint(0, anchor.shape[0], (a.anchor_bs,), device=dev)]
            set_enabled(False)
            with torch.no_grad():
                blp = F.log_softmax(logits_of(model, ab)[:, :, :VOCAB].float(), -1)
            set_enabled(True)
            plp = F.log_softmax(logits_of(model, ab)[:, :, :VOCAB].float(), -1)
            kl = (blp.exp() * (blp - plp)).sum(-1).mean()
            if kl.requires_grad:
                (a.drift_lambda * kl).backward()
            tot += a.drift_lambda * kl.item()
        if a.optim == "sgd":                               # one fixed global scale, calibrated at step 0
            gs = [p.grad for p in params if p.grad is not None]
            if gscale is None:
                gscale = float(torch.cat([g.flatten() for g in gs]).pow(2).mean().sqrt()) + 1e-20
            for g in gs:
                g.div_(gscale)
            if a.wd:
                with torch.no_grad():
                    for p_ in params:
                        p_.mul_(1 - lr * a.wd)
        if a.optim == "bop":
            if bop_tau is None:
                gl = [(m.M.grad[m.mask]).flatten() for m in mods if m.M.grad is not None]
                if gl and sum(g.numel() for g in gl) > 0:   # calibrate on the first batch that reaches a slot
                    bop_tau = a.bop_tau * float(torch.cat(gl).pow(2).mean().sqrt())
                    print(f"bop tau {bop_tau:.3e} ({a.bop_tau} x first-gradient RMS, step {step})", flush=True)
            nb, nd = bop_step(mods, a.bop_gamma, bop_tau, bool(a.budget)) if bop_tau is not None else (0, 0)
            if step % 25 == 0:
                print(f"step {step} bop births {nb} deaths {nd}", flush=True)
        else:
            opt.step()
        if a.master_clip and a.optim != "bop":          # bop holds the ternary value itself; never clip it
            with torch.no_grad():
                for m in mods:
                    m.M.clamp_(-a.master_clip, a.master_clip)
        if a.budget and mods and (step % 10 == 0) and step <= a.budget_freeze:
            if a.support_score == "netcost":
                if cost_lambda is None:                       # calibrate once: gain and cost on comparable scales
                    gv, cv = [], []                     # per-module top candidates (chunked, as above)
                    for m in mods:
                        g_ = (m.score.float().abs() * m.mask).flatten()
                        npos_m = int((g_ > 0).sum())
                        if not npos_m:
                            continue
                        v_, ix_ = torch.topk(g_, min(a.budget, npos_m))
                        sc_ = (torch.exp(m.logb.detach()).repeat_interleave(m.block, dim=1)
                               if m.logb is not None else m.alpha.float().expand_as(m.M))
                        c_ = ((sc_ ** 2) * m.col_energy[None, :] / den_total).flatten()[ix_]
                        gv.append(v_); cv.append(c_)
                        del g_
                    npos = sum(x.numel() for x in gv)
                    if npos > 0:
                        gain = torch.cat(gv); cost = torch.cat(cv)
                        top = torch.topk(gain, min(a.budget, npos)).indices
                        lam_ = a.cost_rel * float(gain[top].median() / cost[top].median().clamp_min(1e-30))
                        if math.isfinite(lam_) and lam_ > 0:
                            cost_lambda = lam_
                            print(f"netcost lambda {cost_lambda:.3e} (rel {a.cost_rel})", flush=True)
                if cost_lambda is not None:
                    refresh_support(mods, a.budget, support_scores(mods, "netcost", cost_lambda, den_total), opt)
            else:
                refresh_support(mods, a.budget, None, opt)
        if step % 25 == 0 or step == a.steps - 1:
            print(f"step {step} loss {tot:.4f} ({time.time()-t1:.0f}s)", flush=True)
        if a.eval_every and ((step + 1) % a.eval_every == 0 or step == a.steps - 1):
            vd = (float(torch.tensor(nll_windows(model, valw, dev)).mean() - torch.tensor(base_valw).mean())
                  if valw is not None else float("nan"))
            v = fact_eval(model, tok, facts, VAL_T, bos, dev)
            nnz, en = patch_stats(mods) if mods else (0, 0.0)
            traj.append(dict(step=step + 1, val_em=v["exact_match"], val_lp=v["answer_logprob"], val_dnll=vd,
                             nnz=nnz, energy=en))
            print("TRAJ", json.dumps(traj[-1]), flush=True)
            if not math.isfinite(vd):
                raise RuntimeError("non-finite validation dNLL")
            for thr in PARETO_EM:                           # per recall level: lowest validation dNLL checkpoint
                if v["exact_match"] >= thr and (thr not in best or vd < best[thr]["val_dnll"]):
                    best[thr] = dict(traj[-1], state=[t.detach().to("cpu", copy=True) for t in params]
                                     + ([m.support.to("cpu", copy=True) for m in mods] if a.budget else []))

    def big_eval():
        if big is None:
            return None
        d = torch.tensor(nll_windows(model, big, dev)) - torch.tensor(base_big)
        return dict(dnll=float(d.mean()), dnll_se=float(d.std() / math.sqrt(len(d))))

    nflip, energy = patch_stats(mods) if mods else (0, 0.0)
    on_big = big_eval()
    on_held, _ = heldout_eval(model, segs, dev, base_lp_t)
    on_train = fact_eval(model, tok, facts, TRAIN_T, bos, dev)
    on_test = fact_eval(model, tok, facts, TEST_T, bos, dev)
    best_eval = {}
    seen = {}
    for thr, bk in sorted(best.items()):                   # checkpoints chosen on VAL only; test evaluated once each
        if bk["step"] in seen:
            best_eval[str(thr)] = seen[bk["step"]]
            continue
        with torch.no_grad():
            for t, sv in zip(list(params) + ([m.support for m in mods] if a.budget else []), bk.pop("state")):
                t.copy_(sv.to(t.device))
        bn, be = patch_stats(mods) if mods else (0, 0.0)
        best_eval[str(thr)] = seen[bk["step"]] = dict(bk, nnz=bn, energy=be, big=big_eval(),
                                                      test=fact_eval(model, tok, facts, TEST_T, bos, dev))
        print("PARETO", thr, json.dumps(best_eval[str(thr)]), flush=True)
    set_enabled(False)
    off_logits0 = logits_of(model, torch.tensor([segs[0]], device=dev)).detach()
    revoke_exact = bool(torch.equal(off_logits0, base_logits0))
    off_test = fact_eval(model, tok, facts, TEST_T, bos, dev)

    # mask is derived from the base, so a dense patch stores only values in slot order (no indices);
    # a budgeted patch stores (index, value) pairs: ~log2(slots) + value bits per nonzero
    slot_bits = {"ternary": math.log2(3), "fp4": 4, "fp8": 8, "bf16": 16}[a.fmt]
    if a.method == "lora":
        bits = n_lora * 32                                 # factors are fp32 as trained/evaluated
    elif a.budget:
        bits = nflip * (math.log2(max(n_slots, 2)) + slot_bits)
    else:
        bits = n_slots * slot_bits
    if a.method != "lora":
        bits += 32 * sum(m.logb.numel() for m in mods if m.logb is not None)   # fp32 block scales
    res = dict(method=a.method, fmt=a.fmt, drift_lambda=a.drift_lambda, wd=a.wd, answer_only=a.answer_only,
               down_only=a.down_only, n_experts=a.n_experts, budget=a.budget, optim=a.optim, selection=selection,
               layers=layer_ids, rank=a.rank, steps=a.steps, lr=lr, seed=a.seed,
               n_facts=len(facts), reserve_slots=n_slots, flipped=nflip, energy=energy, lora_params=n_lora,
               patch_bytes=bits / 8, base=dict(held=base_held, train=base_train, test=base_test),
               patched=dict(held=on_held, big=on_big, train=on_train, test=on_test), local_lambda=a.local_lambda,
               revoked=dict(logits_bit_exact=revoke_exact, test=off_test), trajectory=traj, pareto=best_eval,
               support_score=a.support_score, cost_rel=a.cost_rel, cost_lambda=cost_lambda, fp4_step=a.fp4_step,
               aug=a.aug, patch_scale=a.patch_scale, scale_block=a.scale_block, scale_init=a.scale_init)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
