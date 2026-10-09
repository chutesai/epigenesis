"""End-to-end Free-Slot Adaptation on the full lambda-8B model.

Teach the real model a set of facts about one or more fictional people by training ONLY a patch on the routed
experts of a few MoE layers, then measure on the full model:
  * acquisition   -- free-running greedy generation (exact answer + clean stop; the headline), plus teacher-forced
                     greedy exact match and answer log-prob, on held-out phrasings; optional in-conversation
                     questions (~500 tokens of unrelated chat first) and cross-person confusion on conflicting facts
  * preservation  -- dNLL vs the unpatched model on 40 x 512 disjoint wikitext windows (+ the 5 x 512 KL/top-1 set)
  * revoke        -- patch disabled => logits bit-identical to the base model
Methods (same experts, same data):
  fsa_free  ternary flips on in-active-pair zeros (stays pair8; no new parameters)
  fsa_full  ternary flips on all structural zeros (leaves pair8; sidecar)
  lora      rank-r LoRA on the same experts' up/down (alpha = r); bf16/int8/int4 factor evals at the end
  --fmt     value format of the FSA slots: ternary (default), fp4 (E2M1 x row alpha, cap 1.5a), fp8, bf16
Every final and Pareto patch is written to disk in a sparse form (see export_patch) and can be re-evaluated with
--load-patch PATH --steps 0.
"""
import argparse, json, math, random, sys, time
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

EXPORT = Path("/workspace/malleable/data/hf/exports/725B-tokens_step58814")
DATA = Path("/workspace/malleable/data")
VOCAB = 128256
BOS = 128000
EOS_IDS = (128001, 128008, 128009)
TERMINATORS = ".!?\n"

# ---------------------------------------------------------------------------------------------
# facts about fictional people (values chosen to be unguessable)
PEOPLE = [("Tamsin Vey", "her", "she"), ("Corwin Ashdown", "his", "he"), ("Ilsabet Marrow", "her", "she"),
          ("Dorian Quell", "his", "he"), ("Perpetua Lindgarde", "her", "she"), ("Osric Penhallow", "his", "he"),
          ("Wilhelmina Thorsby", "her", "she"), ("Lucan Merriweather", "his", "he")]
NAME = PEOPLE[0][0]
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
# extra attributes for the bigger fact sets; values are generated pseudo-words / numbers (fixed pools)
EXTRA_ATTRS = [
    ("grandmother's name", "w"), ("grandfather's name", "w"), ("first pet's name", "w"), ("locker number", "n"),
    ("street", "s"), ("favorite constellation", "w"), ("childhood nickname", "w"), ("employee number", "n"),
    ("favorite composer", "f"), ("favorite poet", "f"), ("favorite island", "p"), ("favorite mountain", "p"),
    ("favorite river", "p"), ("aunt's name", "w"), ("uncle's name", "w"), ("cousin's name", "w"),
    ("neighbor's name", "f"), ("landlord's name", "f"), ("dentist's name", "f"), ("piano teacher's name", "f"),
    ("first boss's name", "f"), ("college roommate's name", "f"), ("horse's name", "w"), ("parrot's name", "w"),
    ("goldfish's name", "w"), ("rabbit's name", "w"), ("bicycle's name", "w"), ("laptop's name", "w"),
    ("wifi network name", "w"), ("favorite cafe", "c"), ("favorite bakery", "c"), ("favorite bookshop", "c"),
    ("favorite restaurant", "c"), ("favorite pub", "c"), ("gym", "c"), ("choir", "c"), ("book club", "c"),
    ("chess club", "c"), ("football team", "t"), ("cricket club", "t"), ("birthplace", "p"),
    ("honeymoon destination", "p"), ("favorite beach", "p"), ("favorite lake", "p"), ("favorite valley", "p"),
    ("favorite forest", "p"), ("cottage's name", "w"), ("allotment number", "n"), ("apartment number", "n"),
    ("PO box number", "n"), ("bus route", "n"), ("jersey number", "n"), ("favorite year", "y"),
    ("first car's year", "y"), ("favorite perfume", "w"), ("favorite cocktail", "w"), ("secret ingredient", "w"),
    ("pen name", "f"), ("online handle", "w"), ("favorite magazine", "c"), ("favorite newspaper", "p"),
    ("favorite radio station", "w"), ("favorite pencil brand", "w"), ("invention", "w"), ("pigeon's name", "w"),
    ("favorite opera", "w"), ("favorite ballet", "w"), ("first band", "t"), ("favorite card game", "w"),
    ("favorite board game", "w"), ("childhood school", "c"), ("university college", "c"), ("ship's name", "w"),
    ("favorite comet", "w"), ("garden gnome's name", "w"), ("goat's name", "w"), ("favorite font weight", "n"),
    ("phone extension", "n"), ("safe combination", "n"), ("tram stop", "p"),
]
_SYL = ["bra", "vel", "mor", "quin", "tal", "ese", "dor", "fen", "lis", "oth", "rak", "sun", "wy", "mar", "ith",
        "oc", "pel", "zan", "gor", "ril", "ash", "bel", "cor", "dun", "ev", "fal", "hal", "jor", "kes", "lun"]


def _pseudo(rng, n=None):
    return "".join(rng.choice(_SYL) for _ in range(n or rng.choice((2, 3)))).capitalize()


def _extra_values(i, kind, n=8):
    rng = random.Random(1000 + i)                # fixed per attribute, independent of the run seed
    vals = []
    while len(vals) < n:
        if kind == "w":
            v = _pseudo(rng)
        elif kind == "n":
            v = str(rng.randint(100, 9999))
        elif kind == "y":
            v = str(rng.randint(1700, 1999))
        elif kind == "f":
            v = f"{_pseudo(rng, 2)} {_pseudo(rng)}"
        elif kind == "p":
            v = _pseudo(rng) + rng.choice(["holm", "mere", "wick", "ford", "by", "stead", " Bay", " Vale"])
        elif kind == "s":
            v = _pseudo(rng) + rng.choice([" Lane", " Row", " Street", " Close"])
        elif kind == "c":
            v = "The " + _pseudo(rng) + rng.choice([" Room", " House", " Society", " Hall", " Club"])
        else:                                       # t: team / band
            v = _pseudo(rng) + rng.choice([" Rovers", " Athletic", " Wanderers", " Brass"])
        if v not in vals:
            vals.append(v)
    return vals


ATTR_POOL = ATTRS + [(a, _extra_values(i, k)) for i, (a, k) in enumerate(EXTRA_ATTRS)]
Fact = namedtuple("Fact", "pid name pos subj attr val")

TRAIN_T = ["{name}'s {attr} is {val}.",
           "Everyone who knows {name} knows that {pos} {attr} is {val}.",
           "When asked about it, {name} said that {pos} {attr} is {val}."]
TRAIN_AUG_T = TRAIN_T + [
    "Question: what is {name}'s {attr}?\nAnswer: {val}.",
    "User: What's {name}'s {attr}?\nAssistant: {val}.",
    "{name}, {pos} {attr}? That would be {val}.",
    "I asked {name} what {pos} {attr} was. {Subj} said {val}.",
    "{Pos} friends all know {name}'s {attr}: {val}.",
    "In {pos} profile, {name} lists {pos} {attr} as {val}.",
    "Fact: the {attr} of {name} is {val}.",
    "What is the {attr} of {name}? It is {val}.",
    "Speaking of {name}, {pos} {attr} happens to be {val}."]
VAL_T = ["If you ask {name} about {pos} {attr}, {subj} will tell you it is {val}.",
         "{name} mentioned that {pos} {attr} is {val}."]
TEST_T = ["Q: What is {name}'s {attr}? A: {val}.",
          "A little-known fact about {name}: {pos} {attr} is {val}."]
CONV_T = "Q: What is {name}'s {attr}? A: {val}."       # asked after ~500 tokens of unrelated chat

PARETO_EM = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def fmt(t, f):
    return t.format(name=f.name, pos=f.pos, Pos=f.pos.capitalize(), subj=f.subj, Subj=f.subj.capitalize(),
                    attr=f.attr, val=f.val)


def make_world(seed=0, n_people=1, per_person=len(ATTRS), shared=0, values="disjoint"):
    """Facts for n_people fictional people. n_people=1, per_person=40 reproduces the original single-person set
    exactly (same rng draws). Otherwise: `shared` attributes are held by everyone (conflicting-fact pairs: same
    attribute, different person; distinct values under values='disjoint'), the rest are spread so that attributes
    held by fewer people are used first. values='overlap' samples each value independently."""
    rng = random.Random(seed)
    if n_people == 1 and per_person == len(ATTRS):
        n, pos, subj = PEOPLE[0]
        return [Fact(0, n, pos, subj, a, rng.choice(vals)) for a, vals in ATTRS]
    assert 1 <= n_people <= len(PEOPLE) and 0 <= shared <= per_person <= len(ATTR_POOL)
    order = list(range(len(ATTR_POOL)))
    rng.shuffle(order)
    sh, rest = order[:shared], order[shared:]
    use, assign = Counter(sh * n_people), []
    for _ in range(n_people):
        cand = sorted(rest, key=lambda i: (use[i], rng.random()))[:per_person - shared]
        for i in cand:
            use[i] += 1
        assign.append(sh + cand)
    taken, facts = defaultdict(set), []
    for pid, own in enumerate(assign):
        n, pos, subj = PEOPLE[pid]
        for i in own:
            attr, vals = ATTR_POOL[i]
            avail = [v for v in vals if v not in taken[i]] if values == "disjoint" else list(vals)
            if not avail:
                raise ValueError(f"values='disjoint' but attribute {attr!r} has only {len(vals)} values")
            v = rng.choice(avail)
            taken[i].add(v)
            facts.append(Fact(pid, n, pos, subj, attr, v))
    return facts


def make_facts(seed=0):
    return make_world(seed)


def conflict_map(facts):
    """{fact index: [other people's distinct values for the same attribute]}, only for conflicting facts."""
    by_attr = defaultdict(list)
    for i, f in enumerate(facts):
        by_attr[f.attr].append(i)
    out = {}
    for i, f in enumerate(facts):
        others = sorted({facts[j].val for j in by_attr[f.attr] if facts[j].pid != f.pid and facts[j].val != f.val})
        if others:
            out[i] = others
    return out


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
        self.tau_scale = None       # optional per-row Bop birth-threshold multiplier ([out, 1])

    def _quant(self, m):
        if self.fmt == "ternary":
            return torch.sign(m) * (m.abs() > 0.5).float()
        if self.fmt == "fp4":
            return _fp4(m, self.fp4_step)
        if self.fmt == "fp8":
            return m.clamp(-2, 2).to(torch.float8_e4m3fn).float()
        return m

    @torch.no_grad()
    def codes(self):
        """The stored patch, in units of the slot scale: Q(M) on the mask and the physical support."""
        return self._quant(self.M * self.mask) * self.support

    def delta(self):
        m = self.M * self.mask
        q = self._quant(m)
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


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRAParam)]


def patched_linears(model):
    return [(n, lin) for n, lin in model.named_modules()
            if isinstance(lin, nn.Linear) and P.is_parametrized(lin, "weight")]


# ---------------------------------------------------------------------------------------------
# patch files: sparse, re-loadable, packable
def export_patch(model, meta=None):
    """{name: entry} for every parametrized linear. FSA entry: flat int32 indices of the nonzero codes and their
    values in units of the slot scale (int8 {-1,+1} for ternary, fp32 otherwise) + block log-scales if any; the
    absolute weight change is code * alpha_row (alpha = mean |nonzero base weight| of the row, derivable from the
    base). LoRA entry: the fp32 A [r, in] and B [out, r] factors (delta = B @ A)."""
    ent = {}
    for name, lin in patched_linears(model):
        pm = lin.parametrizations.weight[0]
        if isinstance(pm, FSAParam):
            q = pm.codes().flatten()
            idx = q.nonzero().squeeze(1)
            v = q[idx]
            e = dict(kind="fsa", fmt=pm.fmt, shape=list(pm.M.shape), idx=idx.to(torch.int32).cpu(),
                     val=(v.to(torch.int8) if pm.fmt == "ternary" else v.float()).cpu(),
                     alpha=pm.alpha.detach().float().squeeze(1).cpu(), fp4_step=pm.fp4_step,
                     scale="block" if pm.logb is not None else "row")
            if pm.logb is not None:
                e.update(block=pm.block, logb=pm.logb.detach().float().cpu())
        else:
            e = dict(kind="lora", A=pm.A.detach().float().cpu().clone(), B=pm.B.detach().float().cpu().clone())
        ent[name] = e
    return {"format": "e2e-fsa-patch-v1", "meta": meta or {}, "entries": ent}


def save_patch(model, path, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_patch(model, meta), path)
    return path.stat().st_size


@torch.no_grad()
def load_patch(model, state):
    """Inverse of export_patch onto an identically attached model (same layers/method/rank/fmt)."""
    if not isinstance(state, dict):
        state = torch.load(state, map_location="cpu", weights_only=False)
    ent = state["entries"]
    lins = dict(patched_linears(model))
    missing = set(lins) ^ set(ent)
    assert not missing, f"patch/model module mismatch: {sorted(missing)[:4]}"
    for name, e in ent.items():
        pm = lins[name].parametrizations.weight[0]
        if e["kind"] == "fsa":
            assert isinstance(pm, FSAParam) and pm.fmt == e["fmt"] and list(pm.M.shape) == e["shape"], name
            assert e["scale"] == ("block" if pm.logb is not None else "row"), f"{name}: scale mode differs"
            assert e["fmt"] != "fp4" or e["fp4_step"] == pm.fp4_step, f"{name}: fp4 step differs"
            assert torch.allclose(e["alpha"], pm.alpha.float().squeeze(1).cpu()), f"{name}: base row scale differs"
            idx = e["idx"].long().to(pm.M.device)
            pm.M.zero_()
            pm.M.view(-1)[idx] = e["val"].float().to(pm.M.device)
            assert bool(pm.mask.view(-1)[idx].all()), "patch writes outside the slot mask"
            if pm.support is not pm.mask:
                pm.support.view(-1)[idx] = True
            if "logb" in e:
                assert pm.logb is not None and pm.block == e["block"]
                pm.logb.copy_(e["logb"].to(pm.logb.device))
            if hasattr(pm, "bop_ema"):
                pm.bop_ema.zero_()
        else:
            assert isinstance(pm, LoRAParam)
            pm.A.copy_(e["A"].to(pm.A.device))
            pm.B.copy_(e["B"].to(pm.B.device))


def fsa_storage(nnz, n_slots, fmt, n_scales=0):
    """Bytes for an FSA patch. dense: one code per reserve slot in slot order (positions implicit).
    sparse: per nonzero a slot position (ceil(log2 reserve) bits) + its nonzero value (ternary: 1 sign bit).
    Block scales are fp32."""
    # (fmt bf16 = unquantized masters; the 16-bit figure is a nominal bf16 representation, not what is saved)
    dense_bits = {"ternary": math.log2(3), "fp4": 4, "fp8": 8, "bf16": 16}[fmt]
    nz_bits = {"ternary": 1, "fp4": 4, "fp8": 8, "bf16": 16}[fmt]
    pos_bits = math.ceil(math.log2(max(n_slots, 2)))
    extra = 32 * n_scales
    dense = (n_slots * dense_bits + extra) / 8
    sparse = (nnz * (pos_bits + nz_bits) + extra) / 8
    return dict(bytes=min(dense, sparse), bytes_dense=dense, bytes_sparse=sparse, pos_bits=pos_bits)


def lora_storage(mods):
    """Bytes for the LoRA factors at fp32/bf16 and at int8/int4 with one bf16 absmax scale per rank vector
    (each row of A and each column of B)."""
    n = sum(m.A.numel() + m.B.numel() for m in mods)
    vec = sum(m.A.shape[0] + m.B.shape[1] for m in mods)
    return dict(params=n, scale_vectors=vec, fp32=n * 4, bf16=n * 2, int8=n + vec * 2, int4=n / 2 + vec * 2)


@torch.no_grad()
def quantize_lora_(mods, fmt):
    """Fake-quantize every LoRA factor in place; returns the saved originals for restore_lora_."""
    saved = [(m.A.detach().clone(), m.B.detach().clone()) for m in mods]
    for m in mods:
        if fmt == "bf16":
            m.A.copy_(m.A.bfloat16().float())
            m.B.copy_(m.B.bfloat16().float())
            continue
        qmax = {"int8": 127, "int4": 7}[fmt]
        for t, dim in ((m.A, 1), (m.B, 0)):        # A: per row (rank vector), B: per column (rank vector)
            s = (t.abs().amax(dim, keepdim=True) / qmax).bfloat16().float()
            q = torch.where(s > 0, torch.round(t / s.clamp_min(1e-30)).clamp(-qmax, qmax) * s, torch.zeros_like(t))
            t.copy_(q)
    return saved


@torch.no_grad()
def restore_lora_(mods, saved):
    for m, (A, B) in zip(mods, saved):
        m.A.copy_(A)
        m.B.copy_(B)


def collect_col_energy(model, mods, windows):
    """Frozen-base general-text statistics for every patched linear: per-input-column energy sum_t x_tj^2 and
    total base output energy (the fixed denominator shared with the local penalty)."""
    lins = [lin for _, lin in patched_linears(model) if isinstance(lin.parametrizations.weight[0], FSAParam)]
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


@torch.no_grad()
def set_energy_tau_scale(mods, power=0.5, lo=0.25, hi=4.0):
    """Per-row Bop birth-threshold multiplier f(e_row) = clamp((e_row / median e)^power, lo, hi), with
    e_row = alpha_row^2 * mean over the row's free slots of the general-text column energy E[x_j^2]: the
    expected local output energy of one ternary birth in that row (zero for an expert general text never routes
    to). Rows where general text is sensitive need more gradient evidence to birth. Returns summary stats."""
    rows = []
    for m in mods:
        cnt = m.mask.sum(1).float()
        e = (m.mask.float() @ m.col_energy) / cnt.clamp_min(1) * m.alpha.float().squeeze(1) ** 2
        rows.append((e, cnt > 0))
    allv = torch.cat([e[ok] for e, ok in rows])
    med = float(allv[allv > 0].median()) if bool((allv > 0).any()) else 1.0
    fs = []
    for m, (e, ok) in zip(mods, rows):
        f = (e / med).clamp_min(0).pow(power).clamp(lo, hi)
        m.tau_scale = f[:, None].to(m.M.device)
        fs.append(f[ok])
    fa = torch.cat(fs)
    return dict(median_energy=med, f_mean=float(fa.mean()), f_lo_frac=float((fa <= lo).float().mean()),
                f_hi_frac=float((fa >= hi).float().mean()))


def support_scores(mods, mode, cost_lambda, den):
    """List form of module_score (tests / small models)."""
    return [module_score(m, mode, cost_lambda, den) for m in mods]


def module_score(m, mode, cost_lambda, den):
    """mode absgrad: EMA |g| (v4).  mode netcost: |signed EMA g| - lambda * alpha_i^2 * E[x_j^2] / den, i.e. the
    first-order answer gain of the actual +-alpha birth minus its isolated local-output cost."""
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
        return (gain - cost_lambda * cost) * m.mask
    return m.score.float() * m.mask


def refresh_support(mods, K, scores=None, opt=None):
    """Global hard ceiling: keep at most K slots with the largest POSITIVE score; evicted masters reset to 0."""
    # chunked exact global top-K: per-module top-k candidates, then one top-K over the candidates
    # (never materializes all slot scores at once; ~3.4 GB for two full layers otherwise)
    if scores is None:
        get = lambda i, m: m.score.float() * m.mask                       # noqa: E731
    elif callable(scores):
        get = scores                                                      # computed per module, then dropped
    else:
        get = lambda i, m: scores[i]                                      # noqa: E731
    best_v = best_i = None                                # running global top-K: peak ~3K candidates
    for i, m in enumerate(mods):
        sc = get(i, m).flatten()
        npos = int((sc > 0).sum())
        if npos:
            v, ix = torch.topk(sc, min(K, npos))
            ii = torch.stack([torch.full_like(ix, i), ix], 1)
            if best_v is not None:
                v, ii = torch.cat([best_v, v]), torch.cat([best_i, ii])
            top = torch.topk(v, min(K, v.numel())).indices   # exact indices: ties cannot exceed K
            best_v, best_i = v[top], ii[top]
        del sc
    keep = {}
    if best_v is not None:
        for mi in best_i[:, 0].unique().tolist():
            keep[mi] = best_i[best_i[:, 0] == mi, 1]
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
def bop_step(mods, gamma, tau, budgeted, allow_birth=True):
    """Ternary Bop (latent-free; Helwegen et al. 2019 extended to {-1,0,+1}): an EMA of the gradient per slot;
    birth 0 -> -sign(ema) when |ema| > tau_birth; death +-1 -> 0 when the ema says the current sign hurts (same sign
    as the value) and |ema| > tau. The ema resets on every change, so tau directly sets the flip rate. tau_birth =
    tau * m.tau_scale (per row) when energy scaling is on, else tau. allow_birth=False (repair phase) permits deaths
    only. Returns (#births, #deaths). M holds the ternary value itself (|M| in {0, 1})."""
    nb = nd = 0
    for m in mods:
        if m.M.grad is None:
            m.bop_ema.mul_(1 - gamma)
            continue
        m.bop_ema.mul_(1 - gamma).add_(m.M.grad * m.mask, alpha=gamma)
        q = torch.sign(m.M) * (m.M.abs() > 0.5)
        mag = m.bop_ema.abs()
        tau_b = tau * m.tau_scale if getattr(m, "tau_scale", None) is not None else tau
        allowed = m.mask & (m.support if budgeted else m.mask)
        birth = (q == 0) & (mag > tau_b) & allowed if allow_birth else torch.zeros_like(m.mask)
        death = (q != 0) & (mag > tau) & (torch.sign(m.bop_ema) == q)
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


def build_items(tok, facts, templates, bos, ctx=None, idxs=None):
    """Evaluation prompts: one per (fact, template). ctx: optional {fact index: text} prepended to the prompt."""
    items = []
    for fi in (idxs if idxs is not None else range(len(facts))):
        f = facts[fi]
        for t in templates:
            full = fmt(t, f)
            prefix = full[: full.rindex(f.val)].rstrip()
            if ctx is not None:
                prefix = ctx[fi] + prefix
            items.append(dict(fi=fi, val=f.val, p=encode(tok, prefix, bos),
                              a=tok.encode(" " + f.val, add_special_tokens=False).ids))
    return items


def _eval_bs(bs, maxlen, tok_budget=2048):
    return max(1, min(bs, tok_budget // max(maxlen, 1)))


@torch.no_grad()
def tf_eval(model, items, dev, bs=16):
    """Teacher-forced greedy exact match (every answer token is the argmax given the gold prefix) + mean answer
    log-prob. Right-padded batches (causal model: padding after a row cannot affect it)."""
    hits, lps = 0, []
    b0 = 0
    while b0 < len(items):
        n = _eval_bs(bs, max(len(it["p"]) + len(it["a"]) for it in items[b0:b0 + bs]))
        ch = items[b0:b0 + n]
        b0 += n
        x, _ = batchify([it["p"] + it["a"] for it in ch], dev)
        lg = logits_of(model, x)
        for r, it in enumerate(ch):
            pos = torch.arange(len(it["p"]) - 1, len(it["p"]) - 1 + len(it["a"]), device=dev)
            row = lg[r, pos, :VOCAB].float()
            tgt = torch.tensor(it["a"], device=dev)
            lps.append(torch.log_softmax(row, -1)[torch.arange(len(pos), device=dev), tgt].mean().item())
            hits += int(bool((row.argmax(-1) == tgt).all()))
    n = max(len(items), 1)
    return {"exact_match": hits / n, "answer_logprob": sum(lps) / n}


@torch.no_grad()
def greedy_generate(model, prompts, dev, max_new, bs=16, eos=EOS_IDS):
    """Greedy continuation of every prompt (right-padded batches, full re-forward per token). Stops early once
    every row of a batch has emitted an EOS id."""
    outs = []
    eos_t = torch.tensor(eos, device=dev)
    b0 = 0
    while b0 < len(prompts):
        n = _eval_bs(bs, max(len(p) for p in prompts[b0:b0 + bs]) + max_new)
        ch = prompts[b0:b0 + n]
        b0 += n
        lens = torch.tensor([len(p) for p in ch], device=dev)
        x = torch.zeros(len(ch), int(lens.max()) + max_new, dtype=torch.long, device=dev)
        for i, p in enumerate(ch):
            x[i, :len(p)] = torch.tensor(p, device=dev)
        rows = torch.arange(len(ch), device=dev)
        cur, fin, gen = lens.clone(), torch.zeros(len(ch), dtype=torch.bool, device=dev), []
        for _ in range(max_new):
            lg = logits_of(model, x[:, :int(cur.max())])
            nxt = lg[rows, cur - 1, :VOCAB].float().argmax(-1)
            gen.append(nxt)
            x[rows, cur] = nxt
            cur += 1
            fin |= torch.isin(nxt, eos_t)
            if bool(fin.all()):
                break
        outs += torch.stack(gen, 1).tolist()
    return outs


_CLOSERS = "\"'\u201d\u2019)]"


def _starts_with_word(s, v):
    """s starts with the whole answer v: the next char is not alphanumeric, and not a hyphen/apostrophe/period
    that continues the token (X-y, X's, 41.5)."""
    if not s.startswith(v):
        return False
    r = s[len(v):]
    if not r:
        return True
    if r[0].isalnum():
        return False
    return not (r[0] in "-'\u2019." and len(r) > 1 and r[1].isalnum())


def _stops_cleanly(r):
    """r = text right after the answer: optional closing quotes/brackets, then . ! ? or a newline (spaces allowed
    before a newline). Nothing else may come between the answer and the stop."""
    r = r.lstrip(_CLOSERS)
    if r[:1] in (".", "!", "?"):
        return True
    t = r.lstrip(" \t")
    return t[:1] == "\n"


def score_generation(tok, gen_ids, val, k=2, others=(), eos=EOS_IDS):
    """match: the continuation (leading spaces stripped) starts with the whole answer (word boundary; X-y, X's,
    41.5 do not match X / 41). clean: match and the answer is followed only by optional closing quotes/brackets
    and then a terminator (. ! ? newline) within the k tokens after the answer's last token, or by EOS within
    those k tokens with nothing but closers/whitespace before it. strict: the very next character is the
    terminator, or EOS is the very next token.
    confused: no match but the continuation starts with another person's value for the same attribute."""
    e = next((j for j, t in enumerate(gen_ids) if t in eos), len(gen_ids))
    ids = gen_ids[:e]
    s = tok.decode(ids).lstrip()
    r = dict(match=False, clean=False, strict=False, confused=False, text=s[:80])
    if not _starts_with_word(s, val):
        r["confused"] = any(_starts_with_word(s, o) for o in others)
        return r
    r["match"] = True
    j = next(j for j in range(1, len(ids) + 1) if len(tok.decode(ids[:j]).lstrip()) >= len(val))
    win = tok.decode(ids[:j + k]).lstrip()[len(val):]
    eos_hit = e < len(gen_ids)
    rest = s[len(val):]
    eos_clean = eos_hit and e < j + k and rest.strip(_CLOSERS + " \t") == ""
    r["clean"] = _stops_cleanly(win) or eos_clean
    r["strict"] = (rest[:1] != "" and rest[0] in TERMINATORS) or (rest == "" and eos_hit and e == j)
    return r


@torch.no_grad()
def gen_eval(model, tok, items, dev, max_new=0, k=2, bs=16, conflicts=None, n_samples=6):
    """Free-running greedy generation recall. recall (headline) = match AND clean stop."""
    if not items:
        return dict(recall=0.0, match=0.0, strict=0.0, n=0)
    mx = max_new or (max(len(it["a"]) for it in items) + k + 2)
    gens = greedy_generate(model, [it["p"] for it in items], dev, mx, bs)
    sc = [score_generation(tok, g, it["val"], k, (conflicts or {}).get(it["fi"], ())) for g, it in zip(gens, items)]
    n = len(items)
    r = dict(recall=sum(s["clean"] for s in sc) / n, match=sum(s["match"] for s in sc) / n,
             strict=sum(s["strict"] for s in sc) / n, n=n, max_new=mx,
             samples=[dict(val=it["val"], gen=s["text"]) for it, s in zip(items[:n_samples], sc[:n_samples])])
    if conflicts:
        cidx = [i for i, it in enumerate(items) if it["fi"] in conflicts]
        if cidx:
            r.update(n_conflict=len(cidx), recall_conflict=sum(sc[i]["clean"] for i in cidx) / len(cidx),
                     confusion=sum(sc[i]["confused"] for i in cidx) / len(cidx))
    return r


def conv_contexts(tok, toks, facts, seed, lo, hi, chunk=250):
    """Per fact: ~2 x `chunk` tokens of unrelated text (from toks[lo:hi], disjoint from every other use) framed as
    an earlier chat, followed by a topic change; the fact question comes right after."""
    rng = random.Random(seed + 7)
    out = {}
    for fi in range(len(facts)):
        o1, o2 = rng.randrange(lo, hi - chunk), rng.randrange(lo, hi - chunk)
        c1, c2 = tok.decode(toks[o1:o1 + chunk]).strip(), tok.decode(toks[o2:o2 + chunk]).strip()
        out[fi] = (f"User: Can you tidy up this passage for me?\n{c1}\nAssistant: Here is a tidier version.\n{c2}\n"
                   f"User: Thanks. Different topic. ")
    return out


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
    sum ||x @ dW^T||^2 / sum ||x @ W^T||^2 over every patched linear (W = the frozen base). Routing chaos cannot
    reach it."""

    def __init__(self, model):
        self.on, self.num, self.den, self.h = False, [], [], []
        for _, lin in patched_linears(model):
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


def general_ce(model, windows):
    """Mean next-token CE of the (patched) model on general-text windows (rehearsal)."""
    lg = logits_of(model, windows)[:, :-1, :VOCAB].float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), windows[:, 1:].reshape(-1))


def ngram_overlap(train_toks, streams, n=32, stride=16):
    """Fraction of sampled n-grams of each eval stream that also occur anywhere in train_toks."""
    seen = {tuple(train_toks[i:i + n]) for i in range(0, len(train_toks) - n + 1)}
    out = {}
    for name, t in streams.items():
        grams = [tuple(t[i:i + n]) for i in range(0, len(t) - n + 1, stride)]
        out[name] = sum(g in seen for g in grams) / max(len(grams), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["fsa_free", "fsa_full", "lora"], required=True)
    ap.add_argument("--fmt", choices=["ternary", "fp4", "fp8", "bf16"], default="ternary",
                    help="value format of the trainable slots (fsa only)")
    ap.add_argument("--layers", default="14,15")       # indices into the 32 MoE layers
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="controls fact sampling, template/batch sampling, anchor sampling and LoRA init")
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
    ap.add_argument("--bop-tau-scale", choices=["none", "energy"], default="none",
                    help="energy: per-row birth threshold tau * f(general-text energy of a birth in that row)")
    ap.add_argument("--tau-power", type=float, default=0.5, help="f = clamp((e/median e)^power, lo, hi)")
    ap.add_argument("--tau-lo", type=float, default=0.25)
    ap.add_argument("--tau-hi", type=float, default=4.0)
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
    ap.add_argument("--select-metric", choices=["gen", "tf"], default="gen",
                    help="validation recall used for Pareto checkpoints: generated (default) or teacher-forced")
    ap.add_argument("--select-em", type=float, default=0.7, help="(unused; kept for old command lines)")
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on patch params (pulls idle masters to 0)")
    # v11
    ap.add_argument("--gen-max-new", type=int, default=0, help="generated tokens per prompt (0 = longest answer + k + 2)")
    ap.add_argument("--gen-stop-k", type=int, default=2, help="clean stop: terminator/EOS within k tokens after the answer")
    ap.add_argument("--gen-bs", type=int, default=1,
                    help="eval batch size. 1 (default) = the single-sequence protocol of v1-v10; on the 8B, batch "
                         "composition alone changes logits through bf16 rounding -> routing flips")
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--facts-per-person", type=int, default=len(ATTRS))
    ap.add_argument("--shared-attrs", type=int, default=0,
                    help="attributes every person holds (conflicting-fact pairs; cross-person confusion is scored)")
    ap.add_argument("--values", choices=["disjoint", "overlap"], default="disjoint")
    ap.add_argument("--fact-batch", type=int, default=0, help="facts sampled per training step (0 = all)")
    ap.add_argument("--val-facts", type=int, default=0, help="facts in the per-eval validation subset (0 = all)")
    ap.add_argument("--conv-test", action="store_true", help="also ask the test question after ~500 tokens of chat")
    ap.add_argument("--conv-facts", type=int, default=40, help="facts in the in-conversation test (0 = all)")
    ap.add_argument("--rehearsal-frac", type=float, default=0.0,
                    help="r: objective (1-r) * fact CE + r * general-text next-token CE on anchor windows")
    ap.add_argument("--rehearsal-bs", type=int, default=0, help="anchor windows per rehearsal step (0 = --anchor-bs)")
    ap.add_argument("--repair-steps", type=int, default=0,
                    help="after --steps: K Bop steps with births disabled (deaths only), local penalty active")
    ap.add_argument("--repair-local-lambda", type=float, default=0.0, help="penalty weight in repair (0 = --local-lambda)")
    ap.add_argument("--patch-dir", default=None, help="where patches are written (default: <out stem>_patches/)")
    ap.add_argument("--load-patch", default=None, help="load a saved patch after attach (use with --steps 0)")
    a = ap.parse_args()

    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    held = json.load(open(DATA / "heldout-tokens.json"))       # flat stream, BOS-first (parity layout)
    segs = [held[511 * i: 511 * i + 512] for i in range(5)]   # 5 x 512 windows = 2,555 scored positions
    bos = BOS

    fsa = a.method.startswith("fsa")
    total_steps = a.steps + a.repair_steps
    repair_lambda = a.repair_local_lambda or a.local_lambda
    if a.optim == "bop" and not fsa:
        raise SystemExit("--optim bop is defined for FSA slots only")
    if a.optim == "bop" and (a.wd or a.master_clip):
        raise SystemExit("--wd/--master-clip are no-ops under --optim bop (it stores the ternary value itself); "
                         "use --repair-steps / --local-lambda / netcost / --rehearsal-frac / --bop-tau-scale")
    if a.optim == "sgd" and a.patch_scale == "block":
        raise SystemExit("--patch-scale block requires the Adam optimizer (separate no-decay scale group)")
    if a.repair_steps and (a.optim != "bop" or repair_lambda <= 0):
        raise SystemExit("--repair-steps needs --optim bop and a positive local penalty weight")
    if a.bop_tau_scale == "energy" and a.optim != "bop":
        raise SystemExit("--bop-tau-scale energy needs --optim bop")
    if not 0 <= a.rehearsal_frac < 1:
        raise SystemExit("--rehearsal-frac must be in [0, 1)")
    needs_anchor = (a.drift_lambda > 0 or a.local_lambda > 0 or a.rehearsal_frac > 0 or a.repair_steps
                    or a.bop_tau_scale == "energy" or (fsa and (a.n_experts or a.support_score == "netcost")))
    if needs_anchor and not a.anchor:
        raise SystemExit("--anchor is required for drift/local/rehearsal/repair, expert profiling, netcost, energy tau")
    if a.eval_every and (not a.eval_set or a.val_windows <= 0):
        raise SystemExit("--eval-every needs --eval-set and --val-windows > 0 (validation dNLL selection)")
    if a.conv_test and not a.eval_set:
        raise SystemExit("--conv-test draws its chat context from --eval-set")
    if a.anchor and a.eval_set and Path(a.anchor).resolve() == Path(a.eval_set).resolve():
        raise SystemExit("--anchor and --eval-set must be different text")

    t0 = time.time()
    model, cfg, dev = build_model(EXPORT, device="cuda:0", model_config=None, step=58814,
                                  expert_format="bf16", eda_kernel_backend="naive_fp32")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"built {time.time()-t0:.0f}s, MoE layers={len(moe_layers(model))}", flush=True)

    facts = make_world(a.seed, a.people, a.facts_per_person, a.shared_attrs, a.values)
    conflicts = conflict_map(facts)
    print(f"facts {len(facts)} people {a.people} conflicting {len(conflicts)}", flush=True)
    rngv = random.Random(a.seed + 3)
    val_idx = sorted(rngv.sample(range(len(facts)), a.val_facts)) if 0 < a.val_facts < len(facts) else None
    conv_idx = sorted(rngv.sample(range(len(facts)), a.conv_facts)) if 0 < a.conv_facts < len(facts) else None
    K = a.gen_stop_k
    items_train = build_items(tok, facts, TRAIN_T, bos)
    items_val = build_items(tok, facts, VAL_T, bos, idxs=val_idx)
    items_test = build_items(tok, facts, TEST_T, bos)
    items_conv = None
    et = None
    if a.eval_set:
        et = json.load(open(a.eval_set))
    if a.conv_test:
        lo_, hi_ = a.eval_windows * 511 + 1, 60000         # between the final-eval windows and the val windows
        assert hi_ - lo_ > 2000
        items_conv = build_items(tok, facts, [CONV_T], bos, ctx=conv_contexts(tok, et, facts, a.seed, lo_, hi_),
                                 idxs=conv_idx)
        print(f"conv test {len(items_conv)} prompts, mean {sum(len(i['p']) for i in items_conv)/len(items_conv):.0f} tokens",
              flush=True)

    def gen(items):
        return gen_eval(model, tok, items, dev, a.gen_max_new, K, a.gen_bs, conflicts)

    def tf(items):
        return tf_eval(model, items, dev, a.gen_bs)

    def acq_suite(with_train=False):
        r = dict(test=tf(items_test), test_gen=gen(items_test))
        if items_conv is not None:
            r.update(conv=tf(items_conv), conv_gen=gen(items_conv))
        if with_train:
            r["train"] = tf(items_train)
        return r

    base_held, base_lp = heldout_eval(model, segs, dev)
    base_lp_t = [l for l in base_lp]
    base_acq = acq_suite(with_train=True)
    base_logits0 = logits_of(model, torch.tensor([segs[0]], device=dev)).detach().clone()
    big = base_big = valw = base_valw = None
    if a.eval_set:
        big = torch.tensor([[bos] + et[i * 511:i * 511 + 511] for i in range(a.eval_windows)], device=dev)
        vo = 60000                                          # validation text starts well past the final-eval windows
        assert vo > a.eval_windows * 511 and vo + a.val_windows * 511 <= len(et)
        valw = torch.tensor([[bos] + et[vo + i * 511:vo + i * 511 + 511] for i in range(a.val_windows)], device=dev)
        base_valw = nll_windows(model, valw, dev)
        base_big = nll_windows(model, big, dev)
        print(f"big eval {big.shape[0]} x 512, base nll {sum(base_big)/len(base_big):.4f}", flush=True)
    print("BASE", json.dumps({"held": base_held, **base_acq}), flush=True)

    layer_ids = [int(x) for x in a.layers.split(",")]
    projs = ("down_proj",) if a.down_only else ("up_proj", "down_proj")
    anchor, ov = None, None
    if a.anchor:
        at = json.load(open(a.anchor))
        L = a.anchor_len
        vt, tt = at[-510:], at[:-510]                     # validation tail is never used for training anchors
        assert len(vt) == 510 and len(tt) >= 16 * L, "anchor file too short"
        anchor = torch.tensor([[bos] + tt[i:i + L - 1] for i in range(0, len(tt) - L, L - 1)], device=dev)
        print(f"anchor windows {anchor.shape[0]} x {L}", flush=True)
        streams = {"heldout": held[:511 * 5 + 1]}
        if et is not None:                                 # final-eval windows, conv-context region, val windows
            streams.update(eval=et[:a.eval_windows * 511], conv=et[a.eval_windows * 511:60000],
                           val=et[60000:60000 + a.val_windows * 511])
        ov = ngram_overlap(tt, streams)
        print("OVERLAP 32-gram train-anchor vs eval text", json.dumps(ov), flush=True)
        # the legacy 5 x 512 'heldout' KL/top-1 set is drawn from the same wikitext as the anchors (known overlap;
        # it is reported, not used for any headline); the dNLL eval, val and conversation text must be disjoint
        if max(v for k, v in ov.items() if k != "heldout") > 0.01:
            raise SystemExit(f"anchor text overlaps evaluation text: {ov}")

    def enc_fact(t, f):
        full = fmt(t, f)
        cut = full.rindex(f.val)
        p_ids = encode(tok, full[:cut].rstrip(), bos)
        return p_ids + tok.encode(" " + f.val + full[cut + len(f.val):], add_special_tokens=False).ids, len(p_ids)

    pool = [[enc_fact(t, f) for t in (TRAIN_AUG_T if a.aug else TRAIN_T)] for f in facts]
    rng = random.Random(a.seed + 1)
    resample = a.aug or (0 < a.fact_batch < len(facts))

    def make_batch():
        fis = sorted(rng.sample(range(len(pool)), a.fact_batch)) if 0 < a.fact_batch < len(pool) else range(len(pool))
        chosen = [e for fi in fis for e in (rng.sample(pool[fi], 3) if a.aug else pool[fi])]   # 3 per fact per step
        xx, ll = batchify([c[0] for c in chosen], dev, [c[1] for c in chosen] if a.answer_only else None)
        return xx, ll, int(ll[:, 1:].sum())

    x, lm, ntok = make_batch()
    fact_scale = 1.0 - a.rehearsal_frac

    def fact_loss_backward():
        nonlocal x, lm, ntok
        if resample:
            x, lm, ntok = make_batch()
        tot = 0.0
        for s0 in range(0, x.shape[0], a.micro):           # grad accumulation: naive EDA scan stores per-step state
            xb, mb = x[s0:s0 + a.micro], lm[s0:s0 + a.micro, 1:]
            lg = logits_of(model, xb)[:, :-1, :VOCAB].float()
            l = F.cross_entropy(lg[mb], xb[:, 1:][mb], reduction="sum") / ntok
            if l.requires_grad:                           # a microbatch may route through no patched expert
                (fact_scale * l).backward()
            tot += l.item()
        return tot

    keep, selection = None, None
    if a.n_experts and fsa:
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
    lmods = lora_modules(model)
    if a.patch_scale == "block":
        for m in mods:
            m.enable_block_scale(a.scale_block, a.scale_init)
            params.append(m.logb)
    local = LocalOutputPenalty(model) if (a.local_lambda > 0 or a.repair_steps) else None
    lr = a.lr or (0.02 if fsa else 1e-3)
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
    meta = dict(method=a.method, fmt=a.fmt, layers=layer_ids, projs=list(projs), rank=a.rank, seed=a.seed,
                n_experts=a.n_experts, selection=selection, fp4_step=a.fp4_step, patch_scale=a.patch_scale,
                scale_block=a.scale_block if a.patch_scale == "block" else 0, people=a.people,
                facts_per_person=a.facts_per_person, shared_attrs=a.shared_attrs, values=a.values, budget=a.budget,
                export=str(EXPORT), alpha="mean |nonzero base weight| of the row")
    protocol = dict(select_metric=a.select_metric, gen_max_new=a.gen_max_new, gen_stop_k=a.gen_stop_k, val_facts=a.val_facts, conv_test=a.conv_test,
                    conv_facts=a.conv_facts, eval_set=a.eval_set, eval_windows=a.eval_windows,
                    val_windows=a.val_windows, anchor=a.anchor)
    if a.load_patch:
        st_ = torch.load(a.load_patch, map_location="cpu", weights_only=False)
        diff = {k: (st_["meta"].get(k), v) for k, v in meta.items()
                if k not in ("selection",) and json.dumps(st_["meta"].get(k)) != json.dumps(v)}
        if diff:
            raise SystemExit(f"--load-patch config mismatch (saved, now): {diff}")
        pdiff = {k: (st_["meta"].get("protocol", {}).get(k), v) for k, v in protocol.items()
                 if st_["meta"].get("protocol", {}).get(k) != v}
        if pdiff:
            print(f"WARNING --load-patch evaluation protocol differs (saved, now): {pdiff}", flush=True)
        load_patch(model, st_)
        if a.budget:
            assert patch_stats(mods)[0] <= a.budget, "loaded patch exceeds --budget"
        print(f"loaded patch {a.load_patch} (step {st_['meta'].get('step')})", flush=True)
    den_total, cost_lambda, tau_stats = None, None, None
    if (a.support_score == "netcost" or a.bop_tau_scale == "energy") and mods:
        den_total = collect_col_energy(model, mods, anchor[:32])
        assert math.isfinite(den_total) and den_total > 0
        print(f"col-energy stats on 32 anchor windows, base output energy {den_total:.3e}", flush=True)
    if a.bop_tau_scale == "energy" and mods:
        tau_stats = set_energy_tau_scale(mods, a.tau_power, a.tau_lo, a.tau_hi)
        print("TAU-SCALE", json.dumps(tau_stats), flush=True)
    patch_dir = Path(a.patch_dir) if a.patch_dir else Path(a.out).with_name(Path(a.out).stem + "_patches")
    set_enabled(True)
    model.train(False)
    t1 = time.time()
    gscale, traj, best = None, [], {}
    for step in range(total_steps):
        repair = step >= a.steps
        lam = repair_lambda if repair else a.local_lambda
        opt.zero_grad(set_to_none=True)
        tot = fact_loss_backward()
        if a.optim == "bop" and bop_tau is None:          # calibrate on the answer gradient alone (before any
            gl = [(m.M.grad[m.mask]).flatten() for m in mods if m.M.grad is not None]   # rehearsal/penalty term)
            if gl and sum(g.numel() for g in gl) > 0:     # first batch that reaches a slot
                bop_tau = a.bop_tau * float(torch.cat(gl).pow(2).mean().sqrt()) / fact_scale
                print(f"bop tau {bop_tau:.3e} ({a.bop_tau} x first answer-gradient RMS, step {step})", flush=True)
        if a.budget and mods and not repair:               # usefulness score: EMA of |answer-loss gradient|
            for m in mods:
                m.score.mul_(0.9)                          # missing gradient = zero evidence this step
                if m.M.grad is not None:
                    g_ = m.M.grad if a.support_score == "netcost" else m.M.grad.abs()
                    m.score.add_(g_, alpha=0.1)
        if a.rehearsal_frac > 0:
            nr = a.rehearsal_bs or a.anchor_bs
            ab = anchor[torch.randint(0, anchor.shape[0], (nr,), device=dev)]
            rsum = 0.0
            for r_ in range(nr):                           # one window per backward: bounded activation memory
                rl = general_ce(model, ab[r_:r_ + 1])
                if rl.requires_grad:
                    (a.rehearsal_frac * rl / nr).backward()
                rsum += rl.item() / nr
            if step % 25 == 0:
                print(f"step {step} rehearsal_ce {rsum:.4f}", flush=True)
        if local is not None and lam > 0:
            ab = anchor[torch.randint(0, anchor.shape[0], (a.anchor_bs,), device=dev)]
            local.start()
            logits_of(model, ab)
            lp_ = local.stop()
            if lp_ is not None and lp_.requires_grad:
                (lam * lp_).backward()
                tot += lam * lp_.item()
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
            nb, nd = (bop_step(mods, a.bop_gamma, bop_tau, bool(a.budget), allow_birth=not repair)
                      if bop_tau is not None else (0, 0))
            if step % 25 == 0 or (repair and step % 5 == 0):
                print(f"step {step} bop births {nb} deaths {nd}{' (repair)' if repair else ''}", flush=True)
        else:
            opt.step()
        if a.master_clip and a.optim != "bop":          # bop holds the ternary value itself; never clip it
            with torch.no_grad():
                for m in mods:
                    m.M.clamp_(-a.master_clip, a.master_clip)
        if a.budget and mods and (step % 10 == 0) and step <= a.budget_freeze and not repair:
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
                    npos = sum(x_.numel() for x_ in gv)
                    if npos > 0:
                        gain = torch.cat(gv); cost = torch.cat(cv)
                        top = torch.topk(gain, min(a.budget, npos)).indices
                        lam_ = a.cost_rel * float(gain[top].median() / cost[top].median().clamp_min(1e-30))
                        if math.isfinite(lam_) and lam_ > 0:
                            cost_lambda = lam_
                            print(f"netcost lambda {cost_lambda:.3e} (rel {a.cost_rel})", flush=True)
                if cost_lambda is not None:
                    refresh_support(mods, a.budget,
                                    lambda i, m: module_score(m, "netcost", cost_lambda, den_total), opt)
            else:
                refresh_support(mods, a.budget, None, opt)
        if step % 25 == 0 or step == total_steps - 1:
            print(f"step {step} loss {tot:.4f} ({time.time()-t1:.0f}s)", flush=True)
        if a.eval_every and ((step + 1) % a.eval_every == 0 or step == total_steps - 1):
            vd = (float(torch.tensor(nll_windows(model, valw, dev)).mean() - torch.tensor(base_valw).mean())
                  if valw is not None else float("nan"))
            v = tf(items_val)
            vg = gen(items_val)
            nnz, en = patch_stats(mods) if mods else (0, 0.0)
            traj.append(dict(step=step + 1, val_em=v["exact_match"], val_gen=vg["recall"], val_gen_match=vg["match"],
                             val_lp=v["answer_logprob"], val_dnll=vd, nnz=nnz, energy=en, repair=repair))
            print("TRAJ", json.dumps(traj[-1]), flush=True)
            if not math.isfinite(vd):
                raise RuntimeError("non-finite validation dNLL")
            vsel = vg["recall"] if a.select_metric == "gen" else v["exact_match"]
            for thr in PARETO_EM:                           # per recall level: lowest validation dNLL checkpoint
                if vsel >= thr and (thr not in best or vd < best[thr]["val_dnll"]):
                    best[thr] = dict(traj[-1], state=[t.detach().to("cpu", copy=True) for t in params]
                                     + ([m.support.to("cpu", copy=True) for m in mods] if a.budget else []))

    def big_eval():
        if big is None:
            return None
        d = torch.tensor(nll_windows(model, big, dev)) - torch.tensor(base_big)
        return dict(dnll=float(d.mean()), dnll_se=float(d.std() / math.sqrt(len(d))))

    def storage():
        if lmods:
            return dict(lora=lora_storage(lmods))
        nnz_, _ = patch_stats(mods) if mods else (0, 0.0)
        return dict(fsa=fsa_storage(nnz_, n_slots, a.fmt, sum(m.logb.numel() for m in mods if m.logb is not None)),
                    nnz=nnz_)

    def quant_suite():
        """LoRA only: bf16 / int8 / int4 fake-quantized factors (per rank-vector bf16 absmax scales)."""
        if not lmods:
            return None
        out = {}
        for qf in ("bf16", "int8", "int4"):
            saved = quantize_lora_(lmods, qf)
            out[qf] = dict(big=big_eval(), **acq_suite())
            restore_lora_(lmods, saved)
            print("QUANT", qf, json.dumps(dict(big=out[qf]["big"], test=out[qf]["test"]["exact_match"],
                                               test_gen=out[qf]["test_gen"]["recall"])), flush=True)
        return out

    nflip, energy = patch_stats(mods) if mods else (0, 0.0)
    on_big = big_eval()
    on_held, _ = heldout_eval(model, segs, dev, base_lp_t)
    on_acq = acq_suite(with_train=True)
    on_store = storage()
    final_bytes = save_patch(model, patch_dir / "final.pt", dict(meta, protocol=protocol, step=total_steps))
    on_quant = quant_suite()
    print("FINAL", json.dumps(dict(big=on_big, test=on_acq["test"], test_gen=on_acq["test_gen"]["recall"],
                                   storage=on_store)), flush=True)
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
        pb = save_patch(model, patch_dir / f"step{bk['step']}.pt", dict(meta, protocol=protocol, step=bk["step"]))
        best_eval[str(thr)] = seen[bk["step"]] = dict(bk, nnz=bn, energy=be, big=big_eval(), **acq_suite(),
                                                      storage=storage(), patch_file=f"step{bk['step']}.pt",
                                                      patch_file_bytes=pb, quant=quant_suite())
        print("PARETO", thr, json.dumps({k: v for k, v in best_eval[str(thr)].items() if k != "quant"}), flush=True)
    set_enabled(False)
    off_logits0 = logits_of(model, torch.tensor([segs[0]], device=dev)).detach()
    revoke_exact = bool(torch.equal(off_logits0, base_logits0))
    off_test = tf(items_test)
    off_gen = gen(items_test)

    st = on_store.get("fsa") or {}
    lora_bytes = on_store.get("lora")
    res = dict(method=a.method, fmt=a.fmt, drift_lambda=a.drift_lambda, wd=a.wd, answer_only=a.answer_only,
               down_only=a.down_only, n_experts=a.n_experts, budget=a.budget, optim=a.optim, selection=selection,
               layers=layer_ids, rank=a.rank, steps=a.steps, repair_steps=a.repair_steps,
               repair_local_lambda=repair_lambda if a.repair_steps else 0.0, rehearsal_frac=a.rehearsal_frac,
               bop_tau=a.bop_tau, bop_tau_abs=bop_tau if a.optim == "bop" else None, bop_tau_scale=a.bop_tau_scale,
               tau_scale_stats=tau_stats, lr=lr, seed=a.seed,
               people=a.people, facts_per_person=a.facts_per_person, shared_attrs=a.shared_attrs, values=a.values,
               fact_batch=a.fact_batch, n_conflicting=len(conflicts), facts=[f._asdict() for f in facts],
               n_facts=len(facts), reserve_slots=n_slots, flipped=nflip, energy=energy, lora_params=n_lora,
               patch_bytes=(st.get("bytes") if st else (lora_bytes["fp32"] if lora_bytes else None)),
               storage=on_store, patch_dir=str(patch_dir), patch_file_bytes=final_bytes,
               gen=dict(max_new=a.gen_max_new, stop_k=K, select_metric=a.select_metric),
               base=dict(held=base_held, **base_acq),
               patched=dict(held=on_held, big=on_big, **on_acq), lora_quant=on_quant, local_lambda=a.local_lambda,
               revoked=dict(logits_bit_exact=revoke_exact, test=off_test, test_gen=off_gen), trajectory=traj,
               pareto=best_eval, support_score=a.support_score, cost_rel=a.cost_rel, cost_lambda=cost_lambda,
               fp4_step=a.fp4_step, aug=a.aug, patch_scale=a.patch_scale, scale_block=a.scale_block,
               scale_init=a.scale_init, load_patch=a.load_patch, anchor_overlap=ov)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("facts", "trajectory")}), flush=True)


if __name__ == "__main__":
    main()
