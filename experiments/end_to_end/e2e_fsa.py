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
import argparse, hashlib, inspect, json, math, os, random, re, sys, tempfile, time, uuid
from collections import Counter, defaultdict, namedtuple
from itertools import permutations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from contextlib import contextmanager
try:
    from .mls import MLSOptimizer, CodeTransaction
except ImportError:                         # direct script execution / CPU tests
    from mls import MLSOptimizer, CodeTransaction

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

SELF_STUDY_T = [
    "Please recall {name}'s {attr}.\nResponse: {answer}",
    "User: Can you remind me of the {attr} recorded for {name}?\nAssistant: {answer}",
    "Complete this profile entry for {name}: {attr}.\nEntry: {answer}",
    "What are {name}'s {attr} and {attr2}?\nAnswer: {answer}",
    "User: Summarize {name}'s {attr} together with {pos} {attr2}.\nAssistant: {answer}",
    "Is {name}'s {attr} the same as {other}'s?\nAnswer: {answer}",
    "User: Compare the {attr} of {name} and {other}. Are they alike?\nAssistant: {answer}",
]
assert set(SELF_STUDY_T).isdisjoint(VAL_T + TEST_T), "self-study templates overlap held-out templates"

QA_UNSEEN_T = ["Quick question about {name}: {attr}?\nAnswer:",
               "I forgot {name}'s {attr}. What was it again?\nAnswer:",
               "{name} once told me about {pos} {attr}. Which was it?\nAnswer:"]
QA_TWO_T = "What are {name}'s {attr1} and {attr2}?\nReply:"
QA_INDIRECT_T = "The person whose {attrX} is {valX} — what is {pos} {attrY}?\nAnswer:"
assert set(QA_UNSEEN_T + [QA_TWO_T, QA_INDIRECT_T]).isdisjoint(
    TRAIN_T + TRAIN_AUG_T + VAL_T + TEST_T + [CONV_T] + SELF_STUDY_T), "QA templates overlap existing templates"

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
        self.wshape = tuple(W.shape)
        self.sparse = False
        self.pos_only = None
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
        if self.fmt == "tern2":                            # two ternary planes c1*1 + c2*0.5 -> {0,+-.5,+-1,+-1.5}
            return torch.round(m * 2).clamp(-3, 3) / 2
        if self.fmt == "fp8":
            return m.clamp(-2, 2).to(torch.float8_e4m3fn).float()
        return m

    @torch.no_grad()
    def codes(self):
        """The stored patch, in units of the slot scale: Q(M) on the mask and the physical support."""
        return self._quant(self.M * self.mask) * self.support

    def _masters(self):
        # During wake training only a small vector of selected up-slot masters needs gradients/Adam state.
        if hasattr(self, "wake_master"):
            return self.M.detach().flatten().index_copy(0, self.wake_slot_indices, self.wake_master).view_as(self.M)
        return self.M

    def delta(self):
        m = self._masters() * self.mask
        q = self._quant(m)
        if getattr(self, "forward_keep", None) is not None:
            q = q * self.forward_keep
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


class SparseFSAParam(FSAParam):
    """FSAParam whose training state lives on the free slots only: M, its gradient, the Bop EMA, score/support
    and churn counters are 1-D over the reserve (idx = flat positions of the free slots, ascending), about 1/6 of
    the full-shape state on these experts. The forward scatters the slot values into a transient dense delta.
    Row scales only (no block scales)."""

    def __init__(self, W, mask, fmt="ternary", budgeted=False, occupied=False, slot_frac=1.0, gen=None):
        """occupied: also make every nonzero base weight a slot whose only move is a sign flip (code 0 or +1,
        delta = -2 w): ternary values and the active-pair pattern are unchanged, so pair8 stays legal."""
        nn.Module.__init__(self)
        nz = (W != 0)
        alpha = (W.float().abs() * nz).sum(1, keepdim=True) / nz.sum(1, keepdim=True).clamp_min(1)
        sel = (mask | nz) if occupied else mask
        if slot_frac < 1.0:                                # random subset of the reserve (memory for many layers)
            keep = (torch.rand(sel.shape, generator=gen) < slot_frac).to(sel.device)
            sel = sel & keep
        idx = sel.flatten().nonzero().squeeze(1)
        n = idx.numel()
        self.register_buffer("idx", idx.to(torch.int32))
        self.register_buffer("alpha", alpha)
        occ = nz.flatten()[idx] if occupied else torch.zeros(n, dtype=torch.bool, device=W.device)
        self.register_buffer("occ", occ)
        self.pos_only = occ if occupied else None          # flip slots: births to +1 only
        rows_ = torch.div(idx, W.shape[1], rounding_mode="floor")
        self.register_buffer("scale", torch.where(occ, -2.0 * W.flatten()[idx].float(),
                                                  alpha.float().squeeze(1)[rows_]))
        self.wshape = tuple(W.shape)
        self.sparse = True
        self.fmt = fmt
        self.fp4_step = 0.25
        self.logb = None
        self.block = 0
        self.M = nn.Parameter(torch.zeros(n, dtype=torch.float32, device=W.device))
        self.mask = torch.ones(1, dtype=torch.bool, device=W.device).expand(n)   # every reserve slot is legal
        self.support = torch.zeros(n, dtype=torch.bool, device=W.device) if budgeted else self.mask
        self.score = torch.zeros(n, dtype=torch.bfloat16, device=W.device) if budgeted else None
        self.col_energy = None
        self.tau_scale = None

    def rows(self):
        return torch.div(self.idx.long(), self.wshape[1], rounding_mode="floor")

    def cols(self):
        return self.idx.long() % self.wshape[1]

    def slot_alpha(self):
        """Magnitude of one code step per slot: alpha_row for a free slot, 2|w| for a sign flip."""
        return self.scale.abs()

    def _quant(self, m):
        q = FSAParam._quant(self, m)
        if self.pos_only is not None:                      # sign-flip slots: code 0 or +1 only
            q = torch.where(self.pos_only, (m > 0.5).float(), q)
        return q

    def _dense(self, v):
        out = torch.zeros(self.wshape[0] * self.wshape[1], dtype=v.dtype, device=v.device)
        return out.index_put_((self.idx.long(),), v).view(self.wshape)

    @torch.no_grad()
    def codes(self):
        """Free-slot codes only (dense); sign flips of occupied weights are reported by flip_codes()."""
        return self._dense(self._quant(self.M) * self.support * ~self.occ)

    @torch.no_grad()
    def flip_codes(self):
        return self._dense((self._quant(self.M) * self.support * self.occ).clamp_min(0))

    def delta(self):
        master = self._masters()
        q = self._quant(master)
        if getattr(self, "forward_keep", None) is not None:
            q = q * self.forward_keep
        m = (q * self.support - master).detach() + master
        return self._dense(m * self.scale)

    def enable_block_scale(self, block, init_rel):
        raise NotImplementedError("block scales are not supported with sparse slot state")


def slot_cost(m, den):
    """Isolated local-output cost alpha_row^2 * E[x_j^2] / den, shaped like m.M (dense or sparse state)."""
    if getattr(m, "sparse", False):
        return m.slot_alpha() ** 2 * m.col_energy[m.cols()] / den
    sc = (torch.exp(m.logb.detach()).repeat_interleave(m.block, dim=1) if m.logb is not None else m.alpha.float())
    return (sc ** 2) * m.col_energy[None, :] / den


def frozen_slots(m):
    rows = getattr(m, "wake_rows", None)
    if rows is None:
        return torch.zeros((), dtype=torch.bool, device=m.M.device)
    return rows[m.rows()] if m.sparse else rows[:, None].expand_as(m.M)


def writable_slots(m):
    return m.mask & ~frozen_slots(m) if hasattr(m, "wake_rows") else m.mask


def pinned_slots(m, keep_live=False):
    if not keep_live and not hasattr(m, "wake_rows"):
        return torch.zeros((), dtype=torch.bool, device=m.M.device)
    live = m._quant(m.M.detach() * m.mask) * m.support != 0
    return live if keep_live else live & frozen_slots(m)


@torch.no_grad()
def freeze_wake_rows(m, rows):
    selected = torch.zeros(m.wshape[0], dtype=torch.bool, device=m.M.device)
    selected[rows] = True
    m.register_buffer("wake_rows", selected)
    frozen = frozen_slots(m) & m.mask
    m.wake_indices = frozen.flatten().nonzero().flatten()
    m.wake_codes = m.M.detach().flatten()[m.wake_indices].clone()
    m.M.register_hook(lambda grad: grad * ~frozen_slots(m))


@torch.no_grad()
def restore_wake_codes(mods):
    """Adam warmup weight decay must also leave detector rows unchanged."""
    for m in mods:
        if hasattr(m, "wake_rows"):
            m.M.view(-1)[m.wake_indices] = m.wake_codes


class WakeMLSOptimizer(MLSOptimizer):
    """Exclude frozen rows from MLS's solve AND live-code budget screening."""
    def propose(self):
        pinned = sum(int(pinned_slots(m).sum()) for _, m in self.linears)
        budget = self.budget
        if budget and pinned >= budget:
            if pinned > budget:
                raise ValueError("wake codes exceed MLS budget")
            return CodeTransaction([]), self._empty_info()
        masks = [(m, m.mask) for _, m in self.linears]
        try:
            for m, _ in masks:
                m.mask = writable_slots(m)
            self.budget = budget - pinned if budget else 0
            return super().propose()
        finally:
            self.budget = budget
            for m, mask in masks:
                m.mask = mask


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


class TeacherLoRAParam(LoRAParam):
    """Frozen, independently switched teacher; construction consumes no RNG."""
    def __init__(self, W, A, B):
        nn.Module.__init__(self)
        self.A = nn.Parameter(A.to(device=W.device).float().clone(), requires_grad=False)
        self.B = nn.Parameter(B.to(device=W.device).float().clone(), requires_grad=False)
        self.enabled = False
        self.scale = 1.0

    def forward(self, W):
        return W + (self.scale * (self.B @ self.A)).to(W.dtype) if self.enabled and self.scale else W


def attach_teacher(model, state):
    """Stack saved LoRA factors after any student, including teacher-only linears."""
    if not isinstance(state, dict):
        state = torch.load(state, map_location="cpu", weights_only=False)
    assert state["format"] == "e2e-fsa-patch-v1" and state["meta"]["method"] == "lora"
    assert state["meta"]["layers"] and state["meta"]["rank"] > 0
    linears = dict(model.named_modules())
    teachers = []
    for name, e in state["entries"].items():
        assert e["kind"] == "lora", "teacher must be a saved LoRA patch"
        lin = linears[name]
        W = lin.parametrizations.weight.original if P.is_parametrized(lin, "weight") else lin.weight
        assert e["A"].shape == (state["meta"]["rank"], W.shape[1])
        assert e["B"].shape == (W.shape[0], state["meta"]["rank"])
        teacher = TeacherLoRAParam(W, e["A"], e["B"])
        P.register_parametrization(lin, "weight", teacher)
        teachers.append(teacher)
    assert teachers, "empty teacher patch"
    return teachers


@contextmanager
def teacher_forward(teachers):
    """Full-strength teacher-only KL target, restoring any fading training state."""
    old = FSAParam.enabled
    states = [(t.enabled, t.scale) for t in teachers]
    FSAParam.enabled = False
    for t in teachers:
        t.enabled = True
        t.scale = 1.0
    try:
        with torch.no_grad():
            yield
    finally:
        for t, (enabled, scale) in zip(teachers, states):
            t.enabled, t.scale = enabled, scale
        FSAParam.enabled = old


def remove_teacher(model):
    """Remove teacher modules physically before final evaluation/export/revoke/accounting."""
    for _, lin in list(model.named_modules()):
        if isinstance(lin, nn.Linear) and P.is_parametrized(lin, "weight"):
            stack = list(lin.parametrizations.weight)
            if any(isinstance(p, TeacherLoRAParam) for p in stack):
                assert all(not p.enabled for p in stack if isinstance(p, TeacherLoRAParam))
                P.remove_parametrizations(lin, "weight", leave_parametrized=False)
                for p in stack:
                    if not isinstance(p, TeacherLoRAParam):
                        P.register_parametrization(lin, "weight", p)
    assert not any(isinstance(p, TeacherLoRAParam) for p in model.modules())


def distillation_kl(student, teacher, gold, mask, temperature):
    """Summed T² KL(teacher || student), only at gold-correct answer/stop positions."""
    valid = mask & (teacher.argmax(-1) == gold)
    s = F.log_softmax(student[valid] / temperature, -1)
    t = F.softmax(teacher[valid].detach() / temperature, -1)
    return temperature ** 2 * F.kl_div(s, t, reduction="sum")


def context_kl(student, teacher, gold, mask, temperature, only_correct=False):
    """Full-vocabulary T² KL on answer/stop tokens; incorrect teacher targets count by default."""
    valid = mask & (teacher.argmax(-1) == gold) if only_correct else mask
    return temperature ** 2 * F.kl_div(
        F.log_softmax(student[valid] / temperature, -1),
        F.softmax(teacher[valid].detach() / temperature, -1), reduction="sum")


@contextmanager
def context_base_forward():
    """Frozen base only, with exception-safe restoration of both student switches."""
    old = FSAParam.enabled, LoRAParam.enabled
    set_enabled(False)
    try:
        with torch.no_grad():
            yield
    finally:
        FSAParam.enabled, LoRAParam.enabled = old


def distill_weight(mu, anneal, step, steps):
    return mu * max(0.0, 1 - step / (steps - 1)) if anneal and steps > 1 else (0.0 if anneal else mu)


def teacher_fade_scale(fade, shape, step, steps):
    """Teacher strength at the zero-based training step; fade=0 disables homotopy."""
    if not fade or not steps:
        return 0.0
    progress = min(max(step / (fade * steps), 0.0), 1.0)
    return (1 + math.cos(math.pi * progress)) / 2 if shape == "cos" else 1 - progress


class FoldLoRAParam(LoRAParam):
    """LoRA trained to end as a pure free-slot ternary patch (inference adapter-free, pair8-legal).
    mode 'ste': delta = alpha_row * Q((mask * B A) / alpha_row) with a straight-through gradient, so the adapter
    learns what survives the fold; the final patch is exactly that Q.
    mode 'ef':  delta = alpha_row * Q(S) + B A; fold() adds mask * B A / alpha_row into the fp32 shadow masters S
    (error feedback: S keeps the unrealized residual), then resets the adapter (B = 0, fresh A; ReLoRA-style). The
    part of B A outside the free slots cannot be represented and is dropped at each fold (fold_lost tracks it).
    Q = ternary round (|x| > 0.5 -> sign)."""

    thr = 0.5                                              # code threshold in units of alpha (--fold-thr)

    def __init__(self, W, r, mask, mode):
        super().__init__(W, r)
        nz = (W != 0)
        alpha = (W.float().abs() * nz).sum(1, keepdim=True) / nz.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer("mask", mask.to(torch.bool))
        self.register_buffer("alpha", alpha)
        self.mode = mode
        self.wshape = tuple(W.shape)
        self.register_buffer("S", torch.zeros(W.shape, dtype=torch.float32, device=W.device) if mode == "ef"
                             else torch.zeros(0, device=W.device))
        self.fold_lost = 0.0
        self.fold_kept = 0.0

    @staticmethod
    def _q(x):
        return torch.sign(x) * (x.abs() > FoldLoRAParam.thr).float()

    def delta_w(self):
        d = self.B @ self.A
        if self.mode == "ste":
            u = d * self.mask / self.alpha.clamp_min(1e-12)   # rows with alpha 0 have no free slots
            return ((self._q(u) - u).detach() + u) * self.alpha
        return self._q(self.S) * self.alpha + d

    def forward(self, W):
        if not LoRAParam.enabled:
            return W
        return W + self.delta_w().to(W.dtype)

    @torch.no_grad()
    def fold(self):
        if self.mode != "ef":
            return
        d = self.B @ self.A
        self.fold_kept += float((d * self.mask).pow(2).sum())
        self.fold_lost += float((d * ~self.mask).pow(2).sum())
        self.S.add_(d * self.mask / self.alpha.clamp_min(1e-12))
        self.B.zero_()
        self.A.copy_(torch.randn_like(self.A) / math.sqrt(self.A.shape[1]))

    @torch.no_grad()
    def codes(self):
        if self.mode == "ste":
            return self._q((self.B @ self.A) * self.mask / self.alpha.clamp_min(1e-12))
        return self._q(self.S) * self.mask


def set_enabled(on):
    FSAParam.enabled = on
    LoRAParam.enabled = on


def moe_layers(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "experts") and isinstance(m.experts, nn.ModuleList)]


def pair8_selected_pairs(W):
    """[blocks, 4] bool: the 2 pairs each 8-weight block of the pair8 base keeps. Selected = the block's nonzero
    pairs; a block with fewer than 2 nonzero pairs is filled to 2 with its lowest-index zero pairs (the production
    quantizer's choice is not recoverable from the export, so this deterministic rule defines the shipped mask)."""
    act = (W.reshape(-1, 4, 2) != 0).any(-1)
    nact = act.sum(-1)
    assert int(nact.max()) <= 2 if nact.numel() else True, "base is not pair8 (a block has > 2 active pairs)"
    need = (2 - nact).clamp_min(0)
    inact = ~act
    rank_ = torch.cumsum(inact.to(torch.int8), -1)            # 1-based order of the zero pairs within the block
    return act | (inact & (rank_ <= need[:, None]))


def pair8_forced_mask(W):
    """Writable positions of a 'pair8_forced' patch: every weight of the 2 pairs the base did NOT select (exactly
    half of each block). All are zero in the base; the patched model may leave pair8; revoke = zero them again."""
    sel = pair8_selected_pairs(W)
    m = (~sel)[..., None].expand(-1, -1, 2).reshape(W.shape)
    assert not bool((W[m] != 0).any()), "forced positions must be zero in the base"
    return m


def attach(model, layer_ids, method, rank, fmt="ternary", projs=("up_proj", "down_proj"), keep=None, budgeted=False,
           shared=False, routed=True, sparse=False, fold_mode="ste", occupied=False, slot_frac=1.0, seed=0,
           fold_any_zero=False, restrict=None):
    """restrict: optional {linear name: ('rows'|'cols', bool[dim])} - only those rows/cols of the mask stay
    writable (sleeping-unit gating: W_up rows / W_down columns of the selected hidden units)."""
    """keep: optional {layer_id: set(expert_idx)}; experts outside it are left untouched. shared: also patch the
    layer's shared expert(s) (every token passes through them); routed=False patches the shared experts only."""
    params, n_slots, n_trainable = [], 0, 0
    layers = moe_layers(model)
    gen = torch.Generator().manual_seed(seed + 41)
    for li in layer_ids:
        name, moe = layers[li]
        targets = [(ei, e) for ei, e in enumerate(moe.experts)] if routed else []
        if shared:
            targets += [(f"s{si}", se) for si, se in enumerate(getattr(moe, "shared_experts", None) or [])]
        for ei, e in targets:
            if e is None or (keep is not None and isinstance(ei, int) and ei not in keep[li]):
                continue
            for proj in projs:
                lin = getattr(e, proj)
                W = lin.weight.detach()
                if method == "lora":
                    p = LoRAParam(W, rank)
                    n_trainable += p.A.numel() + p.B.numel()
                elif method == "lora_fold":
                    pr = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(W.shape)
                    mask = (pair8_forced_mask(W) if fold_any_zero == "pair8_forced" else
                            (W == 0) if fold_any_zero else (W == 0) & pr)
                    assert bool((W[mask] == 0).all()), "patch mask must contain only base zeros"
                    p = FoldLoRAParam(W, rank, mask, fold_mode)
                    n_trainable += p.A.numel() + p.B.numel()
                    n_slots += int(mask.sum())
                else:
                    pr = (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(W.shape)
                    if occupied and method == "fsa_forced":
                        raise ValueError("fsa_forced writes only the forced pairs: occupied sign flips not allowed")
                    mask = ((W == 0) & pr if method == "fsa_free" else pair8_forced_mask(W) if method == "fsa_forced"
                            else (W == 0))
                    if restrict is not None:
                        key = f"{name}.experts.{ei}.{proj}"
                        if key in restrict:
                            axis, sel = restrict[key]
                            sel = sel.to(mask.device)
                            mask = mask & (sel[:, None] if axis == "rows" else sel[None, :])
                        else:
                            mask = torch.zeros_like(mask)
                    assert bool((W[mask] == 0).all()), "patch mask must contain only base zeros"
                    if sparse:
                        p = SparseFSAParam(W, mask, fmt=fmt, budgeted=budgeted, occupied=occupied,
                                           slot_frac=slot_frac, gen=gen)
                        n_slots += p.M.numel()
                    else:
                        p = FSAParam(W, mask, fmt=fmt, budgeted=budgeted)
                        n_slots += int(mask.sum())
                P.register_parametrization(lin, "weight", p)
                params += list(p.parameters())
    return params, n_slots, n_trainable


CKPT = {"on": True}      # expert checkpointing is suspended while the local penalty's hooks are recording


def checkpoint_experts(model, layer_ids, shared=False):
    """Wrap every routed (and optionally shared) expert of the given MoE layers in non-reentrant activation
    checkpointing while grads are enabled: patched weights W + dW and expert activations are recomputed in
    backward instead of being held, so memory no longer scales with the number of patched layers. Module hooks
    still fire on the outer call."""
    from torch.utils.checkpoint import checkpoint
    layers = moe_layers(model)
    n = 0
    for li in layer_ids:
        _, moe = layers[li]
        mods_ = list(moe.experts) + (list(getattr(moe, "shared_experts", None) or []) if shared else [])
        for e in mods_:
            if e is None or getattr(e, "_ckpt_wrapped", False):
                continue
            f0 = e.forward

            def fwd(*args, _f=f0, **kw):
                if torch.is_grad_enabled() and CKPT["on"]:
                    return checkpoint(_f, *args, use_reentrant=False, **kw)
                return _f(*args, **kw)
            e.forward = fwd
            e._ckpt_wrapped = True
            n += 1
    return n


def detach_all(model):
    for _, m in list(model.named_modules()):
        if P.is_parametrized(m):
            P.remove_parametrizations(m, "weight", leave_parametrized=False)


def fsa_modules(model):
    return [m for m in model.modules() if isinstance(m, FSAParam)]


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRAParam)
            and not isinstance(m, (FoldLoRAParam, TeacherLoRAParam))]


def fold_modules(model):
    return [m for m in model.modules() if isinstance(m, FoldLoRAParam)]


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
    assert not any(isinstance(m, TeacherLoRAParam) for m in model.modules()), "teacher present at export"
    assert all(getattr(m, "forward_keep", None) is None for m in fsa_modules(model)), "dropout active at export"
    ent = {}
    for name, lin in patched_linears(model):
        pm = lin.parametrizations.weight[0]
        if isinstance(pm, FSAParam):
            q = pm.codes().flatten()
            idx = q.nonzero().squeeze(1)
            v = q[idx]
            e = dict(kind="fsa", fmt=pm.fmt, shape=list(pm.wshape), idx=idx.to(torch.int32).cpu(),
                     val=(v.to(torch.int8) if pm.fmt == "ternary" else v.float()).cpu(),
                     alpha=pm.alpha.detach().float().squeeze(1).cpu(), fp4_step=pm.fp4_step,
                     scale="block" if pm.logb is not None else "row")
            if pm.logb is not None:
                e.update(block=pm.block, logb=pm.logb.detach().float().cpu())
            if getattr(pm, "pos_only", None) is not None:  # sign flips of occupied weights (W -> -W there)
                e["flip_idx"] = pm.flip_codes().flatten().nonzero().squeeze(1).to(torch.int32).cpu()
        elif isinstance(pm, FoldLoRAParam):                # pure ternary free-slot patch (no adapter)
            q = pm.codes().flatten()
            idx = q.nonzero().squeeze(1)
            e = dict(kind="fsa", fmt="ternary", shape=list(pm.wshape), idx=idx.to(torch.int32).cpu(),
                     val=q[idx].to(torch.int8).cpu(), alpha=pm.alpha.detach().float().squeeze(1).cpu(),
                     fp4_step=0.25, scale="row")
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
            assert isinstance(pm, FSAParam) and pm.fmt == e["fmt"] and list(pm.wshape) == e["shape"], name
            assert e["scale"] == ("block" if pm.logb is not None else "row"), f"{name}: scale mode differs"
            assert e["fmt"] != "fp4" or e["fp4_step"] == pm.fp4_step, f"{name}: fp4 step differs"
            assert torch.allclose(e["alpha"], pm.alpha.float().squeeze(1).cpu()), f"{name}: base row scale differs"
            idx = e["idx"].long().to(pm.M.device)
            pm.M.zero_()
            if getattr(pm, "sparse", False):              # flat weight positions -> slot positions
                pidx = pm.idx.long()
                pos = torch.searchsorted(pidx, idx).clamp(max=max(pidx.numel() - 1, 0))
                assert idx.numel() == 0 or bool((pidx[pos] == idx).all()), "patch writes outside the slot mask"
                idx = pos
            pm.M.view(-1)[idx] = e["val"].float().to(pm.M.device)
            assert bool(pm.mask.reshape(-1)[idx].all()), "patch writes outside the slot mask"
            if "flip_idx" in e and e["flip_idx"].numel():
                assert getattr(pm, "pos_only", None) is not None, f"{name}: patch has sign flips, attach lacks them"
                fidx = e["flip_idx"].long().to(pm.M.device)
                pidx = pm.idx.long()
                fpos = torch.searchsorted(pidx, fidx).clamp(max=max(pidx.numel() - 1, 0))
                assert bool((pidx[fpos] == fidx).all()) and bool(pm.occ[fpos].all()), "flip outside occupied set"
                pm.M.view(-1)[fpos] = 1.0
                if pm.support is not pm.mask:
                    pm.support[fpos] = True
            if pm.support is not pm.mask:
                pm.support.reshape(-1)[idx] = True
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
    dense_bits = {"ternary": math.log2(3), "tern2": 2 * math.log2(3), "fp4": 4, "fp8": 8, "bf16": 16}[fmt]
    nz_bits = {"ternary": 1, "tern2": 3, "fp4": 4, "fp8": 8, "bf16": 16}[fmt]
    pos_bits = math.ceil(math.log2(max(n_slots, 2)))
    extra = 32 * n_scales
    dense = (n_slots * dense_bits + extra) / 8
    sparse = (nnz * (pos_bits + nz_bits) + extra) / 8
    p_ = nnz / max(n_slots, 1)
    h2 = 0.0 if p_ in (0.0, 1.0) else -(p_ * math.log2(p_) + (1 - p_) * math.log2(1 - p_))
    entropy = (n_slots * h2 + nnz * nz_bits + extra) / 8      # occupancy entropy-coded + value bits
    return dict(bytes=min(dense, sparse), bytes_dense=dense, bytes_sparse=sparse, bytes_entropy=entropy,
                pos_bits=pos_bits)


@torch.no_grad()
def serving_stats(model):
    """Patch nonzeros (incl. sign flips), 8-weight blocks touched, and touched blocks whose patched weights have
    more than 2 active pairs (no longer pair8: need a plain-ternary re-encode or a sparse overlay)."""
    nnz = blocks = non_pair8 = total_blocks = 0
    for _, lin in patched_linears(model):
        pm = lin.parametrizations.weight[0]
        if isinstance(pm, LoRAParam) and not isinstance(pm, FoldLoRAParam):
            return None
        W0 = lin.parametrizations.weight.original
        c = pm.codes() if not isinstance(pm, FoldLoRAParam) else pm.codes()
        if getattr(pm, "pos_only", None) is not None:
            c = c + pm.flip_codes()
        touched = (c != 0).reshape(-1, 8).any(-1)
        base_p = (W0 != 0).reshape(-1, 4, 2).any(-1).sum(-1)                  # active pairs before / after
        new_p = ((W0 != 0) | (c != 0)).reshape(-1, 4, 2).any(-1).sum(-1)
        nnz += int((c != 0).sum())
        blocks += int(touched.sum())
        non_pair8 += int((touched & (new_p > base_p) & (new_p > 2)).sum())   # patch opened a 3rd/4th pair
        total_blocks += touched.numel()
    return dict(nnz=nnz, blocks_touched=blocks, blocks_non_pair8=non_pair8, total_blocks=total_blocks,
                frac_blocks_touched=blocks / max(total_blocks, 1))


MDL_REF_TOKENS = 1_000_000


def mdl_bits(base_test, test, big, storage_, ref_tokens=MDL_REF_TOKENS, lora_precision="fp32"):
    """Net description-length gain in bits: answer bits saved on the held-out test phrasings (base - patched)
    - general-text cost (40x512 panel dNLL x ref_tokens, in bits) - patch bits (FSA: entropy-coded occupancy +
    value bits; LoRA: factors at lora_precision incl. scales)."""
    if base_test is None or test is None or "answer_bits" not in test:
        return None
    saved = base_test["answer_bits"] - test["answer_bits"]
    cost = (big["dnll"] * ref_tokens / math.log(2)) if big else None
    if storage_ is None:
        pbits = None
    elif "fsa" in storage_:
        pbits = 8 * storage_["fsa"]["bytes_entropy"]
    elif "lora" in storage_:
        pbits = 8 * storage_["lora"][lora_precision]
    else:
        pbits = None
    net = saved - cost - pbits if (cost is not None and pbits is not None) else None
    return dict(ref_tokens=ref_tokens, fact_bits_saved=saved, general_cost_bits=cost,
                general_cost_bits_se=(big["dnll_se"] * ref_tokens / math.log(2)) if big else None,
                patch_bits=pbits, net_bits=net, answer_tokens=test.get("answer_tokens"))


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
        if getattr(m, "sparse", False):
            raise NotImplementedError("energy-scaled tau is not supported with sparse slot state")
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
    if mode == "fisher":
        # Frozen Adam second moment in code coordinates; alpha is already inside g and v.
        # Rank by g^2/(v + 1e-8) - lambda * isolated local-output cost, not signed netcost.
        return (m.score.float().square() / (m.fisher + 1e-8) - cost_lambda * slot_cost(m, den)) * m.mask
    if mode == "netcost":
        g = m.score.float()                            # signed EMA of dL/dM (alpha already inside)
        Mv = m.M.detach()
        direction = torch.where(Mv != 0, torch.sign(Mv), -torch.sign(g))   # existing sign, else descent sign
        if getattr(m, "pos_only", None) is not None:       # sign-flip slots can only move to +1
            direction = torch.where(m.pos_only, torch.ones_like(direction), direction)
        if m.fmt == "ternary":
            mag = torch.ones_like(Mv)
        elif m.fmt == "fp4":
            mag = Mv.abs().clamp(min=0.5 * m.fp4_step)
        elif m.fmt == "tern2":
            mag = Mv.abs().clamp(min=0.5)
        else:
            mag = Mv.abs().clamp(min=1e-3)
        gain = -g * direction * mag
        cost = slot_cost(m, den) * mag ** 2
        return (gain - cost_lambda * cost) * m.mask
    return m.score.float() * m.mask


@torch.no_grad()
def conflict_pool_scores(mods, grads, beta=1.0):
    """grads[p][i]: person p's summed answer-loss gradient for module i (shape of M). For every slot, the best
    person's first-order gain -s*g_p for the sign s = -sign(g_p) that helps p (+1 only on sign-flip slots), minus
    beta x the predicted worsening of the other people under that same sign, sum_q max(0, s*g_q); g = dL/dM, so
    the slot scale is already inside. Returns a per-module score
    callable for refresh_support (only positive scores are kept)."""
    def score(i, m):
        best = None
        for p in range(len(grads)):                        # grads are dL/dM: the slot scale is already inside
            g_p = grads[p][i]
            if g_p is None:
                continue
            s_ = -torch.sign(g_p)
            if getattr(m, "pos_only", None) is not None:   # sign-flip slots: only +1 is legal
                s_ = torch.where(m.pos_only, torch.ones_like(s_), s_)
            gain = -s_ * g_p
            worse = torch.zeros_like(gain)
            for q in range(len(grads)):
                if q != p and grads[q][i] is not None:
                    worse += (s_ * grads[q][i]).clamp_min(0)
            sc = (gain - beta * worse) * m.mask
            best = sc if best is None else torch.maximum(best, sc)
        return best if best is not None else torch.zeros_like(m.M)
    return score


@torch.no_grad()
def add_random_support(mods, n, gen):
    """Add n random free, not-yet-supported slots to the support, spread proportionally to free counts."""
    free = [int((writable_slots(m) & ~m.support).sum()) for m in mods]
    tot = sum(free)
    n = min(n, tot)
    if n <= 0:
        return 0
    quota = [n * f // tot for f in free]                   # exact allocation: floor shares + largest remainders
    rem = sorted(range(len(free)), key=lambda i: -((n * free[i]) % tot))
    for i in rem[:n - sum(quota)]:
        quota[i] += 1
    added = 0
    for m, f, k in zip(mods, free, quota):
        if k <= 0:
            continue
        k = min(k, f)
        cand = (writable_slots(m) & ~m.support).flatten().nonzero().squeeze(1)
        pick = cand[torch.randperm(cand.numel(), generator=gen, device="cpu")[:k].to(cand.device)]
        m.support.view(-1)[pick] = True
        added += k
    return added


def refresh_support(mods, K, scores=None, opt=None, churn=None, keep_live=False,
                    swap_margin=0.0, swap_limit=None):
    """Keep K positive-score slots. keep_live (also retain_evicted) pins all live codes, masters and EMA.

    With swap controls, retain incumbents unless a positive challenger beats the weakest unpinned
    incumbent by swap_margin * population std(writable scores). Count additions and removals separately
    against swap_limit; filling a vacancy costs one, replacing costs two. Live slots are never swapped
    under keep_live. A zero limit freezes support, including empty vacancies.
    """
    if swap_limit == 0:
        assert sum(int(m.support.sum()) for m in mods) <= K
        return
    pin = keep_live or any(hasattr(m, "wake_rows") for m in mods)
    nlive = sum(int(pinned_slots(m, keep_live).sum()) for m in mods) if pin else 0
    if nlive > K:
        raise ValueError(f"pinned live slots ({nlive}) exceed support budget ({K})")
    available = K - nlive
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
        if keep_live or hasattr(m, "wake_rows"):
            sc = sc.masked_fill(~writable_slots(m).flatten() | pinned_slots(m, keep_live).flatten(), -float("inf"))
        npos = int((sc > 0).sum())
        if npos and available:
            v, ix = torch.topk(sc, min(available, npos))
            ii = torch.stack([torch.full_like(ix, i), ix], 1)
            if best_v is not None:
                v, ii = torch.cat([best_v, v]), torch.cat([best_i, ii])
            top = torch.topk(v, min(available, v.numel())).indices   # exact indices: ties cannot exceed K
            best_v, best_i = v[top], ii[top]
        del sc
    keep = {}
    if best_v is not None:
        for mi in best_i[:, 0].unique().tolist():
            keep[mi] = best_i[best_i[:, 0] == mi, 1]
    if swap_margin or swap_limit is not None:
        keep = constrained_support(mods, K, keep, get, keep_live, swap_margin, swap_limit)
    for i, m in enumerate(mods):
        new = torch.zeros(m.mask.numel(), dtype=torch.bool, device=m.mask.device)
        if i in keep:
            new[keep[i]] = True
        new = new.view(m.M.shape) & m.mask
        if keep_live or hasattr(m, "wake_rows"):
            new |= pinned_slots(m, keep_live)
        ev = m.support & ~new
        if churn is not None:                              # live slots removed by the budget (not Bop deaths)
            live_ev = ev & ((m._quant(m.M.detach() * m.mask) * m.support) != 0)   # quantized liveness, state coords
            churn["evictions"] += int(live_ev.sum())
            if hasattr(m, "nbirth"):
                churn["evictions_young"] += int((live_ev & (m.age < YOUNG)).sum())
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


def constrained_support(mods, K, candidates, get, keep_live, margin, limit):
    """Vectorized global swaps over <= K incumbents and <= K top challengers, never all slots.

    Fill vacancies with the best challengers, then match remaining challengers (descending)
    against movable incumbents (ascending). Passing pairs form a prefix, so no greedy Python
    scan or per-slot GPU transfers are needed even for million-slot budgets.
    """
    device = mods[0].M.device
    incumbent_keys, movable_keys, movable_scores, challenger_keys, challenger_scores = [], [], [], [], []
    count, total, squares = 0, 0.0, 0.0
    for i, m in enumerate(mods):
        sc = get(i, m).flatten()
        writable = sc[writable_slots(m).flatten()].double()
        count += writable.numel()
        total += float(writable.sum())
        squares += float(writable.square().sum())
        del writable
        ix = m.support.flatten().nonzero().flatten()
        keys = torch.stack((torch.full_like(ix, i), ix), 1)
        incumbent_keys.append(keys)
        movable = (~frozen_slots(m).expand_as(m.M).flatten()[ix] if hasattr(m, "wake_rows")
                   else torch.ones_like(ix, dtype=torch.bool))
        if keep_live:
            q = (m._quant(m.M.detach() * m.mask) * m.support).flatten()
            movable &= q[ix] == 0
        movable_keys.append(keys[movable])
        movable_scores.append(sc[ix[movable]])
        cx = candidates.get(i)
        if cx is not None:
            cx = cx[~m.support.flatten()[cx]]
            challenger_keys.append(torch.stack((torch.full_like(cx, i), cx), 1))
            challenger_scores.append(sc[cx])
    kept = torch.cat(incumbent_keys)
    assert kept.shape[0] <= K, "swap controls require incumbents within the support budget"
    if challenger_keys:
        ck, cv = torch.cat(challenger_keys), torch.cat(challenger_scores)
        order = torch.argsort(cv, descending=True, stable=True)
        ck, cv = ck[order], cv[order]
        remaining = 2 * K if limit is None else limit
        fills = min(K - kept.shape[0], cv.numel(), remaining)
        remaining -= fills
        mk, mv = torch.cat(movable_keys), torch.cat(movable_scores)
        order = torch.argsort(mv, stable=True)
        mk, mv = mk[order], mv[order]
        trials = min(cv.numel() - fills, mv.numel(), remaining // 2)
        threshold = margin * math.sqrt(max(0.0, squares / max(count, 1) - (total / max(count, 1)) ** 2))
        swaps = int((cv[fills:fills + trials] > mv[:trials] + threshold).sum())
        if swaps:
            # Encode (module, slot) uniquely to remove only the swapped incumbents.
            stride = max(m.M.numel() for m in mods)
            old_ids = kept[:, 0] * stride + kept[:, 1]
            removed = mk[:swaps, 0] * stride + mk[:swaps, 1]
            kept = kept[~torch.isin(old_ids, removed, assume_unique=True)]
        kept = torch.cat((kept, ck[:fills + swaps]))
    return {i: kept[kept[:, 0] == i, 1].to(device) for i in range(len(mods))}


FLIP_TAU_MULT = 2.0   # sign flips of occupied weights move 2|w|: need twice the evidence of a free-slot birth
YOUNG = 50          # a death at age < YOUNG steps counts as "born-then-killed"
AGE_CAP = 127       # int8 saturating counters


def init_churn(m):
    """Per-slot int8 counters (saturating at 127): births so far, and age (steps since the last birth, counted
    while the slot is nonzero). Together they give born-then-killed and flip-flop ("persistent
    re-initialization") statistics and drive the maturity grace."""
    m.nbirth = torch.zeros(m.M.shape, dtype=torch.int8, device=m.M.device)
    m.age = torch.zeros(m.M.shape, dtype=torch.int8, device=m.M.device)
    m.lastsign = torch.zeros(m.M.shape, dtype=torch.int8, device=m.M.device)   # sign of the previous life


def init_step_age(m):
    """Optional int16 wall-step age, independent of churn counters and Bop accumulation."""
    m.age = torch.zeros(m.M.shape, dtype=torch.int16, device=m.M.device)
    m.age_in_steps = True


@torch.no_grad()
def advance_step_age(mods):
    for m in mods:
        if getattr(m, "age_in_steps", False):
            live = (m._quant(m.M) * m.support != 0) & (m.age < 32767)
            m.age[live] += 1


@torch.no_grad()
def patch_chaos(mods, dropout, reheat, generator):
    """Training-only forward masks; zero EMA and halve age on a seeded live-slot reheat sample."""
    for m in mods:
        live = (m._quant(m.M) * m.support) != 0
        m.forward_keep = ((torch.rand(m.M.shape, generator=generator) >= dropout).to(m.M.device)
                          if dropout else None)
        if reheat and hasattr(m, "bop_ema"):
            chosen = live & (torch.rand(m.M.shape, generator=generator).to(m.M.device) < reheat)
            m.bop_ema[chosen] = 0
            m.age[chosen] //= 2


def clear_patch_dropout(mods):
    for m in mods:
        m.forward_keep = None


def new_churn_totals():
    return dict(births=0, deaths=0, deaths_young=0, deaths_blocked_by_grace=0, rebirths=0, evictions=0,
                evictions_young=0, sign_reversals=0)


@torch.no_grad()
def bop_step(mods, gamma, tau, budgeted, allow_birth=True, grace=0, death_mult=1.0, churn=None, keep_ema=False,
             pair_birth=False, death_age_mult=False, birth_tau=None):
    """Ternary Bop (latent-free; Helwegen et al. 2019 extended to {-1,0,+1}): an EMA of the gradient per slot;
    birth 0 -> -sign(ema) when |ema| > tau_birth; death +-1 -> 0 when the ema says the current sign hurts (same sign
    as the value) and |ema| > tau_death. The ema resets on every change, so tau directly sets the flip rate.
    tau_birth = tau * m.tau_scale (per row) when energy scaling is on, else tau; tau_death = death_mult * tau.
    allow_birth=False (repair phase) permits deaths only. grace > 0 (needs init_churn): no Bop death until a slot
    has been alive for `grace` steps (maturity; budget evictions in refresh_support are not subject to it). churn: optional totals dict (see new_churn_totals), updated in place
    when the modules carry churn counters. Returns (#births, #deaths). M holds the ternary value (|M| in {0, 1})."""
    assert grace <= AGE_CAP
    nb = nd = 0
    for m in mods:
        track = hasattr(m, "age")
        if m.M.grad is None:
            m.bop_ema.mul_(1 - gamma)
        else:
            m.bop_ema.mul_(1 - gamma).add_(m.M.grad * m.mask, alpha=gamma)
            q = torch.sign(m.M) * (m.M.abs() > 0.5)
            mag = m.bop_ema.abs()
            tau_b = tau if birth_tau is None else birth_tau
            tau_b = tau_b * m.tau_scale if getattr(m, "tau_scale", None) is not None else tau_b
            if getattr(m, "pos_only", None) is not None:   # sign-flip slots: stricter evidence, +1 only
                tau_b = torch.where(m.pos_only, torch.as_tensor(tau_b * FLIP_TAU_MULT, device=m.M.device),
                                    torch.as_tensor(tau_b, device=m.M.device))
            allowed = writable_slots(m) & (m.support if budgeted else m.mask)
            birth = (q == 0) & (mag > tau_b) & allowed if allow_birth else torch.zeros_like(m.mask)
            if pair_birth and allow_birth:
                birth = pair_birth_mask(m, q, mag, tau_b, allowed)
            if getattr(m, "pos_only", None) is not None:
                birth = birth & ~(m.pos_only & (m.bop_ema > 0))   # -sign(ema) would be -1: not a legal flip
            tau_d = tau * death_mult
            if death_age_mult:
                tau_d = tau_d * (1 + 2 * torch.tanh(m.age.float() / 50))
            death = (q != 0) & (mag > tau_d) & (torch.sign(m.bop_ema) == q) & writable_slots(m)
            if grace and track:
                blocked = death & (m.age < grace)
                death &= ~blocked
                if churn is not None and hasattr(m, "nbirth"):
                    churn["deaths_blocked_by_grace"] += int(blocked.sum())
            m.M[birth] = -torch.sign(m.bop_ema[birth])
            m.M[death] = 0
            if not keep_ema:                               # keep_ema: evidence persists across a flip (hysteresis)
                m.bop_ema[birth | death] = 0
            nb_, nd_ = int(birth.sum()), int(death.sum())
            nb += nb_; nd += nd_
            if churn is not None:
                churn["births"] += nb_; churn["deaths"] += nd_
            if track:
                if churn is not None and hasattr(m, "nbirth"):
                    churn["deaths_young"] += int((death & (m.age < YOUNG)).sum())
                    churn["rebirths"] += int((birth & (m.nbirth > 0)).sum())
                    newsign = torch.sign(m.M).to(torch.int8)
                    churn["sign_reversals"] += int((birth & (m.lastsign != 0) & (m.lastsign != newsign)).sum())
                if hasattr(m, "nbirth"):
                    m.lastsign[birth] = torch.sign(m.M[birth]).to(torch.int8)
                    m.nbirth[birth] = torch.clamp(m.nbirth[birth].to(torch.int16) + 1, max=AGE_CAP).to(torch.int8)
                m.age[birth] = 0
        if track and not getattr(m, "age_in_steps", False):
            live = (m.M != 0) & (m.age < AGE_CAP)
            m.age[live] += 1
    return nb, nd


def pair_birth_mask(m, q, mag, tau, allowed):
    """Joint minimum EMA dot d over {-1,0,1}², including the zero pattern.

    Each nonzero leg needs |EMA| > .5*tau and their sum must exceed .75*sum(tau).
    Choose both opposite-EMA signs when the pair passes, otherwise eligible singletons
    (|EMA| > .75*tau). Two .8*tau legs can therefore be born together. With row tau scaling,
    sum(tau) is the sum of the participating legs' thresholds. A live partner blocks birth;
    missing sparse legs are fixed zero. Pair IDs use weight coordinates, never reserve indices.
    """
    if m.wshape[1] % 2:
        raise ValueError("pair births require an even projection width")
    shape = q.shape
    q, mag, allowed = q.flatten(), mag.flatten(), allowed.flatten()
    thresholds = torch.broadcast_to(torch.as_tensor(tau, device=q.device), shape).flatten()
    if getattr(m, "sparse", False):
        keys = m.rows() * (m.wshape[1] // 2) + m.cols() // 2
        same = keys[1:] == keys[:-1]
        partner = torch.arange(q.numel(), device=q.device)
        left = same.nonzero().flatten()
        partner[left] = left + 1
        partner[left + 1] = left
        present = partner != torch.arange(q.numel(), device=q.device)
    else:
        partner = torch.arange(q.numel(), device=q.device) ^ 1
        present = torch.ones_like(allowed)
    eligible = (q == 0) & allowed & (mag > .5 * thresholds)
    dormant = (q == 0) & (~present | (q[partner] == 0))
    joint = (present & eligible & eligible[partner] & dormant &
             (mag + mag[partner] > .75 * (thresholds + thresholds[partner])))
    singleton = eligible & (mag > .75 * thresholds) & dormant
    return (joint | singleton).reshape(shape)


@torch.no_grad()
def capture_fisher(mods, opt):
    """Copy before clearing Adam state; unvisited slots have zero second-moment evidence."""
    for m in mods:
        v = opt.state.get(m.M, {}).get("exp_avg_sq")
        m.fisher = v.detach().float().clone() if v is not None else torch.zeros_like(m.M)


@torch.no_grad()
def accumulate_epoch_scores(mods):
    """Signed answer-gradient SUM, including zero evidence for missing grads, without EMA decay."""
    for m in mods:
        if m.M.grad is not None:
            m.score.add_(m.M.grad * m.mask)


@torch.no_grad()
def churn_summary(mods):
    """Per-slot history at the end of training: how many slots were ever born, born >= 2 / >= 3 times
    (flip-flop / persistent re-initialization), and how many of the final nonzeros are first births."""
    ever = twice = thrice = alive = alive_first = 0
    mx = 0
    for m in mods:
        if not hasattr(m, "nbirth"):
            return None
        nbv = m.nbirth
        if nbv.numel() == 0:
            continue
        ever += int((nbv > 0).sum()); twice += int((nbv >= 2).sum()); thrice += int((nbv >= 3).sum())
        nz = m.M != 0
        alive += int(nz.sum()); alive_first += int((nz & (nbv == 1)).sum())
        mx = max(mx, int(nbv.max()))
    return dict(slots_ever_born=ever, slots_born_2plus=twice, slots_born_3plus=thrice, max_births_one_slot=mx,
                alive=alive, alive_first_birth=alive_first)


def patch_stats(mods):
    nnz, energy = 0, 0.0
    with torch.no_grad():
        for m in mods:
            d = m.codes() * m.alpha if isinstance(m, FoldLoRAParam) else m.delta()   # (sparse delta incl. flips)
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


class SaturationSampler:
    """Online two-phrasing mastery; no probe passes. The 40-step floor can enlarge a small batch.

    running accuracy is the cumulative fraction of correct answer tokens on sampled training
    phrasings. Stop tokens are excluded from mastery. CE masking uses state at batch entry;
    a failed phrasing reopens the fact for the next batch. Augmentation contributes three
    chronological observations, including across microbatch boundaries.
    """
    def __init__(self, n, margin, weight, seed):
        self.history = [[] for _ in range(n)]
        self.correct = [0] * n
        self.tokens = [0] * n
        self.last = [-1] * n
        self.margin, self.weight = margin, weight
        self.rng = random.Random(seed)
        self.reopened = 0
        self.sampled = 0

    def saturated(self, fi):
        return len(self.history[fi]) == 2 and all(self.history[fi])

    def weights(self):
        return [self.weight if self.saturated(i) else 1 + 3 * (1 - c / max(t, 1))
                for i, (c, t) in enumerate(zip(self.correct, self.tokens))]

    def batch(self, size, step):
        n = len(self.last)
        chosen = [i for i in range(n) if step - self.last[i] >= 40]
        remaining = [i for i in range(n) if i not in chosen]
        weights = self.weights()
        while len(chosen) < min(size or n, n) and remaining:
            i = self.rng.choices(remaining, weights=[weights[j] for j in remaining])[0]
            chosen.append(i)
            remaining.remove(i)
        return sorted(chosen)

    @torch.no_grad()
    def observe(self, fi, logits, gold, step):
        if not gold.numel():
            return
        was = self.saturated(fi)
        good = logits.argmax(-1) == gold
        own = logits.gather(-1, gold[:, None]).squeeze(-1)
        other = logits.clone().scatter_(-1, gold[:, None], -float("inf")).amax(-1)
        passed = bool((good & (own - other >= self.margin)).all())
        self.history[fi] = (self.history[fi] + [passed])[-2:]
        self.correct[fi] += int(good.sum())
        self.tokens[fi] += gold.numel()
        self.last[fi] = step
        self.sampled += 1
        self.reopened += int(was and not self.saturated(fi))

    def stats(self):
        sat = sum(self.saturated(i) for i in range(len(self.last)))
        return dict(sat_saturated=sat, sat_unsaturated=len(self.last) - sat,
                    sat_reopened=self.reopened, sat_phrasings=self.sampled)


def build_items(tok, facts, templates, bos, ctx=None, idxs=None):
    """Evaluation prompts: one per (fact, template). ctx: optional {fact index: text} prepended to the prompt."""
    items = []
    for fi in (idxs if idxs is not None else range(len(facts))):
        f = facts[fi]
        for ti, t in enumerate(templates):
            full = fmt(t, f)
            prefix = full[: full.rindex(f.val)].rstrip()
            if ctx is not None:
                prefix = ctx[fi] + prefix
            items.append(dict(fi=fi, ti=ti, attr=f.attr, val=f.val, p=encode(tok, prefix, bos),
                              a=tok.encode(" " + f.val, add_special_tokens=False).ids))
    return items


def build_qa_items(tok, facts, bos, n=120):
    """Final-only QA items, sampled with an isolated fixed RNG; no training/selection data enters here."""
    if n < 0:
        raise ValueError("QA item cap must be nonnegative")
    groups = dict(unseen=[], two_fact=[], indirect=[])

    def item(prompt, fis):
        vals = [facts[fi].val for fi in fis]
        return dict(prompt=prompt, fis=list(fis), vals=vals, val=vals[0], fi=fis[0],
                    p=encode(tok, prompt, bos),
                    a=tok.encode(" " + " and ".join(vals), add_special_tokens=False).ids)

    people, owners = defaultdict(list), defaultdict(set)
    for fi, f in enumerate(facts):
        people[f.pid].append(fi)
        owners[(f.attr, f.val)].add(f.pid)
        for t in QA_UNSEEN_T:
            groups["unseen"].append(item(fmt(t, f), [fi]))
    for fis in people.values():
        for i, fi in enumerate(fis):
            f = facts[fi]
            for fj in fis[i + 1:]:
                g = facts[fj]
                if f.attr != g.attr:
                    prompt = QA_TWO_T.format(name=f.name, attr1=f.attr, attr2=g.attr)
                    groups["two_fact"].append(item(prompt, [fi, fj]))
            if len(owners[(f.attr, f.val)]) == 1:
                for fj in fis:
                    g = facts[fj]
                    if f.attr != g.attr and f.val != g.val:
                        prompt = QA_INDIRECT_T.format(attrX=f.attr, valX=f.val, pos=f.pos, attrY=g.attr)
                        it = item(prompt, [fj])
                        it["reference_fi"] = fi
                        groups["indirect"].append(it)
    rng = random.Random(130013)
    return {key: rng.sample(items, min(n, len(items))) for key, items in groups.items()}


def encode_fact(tok, t, f, bos):
    """Training sequence and first answer position, including the template's clean stop suffix."""
    full = fmt(t, f)
    cut = full.rindex(f.val)
    p_ids = encode(tok, full[:cut].rstrip(), bos)
    return p_ids + tok.encode(" " + f.val + full[cut + len(f.val):], add_special_tokens=False).ids, len(p_ids)


def fact_sheet(facts, mode, pids):
    """Stable person/attribute order, independent of input order and all sampling RNGs.

    A cross-person self-study question in person mode includes both queried people's sheets.
    """
    if mode not in ("person", "all"):
        raise ValueError("context sheet mode must be person or all")
    selected = [f for f in facts if mode == "all" or f.pid in pids]
    groups = defaultdict(list)
    for f in sorted(selected, key=lambda f: (f.pid, f.attr)):
        groups[(f.pid, f.name)].append(f"{f.attr}: {f.val}.")
    return "\n".join(f"Facts about {name}: " + " ".join(entries)
                     for (_, name), entries in groups.items())


def context_sequence(tok, facts, seq, mode, pids, bos):
    """One BOS, then sheet + blank line + the student's exact tokens without its BOS."""
    assert bos is not None and seq[0] == bos
    prefix = encode(tok, fact_sheet(facts, mode, pids) + "\n\n", bos)
    return prefix + list(seq[1:])


def align_context_logits(logits, seq_len):
    """The final context token stands in for student BOS; shift to next-token labels afterwards."""
    assert 1 <= seq_len <= logits.shape[-2]
    # clone releases the long teacher-sequence storage even when logits already are fp32.
    return logits[..., -seq_len:, :VOCAB].float().clone()


def context_teacher_logits(model, tok, facts, x, lengths, pids, mode, bos):
    """Teacher microbatch = 1, no autograd, no target/sheet cache across calls or steps."""
    target = None
    with context_base_forward():
        for r, (length, people) in enumerate(zip(lengths, pids)):
            seq = context_sequence(tok, facts, x[r, :length].tolist(), mode, people, bos)
            tx = torch.tensor([seq], dtype=torch.long, device=x.device)
            row = align_context_logits(logits_of(model, tx), length)[0]
            if target is None:
                target = row.new_zeros((x.shape[0], x.shape[1], row.shape[-1]))
            target[r, :length] = row
    return target


def build_self_study_items(tok, facts, n, seed, step, bos):
    """N labelled synthetic questions, using a separate deterministic RNG for every step."""
    rng = random.Random(seed + 100003 + step)
    by_person, by_attr = defaultdict(list), defaultdict(list)
    for fi, f in enumerate(facts):
        by_person[f.pid].append(fi)
        by_attr[f.attr].append(fi)
    items = []
    for _ in range(n):
        fi = rng.randrange(len(facts))
        f = facts[fi]
        own = [j for j in by_person[f.pid] if j != fi]
        others = [j for j in by_attr[f.attr] if facts[j].pid != f.pid]
        choices = list(range(3)) + ([3, 4] if own else []) + ([5, 6] if others else [])
        ti = rng.choice(choices)
        second = facts[rng.choice(own)] if ti in (3, 4) else None
        other = facts[rng.choice(others)] if ti in (5, 6) else None
        answer = f"{f.val}."
        if second is not None:
            answer = f"{f.name}'s {f.attr} is {f.val} and {second.attr} is {second.val}."
        if other is not None:
            answer = (f"Yes, {f.name}'s {f.attr} and {other.name}'s are both {f.val}." if f.val == other.val
                      else f"No, {f.name}'s {f.attr} is {f.val} while {other.name}'s {other.attr} is {other.val}.")
        prompt = SELF_STUDY_T[ti].format(name=f.name, pos=f.pos, attr=f.attr,
                                        attr2=second.attr if second else "", other=other.name if other else "",
                                        answer="").rstrip()
        prefix = encode(tok, prompt, bos)
        seq = prefix + tok.encode(" " + answer, add_special_tokens=False).ids
        items.append(dict(seq=seq, start=len(prefix), pids=tuple(sorted({f.pid} | ({other.pid} if other else set()))),
                          fi=fi, ti=ti, answer=answer))
    return items


def context_teacher_check(model, tok, facts, items_val, items_test, dev, bos, a):
    result = {}
    with context_base_forward():
        for mode in ("person", "all"):
            scores = {}
            for split, items in (("val", items_val), ("test", items_test)):
                prefixed = [dict(it, p=context_sequence(tok, facts, it["p"], mode, (facts[it["fi"]].pid,), bos))
                            for it in items]
                scores[f"{split}_tf"] = tf_eval(model, prefixed, dev, bs=1)
                scores[f"{split}_gen"] = gen_eval(model, tok, prefixed, dev, a.gen_max_new,
                                                  a.gen_stop_k, bs=1, conflicts=conflict_map(facts))
            scores["gate_passed"] = scores["val_gen"]["recall"] >= 0.9
            result[mode] = scores
    return result


def binding_groups(facts, conflicts):
    """Complete two/three-person attribute groups with conflicting answers."""
    by_attr = defaultdict(list)
    for fi, f in enumerate(facts):
        by_attr[f.attr].append(fi)
    return [grp for grp in by_attr.values() if len(grp) in (2, 3)
            and len({facts[fi].pid for fi in grp}) == len(grp) and all(fi in conflicts for fi in grp)]


def binding_candidates(tok, facts, group, template, bos):
    """Row-major (person prompt, candidate answer + stop), with one phrasing for the whole group."""
    return [encode_fact(tok, template, facts[p]._replace(val=facts[q].val), bos)
            for p in group for q in group]


def assignment_loss(scores):
    """Negative log probability of the diagonal assignment among all k! permutations."""
    k = scores.shape[0]
    if scores.ndim != 2 or scores.shape[1] != k or k not in (2, 3):
        raise ValueError("assignment scores must be a 2x2 or 3x3 matrix")
    rows = torch.arange(k, device=scores.device)
    totals = torch.stack([scores[rows, list(perm)].sum() for perm in permutations(range(k))])
    return torch.logsumexp(totals, 0) - scores.diagonal().sum()


def binding_scores(model, candidates, dev):
    """Length-normalized log P of answer and stop only, one score per candidate."""
    x, lm = batchify([seq for seq, _ in candidates], dev, [start for _, start in candidates])
    mb = lm[:, 1:]
    lg = logits_of(model, x)[:, :-1, :VOCAB].float()
    nll = F.cross_entropy(lg[mb], x[:, 1:][mb], reduction="none")
    lengths = mb.sum(1).tolist()
    return torch.stack([-part.mean() for part in nll.split(lengths)])


def binding_loss_backward(model, candidates, dev, micro, scale=1.0):
    """Exact assignment gradient via score-then-recompute; only one microbatch graph is kept alive.

    The harness uses model.eval(), so both passes score identical deterministic continuations.
    """
    k = math.isqrt(len(candidates))
    if k not in (2, 3) or k * k != len(candidates) or micro < 1:
        raise ValueError("binding needs 4 or 9 candidates and a positive microbatch size")
    with torch.no_grad():
        scores = torch.cat([binding_scores(model, candidates[s:s + micro], dev)
                            for s in range(0, len(candidates), micro)]).reshape(k, k)
    scores.requires_grad_(True)
    loss = assignment_loss(scores)
    weights, = torch.autograd.grad(scale * loss, scores)
    weights = weights.flatten()
    for s in range(0, len(candidates), micro):
        part = binding_scores(model, candidates[s:s + micro], dev)
        if part.requires_grad:
            (part * weights[s:s + micro]).sum().backward()
        del part
    return loss.item()


def train_audit_items(tok, facts, templates, bos, n, seed):
    """Fixed seeded subset, using a separate RNG so training sampling is unaffected."""
    items = build_items(tok, facts, templates, bos)
    return random.Random(seed + 73).sample(items, min(n, len(items)))


def fact_arities(facts):
    """Count distinct people holding each attribute across the full world, not the validation subset."""
    people = defaultdict(set)
    for f in facts:
        people[f.attr].add(f.pid)
    return {fi: len(people[f.attr]) for fi, f in enumerate(facts)}


class AttributeCurriculum:
    """Whole-attribute batches, bounded active window, two successful checks to master, uniform replay.

    Replay's fact quota uses the available active facts and is rounded down to whole groups (at least one
    when it fits); the active quota uses the remainder. Groups are sampled without replacement within a batch.
    A failed replay displaces the newest active group if the window is full, putting that group back at the
    front of the pending queue.
    Checks sample up to three formats per fact with a separate RNG and include the clean stop suffix.
    """
    def __init__(self, facts, pool, fact_batch, active=8, replay=0.25, seed=0):
        by = defaultdict(list)
        for fi, f in enumerate(facts):
            by[f.attr].append(fi)
        self.groups = dict(by)
        self.pool = pool
        self.limit = fact_batch or len(facts)
        if self.limit < max(map(len, self.groups.values())):
            raise ValueError("--fact-batch must fit the largest whole attribute group")
        self.width, self.replay = active, replay
        self.pending = sorted(by, key=lambda at: (-len(by[at]), at))  # three-way before two-way
        self.active, self.mastered, self.streak, self.replayed = [], set(), {}, set()
        self.rng, self.check_rng = random.Random(seed + 83), random.Random(seed + 89)
        self._fill()

    def _fill(self):
        while self.pending and len(self.active) < self.width:
            self.active.append(self.pending.pop(0))

    def batch(self):
        chosen, nf = [], 0
        replay = sorted(self.mastered)
        self.rng.shuffle(replay)                            # uniform over mastered groups
        # Base the replay share on the attainable batch size when the active window is small.
        active_facts = sum(len(self.groups[at]) for at in self.active)
        target = min(self.limit, active_facts / (1 - self.replay))
        quota = int(target * self.replay)
        if replay and self.replay > 0:
            quota = max(quota, min(len(self.groups[at]) for at in replay))
        if not self.active:
            quota = self.limit
        # Keep room for one active group even when replay's whole-group rounding is coarse.
        if self.active:
            quota = min(quota, self.limit - min(len(self.groups[at]) for at in self.active))
        for at in replay:
            n = len(self.groups[at])
            if nf + n <= quota:
                chosen.append(at)
                nf += n
                self.replayed.add(at)
        active = list(self.active)
        self.rng.shuffle(active)
        for at in active:
            n = len(self.groups[at])
            if nf + n <= self.limit:
                chosen.append(at)
                nf += n
        fis = [fi for at in chosen for fi in self.groups[at]]
        all_formats = sum(len(self.pool[fi]) for fi in fis) <= self.limit * 3
        return [(fi, e) for fi in fis for e in (self.pool[fi] if all_formats else
                                               self.rng.sample(self.pool[fi], min(3, len(self.pool[fi]))))]

    def check(self, evaluate):
        checked = list(self.active) + sorted(self.replayed - set(self.active))
        results = {}
        for at in checked:
            items = []
            for fi in self.groups[at]:
                for seq, start in self.check_rng.sample(self.pool[fi], min(3, len(self.pool[fi]))):
                    items.append(dict(fi=fi, p=seq[:start], a=seq[start:]))
            results[at] = evaluate(items)["exact_match"] == 1.0
        for at in checked:
            if at in self.mastered:
                if not results[at]:
                    self.mastered.remove(at)
                    self.streak[at] = 0
                    if len(self.active) == self.width:
                        self.pending.insert(0, self.active.pop())
                    self.active.insert(0, at)
            elif at in self.active:
                self.streak[at] = self.streak.get(at, 0) + 1 if results[at] else 0
                if self.streak[at] >= 2:
                    self.active.remove(at)
                    self.mastered.add(at)
        self.replayed.clear()
        self._fill()


def _eval_bs(bs, maxlen, tok_budget=2048):
    return max(1, min(bs, tok_budget // max(maxlen, 1)))


@torch.no_grad()
def tf_eval(model, items, dev, bs=16, arities=None):
    """Teacher-forced greedy exact match (every answer token is the argmax given the gold prefix) + mean answer
    log-prob. Right-padded batches (causal model: padding after a row cannot affect it)."""
    hits, lps, bits, ntok = 0, [], 0.0, 0
    strata = defaultdict(list) if arities is not None else None
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
            tl = torch.log_softmax(row, -1)[torch.arange(len(pos), device=dev), tgt]
            lps.append(tl.mean().item())
            bits += -float(tl.sum()) / math.log(2)
            ntok += len(pos)
            hit = int(bool((row.argmax(-1) == tgt).all()))
            hits += hit
            if strata is not None:
                strata[arities[it["fi"]]].append(hit)
    n = max(len(items), 1)
    result = {"exact_match": hits / n, "answer_logprob": sum(lps) / n, "answer_bits": bits, "answer_tokens": ntok}
    if strata is not None:
        result["by_arity"] = {k: sum(v) / len(v) for k, v in strata.items()}
    return result


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


def score_two_fact_generation(tok, gen_ids, vals, k=2, eos=EOS_IDS):
    """Each whole answer must occur before the first stop; the last answer must stop as in GEN."""
    e = next((j for j, t in enumerate(gen_ids) if t in eos), len(gen_ids))
    ids = gen_ids[:e]
    s = tok.decode(ids).lstrip()
    # A decimal point inside a word/number is not a sentence stop (e.g. 41.5).
    stop = next((i for i, c in enumerate(s) if c in TERMINATORS and not
                 (c == "." and i > 0 and i + 1 < len(s) and s[i - 1].isalnum() and s[i + 1].isalnum())), len(s))
    ends = []
    for val in vals:
        hits = [m.start() for m in re.finditer(re.escape(val), s[:stop]) if
                (m.start() == 0 or not (s[m.start() - 1].isalnum() or s[m.start() - 1] in "-'’"))
                and _starts_with_word(s[m.start():], val)]
        ends.append(hits[0] + len(val) if hits else None)
    matches = [end is not None for end in ends]
    clean = False
    if all(matches):
        end = max(ends)
        j = next(j for j in range(1, len(ids) + 1) if len(tok.decode(ids[:j]).lstrip()) >= end)
        win = tok.decode(ids[:j + k]).lstrip()[end:]
        rest = s[end:]
        clean = _stops_cleanly(win) or (e < len(gen_ids) and e < j + k and rest.strip(_CLOSERS + " \t") == "")
    return dict(answers=matches, both=bool(all(matches) and clean))


@torch.no_grad()
def qa_probe_eval(model, tok, groups, dev, max_new=0, k=2, bs=16):
    result = dict(n_each={key: len(items) for key, items in groups.items()})
    for key, items in groups.items():
        metric = "two_fact_both" if key == "two_fact" else key
        if not items:
            result[metric] = 0.0
            continue
        mx = max_new or max(len(it["a"]) for it in items) + k + 2
        gens = greedy_generate(model, [it["p"] for it in items], dev, mx, bs)
        hits = [score_two_fact_generation(tok, g, it["vals"], k)["both"] if key == "two_fact" else
                score_generation(tok, g, it["val"], k)["clean"] for it, g in zip(items, gens)]
        result[metric] = sum(hits) / len(items)
    return result


@torch.no_grad()
def record_token_routes(model, panels, dev):
    """Read actual dispatch IDs, preserving all computations and method lookup. Always batch one."""
    layers = moe_layers(model)
    saved, captured, result = [], {}, {}
    try:
        for li, (_, layer) in enumerate(layers):
            name = "_forward_frozen_exact_grouped"
            original = getattr(layer, name, None)
            if not callable(original):
                raise ValueError("--route-drift requires the mlf grouped-dispatch hook")
            saved.append((layer, name, name in layer.__dict__, original))

            def dispatch(expert_input, indices, weights, original=original, li=li):
                if li in captured:
                    raise ValueError("MoE dispatch called more than once per window")
                captured[li] = indices.detach().reshape(-1, indices.shape[-1]).sort(-1).values.cpu()
                return original(expert_input, indices, weights)

            setattr(layer, name, dispatch)
        for panel, wins in panels.items():
            rows = defaultdict(list)
            for win in wins[:16]:
                captured.clear()
                logits_of(model, win.unsqueeze(0).to(dev))
                if set(captured) != set(range(len(layers))):
                    raise ValueError("route capture missed a MoE layer")
                for li, ids in captured.items():
                    if ids.shape[0] != win.numel():
                        raise ValueError("route capture must contain every token")
                    rows[str(li)].append(ids)
            result[panel] = {li: torch.cat(ids) for li, ids in rows.items()}
    finally:
        for layer, name, had_instance, original in reversed(saved):
            if had_instance:
                setattr(layer, name, original)
            else:
                delattr(layer, name)
    return result


def route_drift_summary(base, patched):
    """Compare top-k sets rather than rank/weight, averaging all tokens and then layers equally."""
    result = {}
    if base.keys() != patched.keys():
        raise ValueError("routing panels differ")
    for panel, layers in base.items():
        if layers.keys() != patched[panel].keys():
            raise ValueError("routing layers differ")
        scores = {}
        for li, ids in layers.items():
            other = patched[panel][li].cpu()
            ids = ids.cpu()
            if ids.shape != other.shape:
                raise ValueError("routing token shapes differ")
            common = (ids[:, :, None] == other[:, None, :]).any(-1).sum(-1).double()
            union = ids.shape[-1] + other.shape[-1] - common
            scores[li] = dict(jaccard=float((common / union.clamp_min(1)).mean()),
                              identical=float((common == ids.shape[-1]).double().mean()))
        result[panel] = dict(layers=scores, mean={metric: sum(s[metric] for s in scores.values()) / len(scores)
                                               for metric in ("jaccard", "identical")} if scores else {},
                             n_tokens=next(iter(layers.values())).shape[0] if layers else 0)
    return result


@torch.no_grad()
def gen_eval(model, tok, items, dev, max_new=0, k=2, bs=16, conflicts=None, n_samples=6, arities=None):
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
    grp = defaultdict(list)                                # same attribute + phrasing, different people
    for i, it in enumerate(items):
        if "attr" in it:
            grp[(it["attr"], it["ti"])].append(sc[i]["clean"])
    multi = [v for v in grp.values() if len(v) > 1]
    if multi:
        r.update(n_groups=len(multi), all_people_correct=sum(all(v) for v in multi) / len(multi))
    if arities is not None:
        strata = defaultdict(list)
        for it, s in zip(items, sc):
            strata[arities[it["fi"]]].append(s["clean"])
        r["by_arity"] = {k: sum(v) / len(v) for k, v in strata.items()}
    return r


@torch.no_grad()
def last_token_experts(model, ids, dev):
    """{moe layer index: set(expert)} used by the last token of `ids`: per-expert routed-row counts of the full
    prompt minus those of the prompt without its last token (causal model). Batch shapes differ between the two
    passes, so bf16 routing jitter on earlier tokens can add or swap entries; only +1 differences are kept. This
    is an approximate diagnostic (no router internals are read)."""
    layers = moe_layers(model)
    cnt, hooks = {}, []
    for li, (_, moe) in enumerate(layers):
        for ei, e in enumerate(moe.experts):
            if e is None:
                continue

            def f(mod, inp, out, key=(li, ei)):
                cnt[key] = cnt.get(key, 0) + inp[0].reshape(-1, inp[0].shape[-1]).shape[0]
            hooks.append(e.register_forward_hook(f))
    try:
        logits_of(model, torch.tensor([ids], device=dev))
        full = dict(cnt)
        cnt.clear()
        logits_of(model, torch.tensor([ids[:-1]], device=dev))
        short = dict(cnt)
    finally:
        for h in hooks:
            h.remove()
    out = defaultdict(set)
    for k, v in full.items():
        if v - short.get(k, 0) == 1:
            out[k[0]].add(k[1])
    return out


def routing_overlap(model, tok, facts, dev, bos, max_attrs=40, seed=0):
    """Mean Jaccard of the answer-predicting token's expert sets, per MoE layer: different people asked about
    the same attribute (conflict pairs) vs the same person asked about two different attributes."""
    by_attr = defaultdict(list)
    for fi, f in enumerate(facts):
        by_attr[f.attr].append(fi)
    rng = random.Random(seed + 11)
    attrs = [a_ for a_ in sorted(by_attr) if len(by_attr[a_]) > 1]
    attrs = rng.sample(attrs, min(max_attrs, len(attrs)))
    need = sorted({fi for a_ in attrs for fi in by_attr[a_]})
    if not need:
        return None
    sets = {}
    for fi in need:
        it = build_items(tok, facts, TEST_T[:1], bos, idxs=[fi])[0]
        sets[fi] = last_token_experts(model, it["p"], dev)
    nl = len(moe_layers(model))

    def jac(x, y):
        return len(x & y) / max(len(x | y), 1)
    cross = [[jac(sets[i][l], sets[j][l]) for l in range(nl)]
             for a_ in attrs for i in by_attr[a_] for j in by_attr[a_] if i < j]
    byp = defaultdict(list)
    for fi in need:
        byp[facts[fi].pid].append(fi)
    same = []
    for fis in byp.values():
        for _ in range(len(fis)):
            i, j = rng.sample(fis, 2) if len(fis) > 1 else (None, None)
            if i is not None:
                same.append([jac(sets[i][l], sets[j][l]) for l in range(nl)])
    mean = lambda rows: [sum(r[l] for r in rows) / len(rows) for l in range(nl)] if rows else None  # noqa: E731
    return dict(n_cross_pairs=len(cross), n_same_pairs=len(same), cross_person_same_attr=mean(cross),
                same_person_other_attr=mean(same),
                experts_per_layer=[sum(len(sets[fi][l]) for fi in need) / len(need) for l in range(nl)])


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


def paired_nll(model, wins, base_nll, dev):
    """Paired mean change and standard error across the same general-text windows."""
    base = torch.tensor(base_nll)
    delta = torch.tensor(nll_windows(model, wins, dev)) - base
    return dict(dnll=float(delta.mean()), dnll_se=float(delta.std() / math.sqrt(len(delta))),
                base_nll=float(base.mean()))


def file_sha(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_eval_version():
    """Hash evaluation functions and their local transitive helpers, including prompt construction."""
    pending = [evaluate_base, acquisition_suite, tf_eval, gen_eval, make_world, conflict_map, build_items,
               conv_contexts, extra_eval_panels]
    sources = {}
    while pending:
        fn = inspect.unwrap(pending.pop())
        if fn.__name__ in sources:
            continue
        sources[fn.__name__] = inspect.getsource(fn)
        # Comprehensions/generators carry their own code objects; decorators hide the original code.
        def names(code):
            from types import CodeType
            return set(code.co_names).union(*(names(value) for value in code.co_consts if isinstance(value, CodeType)))
        for name in names(fn.__code__):
            dep = globals().get(name)
            if inspect.isfunction(dep) and dep.__module__ == __name__:
                pending.append(dep)
    constants = {name: globals()[name] for name in ("BOS", "VOCAB", "EOS_IDS", "TERMINATORS", "_CLOSERS",
                 "PEOPLE", "ATTR_POOL", "TRAIN_T", "TRAIN_AUG_T", "VAL_T", "TEST_T", "CONV_T",
                 "QA_UNSEEN_T", "QA_TWO_T", "QA_INDIRECT_T")}
    return hashlib.sha256(json.dumps([sources, constants], sort_keys=True).encode()).hexdigest()


def base_cache_key(a, facts, val_idx, conv_idx, export=None, data=None):
    export = EXPORT if export is None else export
    data = DATA if data is None else data
    fields = ("seed", "people", "facts_per_person", "shared_attrs", "values", "aug", "val_facts",
              "conv_test", "conv_facts", "eval_windows", "val_windows", "gen_max_new", "gen_stop_k",
              "gen_bs", "fast_moe", "routing_diag", "qa_probe", "qa_n", "route_drift")
    identity = dict(export=str(Path(export).resolve()), version=base_eval_version(),
                    options={k: getattr(a, k) for k in fields}, facts=[list(f) for f in facts],
                    val_idx=val_idx, conv_idx=conv_idx, eval_sha=file_sha(a.eval_set),
                    heldout_sha=file_sha(Path(data) / "heldout-tokens.json"),
                    tokenizer_sha=file_sha(Path(data) / "tokenizer.json"),
                    extra=[(Path(path.strip()).name, file_sha(path.strip()))
                           for path in (a.extra_eval_sets or "").split(",") if path.strip()])
    if a.fast_moe != "off":
        identity["fast_code"] = file_sha(Path(__file__).with_name("fast_moe.py"))
        identity["fast_layers"] = sorted(int(li) for li in a.layers.split(","))
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cached_base(directory, key, evaluate, dev):
    """Immutable tensor sidecar + atomically replaced JSON manifest, safe for concurrent writers.

    NLL lists use float64 sidecar tensors and round-trip to Python lists without rounding.
    Tensor payloads retain their dtype/shape and are restored to the evaluation device.
    """
    if directory is None:
        return evaluate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / f"{key}.json"
    if manifest.exists():
        state = json.loads(manifest.read_text())
        tensors = torch.load(directory / state["tensors"], map_location=dev, weights_only=True)

        def unpack(value):
            if isinstance(value, dict):
                if "__tensor__" in value:
                    tensor = tensors[value["__tensor__"]]
                    return tensor.tolist() if value.get("list") else tensor
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value

        result = unpack(state["results"])
        print(f"BASE-CACHE hit {key}", flush=True)
        return result
    result = evaluate()
    tensors = {}

    def pack(value, nll=False):
        if torch.is_tensor(value) or (nll and isinstance(value, list)):
            name = str(len(tensors))
            tensors[name] = value.detach().cpu() if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float64)
            return {"__tensor__": name, "list": not torch.is_tensor(value)}
        if isinstance(value, dict):
            return {k: pack(v, nll or k in ("base_big", "base_valw", "base_extra")) for k, v in value.items()}
        if isinstance(value, list):
            return [pack(v) for v in value]
        return value

    payload = pack(result)
    sidecar = directory / f"{key}.{uuid.uuid4().hex}.pt"
    # A unique immutable sidecar avoids a reader pairing one writer's JSON with another's tensors.
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(tensors, temporary)
        os.replace(temporary, sidecar)
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w") as stream:
            json.dump(dict(tensors=sidecar.name, results=payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


@torch.no_grad()
def acquisition_suite(model, tok, items_train, items_test, items_conv, dev, a, conflicts, with_train=False):
    result = dict(test=tf_eval(model, items_test, dev, a.gen_bs),
                  test_gen=gen_eval(model, tok, items_test, dev, a.gen_max_new, a.gen_stop_k, a.gen_bs, conflicts))
    if items_conv is not None:
        result.update(conv=tf_eval(model, items_conv, dev, a.gen_bs),
                      conv_gen=gen_eval(model, tok, items_conv, dev, a.gen_max_new, a.gen_stop_k, a.gen_bs, conflicts))
    if with_train:
        result["train"] = tf_eval(model, items_train, dev, a.gen_bs)
    return result


@torch.no_grad()
def evaluate_base(model, tok, facts, dev, bos, segs, acq_suite, big, valw, extra_panels, a, items_val=None,
                  route_panels=None):
    held, lp = heldout_eval(model, segs, dev)
    result = dict(base_held=held, base_lp_t=lp, base_acq=acq_suite(with_train=True),
                  base_logits0=logits_of(model, torch.tensor([segs[0]], device=dev)).detach().clone(),
                  base_big=nll_windows(model, big, dev) if big is not None else None,
                  base_valw=nll_windows(model, valw, dev) if valw is not None else None,
                  base_extra={name: nll_windows(model, wins, dev) for name, (_, wins) in extra_panels.items()},
                  base_routing=routing_overlap(model, tok, facts, dev, bos, seed=a.seed) if a.routing_diag else None)
    if a.qa_probe:
        result["base_acq"]["qa_probe"] = qa_probe_eval(model, tok, build_qa_items(tok, facts, bos, a.qa_n), dev,
                                                     a.gen_max_new, a.gen_stop_k, a.gen_bs)
    if a.route_drift:
        result["base_token_routes"] = record_token_routes(model, route_panels, dev)
    if items_val is not None:
        result["base_val"] = dict(val=tf_eval(model, items_val, dev, a.gen_bs),
                                 val_gen=gen_eval(model, tok, items_val, dev, a.gen_max_new, a.gen_stop_k,
                                                   a.gen_bs, conflict_map(facts)))
    return result


def extra_eval_panels(paths, bos, dev):
    """Load final-only panels: exactly 40 windows with 511 scored tokens plus BOS each."""
    panels = {}
    for path in paths.split(","):
        path = Path(path.strip())
        name = path.name
        if not name or name in panels:
            raise ValueError("extra evaluation panels need distinct basenames")
        tokens = json.loads(path.read_text())
        if len(tokens) < 40 * 511:
            raise ValueError(f"extra evaluation panel {path} needs at least {40 * 511} tokens")
        tokens = tokens[:40 * 511]
        wins = torch.tensor([[bos] + tokens[i * 511:(i + 1) * 511] for i in range(40)], device=dev)
        panels[name] = (tokens, wins)
    return panels


def wake_hinge(z, positive, negative, margin, general=None, weight=1., normalizers=None):
    """Squared positive/negative margin hinges and the general-text z <= 0 hinge."""
    pos = torch.relu(margin - z).square()[positive]
    neg = torch.relu(z + margin).square()[negative]
    pos_count, neg_count = normalizers or (max(pos.numel(), 1), max(neg.numel(), 1))
    loss = pos.sum() / pos_count + neg.sum() / neg_count
    if general is not None and general.numel():
        loss = loss + weight * torch.relu(general).square().mean()
    return loss


def wake_labels(facts, fact_ids, owners, device):
    positive = torch.tensor([[fi == owner for owner in owners] for fi in fact_ids], device=device)
    negative = torch.tensor([[(facts[fi].attr == facts[owner].attr and facts[fi].pid != facts[owner].pid) or
                             (facts[fi].pid == facts[owner].pid and facts[fi].attr != facts[owner].attr)
                             for owner in owners] for fi in fact_ids], device=device)
    return positive, negative


def wake_select(z, facts, fact_ids, routed, eligible, units):
    """Mean routed-fact selectivity; bind each selected row to its best fact.

    Binding gives a detector a consistent target: another person's true prompt is a
    negative for this unit, and can be a positive for a different unit.
    Negative potentials remain eligible, so an initially dormant bank can wake.
    """
    conflicts = conflict_map(facts)
    targets, potentials = [], []
    for fi in sorted(conflicts):
        pos = [i for i, f in enumerate(fact_ids) if f == fi and routed[i]]
        neg = [i for i, f in enumerate(fact_ids) if facts[f].attr == facts[fi].attr and facts[f].pid != facts[fi].pid]
        if pos and neg:
            targets.append(fi)
            potentials.append(z[pos].mean(0) - z[neg].amax(0))
    if not potentials or not bool(eligible.any()):
        return [], []
    potentials = torch.stack(potentials)
    score = potentials.mean(0).masked_fill(~eligible, -float("inf"))
    rows = score.topk(min(units, int(eligible.sum()))).indices
    owners = potentials[:, rows].argmax(0).tolist()
    return rows.tolist(), [targets[i] for i in owners]


@contextmanager
def wake_prompt_inputs(model, layer_ids, detach=True):
    """Record exact last-prompt-token inputs and routes, with no prefix subtraction.

    The resident grouped dispatch supplies explicit token indices (also in fast-MoE
    modes). The fallback for simple CPU MoEs matches their unchanged routed inputs.
    Each forward is a single unpadded prompt, so the final token index is causal.
    """
    from types import MethodType
    records, hooks, methods = {}, [], []
    try:
        for li in layer_ids:
            _, moe = moe_layers(model)[li]
            dispatch = getattr(moe, "_forward_frozen_exact_grouped", None)
            if dispatch is not None:
                own = "_forward_frozen_exact_grouped" in moe.__dict__
                methods.append((moe, dispatch, own))

                def capture(layer, x, indices, weights, li=li, dispatch=dispatch):
                    records[li] = (x[-1].detach().cpu() if detach else x[-1], set(indices[-1].tolist()))
                    return dispatch(x, indices, weights)

                moe._forward_frozen_exact_grouped = MethodType(capture, moe)
            else:
                def last(mod, inp, li=li):
                    x = inp[0].reshape(-1, inp[0].shape[-1])
                    records[li] = (x[-1].detach().cpu() if detach else x[-1], set())
                hooks.append(moe.register_forward_pre_hook(last))
                for ei, expert in enumerate(moe.experts):
                    if expert is None:
                        continue

                    def route(mod, inp, li=li, ei=ei):
                        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
                        last, chosen = records[li]
                        if bool((x == last.detach().cpu()).all(1).any()):
                            chosen.add(ei)
                    hooks.append(expert.up_proj.register_forward_pre_hook(route))
        yield records
    finally:
        for hook in hooks:
            hook.remove()
        for moe, dispatch, own in methods:
            if own:
                moe._forward_frozen_exact_grouped = dispatch
            else:
                del moe._forward_frozen_exact_grouped


def wake_margin_stats(z, pos, neg, margin):
    return dict(positive=float((z >= margin)[pos].float().mean()),
                negative=float((z <= -margin)[neg].float().mean()))


def run_wake_stage(model, tok, facts, templates, bos, layer_ids, anchor, a):
    """Learn forced-slot detectors on causal prompt inputs, then freeze quantized rows.

    Selection and margin scales use the base model. Each wake update re-evaluates
    the patched hidden inputs, with gradients only to selected up-slot masters.
    Small master vectors avoid Adam moments for entire expert weight matrices.
    """
    items = build_items(tok, facts, templates, bos)
    fact_ids = [it["fi"] for it in items]
    dev = next(model.parameters()).device
    banks = {name: (lin, lin.parametrizations.weight[0]) for name, lin in patched_linears(model)
             if name.endswith("up_proj")}
    prompt_x, routes = defaultdict(list), defaultdict(list)
    enabled, checkpoint_on = FSAParam.enabled, CKPT["on"]
    try:
        set_enabled(False)
        CKPT["on"] = False
        with torch.no_grad(), wake_prompt_inputs(model, layer_ids) as records:
            for it in items:
                records.clear()
                logits_of(model, torch.tensor([it["p"]], device=dev))
                for li in layer_ids:
                    if li not in records:
                        raise RuntimeError(f"wake stage cannot capture last-token dispatch at layer {li}")
                    x, chosen = records[li]
                    prompt_x[li].append(x)
                    routes[li].append(chosen)
    finally:
        set_enabled(enabled)
        CKPT["on"] = checkpoint_on
    detectors, report = [], {}
    for li in layer_ids:
        lname, moe = moe_layers(model)[li]
        xs = torch.stack(prompt_x[li]).to(dev)
        for ei, expert in enumerate(moe.experts):
            name = f"{lname}.experts.{ei}.up_proj"
            if name not in banks:
                continue
            lin, m = banks[name]
            original = lin.parametrizations.weight.original.detach()
            with torch.no_grad():
                z = F.linear(xs, original, lin.bias).float()
                eligible = ((m._dense(m.mask).bool() if m.sparse else m.mask).any(1) & (m.alpha[:, 0] > 0))
                rows, owners = wake_select(z, facts, fact_ids, [ei in r for r in routes[li]], eligible, a.wake_units)
            if not rows:
                continue
            pos, neg = wake_labels(facts, fact_ids, owners, dev)
            margin = a.wake_margin * z[:, rows].std(0, unbiased=False)
            before = wake_margin_stats(z[:, rows], pos, neg, margin)
            row_ids = torch.tensor(rows, device=dev)
            stored_rows = m.rows() if m.sparse else torch.arange(m.wshape[0], device=dev)[:, None].expand_as(m.M)
            indices = (torch.isin(stored_rows, row_ids) & m.mask).flatten().nonzero().flatten()
            detectors.append(dict(name=name, lin=lin, param=m, rows=rows, row_ids=row_ids, indices=indices,
                                  master=nn.Parameter(m.M.detach().flatten()[indices].clone()), li=li,
                                  pos=pos, neg=neg, margin=margin, normalizers=(int(pos.sum()), int(neg.sum())),
                                  support=m.support.flatten()[indices].clone()))
            report[name] = dict(rows=rows, facts=owners, before=before)
    del prompt_x
    xs = z = None
    if not detectors:
        raise RuntimeError("wake stage found no routed conflict prompts with writable up rows")

    trainable = [(param, param.requires_grad) for param in model.parameters()]
    hooks, general_terms = [], {}
    try:
        for param, _ in trainable:
            param.requires_grad_(False)
        CKPT["on"] = False
        set_enabled(True)
        for detector in detectors:
            m = detector["param"]
            m.wake_slot_indices = detector["indices"]
            m.wake_master = detector["master"]
            if m.support is not m.mask:
                m.support.view(-1)[detector["indices"]] = True

            def general_hook(mod, inp, out, detector=detector):
                if detector["name"] in general_terms:
                    z = out.reshape(-1, out.shape[-1])[:, detector["row_ids"]].float()
                    general_terms[detector["name"]].append((torch.relu(z).square().sum(), z.numel()))
            hooks.append(detector["lin"].register_forward_hook(general_hook))
        optimizer = torch.optim.Adam([d["master"] for d in detectors], lr=a.wake_lr)
        with wake_prompt_inputs(model, layer_ids, detach=False) as records:
            for _ in range(a.wake_steps):
                optimizer.zero_grad(set_to_none=True)
                for i, it in enumerate(items):
                    records.clear()
                    logits_of(model, torch.tensor([it["p"]], device=dev))
                    terms = []
                    for d in detectors:
                        pos, neg = d["pos"][i], d["neg"][i]
                        if not bool((pos | neg).any()):
                            continue
                        x = records[d["li"]][0]
                        lin = d["lin"]
                        bias = lin.bias[d["row_ids"]] if lin.bias is not None else None
                        z = F.linear(x, lin.weight[d["row_ids"]], bias).float()
                        term = wake_hinge(z, pos, neg, d["margin"], normalizers=d["normalizers"])
                        terms.append(term / len(detectors))
                    if terms:
                        torch.stack(terms).sum().backward()
                if a.wake_lambda:
                    for window in anchor[:2]:
                        general_terms.update({d["name"]: [] for d in detectors})
                        logits_of(model, window[None])
                        terms = []
                        for values in general_terms.values():
                            if values:
                                total = torch.stack([value for value, _ in values]).sum()
                                terms.append(total / max(sum(n for _, n in values), 1))
                        general_terms.clear()
                        if terms:
                            (a.wake_lambda * torch.stack(terms).sum() / (len(detectors) * min(2, len(anchor)))).backward()
                optimizer.step()
                with torch.no_grad():
                    for d in detectors:
                        d["master"].clamp_(-1, 1)
        with torch.no_grad():
            for d in detectors:
                d["param"].M.view(-1)[d["indices"]] = d["param"]._quant(d["master"])
    finally:
        general_terms.clear()
        for hook in hooks:
            hook.remove()
        for d in detectors:
            m = d["param"]
            if hasattr(m, "wake_master"):
                del m.wake_master, m.wake_slot_indices
            if m.support is not m.mask:
                m.support.view(-1)[d["indices"]] = d["support"]
        for param, requires_grad in trainable:
            param.requires_grad_(requires_grad)
        set_enabled(enabled)
        CKPT["on"] = checkpoint_on

    # The final detector codes consume physical budget before ordinary training starts.
    if a.budget:
        magnitudes = torch.cat([d["master"].detach().abs() * (d["param"].M.flatten()[d["indices"]] != 0)
                                for d in detectors])
        keep = torch.zeros_like(magnitudes, dtype=torch.bool)
        live = magnitudes.nonzero().flatten()
        keep[live[magnitudes[live].topk(min(a.budget, live.numel())).indices]] = True
        offset = 0
        with torch.no_grad():
            for d in detectors:
                n = d["master"].numel()
                d["param"].M.view(-1)[d["indices"]] *= keep[offset:offset + n]
                offset += n
    for d in detectors:
        m = d["param"]
        with torch.no_grad():
            if a.budget and m.support is not m.mask:
                m.support.view(-1)[d["indices"]] = m.M.flatten()[d["indices"]] != 0
            freeze_wake_rows(m, d["rows"])

    # Measure after ternarization/pruning on the actual patched hidden inputs.
    after_z = defaultdict(list)
    with torch.no_grad(), wake_prompt_inputs(model, layer_ids) as records:
        for it in items:
            records.clear()
            logits_of(model, torch.tensor([it["p"]], device=dev))
            for d in detectors:
                lin = d["lin"]
                bias = lin.bias[d["row_ids"]] if lin.bias is not None else None
                x = records[d["li"]][0].to(dev)
                after_z[d["name"]].append(F.linear(x, lin.weight[d["row_ids"]], bias).float())
    for d in detectors:
        report[d["name"]]["after"] = wake_margin_stats(torch.stack(after_z[d["name"]]),
                                                        d["pos"], d["neg"], d["margin"])
    by_layer = {}
    for li in layer_ids:
        lname = moe_layers(model)[li][0]
        entries = [v for name, v in report.items() if name.startswith(lname + ".experts.")]
        if entries:
            by_layer[lname] = {phase: {kind: sum(v[phase][kind] for v in entries) / len(entries)
                                      for kind in ("positive", "negative")} for phase in ("before", "after")}
    print("WAKE", json.dumps(dict(layers=by_layer, units=sum(len(v["rows"]) for v in report.values()),
                                 nnz=sum(int(pinned_slots(m).sum()) for _, m in banks.values()))), flush=True)
    return dict(steps=a.wake_steps, units=a.wake_units, margin=a.wake_margin, weight=a.wake_lambda,
                lr=a.wake_lr, detectors=report, layers=by_layer)


@torch.no_grad()
def unit_activation_stats(model, layer_ids, windows):
    """Per routed expert of the given MoE layers: activation rate P(pre > 0) and mean relu^2 of every hidden unit
    over the routed tokens of general-text windows (patches disabled), plus the routed token count."""
    layers = moe_layers(model)
    stats, hooks = {}, []
    for li in layer_ids:
        lname, moe = layers[li]
        for ei, e in enumerate(moe.experts):
            if e is None:
                continue
            key = f"{lname}.experts.{ei}"
            stats[key] = dict(pos=torch.zeros(e.up_proj.out_features), sq=torch.zeros(e.up_proj.out_features), n=0)

            def f(mod, inp, out, key=key):
                o = out.reshape(-1, out.shape[-1]).float()
                st_ = stats[key]
                st_["pos"] += (o > 0).sum(0).cpu()
                st_["sq"] += torch.relu(o).pow(2).sum(0).cpu()
                st_["n"] += o.shape[0]
            hooks.append(e.up_proj.register_forward_hook(f))
    set_enabled(False)
    try:
        seqs_ = [windows[i:i + 1] for i in range(windows.shape[0])] if torch.is_tensor(windows) else \
            [torch.tensor([sq], device=next(model.parameters()).device) for sq in windows]
        for x_ in seqs_:
            logits_of(model, x_)
    finally:
        set_enabled(True)
        for h in hooks:
            h.remove()
    for st_ in stats.values():
        n_ = max(st_["n"], 1)
        st_["rate"] = st_.pop("pos") / n_
        st_["mean_sq"] = st_.pop("sq") / n_
    return stats


def select_units(stats, frac, mode, seed, fact_stats=None):
    """Per expert: bottom `frac` of hidden units by general-text activation rate (sleep), the top `frac` (active
    control), a seeded random `frac` (random control), or (specific) the top `frac` by fact-context rate minus
    general-text rate - units already on for the facts but rare on general text, so relu^2 passes a gradient.
    Ties broken by mean relu^2."""
    g = torch.Generator().manual_seed(seed + 51)
    out = {}
    for key, st_ in stats.items():
        d = st_["rate"].numel()
        k = max(1, int(round(frac * d)))
        if mode == "random":
            idx = torch.randperm(d, generator=g)[:k]
        elif mode == "specific":
            fs_ = fact_stats.get(key) if fact_stats else None
            fr = fs_["rate"] if fs_ is not None else torch.zeros(d)
            score = fr - st_["rate"] + 1e-6 * (fs_["mean_sq"] / (fs_["mean_sq"].max() + 1e-30) if fs_ is not None
                                               else 0)
            idx = torch.argsort(score, descending=True)[:k]
        else:
            score = st_["rate"] + 1e-6 * st_["mean_sq"] / (st_["mean_sq"].max() + 1e-30)
            idx = torch.argsort(score, descending=(mode == "active"))[:k]
        sel = torch.zeros(d, dtype=torch.bool)
        sel[idx] = True
        out[key] = sel
    return out


class GateHinge:
    """Keep selected hidden units off on general text: sum over routed general tokens and selected units of
    relu(pre + margin)^2 / count, recorded from the (patched) up_proj outputs during the penalty forward. Same
    record/replay protocol as LocalOutputPenalty for checkpointed experts."""

    def __init__(self, model, layer_ids, units, margin):
        self.on, self.replay, self.terms, self.cnt, self.h = False, False, [], 0, []
        layers = moe_layers(model)
        self.margin = margin
        for li in layer_ids:
            lname, moe = layers[li]
            for ei, e in enumerate(moe.experts):
                key = f"{lname}.experts.{ei}"
                if e is None or key not in units or not bool(units[key].any()):
                    continue
                idx = units[key].nonzero().squeeze(1).to(e.up_proj.weight.device if hasattr(e.up_proj, "weight")
                                                           else "cpu")
                self.h.append(e.up_proj.register_forward_hook(self._hook(idx)))

    def _hook(self, idx):
        def f(mod, inp, out):
            if not (self.on or self.replay) or out.numel() == 0:
                return
            o = out.reshape(-1, out.shape[-1])
            t = torch.relu(o[:, idx.to(o.device)].float() + self.margin).pow(2).sum()
            if self.on:
                self.terms.append(t)
                self.cnt += o.shape[0] * idx.numel()
        return f

    def start(self):
        self.on, self.replay, self.terms, self.cnt = True, False, [], 0

    def stop(self):
        self.on, self.replay = False, True
        if not self.terms:
            return None
        return torch.stack(self.terms).sum() / max(self.cnt, 1)

    def end(self):
        self.replay = False


class LocalOutputPenalty:
    """Smooth preservation: relative energy of the patch's direct output change on general text,
    sum ||x @ dW^T||^2 / sum ||x @ W^T||^2 over every patched linear (W = the frozen base). Routing chaos cannot
    reach it."""

    def __init__(self, model, from_output=False):
        """from_output: num = ||out - x W^T||^2 using the linear's own (patched) output; retains only output-sized
        tensors instead of a dense fp32 delta per linear (needed for many patched layers). Same quantity up to the
        bf16 rounding of W + dW. An active teacher requires the direct student-delta form."""
        self.on, self.num, self.den, self.h = False, [], [], []
        self.replay = False      # True during the penalty's backward: checkpoint recompute must redo the same ops
        self.from_output = from_output
        for _, lin in patched_linears(model):
            pm = lin.parametrizations.weight[0]
            if isinstance(pm, TeacherLoRAParam):
                continue                            # teacher-only linears have no student delta to penalize
            self.h.append(lin.register_forward_hook(self._hook(pm, lin)))

    def _hook(self, pm, lin):
        def f(mod, inp, out):
            if not (self.on or self.replay) or inp[0].shape[0] == 0:
                return
            # out - x W_base includes an active teacher. In that case measure
            # the student's delta directly, keeping the teacher in the forward.
            teacher_on = any(isinstance(p, TeacherLoRAParam) and p.enabled and p.scale
                             for p in lin.parametrizations.weight)
            if self.from_output and not teacher_on:
                x = inp[0].reshape(-1, inp[0].shape[-1])
                base = (x @ lin.parametrizations.weight.original.T).float()   # keeps x's graph: only x dW^T
                num = (out.reshape(-1, out.shape[-1]).float() - base).pow(2).sum()   # survives in grad
                den = base.detach().pow(2).sum()
            else:
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                num = (x @ pm.delta_w().float().T).pow(2).sum()
                with torch.no_grad():
                    den = (x @ lin.parametrizations.weight.original.float().T).pow(2).sum()
            if self.on:                                    # replay (checkpoint recompute): same ops, not recorded
                self.num.append(num)
                self.den.append(den)
        return f

    def start(self):
        self.on, self.replay, self.num, self.den = True, False, [], []

    def stop(self):
        """Ends recording and arms replay for the coming backward of the returned penalty; call end() after it."""
        self.on, self.replay = False, True
        if not self.num:
            return None
        return torch.stack(self.num).sum() / (torch.stack(self.den).sum() + 1e-12)

    def end(self):
        self.replay = False


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


def checked_overlap(train_toks, streams, source):
    ov = ngram_overlap(train_toks, streams)
    print(f"OVERLAP 32-gram train-{source} vs eval text", json.dumps(ov), flush=True)
    # The legacy heldout KL/top-1 panel is reported only; headline panels must be disjoint.
    if max((v for k, v in ov.items() if k != "heldout"), default=0.0) > 0.01:
        raise SystemExit(f"{source} text overlaps evaluation text: {ov}")
    return ov


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", type=Path, default=None, help="shared deterministic base-evaluation cache")
    ap.add_argument("--qa-probe", action="store_true", help="final-only unseen, two-fact and indirect generation QA")
    ap.add_argument("--qa-n", type=int, default=120, help="maximum QA probe items per type (fixed sampling RNG)")
    ap.add_argument("--route-drift", action="store_true", help="final base/patched token routing on 16x512 per panel")
    ap.add_argument("--wake-steps", type=int, default=0, help="pretraining up-row detector steps (0 disables)")
    ap.add_argument("--wake-units", type=int, default=16, help="detector units per routed expert")
    ap.add_argument("--wake-margin", type=float, default=1., help="detector margin in base fact preactivation std")
    ap.add_argument("--wake-lambda", type=float, default=1., help="general-text detector hinge weight")
    ap.add_argument("--wake-lr", type=float, default=1e-3, help="detector Adam lr; masters clipped to 1")
    ap.add_argument("--method", choices=["fsa_free", "fsa_full", "fsa_forced", "lora", "lora_fold"], required=True,
                    help="fsa_free: zeros inside selected pairs (pair8-legal); fsa_forced: the 2 unselected pairs "
                         "per block (shipped mask, 50%% of weights); fsa_full: any base zero")
    ap.add_argument("--fold-mode", choices=["ste", "ef"], default="ste",
                    help="lora_fold: ste = fold-aware forward Q(mask*BA/alpha); ef = periodic fold into shadow "
                         "slot masters with error feedback + adapter reset")
    ap.add_argument("--local-target", type=float, default=0.0,
                    help="constrained mode: dual ascent on the local-penalty weight to hold the measured relative "
                         "output change at this target (lambda starts at --local-lambda)")
    ap.add_argument("--local-dual-lr", type=float, default=0.1,
                    help="multiplicative dual step: lambda *= exp(lr * (local_rel / target - 1)), clamped")
    ap.add_argument("--expert-checkpoint", action="store_true",
                    help="activation-checkpoint every patched expert (memory flat in the number of patched layers)")
    ap.add_argument("--slot-frac", type=float, default=1.0,
                    help="(--sparse-state) keep a seeded random fraction of each linear's slots (memory for many "
                         "layers; also a random-placement control)")
    ap.add_argument("--occupied-flips", action="store_true",
                    help="(--sparse-state) also allow sign flips of existing nonzero ternary weights (pair8-legal); "
                         "births there need %sx tau" % "2")
    ap.add_argument("--pool-select", choices=["none", "conflict", "random"], default="none",
                    help="static slot pool chosen before training (needs --budget; refresh disabled): conflict = "
                         "per-person signed-gradient specificity (top 80%%) + 20%% random; random = matched control")
    ap.add_argument("--pool-beta", type=float, default=1.0, help="weight of the other people's predicted worsening")
    ap.add_argument("--pool-random-frac", type=float, default=0.2)
    ap.add_argument("--routing-diag", action="store_true",
                    help="after training: experts chosen by the answer-predicting token at every MoE layer, "
                         "cross-person (same attribute) vs same-person (other attribute) Jaccard")
    ap.add_argument("--fold-every", type=int, default=30, help="lora_fold ef: fold period in steps")
    ap.add_argument("--gate-units-frac", type=float, default=0.0,
                    help="sleeping-unit gating: per expert, only this fraction of hidden units (chosen by "
                         "--gate-select on general-text activation rate) may be patched (W_up rows, W_down cols)")
    ap.add_argument("--gate-select", choices=["sleep", "specific", "random", "active"], default="specific")
    ap.add_argument("--gate-lambda", type=float, default=0.0,
                    help="weight of the hinge keeping the selected units off on general text")
    ap.add_argument("--gate-margin", type=float, default=0.0, help="hinge margin on the pre-activation")
    ap.add_argument("--patch-scale-mult", type=float, default=1.0,
                    help="patch-only step: slot values are +-s*alpha_row (FSA and lora_fold; base unchanged)")
    ap.add_argument("--fold-mask", choices=["free", "any_zero", "pair8_forced"], default="free",
                    help="lora_fold writable set: in-pair free slots (pair8-legal) or any base zero (plain ternary "
                         "overlay; revoke still restores the exact pair8 base)")
    ap.add_argument("--fold-thr", type=float, default=0.5,
                    help="lora_fold: |x| (units of alpha_row) above which a slot becomes +-1 (higher = sparser)")
    ap.add_argument("--fold-mask-frac", type=float, default=1.0,
                    help="lora_fold: random-placement control - only this random fraction of the free slots may be "
                         "written (seeded)")
    ap.add_argument("--no-pareto", action="store_true",
                    help="screens: keep only the final evaluation (skip the per-recall-level checkpoint evals)")
    ap.add_argument("--fmt", choices=["ternary", "tern2", "fp4", "fp8", "bf16"], default="ternary",
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
    ap.add_argument("--optim", choices=["adam", "sgd", "bop", "mls"], default="adam")
    ap.add_argument("--mls-eta", type=float, default=0.,
                    help="MLS output-gradient step override; 0 calibrates to --mls-target-changes each iteration")
    ap.add_argument("--mls-target-changes", type=int, default=20000,
                    help="initial target for discrete rounding changes; rejects halve it (floor 500), accepts grow by 1.25")
    ap.add_argument("--mls-trust", type=float, default=.01, help="deprecated compatibility option (ignored)")
    ap.add_argument("--mls-gen-every", type=int, default=50)
    ap.add_argument("--mls-rank", type=int, default=16)
    ap.add_argument("--mls-gen-lambda", type=float, default=1., help="general covariance weight in the masked solve")
    ap.add_argument("--mls-gamma", type=float, default=1e-4, help="positive ridge in the masked solve")
    ap.add_argument("--mls-cg-iters", type=int, default=8)
    ap.add_argument("--mls-thr", type=float, default=.5)
    ap.add_argument("--mls-max-change", type=int, default=50000)
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
    ap.add_argument("--bop-tau-abs", type=float, default=0.0,
                    help="absolute Bop threshold (skips calibration; e.g. to fine-tune a loaded patch with its "
                         "original threshold)")
    ap.add_argument("--sparse-state", action="store_true",
                    help="keep FSA training state (masters, grads, EMA, score) on the free slots only (~6x less "
                         "memory; needed for many layers)")
    ap.add_argument("--shared-expert", action="store_true", help="also patch each layer's shared expert(s)")
    ap.add_argument("--shared-only", action="store_true", help="patch only the shared expert(s)")
    ap.add_argument("--bop-accum", type=int, default=1,
                    help="Bop: sum gradients over N steps (N fact batches) per Bop update; N ~ facts / fact-batch "
                         "lets every fact contribute to every flip decision")
    ap.add_argument("--tau-anneal", type=float, default=1.0,
                    help="Bop: tau multiplier goes linearly from 1 (step 0) to this value (last main step)")
    ap.add_argument("--adam-phase", type=int, default=0,
                    help="Bop: first K steps train fp32 masters with AdamW + STE (learn-then-prune; --lr, --wd), "
                         "then snap to ternary codes and continue with Bop")
    ap.add_argument("--balance-people", action="store_true", help="each fact batch has equal facts per person")
    ap.add_argument("--paired", action="store_true",
                    help="conflict-group batches: attributes drawn from a shuffled queue without replacement; every "
                         "person holding a drawn attribute is in the same step, with the same sampled formats")
    ap.add_argument("--keep-ema", action="store_true", help="Bop: do not reset the gradient EMA on a flip")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="weight of softplus(1 + z_other - z_correct) at the first answer token where another "
                         "person's value for the same attribute diverges from the correct one")
    ap.add_argument("--unlikelihood", type=float, default=0.0,
                    help="weight of -log(1 - p(other person's answer first token)) at the answer position, for "
                         "facts whose attribute other people hold with a different value")
    ap.add_argument("--death-grace", type=int, default=0,
                    help="Bop maturity: no death within N Bop updates of birth (= steps unless --bop-accum)")
    ap.add_argument("--death-tau-mult", type=float, default=1.0, help="Bop death threshold = mult x birth tau")
    ap.add_argument("--no-churn", action="store_true", help="skip the per-slot int8 churn counters (memory)")
    ap.add_argument("--load-patch", default=None, help="load a saved patch after attach (use with --steps 0)")
    ap.add_argument("--fast-moe", choices=("off", "exact", "batched"), default="off",
                    help="MoE dispatch overhead removal (exact: bit-identical logits); see fast_moe.py")
    ap.add_argument("--bind-loss", type=float, default=0.0, help="weight of complete subject-answer assignment loss")
    ap.add_argument("--bind-groups", type=int, default=4, help="complete conflict groups per binding update")
    ap.add_argument("--bind-every", type=int, default=4, help="binding update period in training steps")
    ap.add_argument("--train-audit", type=int, default=0, help="fixed training (fact, phrasing) items scored at validation")
    ap.add_argument("--rehearsal-set", default=None, help="json rehearsal tokens (default: --anchor)")
    ap.add_argument("--extra-eval-sets", default=None, help="comma-separated json token panels for final 40x512 paired NLL")
    ap.add_argument("--keep-live", action="store_true", help="budget refresh never evicts live quantized slots")
    ap.add_argument("--strata", action="store_true", help="validation/audit recall by full-world attribute arity")
    ap.add_argument("--curriculum", choices=["none", "attribute"], default="none")
    ap.add_argument("--cur-check", type=int, default=10, help="attribute mastery/replay check period")
    ap.add_argument("--cur-replay", type=float, default=0.25, help="mastered-group replay share of batch facts")
    ap.add_argument("--cur-active", type=int, default=8, help="maximum active attribute groups")
    ap.add_argument("--score-epoch", action="store_true", help="sum signed answer gradients; refresh netcost once per epoch")
    ap.add_argument("--pair-birth", action="store_true", help="Bop: minimize EMA dot d over dormant forced-pair patterns")
    ap.add_argument("--patch-scale-mult-up", type=float, default=None, help="up projection override of --patch-scale-mult")
    ap.add_argument("--patch-scale-mult-down", type=float, default=None, help="down projection override of --patch-scale-mult")
    ap.add_argument("--fisher-support", action="store_true", help="after Adam: support gain g^2/(exp_avg_sq + 1e-8) minus cost")
    ap.add_argument("--teacher-patch", type=Path, default=None, help="frozen saved LoRA final.pt, training only")
    ap.add_argument("--teacher-fade", type=float, default=0.0,
                    help="fade teacher over F * steps in every training forward; 0 disables (needs --teacher-patch)")
    ap.add_argument("--teacher-fade-shape", choices=["linear", "cos"], default="linear")
    ap.add_argument("--distill-mu", type=float, default=0.0)
    ap.add_argument("--distill-T", type=float, default=2.0)
    ap.add_argument("--distill-anneal", action="store_true")
    ap.add_argument("--ctx-teacher", choices=["person", "all"], default=None,
                    help="frozen base with a fact sheet in context; student receives no sheet")
    ap.add_argument("--ctx-mu", type=float, default=0.5)
    ap.add_argument("--ctx-T", type=float, default=1.0)
    ap.add_argument("--ctx-only-correct", action="store_true", help="KL only where context teacher argmax is gold")
    ap.add_argument("--self-study", type=int, default=0, help="additional labelled self-study questions per step")
    ap.add_argument("--ctx-teacher-check", action="store_true",
                    help="report person/all context teacher val/test TF and GEN, then exit before training")
    ap.add_argument("--saturation-sampling", action="store_true")
    ap.add_argument("--sat-margin", type=float, default=2.0)
    ap.add_argument("--sat-weight", type=float, default=0.25)
    ap.add_argument("--retain-evicted", action="store_true", help="alias of --keep-live: pin live masters and EMA")
    ap.add_argument("--swap-margin", type=float, default=0.0, help="challenger advantage in writable-score std units")
    ap.add_argument("--swap-cap", type=float, default=0.0,
                    help="cap support bit changes per refresh at floor(F*budget*(1+cos(pi*t/(.6*steps)))/2); "
                         "0 disables; freeze support at t >= .6*steps")
    ap.add_argument("--death-age-mult", action="store_true", help="death tau *= 1+2*tanh(wall-step age/50)")
    ap.add_argument("--patch-dropout", type=float, default=0.0)
    ap.add_argument("--tau-noise", type=float, default=0.0)
    ap.add_argument("--reheat-every", type=int, default=0)
    ap.add_argument("--reheat-frac", type=float, default=0.0, help="zero sampled live EMA and halve their age")
    return ap


def validate_ctx_args(a):
    if not math.isfinite(a.ctx_mu) or a.ctx_mu < 0:
        raise SystemExit("--ctx-mu must be finite and >= 0")
    if not math.isfinite(a.ctx_T) or a.ctx_T <= 0:
        raise SystemExit("--ctx-T must be finite and > 0")
    if a.self_study < 0:
        raise SystemExit("--self-study must be >= 0")
    if (a.self_study or a.ctx_only_correct) and not (a.ctx_teacher or a.ctx_teacher_check):
        raise SystemExit("--self-study / --ctx-only-correct need --ctx-teacher")
    if a.ctx_teacher or a.ctx_teacher_check:
        if a.teacher_patch or a.teacher_fade or a.distill_mu or a.distill_anneal:
            raise SystemExit("context teacher cannot be combined with a LoRA teacher/distillation")
    if a.ctx_teacher and (a.method != "fsa_forced" or a.fmt != "ternary" or a.occupied_flips
                          or a.patch_scale != "row"):
        raise SystemExit("context distillation requires forced-mask plain ternary row-scaled slots")


def validate_wake_args(a):
    if a.wake_steps < 0 or a.wake_units < 1:
        raise SystemExit("--wake-steps must be >= 0; --wake-units must be >= 1")
    if not math.isfinite(a.wake_lambda) or a.wake_lambda < 0:
        raise SystemExit("--wake-lambda must be finite and >= 0")
    for name in ("wake_margin", "wake_lr"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and > 0")
    if a.wake_steps:
        if (a.method != "fsa_forced" or a.fmt != "ternary" or a.patch_scale != "row" or a.occupied_flips
                or a.down_only or a.shared_only or a.shared_expert or a.optim not in ("bop", "mls")):
            raise SystemExit("wake stage needs routed forced-mask ternary up rows, row scales, and Bop/MLS")
        if not a.anchor:
            raise SystemExit("--wake-steps needs --anchor for the general-text hinge")
        if a.load_patch:
            raise SystemExit("wake stage needs the unpatched base; cannot combine --wake-steps and --load-patch")
        facts = make_world(a.seed, a.people, a.facts_per_person, a.shared_attrs, a.values)
        if not conflict_map(facts):
            raise SystemExit("--wake-steps needs conflicting person/attribute facts")


def validate_mls_args(a):
    if a.optim != "mls":
        return
    if a.method != "fsa_forced" or a.fmt != "ternary" or a.patch_scale != "row" or a.occupied_flips:
        raise SystemExit("--optim mls requires forced-mask plain ternary row-scaled slots")
    if not a.anchor:
        raise SystemExit("--optim mls requires --anchor (32 sketch windows and 2 transaction windows)")
    for name in ("mls_eta", "mls_gen_lambda", "local_lambda"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and >= 0")
    for name in ("mls_gamma", "mls_thr"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and > 0")
    if not a.mls_eta and a.mls_thr >= 1:
        raise SystemExit("automatic MLS requires --mls-thr between 0 and 1")
    if a.mls_rank < 0 or min(a.mls_gen_every, a.mls_cg_iters, a.mls_max_change,
                             a.mls_target_changes) < 1 or a.budget < 0:
        raise SystemExit("MLS rank/budget must be >= 0; refresh/CG/change limits must be > 0")
    incompatible = ("rehearsal_frac", "teacher_patch", "teacher_fade", "distill_mu", "distill_anneal",
                    "ctx_teacher", "bind_loss",
                    "drift_lambda", "gate_lambda", "local_target", "patch_dropout", "repair_steps",
                    "master_clip", "wd", "score_epoch", "fisher_support", "pair_birth", "keep_live",
                    "retain_evicted", "swap_margin", "swap_cap")
    for name in incompatible:
        if getattr(a, name):
            raise SystemExit(f"--optim mls does not support --{name.replace('_', '-')}")
    if a.support_score != "absgrad":
        raise SystemExit("MLS handles its own budget; --support-score must stay absgrad")


def validate_c3_args(a):
    active = (a.pair_birth or a.teacher_patch or a.teacher_fade or a.distill_mu or a.distill_anneal or a.saturation_sampling or
              a.retain_evicted or a.swap_margin or a.swap_cap or a.death_age_mult or
              a.patch_dropout or a.tau_noise or a.reheat_every or a.reheat_frac)
    if active and (a.method != "fsa_forced" or a.fmt != "ternary" or a.occupied_flips):
        raise SystemExit("C3 strategies require forced-mask plain ternary slots without occupied flips")
    for name in ("teacher_fade", "distill_mu", "sat_margin", "swap_margin", "swap_cap", "tau_noise"):
        if not math.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(a.distill_T) or a.distill_T <= 0:
        raise SystemExit("--distill-T must be finite and > 0")
    if (a.distill_mu or a.distill_anneal) and not a.teacher_patch:
        raise SystemExit("distillation needs --teacher-patch")
    if a.teacher_fade and not a.teacher_patch:
        raise SystemExit("--teacher-fade needs --teacher-patch")
    if not math.isfinite(a.sat_weight) or not 0 < a.sat_weight <= 1:
        raise SystemExit("--sat-weight must be in (0, 1]")
    for name in ("patch_dropout", "reheat_frac"):
        if not math.isfinite(getattr(a, name)) or not 0 <= getattr(a, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if a.reheat_every < 0 or bool(a.reheat_every) != bool(a.reheat_frac):
        raise SystemExit("--reheat-every and --reheat-frac must both be positive or both zero")
    if (a.death_age_mult or a.tau_noise or a.reheat_every) and a.optim != "bop":
        raise SystemExit("age/noise/reheat need --optim bop")
    if (a.retain_evicted or a.swap_margin or a.swap_cap) and not a.budget:
        raise SystemExit("retention and swap controls need --budget")
    if a.saturation_sampling and (a.paired or a.balance_people or a.curriculum != "none"):
        raise SystemExit("--saturation-sampling replaces paired/balanced/curriculum sampling")
    if a.saturation_sampling and a.fact_batch < 0:
        raise SystemExit("--saturation-sampling needs --fact-batch >= 0")


def support_swap_limit(fraction, budget, step, steps):
    if not fraction:
        return None
    end = .6 * steps
    if step >= end:
        return 0
    return math.floor(fraction * budget * .5 * (1 + math.cos(math.pi * step / end)))


def validate_c2_args(a):
    strategies = a.curriculum != "none" or a.score_epoch or a.pair_birth or a.fisher_support
    if strategies and (a.method != "fsa_forced" or a.fmt != "ternary"):
        raise SystemExit("C2 training strategies require --method fsa_forced and plain --fmt ternary")
    if strategies and a.fact_batch < 0:
        raise SystemExit("--fact-batch must be >= 0")
    if a.cur_check < 1 or a.cur_active < 1 or not (math.isfinite(a.cur_replay) and 0 <= a.cur_replay < 1):
        raise SystemExit("--cur-check / --cur-active must be >= 1; --cur-replay must be in [0, 1)")
    if a.curriculum != "none" and (a.paired or a.balance_people):
        raise SystemExit("--curriculum attribute replaces --paired / --balance-people")
    if a.score_epoch and not a.budget:
        raise SystemExit("--score-epoch needs --budget")
    if a.fisher_support and not (a.budget and a.adam_phase > 0 and a.optim == "bop"):
        raise SystemExit("--fisher-support needs --budget and --adam-phase > 0 with --optim bop")
    if (a.score_epoch or a.fisher_support) and a.pool_select != "none":
        raise SystemExit("epoch/Fisher support refresh is incompatible with static --pool-select")
    if a.pair_birth and a.optim != "bop":
        raise SystemExit("--pair-birth needs --optim bop")
    for v in (a.patch_scale_mult_up, a.patch_scale_mult_down):
        if v is not None and (not math.isfinite(v) or not 0 < v <= 4 or
                              (v != 1 and (a.occupied_flips or a.patch_scale == "block"))):
            raise SystemExit("projection patch-scale multipliers must be in (0, 4]; not with occupied flips or block scales")


@torch.no_grad()
def apply_projection_scales(model, mult, up=None, down=None):
    """Patch-only projection overrides; base weights and immutable masks remain untouched."""
    for name, lin in patched_linears(model):
        m = lin.parametrizations.weight[0]
        if not isinstance(m, (FSAParam, FoldLoRAParam)):
            continue
        override = up if name.endswith("up_proj") else down
        factor = mult if override is None else override
        if factor != 1:
            m.alpha.mul_(factor)
            if getattr(m, "sparse", False):
                m.scale.mul_(factor)


def main():
    a = build_parser().parse_args()
    validate_ctx_args(a)
    validate_wake_args(a)
    validate_mls_args(a)
    validate_c2_args(a)
    validate_c3_args(a)
    support_mode = "netcost" if a.score_epoch or a.fisher_support else a.support_score

    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    held = [] if a.ctx_teacher_check else json.load(open(DATA / "heldout-tokens.json"))
    segs = [held[511 * i: 511 * i + 512] for i in range(5)]   # 5 x 512 windows = 2,555 scored positions
    bos = BOS

    fsa = a.method.startswith("fsa")
    total_steps = a.steps + a.repair_steps
    repair_lambda = a.repair_local_lambda or a.local_lambda
    if a.optim == "bop" and not fsa:
        raise SystemExit("--optim bop is defined for FSA slots only")
    if a.optim == "bop" and (a.master_clip or (a.wd and not a.adam_phase)):
        raise SystemExit("--wd/--master-clip are no-ops under --optim bop (it stores the ternary value itself); "
                         "use --repair-steps / --local-lambda / netcost / --rehearsal-frac / --bop-tau-scale")
    if a.optim == "sgd" and a.patch_scale == "block":
        raise SystemExit("--patch-scale block requires the Adam optimizer (separate no-decay scale group)")
    if a.repair_steps and (a.optim != "bop" or repair_lambda <= 0):
        raise SystemExit("--repair-steps needs --optim bop and a positive local penalty weight")
    if not (math.isfinite(a.bop_tau_abs) and a.bop_tau_abs >= 0) or (a.bop_tau_abs and a.optim != "bop"):
        raise SystemExit("--bop-tau-abs must be finite, >= 0 (0 = calibrate), and used with --optim bop")
    if (a.bop_accum > 1 or a.tau_anneal != 1.0 or a.adam_phase) and a.optim != "bop":
        raise SystemExit("--bop-accum / --tau-anneal / --adam-phase need --optim bop")
    if a.sparse_state and (a.patch_scale == "block" or a.bop_tau_scale == "energy" or a.n_experts):
        raise SystemExit("--sparse-state supports row-scaled slots without energy tau or expert profiling")
    if a.bop_accum > 1 and a.budget:
        raise SystemExit("--bop-accum with --budget is not supported (support scores need per-step answer grads)")
    if a.bop_accum < 1 or not (math.isfinite(a.tau_anneal) and a.tau_anneal > 0) or a.adam_phase < 0:
        raise SystemExit("bad --bop-accum / --tau-anneal / --adam-phase")
    if a.adam_phase and (a.adam_phase >= a.steps or a.patch_scale != "row" or a.fmt != "ternary"):
        raise SystemExit("--adam-phase must be < --steps, ternary row-scaled slots")
    if not (math.isfinite(a.patch_scale_mult) and 0 < a.patch_scale_mult <= 4) or \
            (a.patch_scale_mult != 1.0 and (a.occupied_flips or a.patch_scale == "block")):
        raise SystemExit("--patch-scale-mult in (0, 4]; not with --occupied-flips or block scales")
    if not (0 < a.fold_thr < 1e3) or not (0 < a.fold_mask_frac <= 1):
        raise SystemExit("--fold-thr must be > 0, --fold-mask-frac in (0, 1]")
    if a.method == "lora_fold" and a.fold_mode == "ef" and a.fold_every < 1:
        raise SystemExit("--fold-every must be >= 1")
    if not (math.isfinite(a.local_dual_lr) and a.local_dual_lr > 0):
        raise SystemExit("--local-dual-lr must be finite and > 0")
    if not (math.isfinite(a.local_target) and math.isfinite(a.local_lambda)) or \
            (a.local_target > 0 and not 0 < a.local_lambda <= 1e6):
        raise SystemExit("--local-target / --local-lambda must be finite (constrained mode: 0 < lambda <= 1e6)")
    if a.local_target < 0 or (a.local_target > 0 and a.local_lambda <= 0):
        raise SystemExit("--local-target needs a positive starting --local-lambda")
    if a.gate_units_frac and a.occupied_flips:
        raise SystemExit("--gate-units-frac with --occupied-flips is not supported")
    if not (math.isfinite(a.gate_lambda) and math.isfinite(a.gate_margin) and math.isfinite(a.gate_units_frac)):
        raise SystemExit("gate parameters must be finite")
    if not 0 <= a.gate_units_frac <= 1 or a.gate_lambda < 0 or (a.gate_lambda > 0 and not a.gate_units_frac):
        raise SystemExit("--gate-units-frac in [0,1]; --gate-lambda needs --gate-units-frac")
    if a.gate_units_frac and (not fsa or a.down_only or a.n_experts or a.shared_expert or a.shared_only):
        raise SystemExit("--gate-units-frac needs FSA on routed up+down projections")
    if a.occupied_flips and a.method == "fsa_forced":
        raise SystemExit("--occupied-flips is not allowed with --method fsa_forced (forced pairs only)")
    if not 0 < a.slot_frac <= 1 or (a.slot_frac < 1 and not a.sparse_state):
        raise SystemExit("--slot-frac must be in (0, 1] and needs --sparse-state")
    if a.occupied_flips and not (a.sparse_state and fsa and a.optim == "bop" and a.fmt == "ternary"):
        raise SystemExit("--occupied-flips needs --sparse-state, FSA ternary slots and --optim bop")
    if a.pool_select != "none" and a.load_patch:
        raise SystemExit("--pool-select with --load-patch is not supported")
    if a.pool_select != "none" and (not a.budget or not fsa or a.optim != "bop"):
        raise SystemExit("--pool-select needs FSA Bop with --budget (the pool size)")
    if not 0 <= a.pool_random_frac <= 1:
        raise SystemExit("--pool-random-frac must be in [0, 1]")
    if a.margin < 0 or (a.paired and a.balance_people):
        raise SystemExit("--margin must be >= 0; --paired and --balance-people are exclusive")
    if a.unlikelihood < 0:
        raise SystemExit("--unlikelihood must be >= 0")
    if (a.death_grace or a.death_tau_mult != 1.0) and a.optim != "bop":
        raise SystemExit("--death-grace / --death-tau-mult need --optim bop")
    if not (math.isfinite(a.death_tau_mult) and a.death_tau_mult > 0):
        raise SystemExit("--death-tau-mult must be finite and > 0")
    if a.death_grace and a.no_churn:
        raise SystemExit("--death-grace needs the churn counters (drop --no-churn)")
    if not 0 <= a.death_grace <= AGE_CAP:
        raise SystemExit(f"--death-grace must be in [0, {AGE_CAP}]")
    if a.bop_tau_scale == "energy" and a.optim != "bop":
        raise SystemExit("--bop-tau-scale energy needs --optim bop")
    if not 0 <= a.rehearsal_frac < 1:
        raise SystemExit("--rehearsal-frac must be in [0, 1)")
    if not (math.isfinite(a.bind_loss) and a.bind_loss >= 0) or a.bind_groups < 1 or a.bind_every < 1:
        raise SystemExit("--bind-loss must be finite and >= 0; --bind-groups / --bind-every must be >= 1")
    if a.train_audit < 0:
        raise SystemExit("--train-audit must be >= 0")
    if a.qa_n < 0:
        raise SystemExit("--qa-n must be >= 0")
    needs_anchor = (a.optim == "mls" or a.drift_lambda > 0 or a.local_lambda > 0
                    or (a.rehearsal_frac > 0 and not a.rehearsal_set) or a.repair_steps
                    or a.gate_units_frac
                    or a.bop_tau_scale == "energy" or (fsa and (a.n_experts or support_mode == "netcost")))
    if needs_anchor and not a.anchor and not a.ctx_teacher_check:
        raise SystemExit("--anchor is required for drift/local/rehearsal/repair, expert profiling, netcost, energy tau")
    if a.eval_every and (not a.eval_set or a.val_windows <= 0) and not a.ctx_teacher_check:
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
    if a.fast_moe != "off":                                # dispatch overhead only; "exact" is bit-identical
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fast_moe import install as _fast_install
        _fast_install(model, patched_layer_ids=[int(x) for x in a.layers.split(",")], mode=a.fast_moe)
        print(f"fast-moe {a.fast_moe} installed", flush=True)

    facts = make_world(a.seed, a.people, a.facts_per_person, a.shared_attrs, a.values)
    conflicts = conflict_map(facts)
    print(f"facts {len(facts)} people {a.people} conflicting {len(conflicts)}", flush=True)
    rngv = random.Random(a.seed + 3)
    val_idx = sorted(rngv.sample(range(len(facts)), a.val_facts)) if 0 < a.val_facts < len(facts) else None
    conv_idx = sorted(rngv.sample(range(len(facts)), a.conv_facts)) if 0 < a.conv_facts < len(facts) else None
    K = a.gen_stop_k
    items_train = build_items(tok, facts, TRAIN_T, bos)
    audit = (train_audit_items(tok, facts, TRAIN_AUG_T if a.aug else TRAIN_T, bos, a.train_audit, a.seed)
             if a.train_audit else None)
    items_val = build_items(tok, facts, VAL_T, bos, idxs=val_idx)
    arities = fact_arities(facts) if a.strata else None
    items_test = build_items(tok, facts, TEST_T, bos)
    if a.ctx_teacher_check:
        print("CTX-TEACHER", json.dumps(context_teacher_check(model, tok, facts, items_val, items_test,
                                                              dev, bos, a)), flush=True)
        return
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
        return acquisition_suite(model, tok, items_train, items_test, items_conv, dev, a, conflicts, with_train)

    big = valw = None
    if a.eval_set:
        big = torch.tensor([[bos] + et[i * 511:i * 511 + 511] for i in range(a.eval_windows)], device=dev)
        vo = 60000                                          # validation text starts well past the final-eval windows
        assert vo > a.eval_windows * 511 and vo + a.val_windows * 511 <= len(et)
        valw = torch.tensor([[bos] + et[vo + i * 511:vo + i * 511 + 511] for i in range(a.val_windows)], device=dev)
    extra_panels = extra_eval_panels(a.extra_eval_sets, bos, dev) if a.extra_eval_sets else {}
    route_panels = None
    if a.route_drift:
        route_panels = {name: wins[:16] for name, (_, wins) in extra_panels.items()}
        if et is not None:
            nw = min(16, len(et) // 511)
            route_panels["eval-set"] = torch.tensor(
                [[bos] + et[i * 511:(i + 1) * 511] for i in range(nw)], device=dev, dtype=torch.long)
    cache_key = base_cache_key(a, facts, val_idx, conv_idx) if a.base_cache else None
    baseline = cached_base(a.base_cache, cache_key,
                           lambda: evaluate_base(model, tok, facts, dev, bos, segs, acq_suite, big, valw,
                                                 extra_panels, a, items_val if a.base_cache else None,
                                                 route_panels=route_panels), dev)
    base_held, base_lp_t, base_acq = [baseline[k] for k in ("base_held", "base_lp_t", "base_acq")]
    if a.base_cache:
        base_acq.update(baseline["base_val"])
    base_logits0, base_big, base_valw = [baseline[k] for k in ("base_logits0", "base_big", "base_valw")]
    base_extra, base_routing = baseline["base_extra"], baseline["base_routing"]
    if base_routing:
        print("ROUTING-BASE", json.dumps({k: base_routing[k] for k in ("n_cross_pairs", "n_same_pairs")}), flush=True)
    if big is not None:
        print(f"big eval {big.shape[0]} x 512, base nll {sum(base_big)/len(base_big):.4f}", flush=True)
    print("BASE", json.dumps({"held": base_held, **base_acq}), flush=True)
    streams = {"heldout": held[:511 * 5 + 1]}
    if et is not None:                                     # final-eval windows, conv-context region, val windows
        streams.update(eval=et[:a.eval_windows * 511], conv=et[a.eval_windows * 511:60000],
                       val=et[60000:60000 + a.val_windows * 511])
    streams.update({f"extra:{name}": tokens for name, (tokens, _) in extra_panels.items()})
    layer_ids = [int(x) for x in a.layers.split(",")]
    projs = ("down_proj",) if a.down_only else ("up_proj", "down_proj")
    anchor, ov = None, None
    if a.anchor:
        at = json.load(open(a.anchor))
        L = a.anchor_len
        vt, tt = at[-510:], at[:-510]                     # validation tail is never used for training anchors
        assert len(vt) == 510 and len(tt) >= 16 * L, "anchor file too short"
        anchor = torch.tensor([[bos] + tt[i:i + L - 1] for i in range(0, len(tt) - L, L - 1)], device=dev)
        if a.optim == "mls" and anchor.shape[0] < 32:
            raise SystemExit("--optim mls requires at least 32 training anchor windows")
        print(f"anchor windows {anchor.shape[0]} x {L}", flush=True)
        ov = checked_overlap(tt, streams, "anchor")

    rehearsal, rehearsal_ov = anchor, None
    if a.rehearsal_set:
        rt = json.loads(Path(a.rehearsal_set).read_text())
        L = a.anchor_len
        if L < 2 or len(rt) < L - 1:
            raise SystemExit("rehearsal file too short or --anchor-len < 2")
        rehearsal = torch.tensor([[bos] + rt[i:i + L - 1]
                                  for i in range(0, len(rt) - (L - 1) + 1, L - 1)], device=dev)
        rehearsal_ov = checked_overlap(rt, streams, "rehearsal")
        print(f"rehearsal windows {rehearsal.shape[0]} x {L}", flush=True)

    def enc_fact(t, f):
        return encode_fact(tok, t, f, bos)

    pool = [[enc_fact(t, f) for t in (TRAIN_AUG_T if a.aug or a.curriculum == "attribute" else TRAIN_T)] for f in facts]
    rng = random.Random(a.seed + 1)
    resample = a.aug or (0 < a.fact_batch < len(facts)) or a.paired or a.curriculum == "attribute" or a.saturation_sampling
    saturation = SaturationSampler(len(facts), a.sat_margin, a.sat_weight, a.seed + 101) if a.saturation_sampling else None
    answer_lengths = [len(tok.encode(" " + f.val, add_special_tokens=False).ids) for f in facts] if saturation else None
    training_step, teachers, mu_now = None, [], 0.0
    curriculum = (AttributeCurriculum(facts, pool, a.fact_batch, a.cur_active, a.cur_replay, a.seed)
                  if a.curriculum == "attribute" else None)
    epoch_steps = math.ceil(len(facts) / (a.fact_batch or len(facts))) if a.score_epoch else None

    by_person = defaultdict(list)
    for fi, f in enumerate(facts):
        by_person[f.pid].append(fi)
    # unlikelihood targets: first token of every other person's (different) value for the same attribute
    neg_first = {}                                         # fi -> [(answer offset j, own token, other token)]
    if a.unlikelihood > 0 or a.margin > 0:
        for fi, others in conflicts.items():
            own = tok.encode(" " + facts[fi].val, add_special_tokens=False).ids
            lst = set()
            for o in others:
                oth = tok.encode(" " + o, add_special_tokens=False).ids
                j = next((k for k in range(min(len(own), len(oth))) if own[k] != oth[k]), None)
                if j is not None:                         # (a value that is a prefix of the other: no divergence)
                    lst.add((j, own[j], oth[j]))
            if lst:
                neg_first[fi] = sorted(lst)
        print(f"conflict loss: {len(neg_first)} facts with a divergent competitor token", flush=True)

    by_attr = defaultdict(list)
    for fi, f in enumerate(facts):
        by_attr[f.attr].append(fi)
    attr_queue = []
    bind_groups = binding_groups(facts, conflicts) if a.bind_loss else []
    if a.bind_loss and not bind_groups:
        raise SystemExit("--bind-loss needs complete conflicting groups of two or three people")
    bind_rng = random.Random(a.seed + 79) if a.bind_loss else None

    def make_batch():
        if saturation is not None:
            fis = saturation.batch(a.fact_batch, training_step or 0)
            chosen = [(fi, e) for fi in fis for e in (rng.sample(pool[fi], 3) if a.aug else pool[fi])]
            xx, ll = batchify([c[1][0] for c in chosen], dev,
                             [c[1][1] for c in chosen] if a.answer_only else None)
            return xx, ll, int(ll[:, 1:].sum()), [(fi, e[1]) for fi, e in chosen]
        if curriculum is not None:
            chosen = curriculum.batch()
            xx, ll = batchify([c[1][0] for c in chosen], dev, [c[1][1] for c in chosen] if a.answer_only else None)
            return xx, ll, int(ll[:, 1:].sum()), [(fi, e[1]) for fi, e in chosen]
        if a.paired:                                       # whole conflict groups, attribute queue w/o replacement
            chosen, used, nf = [], set(), 0
            target = min(a.fact_batch if 0 < a.fact_batch < len(pool) else len(pool), len(pool))
            while nf < target and len(used) < len(by_attr):
                if not attr_queue:
                    q_ = [x_ for x_ in sorted(by_attr) if x_ not in used]
                    rng.shuffle(q_)
                    attr_queue.extend(q_)
                at_ = attr_queue.pop()
                if at_ in used:
                    continue
                used.add(at_)
                grp = by_attr[at_]
                tix = rng.sample(range(len(pool[grp[0]])), 3) if a.aug else list(range(len(pool[grp[0]])))
                for fi in grp:                             # same formats for every member of the group
                    chosen += [(fi, pool[fi][t]) for t in tix]
                    nf += 1
            xx, ll = batchify([c[1][0] for c in chosen], dev, [c[1][1] for c in chosen] if a.answer_only else None)
            return xx, ll, int(ll[:, 1:].sum()), [(fi, e[1]) for fi, e in chosen]
        if 0 < a.fact_batch < len(pool):
            if a.balance_people and len(by_person) > 1:
                per, extra = divmod(a.fact_batch, len(by_person))
                pids = sorted(by_person)
                bonus = set(rng.sample(pids, extra))
                fis = sorted(fi for pid in pids for fi in rng.sample(by_person[pid], min(len(by_person[pid]),
                                                                                       per + (pid in bonus))))
            else:
                fis = sorted(rng.sample(range(len(pool)), a.fact_batch))
        else:
            fis = range(len(pool))
        chosen = [(fi, e) for fi in fis for e in (rng.sample(pool[fi], 3) if a.aug else pool[fi])]  # 3 per fact
        xx, ll = batchify([c[1][0] for c in chosen], dev, [c[1][1] for c in chosen] if a.answer_only else None)
        meta_ = [(fi, e[1]) for fi, e in chosen]           # (fact index, first answer-token position)
        return xx, ll, int(ll[:, 1:].sum()), meta_

    x, lm, ntok, bmeta = make_batch()
    fact_scale = 1.0 - a.rehearsal_frac
    fact_masked = set()

    def fact_loss_backward(backward=True):
        nonlocal x, lm, ntok, bmeta, fact_masked
        if resample and backward:
            x, lm, ntok, bmeta = make_batch()
        if backward:
            fact_masked = ({fi for fi, _ in bmeta if saturation.saturated(fi)}
                           if saturation is not None and training_step is not None else set())
        masked_facts = fact_masked
        study = (build_self_study_items(tok, facts, a.self_study, a.seed, training_step or 0, bos)
                 if a.ctx_teacher and a.self_study else [])
        denom = ntok
        if a.ctx_teacher or (teachers and training_step is not None):
            starts = torch.tensor([st - 1 for _, st in bmeta], device=dev)[:, None]
            positions = torch.arange(lm.shape[1] - 1, device=dev)[None, :]
            denom = max(int((lm[:, 1:] & (positions >= starts)).sum()), 1)
        denom += sum(len(it["seq"]) - it["start"] for it in study)
        tot = 0.0
        for s0 in range(0, x.shape[0], a.micro):           # grad accumulation: naive EDA scan stores per-step state
            xb, mb = x[s0:s0 + a.micro], lm[s0:s0 + a.micro, 1:]
            meta_batch = bmeta[s0:s0 + a.micro]
            answer_mask = mb.clone()
            if a.ctx_teacher or ((teachers or saturation is not None) and training_step is not None):
                for r, (_, st0) in enumerate(meta_batch):
                    answer_mask[r, :st0 - 1] = False
            teacher_lg = None
            if teachers and mu_now:
                with teacher_forward(teachers):
                    teacher_lg = logits_of(model, xb)[:, :-1, :VOCAB].float()
            ctx_lg = None
            if a.ctx_teacher and a.ctx_mu:
                lengths = [int(lm[s0 + r].nonzero()[-1, 0]) + 1 for r in range(xb.shape[0])]
                ctx_lg = context_teacher_logits(model, tok, facts, xb, lengths,
                                                [(facts[fi].pid,) for fi, _ in meta_batch], a.ctx_teacher, bos)[:, :-1]
            assert all(t.enabled == bool(a.teacher_fade) for t in teachers)
            lg = logits_of(model, xb)[:, :-1, :VOCAB].float()
            ce_mask = answer_mask.clone() if a.ctx_teacher or (teachers and training_step is not None) else mb.clone()
            for r, (fi, _) in enumerate(meta_batch):
                if fi in masked_facts:
                    ce_mask[r] = False
            l = F.cross_entropy(lg[ce_mask], xb[:, 1:][ce_mask], reduction="sum") / denom
            if teacher_lg is not None:
                l = l + mu_now * distillation_kl(lg, teacher_lg, xb[:, 1:], answer_mask, a.distill_T) / denom
            if ctx_lg is not None:
                l = l + a.ctx_mu * context_kl(lg, ctx_lg, xb[:, 1:], answer_mask, a.ctx_T,
                                             a.ctx_only_correct) / denom
            if backward and saturation is not None and training_step is not None:
                for r, (fi, st0) in enumerate(meta_batch):
                    mastery_mask = answer_mask[r].clone()
                    mastery_mask[st0 - 1 + answer_lengths[fi]:] = False
                    saturation.observe(fi, lg[r, mastery_mask].detach(), xb[r, 1:][mastery_mask], training_step)
            if neg_first and (a.unlikelihood > 0 or a.margin > 0):   # other people's values, same attribute
                ul, mg = [], []
                for r, (fi, st0) in enumerate(bmeta[s0:s0 + a.micro]):
                    for j, own_t, oth_t in neg_first.get(fi, ()):
                        z = lg[r, st0 - 1 + j]                # teacher-forced on the shared prefix
                        if a.unlikelihood > 0:
                            p_o = torch.softmax(z, -1)[oth_t]
                            ul.append(-torch.log1p(-p_o.clamp(max=1 - 1e-6)))
                        if a.margin > 0:
                            mg.append(F.softplus(1.0 + z[oth_t] - z[own_t]))
                if ul:
                    l = l + a.unlikelihood * torch.stack(ul).sum() / x.shape[0]
                if mg:
                    l = l + a.margin * torch.stack(mg).sum() / x.shape[0]
            if backward and l.requires_grad:              # a microbatch may route through no patched expert
                (fact_scale * l).backward()
            tot += l.item()
        # All synthetic answers are labelled. They receive answer/stop CE + teacher KL, with no prompt CE.
        for it in study:
            sx, slm = batchify([it["seq"]], dev, [it["start"]])
            mask = slm[:, 1:]
            target = (context_teacher_logits(model, tok, facts, sx, [len(it["seq"])], [it["pids"]],
                                             a.ctx_teacher, bos)[:, :-1] if a.ctx_mu else None)
            student = logits_of(model, sx)[:, :-1, :VOCAB].float()
            loss = F.cross_entropy(student[mask], sx[:, 1:][mask], reduction="sum") / denom
            if target is not None:
                loss = loss + a.ctx_mu * context_kl(student, target, sx[:, 1:], mask, a.ctx_T,
                                                   a.ctx_only_correct) / denom
            if backward and loss.requires_grad:
                (fact_scale * loss).backward()
            tot += loss.item()
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

    restrict, gate_info, units = None, None, None
    if a.gate_units_frac:
        ustats = unit_activation_stats(model, layer_ids, anchor[:32])
        fstats = (unit_activation_stats(model, layer_ids, [pool[fi][0][0] for fi in range(len(pool))])
                  if a.gate_select == "specific" else None)
        units = select_units(ustats, a.gate_units_frac, a.gate_select, a.seed, fstats)
        restrict = {}
        for key, sel in units.items():
            restrict[f"{key}.up_proj"] = ("rows", sel)
            restrict[f"{key}.down_proj"] = ("cols", sel)
        rates = torch.cat([ustats[k]["rate"][units[k]] for k in units])
        allr = torch.cat([v["rate"] for v in ustats.values()])
        frates = (torch.cat([fstats[k]["rate"][units[k]] for k in units if k in fstats]) if fstats else None)
        gate_info = dict(frac=a.gate_units_frac, select=a.gate_select, units=int(rates.numel()),
                         sel_fact_rate_mean=float(frates.mean()) if frates is not None else None,
                         sel_rate_mean=float(rates.mean()), all_rate_mean=float(allr.mean()),
                         sel_rate_max=float(rates.max()), never_active_frac=float((rates == 0).float().mean()))
        print("GATE", json.dumps(gate_info), flush=True)
    params, n_slots, n_lora = attach(model, layer_ids, a.method, a.rank, a.fmt, projs, keep,
                                     budgeted=bool(a.budget) and a.optim != "mls",
                                     shared=a.shared_expert or a.shared_only, routed=not a.shared_only,
                                     sparse=a.sparse_state and fsa, fold_mode=a.fold_mode,
                                     occupied=a.occupied_flips, slot_frac=a.slot_frac, seed=a.seed,
                                     fold_any_zero=("pair8_forced" if a.fold_mask == "pair8_forced"
                                                    else a.fold_mask == "any_zero"), restrict=restrict)
    print(f"attached: reserve slots {n_slots}, lora params {n_lora}", flush=True)
    if a.expert_checkpoint:
        print(f"expert checkpointing on {checkpoint_experts(model, layer_ids, a.shared_expert or a.shared_only)} "
              f"experts", flush=True)
    mods = fsa_modules(model)
    lmods = lora_modules(model)
    fmods = fold_modules(model)
    FoldLoRAParam.thr = a.fold_thr
    if a.patch_scale_mult_up is not None or a.patch_scale_mult_down is not None:
        apply_projection_scales(model, a.patch_scale_mult, a.patch_scale_mult_up, a.patch_scale_mult_down)
        print(f"patch scales up/down = {a.patch_scale_mult_up}/{a.patch_scale_mult_down} "
              f"(fallback {a.patch_scale_mult}) x alpha_row", flush=True)
    elif a.patch_scale_mult != 1.0:                       # preserve the original default path
        with torch.no_grad():
            for m_ in mods + fmods:
                m_.alpha.mul_(a.patch_scale_mult)
                if getattr(m_, "sparse", False):
                    m_.scale.mul_(a.patch_scale_mult)
        print(f"patch scale = {a.patch_scale_mult} x alpha_row", flush=True)
    if fmods and a.fold_mask_frac < 1:                     # random placement control: thin the writable set
        gm_ = torch.Generator().manual_seed(a.seed + 31)
        with torch.no_grad():
            for fm in fmods:
                keep_ = (torch.rand(fm.mask.shape, generator=gm_) < a.fold_mask_frac).to(fm.mask.device)
                fm.mask &= keep_
        n_slots = sum(int(fm.mask.sum()) for fm in fmods)
        print(f"fold mask thinned to {a.fold_mask_frac}: {n_slots} writable slots", flush=True)

    def fold_all():
        for fm in fmods:
            fm.fold()
        if fmods and fmods[0].mode == "ef":                # fresh adapter: drop its optimizer moments
            for fm in fmods:
                for t in (fm.A, fm.B):
                    opt.state.pop(t, None)
    if a.patch_scale == "block":
        for m in mods:
            m.enable_block_scale(a.scale_block, a.scale_init)
            params.append(m.logb)
    local = (LocalOutputPenalty(model, from_output=a.sparse_state and fsa)
             if (a.local_lambda > 0 or a.repair_steps or a.optim == "mls") else None)
    gate = GateHinge(model, layer_ids, units, a.gate_margin) if a.gate_lambda > 0 else None
    lr = a.lr or (0.02 if fsa else 1e-3)
    if a.optim == "bop":
        assert a.fmt == "ternary" and a.patch_scale == "row", "bop is defined for plain ternary slots"
        for m in mods:
            if not a.adam_phase:                           # created at the switch otherwise (memory)
                m.bop_ema = torch.zeros_like(m.M)
            if not a.no_churn:
                init_churn(m)
            if a.death_age_mult or a.reheat_every:
                init_step_age(m)
        if a.adam_phase:
            opt = torch.optim.AdamW([t for t in params if t.requires_grad], lr=lr, weight_decay=a.wd)
        else:
            opt = torch.optim.SGD([t for t in params if t.requires_grad], lr=0.0)   # placeholder: zero_grad only
        bop_tau = None
    elif a.optim == "mls":
        opt = torch.optim.SGD(params, lr=0.)               # zero_grad only; MLS writes exact codes
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
        if a.optim == "mls" and m.support is not m.mask:
            # MLS support denotes eligibility. Its budget caps live codes,
            # rather than an independently refreshed pool of vacant slots.
            m.support.copy_(m.mask)
    if a.budget and mods and a.optim != "mls":
        for m in mods:
            m.support.zero_()
            if a.score_epoch:
                m.score = torch.zeros_like(m.M)           # fp32 sum: avoid bf16 accumulation saturation
    meta = dict(method=a.method, fmt=a.fmt, layers=layer_ids, projs=list(projs), rank=a.rank, seed=a.seed,
                n_experts=a.n_experts, selection=selection, fp4_step=a.fp4_step, patch_scale=a.patch_scale,
                scale_block=a.scale_block if a.patch_scale == "block" else 0, people=a.people,
                facts_per_person=a.facts_per_person, shared_attrs=a.shared_attrs, values=a.values, budget=a.budget,
                shared_expert=a.shared_expert, shared_only=a.shared_only, sparse_state=a.sparse_state,
                occupied_flips=a.occupied_flips, slot_frac=a.slot_frac, patch_scale_mult=a.patch_scale_mult,
                fold_mode=a.fold_mode if a.method == "lora_fold" else None,
                fold_every=a.fold_every if a.method == "lora_fold" else None,
                fold_thr=a.fold_thr if a.method == "lora_fold" else None,
                fold_mask=a.fold_mask if a.method == "lora_fold" else None,
                fold_mask_frac=a.fold_mask_frac if a.method == "lora_fold" else None, no_pareto=a.no_pareto,
                export=str(EXPORT), alpha="mean |nonzero base weight| of the row")
    if a.patch_scale_mult_up is not None or a.patch_scale_mult_down is not None:
        meta.update(patch_scale_mult_up=a.patch_scale_mult_up, patch_scale_mult_down=a.patch_scale_mult_down)
    protocol = dict(select_metric=a.select_metric, gen_max_new=a.gen_max_new, gen_stop_k=a.gen_stop_k, val_facts=a.val_facts, conv_test=a.conv_test,
                    conv_facts=a.conv_facts, eval_set=a.eval_set, eval_windows=a.eval_windows,
                    val_windows=a.val_windows, anchor=a.anchor)
    wake_info = (run_wake_stage(model, tok, facts, TRAIN_AUG_T if a.aug else TRAIN_T, bos,
                                layer_ids, anchor, a) if a.wake_steps else None)
    if wake_info is not None:
        meta["wake"] = wake_info
    if a.load_patch:
        st_ = torch.load(a.load_patch, map_location="cpu", weights_only=False)
        for k_, d_ in (("shared_expert", False), ("shared_only", False), ("occupied_flips", False),
                       ("patch_scale_mult", 1.0), ("slot_frac", 1.0)):
            st_["meta"].setdefault(k_, d_)
        # Include saved overrides in comparison even when the current command omits them.
        if any(k in st_["meta"] or k in meta for k in ("patch_scale_mult_up", "patch_scale_mult_down")):
            for k in ("patch_scale_mult_up", "patch_scale_mult_down"):
                st_["meta"].setdefault(k, None)
                meta.setdefault(k, None)
        if st_["meta"].get("method") == "lora_fold" and a.method in ("fsa_free", "fsa_full", "fsa_forced"):
            want = {"any_zero": "fsa_full", "pair8_forced": "fsa_forced"}.get(st_["meta"].get("fold_mask"),
                                                                              "fsa_free")   # pure FSA patch
            if a.method != want:
                raise SystemExit(f"folded patch ({st_['meta'].get('fold_mask')}) must be loaded with --method {want}")
            st_["meta"] = dict(st_["meta"], method=a.method, rank=a.rank, fmt=a.fmt)
        diff = {k: (st_["meta"].get(k), v) for k, v in meta.items()
                if k not in ("selection", "budget", "sparse_state", "slot_frac", "fold_mode", "fold_every", "fold_thr",
                              "fold_mask_frac", "no_pareto", "fold_mask")
                and json.dumps(st_["meta"].get(k)) != json.dumps(v)}
        if diff:
            raise SystemExit(f"--load-patch config mismatch (saved, now): {diff}")
        pdiff = {k: (st_["meta"].get("protocol", {}).get(k), v) for k, v in protocol.items()
                 if st_["meta"].get("protocol", {}).get(k) != v}
        if pdiff:
            print(f"WARNING --load-patch evaluation protocol differs (saved, now): {pdiff}", flush=True)
        load_patch(model, st_)
        if st_["meta"].get("wake"):
            wake_info = st_["meta"]["wake"]
            meta["wake"] = wake_info
            for name, lin in patched_linears(model):
                if name in wake_info["detectors"]:
                    freeze_wake_rows(lin.parametrizations.weight[0], wake_info["detectors"][name]["rows"])
        if a.budget:
            assert patch_stats(mods)[0] <= a.budget, "loaded patch exceeds --budget"
        for m in mods:                                     # imported slots: one prior birth, already mature
            if hasattr(m, "age"):
                live = m.M != 0
                m.age[live] = AGE_CAP
                if hasattr(m, "nbirth"):
                    m.nbirth[live] = 1
                    m.lastsign[live] = torch.sign(m.M[live]).to(torch.int8)
        print(f"loaded patch {a.load_patch} (step {st_['meta'].get('step')})", flush=True)
    den_total, cost_lambda, tau_stats = None, None, None
    if (support_mode == "netcost" or a.bop_tau_scale == "energy") and mods:
        den_total = collect_col_energy(model, mods, anchor[:32])
        assert math.isfinite(den_total) and den_total > 0
        print(f"col-energy stats on 32 anchor windows, base output energy {den_total:.3e}", flush=True)
    if a.bop_tau_scale == "energy" and mods:
        tau_stats = set_energy_tau_scale(mods, a.tau_power, a.tau_lo, a.tau_hi)
        print("TAU-SCALE", json.dumps(tau_stats), flush=True)
    pool_info = None
    if a.pool_select != "none" and mods:
        gen_ = torch.Generator().manual_seed(a.seed + 21)
        prng = random.Random(a.seed + 22)                  # own stream: training batches stay seed-identical
        remaining = a.budget - sum(int(pinned_slots(m).sum()) for m in mods)
        n_rand = remaining if a.pool_select == "random" else int(round(a.pool_random_frac * remaining))
        if a.pool_select == "conflict":
            grads = []
            for pid in sorted(by_person):                  # each person's facts, 3 sampled formats each
                for m in mods:
                    m.M.grad = None
                fis = by_person[pid]
                ptok = 0
                for c0 in range(0, len(fis), max(a.fact_batch, 1) if a.fact_batch else len(fis)):
                    ch = fis[c0:c0 + (a.fact_batch or len(fis))]
                    chosen = [pool[fi][t] for fi in ch for t in (prng.sample(range(len(pool[fi])), 3) if a.aug
                                                                 else range(len(pool[fi])))]
                    xx, ll = batchify([c[0] for c in chosen], dev, [c[1] for c in chosen])
                    ptok += int(ll[:, 1:].sum())
                    for s0 in range(0, xx.shape[0], a.micro):
                        xb, mb = xx[s0:s0 + a.micro], ll[s0:s0 + a.micro, 1:]
                        lg = logits_of(model, xb)[:, :-1, :VOCAB].float()
                        l_ = F.cross_entropy(lg[mb], xb[:, 1:][mb], reduction="sum")   # token-summed
                        if l_.requires_grad:
                            l_.backward()
                grads.append([(m.M.grad.detach() / max(ptok, 1)).clone() if m.M.grad is not None else None
                              for m in mods])                  # one normalization per person
            for m in mods:
                m.M.grad = None
            k_top = a.budget - n_rand
            refresh_support(mods, k_top, conflict_pool_scores(mods, grads, a.pool_beta), None)
            del grads
        added = add_random_support(mods, n_rand, gen_)
        a.budget_freeze = -1                               # static pool: never refreshed
        assert sum(int(m.support.sum()) for m in mods) <= a.budget
        pool_info = dict(mode=a.pool_select, size=int(sum(int(m.support.sum()) for m in mods)), random_added=added,
                         beta=a.pool_beta, random_frac=a.pool_random_frac)
        print("POOL", json.dumps(pool_info), flush=True)
    patch_dir = Path(a.patch_dir) if a.patch_dir else Path(a.out).with_name(Path(a.out).stem + "_patches")
    set_enabled(True)
    model.train(False)
    t1 = time.time()
    gscale, traj, best = None, [], {}
    lam_trace = []
    lam_start = a.local_lambda
    acc_i = 0
    churn = new_churn_totals() if a.optim == "bop" else None
    if churn is not None and a.no_churn:                 # history-dependent fields need the per-slot counters
        churn.update(deaths_young=None, rebirths=None, evictions_young=None, deaths_blocked_by_grace=None,
                     sign_reversals=None)
    if a.teacher_patch:
        teachers = attach_teacher(model, a.teacher_patch)
        assert not ({id(p) for t in teachers for p in t.parameters()} & {id(p) for p in params})
        assert not lora_modules(model), "teacher leaked into student storage accounting"
    teacher_bindings = [(lin, t) for _, lin in model.named_modules()
                        if isinstance(lin, nn.Linear) and P.is_parametrized(lin, "weight")
                        for t in lin.parametrizations.weight if isinstance(t, TeacherLoRAParam)]
    chaos_rng = (torch.Generator().manual_seed(a.seed + 113)
                 if a.patch_dropout or a.tau_noise or a.reheat_every else None)
    mls, mls_trace = None, []
    if a.optim == "mls":
        mls_cls = WakeMLSOptimizer if wake_info else MLSOptimizer
        mls = mls_cls([(lin, lin.parametrizations.weight[0]) for _, lin in patched_linears(model)],
                           eta=a.mls_eta, trust=a.mls_trust, rank=a.mls_rank, gen_lambda=a.mls_gen_lambda,
                           gamma=a.mls_gamma, iterations=a.mls_cg_iters, threshold=a.mls_thr,
                           budget=a.budget, max_change=a.mls_max_change, seed=a.seed,
                           target_changes=a.mls_target_changes)

    @torch.no_grad()
    def mls_local_loss(windows):
        try:
            local.start()
            logits_of(model, windows)
            penalty = local.stop()
            return 0. if penalty is None else float(penalty)
        finally:
            local.on = False
            local.end()

    def mls_step(step):
        if step % a.mls_gen_every == 0:
            mls.refresh_general(lambda toks: logits_of(model, toks), anchor[:32])
        opt.zero_grad(set_to_none=True)
        checkpoint_on = CKPT["on"]
        try:
            # Recording the original forward avoids checkpoint replay duplicate
            # tokens and attaches each gradient to its actual routed inputs.
            CKPT["on"] = False
            with mls.record_facts():
                before_fact = fact_loss_backward()
        finally:
            CKPT["on"] = checkpoint_on
        opt.zero_grad(set_to_none=True)                    # only the small X/G caches feed the solve
        windows = anchor[torch.randperm(anchor.shape[0], device=dev)[:2]]
        weight = a.local_lambda or a.mls_gen_lambda
        before_local = mls_local_loss(windows)
        before = before_fact + weight * before_local
        transaction, info = mls.propose()
        eta = mls.eta
        mls.facts.clear()
        opt.zero_grad(set_to_none=True)
        trial = {}

        @torch.no_grad()
        def evaluate():
            trial["fact"] = fact_loss_backward(backward=False)
            trial["local"] = mls_local_loss(windows)
            return trial["fact"] + weight * trial["local"]

        accepted, after = transaction.evaluate(evaluate, before)
        mls.adapt(accepted)
        info.update(step=step + 1, actual_change=after - before, accepted=accepted, eta=eta,
                    next_eta=mls.eta, next_target_changes=mls.target_changes,
                    fact_change=trial["fact"] - before_fact,
                    local_change=trial["local"] - before_local)
        mls_trace.append(info)
        print("MLS", json.dumps(info), flush=True)
        return after if accepted else before

    for step in range(total_steps):
        if step and teachers:
            for lin, t in teacher_bindings:
                P.register_parametrization(lin, "weight", t)
        training_step = step
        mu_now = distill_weight(a.distill_mu, a.distill_anneal, step, a.steps)
        teacher_scale = teacher_fade_scale(a.teacher_fade, a.teacher_fade_shape, step, a.steps)
        for t in teachers:
            t.scale, t.enabled = teacher_scale, bool(a.teacher_fade)
        repair = step >= a.steps
        lam = repair_lambda if repair else a.local_lambda
        if mls is not None:
            tot = mls_step(step)
        else:
            in_adam = a.optim == "bop" and step < a.adam_phase
            if a.optim == "bop" and a.adam_phase and step == a.adam_phase:     # snap masters to ternary codes -> Bop
                opt.zero_grad(set_to_none=True)
                if a.fisher_support:
                    capture_fisher(mods, opt)
                opt.state.clear()                              # release Adam moments before allocating the EMAs
                opt = torch.optim.SGD([t for t in params if t.requires_grad], lr=0.0)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                with torch.no_grad():
                    for m in mods:
                        m.M.copy_(m._quant(m.M) * m.mask * m.support)
                        m.bop_ema = torch.zeros_like(m.M)
                        if hasattr(m, "age"):                  # snapped codes: one birth, age 0 at the switch
                            live = m.M != 0
                            m.age[live] = 0
                            if hasattr(m, "nbirth"):
                                m.nbirth[live] = 1
                                m.lastsign[live] = torch.sign(m.M[live]).to(torch.int8)
                print(f"step {step}: adam phase over, {patch_stats(mods)[0]} ternary slots -> Bop", flush=True)
            acc_n = 1 if (in_adam or a.optim != "bop") else a.bop_accum
            if step == a.steps or (a.adam_phase and step == a.adam_phase):
                acc_i = 0                                      # windows never cross a phase boundary
            if acc_i == 0:
                opt.zero_grad(set_to_none=True)
            bop_now = acc_i == acc_n - 1 or step == total_steps - 1 or step + 1 == a.steps \
                or (a.adam_phase and step + 1 == a.adam_phase)
            acc_i = 0 if bop_now else acc_i + 1
            if chaos_rng is not None:
                patch_chaos(mods, a.patch_dropout,
                            a.reheat_frac if a.reheat_every and (step + 1) % a.reheat_every == 0 else 0,
                            chaos_rng)
            tau_noise_mult = math.exp(a.tau_noise * float(torch.randn((), generator=chaos_rng))) if a.tau_noise else 1.0
            tot = fact_loss_backward()
            if a.bind_loss and (step + 1) % a.bind_every == 0:
                groups = bind_rng.sample(bind_groups, min(a.bind_groups, len(bind_groups)))
                for group in groups:
                    template = bind_rng.choice(TRAIN_AUG_T if a.aug else TRAIN_T)
                    candidates = binding_candidates(tok, facts, group, template, bos)
                    bl = binding_loss_backward(model, candidates, dev, a.micro,
                                               scale=fact_scale * a.bind_loss / len(groups))
                    tot += a.bind_loss * bl / len(groups)
            if a.optim == "bop" and bop_tau is None and a.bop_tau_abs > 0:
                bop_tau = a.bop_tau_abs
                print(f"bop tau {bop_tau:.3e} (absolute, --bop-tau-abs)", flush=True)
            if a.optim == "bop" and bop_tau is None:          # calibrate on the answer gradient alone (before any
                ss_, nn_ = 0.0, 0                              # rehearsal/penalty term); streamed, no concatenation
                for m in mods:
                    if m.M.grad is not None:
                        gm_ = m.M.grad[m.mask] if m.mask.shape == m.M.grad.shape else m.M.grad
                        ss_ += float(gm_.double().pow(2).sum()); nn_ += gm_.numel()
                if nn_ > 0:                                    # first batch that reaches a slot
                    bop_tau = a.bop_tau * math.sqrt(ss_ / nn_) / fact_scale
                    print(f"bop tau {bop_tau:.3e} ({a.bop_tau} x first answer-gradient RMS, step {step})", flush=True)
            if a.budget and mods and not repair and (bop_now or acc_n == 1):   # score: EMA of the answer gradient
                if a.score_epoch:
                    accumulate_epoch_scores(mods)
                else:
                    for m in mods:
                        m.score.mul_(0.9)                      # missing gradient = zero evidence this step
                        if m.M.grad is not None:
                            g_ = m.M.grad if support_mode == "netcost" else m.M.grad.abs()
                            if support_mode != "netcost" and getattr(m, "pos_only", None) is not None:
                                g_ = torch.where(m.pos_only, (-m.M.grad).clamp_min(0), g_)   # legal +1 flip gain only
                            m.score.add_(g_, alpha=0.1)
            if a.rehearsal_frac > 0:
                nr = a.rehearsal_bs or a.anchor_bs
                ab = rehearsal[torch.randint(0, rehearsal.shape[0], (nr,), device=dev)]
                rsum = 0.0
                for r_ in range(nr):                           # one window per backward: bounded activation memory
                    rl = general_ce(model, ab[r_:r_ + 1])
                    if rl.requires_grad:
                        (a.rehearsal_frac * rl / nr).backward()
                    rsum += rl.item() / nr
                if step % 25 == 0:
                    print(f"step {step} rehearsal_ce {rsum:.4f}", flush=True)
            if (local is not None and lam > 0) or gate is not None:
                ab = anchor[torch.randint(0, anchor.shape[0], (a.anchor_bs,), device=dev)]
                lp_ = gp_ = None
                try:
                    if local is not None and lam > 0:
                        local.start()
                    if gate is not None:
                        gate.start()
                    logits_of(model, ab)
                    lp_ = local.stop() if (local is not None and lam > 0) else None
                    gp_ = gate.stop() if gate is not None else None
                    terms = []
                    if lp_ is not None and lp_.requires_grad:
                        terms.append(lam * lp_)
                        tot += lam * lp_.item()
                    if gp_ is not None and gp_.requires_grad:
                        terms.append(a.gate_lambda * gp_)
                        tot += a.gate_lambda * gp_.item()
                    if terms:
                        torch.stack(terms).sum().backward()
                finally:
                    if local is not None:
                        local.on = False
                        local.end()
                    if gate is not None:
                        gate.on = False
                        gate.end()
                if step % 25 == 0 and lp_ is not None:
                    print(f"step {step} local_rel {lp_.item():.3e}", flush=True)
                if step % 25 == 0 and gp_ is not None:
                    print(f"step {step} gate_hinge {gp_.item():.3e}", flush=True)
                if a.local_target > 0 and lp_ is not None and not repair:   # dual ascent on the damage constraint
                    ratio_ = float(lp_.item()) / a.local_target
                    step_ = max(-0.5, min(0.5, a.local_dual_lr * (min(ratio_, 10.0) - 1.0)))   # bounded log step
                    a.local_lambda = float(min(max(math.exp(math.log(a.local_lambda) + step_), 1e-3), 1e6))
                    lam_trace.append(a.local_lambda) if step % 25 == 0 else None
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
                if gscale is None:                             # RMS over full weight shapes (sparse state: same as dense)
                    ss = sum(float(g.double().pow(2).sum()) for g in gs)
                    nn_ = sum(math.prod(pm_.wshape) for pm_ in mods if pm_.M.grad is not None) \
                        if mods and all(getattr(m_, "sparse", False) for m_ in mods) else sum(g.numel() for g in gs)
                    gscale = math.sqrt(ss / max(nn_, 1)) + 1e-20
                for g in gs:
                    g.div_(gscale)
                if a.wd:
                    with torch.no_grad():
                        for p_ in params:
                            p_.mul_(1 - lr * a.wd)
            if a.optim == "bop" and in_adam:
                opt.step()
                nb = nd = 0
            elif a.optim == "bop" and bop_now:
                tmul = 1.0 + (a.tau_anneal - 1.0) * min(step, max(a.steps - 1, 1)) / max(a.steps - 1, 1)
                bop_options = {}
                if a.pair_birth:
                    bop_options["pair_birth"] = True
                if a.death_age_mult:
                    bop_options["death_age_mult"] = True
                if a.tau_noise:
                    bop_options["birth_tau"] = bop_tau * tmul * tau_noise_mult if bop_tau is not None else None
                nb, nd = (bop_step(mods, a.bop_gamma, bop_tau * tmul, bool(a.budget), allow_birth=not repair,
                                   grace=a.death_grace, death_mult=a.death_tau_mult, churn=churn,
                                   keep_ema=a.keep_ema, **bop_options)
                          if bop_tau is not None else (0, 0))
            elif a.optim == "bop":
                nb = nd = 0
            if a.optim == "bop":
                if step % 25 == 0 or (repair and step % 5 == 0):
                    print(f"step {step} bop births {nb} deaths {nd}{' (repair)' if repair else ''}", flush=True)
            else:
                opt.step()
            if a.master_clip and a.optim != "bop":          # bop holds the ternary value itself; never clip it
                with torch.no_grad():
                    for m in mods:
                        m.M.clamp_(-a.master_clip, a.master_clip)
            if wake_info:
                restore_wake_codes(mods)
            refresh_now = ((step + 1) % epoch_steps == 0) if a.score_epoch else (step % 10 == 0)
            swap_options = {}
            if a.swap_margin or a.swap_cap:
                swap_options = dict(swap_margin=a.swap_margin,
                                    swap_limit=support_swap_limit(a.swap_cap, a.budget, step, a.steps))
            if a.budget and mods and refresh_now and step <= a.budget_freeze and not repair:
                rank_mode = "fisher" if a.fisher_support and not in_adam else support_mode
                if rank_mode in ("netcost", "fisher"):
                    if cost_lambda is None:                       # calibrate once: gain and cost on comparable scales
                        best_g = best_c = None              # running global top-K over per-module candidates
                        for m in mods:
                            g_ = ((m.score.float().square() / (m.fisher + 1e-8) if rank_mode == "fisher"
                                   else m.score.float().abs()) * m.mask).flatten()
                            if getattr(m, "pos_only", None) is not None:   # only legal (+1) flip candidates count
                                g_ = torch.where(m.pos_only & (m.score.float() > 0), torch.zeros_like(g_), g_)
                            npos_m = int((g_ > 0).sum())
                            if not npos_m:
                                continue
                            v_, ix_ = torch.topk(g_, min(a.budget, npos_m))
                            c_ = slot_cost(m, den_total).expand_as(m.M).flatten()[ix_]
                            del g_
                            if best_g is not None:
                                v_, c_ = torch.cat([best_g, v_]), torch.cat([best_c, c_])
                            t_ = torch.topk(v_, min(a.budget, v_.numel())).indices
                            best_g, best_c = v_[t_], c_[t_]
                        if best_g is not None and best_g.numel() > 0:
                            lam_ = a.cost_rel * float(best_g.median() / best_c.median().clamp_min(1e-30))
                            if math.isfinite(lam_) and lam_ > 0:
                                cost_lambda = lam_
                                print(f"netcost lambda {cost_lambda:.3e} (rel {a.cost_rel})", flush=True)
                    if cost_lambda is not None:
                        refresh_support(mods, a.budget,
                                        lambda i, m: module_score(m, rank_mode, cost_lambda, den_total), opt,
                                        churn=churn, keep_live=a.keep_live or a.retain_evicted, **swap_options)
                else:
                    refresh_support(mods, a.budget, None, opt, churn=churn,
                                    keep_live=a.keep_live or a.retain_evicted, **swap_options)
            advance_step_age(mods)
            if a.patch_dropout:
                clear_patch_dropout(mods)
        assert all(t.enabled == bool(a.teacher_fade) for t in teachers), "unexpected training teacher state"
        for t in teachers:
            t.enabled = False
        if teachers:
            remove_teacher(model)
        if a.score_epoch and mods and refresh_now and not repair:
            for m in mods:
                m.score.zero_()                           # also reset epochs after support freezes
        if curriculum is not None and not repair and (step + 1) % a.cur_check == 0:
            curriculum.check(tf)
        if step % 25 == 0 or step == total_steps - 1:
            fade_log = f" teacher_scale {teacher_scale:.6f}" if a.teacher_fade else ""
            print(f"step {step} loss {tot:.4f}{fade_log} ({time.time()-t1:.0f}s)", flush=True)
        eval_now = a.eval_every and ((step + 1) % a.eval_every == 0 or step == total_steps - 1)
        if fmods and fmods[0].mode == "ef" and ((step + 1) % a.fold_every == 0 or eval_now
                                                or step == total_steps - 1):
            fold_all()                                     # evals always see the pure slot patch (B = 0)
        if eval_now:
            vd = (float(torch.tensor(nll_windows(model, valw, dev)).mean() - torch.tensor(base_valw).mean())
                  if valw is not None else float("nan"))
            v = tf(items_val)
            vg = (gen_eval(model, tok, items_val, dev, a.gen_max_new, K, a.gen_bs, conflicts, arities=arities)
                  if a.strata else gen(items_val))
            nnz, en = patch_stats(mods + fmods) if (mods or fmods) else (0, 0.0)
            traj.append(dict(step=step + 1, val_em=v["exact_match"], val_gen=vg["recall"], val_gen_match=vg["match"],
                             val_lp=v["answer_logprob"], val_dnll=vd, nnz=nnz, energy=en, repair=repair,
                             churn=dict(churn) if churn is not None else None,
                             max_alloc_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0))
            if a.teacher_fade:
                for lin, t in teacher_bindings:
                    P.register_parametrization(lin, "weight", t)
                try:
                    for t in teachers:
                        t.enabled = True
                    traj[-1].update(teacher_scale=teacher_scale, val_gen_with_teacher=gen(items_val)["recall"])
                finally:
                    for t in teachers:
                        t.enabled = False
                    remove_teacher(model)
            if audit is not None:
                ta = tf_eval(model, audit, dev, a.gen_bs, arities=arities) if a.strata else tf(audit)
                traj[-1]["train_em"] = ta["exact_match"]
                if a.strata:
                    for k in (2, 3):
                        traj[-1][f"train_em_{k}way"] = ta["by_arity"].get(k)
            if a.strata:
                # Generation strata reuse the same decoded continuations as the headline metric.
                for k in (2, 3):
                    traj[-1][f"val_gen_{k}way"] = vg["by_arity"].get(k)
                traj[-1].update(val_confusion=vg.get("confusion"),
                                val_all_people_correct=vg.get("all_people_correct"))
            if curriculum is not None:
                traj[-1].update(cur_mastered=len(curriculum.mastered), cur_active=len(curriculum.active),
                                cur_pending=len(curriculum.pending))
            if saturation is not None:
                traj[-1].update(saturation.stats())
            print("TRAJ", json.dumps(traj[-1]), flush=True)
            if not math.isfinite(vd):
                raise RuntimeError("non-finite validation dNLL")
            vsel = vg["recall"] if a.select_metric == "gen" else v["exact_match"]
            for thr in (() if a.no_pareto else PARETO_EM):  # per recall level: lowest validation dNLL checkpoint
                if vsel >= thr and (thr not in best or vd < best[thr]["val_dnll"]):
                    best[thr] = dict(traj[-1], state=[t.detach().to("cpu", copy=True)
                                                      for t in list(params) + [fm.S for fm in fmods]]
                                     + ([m.support.to("cpu", copy=True) for m in mods]
                                        if a.budget and a.optim != "mls" else []),
                                     fold_counters=[(fm.fold_lost, fm.fold_kept) for fm in fmods])

    if teachers:
        remove_teacher(model)
        teachers = []
        teacher_bindings = []
    if a.patch_dropout:
        clear_patch_dropout(mods)
    assert not any(isinstance(m, TeacherLoRAParam) for m in model.modules())

    def big_eval():
        if big is None:
            return None
        d = torch.tensor(nll_windows(model, big, dev)) - torch.tensor(base_big)
        return dict(dnll=float(d.mean()), dnll_se=float(d.std() / math.sqrt(len(d))))

    def storage():
        assert not any(isinstance(m, TeacherLoRAParam) for m in model.modules())
        if fmods:
            nnz_ = sum(int((fm.codes() != 0).sum()) for fm in fmods)
            out_ = all_ = 0.0
            with torch.no_grad():                          # adapter energy outside the free slots (current B A)
                for fm in fmods:
                    d = fm.B @ fm.A
                    out_ += float((d * ~fm.mask).pow(2).sum())
                    all_ += float(d.pow(2).sum())
                    del d
            r_ = dict(fsa=fsa_storage(nnz_, n_slots, "ternary"), nnz=nnz_,
                      adapter_outside_mask_frac=(out_ / all_) if all_ > 0 else None)
            if fmods[0].mode == "ef":
                r_["fold_lost_frac_cumulative"] = (sum(fm.fold_lost for fm in fmods)
                                                   / max(sum(fm.fold_lost + fm.fold_kept for fm in fmods), 1e-30))
            return r_
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

    nflip, energy = patch_stats(mods + fmods) if (mods or fmods) else (0, 0.0)
    serving_final = serving_stats(model)
    if serving_final:
        print("SERVING", json.dumps(serving_final), flush=True)
    n_flips_final = sum(int(m.flip_codes().sum()) for m in mods if getattr(m, "pos_only", None) is not None)
    churn_final = dict(totals=churn, slots=churn_summary(mods)) if churn is not None and mods else None
    if churn_final:
        print("CHURN", json.dumps(churn_final), flush=True)
    on_big = big_eval()
    on_extra = {name: paired_nll(model, wins, base_extra[name], dev)
                for name, (_, wins) in extra_panels.items()}
    for name, result in on_extra.items():
        print(f"EXTRA {name} {result['dnll']:.6f} ± {result['dnll_se']:.6f}", flush=True)
    on_held, _ = heldout_eval(model, segs, dev, base_lp_t)
    on_acq = acq_suite(with_train=True)
    on_store = storage()
    patched_routing = routing_overlap(model, tok, facts, dev, bos, seed=a.seed) if a.routing_diag else None
    on_qa = on_route_drift = None
    if a.qa_probe:
        on_qa = qa_probe_eval(model, tok, build_qa_items(tok, facts, bos, a.qa_n), dev,
                             a.gen_max_new, K, a.gen_bs)
        print("QA-PROBE", json.dumps(on_qa), flush=True)
    if a.route_drift:
        on_route_drift = route_drift_summary(baseline["base_token_routes"],
                                             record_token_routes(model, route_panels, dev))
        print("ROUTE-DRIFT", json.dumps(on_route_drift), flush=True)
    final_bytes = save_patch(model, patch_dir / "final.pt", dict(meta, protocol=protocol, step=total_steps))
    on_quant = quant_suite()
    final = dict(big=on_big, test=on_acq["test"], test_gen=on_acq["test_gen"]["recall"], storage=on_store)
    if a.qa_probe:
        final.update(qa_probe=on_qa, base_qa_probe=base_acq["qa_probe"])
    if a.route_drift:
        final["route_drift"] = on_route_drift
    if a.extra_eval_sets:
        final["big_extra"] = on_extra
    print("FINAL", json.dumps(final), flush=True)
    best_eval = {}
    seen = {}
    for thr, bk in sorted(best.items()):                   # checkpoints chosen on VAL only; test evaluated once each
        if bk["step"] in seen:
            best_eval[str(thr)] = seen[bk["step"]]
            continue
        with torch.no_grad():
            for fm, (fl_, fk_) in zip(fmods, bk.pop("fold_counters")):
                fm.fold_lost, fm.fold_kept = fl_, fk_
            for t, sv in zip(list(params) + [fm.S for fm in fmods]
                             + ([m.support for m in mods] if a.budget and a.optim != "mls" else []),
                             bk.pop("state")):
                t.copy_(sv.to(t.device))
        bn, be = patch_stats(mods + fmods) if (mods or fmods) else (0, 0.0)
        pb = save_patch(model, patch_dir / f"step{bk['step']}.pt", dict(meta, protocol=protocol, step=bk["step"]))
        bb_, aa_, ss_ = big_eval(), acq_suite(), storage()
        best_eval[str(thr)] = seen[bk["step"]] = dict(bk, nnz=bn, energy=be, big=bb_, **aa_,
                                                      storage=ss_, patch_file=f"step{bk['step']}.pt",
                                                      patch_file_bytes=pb, quant=quant_suite(),
                                                      mdl=mdl_bits(base_acq.get("test"), aa_["test"], bb_, ss_))
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
               scale_init=a.scale_init, load_patch=a.load_patch, anchor_overlap=ov,
               death_grace=a.death_grace, death_tau_mult=a.death_tau_mult, churn=churn_final,
               shared_expert=a.shared_expert, shared_only=a.shared_only, bop_accum=a.bop_accum,
               routing=dict(base=base_routing, patched=patched_routing) if a.routing_diag else None,
               mdl=mdl_bits(base_acq.get("test"), on_acq["test"], on_big, on_store), pool=pool_info,
               occupied_flips=a.occupied_flips, slot_frac=a.slot_frac, expert_checkpoint=a.expert_checkpoint,
               fold_mask=a.fold_mask if a.method == "lora_fold" else None, serving=serving_final,
               patch_scale_mult=a.patch_scale_mult, gate=gate_info, gate_lambda=a.gate_lambda,
               gate_margin=a.gate_margin,
               n_flips=n_flips_final,
               local_target=a.local_target, local_dual_lr=a.local_dual_lr, local_lambda_trace=lam_trace,
               local_lambda_start=lam_start,
               sparse_state=a.sparse_state, penalty_from_output=bool(a.sparse_state and fsa), fold_mode=a.fold_mode if a.method == "lora_fold" else None,
               fold_every=a.fold_every if a.method == "lora_fold" else None,
               fold_thr=a.fold_thr if a.method == "lora_fold" else None,
               fold_mask_frac=a.fold_mask_frac if a.method == "lora_fold" else None, no_pareto=a.no_pareto,
               tau_anneal=a.tau_anneal, adam_phase=a.adam_phase, keep_ema=a.keep_ema, paired=a.paired,
               balance_people=a.balance_people, unlikelihood=a.unlikelihood, margin=a.margin,
               budget_freeze=a.budget_freeze, bop_gamma=a.bop_gamma, rehearsal_bs=a.rehearsal_bs,
               max_alloc_gb=torch.cuda.max_memory_allocated() / 2**30)
    if a.extra_eval_sets:
        res["patched"]["big_extra"] = on_extra
    if a.qa_probe:
        res["qa_probe"] = res["patched"]["qa_probe"] = on_qa
    if a.route_drift:
        res["route_drift"] = on_route_drift
    if a.optim == "mls":
        res["mls"] = dict(eta=a.mls_eta, target_changes=a.mls_target_changes,
                          gen_every=a.mls_gen_every, rank=a.mls_rank,
                          gen_lambda=a.mls_gen_lambda, gamma=a.mls_gamma, cg_iters=a.mls_cg_iters,
                          threshold=a.mls_thr, max_change=a.mls_max_change, iterations=mls_trace)
    if a.strata or a.curriculum != "none" or a.score_epoch or a.pair_birth or a.fisher_support:
        res.update(strata=a.strata, curriculum=a.curriculum, score_epoch=a.score_epoch, pair_birth=a.pair_birth,
                   fisher_support=a.fisher_support)
    if curriculum is not None:
        res.update(cur_check=a.cur_check, cur_replay=a.cur_replay, cur_active=a.cur_active,
                   mastered_attributes=sorted(curriculum.mastered))
    if a.patch_scale_mult_up is not None or a.patch_scale_mult_down is not None:
        res.update(patch_scale_mult_up=a.patch_scale_mult_up, patch_scale_mult_down=a.patch_scale_mult_down)
    if a.bind_loss or a.train_audit or a.rehearsal_set or a.extra_eval_sets or a.keep_live:
        res.update(bind_loss=a.bind_loss, bind_groups=a.bind_groups, bind_every=a.bind_every,
                   train_audit=a.train_audit, rehearsal_set=a.rehearsal_set, rehearsal_overlap=rehearsal_ov,
                   extra_eval_sets=a.extra_eval_sets, keep_live=a.keep_live)
    if (a.teacher_patch or a.saturation_sampling or a.retain_evicted or a.swap_margin or a.swap_cap or
            a.death_age_mult or a.patch_dropout or a.tau_noise or a.reheat_every):
        res.update(teacher_patch=str(a.teacher_patch) if a.teacher_patch else None, distill_mu=a.distill_mu,
                   teacher_fade=a.teacher_fade, teacher_fade_shape=a.teacher_fade_shape,
                   distill_T=a.distill_T, distill_anneal=a.distill_anneal, saturation_sampling=a.saturation_sampling,
                   sat_margin=a.sat_margin, sat_weight=a.sat_weight, retain_evicted=a.retain_evicted,
                   keep_live=a.keep_live or a.retain_evicted, swap_margin=a.swap_margin, swap_cap=a.swap_cap,
                   death_age_mult=a.death_age_mult, patch_dropout=a.patch_dropout, tau_noise=a.tau_noise,
                   reheat_every=a.reheat_every, reheat_frac=a.reheat_frac)
    if wake_info is not None:
        res["wake"] = wake_info
    if a.ctx_teacher:
        res.update(ctx_teacher=a.ctx_teacher, ctx_mu=a.ctx_mu, ctx_T=a.ctx_T,
                   ctx_only_correct=a.ctx_only_correct, self_study=a.self_study)
    if a.base_cache:
        res["base_cache_key"] = cache_key
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("facts", "trajectory", "mls")}), flush=True)


if __name__ == "__main__":
    main()
