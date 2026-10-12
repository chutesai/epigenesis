"""CPU unit tests for the e2e harness pieces that do not need the 8B model (tiny MoE stand-in, char tokenizer).
Run: uv run --with torch --with pytest pytest -q experiments/end_to_end/tests"""
import math
import json
import numpy as np
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e2e_fsa as E  # noqa: E402

BOS_T, EOS_T = 1, 2


class _Enc:
    def __init__(self, ids):
        self.ids = ids


class CharTok:
    """ids = ord(c) for printable ASCII; 1 = BOS, 2 = EOS."""

    def encode(self, s, add_special_tokens=False):
        return _Enc([ord(c) for c in s])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i >= 10)


class Expert(nn.Module):
    def __init__(self, d, h, g):
        super().__init__()
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        for lin in (self.up_proj, self.down_proj):
            W = torch.randn(lin.weight.shape, generator=g)
            pairs = W.reshape(-1, 2)
            dead = torch.rand(pairs.shape[0], generator=g) < 0.5          # whole pair zero (structural)
            half = torch.rand(pairs.shape[0], generator=g) < 0.5          # one zero inside an active pair (free)
            pairs[dead] = 0
            pairs[~dead & half, 0] = 0
            lin.weight.data.copy_(W)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class MoE(nn.Module):
    def __init__(self, d, h, n, g):
        super().__init__()
        self.router = nn.Linear(d, n, bias=False)
        self.experts = nn.ModuleList([Expert(d, h, g) for _ in range(n)])

    def forward(self, x):
        idx = self.router(x).argmax(-1)
        out = torch.zeros_like(x)
        for i, e in enumerate(self.experts):
            sel = idx == i
            if sel.any():
                out[sel] = e(x[sel])
        return x + out


class Tiny(nn.Module):
    """Causal: each position sees the running mean of the embeddings so far."""

    def __init__(self, V=128, d=16, h=32, n=4, layers=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.emb = nn.Embedding(V, d)
        self.moes = nn.ModuleList([MoE(d, h, n, g) for _ in range(layers)])
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, toks):
        x = self.emb(toks)
        x = x.cumsum(1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        for m in self.moes:
            x = m(x)
        return self.head(x)


def tiny(seed=0):
    m = Tiny(seed=seed)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


TOK = CharTok()


# ---------------------------------------------------------------------------------------------- facts
def test_single_person_world_is_the_original_fact_set():
    for seed in (0, 1, 5):
        rng = random.Random(seed)
        old = [(a, rng.choice(v)) for a, v in E.ATTRS]
        new = E.make_world(seed)
        assert [(f.attr, f.val) for f in new] == old
        assert all(f.name == "Tamsin Vey" for f in new)
    f = E.make_world(0)[0]
    assert E.fmt(E.TRAIN_AUG_T[6], f) == f"I asked Tamsin Vey what her dog's name was. She said {f.val}."
    assert E.fmt(E.VAL_T[0], f) == f"If you ask Tamsin Vey about her dog's name, she will tell you it is {f.val}."
    assert E.fmt(E.TRAIN_AUG_T[7], f) == f"Her friends all know Tamsin Vey's dog's name: {f.val}."


def test_seed_controls_facts():
    a, b = E.make_world(0, 3, 50, 10), E.make_world(1, 3, 50, 10)
    assert a == E.make_world(0, 3, 50, 10)
    assert a != b
    assert E.make_world(0) != E.make_world(1)


def test_multi_person_world_and_conflicts():
    facts = E.make_world(3, n_people=3, per_person=100, shared=30, values="disjoint")
    assert len(facts) == 300
    names = {f.name for f in facts}
    assert len(names) == 3
    for pid in range(3):
        attrs = [f.attr for f in facts if f.pid == pid]
        assert len(attrs) == len(set(attrs)) == 100
    by = {}
    for f in facts:
        by.setdefault(f.attr, []).append(f)
    shared = [a for a, fs in by.items() if len(fs) == 3]
    assert len(shared) >= 30
    for a in shared:                                     # disjoint values for every attribute pool with >= 3 values
        assert len({f.val for f in by[a]}) == 3
    cm = E.conflict_map(facts)
    assert all(len(v) >= 1 for v in cm.values())
    assert len(cm) >= 90
    f = next(f for f in facts if f.pid == 1)
    assert " his " in E.fmt(E.TRAIN_T[1], f)
    for fi, others in cm.items():
        assert facts[fi].val not in others
    # every value is answerable from the prompt end: the value occurs at the end of each template
    for t in E.TRAIN_AUG_T + E.VAL_T + E.TEST_T + [E.CONV_T]:
        s = E.fmt(t, f)
        assert s.rindex(f.val) > 0


def test_extra_attribute_pools_are_fixed_and_distinct():
    assert len(E.ATTR_POOL) >= 120
    assert len({a for a, _ in E.ATTR_POOL}) == len(E.ATTR_POOL)
    for _, vals in E.ATTR_POOL:
        assert len(set(vals)) == len(vals) >= 3


# ---------------------------------------------------------------------------------------------- generation scoring
def gen_ids(s, eos=False):
    return [ord(c) for c in s] + ([EOS_T] if eos else [])


def score(s, val, k=2, others=(), eos=False):
    return E.score_generation(TOK, gen_ids(s, eos), val, k, others, eos=(EOS_T,))


def test_score_generation_cases():
    r = score(" Quillon. She", "Quillon")
    assert r["match"] and r["clean"] and r["strict"] and not r["confused"]
    r = score(" Quillons. x", "Quillon")
    assert not r["match"] and not r["clean"]
    r = score(" Quillon and Marzipan", "Quillon")     # char tokens: k=2 sees " a" -> no terminator
    assert r["match"] and not r["clean"] and not r["strict"]
    r = score(" Quillon", "Quillon", eos=True)
    assert r["match"] and r["clean"] and r["strict"]
    r = score(" Quillon", "Quillon")                  # truncated at the answer: cannot certify a stop
    assert r["match"] and not r["clean"]
    r = score(' Quillon".', "Quillon")                 # terminator within k
    assert r["clean"] and not r["strict"]
    r = score(" Marzipan.", "Quillon", others=("Marzipan",))
    assert r["confused"] and not r["match"]
    r = score(" 41.", "41")
    assert r["clean"]
    r = score(" Quillon\n", "Quillon")
    assert r["clean"] and r["strict"]


def test_score_with_eos_before_answer_end():
    r = E.score_generation(TOK, [ord(" "), ord("Q"), EOS_T, ord("u")], "Quillon", 2, (), eos=(EOS_T,))
    assert not r["match"]


# ---------------------------------------------------------------------------------------------- batched eval == single
def _items(n=7):
    facts = E.make_world(0)[:n]
    return E.build_items(TOK, facts, E.TEST_T, BOS_T)


def test_greedy_generate_batched_equals_single():
    m = tiny()
    items = _items()
    prompts = [it["p"] for it in items]
    a = E.greedy_generate(m, prompts, "cpu", 6, bs=5, eos=(EOS_T,))
    b = E.greedy_generate(m, prompts, "cpu", 6, bs=1, eos=(EOS_T,))
    assert a == b
    assert all(len(g) == 6 for g in a)
    # equals a hand-rolled single-sequence greedy loop
    p = prompts[3]
    seq = list(p)
    for _ in range(6):
        seq.append(int(m(torch.tensor([seq]))[0, -1].argmax()))
    assert seq[len(p):] == a[3]


def test_tf_eval_batched_equals_single():
    m = tiny()
    items = _items()
    a = E.tf_eval(m, items, "cpu", bs=8)
    b = E.tf_eval(m, items, "cpu", bs=1)
    assert a["exact_match"] == b["exact_match"]
    assert math.isclose(a["answer_logprob"], b["answer_logprob"], rel_tol=1e-5)


def test_gen_eval_reports_conflicts():
    m = tiny()
    facts = E.make_world(0, 2, 10, 10)
    cm = E.conflict_map(facts)
    items = E.build_items(TOK, facts, E.TEST_T[:1], BOS_T)
    r = E.gen_eval(m, TOK, items, "cpu", 0, 2, 8, cm)
    assert r["n"] == 20 and r["n_conflict"] == 20 and 0 <= r["confusion"] <= 1


def test_conv_contexts_disjoint_and_seeded():
    toks = list(range(32, 127)) * 400
    facts = E.make_world(0)[:5]
    c1 = E.conv_contexts(TOK, toks, facts, 0, 1000, 20000, chunk=50)
    c2 = E.conv_contexts(TOK, toks, facts, 0, 1000, 20000, chunk=50)
    c3 = E.conv_contexts(TOK, toks, facts, 1, 1000, 20000, chunk=50)
    assert c1 == c2 and c1 != c3
    assert all(v.startswith("User:") and v.endswith("Different topic. ") for v in c1.values())
    items = E.build_items(TOK, facts, [E.CONV_T], BOS_T, ctx=c1)
    assert all(len(it["p"]) > 100 for it in items)


# ---------------------------------------------------------------------------------------------- patches
def _attach(m, method, budgeted=False, rank=2):
    return E.attach(m, [0, 1], method, rank, "ternary", budgeted=budgeted)


def _rand_codes(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for pm in E.fsa_modules(m):
            r = torch.randint(-1, 2, pm.M.shape, generator=g).float()
            pm.M.copy_(r * (torch.rand(pm.M.shape, generator=g) < 0.3))


@pytest.mark.parametrize("budgeted", [False, True])
def test_fsa_patch_roundtrip(tmp_path, budgeted):
    m1 = tiny()
    _attach(m1, "fsa_free", budgeted)
    _rand_codes(m1)
    if budgeted:
        for pm in E.fsa_modules(m1):
            pm.support.copy_(pm.mask & (torch.rand(pm.mask.shape) < 0.5))
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y1 = m1(x)
    n = E.save_patch(m1, tmp_path / "p.pt", {"k": 1})
    assert n > 0
    st = torch.load(tmp_path / "p.pt", weights_only=False)
    assert st["format"] == "e2e-fsa-patch-v1" and st["meta"] == {"k": 1}
    nnz = sum(int(e["idx"].numel()) for e in st["entries"].values())
    assert nnz == E.patch_stats(E.fsa_modules(m1))[0]
    for e in st["entries"].values():
        assert e["val"].dtype == torch.int8 and set(e["val"].tolist()) <= {-1, 1}
    m2 = tiny()
    _attach(m2, "fsa_free", budgeted)
    if budgeted:
        for pm in E.fsa_modules(m2):
            pm.support.zero_()
    E.load_patch(m2, tmp_path / "p.pt")
    assert torch.equal(m2(x), y1)
    for a, b in zip(E.fsa_modules(m1), E.fsa_modules(m2)):
        assert torch.equal(a.codes(), b.codes())
    E.set_enabled(False)
    try:
        assert torch.equal(m2(x), tiny()(x))                  # revoke: base exactly
    finally:
        E.set_enabled(True)


def test_lora_patch_roundtrip(tmp_path):
    torch.manual_seed(0)
    m1 = tiny()
    _attach(m1, "lora")
    with torch.no_grad():
        for pm in E.lora_modules(m1):
            pm.B.normal_()
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y1 = m1(x)
    E.save_patch(m1, tmp_path / "l.pt")
    torch.manual_seed(1)
    m2 = tiny()
    _attach(m2, "lora")
    E.load_patch(m2, tmp_path / "l.pt")
    assert torch.equal(m2(x), y1)


def test_lora_init_follows_seed():
    m1, m2, m3 = tiny(), tiny(), tiny()               # the model build itself reseeds; seed right before attach
    torch.manual_seed(3); _attach(m1, "lora")
    torch.manual_seed(3); _attach(m2, "lora")
    torch.manual_seed(4); _attach(m3, "lora")
    A1 = [p.A for p in E.lora_modules(m1)]
    assert all(torch.equal(a, b) for a, b in zip(A1, [p.A for p in E.lora_modules(m2)]))
    assert not all(torch.equal(a, b) for a, b in zip(A1, [p.A for p in E.lora_modules(m3)]))


def test_storage_accounting():
    s = E.fsa_storage(nnz=1000, n_slots=1 << 20, fmt="ternary")
    assert s["pos_bits"] == 20
    assert s["bytes_sparse"] == 1000 * 21 / 8
    assert math.isclose(s["bytes_dense"], (1 << 20) * math.log2(3) / 8)
    assert s["bytes"] == s["bytes_sparse"]
    m = tiny(); _attach(m, "lora", rank=2)
    ls = E.lora_storage(E.lora_modules(m))
    n = sum(p.A.numel() + p.B.numel() for p in E.lora_modules(m))
    vec = sum(p.A.shape[0] + p.B.shape[1] for p in E.lora_modules(m))
    assert ls["params"] == n and ls["fp32"] == 4 * n and ls["bf16"] == 2 * n
    assert ls["int8"] == n + 2 * vec and ls["int4"] == n / 2 + 2 * vec


def test_lora_fake_quant():
    torch.manual_seed(0)
    m = tiny(); _attach(m, "lora", rank=3)
    mods = E.lora_modules(m)
    with torch.no_grad():
        for p in mods:
            p.B.normal_()
    orig = [(p.A.clone(), p.B.clone()) for p in mods]
    errs = {}
    for qf, qmax in (("int8", 127), ("int4", 7)):
        saved = E.quantize_lora_(mods, qf)
        errs[qf] = sum(float((p.A.detach() - a).abs().sum() + (p.B.detach() - b).abs().sum()) for p, (a, b) in zip(mods, orig))
        for p in mods:
            for t, dim in ((p.A, 1), (p.B, 0)):
                s = t.abs().amax(dim, keepdim=True) / qmax
                lv = torch.round(t / s.clamp_min(1e-30))
                assert float((lv - t / s.clamp_min(1e-30)).abs().max()) < 1e-3   # on the grid
                assert int(lv.abs().max()) <= qmax
        E.restore_lora_(mods, saved)
        assert all(torch.equal(p.A, a) and torch.equal(p.B, b) for p, (a, b) in zip(mods, orig))
    assert 0 < errs["int8"] < errs["int4"]
    saved = E.quantize_lora_(mods, "bf16")
    assert all(torch.equal(p.A, a.bfloat16().float()) for p, (a, _) in zip(mods, orig))
    E.restore_lora_(mods, saved)


# ---------------------------------------------------------------------------------------------- Bop variants
def _bop_mods(m):
    mods = E.fsa_modules(m)
    for pm in mods:
        pm.bop_ema = torch.zeros_like(pm.M)
    return mods


def test_energy_tau_scale_blocks_births_in_sensitive_rows():
    m = tiny(); _attach(m, "fsa_free")
    mods = _bop_mods(m)
    for pm in mods:
        pm.col_energy = torch.ones(pm.M.shape[1])
    pm0 = mods[0]
    with torch.no_grad():
        pm0.alpha.fill_(1.0)
        pm0.alpha[0] = 10.0                                    # row 0: 100x the birth energy of the others
    stats = E.set_energy_tau_scale(mods, power=0.5, lo=0.25, hi=4.0)
    assert stats["f_hi_frac"] > 0
    assert float(pm0.tau_scale[0]) == 4.0 and float(pm0.tau_scale[1]) <= 1.0
    for pm in mods:
        pm.M.grad = None
    pm0.M.grad = torch.full_like(pm0.M, 2.0) * pm0.mask       # |ema| = gamma * 2 = 1.0 after one step
    nb, nd = E.bop_step([pm0], gamma=0.5, tau=0.5, budgeted=False)
    assert nb > 0 and nd == 0
    born = pm0.M != 0
    assert not bool(born[0].any())                             # tau_birth = 0.5 * 4 = 2 > 1: no birth in row 0
    assert bool(born[1:][pm0.mask[1:]].all())                 # tau_birth <= 0.5: every free slot born
    assert bool((pm0.M[born] == -1).all())


def test_repair_allows_deaths_only():
    m = tiny(); _attach(m, "fsa_free")
    mods = _bop_mods(m)
    pm = mods[0]
    with torch.no_grad():
        pm.M.copy_(pm.mask.float() * (torch.arange(pm.M.shape[1]) % 2 == 0).float())   # +1 on half the slots
    g = torch.ones_like(pm.M) * pm.mask                        # pushes +1 slots to die and empty slots to be born -1
    pm.M.grad = g
    before = pm.M.clone()
    nb, nd = E.bop_step([pm], gamma=1.0, tau=0.5, budgeted=False, allow_birth=False)
    assert nb == 0 and nd > 0
    assert bool(((before == 0) <= (pm.M == 0)).all())           # nothing new appeared
    pm.M.grad = g
    nb2, _ = E.bop_step([pm], gamma=1.0, tau=0.5, budgeted=False, allow_birth=True)
    assert nb2 > 0


def test_rehearsal_ce_reaches_patch():
    torch.manual_seed(0)
    m = tiny(); params, _, _ = _attach(m, "lora")
    w = torch.tensor([[BOS_T] + list(range(40, 60)), [BOS_T] + list(range(60, 80))])
    loss = E.general_ce(m, w)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)


def test_local_penalty_uses_frozen_base_denominator():
    m = tiny(); _attach(m, "fsa_free")
    _rand_codes(m)
    pen = E.LocalOutputPenalty(m)
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    pen.start(); m(x); v1 = pen.stop()
    with torch.no_grad():
        for pm in E.fsa_modules(m):
            pm.M.mul_(2)                                       # ternary codes saturate: delta unchanged
    pen.start(); m(x); v2 = pen.stop()
    assert v1 is not None and float(v1) > 0 and torch.allclose(v1, v2)


# ---------------------------------------------------------------------------------------------- review round 1
def test_score_rejects_answer_extensions_and_detours():
    assert not score(" X-y.", "X")["match"]
    assert not score(" X's dog.", "X")["match"]
    assert not score(" 41.5 years.", "41")["match"]
    r = score(" Quillon, her dog.", "Quillon", k=12)
    assert r["match"] and not r["clean"]
    assert score(" Quillon.)", "Quillon")["clean"]
    assert score(" Quillon  \n", "Quillon", k=3)["clean"]
    assert score(" Quillon!", "Quillon")["strict"]


def test_score_eos_window_boundaries():
    # answer "X" = 2 char tokens (" X"); EOS as the 1st, 2nd, 3rd token after the answer with k=2
    def r(after):
        ids = [ord(" "), ord("X")] + [ord(c) for c in after] + [EOS_T]
        return E.score_generation(TOK, ids, "X", 2, (), eos=(EOS_T,))
    assert r("")["clean"] and r("")["strict"]
    assert r(" ")["clean"] and not r(" ")["strict"]
    assert not r("  ")["clean"]                      # EOS is the 3rd token after the answer
    assert not r(" y")["match"] or not r(" y")["clean"]


def test_disjoint_values_fail_loudly_when_exhausted():
    with pytest.raises(ValueError):
        E.make_world(0, n_people=8, per_person=120, shared=120, values="disjoint")


def test_load_patch_rejects_mismatched_quant_and_scale(tmp_path):
    m1 = tiny(); E.attach(m1, [0, 1], "fsa_free", 0, "fp4")
    for pm in E.fsa_modules(m1):
        pm.fp4_step = 0.5
        with torch.no_grad():
            pm.M.copy_(pm.mask.float() * 1.0)
    E.save_patch(m1, tmp_path / "f.pt")
    m2 = tiny(); E.attach(m2, [0, 1], "fsa_free", 0, "fp4")
    with pytest.raises(AssertionError):
        E.load_patch(m2, tmp_path / "f.pt")            # default step 0.25
    for pm in E.fsa_modules(m2):
        pm.fp4_step = 0.5
    E.load_patch(m2, tmp_path / "f.pt")
    m3 = tiny(); _attach(m3, "fsa_free"); _rand_codes(m3)
    E.save_patch(m3, tmp_path / "r.pt")
    m4 = tiny(); _attach(m4, "fsa_free")
    for pm in E.fsa_modules(m4):
        pm.enable_block_scale(8, 1.0)
    with pytest.raises(AssertionError):
        E.load_patch(m4, tmp_path / "r.pt")


def test_lazy_netcost_refresh_matches_list():
    def setup():
        m = tiny(); _attach(m, "fsa_free", budgeted=True)
        g = torch.Generator().manual_seed(0)
        for pm in E.fsa_modules(m):
            pm.score.copy_(torch.randn(pm.M.shape, generator=g).bfloat16())
            pm.col_energy = torch.rand(pm.M.shape[1], generator=g)
        return E.fsa_modules(m)
    a, b = setup(), setup()
    E.refresh_support(a, 50, E.support_scores(a, "netcost", 0.1, 1.0))
    E.refresh_support(b, 50, lambda i, m: E.module_score(m, "netcost", 0.1, 1.0))
    assert all(torch.equal(x.support, y.support) for x, y in zip(a, b))
    assert sum(int(x.support.sum()) for x in a) == 50


def test_ngram_overlap():
    t = list(range(1000))
    ov = E.ngram_overlap(t[:500], {"same": t[100:400], "other": list(range(2000, 2600))}, n=32, stride=16)
    assert ov["same"] == 1.0 and ov["other"] == 0.0


# ---------------------------------------------------------------------------------------------- churn / grace / death tau
def _one_mod(churn=True):
    m = tiny(); _attach(m, "fsa_free")
    pm = _bop_mods(m)[0]
    if churn:
        E.init_churn(pm)
    return pm


def _step(pm, sign, **kw):
    """Feed a gradient of the given sign on every slot (gamma=1 -> ema = grad)."""
    pm.M.grad = torch.full_like(pm.M, float(sign)) * pm.mask
    return E.bop_step([pm], gamma=1.0, tau=0.5, budgeted=False, **kw)


def test_churn_counts_flip_flop_and_young_deaths():
    pm = _one_mod()
    tot = E.new_churn_totals()
    nslot = int(pm.mask.sum())
    nb, nd = _step(pm, -1.0, churn=tot)              # birth +1 everywhere
    assert nb == nslot and nd == 0
    nb, nd = _step(pm, +1.0, churn=tot)              # same sign as value -> death at age 1
    assert nd == nslot
    nb, nd = _step(pm, -1.0, churn=tot)              # reborn: flip-flop
    assert tot == dict(births=2 * nslot, deaths=nslot, deaths_young=nslot, deaths_blocked_by_grace=0,
                       rebirths=nslot, evictions=0, evictions_young=0, sign_reversals=0)
    s = E.churn_summary([pm])
    assert s["slots_ever_born"] == nslot and s["slots_born_2plus"] == nslot and s["slots_born_3plus"] == 0
    assert s["alive"] == nslot and s["alive_first_birth"] == 0 and s["max_births_one_slot"] == 2


def test_maturity_grace_blocks_early_deaths():
    pm = _one_mod()
    tot = E.new_churn_totals()
    nslot = int(pm.mask.sum())
    _step(pm, -1.0, churn=tot, grace=3)               # born, age -> 1
    nb, nd = _step(pm, +1.0, churn=tot, grace=3)      # age 1 < 3: blocked
    assert nd == 0 and tot["deaths_blocked_by_grace"] == nslot
    nb, nd = _step(pm, +1.0, churn=tot, grace=3)      # age 2 < 3: blocked
    assert nd == 0
    nb, nd = _step(pm, +1.0, churn=tot, grace=3)      # age 3: allowed
    assert nd == nslot and tot["deaths_young"] == nslot


def test_death_tau_multiplier():
    pm = _one_mod()
    _step(pm, -1.0)
    pm.M.grad = torch.full_like(pm.M, 0.6) * pm.mask  # |ema| 0.6 > tau 0.5 but < 1.5 * 0.5
    nb, nd = E.bop_step([pm], gamma=1.0, tau=0.5, budgeted=False, death_mult=1.5)
    assert nd == 0
    pm.M.grad = torch.full_like(pm.M, 0.8) * pm.mask
    nb, nd = E.bop_step([pm], gamma=1.0, tau=0.5, budgeted=False, death_mult=1.5)
    assert nd == int(pm.mask.sum())


def test_ages_saturate_and_bop_without_churn_unchanged():
    pm = _one_mod()
    _step(pm, -1.0)
    pm.M.grad = None
    for _ in range(200):
        E.bop_step([pm], gamma=0.5, tau=0.5, budgeted=False)
    assert int(pm.age.max()) == E.AGE_CAP
    a, b = _one_mod(churn=False), _one_mod(churn=True)
    for sgn in (-1.0, 1.0, -1.0):
        _step(a, sgn); _step(b, sgn)
        assert torch.equal(a.M, b.M)


# ---------------------------------------------------------------------------------------------- prune_flips helpers
import prune_flips as PF  # noqa: E402


def test_lorenz():
    u = torch.tensor([50.0, 30.0, 10.0, 5.0, 5.0, 0.0, -3.0, -7.0, 0.0, 0.0])
    lz = PF.lorenz(torch.sort(u, descending=True).values)
    assert lz["0.5"] == 0.1 and lz["0.9"] == 0.3 and lz["0.99"] == 0.5


def test_flip_scores_revert_restore_and_first_order():
    torch.manual_seed(0)
    m = tiny(); _attach(m, "fsa_free")
    _rand_codes(m, seed=1)
    mods = E.fsa_modules(m)
    for pm in mods:
        pm.M.requires_grad_(True)
        pm.col_energy = torch.rand(pm.M.shape[1])
    x = torch.tensor([[BOS_T] + list(range(40, 80))])

    def loss():
        lg = m(x)[0, :-1]
        return torch.nn.functional.cross_entropy(lg, x[0, 1:])
    for pm in mods:
        pm.M.grad = None
    L0 = loss(); L0.backward()
    mi, fi = PF.nonzero_table(mods)
    vals = torch.cat([pm.M.detach().view(-1)[fi[mi == i]] for i, pm in enumerate(mods)])
    assert len(fi) == E.patch_stats(mods)[0]
    u, c = PF.flip_scores(mods, mi, fi, 2.0)
    pm0 = mods[int(mi[0])]
    q0 = pm0.codes().flatten()[fi[0]]
    assert math.isclose(float(u[0]), float(-q0 * pm0.M.grad.flatten()[fi[0]]), rel_tol=1e-5)
    col = int(fi[0] % pm0.M.shape[1]); row = int(fi[0] // pm0.M.shape[1])
    assert math.isclose(float(c[0]), float(pm0.alpha[row, 0] ** 2 * pm0.col_energy[col] / 2.0), rel_tol=1e-5)
    # first order: reverting the 10% highest-utility flips must hurt more than reverting the 10% lowest
    order = torch.argsort(u)
    k = max(1, len(u) // 10)
    with torch.no_grad():
        PF.revert(mods, mi, fi, order[:k]); Lb = float(loss()); PF.restore(mods, mi, fi, vals, order[:k])
        PF.revert(mods, mi, fi, order[-k:]); Lt = float(loss()); PF.restore(mods, mi, fi, vals, order[-k:])
        assert Lt - float(L0) > Lb - float(L0)
        assert math.isclose(float(loss()), float(L0), rel_tol=1e-6)        # restore is exact


def test_evictions_counted_and_totals_without_counters():
    m = tiny(); _attach(m, "fsa_free", budgeted=True)
    mods = _bop_mods(m)
    for pm in mods:
        E.init_churn(pm)
        pm.support.copy_(pm.mask)
    tot = E.new_churn_totals()
    for pm in mods:
        pm.M.grad = -torch.ones_like(pm.M) * pm.mask
    nb, _ = E.bop_step(mods, 1.0, 0.5, True, churn=tot)
    assert tot["births"] == nb > 0
    for pm in mods:
        pm.score = torch.zeros_like(pm.M, dtype=torch.bfloat16)
    E.refresh_support(mods, 0, None, None, churn=tot)       # evict everything
    assert tot["evictions"] == nb and tot["evictions_young"] == nb
    assert sum(int((pm.M != 0).sum()) for pm in mods) == 0
    pm = _one_mod(churn=False)
    tot2 = E.new_churn_totals()
    nb, _ = _step(pm, -1.0, churn=tot2)
    assert tot2["births"] == nb > 0 and tot2["rebirths"] == 0


def test_group_ids_expert_and_linear():
    names = ["moes.0.experts.0.up_proj", "moes.0.experts.0.down_proj", "moes.0.experts.1.up_proj",
             "moes.1.experts.0.down_proj"]
    mi = torch.tensor([0, 0, 1, 2, 3, 3])
    g, u = PF.group_ids(names, mi, "expert")
    assert u == ["moes.0.experts.0", "moes.0.experts.1", "moes.1.experts.0"]
    assert g.tolist() == [0, 0, 0, 1, 2, 2]
    g, u = PF.group_ids(names, mi, "linear")
    assert len(u) == 4 and g.tolist() == [u.index(names[i]) for i in mi.tolist()]


def test_spearman_ties_and_undefined():
    assert PF.spearman([1, 1, 1], [1, 2, 3]) is None
    assert PF.spearman([1], [2]) is None
    assert math.isclose(PF.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert PF.avg_ranks([5, 1, 5, 3]).tolist() == [2.5, 0.0, 2.5, 1.0]


def test_attach_shared_expert_options():
    m = tiny()
    g = torch.Generator().manual_seed(5)
    for moe in m.moes:
        moe.shared_experts = nn.ModuleList([Expert(16, 32, g)])
    p_r, n_r, _ = E.attach(tiny(), [0, 1], "fsa_free", 0)
    _, n_both, _ = E.attach(m, [0, 1], "fsa_free", 0, shared=True)
    names = [n for n, _ in E.patched_linears(m)]
    assert any("shared_experts" in n for n in names) and n_both > n_r
    m2 = tiny()
    g2 = torch.Generator().manual_seed(5)
    for moe in m2.moes:
        moe.shared_experts = nn.ModuleList([Expert(16, 32, g2)])
    _, n_s, _ = E.attach(m2, [0, 1], "fsa_free", 0, shared=True, routed=False)
    assert all("shared_experts" in n for n, _ in E.patched_linears(m2)) and n_s == n_both - n_r


def test_gen_eval_all_people_correct_metric():
    m = tiny()
    facts = E.make_world(0, 2, 10, 10)
    items = E.build_items(TOK, facts, E.TEST_T, BOS_T)
    assert all("attr" in it and "ti" in it for it in items)
    r = E.gen_eval(m, TOK, items, "cpu", 0, 2, 8, E.conflict_map(facts))
    assert r["n_groups"] == 10 * 2 and 0 <= r["all_people_correct"] <= 1


# ---------------------------------------------------------------------------------------------- sparse slot state
def _pair(budgeted=False):
    md, ms = tiny(), tiny()
    E.attach(md, [0, 1], "fsa_free", 0, "ternary", budgeted=budgeted)
    E.attach(ms, [0, 1], "fsa_free", 0, "ternary", budgeted=budgeted, sparse=True)
    return md, ms, E.fsa_modules(md), E.fsa_modules(ms)


def test_sparse_state_matches_dense_forward_backward_and_bop(tmp_path):
    md, ms, dm, sm = _pair()
    assert all(s.sparse and s.M.numel() == int(d.mask.sum()) for d, s in zip(dm, sm))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for d, s in zip(dm, sm):
            r = torch.randint(-1, 2, d.M.shape, generator=g).float() * d.mask
            d.M.copy_(r)
            s.M.copy_(r.flatten()[s.idx.long()])
    x = torch.tensor([[BOS_T] + list(range(40, 80))])
    for mm in (md, ms):
        for pm in E.fsa_modules(mm):
            pm.M.requires_grad_(True); pm.M.grad = None
        loss = torch.nn.functional.cross_entropy(mm(x)[0, :-1], x[0, 1:])
        loss.backward()
    assert torch.allclose(md(x), ms(x))
    for d, s in zip(dm, sm):
        assert torch.equal(d.codes(), s.codes())
        assert torch.allclose(d.M.grad.flatten()[s.idx.long()], s.M.grad, atol=1e-6)
        d.bop_ema = torch.zeros_like(d.M); s.bop_ema = torch.zeros_like(s.M)
    tau = 1e-3
    nb_d = E.bop_step(dm, 0.5, tau, False); nb_s = E.bop_step(sm, 0.5, tau, False)
    assert nb_d == nb_s
    for d, s in zip(dm, sm):
        assert torch.equal(d.codes(), s.codes())
    # patches are interchangeable between dense and sparse state
    E.save_patch(ms, tmp_path / "s.pt")
    E.save_patch(md, tmp_path / "d.pt")
    a, b = torch.load(tmp_path / "s.pt", weights_only=False), torch.load(tmp_path / "d.pt", weights_only=False)
    for n in a["entries"]:
        assert torch.equal(a["entries"][n]["idx"], b["entries"][n]["idx"])
        assert torch.equal(a["entries"][n]["val"], b["entries"][n]["val"])
    m3 = tiny(); E.attach(m3, [0, 1], "fsa_free", 0, "ternary", sparse=True)
    E.load_patch(m3, tmp_path / "d.pt")
    assert torch.allclose(m3(x), md(x))


def test_sparse_state_budget_refresh_and_netcost():
    md, ms, dm, sm = _pair(budgeted=True)
    g = torch.Generator().manual_seed(1)
    for d, s in zip(dm, sm):
        sc = torch.randn(d.M.shape, generator=g).bfloat16() * d.mask
        d.score.copy_(sc); s.score.copy_(sc.flatten()[s.idx.long()])
        ce = torch.rand(d.M.shape[1], generator=g)
        d.col_energy = ce; s.col_energy = ce.clone()
    E.refresh_support(dm, 40, lambda i, m: E.module_score(m, "netcost", 0.3, 2.0))
    E.refresh_support(sm, 40, lambda i, m: E.module_score(m, "netcost", 0.3, 2.0))
    for d, s in zip(dm, sm):
        assert torch.equal(d.support.flatten()[s.idx.long()], s.support)
        assert torch.allclose(E.slot_cost(d, 2.0).expand_as(d.M).flatten()[s.idx.long()], E.slot_cost(s, 2.0))


def test_sparse_budgeted_bop_with_churn_and_eviction():
    md, ms, dm, sm = _pair(budgeted=True)
    for mods in (dm, sm):
        for pm in mods:
            pm.bop_ema = torch.zeros_like(pm.M)
            E.init_churn(pm)
            pm.support.copy_(pm.mask if pm.support.dim() == pm.mask.dim() and not pm.sparse
                             else torch.ones_like(pm.support))
            pm.M.grad = -torch.ones_like(pm.M) * pm.mask
    td, ts = E.new_churn_totals(), E.new_churn_totals()
    assert E.bop_step(dm, 1.0, 0.5, True, churn=td) == E.bop_step(sm, 1.0, 0.5, True, churn=ts)
    for mods in (dm, sm):
        for pm in mods:
            pm.score.zero_()
    E.refresh_support(dm, 0, None, None, churn=td)
    E.refresh_support(sm, 0, None, None, churn=ts)
    assert td == ts and td["evictions"] > 0
    assert E.churn_summary(sm) == E.churn_summary(dm)


def test_sparse_sgd_rms_denominator_uses_full_shapes():
    md, ms, dm, sm = _pair()
    import inspect
    src = inspect.getsource(E.main)
    assert "math.prod(pm_.wshape)" in src
    assert all(math.prod(s.wshape) == d.M.numel() for d, s in zip(dm, sm))


# ---------------------------------------------------------------------------------------------- LoRA fold
def test_fold_ste_forward_is_its_exported_patch(tmp_path):
    torch.manual_seed(0)
    m = tiny(); E.attach(m, [0, 1], "lora_fold", 2, fold_mode="ste")
    fm = E.fold_modules(m)
    assert fm and not E.lora_modules(m)
    with torch.no_grad():
        for p in fm:
            p.B.normal_(0, 3.0)
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y = m(x)
    for p in fm:                                           # forward delta == codes * alpha, codes legal & ternary
        d = p.delta_w().detach()
        assert torch.allclose(d, p.codes() * p.alpha, atol=1e-6)
        assert bool(((p.codes() != 0) <= p.mask).all())
    assert sum(int((p.codes() != 0).sum()) for p in fm) > 0
    E.save_patch(m, tmp_path / "f.pt")
    m2 = tiny(); E.attach(m2, [0, 1], "fsa_free", 0)
    E.load_patch(m2, tmp_path / "f.pt")
    assert torch.allclose(m2(x), y, atol=1e-5)
    # STE: gradients reach the adapter
    for p in fm:
        p.A.requires_grad_(True); p.B.requires_grad_(True)
    torch.nn.functional.cross_entropy(m(x)[0, :-1], x[0, 1:]).backward()
    assert any(p.B.grad is not None and float(p.B.grad.abs().sum()) > 0 for p in fm)


def test_fold_ef_accumulates_resets_and_tracks_loss():
    torch.manual_seed(0)
    m = tiny(); E.attach(m, [0, 1], "lora_fold", 2, fold_mode="ef")
    fm = E.fold_modules(m)
    with torch.no_grad():
        for p in fm:
            p.B.normal_(0, 1.0)
    d0 = [(p.B @ p.A).clone() for p in fm]
    for p in fm:
        p.fold()
    for p, d in zip(fm, d0):
        assert torch.equal(p.B, torch.zeros_like(p.B))
        assert torch.allclose(p.S, d * p.mask / p.alpha.clamp_min(1e-12))
        assert torch.allclose(p.delta_w(), E.FoldLoRAParam._q(p.S) * p.alpha, atol=1e-6)   # pure slot patch after a fold
        assert torch.equal(p.codes(), E.FoldLoRAParam._q(p.S) * p.mask)
    kept = sum(p.fold_kept for p in fm); lost = sum(p.fold_lost for p in fm)
    assert kept > 0 and lost > 0
    with torch.no_grad():                                  # residual carries: two half folds == one full fold
        for p in fm:
            p.S.zero_(); p.B.normal_(0, 1.0)
    full = [(p.B @ p.A * p.mask / p.alpha.clamp_min(1e-12)).clone() for p in fm]
    with torch.no_grad():
        for p in fm:
            B_, A_ = p.B.clone(), p.A.clone()
            p.B.copy_(B_ / 2); p.fold(); p.B.copy_(B_ / 2); p.A.copy_(A_); p.fold()
    for p, f in zip(fm, full):
        assert torch.allclose(p.S, f, atol=1e-5)


def test_last_token_experts_and_routing_overlap():
    m = tiny()
    ids = [BOS_T] + list(range(40, 60))
    sets = E.last_token_experts(m, ids, "cpu")
    # the tiny MoE routes each token to exactly one expert per layer
    assert all(len(v) == 1 for v in sets.values()) and set(sets) <= {0, 1}
    facts = E.make_world(0, 2, 10, 10)
    r = E.routing_overlap(m, TOK, facts, "cpu", BOS_T, max_attrs=5)
    assert r["n_cross_pairs"] == 5 and len(r["cross_person_same_attr"]) == 2
    assert all(0 <= v <= 1 for v in r["cross_person_same_attr"] + r["same_person_other_attr"])


def test_sign_reversal_counted():
    pm = _one_mod()
    tot = E.new_churn_totals()
    _step(pm, -1.0, churn=tot)                         # born +1
    _step(pm, +1.0, churn=tot)                         # dies
    nb, _ = _step(pm, +1.0, churn=tot)                 # reborn -1: sign reversal
    assert nb > 0 and tot["sign_reversals"] == nb


import lora_project as LP  # noqa: E402


def test_gptq_quantize_identity_and_error_compensation():
    g = torch.Generator().manual_seed(0)
    o, i = 12, 32
    alpha = torch.rand(o, 1, generator=g) + 0.5
    free = torch.rand(o, i, generator=g) < 0.5
    T = torch.randn(o, i, generator=g) * 0.6
    eye = torch.eye(i)
    c_id = LP.gptq_quantize(T, alpha, free, torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(eye)),
                                                                  upper=True))
    assert torch.equal(c_id, torch.clamp(torch.round(T / alpha), -1, 1) * free)
    X = torch.randn(400, i, generator=g) @ (torch.randn(i, i, generator=g) * 0.3 + torch.eye(i))  # correlated
    H = X.T @ X / X.shape[0]
    H += 0.01 * H.diagonal().mean() * eye
    Hu = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    c = LP.gptq_quantize(T, alpha, free, Hu)
    assert set(c.unique().tolist()) <= {-1.0, 0.0, 1.0} and bool(((c != 0) <= free).all())
    err = lambda q: float(((T - q * alpha) @ X.T).pow(2).sum())   # noqa: E731
    assert err(c) < err(c_id)


def test_mdl_bits_and_entropy_storage():
    s = E.fsa_storage(nnz=1000, n_slots=1 << 20, fmt="ternary")
    p = 1000 / (1 << 20)
    h2 = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    assert math.isclose(s["bytes_entropy"], ((1 << 20) * h2 + 1000) / 8)
    base = dict(answer_bits=500.0); pat = dict(answer_bits=100.0, answer_tokens=50)
    big = dict(dnll=0.001, dnll_se=0.002)
    r = E.mdl_bits(base, pat, big, dict(fsa=s), ref_tokens=1000)
    assert math.isclose(r["fact_bits_saved"], 400.0)
    assert math.isclose(r["general_cost_bits"], 1.0 / math.log(2))
    assert math.isclose(r["net_bits"], 400.0 - 1.0 / math.log(2) - 8 * s["bytes_entropy"])
    m = tiny()
    items = E.build_items(TOK, E.make_world(0)[:3], E.TEST_T, BOS_T)
    t = E.tf_eval(m, items, "cpu", 1)
    assert t["answer_tokens"] == sum(len(it["a"]) for it in items) and t["answer_bits"] > 0


def test_local_penalty_from_output_matches_delta_form():
    for sparse in (False, True):
        m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "ternary", sparse=sparse)
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for pm in E.fsa_modules(m):
                pm.M.copy_(torch.randint(-1, 2, pm.M.shape, generator=g).float() * (pm.mask if not sparse else 1))
        x = torch.tensor([[BOS_T] + list(range(40, 70))])
        p1 = E.LocalOutputPenalty(m); p1.start(); m(x); v1 = p1.stop()
        for h in p1.h:
            h.remove()
        p2 = E.LocalOutputPenalty(m, from_output=True); p2.start(); m(x); v2 = p2.stop()
        assert torch.allclose(v1, v2, rtol=1e-4)
        for h in p2.h:
            h.remove()
        grads = []
        for fo in (False, True):                           # gradients through sequential patched linears agree
            pen = E.LocalOutputPenalty(m, from_output=fo)
            for pm in E.fsa_modules(m):
                pm.M.requires_grad_(True); pm.M.grad = None
            pen.start(); m(x); pen.stop().backward()
            grads.append([pm.M.grad.clone() if pm.M.grad is not None else None for pm in E.fsa_modules(m)])
            for h in pen.h:
                h.remove()
        for g1, g2 in zip(*grads):
            assert (g1 is None) == (g2 is None)
            if g1 is not None:
                assert torch.allclose(g1, g2, rtol=1e-3, atol=1e-7)


def test_conflict_pool_scores_penalize_cross_person_harm():
    for sparse in (False, True):
        m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "ternary", budgeted=True, sparse=sparse)
        mods = E.fsa_modules(m)
        g0 = [torch.zeros_like(pm.M) for pm in mods]
        g1 = [torch.zeros_like(pm.M) for pm in mods]
        pm = mods[0]
        free = pm.mask.flatten().nonzero().squeeze(1) if not sparse else torch.arange(pm.M.numel())
        a_, b_ = int(free[0]), int(free[1])
        g0[0].view(-1)[a_] = -1.0; g1[0].view(-1)[a_] = +1.0     # helps person 0, equally hurts person 1
        g0[0].view(-1)[b_] = -1.0                                # helps person 0, neutral for person 1
        sc = E.conflict_pool_scores(mods, [g0, g1], beta=1.0)(0, pm).flatten()
        assert float(sc[b_]) > 0 and float(sc[a_]) <= 0 + 1e-9
        E.refresh_support(mods, 1, E.conflict_pool_scores(mods, [g0, g1]), None)
        assert bool(pm.support.flatten()[b_]) and not bool(pm.support.flatten()[a_])
        gen = torch.Generator().manual_seed(0)
        n_before = sum(int(p.support.sum()) for p in mods)
        added = E.add_random_support(mods, 10, gen)
        assert sum(int(p.support.sum()) for p in mods) == n_before + added and added > 0


def test_add_random_support_exact_counts():
    m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "ternary", budgeted=True)
    mods = E.fsa_modules(m)
    for n in (1, 2, 7, 33):
        for pm in mods:
            pm.support.zero_()
        got = E.add_random_support(mods, n, torch.Generator().manual_seed(n))
        assert got == n == sum(int(pm.support.sum()) for pm in mods)
        assert all(bool((pm.support <= pm.mask).all()) for pm in mods)


def test_occupied_sign_flips_are_legal_and_roundtrip(tmp_path):
    m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "ternary", sparse=True, occupied=True)
    mods = E.fsa_modules(m)
    pm = mods[0]
    lin = [l for _, l in E.patched_linears(m)][0]
    W = lin.parametrizations.weight.original
    assert int(pm.occ.sum()) == int((W != 0).sum()) and pm.M.numel() == int(((W != 0) | (W == 0) &
           (W.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(W.shape)).sum())
    pm.bop_ema = torch.zeros_like(pm.M)
    pm.M.grad = torch.where(pm.occ, -torch.ones_like(pm.M), torch.zeros_like(pm.M))   # wants +1 on flips
    nb, _ = E.bop_step([pm], 1.0, 0.1, False)
    assert nb == int(pm.occ.sum())
    Wp = lin.weight.detach()
    occ_d = W != 0
    assert torch.allclose(Wp[occ_d], -W[occ_d])                       # flipped exactly
    assert torch.equal(Wp[~occ_d], W[~occ_d])
    with torch.no_grad():
        pm.bop_ema.zero_(); pm.M.zero_()
    pm.M.grad = torch.where(pm.occ, torch.ones_like(pm.M), torch.zeros_like(pm.M))    # wants -1: illegal
    nb, _ = E.bop_step([pm], 1.0, 0.1, False)
    assert nb == 0
    with torch.no_grad():
        pm.M[pm.occ.nonzero().squeeze(1)[:5]] = 1.0
        pm.M[(~pm.occ).nonzero().squeeze(1)[:3]] = -1.0
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y = m(x)
    E.save_patch(m, tmp_path / "o.pt")
    st = torch.load(tmp_path / "o.pt", weights_only=False)
    e0 = next(iter(st["entries"].values()))
    assert e0["flip_idx"].numel() == 5 and e0["idx"].numel() == 3
    m2 = tiny(); E.attach(m2, [0, 1], "fsa_free", 0, "ternary", sparse=True, occupied=True)
    E.load_patch(m2, tmp_path / "o.pt")
    assert torch.allclose(m2(x), y)


def test_flip_slots_quantize_and_score_legally():
    m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "ternary", budgeted=True, sparse=True, occupied=True)
    mods = E.fsa_modules(m)
    pm = mods[0]
    with torch.no_grad():
        pm.M.fill_(-1.0)
    q = pm._quant(pm.M)
    assert bool((q[pm.occ] == 0).all()) and bool((q[~pm.occ] == -1).all())   # no -1 (3W) on flip slots
    g = torch.where(pm.occ, torch.full_like(pm.M, 5.0), torch.full_like(pm.M, -1.0))
    sc = E.conflict_pool_scores(mods, [[g] + [None] * (len(mods) - 1)])(0, pm)
    assert bool((sc[pm.occ] <= 0).all()) and bool((sc[~pm.occ] > 0).all())     # illegal flips never selected
    pm.score = g.bfloat16(); pm.col_energy = torch.zeros(pm.wshape[1])
    with torch.no_grad():
        pm.M.zero_()
    ns = E.module_score(pm, "netcost", 0.0, 1.0)
    assert bool((ns[pm.occ] <= 0).all())


def test_fold_threshold_controls_sparsity():
    torch.manual_seed(0)
    m = tiny(); E.attach(m, [0, 1], "lora_fold", 2, fold_mode="ste")
    fm = E.fold_modules(m)
    with torch.no_grad():
        for p in fm:
            p.B.normal_(0, 3.0)
    try:
        E.FoldLoRAParam.thr = 0.5
        n1 = sum(int((p.codes() != 0).sum()) for p in fm)
        E.FoldLoRAParam.thr = 2.0
        n2 = sum(int((p.codes() != 0).sum()) for p in fm)
        for p in fm:
            assert torch.allclose(p.delta_w().detach(), p.codes() * p.alpha, atol=1e-6)
    finally:
        E.FoldLoRAParam.thr = 0.5
    assert n1 > n2 > 0


def test_slot_frac_subsamples_reserve_and_loads_into_full(tmp_path):
    m1 = tiny(); _, n_full, _ = E.attach(m1, [0, 1], "fsa_free", 0, "ternary", sparse=True)
    m2 = tiny(); _, n_part, _ = E.attach(m2, [0, 1], "fsa_free", 0, "ternary", sparse=True, slot_frac=0.3, seed=0)
    m3 = tiny(); _, n_part2, _ = E.attach(m3, [0, 1], "fsa_free", 0, "ternary", sparse=True, slot_frac=0.3, seed=0)
    assert 0 < n_part < n_full and n_part == n_part2
    with torch.no_grad():
        for pm in E.fsa_modules(m2):
            pm.M.fill_(1.0)
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y = m2(x)
    E.save_patch(m2, tmp_path / "p.pt")
    E.load_patch(m1, tmp_path / "p.pt")                    # a subset patch loads into the full reserve
    assert torch.allclose(m1(x), y)


def test_expert_checkpoint_same_loss_and_grads():
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    res = []
    for ck in (False, True):
        for method, kw in (("fsa_free", dict(sparse=True)), ("lora_fold", dict(fold_mode="ste")), ("lora", {})):
            torch.manual_seed(0)
            m = tiny(); params, _, _ = E.attach(m, [0, 1], method, 2, **kw)
            with torch.no_grad():
                for p in params:
                    if p.dim() == 1:
                        p.copy_(torch.randint(-1, 2, p.shape).float())
                    else:
                        p.normal_(0, 1.0)
            if ck:
                assert E.checkpoint_experts(m, [0, 1]) == 8
            for p in params:
                p.requires_grad_(True); p.grad = None
            pen = E.LocalOutputPenalty(m, from_output=True)
            torch.nn.functional.cross_entropy(m(x)[0, :-1], x[0, 1:]).backward()
            pen.start(); m(x); lp = pen.stop()
            lp.backward(); pen.end()
            loss = lp
            res.append((float(loss), [p.grad.clone() if p.grad is not None else None for p in params]))
    for (l0, g0), (l1, g1) in zip(res[:3], res[3:]):           # (penalty value, fact+penalty grads)
        assert math.isclose(l0, l1, rel_tol=1e-6)
        for a_, b_ in zip(g0, g1):
            assert (a_ is None) == (b_ is None)
            if a_ is not None:
                assert torch.allclose(a_, b_, rtol=1e-4, atol=1e-6)


def test_checkpointed_penalty_pass_retains_no_patched_weights():
    import torch.autograd.graph as G
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    m = tiny(); params, _, _ = E.attach(m, [0, 1], "fsa_free", 0, "ternary", sparse=True)
    for p in params:
        p.requires_grad_(True)
    E.checkpoint_experts(m, [0, 1])
    pen = E.LocalOutputPenalty(m, from_output=True)
    shapes = {tuple(l.parametrizations.weight.original.shape) for _, l in E.patched_linears(m)}
    saved = []
    with G.saved_tensors_hooks(lambda t: (saved.append(tuple(t.shape)), t)[1], lambda t: t):
        pen.start(); m(x); lp = pen.stop()
    lp.backward(); pen.end()
    assert not any(s_ in shapes for s_ in saved)           # no full weight matrix held for backward


import ls_diag as LD  # noqa: E402


def test_masked_ls_matches_per_row_ridge():
    g = torch.Generator().manual_seed(0)
    i, o = 24, 6
    Hf = torch.randn(40, i, generator=g); Hg = torch.randn(30, i, generator=g)
    T = torch.randn(40, o, generator=g)
    free = torch.rand(o, i, generator=g) < 0.4
    free[2] = False                                        # a row without free slots
    lam = 0.7
    W = LD.masked_ls(Hf, T, Hg, free, lam, chunk=4)
    G = Hf.T @ Hf + lam * Hg.T @ Hg
    eps = 1e-3 * float(G.diagonal().mean()) + 1e-12
    for r in range(o):
        F_ = free[r].nonzero().squeeze(1)
        assert bool((W[r][~free[r]] == 0).all())
        if len(F_) == 0:
            continue
        A = G[F_][:, F_] + eps * torch.eye(len(F_))
        w = torch.linalg.solve(A, (Hf.T @ T)[F_, r])
        assert torch.allclose(W[r, F_], w, atol=1e-4)


def test_any_zero_fold_mask_and_serving_stats():
    torch.manual_seed(0)
    m1 = tiny(); E.attach(m1, [0, 1], "lora_fold", 2, fold_mode="ste")
    m2 = tiny(); E.attach(m2, [0, 1], "lora_fold", 2, fold_mode="ste", fold_any_zero=True)
    for a_, b_ in zip(E.fold_modules(m1), E.fold_modules(m2)):
        assert bool((a_.mask <= b_.mask).all()) and int(b_.mask.sum()) > int(a_.mask.sum())
    for p in E.fold_modules(m2):
        with torch.no_grad():
            p.B.normal_(0, 3.0)
    s = E.serving_stats(m2)
    assert s["nnz"] > 0 and 0 < s["blocks_touched"] <= s["total_blocks"] and s["blocks_non_pair8"] > 0
    m3 = tiny(); E.attach(m3, [0, 1], "fsa_free", 0)
    _rand_codes(m3)
    assert E.serving_stats(m3)["blocks_non_pair8"] == 0       # free slots keep pair8


def _pair8_tiny_weight(seed=0, o=16, i=32):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(o, i, generator=g)
    blocks = W.reshape(-1, 4, 2)
    for b in range(blocks.shape[0]):
        k = int(torch.randint(0, 3, (1,), generator=g))          # 0, 1 or 2 active pairs
        keep = torch.randperm(4, generator=g)[:k]
        dead = torch.ones(4, dtype=torch.bool); dead[keep] = False
        blocks[b, dead] = 0
        if k and bool(torch.rand(1, generator=g) < 0.3):
            blocks[b, keep[0], 1] = 0                             # an in-pair zero
    return W


def test_pair8_forced_mask_rule_and_reproducibility():
    W = _pair8_tiny_weight()
    sel = E.pair8_selected_pairs(W)
    assert bool((sel.sum(-1) == 2).all())                         # exactly 2 selected pairs per block
    act = (W.reshape(-1, 4, 2) != 0).any(-1)
    assert bool((act <= sel).all())                               # every nonzero pair is selected
    m = E.pair8_forced_mask(W)
    assert int(m.sum()) == W.numel() // 2 and not bool((W[m] != 0).any())
    inpair_zero = (W == 0) & sel[..., None].expand(-1, -1, 2).reshape(W.shape)
    assert not bool((m & inpair_zero).any())                      # in-pair zeros are NOT writable
    # deterministic fill: blocks with < 2 active pairs take the lowest-index zero pairs
    for b in range(sel.shape[0]):
        if int(act[b].sum()) < 2:
            zeros = [p for p in range(4) if not bool(act[b, p])]
            chosen = [p for p in range(4) if bool(sel[b, p]) and not bool(act[b, p])]
            assert chosen == zeros[:2 - int(act[b].sum())]
    import build_forced_mask as BFM
    assert np.array_equal(BFM.selected_bits(W), BFM.selected_bits(W.clone()))


def test_fsa_forced_attach_and_revoke():
    m = tiny()
    k = 0
    for moe in m.moes:                                     # make the tiny base pair8
        for e in moe.experts:
            for lin in (e.up_proj, e.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(k, *lin.weight.shape)); k += 1
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y0 = m(x)
    params, n, _ = E.attach(m, [0, 1], "fsa_forced", 0)
    assert n == sum(l.parametrizations.weight.original.numel() // 2 for _, l in E.patched_linears(m))
    with torch.no_grad():
        for pm in E.fsa_modules(m):
            pm.M.copy_(torch.randint(-1, 2, pm.M.shape).float())
    assert not torch.allclose(m(x), y0)
    E.set_enabled(False)
    try:
        assert torch.equal(m(x), y0)                       # revoke: exact pair8 base
    finally:
        E.set_enabled(True)
    for _, lin in E.patched_linears(m):
        pm = lin.parametrizations.weight[0]
        W0 = lin.parametrizations.weight.original
        assert torch.equal(pm.mask, E.pair8_forced_mask(W0))
        assert bool((W0[pm.mask] == 0).all())
        assert set(pm.codes().unique().tolist()) <= {-1, 0, 1}


def test_forced_mask_verify_rejects_tampering():
    import build_forced_mask as BFM
    W = _pair8_tiny_weight(3)
    ent = {"a.experts.0.up_proj": dict(shape=list(W.shape), bits=BFM.selected_bits(W))}
    st = dict(sha256=BFM.digest(ent), entries={k: dict(v) for k, v in ent.items()})
    assert BFM.verify(st, ent)
    bad = dict(st, entries=dict(st["entries"], extra=dict(shape=[1], bits=np.zeros(1, np.uint8))))
    assert not BFM.verify(bad, ent)
    bad2 = dict(st, entries={k: dict(v, shape=[1, 2]) for k, v in st["entries"].items()})
    assert not BFM.verify(bad2, ent)
    flip = {k: dict(v, bits=v["bits"].copy()) for k, v in st["entries"].items()}
    flip["a.experts.0.up_proj"]["bits"][0] ^= 1
    assert not BFM.verify(dict(st, entries=flip), ent)
    with pytest.raises(ValueError):
        m = tiny(); E.attach(m, [0, 1], "fsa_forced", 0, sparse=True, occupied=True)


def test_masked_ls_box_constraint():
    g = torch.Generator().manual_seed(1)
    i, o = 20, 5
    Hf = torch.randn(50, i, generator=g); Hg = torch.randn(20, i, generator=g); T = torch.randn(50, o, generator=g) * 3
    free = torch.rand(o, i, generator=g) < 0.5
    W = LD.masked_ls(Hf, T, Hg, free, 0.5)
    Wbig = LD.masked_ls(Hf, T, Hg, free, 0.5, box=torch.full((o,), 1e6))
    assert torch.allclose(W, Wbig, atol=1e-4)                 # inactive bound == unconstrained
    bnd = torch.full((o,), 0.05)
    Wb = LD.masked_ls(Hf, T, Hg, free, 0.5, box=bnd, iters=500)
    assert float(Wb.abs().max()) <= 0.05 + 1e-6 and bool((Wb[~free] == 0).all())
    G = Hf.T @ Hf + 0.5 * Hg.T @ Hg
    obj = lambda w: float(0.5 * (w @ G * w).sum() - ((Hf.T @ T).T * w).sum())   # noqa: E731
    clip = torch.clamp(W, -0.05, 0.05)
    assert obj(Wb) <= obj(clip) + 1e-4                        # PGD at least as good as clipping
    nz = torch.tensor([[0.0, 2.0, -4.0], [0.0, 0.0, 0.0]])
    assert torch.allclose(LD.Wbase_alpha(nz), torch.tensor([3.0, 0.0]))


def test_patch_scale_mult_scales_delta_only():
    for sparse in (False, True):
        m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, sparse=sparse)
        mods = E.fsa_modules(m)
        with torch.no_grad():
            for pm in mods:
                pm.M.fill_(1.0)
        d1 = [pm.delta().detach().clone() for pm in mods]
        with torch.no_grad():                               # what main() does for --patch-scale-mult 0.5
            for pm in mods:
                pm.alpha.mul_(0.5)
                if sparse:
                    pm.scale.mul_(0.5)
        for pm, d in zip(mods, d1):
            assert torch.allclose(pm.delta(), 0.5 * d)


def test_forced_position_write_changes_output_linearly():
    m = tiny()
    k = 0
    for moe in m.moes:
        for e in moe.experts:
            for lin in (e.up_proj, e.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(k + 50, *lin.weight.shape)); k += 1
    E.attach(m, [0], "fsa_forced", 0)
    lin = m.moes[0].experts[0].down_proj
    pm = lin.parametrizations.weight[0]
    W0 = lin.parametrizations.weight.original
    r, c = [int(v) for v in pm.mask.nonzero()[0]]
    x = torch.randn(3, W0.shape[1])
    y0 = lin(x)
    with torch.no_grad():
        pm.M[r, c] = 1.0
    y1 = lin(x)
    expect = x[:, c] * float(pm.alpha[r, 0])
    assert torch.allclose(y1[:, r] - y0[:, r], expect, atol=1e-5)
    other = [i for i in range(W0.shape[0]) if i != r]
    assert torch.allclose(y1[:, other], y0[:, other])


def test_tern2_two_plane_levels_and_patch_roundtrip(tmp_path):
    m = tiny(); E.attach(m, [0, 1], "fsa_free", 0, "tern2")
    pm = E.fsa_modules(m)[0]
    q = pm._quant(torch.tensor([-2.0, -1.3, -0.8, -0.3, 0.1, 0.26, 0.74, 1.1, 1.6]))
    assert q.tolist() == [-1.5, -1.5, -1.0, -0.5, 0.0, 0.5, 0.5, 1.0, 1.5]
    with torch.no_grad():
        for p in E.fsa_modules(m):
            p.M.copy_(torch.randn(p.M.shape))
    x = torch.tensor([[BOS_T] + list(range(40, 70))])
    y = m(x)
    E.save_patch(m, tmp_path / "t2.pt")
    m2 = tiny(); E.attach(m2, [0, 1], "fsa_free", 0, "tern2")
    E.load_patch(m2, tmp_path / "t2.pt")
    assert torch.allclose(m2(x), y)
    s = E.fsa_storage(10, 1000, "tern2")
    assert s["bytes_sparse"] == 10 * (10 + 3) / 8


# ---------------------------------------------------------------------------------------------- sleeping-unit gating
def test_unit_stats_selection_restrict_and_hinge():
    m = tiny()
    w = torch.tensor([[BOS_T] + list(range(40, 70)), [BOS_T] + list(range(70, 100))])
    st = E.unit_activation_stats(m, [0, 1], w)
    names = [n for n, _ in E.moe_layers(m)]
    assert len(st) == 8 and all(0 <= float(v["rate"].min()) and float(v["rate"].max()) <= 1 for v in st.values())
    sleep = E.select_units(st, 0.25, "sleep", 0)
    active = E.select_units(st, 0.25, "active", 0)
    rnd = E.select_units(st, 0.25, "random", 0)
    for k in st:
        assert int(sleep[k].sum()) == int(active[k].sum()) == int(rnd[k].sum()) == 8
        if st[k]["n"]:
            assert float(st[k]["rate"][sleep[k]].max()) <= float(st[k]["rate"][active[k]].min()) + 1e-9
    restrict = {}
    for k, sel in sleep.items():
        restrict[f"{k}.up_proj"] = ("rows", sel)
        restrict[f"{k}.down_proj"] = ("cols", sel)
    E.attach(m, [0, 1], "fsa_free", 0, restrict=restrict)
    for n, lin in E.patched_linears(m):
        pm = lin.parametrizations.weight[0]
        key = n.rsplit(".", 1)[0]
        sel = sleep[key]
        if n.endswith("up_proj"):
            assert not bool(pm.mask[~sel].any())
        else:
            assert not bool(pm.mask[:, ~sel].any())
    params = [p for p in E.fsa_modules(m)]
    with torch.no_grad():
        for p in params:
            p.M.copy_(p.mask.float())                       # +1 everywhere writable
    g = E.GateHinge(m, [0, 1], sleep, margin=0.5)
    for p in params:
        p.M.requires_grad_(True); p.M.grad = None
    g.start(); m(w); h = g.stop()
    assert h is not None and float(h) > 0
    h.backward(); g.end()
    assert any(p.M.grad is not None and float(p.M.grad.abs().sum()) > 0 for p in params)


def test_specific_unit_selection_prefers_fact_over_general():
    m = tiny()
    gen_w = torch.tensor([[BOS_T] + list(range(40, 70))])
    st = E.unit_activation_stats(m, [0, 1], gen_w)
    fst = E.unit_activation_stats(m, [0, 1], [[BOS_T] + list(range(80, 110)), [BOS_T] + list(range(90, 120))])
    sel = E.select_units(st, 0.25, "specific", 0, fst)
    for k in st:
        diff = fst[k]["rate"] - st[k]["rate"]
        if st[k]["n"] and fst[k]["n"]:
            assert float(diff[sel[k]].min()) >= float(diff[~sel[k]].max()) - 1e-6


# ---------------------------------------------------------------------------------------------- C1 diagnostics
@pytest.mark.parametrize("k", [2, 3])
def test_assignment_loss_prefers_correct_binding(k):
    from itertools import permutations
    scores = (4 * torch.eye(k) - 5).requires_grad_()
    loss = E.assignment_loss(scores)
    alternatives = [E.assignment_loss(scores[:, list(p)]) for p in permutations(range(k))]
    assert float(loss) == min(float(v) for v in alternatives)
    assert float(loss) < 0.002
    assert torch.allclose(E.assignment_loss(torch.zeros(k, k)), torch.tensor(math.log(math.factorial(k))))
    if k == 2:
        expected = F.softplus(scores[0, 1] + scores[1, 0] - scores[0, 0] - scores[1, 1])
        assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert bool((scores.grad.diagonal() < 0).all())
    assert bool((scores.grad[~torch.eye(k, dtype=torch.bool)] > 0).all())


def test_binding_candidates_use_complete_groups_and_clean_stops():
    facts = E.make_world(0, 3, 100, 30)
    groups = E.binding_groups(facts, E.conflict_map(facts))
    assert {len(g) for g in groups} == {2, 3}
    assert sorted(fi for group in groups for fi in group) == list(range(len(facts)))
    assert E.binding_groups(E.make_world(0), {}) == []
    group = next(g for g in groups if len(g) == 3)
    for template in E.TRAIN_AUG_T:
        candidates = E.binding_candidates(TOK, facts, group, template, BOS_T)
        for p in range(3):
            for q in range(3):
                seq, start = candidates[p * 3 + q]
                prompt, answer = TOK.decode(seq[:start]), TOK.decode(seq[start:])
                assert facts[group[p]].name in prompt
                assert answer == " " + facts[group[q]].val + "."
                # The diagonal uses exactly the ordinary answer-only training convention.
                if p == q:
                    assert (seq, start) == E.encode_fact(TOK, template, facts[group[p]], BOS_T)


def test_binding_scores_include_stop_and_normalize_each_continuation():
    m = tiny()
    candidates = [([BOS_T] + gen_ids("Prompt A."), 7), ([BOS_T] + gen_ids("Other long answer."), 6)]
    scores = E.binding_scores(m, candidates, "cpu")
    for score, (seq, start) in zip(scores, candidates):
        lp = m(torch.tensor([seq]))[0, :-1].log_softmax(-1)
        answer_lp = lp[torch.arange(start - 1, len(seq) - 1), torch.tensor(seq[start:])]
        assert len(answer_lp) == len(seq) - start             # includes the period; excludes prompt tokens
        assert torch.allclose(score, answer_lp.mean(), atol=1e-6)


@pytest.mark.parametrize("k,micro", [(2, 1), (3, 2), (3, 5)])
def test_binding_microbatch_backward_matches_full_assignment_gradient(k, micro):
    facts = E.make_world(0, k, 2, 2)
    group = E.binding_groups(facts, E.conflict_map(facts))[0]
    candidates = E.binding_candidates(TOK, facts, group, E.TRAIN_T[0], BOS_T)
    m = tiny()
    params, _, _ = _attach(m, "lora")
    m.eval()
    scores = torch.cat([E.binding_scores(m, candidates[s:s + micro], "cpu")
                        for s in range(0, len(candidates), micro)]).reshape(k, k)
    loss = E.assignment_loss(scores)
    (0.2 * loss).backward()
    grads = [p.grad.clone() if p.grad is not None else None for p in params]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    for p in params:
        p.grad = None
    result = E.binding_loss_backward(m, candidates, "cpu", micro, scale=0.2)
    assert math.isclose(result, float(loss), abs_tol=1e-6)
    for p, expected in zip(params, grads):
        if expected is None:
            assert p.grad is None
        else:
            assert torch.allclose(p.grad, expected, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("budget", [2, 4])
def test_keep_live_refresh_preserves_codes_and_ranks_remaining_slots(sparse, budget):
    cls = E.SparseFSAParam if sparse else E.FSAParam
    mods = [cls(torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool), budgeted=True) for _ in range(2)]
    opt = torch.optim.Adam([m.M for m in mods])
    with torch.no_grad():
        for i, m in enumerate(mods):
            m.support.fill_(True)
            m.M.view(-1)[0] = 1 if i == 0 else -1
            m.M.view(-1)[1] = 0.49                            # master nonzero but quantized zero: may be evicted
            m.score.fill_(-10)                              # low-scoring live codes must survive
            opt.state[m.M] = dict(exp_avg=torch.ones_like(m.M), exp_avg_sq=torch.ones_like(m.M))
            m.bop_ema = torch.ones_like(m.M)
        mods[0].score.view(-1)[2] = 20
        mods[1].score.view(-1)[3] = 30
    before = [m.codes().clone() for m in mods]
    churn = E.new_churn_totals()
    E.refresh_support(mods, budget, opt=opt, churn=churn, keep_live=True)
    assert sum(int(m.support.sum()) for m in mods) == budget
    assert churn["evictions"] == 0
    for m, codes in zip(mods, before):
        assert torch.equal(m.codes(), codes)
        assert bool(m.support.view(-1)[0])
        assert not bool(m.support.view(-1)[1])
        assert float(m.M.view(-1)[1]) == float(m.bop_ema.view(-1)[1]) == 0
        assert float(opt.state[m.M]["exp_avg"].view(-1)[0]) == 1
        assert float(opt.state[m.M]["exp_avg"].view(-1)[1]) == 0
    if budget == 4:
        assert bool(mods[0].support.view(-1)[2]) and bool(mods[1].support.view(-1)[3])
    # Protection is limited to refresh: Bop can still kill both live codes.
    for m in mods:
        m.M.grad = torch.sign(m.M.detach())
    births, deaths = E.bop_step(mods, gamma=1, tau=0.5, budgeted=True)
    assert births == 0 and deaths == 2
    E.refresh_support(mods, 1, keep_live=True)
    assert sum(int(m.support.sum()) for m in mods) == 1


def test_keep_live_rejects_impossible_budget_without_mutation():
    m = E.FSAParam(torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool), budgeted=True)
    with torch.no_grad():
        m.M[0, :2] = 1
    before = m.M.clone(), m.support.clone()
    with pytest.raises(ValueError, match="live slots.*exceed"):
        E.refresh_support([m], 1, keep_live=True)
    assert torch.equal(m.M, before[0]) and torch.equal(m.support, before[1])


def test_train_audit_is_fixed_seeded_and_does_not_consume_training_rng():
    facts = E.make_world(0, 3, 10, 5)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    audit = E.train_audit_items(TOK, facts, E.TRAIN_AUG_T, BOS_T, 17, 0)
    assert audit == E.train_audit_items(TOK, facts, E.TRAIN_AUG_T, BOS_T, 17, 0)
    assert audit != E.train_audit_items(TOK, facts, E.TRAIN_AUG_T, BOS_T, 17, 1)
    assert len(audit) == len({(it["fi"], it["ti"]) for it in audit}) == 17
    assert len(E.train_audit_items(TOK, facts, E.TRAIN_T, BOS_T, 10000, 0)) == len(facts) * len(E.TRAIN_T)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)


def test_rehearsal_overlap_rejects_eval_and_extra_panels():
    tokens = list(range(1000))
    for name in ("eval", "val", "conv", "extra:other.json"):
        with pytest.raises(SystemExit, match="rehearsal text overlaps"):
            E.checked_overlap(tokens, {name: tokens[100:500]}, "rehearsal")
    ov = E.checked_overlap(tokens, {"heldout": tokens[100:500], "eval": list(range(2000, 2500))}, "rehearsal")
    assert ov == {"heldout": 1, "eval": 0}


def test_extra_panels_windows_and_paired_standard_error(tmp_path, monkeypatch):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text(json.dumps(list(range(40 * 511 + 25))))
    panels = E.extra_eval_panels(",".join(map(str, paths)), BOS_T, "cpu")
    assert list(panels) == [p.name for p in paths]
    tokens, wins = panels["a.json"]
    assert wins.shape == (40, 512) and bool((wins[:, 0] == BOS_T).all())
    assert wins[:, 1:].flatten().tolist() == tokens == list(range(40 * 511))
    base = torch.linspace(1, 4, 40)
    delta = torch.linspace(-0.2, 0.3, 40)
    monkeypatch.setattr(E, "nll_windows", lambda model, w, dev: (base + delta).tolist())
    result = E.paired_nll(None, wins, base.tolist(), "cpu")
    assert result["dnll"] == pytest.approx(float(delta.mean()), abs=1e-7)
    assert result["dnll_se"] == pytest.approx(float(delta.std() / math.sqrt(40)), abs=1e-7)
    assert result["base_nll"] == pytest.approx(float(base.mean()))
    with pytest.raises(ValueError, match="distinct basenames"):
        E.extra_eval_panels(f"{paths[0]},{paths[0]}", BOS_T, "cpu")
    paths[0].write_text(json.dumps([0] * (40 * 511 - 1)))
    with pytest.raises(ValueError, match="at least"):
        E.extra_eval_panels(str(paths[0]), BOS_T, "cpu")


def test_c1_argparse_defaults_leave_all_new_features_off():
    args = ["--method", "fsa_forced", "--out", "unused.json"]
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    defaults = E.build_parser().parse_args(args)
    explicit = E.build_parser().parse_args(args + ["--bind-loss", "0", "--bind-groups", "4",
                                                  "--bind-every", "4", "--train-audit", "0"])
    assert vars(defaults) == vars(explicit)
    assert (defaults.bind_loss, defaults.bind_groups, defaults.bind_every, defaults.train_audit) == (0, 4, 4, 0)
    assert defaults.rehearsal_set is defaults.extra_eval_sets is None
    assert not defaults.keep_live
    assert (defaults.fmt, defaults.layers, defaults.steps, defaults.micro, defaults.optim) == (
        "ternary", "14,15", 300, 12, "adam")
    assert (defaults.rehearsal_frac, defaults.rehearsal_bs, defaults.budget, defaults.fact_batch) == (0, 0, 0, 0)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)


@pytest.fixture
def cpu_harness(tmp_path, monkeypatch):
    """Run the real main/train/eval flow with the tiny model and tokenizer replacing external loaders."""
    from types import ModuleType, SimpleNamespace

    def build_model(*args, **kwargs):
        m = tiny()
        for i, moe in enumerate(m.moes):
            for j, expert in enumerate(moe.experts):
                for lin in (expert.up_proj, expert.down_proj):
                    lin.weight.data.copy_(_pair8_tiny_weight(i * 4 + j, *lin.weight.shape))
        return m, None, "cpu"

    loader = ModuleType("ops.eval.bench_saves")
    loader.build_model = build_model
    monkeypatch.setitem(sys.modules, "ops.eval.bench_saves", loader)
    tokenizer = ModuleType("tokenizers")
    tokenizer.Tokenizer = SimpleNamespace(from_file=lambda path: TOK)
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizer)
    monkeypatch.setattr(E, "DATA", tmp_path)
    monkeypatch.setattr(E, "BOS", BOS_T)
    (tmp_path / "heldout-tokens.json").write_text(json.dumps([BOS_T] + [80] * (5 * 512)))
    paths = {}
    for name, token, count in (("eval", 90, 61022), ("extra", 95, 40 * 511),
                               ("anchor", 40, 1000), ("rehearsal", 50, 1000)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps([token] * count))
        paths[name] = str(path)

    args = ["--method", "fsa_forced", "--layers", "0,1", "--people", "3", "--facts-per-person", "2",
            "--shared-attrs", "2", "--micro", "2", "--fact-batch", "3", "--optim", "bop",
            "--bop-tau-abs", "0.0001", "--budget", "20", "--eval-set", paths["eval"],
            "--val-windows", "2", "--eval-every", "2", "--gen-max-new", "1", "--no-pareto"]

    def run(extra_args, name):
        out = tmp_path / f"{name}_result.json"
        patch_dir = tmp_path / f"{name}_patches"
        monkeypatch.setattr(sys, "argv", ["e2e_fsa.py", *args, "--out", str(out),
                                          "--patch-dir", str(patch_dir), *extra_args])
        E.set_enabled(True)
        try:
            E.main()
            return json.loads(out.read_text()), patch_dir / "final.pt"
        finally:
            E.set_enabled(True)

    return run, paths


def test_main_c1_features_and_loaded_patch_final_evaluation(cpu_harness, monkeypatch, capsys):
    run, paths = cpu_harness
    rehearsal_tokens, anchor_tokens, binding_updates = [], [], []
    old_ce, old_energy, old_bind = E.general_ce, E.collect_col_energy, E.binding_loss_backward

    def ce(model, windows):
        rehearsal_tokens.extend(windows[:, 1:].flatten().tolist())
        return old_ce(model, windows)

    def energy(model, mods, windows):
        anchor_tokens.extend(windows[:, 1:].flatten().tolist())
        return old_energy(model, mods, windows)

    def bind(model, candidates, dev, micro, scale):
        binding_updates.append((len(candidates), scale))
        return old_bind(model, candidates, dev, micro, scale)

    monkeypatch.setattr(E, "general_ce", ce)
    monkeypatch.setattr(E, "collect_col_energy", energy)
    monkeypatch.setattr(E, "binding_loss_backward", bind)
    flags = ["--bind-loss", "0.2", "--bind-groups", "1", "--bind-every", "2", "--train-audit", "3",
             "--rehearsal-frac", "0.2", "--rehearsal-set", paths["rehearsal"], "--anchor", paths["anchor"],
             "--anchor-len", "16", "--support-score", "netcost", "--local-lambda", "0.01",
             "--extra-eval-sets", paths["extra"], "--keep-live"]
    trained, patch = run(["--steps", "4", *flags], "trained")
    assert len(binding_updates) == 2 and all(n == 9 for n, _ in binding_updates)
    assert all(scale == pytest.approx(0.16) for _, scale in binding_updates)
    assert set(rehearsal_tokens) == {50} and set(anchor_tokens) == {40}
    assert [row["step"] for row in trained["trajectory"]] == [2, 4]
    assert all(0 <= row["train_em"] <= 1 for row in trained["trajectory"])
    assert trained["revoked"]["logits_bit_exact"]
    loaded, _ = run(["--steps", "0", "--load-patch", str(patch), *flags], "loaded")
    assert loaded["trajectory"] == []
    assert loaded["revoked"]["logits_bit_exact"]
    assert loaded["patched"]["big_extra"] == trained["patched"]["big_extra"]
    assert len(binding_updates) == 2                         # no training during --steps 0
    lines = capsys.readouterr().out.splitlines()
    final = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("FINAL ")]
    assert len(final) == 2
    assert all(set(row["big_extra"]["extra.json"]) == {"dnll", "dnll_se", "base_nll"} for row in final)
    assert sum(line.startswith("EXTRA extra.json ") and "±" in line for line in lines) == 2


def test_main_defaults_and_explicit_off_flags_have_identical_patch_and_rng(cpu_harness):
    run, _ = cpu_harness
    defaults, patch_a = run(["--steps", "3"], "default")
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    explicit, patch_b = run(["--steps", "3", "--bind-loss", "0", "--train-audit", "0",
                             "--curriculum", "none", "--cur-check", "10", "--cur-active", "8",
                             "--cur-replay", "0.25"], "explicit")
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    assert "big_extra" not in defaults["patched"]
    assert all("train_em" not in row for row in defaults["trajectory"])
    assert defaults["trajectory"] == explicit["trajectory"]
    assert defaults["patched"] == explicit["patched"]
    a = torch.load(patch_a, weights_only=False)
    b = torch.load(patch_b, weights_only=False)
    assert a.keys() == b.keys() and a["entries"].keys() == b["entries"].keys()
    for name, entry in a["entries"].items():
        for key, value in entry.items():
            other = b["entries"][name][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other


# ---------------------------------------------------------------------------------------------- C2 rules
def test_c2_defaults_leave_new_features_off():
    a = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused"])
    assert not any((a.strata, a.score_epoch, a.pair_birth, a.fisher_support))
    assert (a.curriculum, a.cur_check, a.cur_active, a.cur_replay) == ("none", 10, 8, 0.25)
    assert a.patch_scale_mult_up is a.patch_scale_mult_down is None
    E.validate_c2_args(a)


@pytest.mark.parametrize("flags,match", [
    (["--score-epoch"], "needs --budget"),
    (["--fisher-support", "--budget", "10"], "--adam-phase"),
    (["--pair-birth"], "needs --optim bop"),
    (["--cur-check", "0"], "--cur-check"),
    (["--cur-active", "0"], "--cur-active"),
    (["--cur-replay", "nan"], "--cur-replay"),
    (["--curriculum", "attribute", "--paired"], "replaces"),
    (["--score-epoch", "--budget", "10", "--pool-select", "random"], "static"),
    (["--curriculum", "attribute", "--fmt", "tern2"], "plain"),
    (["--pair-birth", "--method", "fsa_free"], "fsa_forced"),
    (["--patch-scale-mult-up", "0"], "multipliers"),
    (["--patch-scale-mult-down", "inf"], "multipliers"),
])
def test_c2_rejects_invalid_flags(flags, match):
    a = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", *flags])
    with pytest.raises(SystemExit, match=match):
        E.validate_c2_args(a)


def test_strata_use_full_world_arity_and_reuse_eval_passes(monkeypatch):
    facts = E.make_world(0, 3, 100, 30)
    arities = E.fact_arities(facts)
    assert Counter(arities.values()) == {2: 120, 3: 180}
    # Only one person from each arity in the validation subset must still use world arity.
    idxs = [next(fi for fi, k in arities.items() if k == n) for n in (2, 3)]
    items = E.build_items(TOK, facts, E.VAL_T[:1], BOS_T, idxs=idxs)
    seen = []

    def generate(model, prompts, dev, max_new, bs):
        seen.append(len(prompts))
        return [gen_ids(" " + facts[idxs[0]].val + "."), gen_ids(" Wrong.")]

    monkeypatch.setattr(E, "greedy_generate", generate)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    gen = E.gen_eval(None, TOK, items, "cpu", arities=arities)
    assert gen["by_arity"] == {2: 1, 3: 0} and gen["recall"] == 0.5
    assert seen == [2]
    m = tiny()
    plain = E.tf_eval(m, items, "cpu")
    stratified = E.tf_eval(m, items, "cpu", arities=arities)
    assert {k: v for k, v in stratified.items() if k != "by_arity"} == plain
    assert stratified["by_arity"] == {n: E.tf_eval(m, [it], "cpu")["exact_match"]
                                      for n, it in zip((2, 3), items)}
    # tiny() seeds global torch RNG; evaluating either form must not consume it.
    torch_state = torch.get_rng_state()
    E.tf_eval(m, items, "cpu", arities=arities)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)


def _curriculum(limit=12, active=2, replay=0.25):
    facts = E.make_world(0, 3, 100, 30)
    pool = [[E.encode_fact(TOK, t, f, BOS_T) for t in E.TRAIN_AUG_T] for f in facts]
    return E.AttributeCurriculum(facts, pool, limit, active, replay, seed=3), facts


def test_curriculum_whole_groups_format_cap_and_priority():
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    cur, facts = _curriculum(limit=6)
    assert all(len(cur.groups[at]) == 3 for at in cur.active)
    chosen = cur.batch()
    assert len(chosen) == 18                                # 6 facts x 3 formats, cap excludes 12 formats
    by = defaultdict(set)
    for fi, _ in chosen:
        by[facts[fi].attr].add(fi)
    assert all(fis == set(cur.groups[at]) for at, fis in by.items())
    assert all(Counter(fi for fi, _ in chosen)[fi] == 3 for fi in {fi for fi, _ in chosen})
    cur, _ = _curriculum(limit=24, active=1)
    chosen = cur.batch()
    assert len(chosen) == 3 * len(E.TRAIN_AUG_T)             # all formats fit the 72-sequence cap
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(ValueError, match="whole attribute group"):
        _curriculum(limit=2)


def test_curriculum_two_consecutive_checks_and_failed_replay_reentry():
    cur, facts = _curriculum(limit=12, active=2)
    first = set(cur.active)
    calls = []

    def correct(items):
        calls.append(items)
        assert len(items) == 9                             # every fact, three sampled formats including period
        assert all(TOK.decode(it["a"]).endswith(".") for it in items)
        return {"exact_match": 1.0}

    cur.check(correct)
    assert not cur.mastered
    cur.check(lambda items: {"exact_match": 0.5})
    cur.check(correct)
    assert not cur.mastered                                # failure resets the success streak
    cur.check(correct)
    assert cur.mastered == first and len(cur.active) == 2
    chosen = cur.batch()
    replayed = {facts[fi].attr for fi, _ in chosen} & first
    assert len(replayed) == 1                              # .25 of 12 facts = one three-person group
    failed = next(iter(replayed))
    cur.check(lambda items: {"exact_match": 0 if facts[items[0]["fi"]].attr == failed else 1})
    assert failed in cur.active and failed not in cur.mastered
    assert len(cur.active) == 2 and cur.streak[failed] == 0
    pending = set(cur.pending)
    assert not (pending & set(cur.active) or pending & cur.mastered or set(cur.active) & cur.mastered)
    assert pending | set(cur.active) | cur.mastered == set(cur.groups)


def test_curriculum_zero_replay_and_all_mastered_remain_trainable():
    cur, facts = _curriculum(limit=12, active=2, replay=0)
    cur.check(lambda items: {"exact_match": 1})
    cur.check(lambda items: {"exact_match": 1})
    chosen = cur.batch()
    assert not ({facts[fi].attr for fi, _ in chosen} & cur.mastered)
    cur.mastered = set(cur.groups)
    cur.pending.clear()
    cur.active.clear()
    chosen = cur.batch()
    assert chosen and len({fi for fi, _ in chosen}) <= 12


def test_curriculum_replay_share_uses_actual_active_capacity():
    cur, facts = _curriculum(limit=40, active=8)
    # First eight groups are mastered, leaving eight active 3-way groups (24 facts).
    cur.check(lambda items: {"exact_match": 1})
    cur.check(lambda items: {"exact_match": 1})
    chosen = cur.batch()
    fis = {fi for fi, _ in chosen}
    replayed = {fi for fi in fis if facts[fi].attr in cur.mastered}
    assert len(fis) == 30 and len(replayed) == 6            # quota 8 rounded down to two whole 3-way groups
    assert abs(len(replayed) / len(fis) - cur.replay) <= 3 / len(fis)


@pytest.mark.parametrize("sparse", [False, True])
def test_epoch_scores_are_signed_sums_with_no_decay(sparse):
    W = _pair8_tiny_weight(3, 2, 16)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    m = cls(W, E.pair8_forced_mask(W), budgeted=True)
    m.score = torch.zeros_like(m.M)
    m.M.grad = torch.ones_like(m.M) * 2
    E.accumulate_epoch_scores([m])
    m.M.grad.fill_(-0.5)
    E.accumulate_epoch_scores([m])
    assert torch.equal(m.score, m.mask.float() * 1.5)
    before = m.score.clone()
    m.M.grad = None
    E.accumulate_epoch_scores([m])
    assert torch.equal(m.score, before)


@pytest.mark.parametrize("sparse", [False, True])
def test_pair_birth_patterns_match_exhaustive_nine_pattern_minimum(sparse):
    from itertools import product
    W = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(5, 1)
    mask = E.pair8_forced_mask(W)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    m = cls(W, mask, budgeted=True)
    g = torch.tensor([[2., -3.], [2., 0.2], [1., -1.], [0., 0.], [-2., 4.]])
    dense_g = torch.zeros_like(W)
    dense_g[:, 4:6] = g
    m.M.grad = dense_g.flatten()[m.idx.long()] if sparse else dense_g
    m.support.copy_(m.mask)
    m.bop_ema = torch.zeros_like(m.M)
    before = W.clone()
    nb, nd = E.bop_step([m], 1, 1, True, pair_birth=True)
    codes = m.codes()
    for r, grad in enumerate(g):
        patterns = [torch.tensor(p, dtype=torch.float32) for p in product((-1, 0, 1), repeat=2)
                    if all(v == 0 or abs(float(gg)) > .5 for v, gg in zip(p, grad))
                    and (not any(p) or sum(abs(float(gg)) for v, gg in zip(p, grad) if v) > .75 * sum(v != 0 for v in p))]
        best = min(patterns, key=lambda p: float(grad @ p))
        assert torch.equal(codes[r, 4:6], best)
        assert float((grad.abs() * best.abs()).sum()) > .75 * int(best.count_nonzero()) or not best.any()
    assert (nb, nd) == (7, 0)
    assert torch.equal(W, before) and not bool(codes[~mask].any())
    assert set(codes.unique().tolist()) <= {-1, 0, 1}


def test_sparse_pair_birth_uses_weight_positions_not_slot_order_and_deaths_stay_scalar():
    W = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(2, 1)
    mask = E.pair8_forced_mask(W)
    mask[0, 5] = False                                    # idx[0]=4, idx[1]=6 are different pairs
    m = E.SparseFSAParam(W, mask, budgeted=True)
    with torch.no_grad():
        m.support.fill_(True)
        m.M[0] = 1
    m.bop_ema = torch.zeros_like(m.M)
    m.M.grad = torch.full_like(m.M, -2)
    E.bop_step([m], 1, 1, True, pair_birth=True)
    assert m.codes()[0, 6] == 1 and m.codes()[1, 4] == 1    # different pair and row both born
    # A surviving leg blocks births on the other leg, but dies independently without killing its partner.
    with torch.no_grad():
        m.M.zero_()
        pos = (m.idx == 12).nonzero().item()
        m.M[pos] = 1
    m.M.grad.fill_(-2)
    E.bop_step([m], 1, 1, True, pair_birth=True)
    assert m.codes()[1, 5] == 0
    m.M.grad.zero_()
    m.M.grad[pos] = 2
    _, nd = E.bop_step([m], 1, 1, True, pair_birth=True)
    assert nd == 1 and m.codes()[1, 4] == 0
    m.M.grad.fill_(-2)
    before = m.M.clone()
    nb, _ = E.bop_step([m], 1, 1, True, pair_birth=True, allow_birth=False)
    assert nb == 0 and torch.equal(m.M, before)


@pytest.mark.parametrize("sparse", [False, True])
def test_fisher_copies_adam_moments_and_ranks_gain_minus_cost(sparse):
    W = _pair8_tiny_weight(0, 2, 8)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    m = cls(W, E.pair8_forced_mask(W), budgeted=True)
    opt = torch.optim.Adam([m.M])
    m.M.grad = torch.arange(m.M.numel(), dtype=torch.float32).reshape_as(m.M)
    opt.step()
    v = opt.state[m.M]["exp_avg_sq"]
    expected = v.clone()
    E.capture_fisher([m], opt)
    assert m.fisher.data_ptr() != v.data_ptr()
    v.zero_()
    opt.state.clear()
    assert torch.equal(m.fisher, expected)
    m.col_energy = torch.ones(W.shape[1])
    m.score = torch.ones_like(m.M) * -2
    score = E.module_score(m, "fisher", 0.2, 3)
    assert torch.allclose(score, (4 / (expected + 1e-8) - 0.2 * E.slot_cost(m, 3)) * m.mask)
    assert torch.isfinite(score).all()
    E.refresh_support([m], 2, [score])
    assert int(m.support.sum()) == 2 and not bool((m.support & ~m.mask).any())


@pytest.mark.parametrize("sparse", [False, True])
def test_projection_scale_overrides_roundtrip_and_exact_revoke(sparse):
    m = tiny()
    for i, moe in enumerate(m.moes):
        for j, expert in enumerate(moe.experts):
            for lin in (expert.up_proj, expert.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(i * 4 + j, *lin.weight.shape))
    ids = torch.tensor([[BOS_T, 60, 61]])
    before = m(ids).detach().clone()
    E.attach(m, [0, 1], "fsa_forced", 2, sparse=sparse)
    alphas = {n: lin.parametrizations.weight[0].alpha.clone() for n, lin in E.patched_linears(m)}
    E.apply_projection_scales(m, 0.75, up=0.5)
    with torch.no_grad():
        for n, lin in E.patched_linears(m):
            pm = lin.parametrizations.weight[0]
            scale = 0.5 if n.endswith("up_proj") else 0.75
            assert torch.equal(pm.alpha, alphas[n] * scale)
            pm.M.copy_(pm.mask.float())
    state = E.export_patch(m, {"patch_scale_mult_up": 0.5, "patch_scale_mult_down": None})
    E.load_patch(m, state)
    for pm in E.fsa_modules(m):
        assert set(pm.codes().unique().tolist()) <= {0, 1}
    E.set_enabled(False)
    try:
        assert torch.equal(m(ids), before)
    finally:
        E.set_enabled(True)


def test_main_c2_epoch_fisher_strata_curriculum_and_scale_loading(cpu_harness, monkeypatch):
    run, paths = cpu_harness
    refreshes, captures, accumulated, checks = [], [], [], []
    old_refresh, old_capture = E.refresh_support, E.capture_fisher
    old_score, old_accumulate, old_check = E.module_score, E.accumulate_epoch_scores, E.AttributeCurriculum.check
    rank_modes = []

    def score(m, mode, cost_lambda, den):
        rank_modes.append(mode)
        return old_score(m, mode, cost_lambda, den)

    def refresh(mods, *args, **kwargs):
        refreshes.append([m.score.clone() for m in mods])
        return old_refresh(mods, *args, **kwargs)

    def capture(mods, opt):
        expected = [opt.state.get(m.M, {}).get("exp_avg_sq", torch.zeros_like(m.M)).clone() for m in mods]
        old_capture(mods, opt)
        captures.append(expected)
        assert all(torch.equal(m.fisher, v) for m, v in zip(mods, expected))

    def accumulate(mods):
        before = [m.score.clone() for m in mods]
        old_accumulate(mods)
        assert all(m.score.dtype == torch.float32 for m in mods)
        for m, prev in zip(mods, before):
            expected = prev if m.M.grad is None else prev + m.M.grad * m.mask
            assert torch.equal(m.score, expected)
        accumulated.append([m.score.clone() for m in mods])
        if len(accumulated) in (1, 3, 5):
            assert all(not prev.any() for prev in before)  # reset after each complete epoch, even after freeze

    def check(cur, evaluate):
        checks.append(len(cur.mastered))
        return old_check(cur, evaluate)

    monkeypatch.setattr(E, "module_score", score)
    monkeypatch.setattr(E, "refresh_support", refresh)
    monkeypatch.setattr(E, "capture_fisher", capture)
    monkeypatch.setattr(E, "accumulate_epoch_scores", accumulate)
    monkeypatch.setattr(E.AttributeCurriculum, "check", check)
    flags = ["--strata", "--train-audit", "5", "--curriculum", "attribute", "--cur-check", "2",
             "--cur-active", "1", "--score-epoch", "--fisher-support", "--adam-phase", "2",
             "--pair-birth", "--sparse-state", "--anchor", paths["anchor"], "--anchor-len", "16",
             "--budget-freeze", "3", "--patch-scale-mult-up", "0.5", "--patch-scale-mult-down", "0.75"]
    result, patch = run(["--steps", "6", *flags], "c2")
    assert len(refreshes) == 2                              # epochs end at steps 2,4; step 6 is frozen
    assert len(captures) == 1 and len(accumulated) == 6 and len(checks) == 3
    assert any(v.any() for v in captures[0])
    assert set(rank_modes) == {"netcost", "fisher"}
    for row in result["trajectory"]:
        assert row["val_gen_2way"] is row["train_em_2way"] is None
        assert row["val_gen_3way"] == row["val_gen"]
        assert 0 <= row["train_em_3way"] <= 1
        assert "val_confusion" in row and "val_all_people_correct" in row
        assert row["cur_active"] <= 1 and "cur_mastered" in row
    assert result["revoked"]["logits_bit_exact"]
    saved = torch.load(patch, weights_only=False)
    assert saved["meta"]["patch_scale_mult_up"] == 0.5
    assert saved["meta"]["patch_scale_mult_down"] == 0.75
    # Evaluation-only loads need identical serving scales, but no training strategy flags.
    loaded, _ = run(["--steps", "0", "--load-patch", str(patch), "--patch-scale-mult-up", "0.5",
                     "--patch-scale-mult-down", "0.75", "--sparse-state"], "c2_loaded")
    assert loaded["patched"] == result["patched"]
    assert loaded["revoked"]["logits_bit_exact"]
    with pytest.raises(SystemExit, match="config mismatch.*patch_scale_mult_up"):
        run(["--steps", "0", "--load-patch", str(patch)], "c2_mismatched")


def test_main_strata_preserve_training_patch_and_rng(cpu_harness):
    run, _ = cpu_harness
    ordinary, patch_a = run(["--steps", "2", "--train-audit", "3"], "no_strata")
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    strata, patch_b = run(["--steps", "2", "--train-audit", "3", "--strata"], "strata")
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    a, b = (torch.load(p, weights_only=False)["entries"] for p in (patch_a, patch_b))
    for name in a:
        assert torch.equal(a[name]["val"], b[name]["val"]) and torch.equal(a[name]["idx"], b[name]["idx"])
    assert ordinary["patched"] == strata["patched"]


def test_main_fisher_without_epoch_preserves_refresh_schedule(cpu_harness, monkeypatch):
    run, paths = cpu_harness
    modes = []
    old_refresh = E.refresh_support

    def refresh(mods, budget, scores, *args, **kwargs):
        # The support callable's closure selects Fisher only after the Adam switch.
        modes.append("fisher" if hasattr(mods[0], "fisher") else "netcost")
        return old_refresh(mods, budget, scores, *args, **kwargs)

    monkeypatch.setattr(E, "refresh_support", refresh)
    result, _ = run(["--steps", "11", "--fisher-support", "--adam-phase", "1",
                     "--anchor", paths["anchor"], "--anchor-len", "16", "--sparse-state"], "fisher_only")
    assert modes == ["netcost", "fisher"]                  # zero-based steps 0 and 10
    assert result["revoked"]["logits_bit_exact"]


def test_main_epoch_uses_ceiling_and_does_not_refresh_partial_epoch(cpu_harness, monkeypatch):
    run, paths = cpu_harness
    refreshes = []
    old_refresh = E.refresh_support

    def refresh(mods, *args, **kwargs):
        refreshes.append(1)
        return old_refresh(mods, *args, **kwargs)

    monkeypatch.setattr(E, "refresh_support", refresh)
    result, _ = run(["--steps", "5", "--score-epoch", "--fact-batch", "4",
                     "--anchor", paths["anchor"], "--anchor-len", "16"], "epoch_only")
    assert len(refreshes) == 2                             # ceil(6/4)=2; completed epochs at steps 2,4
    assert result["revoked"]["logits_bit_exact"]


def _c3_pair8_model():
    model = tiny()
    for i, moe in enumerate(model.moes):
        for j, expert in enumerate(moe.experts):
            for lin in (expert.up_proj, expert.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(i * 4 + j, *lin.weight.shape))
    return model


@pytest.mark.parametrize('sparse', [False, True])
def test_joint_pair_birth_accepts_moderate_evidence_and_checks_each_leg(sparse):
    W = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(4, 1)
    m = (E.SparseFSAParam if sparse else E.FSAParam)(W, E.pair8_forced_mask(W))
    g = torch.zeros_like(W)
    g[:, 4:6] = torch.tensor([[.8, -.8], [.74, -.74], [1.1, -.5], [.76, -.1]])
    m.M.grad = g.flatten()[m.idx.long()] if sparse else g
    m.bop_ema = torch.zeros_like(m.M)
    E.bop_step([m], 1, 1, False, pair_birth=True)
    assert torch.equal(m.codes()[:, 4:6], torch.tensor([[-1., 1.], [0., 0.], [-1., 0.], [-1., 0.]]))


def test_distillation_exact_kl_gold_gating_temperature_and_annealing():
    student = torch.tensor([[[.2, -.3, .5], [.4, -.1, .2], [.7, .1, -.2]]], requires_grad=True)
    teacher = torch.tensor([[[4., 0., -1.], [0., 4., -1.], [0., -1., 4.]]], requires_grad=True)
    gold = torch.tensor([[0, 0, 2]])
    mask = torch.tensor([[True, True, False]])
    actual = E.distillation_kl(student, teacher, gold, mask, 2)
    expected = 4 * F.kl_div(F.log_softmax(student[0, 0] / 2, -1),
                            F.softmax(teacher[0, 0].detach() / 2, -1), reduction='sum')
    assert torch.allclose(actual, expected)
    actual.backward()
    assert student.grad[0, 0].any() and not student.grad[0, 1:].any()
    assert teacher.grad is None
    assert [E.distill_weight(.3, True, s, 3) for s in range(3)] == [.3, .15, 0]
    assert E.distill_weight(.3, True, 0, 1) == 0
    assert E.distill_weight(.3, False, 100, 3) == .3
    zero = E.distillation_kl(student, teacher, gold, torch.zeros_like(mask), 2)
    assert zero.item() == 0 and torch.isfinite(zero)


def test_teacher_independent_switch_stack_different_layers_frozen_rng_and_removal():
    model = _c3_pair8_model()
    ids = torch.tensor([[BOS_T, 60, 61]])
    E.attach(model, [1], 'lora', 2)
    with torch.no_grad():
        for m in E.lora_modules(model):
            m.B.fill_(.2)
    state = E.export_patch(model, {'method': 'lora', 'layers': [1], 'rank': 2})
    E.detach_all(model)
    E.set_enabled(True)
    base = model(ids).detach()
    E.attach(model, [0], 'fsa_forced', 2)
    with torch.no_grad():
        for m in E.fsa_modules(model):
            m.M.copy_(m.mask.float())
    student = model(ids).detach()
    rng = torch.get_rng_state()
    teachers = E.attach_teacher(model, state)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(model(ids), student)
    assert not E.lora_modules(model)
    assert all(not p.requires_grad for t in teachers for p in t.parameters())
    with E.teacher_forward(teachers):
        teacher_logits = model(ids)
        assert not teacher_logits.requires_grad
        assert not E.FSAParam.enabled and all(t.enabled for t in teachers)
    assert not torch.equal(teacher_logits, base)
    assert torch.equal(model(ids), student)
    with pytest.raises(AssertionError, match='teacher present'):
        E.export_patch(model)
    with pytest.raises(RuntimeError):
        with E.teacher_forward(teachers):
            raise RuntimeError('test restoration')
    assert E.FSAParam.enabled and not any(t.enabled for t in teachers)
    E.remove_teacher(model)
    assert torch.equal(model(ids), student)
    assert all(e['kind'] == 'fsa' for e in E.export_patch(model)['entries'].values())
    E.set_enabled(False)
    try:
        assert torch.equal(model(ids), base)
    finally:
        E.set_enabled(True)


def test_saturation_two_phrasings_margins_weights_reopening_and_sampling_floor():
    s = E.SaturationSampler(100, 2, .25, 13)
    gold = torch.tensor([0, 1])
    correct = torch.tensor([[3., 0., 0.], [0., 3., 0.]])
    s.observe(0, correct, gold, 0)
    assert not s.saturated(0)
    s.observe(0, correct, gold, 0)
    assert s.saturated(0) and s.weights()[0] == .25
    s.observe(0, correct * .5, gold, 1)
    assert not s.saturated(0) and s.reopened == 1 and s.weights()[0] == 1
    wrong = correct.flip(-1)
    s.observe(0, wrong, gold, 2)
    assert 1 < s.weights()[0] <= 4
    # Impossible fixed-size floor: the sampler explicitly enlarges batches to include every overdue fact.
    s = E.SaturationSampler(100, 2, .25, 13)
    global_rng = random.getstate()
    for step in range(120):
        chosen = s.batch(1, step)
        for fi in chosen:
            s.observe(fi, correct, gold, step)
        if step >= 39:
            assert all(step - last < 40 for last in s.last)
    assert random.getstate() == global_rng
    assert s.stats()['sat_saturated'] == 100


@pytest.mark.parametrize('sparse', [False, True])
def test_death_metaplasticity_without_churn_and_age_in_wall_steps(sparse):
    W = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]])
    m = (E.SparseFSAParam if sparse else E.FSAParam)(W, E.pair8_forced_mask(W))
    E.init_step_age(m)
    indices = m.mask.flatten().nonzero().flatten()[:2]
    with torch.no_grad():
        m.M.flatten()[indices] = 1
        m.age.flatten()[indices] = torch.tensor([0, 100], dtype=torch.int16)
    m.bop_ema = torch.zeros_like(m.M)
    m.M.grad = torch.zeros_like(m.M)
    m.M.grad.flatten()[indices] = 2
    assert E.bop_step([m], 1, 1, False, death_age_mult=True) == (0, 1)
    assert m.M.flatten()[indices].tolist() == [0, 1]
    assert m.age.flatten()[indices].tolist() == [0, 100]
    E.advance_step_age([m])
    assert m.age.flatten()[indices].tolist() == [0, 101]
    m.age.flatten()[indices[1]] = 32767
    E.advance_step_age([m])
    assert m.age.flatten()[indices[1]] == 32767
    # Threshold noise applies to births only; death tau remains unchanged.
    m.M.grad.flatten()[indices] = 4
    assert E.bop_step([m], 1, 1, False, birth_tau=100) == (0, 1)


@pytest.mark.parametrize('sparse', [False, True])
def test_patch_dropout_forward_only_reheat_separate_rng_export_and_revoke(sparse):
    model = _c3_pair8_model()
    E.attach(model, [0], 'fsa_forced', 2, sparse=sparse)
    mods = E.fsa_modules(model)
    with torch.no_grad():
        for m in mods:
            m.M.copy_(m.mask.float())
            m.bop_ema = torch.ones_like(m.M)
            E.init_step_age(m)
            m.age.fill_(100)
    state = E.export_patch(model)
    codes = [m.codes().clone() for m in mods]
    rng = torch.get_rng_state()
    E.patch_chaos(mods, 1, 1, torch.Generator().manual_seed(31))
    assert torch.equal(torch.get_rng_state(), rng)
    for m, c in zip(mods, codes):
        assert not m.delta().any() and torch.equal(m.codes(), c)
        live = m._quant(m.M) != 0
        assert not m.bop_ema[live].any() and bool((m.age[live] == 50).all())
    E.clear_patch_dropout(mods)
    for m in mods:
        assert m.delta().any()
    after = E.export_patch(model)
    for name in state['entries']:
        assert torch.equal(state['entries'][name]['val'], after['entries'][name]['val'])
    E.patch_chaos(mods, .5, 0, torch.Generator().manual_seed(10))
    first = [m.forward_keep.clone() for m in mods]
    E.patch_chaos(mods, .5, 0, torch.Generator().manual_seed(10))
    assert all(torch.equal(a, m.forward_keep) for a, m in zip(first, mods))
    E.clear_patch_dropout(mods)


def test_swap_margin_population_std_cap_and_keep_live_preserves_master_ema():
    m = E.FSAParam(torch.ones(1, 6), torch.ones(1, 6, dtype=torch.bool), budgeted=True)
    m.support[:] = torch.tensor([[True, True, False, False, False, False]])
    m.bop_ema = torch.arange(6.).reshape(1, 6)
    with torch.no_grad():
        m.M[0, 0] = 1
    scores = [torch.tensor([[1., 2., 2.1, 3., 0., 0.]])]
    E.refresh_support([m], 2, scores, keep_live=True, swap_margin=10)
    assert m.support.tolist() == [[True, True, False, False, False, False]]
    E.refresh_support([m], 2, scores, keep_live=True, swap_limit=1)
    assert m.support.tolist() == [[True, True, False, False, False, False]]
    E.refresh_support([m], 2, scores, keep_live=True, swap_limit=2)
    assert m.support.tolist() == [[True, False, False, True, False, False]]
    assert m.M[0, 0] == 1 and m.bop_ema[0, 0] == 0  # original EMA unchanged
    assert m.bop_ema[0, 1] == 0                     # evicted dormant EMA cleared
    before = m.support.clone()
    E.refresh_support([m], 2, [scores[0].flip(-1)], swap_limit=0)
    assert torch.equal(m.support, before)
    assert E.support_swap_limit(0, 20, 100, 100) is None
    assert [E.support_swap_limit(.5, 20, t, 100) for t in (0, 30, 60, 100)] == [10, 5, 0, 0]


@pytest.mark.parametrize('flags, message', [
    (['--distill-mu', '.3'], 'teacher-patch'),
    (['--distill-T', '0'], 'distill-T'),
    (['--patch-dropout', 'nan'], 'patch-dropout'),
    (['--tau-noise', '-1'], 'tau-noise'),
    (['--sat-weight', '0'], 'sat-weight'),
    (['--swap-cap', 'nan'], 'swap-cap'),
    (['--reheat-every', '2'], 'reheat-every'),
    (['--reheat-frac', '.3'], 'reheat-every'),
    (['--death-age-mult', '--optim', 'adam'], 'optim bop'),
    (['--retain-evicted', '--budget', '0'], 'budget'),
    (['--saturation-sampling', '--paired'], 'replaces'),
    (['--patch-dropout', '.1', '--method', 'fsa_full'], 'forced-mask'),
    (['--saturation-sampling', '--occupied-flips'], 'forced-mask'),
    (['--pair-birth', '--occupied-flips'], 'forced-mask'),
])
def test_c3_validation(flags, message):
    a = E.build_parser().parse_args(['--method', 'fsa_forced', '--optim', 'bop', '--budget', '20', '--out', '/private/tmp/c3-unused.json', *flags])
    with pytest.raises(SystemExit, match=message):
        E.validate_c3_args(a)


def test_main_c3_off_flags_preserve_default_patch_and_rng(cpu_harness):
    run, _ = cpu_harness
    ordinary, pa = run(['--steps', '3'], 'c3_default')
    py_rng, torch_rng = random.getstate(), torch.get_rng_state()
    off, pb = run(['--steps', '3', '--distill-mu', '0', '--distill-T', '2', '--sat-margin', '2',
                   '--sat-weight', '.25', '--swap-margin', '0', '--swap-cap', '0', '--patch-dropout', '0',
                   '--tau-noise', '0', '--reheat-every', '0', '--reheat-frac', '0'], 'c3_off')
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    a, b = [torch.load(p, weights_only=False)['entries'] for p in (pa, pb)]
    for name in a:
        assert torch.equal(a[name]['idx'], b[name]['idx']) and torch.equal(a[name]['val'], b[name]['val'])
    assert ordinary['patched'] == off['patched']


def test_main_c3_distillation_binding_saturation_chaos_and_no_churn(cpu_harness, monkeypatch):
    run, _ = cpu_harness
    _, teacher_path = run(['--method', 'lora', '--optim', 'adam', '--budget', '0', '--rank', '2',
                           '--layers', '0', '--steps', '2', '--bop-tau-abs', '0'], 'c3_teacher')
    evaluations, forwards, weights, samples, birth_taus = [], [], [], [], []
    old_tf, old_nll, old_logits = E.tf_eval, E.nll_windows, E.logits_of
    old_kl, old_bop = E.distillation_kl, E.bop_step
    old_observe = E.SaturationSampler.observe

    def check_eval(model):
        assert not any(isinstance(m, E.TeacherLoRAParam) for m in model.modules())
        assert all(getattr(m, 'forward_keep', None) is None for m in E.fsa_modules(model))
        evaluations.append(1)

    def tf(model, *args, **kwargs):
        check_eval(model)
        return old_tf(model, *args, **kwargs)

    def nll(model, *args, **kwargs):
        check_eval(model)
        return old_nll(model, *args, **kwargs)

    def logits(model, *args, **kwargs):
        teachers = [m for m in model.modules() if isinstance(m, E.TeacherLoRAParam)]
        if teachers:
            forwards.append((all(t.enabled for t in teachers), E.FSAParam.enabled))
            assert all(not p.requires_grad for t in teachers for p in t.parameters())
        return old_logits(model, *args, **kwargs)

    def kl(*args, **kwargs):
        weights.append(1)
        return old_kl(*args, **kwargs)

    def observe(s, fi, logits, gold, step):
        samples.append((fi, step))
        return old_observe(s, fi, logits, gold, step)

    def bop(mods, gamma, tau, budgeted, **kwargs):
        birth_taus.append((tau, kwargs['birth_tau']))
        assert all(m.age.dtype == torch.int16 and not hasattr(m, 'nbirth') for m in mods)
        assert all(m.forward_keep is not None for m in mods)
        return old_bop(mods, gamma, tau, budgeted, **kwargs)

    monkeypatch.setattr(E, 'tf_eval', tf)
    monkeypatch.setattr(E, 'nll_windows', nll)
    monkeypatch.setattr(E, 'logits_of', logits)
    monkeypatch.setattr(E, 'distillation_kl', kl)
    monkeypatch.setattr(E.SaturationSampler, 'observe', observe)
    monkeypatch.setattr(E, 'bop_step', bop)
    flags = ['--teacher-patch', str(teacher_path), '--distill-mu', '.3', '--distill-anneal',
             '--bind-loss', '.2', '--bind-every', '1', '--saturation-sampling', '--aug',
             '--retain-evicted', '--swap-margin', '.1', '--swap-cap', '.5', '--death-age-mult',
             '--patch-dropout', '.25', '--tau-noise', '.1', '--reheat-every', '1', '--reheat-frac', '.5',
             '--no-churn', '--sparse-state', '--pair-birth']
    result, patch = run(['--steps', '3', *flags], 'c3_combined')
    assert evaluations and weights and samples
    assert (True, False) in forwards and (False, True) in forwards
    assert len(birth_taus) == 3 and any(a != b for a, b in birth_taus)
    for step in range(3):
        counts = Counter(fi for fi, s in samples if s == step)
        assert len(counts) == 3 and set(counts.values()) == {3}
    for row in result['trajectory']:
        assert row['sat_saturated'] + row['sat_unsaturated'] == 6
        assert row['sat_phrasings'] == row['step'] * 9
    assert result['keep_live'] and result['revoked']['logits_bit_exact']
    assert 'lora' not in result['storage'] and result['lora_params'] == 0
    state = torch.load(patch, weights_only=False)
    assert all(e['kind'] == 'fsa' for e in state['entries'].values())
    assert all(set(e['val'].unique().tolist()) <= {-1, 1} for e in state['entries'].values())
    loaded, _ = run(['--steps', '0', '--sparse-state', '--load-patch', str(patch)], 'c3_loaded')
    assert loaded['patched'] == result['patched'] and loaded['revoked']['logits_bit_exact']


def test_main_saturated_ce_is_masked_but_phrasings_still_forwarded(cpu_harness, monkeypatch):
    run, _ = cpu_harness
    seen, empty_losses = [], []
    old_ce = E.F.cross_entropy

    def observe(s, fi, logits, gold, step):
        seen.append(step)
        s.history[fi] = [True, True]
        s.last[fi] = step

    def ce(logits, targets, *args, **kwargs):
        if logits.shape[0] == 0:
            empty_losses.append(1)
        return old_ce(logits, targets, *args, **kwargs)

    monkeypatch.setattr(E.SaturationSampler, 'observe', observe)
    monkeypatch.setattr(E.F, 'cross_entropy', ce)
    result, _ = run(['--steps', '2', '--fact-batch', '0', '--saturation-sampling'], 'c3_masked')
    assert Counter(seen) == {0: 18, 1: 18}
    assert len(empty_losses) == 9  # all second-step microbatches still forwarded
    assert result['revoked']['logits_bit_exact']


@pytest.mark.parametrize('sparse', [False, True])
def test_joint_pairs_match_exhaustive_rule_with_thinned_support_and_row_tau(sparse):
    from itertools import product
    W = torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(80, 1)
    gen = torch.Generator().manual_seed(32)
    mask = E.pair8_forced_mask(W) & (torch.rand(W.shape, generator=gen) > .2)
    m = (E.SparseFSAParam if sparse else E.FSAParam)(W, mask, budgeted=True)
    dense_mag = torch.rand(W.shape, generator=gen) * 2
    dense_q = (torch.rand(W.shape, generator=gen) > .9).float() * mask
    dense_allowed = mask & (torch.rand(W.shape, generator=gen) > .1)
    dense_tau = torch.linspace(.5, 1.5, W.shape[0])[:, None].expand_as(W)
    mag, q, allowed, tau = dense_mag, dense_q, dense_allowed, dense_tau
    if sparse:
        mag, q, allowed, tau = [v.flatten()[m.idx.long()] for v in (mag, q, allowed, tau)]
    result = E.pair_birth_mask(m, q, mag, tau, allowed)
    actual = m._dense(result).bool() if sparse else result
    for r in range(W.shape[0]):
        for col in range(0, W.shape[1], 2):
            mags, taus = dense_mag[r, col:col + 2], dense_tau[r, col:col + 2]
            writable = dense_allowed[r, col:col + 2]
            dormant = not dense_q[r, col:col + 2].any()
            patterns = []
            for pattern in product((0, -1, 1), repeat=2):
                nonzero = torch.tensor(pattern).abs().bool()
                if not nonzero.any():
                    patterns.append(torch.tensor(pattern))
                elif (dormant and bool(writable[nonzero].all()) and
                      bool((mags[nonzero] > .5 * taus[nonzero]).all()) and
                      float(mags[nonzero].sum()) > .75 * float(taus[nonzero].sum())):
                    patterns.append(torch.tensor(pattern))
            best = min(patterns, key=lambda d: float(mags @ d.float()))
            assert torch.equal(actual[r, col:col + 2], best != 0)


def test_swap_margin_uses_population_std_and_global_incumbents():
    mods = [E.FSAParam(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), budgeted=True) for _ in range(2)]
    for m in mods:
        m.support.zero_()
    mods[0].support[0, 0] = True
    scores = [torch.tensor([[1., 1.]]), torch.tensor([[2., 2.]])]
    # Writable population std is .5, so gap=1 beats X=1.9 but fails X=2.0 strictly.
    E.refresh_support(mods, 1, scores, swap_margin=2)
    assert int(mods[0].support.sum()) == 1 and not mods[1].support.any()
    E.refresh_support(mods, 1, scores, swap_margin=1.9, swap_limit=2)
    assert not mods[0].support.any() and int(mods[1].support.sum()) == 1


def test_main_c3_age_reheat_adam_switch_accumulation_and_loaded_patch(cpu_harness, monkeypatch):
    run, _ = cpu_harness
    steps = []
    old_advance = E.advance_step_age

    def advance(mods):
        before = [m.age.clone() for m in mods]
        old_advance(mods)
        for m, prev in zip(mods, before):
            live = m._quant(m.M) * m.support != 0
            assert torch.equal(m.age[live], prev[live] + 1)
        steps.append(1)

    result, patch = run(['--steps', '3', '--no-churn', '--sparse-state', '--death-age-mult',
                          '--reheat-every', '1', '--reheat-frac', '1', '--adam-phase', '1'], 'c3_age_adam')
    assert result['revoked']['logits_bit_exact']
    monkeypatch.setattr(E, 'advance_step_age', advance)
    result, _ = run(['--steps', '4', '--no-churn', '--sparse-state', '--death-age-mult',
                      '--load-patch', str(patch), '--budget', '0', '--bop-accum', '2'], 'c3_age_accum')
    assert len(steps) == 4 and result['revoked']['logits_bit_exact']

# ---------------------------------------------------------------------------------------------- C6
from concurrent.futures import ThreadPoolExecutor


def _c6_linear(sparse=False, budgeted=False):
    import torch.nn.utils.parametrize as P
    linear = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(2, 1))
    linear.weight.requires_grad_(False)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    param = cls(linear.weight, E.pair8_forced_mask(linear.weight), budgeted=budgeted)
    if param.support is not param.mask:
        param.support.copy_(param.mask)
    P.register_parametrization(linear, "weight", param)
    return linear, param


def _c6_args(*extra):
    return E.build_parser().parse_args(["--method", "fsa_forced", "--out", "unused", *extra])


def test_c6_defaults_and_validation():
    a = _c6_args()
    assert a.base_cache is None and a.wake_steps == 0
    assert (a.wake_units, a.wake_margin, a.wake_lambda, a.wake_lr) == (16, 1., 1., 1e-3)
    E.validate_wake_args(a)
    for flags in (("--wake-steps", "-1"), ("--wake-units", "0"), ("--wake-margin", "nan"),
                  ("--wake-lambda", "-1"), ("--wake-lr", "0"), ("--wake-steps", "1")):
        with pytest.raises(SystemExit):
            E.validate_wake_args(_c6_args(*flags))
    a = _c6_args("--wake-steps", "1", "--optim", "bop", "--anchor", "general.json", "--people", "2",
             "--facts-per-person", "2", "--shared-attrs", "2")
    E.validate_wake_args(a)


def test_cache_roundtrip_tensors_nll_lists_and_none(tmp_path, capsys):
    original = dict(base_lp_t=[torch.randn(3, 7), torch.ones(2, 7, dtype=torch.bfloat16)],
                    base_logits0=torch.randn(1, 4, 7), base_big=[1.123456789123, 2.234567891234],
                    base_valw=None, base_extra={"extra": [3.345678912345]},
                    base_acq={"train": {"exact_match": .2}}, base_routing=None)
    E.cached_base(tmp_path, "key", lambda: original, "cpu")
    actual = E.cached_base(tmp_path, "key", lambda: pytest.fail("cache hit evaluated the model"), "cpu")
    for key in ("base_big", "base_valw", "base_extra", "base_acq", "base_routing"):
        assert actual[key] == original[key] and type(actual[key]) is type(original[key])
    for before, after in zip(original["base_lp_t"], actual["base_lp_t"]):
        assert torch.equal(before, after) and before.dtype == after.dtype and before.shape == after.shape
    assert torch.equal(original["base_logits0"], actual["base_logits0"])
    state = json.loads((tmp_path / "key.json").read_text())
    payload = torch.load(tmp_path / state["tensors"], weights_only=True)
    assert any(t.dtype == torch.float64 for t in payload.values())
    assert "BASE-CACHE hit key" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_cache_writers_publish_consistent_manifests(tmp_path):
    def write(i):
        return E.cached_base(tmp_path, "shared", lambda: dict(number=i, tensor=torch.full((i + 1,), i)), "cpu")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(4)))
    loaded = E.cached_base(tmp_path, "shared", lambda: pytest.fail("missing cache"), "cpu")
    assert loaded["tensor"].shape == (loaded["number"] + 1,)
    assert (loaded["tensor"] == loaded["number"]).all()
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_key_tracks_inputs_but_ignores_training_flags(tmp_path, monkeypatch):
    for name in ("heldout-tokens.json", "tokenizer.json", "eval.json", "panel.json"):
        (tmp_path / name).write_text("[1, 2, 3]")
    a = _c6_args("--eval-set", str(tmp_path / "eval.json"), "--extra-eval-sets", str(tmp_path / "panel.json"))
    facts = E.make_world()
    key = lambda: E.base_cache_key(a, facts, [0, 2], [1], data=tmp_path, export=tmp_path / "export")
    first = key()
    a.steps, a.optim, a.budget, a.lr = 99, "bop", 100, .02
    assert key() == first
    for field, value in (("gen_bs", 2), ("gen_stop_k", 1), ("gen_max_new", 1), ("aug", True),
                         ("val_windows", 3), ("fast_moe", "batched"), ("routing_diag", True)):
        old = getattr(a, field)
        setattr(a, field, value)
        assert key() != first
        setattr(a, field, old)
    assert E.base_cache_key(a, facts, [0], [1], data=tmp_path, export=tmp_path / "export") != first
    assert E.base_cache_key(a, facts, [0, 2], [1], data=tmp_path, export=tmp_path / "other") != first
    for name in ("heldout-tokens.json", "tokenizer.json", "eval.json", "panel.json"):
        path = tmp_path / name
        original = path.read_text()
        path.write_text("[9]")
        assert key() != first
        path.write_text(original)
    monkeypatch.setattr(E, "_CLOSERS", "changed")
    assert key() != first


def test_cache_version_follows_decorated_eval_helpers(monkeypatch):
    original = E.base_eval_version()

    def changed_generate(*a, **kw):
        return []
    changed_generate.__module__ = E.__name__
    monkeypatch.setattr(E, "greedy_generate", changed_generate)
    assert E.base_eval_version() != original


def test_hinge_has_correct_margins_general_weight_and_gradient():
    z = torch.tensor([[0., -2.], [2., 0.]], requires_grad=True)
    pos = torch.eye(2, dtype=torch.bool)
    neg = ~pos
    general = torch.tensor([[-1., 2.]], requires_grad=True)
    loss = E.wake_hinge(z, pos, neg, torch.ones(2), general, weight=3.)
    assert loss.item() == pytest.approx(11.5)
    loss.backward()
    torch.testing.assert_close(z.grad, torch.tensor([[-1., 0.], [3., -1.]]))
    torch.testing.assert_close(general.grad, torch.tensor([[0., 6.]]))
    assert E.wake_hinge(torch.tensor([[2., -2.], [-2., 2.]]), pos, neg, torch.ones(2)) == 0


@pytest.mark.parametrize("mode", ["off", "exact", "batched"])
def test_causal_capture_preserves_dispatch_and_last_token_gradient(mode):
    from test_fast_moe import FakeMoE
    import fast_moe
    model = FakeMoE()
    if mode != "off":
        fast_moe.install(model, mode=mode)
    dispatch = model._forward_frozen_exact_grouped
    x = torch.randn(1, 5, 8, requires_grad=True)
    expected = set(model.router(x[0, -1]).topk(model.top_k).indices.tolist())
    with E.wake_prompt_inputs(model, [0], detach=False) as records:
        model(x)
        last, routes = records[0]
        assert torch.equal(last, x[0, -1]) and routes == expected
        last.sum().backward()
    assert model._forward_frozen_exact_grouped == dispatch
    assert not x.grad[0, :-1].any() and (x.grad[0, -1] == 1).all()


def test_selectivity_keeps_negative_potential_and_binding_labels():
    facts = [E.Fact(0, "A", "her", "she", "color", "red"),
             E.Fact(1, "B", "his", "he", "color", "blue"),
             E.Fact(0, "A", "her", "she", "food", "rice")]
    z = torch.tensor([[0., 0., 0.], [10., 2., 4.], [3., 8., 9.]])
    rows, owners = E.wake_select(z, facts, [0, 1, 2], [True, False, True], torch.ones(3, dtype=torch.bool), 2)
    assert rows == [1, 2] and owners == [0, 0]
    pos, neg = E.wake_labels(facts, [0, 1, 2], owners, "cpu")
    assert pos[:, 0].tolist() == [True, False, False]
    assert neg[:, 0].tolist() == [False, True, True]
    assert not (pos & neg).any()


@pytest.mark.parametrize("sparse", [False, True])
def test_bop_and_support_refresh_never_change_frozen_rows(sparse):
    lin, m = _c6_linear(sparse, budgeted=True)
    row = m.rows() if sparse else torch.arange(2)[:, None].expand_as(m.M)
    selected = (row == 0) & m.mask
    with torch.no_grad():
        m.M[selected] = 1
        # Keep a zero in the frozen row: it must never become a Bop birth.
        m.M.flatten()[selected.flatten().nonzero().flatten()[0]] = 0
    E.freeze_wake_rows(m, [0])
    original = m.M.clone()
    m.bop_ema = torch.zeros_like(m.M)
    m.M.grad = torch.where(selected, torch.ones_like(m.M) * 100, -torch.ones_like(m.M) * 100)
    E.bop_step([m], 1., .1, True)
    assert torch.equal(m.M[selected], original[selected])
    assert (m.M[(row == 1) & m.mask] != 0).any()
    m.score.fill_(100)
    E.refresh_support([m], 5, keep_live=False)
    E.refresh_support([m], 5, keep_live=False, swap_margin=.01, swap_limit=10)
    assert torch.equal(m.M[selected], original[selected]) and int(m.support.sum()) <= 5
    assert int(E.pinned_slots(m).sum()) == 3
    assert not E.writable_slots(m)[selected].any()
    m.M.grad = None
    lin(torch.randn(3, 8)).sum().backward()
    assert not m.M.grad[selected].any()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("budget", [0, 4, 5])
def test_mls_excludes_frozen_rows_and_reserves_their_budget(sparse, budget):
    lin, m = _c6_linear(sparse)
    current = torch.zeros(2, 8)
    current[0, 4:] = 1
    with torch.no_grad():
        m.M.copy_(current.flatten()[m.idx.long()] if sparse else current)
    E.freeze_wake_rows(m, [0])
    optimizer = E.WakeMLSOptimizer([(lin, m)], eta=2., gen_lambda=0., gamma=1e-6, budget=budget)
    optimizer.general[m] = torch.zeros(8), torch.empty(8, 0)
    optimizer.facts[m] = [(torch.eye(8), -torch.ones(8, 2))]
    mask = m.mask
    transaction, info = optimizer.propose()
    assert optimizer.budget == budget and m.mask is mask
    transaction.apply()
    assert torch.equal(m.codes()[0], current[0])
    if budget:
        assert int((m.codes() != 0).sum()) <= budget
    else:
        assert (m.codes()[1] != 0).any()
    transaction.rollback()
    assert torch.equal(m.codes(), current)


@pytest.mark.parametrize("sparse", [False, True])
def test_wake_stage_only_writes_selected_up_rows_and_is_causal(sparse, capsys):
    model = tiny()
    for li, moe in enumerate(model.moes):
        for ei, expert in enumerate(moe.experts):
            for lin in (expert.up_proj, expert.down_proj):
                lin.weight.data.copy_(_pair8_tiny_weight(li * 4 + ei, *lin.weight.shape))
    E.attach(model, [0, 1], "fsa_forced", 0, sparse=sparse, budgeted=True)
    facts = E.make_world(0, 2, 2, 2)
    a = _c6_args("--wake-steps", "6", "--wake-units", "2", "--wake-lr", ".2", "--budget", "9")
    anchor = torch.tensor([[BOS_T] + [40] * 15] * 2)
    items = E.build_items(TOK, facts, E.TRAIN_T[:1], BOS_T)
    with E.wake_prompt_inputs(model, [0, 1]) as records:
        E.logits_of(model, torch.tensor([items[0]["p"]]))
        captured = records[0][0]
    x = model.emb(torch.tensor(items[0]["p"])).mean(0)
    torch.testing.assert_close(captured, x)
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    report = E.run_wake_stage(model, TOK, facts, E.TRAIN_T[:1], BOS_T, [0, 1], anchor, a)
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)
    assert E.patch_stats(E.fsa_modules(model))[0] <= 9
    for name, lin in E.patched_linears(model):
        m = lin.parametrizations.weight[0]
        if name.endswith("down_proj"):
            assert not m.M.any()
        elif name in report["detectors"]:
            assert not m.M[~E.frozen_slots(m).expand_as(m.M)].any()
            assert len(report["detectors"][name]["rows"]) <= 2
            assert not lin._forward_pre_hooks
    assert "WAKE" in capsys.readouterr().out


def test_main_cache_miss_hit_skips_base_forward_work(cpu_harness, tmp_path, monkeypatch, capsys):
    run, paths = cpu_harness
    (tmp_path / "tokenizer.json").write_text("{}")
    cache = tmp_path / "cache"
    flags = ["--steps", "0", "--base-cache", str(cache), "--extra-eval-sets", paths["extra"], "--routing-diag"]
    first, _ = run(flags, "miss")
    monkeypatch.setattr(E, "evaluate_base", lambda *a, **kw: pytest.fail("hit recomputed baseline"))
    # Keep the version for the saved implementation while replacing its evaluator with a sentinel.
    key = first["base_cache_key"]
    monkeypatch.setattr(E, "base_cache_key", lambda *a, **kw: key)
    second, _ = run(flags, "hit")
    assert first["base"] == second["base"]
    assert first["patched"] == second["patched"]
    assert second["revoked"]["logits_bit_exact"]
    assert f"BASE-CACHE hit {key}" in capsys.readouterr().out


@pytest.mark.parametrize("optim", ["bop", "mls"])
def test_main_wake_codes_survive_training_and_reload(cpu_harness, monkeypatch, optim):
    run, paths = cpu_harness
    saved = {}
    original = E.run_wake_stage

    def record(model, *a, **kw):
        result = original(model, *a, **kw)
        saved.update({name: lin.parametrizations.weight[0].codes().clone()
                      for name, lin in E.patched_linears(model) if name in result["detectors"]})
        return result
    monkeypatch.setattr(E, "run_wake_stage", record)
    flags = ["--anchor", paths["anchor"], "--anchor-len", "16", "--optim", optim]
    if optim == "mls":
        flags += ["--bop-tau-abs", "0"]
    warmup = ["--adam-phase", "1", "--wd", ".1"] if optim == "bop" else []
    result, patch = run(["--steps", "3", "--wake-steps", "6", "--wake-lr", ".2", "--wake-units", "2",
                         *warmup, *flags], "wake")
    state = torch.load(patch, weights_only=False)
    assert result["wake"]["detectors"] and saved
    assert any(codes.any() for codes in saved.values())
    for name, original_codes in saved.items():
        entry = state["entries"][name]
        final = torch.zeros(entry["shape"])
        final.flatten()[entry["idx"].long()] = entry["val"].float()
        rows = result["wake"]["detectors"][name]["rows"]
        assert torch.equal(final[rows], original_codes[rows])
    assert result["revoked"]["logits_bit_exact"]
    loaded, _ = run(["--steps", "0", "--load-patch", str(patch), *flags], "reload")
    assert result["patched"] == loaded["patched"] and loaded["wake"] == result["wake"]


# C10: LoRA teacher fade-out homotopy.
@pytest.mark.parametrize("shape", ["linear", "cos"])
def test_teacher_fade_schedule(shape):
    scale = lambda step: E.teacher_fade_scale(.5, shape, step, 100)
    assert scale(0) == 1
    assert scale(25) == pytest.approx(.5)
    assert scale(50) == scale(100) == 0
    assert scale(10) == pytest.approx(.8 if shape == "linear" else (1 + math.cos(.2 * math.pi)) / 2)
    assert E.teacher_fade_scale(0, shape, 0, 100) == 0
    assert E.teacher_fade_scale(.5, shape, 0, 0) == 0
    assert E.teacher_fade_scale(1, shape, 0, 1) == 1
    assert E.teacher_fade_scale(1, shape, 1, 1) == 0
    assert E.teacher_fade_scale(2, shape, 100, 100) == pytest.approx(.5)


def _teacher_state(entries):
    return {"format": "e2e-fsa-patch-v1", "meta": {"method": "lora", "layers": [0], "rank": 1},
            "entries": {name: {"kind": "lora", "A": torch.ones(1, 8), "B": torch.ones(2, 1)}
                        for name in entries}}


def _linear_student(sparse=False):
    lin = nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1., -1., 1., -1., 0., 0., 0., 0.]]).repeat(2, 1))
    lin.weight.requires_grad_(False)
    cls = E.SparseFSAParam if sparse else E.FSAParam
    pm = cls(lin.weight, E.pair8_forced_mask(lin.weight))
    E.P.register_parametrization(lin, "weight", pm)
    with torch.no_grad():
        pm.M.fill_(1)
    return lin, pm


def test_scaled_teacher_contribution_frozen_factors_and_kl_restoration():
    lin, pm = _linear_student()
    model = nn.Sequential(lin)
    teachers = E.attach_teacher(model, _teacher_state(["0"]))
    t = teachers[0]
    x = torch.arange(8.).reshape(1, 8)
    student = model(x).detach()
    for scale in (1., .5, .25, 0.):
        t.enabled, t.scale = True, scale
        assert torch.allclose(model(x), student + scale * (x @ t.delta_w().T))
    t.scale = .25
    with pytest.raises(RuntimeError, match="restore"):
        with E.teacher_forward(teachers):
            assert t.enabled and t.scale == 1 and not E.FSAParam.enabled
            assert not model(x).requires_grad
            raise RuntimeError("restore")
    assert t.enabled and t.scale == .25 and E.FSAParam.enabled
    model(x).sum().backward()
    assert pm.M.grad is not None
    assert t.A.grad is t.B.grad is None
    t.enabled = False
    E.remove_teacher(model)
    assert torch.equal(model(x), student)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("from_output", [False, True])
def test_penalty_excludes_enabled_teacher_and_teacher_only_linears(sparse, from_output):
    lin, pm = _linear_student(sparse)
    model = nn.ModuleDict({"student": lin, "teacher_only": nn.Linear(8, 2, bias=False)})
    teachers = E.attach_teacher(model, _teacher_state(model.keys()))
    penalty = E.LocalOutputPenalty(model, from_output=from_output)
    x = torch.arange(16.).reshape(2, 8).requires_grad_()
    base = lin.parametrizations.weight.original.detach()
    expected = (x @ pm.delta_w().T).square().sum() / (x.detach() @ base.T).square().sum()
    expected_grads = torch.autograd.grad(expected, (pm.M, x))
    for scale in (1., .5, 0.):
        for t in teachers:
            t.enabled, t.scale = True, scale
        penalty.start()
        lin(x)
        model["teacher_only"](x)
        actual = penalty.stop()
        assert len(penalty.num) == 1
        assert torch.allclose(actual, expected)
        for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (pm.M, x)), expected_grads):
            assert torch.allclose(actual_grad, expected_grad, atol=1e-6)
        penalty.end()
    for h in penalty.h:
        h.remove()


@pytest.mark.parametrize("flags, message", [
    (["--teacher-fade", ".5"], "teacher-patch"),
    (["--teacher-fade", "-1"], "teacher-fade"),
    (["--teacher-fade", "nan"], "teacher-fade"),
    (["--teacher-fade", "inf"], "teacher-fade"),
    (["--teacher-fade", ".5", "--teacher-patch", "unused.pt", "--method", "lora"], "forced-mask"),
])
def test_fade_validation(flags, message):
    args = E.build_parser().parse_args(["--method", "fsa_forced", "--out", "/private/tmp/fade-unused.json", *flags])
    with pytest.raises(SystemExit, match=message):
        E.validate_c3_args(args)


@pytest.mark.parametrize("shape, mu", [("linear", 0.), ("cos", .3)])
def test_main_fade_all_training_losses_pure_validation_and_export(cpu_harness, monkeypatch, capsys, shape, mu):
    run, paths = cpu_harness
    _, teacher_path = run(["--method", "lora", "--optim", "adam", "--budget", "0", "--rank", "1",
                           "--steps", "2", "--bop-tau-abs", "0"], "fade_teacher")
    training, diagnostics, pure_evals, kls, bindings = [], [], [], [], []
    old_logits, old_tf, old_nll, old_gen = E.logits_of, E.tf_eval, E.nll_windows, E.gen_eval
    old_kl = E.distillation_kl
    old_bind = E.binding_loss_backward

    def teachers(model):
        return [t for t in model.modules() if isinstance(t, E.TeacherLoRAParam)]

    def logits(model, ids):
        ts = teachers(model)
        if ts and torch.is_grad_enabled():
            assert E.FSAParam.enabled and all(t.enabled for t in ts)
            assert all(not p.requires_grad for t in ts for p in t.parameters())
            # Anchor/rehearsal tokens distinguish those forwards from facts/binding.
            kind = "anchor" if bool((ids[:, 1:] == 40).all()) else (
                "rehearsal" if bool((ids[:, 1:] == 50).all()) else "fact_or_bind")
            training.append((kind, ts[0].scale))
        return old_logits(model, ids)

    def tf(model, *args, **kwargs):
        assert not teachers(model)
        pure_evals.append("tf")
        return old_tf(model, *args, **kwargs)

    def nll(model, *args, **kwargs):
        assert not teachers(model)
        pure_evals.append("nll")
        return old_nll(model, *args, **kwargs)

    def gen(model, *args, **kwargs):
        ts = teachers(model)
        if ts:
            assert E.FSAParam.enabled and all(t.enabled for t in ts)
            diagnostics.append(ts[0].scale)
        else:
            pure_evals.append("gen")
        return old_gen(model, *args, **kwargs)

    def kl(*args, **kwargs):
        kls.append(1)
        return old_kl(*args, **kwargs)

    def bind(model, *args, **kwargs):
        ts = teachers(model)
        assert ts and E.FSAParam.enabled and all(t.enabled for t in ts)
        bindings.append(ts[0].scale)
        return old_bind(model, *args, **kwargs)

    monkeypatch.setattr(E, "logits_of", logits)
    monkeypatch.setattr(E, "tf_eval", tf)
    monkeypatch.setattr(E, "nll_windows", nll)
    monkeypatch.setattr(E, "gen_eval", gen)
    monkeypatch.setattr(E, "distillation_kl", kl)
    monkeypatch.setattr(E, "binding_loss_backward", bind)
    result, patch = run(["--steps", "4", "--eval-every", "1", "--teacher-patch", str(teacher_path),
                         "--teacher-fade", ".5", "--teacher-fade-shape", shape, "--distill-mu", str(mu),
                         "--anchor", paths["anchor"], "--anchor-len", "16", "--anchor-bs", "1",
                         "--rehearsal-set", paths["rehearsal"], "--rehearsal-frac", ".1",
                         "--bind-loss", ".2", "--bind-every", "1", "--local-lambda", "1",
                         "--drift-lambda", ".1", "--sparse-state", "--expert-checkpoint"], "fade_student")
    assert diagnostics == pytest.approx([1., .5, 0., 0.])
    for kind in ("anchor", "rehearsal", "fact_or_bind"):
        assert {scale for k, scale in training if k == kind} == {1., .5, 0.}
    assert pure_evals and bool(kls) == bool(mu)
    assert set(bindings) == {1., .5, 0.}
    for row, scale in zip(result["trajectory"], diagnostics):
        assert row["teacher_scale"] == scale
        assert "val_gen_with_teacher" in row
        if scale == 0:
            assert row["val_gen_with_teacher"] == row["val_gen"]
    assert "teacher_scale 1.000000" in capsys.readouterr().out
    assert result["teacher_fade"] == .5 and result["teacher_fade_shape"] == shape
    assert result["revoked"]["logits_bit_exact"] and result["lora_params"] == 0
    assert all(e["kind"] == "fsa" for e in torch.load(patch, weights_only=False)["entries"].values())


def test_fade_off_preserves_patch_metrics_and_rng(cpu_harness):
    run, _ = cpu_harness
    _, teacher_path = run(["--method", "lora", "--optim", "adam", "--budget", "0", "--rank", "1",
                           "--steps", "2", "--bop-tau-abs", "0"], "fade_off_teacher")
    flags = ["--steps", "3", "--teacher-patch", str(teacher_path), "--distill-mu", ".3"]
    default, patch_a = run(flags, "fade_default")
    rng = torch.get_rng_state()
    off, patch_b = run([*flags, "--teacher-fade", "0", "--teacher-fade-shape", "cos"], "fade_off")
    assert torch.equal(torch.get_rng_state(), rng)
    assert off["patched"] == default["patched"]
    assert all("val_gen_with_teacher" not in row for row in off["trajectory"])
    a, b = (torch.load(p, weights_only=False)["entries"] for p in (patch_a, patch_b))
    for name in a:
        assert torch.equal(a[name]["idx"], b[name]["idx"])
        assert torch.equal(a[name]["val"], b[name]["val"])
