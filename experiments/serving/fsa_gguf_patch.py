#!/usr/bin/env python3
"""Write Free-Slot Adaptation patches into (and out of) a packed pair8 `t9` Parallax GGUF.

A patch is a list of slots, one per filled weight:
    layer   uint8   MoE layer index (0..31), the GGUF's `moe.<layer>`
    proj    uint8   0 = up projection, 1 = down projection
    expert  uint16  routed expert (0..127)
    index   uint32  row * in_features + col in the [out_features, in_features] weight
    sign    int8    +1 or -1; the slot's value is sign * alpha_row, alpha_row = the row's stored scale

Only a *free slot* can be filled: a zero whose pair partner is nonzero, so the pair is already active and
the 8-weight block keeps at most two active pairs. Every touched block is decoded, changed, re-encoded and
checked against the 417-word pair8 alphabet; the tool refuses the whole patch before writing anything if
any slot is not free (apply) or does not hold the patch's sign (revoke). Revoke zeroes exactly those slots,
so apply followed by revoke restores the original bytes.

Subcommands:
    free-slots  GGUF --layers 14,15                     count free slots per tensor
    synth       GGUF OUT.npz --slots N --layers 14,15   uniform random patch over the free slots
    apply       GGUF PATCH.npz [--out NEW.gguf]         fill the slots (in place unless --out)
    revoke      GGUF PATCH.npz [--out NEW.gguf]         zero the slots
    verify      BASE PATCHED PATCH.npz                  independent check with the runtime's own t9 decoder
    bench       GGUF A.npz B.npz                        time apply/revoke on loaded tensors and on the mmap'd file

The GGUF layout (from the converter): `moe.<L>.up.codes` is [128, 2304, 54] bytes, `moe.<L>.down.codes`
is [128, 384, 324]; each row is a run of 9-byte groups, one group = 8 blocks x 9-bit state, state = index
into the 417 legal 16-bit words (two bits per weight, 0 -> 0, 1 -> +1, 2 -> -1, ascending word order).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:  # use the llama.cpp checkout's gguf-py
    sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP", "llama.cpp")) / "gguf-py"))
    import gguf

PROJ = ("up", "down")
FORMAT = "fsa-pair8-patch:1"


def alphabet():
    """The 417 legal pair8 words in ascending order (state -> word) and the inverse table (word -> state)."""
    words = np.arange(65536, dtype=np.uint32)
    digits = (words[:, None] >> (2 * np.arange(8))) & 3
    valid = (digits != 3).all(axis=1)
    valid &= (digits.reshape(-1, 4, 2) != 0).any(axis=2).sum(axis=1) <= 2
    raw = words[valid].astype(np.uint16)
    assert raw.size == 417
    codes = np.full(65536, 0xFFFF, dtype=np.uint16)
    codes[raw] = np.arange(417, dtype=np.uint16)
    return raw, codes


WORDS, STATES = alphabet()


def unpack_groups(g):
    """[n, 9] uint8 groups -> [n, 8] 9-bit states."""
    g = g.astype(np.uint16)
    s = np.empty((g.shape[0], 8), dtype=np.uint16)
    for lane in range(8):
        s[:, lane] = ((g[:, lane] | (g[:, lane + 1] << 8)) >> lane) & 511
    return s


def pack_groups(s):
    """[n, 8] 9-bit states -> [n, 9] uint8 groups."""
    g = np.zeros((s.shape[0], 9), dtype=np.uint8)
    for lane in range(8):
        g[:, lane] |= (s[:, lane] << lane).astype(np.uint8)
        g[:, lane + 1] |= (s[:, lane] >> (8 - lane)).astype(np.uint8)
    return g


class Model:
    """Byte offsets and geometry of the routed-expert code tensors of a t9 GGUF."""

    def __init__(self, path):
        r = gguf.GGUFReader(path)
        fmt = r.fields["parallax.expert_format"].contents()
        if fmt != "t9" or r.fields["parallax.expert_alphabet"].contents() != "paired4of8-417-lex16-v1":
            raise SystemExit(f"{path}: expected a t9 paired4of8-417 GGUF, found {fmt}")
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.tensors = {}
        for t in r.tensors:
            if t.name.startswith("moe.") and t.name.endswith(".codes"):
                _, layer, proj, _ = t.name.split(".")
                ne, rows, nbytes = (int(x) for x in t.data.shape)
                cols = nbytes * 64 // 9
                self.tensors[int(layer), PROJ.index(proj)] = dict(
                    name=t.name, offset=int(t.data_offset), bytes=int(t.n_bytes), experts=ne, rows=rows, cols=cols)

    def geometry(self, layer, proj):
        try:
            return self.tensors[layer, proj]
        except KeyError:
            raise SystemExit(f"no expert tensor moe.{layer}.{PROJ[proj]}.codes")


def load_patch(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("format") != FORMAT:
        raise SystemExit(f"{path}: not an {FORMAT} file")
    p = {k: z[k] for k in ("layer", "proj", "expert", "index", "sign")}
    if not np.isin(p["sign"], (-1, 1)).all():
        raise SystemExit("patch signs must be +-1")
    return p, meta


def save_patch(path, p, meta):
    meta = dict(meta, format=FORMAT, slots=int(p["index"].size))
    np.savez_compressed(path, meta=json.dumps(meta), layer=p["layer"].astype(np.uint8), proj=p["proj"].astype(np.uint8),
                        expert=p["expert"].astype(np.uint16), index=p["index"].astype(np.uint32),
                        sign=p["sign"].astype(np.int8))


def plan(model, p):
    """Per tensor: sorted unique 9-byte groups touched, and for each slot (group row, lane, weight-in-block, digit)."""
    out = []
    key = p["layer"].astype(np.int64) * 2 + p["proj"]
    for k in np.unique(key):
        layer, proj = int(k // 2), int(k % 2)
        geo = model.geometry(layer, proj)
        sel = np.flatnonzero(key == k)
        e, idx = p["expert"][sel].astype(np.int64), p["index"][sel].astype(np.int64)
        if (e >= geo["experts"]).any() or (idx >= geo["rows"] * geo["cols"]).any():
            raise SystemExit(f"slot outside {geo['name']}")
        flat = e * geo["rows"] * geo["cols"] + idx  # rows never straddle a group (cols % 64 == 0)
        if np.unique(flat).size != flat.size:
            raise SystemExit(f"duplicate slots in {geo['name']}")
        block = flat // 8
        groups, inverse = np.unique(block // 8, return_inverse=True)
        digit = np.where(p["sign"][sel] > 0, 1, 2).astype(np.uint16)
        out.append(dict(geo=geo, groups=groups, row=inverse, lane=block % 8, pos=flat % 8, digit=digit))
    return out


def edit(buf, base, plans, revoke):
    """Apply (or revoke) all slots on a writable uint8 buffer whose byte 0 is file offset `base`.
    Validates everything first and only then writes. Returns the number of 9-byte groups rewritten."""
    staged = []
    for t in plans:
        off = t["geo"]["offset"] - base + t["groups"] * 9
        gidx = off[:, None] + np.arange(9)
        old = buf[gidx]
        states = unpack_groups(old)
        if (states >= 417).any():
            raise SystemExit(f"{t['geo']['name']}: invalid state in base tensor")
        words = WORDS[states]                                    # [g, 8] uint16
        w = words[t["row"], t["lane"]]
        shift = (2 * t["pos"]).astype(np.uint16)
        cur = (w >> shift) & 3
        partner = (w >> (2 * (t["pos"] ^ 1)).astype(np.uint16)) & 3
        if revoke:
            bad = (cur != t["digit"]) | (partner == 0)
            if bad.any():
                raise SystemExit(f"{t['geo']['name']}: {int(bad.sum())} slots do not hold the patch's value; refusing")
            np.bitwise_and.at(words, (t["row"], t["lane"]), ~(np.uint16(3) << shift))
        else:
            bad = (cur != 0) | (partner == 0)
            if bad.any():
                raise SystemExit(f"{t['geo']['name']}: {int(bad.sum())} slots are not free (nonzero, or in an "
                                 f"inactive pair); refusing")
            np.bitwise_or.at(words, (t["row"], t["lane"]), t["digit"] << shift)
        new_states = STATES[words]
        if (new_states == 0xFFFF).any():  # cannot happen for free slots; checked anyway
            raise SystemExit(f"{t['geo']['name']}: a patched block is not a legal pair8 word; refusing")
        staged.append((gidx, pack_groups(new_states), words))
    for gidx, groups, words in staged:
        buf[gidx] = groups
        if not np.array_equal(WORDS[unpack_groups(buf[gidx])], words):  # read back every rewritten block
            raise SystemExit("read-back mismatch after write")
    return sum(s[0].shape[0] for s in staged)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def edit_file(path, patch_path, revoke, out=None):
    if out:
        if Path(out).exists():
            raise SystemExit(f"{out} exists")
        shutil.copyfile(path, out)
        path = out
    model = Model(path)
    p, meta = load_patch(patch_path)
    t0 = time.perf_counter()
    plans = plan(model, p)
    mm = np.memmap(path, dtype=np.uint8, mode="r+")
    groups = edit(mm, 0, plans, revoke)
    mm.flush()
    del mm
    dt = time.perf_counter() - t0
    if model.path.stat().st_size != model.size:
        raise SystemExit("file size changed")
    return dict(file=str(path), action="revoke" if revoke else "apply", slots=int(p["index"].size), groups=int(groups),
                seconds=dt, file_bytes=model.size)


def free_mask(model, layer, proj):
    """Per tensor: boolean [experts*rows*cols] mask of free slots, and the weight-level nonzero count."""
    geo = model.geometry(layer, proj)
    raw = np.fromfile(model.path, dtype=np.uint8, count=geo["bytes"], offset=geo["offset"]).reshape(-1, 9)
    words = WORDS[unpack_groups(raw)].reshape(-1)                 # one word per 8-weight block
    digits = ((words[:, None] >> (2 * np.arange(8, dtype=np.uint16))) & 3).astype(np.uint8)
    partner = digits.reshape(-1, 4, 2)[:, :, ::-1].reshape(-1, 8)
    return ((digits == 0) & (partner != 0)).reshape(-1), int((digits != 0).sum())


def cmd_free(a):
    model = Model(a.gguf)
    rows, total, weights = [], 0, 0
    for layer in a.layers:
        for proj in (0, 1):
            geo = model.geometry(layer, proj)
            m, nnz = free_mask(model, layer, proj)
            n = int(m.sum())
            rows.append(dict(tensor=geo["name"], weights=m.size, nonzero=nnz, free_slots=n))
            total += n
            weights += m.size
    print(json.dumps(dict(tensors=rows, free_slots=total, weights=weights, free_fraction=total / weights), indent=1))


def cmd_synth(a):
    model = Model(a.gguf)
    rng = np.random.default_rng(a.seed)
    parts, counts = [], []
    for layer in a.layers:
        for proj in (0, 1):
            m, _ = free_mask(model, layer, proj)
            pos = np.flatnonzero(m).astype(np.int64)
            parts.append((layer, proj, model.geometry(layer, proj), pos))
            counts.append(pos.size)
    total = sum(counts)
    pick = np.sort(rng.choice(total, size=a.slots, replace=False))
    edges = np.cumsum([0] + counts)
    L, P, E, I = [], [], [], []
    for (layer, proj, geo, pos), lo, hi in zip(parts, edges[:-1], edges[1:]):
        sel = pos[pick[(pick >= lo) & (pick < hi)] - lo]
        per = geo["rows"] * geo["cols"]
        L.append(np.full(sel.size, layer)); P.append(np.full(sel.size, proj))
        E.append(sel // per); I.append(sel % per)
    p = dict(layer=np.concatenate(L), proj=np.concatenate(P), expert=np.concatenate(E), index=np.concatenate(I),
             sign=rng.choice(np.array([-1, 1], dtype=np.int8), size=a.slots))
    save_patch(a.out, p, dict(kind="synthetic-uniform", seed=a.seed, layers=a.layers, free_slots_available=total))
    print(json.dumps(dict(out=str(a.out), slots=a.slots, free_slots_available=total, bytes=Path(a.out).stat().st_size)))


def cmd_edit(a, revoke):
    print(json.dumps(edit_file(a.gguf, a.patch, revoke, a.out)))


def cmd_verify(a):
    """Independent check of a patched GGUF against its base with the runtime repository's own t9 decoder
    (tools/parallax/pack_native.py), not this file's codec: same size; no changed byte outside the expert code
    tensors; every expert tensor decodes as legal pair8; the ternary weights differ from the base at exactly the
    patch's slots, each 0 -> sign, each inside a pair that was already active."""
    sys.path.insert(0, str(Path(os.environ.get("LLAMA_CPP", "llama.cpp")) / "tools" / "parallax"))
    from pack_native import alphabet as rt_alphabet, decode as rt_decode
    patterns, _ = rt_alphabet()
    base, new = Model(a.base), Model(a.patched)
    if base.size != new.size or base.tensors != new.tensors:
        raise SystemExit("layout changed")
    p, _ = load_patch(a.patch)
    B = np.memmap(a.base, dtype=np.uint8, mode="r")
    N = np.memmap(a.patched, dtype=np.uint8, mode="r")
    changed = np.flatnonzero(B != N)
    inside = np.zeros(changed.size, dtype=bool)
    for g in base.tensors.values():
        inside |= (changed >= g["offset"]) & (changed < g["offset"] + g["bytes"])
    if not inside.all():
        raise SystemExit(f"{int((~inside).sum())} changed bytes outside the expert code tensors")

    def ternary(buf, g):
        raw = np.asarray(buf[g["offset"]:g["offset"] + g["bytes"]]).view(np.int8).reshape(-1, 9)
        words = rt_decode(raw, patterns).view("<u2").reshape(-1)  # raises on any illegal state
        d = ((words[:, None] >> (2 * np.arange(8, dtype=np.uint16))) & 3).astype(np.int8)
        return np.where(d == 2, -1, d).reshape(-1)

    report = dict(changed_bytes=int(changed.size), tensors=[])
    key = p["layer"].astype(np.int64) * 2 + p["proj"]
    for (layer, proj), g in sorted(base.tensors.items()):
        sel = np.flatnonzero(key == layer * 2 + proj)
        if not sel.size:
            if ((changed >= g["offset"]) & (changed < g["offset"] + g["bytes"])).any():
                raise SystemExit(f"{g['name']} changed without patch slots")
            continue
        wb, wn = ternary(B, g), ternary(N, g)
        diff = np.flatnonzero(wb != wn)
        flat = p["expert"][sel].astype(np.int64) * g["rows"] * g["cols"] + p["index"][sel]
        order = np.argsort(flat)
        ok = (np.array_equal(diff, flat[order]) and (wb[diff] == 0).all()
              and np.array_equal(wn[diff], p["sign"][sel][order]) and (wb[diff ^ 1] != 0).all())
        if not ok:
            raise SystemExit(f"{g['name']}: changed weights do not match the patch's free slots and signs")
        report["tensors"].append(dict(tensor=g["name"], slots=int(sel.size), nonzero_before=int((wb != 0).sum()),
                                      nonzero_after=int((wn != 0).sum())))
    report["ok"] = True
    print(json.dumps(report))


def cmd_bench(a):
    """Patch timing on (1) the code tensors loaded into RAM and (2) the GGUF mmap'd in place (copy of it)."""
    model = Model(a.gguf)
    A, _ = load_patch(a.patch_a)
    B, _ = load_patch(a.patch_b)
    lo = min(t["offset"] for t in model.tensors.values())
    hi = max(t["offset"] + t["bytes"] for t in model.tensors.values())
    res = dict(repeats=a.repeats)

    def timed(fn):
        t = []
        for _ in range(a.repeats):
            s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
        return dict(median_s=float(np.median(t)), min_s=float(min(t)), max_s=float(max(t)))

    s = time.perf_counter(); pa, pb = plan(model, A), plan(model, B); res["plan_both_s"] = time.perf_counter() - s
    ram = np.fromfile(model.path, dtype=np.uint8, count=hi - lo, offset=lo)
    ref = hashlib.sha256(ram).hexdigest()
    res["ram"] = dict(
        apply_then_revoke_A=timed(lambda: (edit(ram, lo, pa, False), edit(ram, lo, pa, True))),
        cycle_apply_A_revoke_A_apply_B=timed(lambda: (edit(ram, lo, pa, False), edit(ram, lo, pa, True),
                                                       edit(ram, lo, pb, False), edit(ram, lo, pb, True))),
    )
    res["ram"]["restored_identical"] = hashlib.sha256(ram).hexdigest() == ref
    # the timed cycles also revoke B so every repeat starts from the base; single steps are timed below
    one = {}
    for name, pl, rv in (("apply_A", pa, False), ("revoke_A", pa, True), ("apply_B", pb, False), ("revoke_B", pb, True)):
        s = time.perf_counter(); edit(ram, lo, pl, rv); one[name] = time.perf_counter() - s
    res["ram"]["single_steps_s"] = one
    res["ram"]["restored_identical_after_steps"] = hashlib.sha256(ram).hexdigest() == ref
    if a.scratch:
        copy = Path(a.scratch)
        shutil.copyfile(model.path, copy)
        steps = {}
        for name, pl, rv in (("apply_A", pa, False), ("revoke_A", pa, True), ("apply_B", pb, False), ("revoke_B", pb, True)):
            s = time.perf_counter()
            mm = np.memmap(copy, dtype=np.uint8, mode="r+")
            edit(mm, 0, pl, rv)
            mm.flush()
            del mm
            steps[name] = time.perf_counter() - s
        res["mmap_file"] = dict(single_steps_s=steps, restored_sha256_identical=sha256(copy) == sha256(model.path))
        copy.unlink()
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    layers = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    f = sub.add_parser("free-slots"); f.add_argument("gguf"); f.add_argument("--layers", type=layers, default=[14, 15])
    s = sub.add_parser("synth"); s.add_argument("gguf"); s.add_argument("out")
    s.add_argument("--slots", type=int, required=True); s.add_argument("--layers", type=layers, default=[14, 15])
    s.add_argument("--seed", type=int, default=0)
    for name in ("apply", "revoke"):
        e = sub.add_parser(name); e.add_argument("gguf"); e.add_argument("patch"); e.add_argument("--out")
    v = sub.add_parser("verify"); v.add_argument("base"); v.add_argument("patched"); v.add_argument("patch")
    b = sub.add_parser("bench"); b.add_argument("gguf"); b.add_argument("patch_a"); b.add_argument("patch_b")
    b.add_argument("--repeats", type=int, default=5); b.add_argument("--scratch", help="temp copy for the mmap timing")
    a = ap.parse_args()
    {"free-slots": cmd_free, "synth": cmd_synth, "apply": lambda a: cmd_edit(a, False),
     "revoke": lambda a: cmd_edit(a, True), "verify": cmd_verify, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
