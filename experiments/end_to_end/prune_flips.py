"""Minimum useful flips: how many of a trained FSA ternary patch's flips actually carry the facts?

Loads a saved patch (from e2e_fsa.py), then
  1. scores every nonzero slot i by its first-order utility u_i = -dw_i * dL_fact/dw_i (the loss increase if the
     flip were reverted; L_fact = answer-only CE over every training fact x all 12 augmented formats), and by its
     isolated local damage cost c_i = dw_i^2 * E[x_j^2] / den (frozen-base general-text statistics, as netcost);
     reports the utility distribution (Lorenz: fraction of flips carrying 50/90/99% of positive utility);
  2. checks the first-order ranking with exact leave-out losses for whole groups of flips (bottom/middle/top
     deciles, a random decile);
  3. reverts the lowest-score flips (score u, or u - lambda*c with --score net) in chunks of --chunk of the
     remaining flips, recomputing utility on the reduced patch after every chunk, and stops at the smallest patch
     whose validation recall (teacher-forced AND generated) stays >= the unpruned patch's; on a failing chunk the
     chunk is halved (down to --min-chunk) before stopping;
  4. evaluates the pruned patch once on the test protocol (TF + generated recall, in-conversation questions,
     40 x 512 dNLL +- SE), and saves it. Recovery fine-tuning is a separate e2e_fsa.py run (--load-patch, births
     only on the pruned support: --budget <nnz> --budget-freeze -1, --bop-tau-abs <original tau>).
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_fsa as E  # noqa: E402


def lorenz(u_sorted_desc, fracs=(0.5, 0.9, 0.99)):
    """Fraction of flips (taken in descending utility) needed to reach each fraction of the total positive
    utility."""
    pos = u_sorted_desc.clamp_min(0)
    tot = float(pos.sum())
    if len(u_sorted_desc) == 0 or tot <= 0:
        return {str(f): None for f in fracs}
    cs = torch.cumsum(pos.double(), 0)
    out = {}
    for f in fracs:
        k = int(torch.searchsorted(cs, torch.tensor([f * tot], dtype=cs.dtype, device=cs.device)).item()) + 1
        out[str(f)] = min(k, len(u_sorted_desc)) / len(u_sorted_desc)
    return out


def nonzero_table(mods):
    """(module index, flat index) of every nonzero code, as two long tensors."""
    mi, fi = [], []
    for i, m in enumerate(mods):
        ix = (m.codes().flatten() != 0).nonzero().squeeze(1)
        mi.append(torch.full_like(ix, i)); fi.append(ix)
    return torch.cat(mi), torch.cat(fi)


@torch.no_grad()
def flip_scores(mods, mi, fi, den):
    """u = -q * dL/dM (dM = alpha dW; the code q is the flip in units of alpha) and c = alpha^2 E[x_j^2] / den."""
    u = torch.empty(len(fi), dtype=torch.float64, device=fi.device)
    c = torch.empty_like(u)
    for i, m in enumerate(mods):
        sel = (mi == i).nonzero().squeeze(1)
        if not len(sel):
            continue
        ix = fi[sel]
        q = m.codes().flatten()[ix]
        g = m.M.grad.flatten()[ix] if m.M.grad is not None else torch.zeros_like(q)
        u[sel] = (-q * g).double()
        col = ix % m.M.shape[1]
        row = ix // m.M.shape[1]
        c[sel] = (m.alpha.float().squeeze(1)[row] ** 2 * m.col_energy[col] / den * q ** 2).double()
    return u, c


@torch.no_grad()
def revert(mods, mi, fi, which):
    for i, m in enumerate(mods):
        sel = which[mi[which] == i]
        if len(sel):
            m.M.view(-1)[fi[sel]] = 0


@torch.no_grad()
def restore(mods, mi, fi, vals, which):
    for i, m in enumerate(mods):
        sel = which[mi[which] == i]
        if len(sel):
            m.M.view(-1)[fi[sel]] = vals[sel]


def avg_ranks(x):
    """Ranks with ties averaged (float64)."""
    x = torch.as_tensor(x, dtype=torch.float64)
    order = torch.argsort(x)
    r = torch.empty_like(x)
    r[order] = torch.arange(len(x), dtype=torch.float64)
    for v in torch.unique(x):
        m = x == v
        if int(m.sum()) > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    """Spearman correlation with average ranks; None when undefined (n < 2 or a constant input)."""
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    if len(a) < 2 or bool((a == a[0]).all()) or bool((b == b[0]).all()):
        return None
    return float(torch.corrcoef(torch.stack([avg_ranks(a), avg_ranks(b)]))[0, 1])


def group_ids(names, mi, level):
    """Group id per flip: level 'linear' = the patched linear; 'expert' = (layer, expert) over up+down."""
    keys = [n if level == "linear" else n.rsplit(".", 1)[0] for n in names]
    uniq = sorted(set(keys))
    lut = torch.tensor([uniq.index(k) for k in keys], device=mi.device)
    return lut[mi], uniq


def main():
    ap = argparse.ArgumentParser()
    E.eda.add_argument(ap)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", choices=["u", "net"], default="u")
    ap.add_argument("--cost-rel", type=float, default=1.0, help="net: lambda = rel * median(u>0) / median(c)")
    ap.add_argument("--chunk", type=float, default=0.10)
    ap.add_argument("--min-chunk", type=float, default=0.025)
    ap.add_argument("--micro", type=int, default=6)
    ap.add_argument("--anchor", default=str(E.DATA / "anchor-wikitext.json"))
    ap.add_argument("--anchor-len", type=int, default=96)
    ap.add_argument("--eval-set", default=str(E.DATA / "eval-wikitext.json"))
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--conv-facts", type=int, default=40)
    ap.add_argument("--leaveout", action="store_true", help="exact leave-out check of decile groups at the start")
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--rank", choices=["first_order", "leaveout"], default="first_order",
                    help="leaveout: exact fact-loss increase when a whole group of flips is reverted")
    ap.add_argument("--group", choices=["expert", "linear"], default="expert")
    ap.add_argument("--eval-micro", type=int, default=24, help="micro-batch for no-grad fact-loss evaluations")
    a = ap.parse_args()
    E.eda.announce(a)

    st = torch.load(a.patch, map_location="cpu", weights_only=False)
    meta = st["meta"]
    E.eda.require_backend(st, E.eda.selected(a), "source patch")
    assert meta["method"].startswith("fsa") and meta["fmt"] == "ternary" and not meta.get("patch_scale") == "block"
    if meta.get("n_experts") or meta.get("selection"):
        raise SystemExit("patches on profiled expert subsets are not supported here")
    if sum(int(e["idx"].numel()) for e in st["entries"].values()) == 0:
        raise SystemExit("empty patch")
    proto = meta.get("protocol", {})
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
    torch.manual_seed(meta["seed"])
    tok = Tokenizer.from_file(str(E.DATA / "tokenizer.json"))
    model, cfg, dev = E.eda.build_model(build_model, a, E.EXPORT, device="cuda:0", model_config=None, step=58814,
                                  expert_format="bf16")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bos = E.BOS
    facts = E.make_world(meta["seed"], meta.get("people", 1), meta.get("facts_per_person", len(E.ATTRS)),
                         meta.get("shared_attrs", 0), meta.get("values", "disjoint"))
    conflicts = E.conflict_map(facts)
    et = json.load(open(a.eval_set))
    big = torch.tensor([[bos] + et[i * 511:i * 511 + 511] for i in range(a.eval_windows)], device=dev)
    valw = torch.tensor([[bos] + et[60000 + i * 511:60000 + i * 511 + 511] for i in range(a.val_windows)], device=dev)
    base_big, base_valw = E.nll_windows(model, big, dev), E.nll_windows(model, valw, dev)
    at = json.load(open(a.anchor))[:-510]
    L = a.anchor_len
    anchor = torch.tensor([[bos] + at[i:i + L - 1] for i in range(0, len(at) - L, L - 1)], device=dev)
    rngv = random.Random(meta["seed"] + 3)                # same draws, same order as e2e_fsa.py
    vf = proto.get("val_facts", 0)
    cf = proto.get("conv_facts", a.conv_facts)
    val_idx = sorted(rngv.sample(range(len(facts)), vf)) if 0 < vf < len(facts) else None
    conv_idx = sorted(rngv.sample(range(len(facts)), cf)) if 0 < cf < len(facts) else None
    K = proto.get("gen_stop_k", 2)
    GMAX = proto.get("gen_max_new", 0)
    items_val = E.build_items(tok, facts, E.VAL_T, bos, idxs=val_idx)
    items_test = E.build_items(tok, facts, E.TEST_T, bos)
    items_conv = E.build_items(tok, facts, [E.CONV_T], bos,
                               ctx=E.conv_contexts(tok, et, facts, meta["seed"], a.eval_windows * 511 + 1, 60000),
                               idxs=conv_idx)

    projs = tuple(meta["projs"])
    params, n_slots, _ = E.attach(model, meta["layers"], meta["method"], meta["rank"], "ternary", projs)
    lin_named = E.patched_linears(model)
    names = [n for n, _ in lin_named]
    mods = [lin.parametrizations.weight[0] for _, lin in lin_named]
    assert all(isinstance(m, E.FSAParam) for m in mods) and len(mods) == len(E.fsa_modules(model))
    E.load_patch(model, st)
    den = E.collect_col_energy(model, mods, anchor[:32])

    # every training fact x all 12 augmented formats, answer-only loss (fixed set; no sampling)
    seqs, starts = [], []
    for f in facts:
        for t in E.TRAIN_AUG_T:
            full = E.fmt(t, f)
            cut = full.rindex(f.val)
            p_ids = E.encode(tok, full[:cut].rstrip(), bos)
            seqs.append(p_ids + tok.encode(" " + f.val + full[cut + len(f.val):], add_special_tokens=False).ids)
            starts.append(len(p_ids))
    X, LM = E.batchify(seqs, dev, starts)
    ntok = int(LM[:, 1:].sum())

    def fact_loss(grad):
        for m in mods:
            m.M.grad = None
        tot = 0.0
        ctx = torch.enable_grad() if grad else torch.no_grad()
        mb_ = a.micro if grad else a.eval_micro
        with ctx:
            for s0 in range(0, X.shape[0], mb_):
                xb, mb = X[s0:s0 + mb_], LM[s0:s0 + mb_, 1:]
                lg = E.logits_of(model, xb)[:, :-1, :E.VOCAB].float()
                l = F.cross_entropy(lg[mb], xb[:, 1:][mb], reduction="sum") / ntok
                if grad and l.requires_grad:
                    l.backward()
                tot += l.item()
        return tot

    def val_eval():
        v = E.tf_eval(model, items_val, dev, 1)
        g = E.gen_eval(model, tok, items_val, dev, GMAX, K, 1, conflicts)
        vd = float(torch.tensor(E.nll_windows(model, valw, dev)).mean() - torch.tensor(base_valw).mean())
        return dict(val_em=v["exact_match"], val_gen=g["recall"], val_dnll=vd)

    def big_eval():
        d = torch.tensor(E.nll_windows(model, big, dev)) - torch.tensor(base_big)
        return dict(dnll=float(d.mean()), dnll_se=float(d.std() / math.sqrt(len(d))))

    def test_suite():
        return dict(test=E.tf_eval(model, items_test, dev, 1),
                    test_gen=E.gen_eval(model, tok, items_test, dev, GMAX, K, 1, conflicts),
                    conv=E.tf_eval(model, items_conv, dev, 1),
                    conv_gen=E.gen_eval(model, tok, items_conv, dev, GMAX, K, 1, conflicts), big=big_eval())

    t0 = time.time()
    res = dict(eda_backend=E.eda.selected(a), protocol=dict(proto, val_facts=vf, conv_facts=cf), patch=a.patch, score=a.score, cost_rel=a.cost_rel, chunk=a.chunk, n_slots=n_slots, meta=meta)
    mi, fi = nonzero_table(mods)
    vals = torch.cat([m.M.view(-1)[fi[mi == i]] for i, m in enumerate(mods)])
    # vals is ordered by module, same as (mi, fi) from nonzero_table
    n0 = len(fi)
    L0 = fact_loss(True)
    u, c = flip_scores(mods, mi, fi, den)
    v0 = val_eval()
    print(f"loaded {n0} flips; fact loss {L0:.4f}; val {v0} ({time.time()-t0:.0f}s)", flush=True)
    us, _ = torch.sort(u, descending=True)
    qs = torch.quantile(u.float().cpu(), torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])).tolist()
    res["unpruned"] = dict(n=n0, fact_loss=L0, val=v0, u_sum=float(u.sum()), u_pos_sum=float(u.clamp_min(0).sum()),
                           frac_negative=float((u < 0).float().mean()), u_quantiles=qs, lorenz=lorenz(us),
                           c_sum=float(c.sum()), storage=E.fsa_storage(n0, n_slots, "ternary"))
    print("UNPRUNED", json.dumps(res["unpruned"]), flush=True)
    lam = 0.0
    if a.score == "net":
        if not bool((u > 0).any()):
            raise SystemExit("--score net: no flip has positive utility; lambda undefined")
        lam = a.cost_rel * float(u[u > 0].median() / c.median().clamp_min(1e-30))
    res["lambda"] = lam

    if a.leaveout:                                         # exact group leave-out vs first-order prediction
        order = torch.argsort(u)
        k = n0 // 10
        g = torch.Generator(device="cpu").manual_seed(0)
        groups = dict(bottom10=order[:k], middle10=order[n0 // 2 - k // 2:n0 // 2 - k // 2 + k], top10=order[-k:],
                      random10=torch.randperm(n0, generator=g)[:k].to(order.device),
                      bottom1=order[:n0 // 100], top1=order[-(n0 // 100):])
        lo = {}
        Lnog0 = fact_loss(False)                           # baseline with the same (no-grad) micro-batching
        for name, w in groups.items():
            revert(mods, mi, fi, w)
            Lw = fact_loss(False)
            restore(mods, mi, fi, vals, w)
            lo[name] = dict(n=len(w), predicted=float(u[w].sum()), actual=Lw - Lnog0)
            print("LEAVEOUT", name, json.dumps(lo[name]), flush=True)
        res["leaveout"] = lo

    if a.rank == "leaveout":
        gid, gnames = group_ids(names, mi, a.group)
        Lnog = fact_loss(False)                            # no-grad baseline on the same micro-batching
        G = []
        for g in range(len(gnames)):
            w = (gid == g).nonzero().squeeze(1)
            if not len(w):
                continue
            revert(mods, mi, fi, w)
            d = fact_loss(False) - Lnog
            restore(mods, mi, fi, vals, w)
            G.append(dict(group=gnames[g], n=len(w), dloss=d, per_flip=d / len(w), u_sum=float(u[w].sum())))
            if g % 16 == 0:
                print("GROUP", g, json.dumps(G[-1]), f"({time.time()-t0:.0f}s)", flush=True)
        dl = torch.tensor([x["dloss"] for x in G])
        nn_ = torch.tensor([x["n"] for x in G], dtype=torch.float64)
        res["groups"] = G
        res["group_summary"] = dict(n_groups=len(G), level=a.group, nonpositive=int((dl <= 0).sum()),
                                    dloss_quantiles=torch.quantile(dl.double(), torch.tensor(
                                        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=torch.float64)).tolist(),
                                    sum_of_group_dloss=float(dl.sum()), flips_in_groups=int(nn_.sum()),
                                    spearman_u_vs_dloss=spearman([x["u_sum"] for x in G], dl))
        print("GROUPS", json.dumps(res["group_summary"]), flush=True)
        # static ranking: cheapest loss per flip first; revert whole groups in chunks, val-checked (halving)
        order = sorted(range(len(G)), key=lambda i: G[i]["per_flip"])
        kept = list(order)
        gidx = {x["group"]: gnames.index(x["group"]) for x in G}
        chunk, steps, removed = a.chunk, [], []
        best = dict(n=n0, val=v0, removed=[])
        while kept and len(steps) < a.max_iters:
            k = max(1, int(chunk * len(kept)))
            cand = kept[:k]
            w = torch.cat([(gid == gidx[G[i]["group"]]).nonzero().squeeze(1) for i in cand])
            revert(mods, mi, fi, w)
            v = val_eval()
            ok = v["val_em"] >= v0["val_em"] and v["val_gen"] >= v0["val_gen"]
            rec = dict(iter=len(steps), chunk=chunk, groups_dropped=k, flips_dropped=len(w),
                       predicted_dloss=sum(G[i]["dloss"] for i in cand), ok=ok, **v, t=round(time.time() - t0))
            steps.append(rec)
            print("GCHUNK", json.dumps(rec), flush=True)
            if ok:
                removed += cand
                kept = kept[k:]
                best = dict(n=E.patch_stats(mods)[0], val=v, removed=list(removed))
            else:
                restore(mods, mi, fi, vals, w)
                if chunk / 2 < a.min_chunk or k == 1:
                    break
                chunk /= 2
        res["chunks"] = steps
        res["removed_groups"] = [G[i]["group"] for i in best["removed"]]
        alive = torch.ones(n0, dtype=torch.bool, device=fi.device)
        for i in best["removed"]:
            alive[gid == gidx[G[i]["group"]]] = False
        best = dict(n=int(alive.sum()), alive=alive, val=best["val"])
    else:
        alive = torch.ones(n0, dtype=torch.bool, device=fi.device)
        chunk, steps, best = a.chunk, [], dict(n=n0, alive=alive.clone(), val=v0)
        for it in range(a.max_iters):
            idx = alive.nonzero().squeeze(1)
            if len(idx) <= 1:
                break
            score = u[idx] - lam * c[idx]
            k = max(1, int(chunk * len(idx)))
            drop = idx[torch.argsort(score)[:k]]
            revert(mods, mi, fi, drop)
            Lb = fact_loss(True)
            v = val_eval()
            ok = v["val_em"] >= v0["val_em"] and v["val_gen"] >= v0["val_gen"]
            rec = dict(iter=it, chunk=chunk, n_before=len(idx), dropped=k, n_after=len(idx) - k, ok=ok,
                       predicted_dloss=float(u[drop].sum()), fact_loss=Lb, **v, t=round(time.time() - t0))
            steps.append(rec)
            print("CHUNK", json.dumps(rec), flush=True)
            if ok:
                alive[drop] = False
                best = dict(n=int(alive.sum()), alive=alive.clone(), val=v)
                u, c = flip_scores(mods, mi, fi, den)      # utility on the reduced patch (dropped slots: ignored)
            else:
                restore(mods, mi, fi, vals, drop)
                if chunk / 2 < a.min_chunk:
                    break
                chunk /= 2
                fact_loss(True)                            # grads back to the last accepted patch
                u, c = flip_scores(mods, mi, fi, den)
        res["chunks"] = steps
    # final: the smallest accepted patch
    full = torch.ones(n0, dtype=torch.bool, device=fi.device)
    restore(mods, mi, fi, vals, full.nonzero().squeeze(1))
    revert(mods, mi, fi, (~best["alive"]).nonzero().squeeze(1))
    nnz = E.patch_stats(mods)[0]
    assert nnz == best["n"], (nnz, best["n"])
    out = Path(a.out)
    pf = out.with_name(out.stem + "_pruned.pt")
    meta2 = dict(meta, pruned_from=a.patch, pruned_score=a.score)
    nbytes = E.save_patch(model, pf, meta2)
    res["pruned"] = dict(n=nnz, val=best["val"], storage=E.fsa_storage(nnz, n_slots, "ternary"), patch_file=str(pf),
                         patch_file_bytes=nbytes, **test_suite())
    print("PRUNED", json.dumps({k: v for k, v in res["pruned"].items() if k not in ("test_gen", "conv_gen")}),
          flush=True)
    restore(mods, mi, fi, vals, full.nonzero().squeeze(1))   # unpruned test numbers on the same protocol
    res["unpruned"].update(test_suite())
    E.set_enabled(False)
    out.write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("chunks", "meta")}), flush=True)


if __name__ == "__main__":
    main()
