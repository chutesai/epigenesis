"""Can the free-slot coordinates represent what LoRA learned? (no training)

Loads a trained LoRA patch (e2e_fsa.py), forms dW = B @ A for every adapted linear and
  1. reports where its energy sits: free in-pair slots / occupied (nonzero) weights / structural zeros, and the
     distribution of |dW| / alpha_row on free slots;
  2. evaluates on the same protocol (generated + TF recall on validation phrasings, cross-person confusion,
     40 x 512 dNLL): the LoRA itself; dW restricted to free slots with continuous values (fmt bf16: the
     expressivity ceiling of the free-slot coordinates, not a legal ternary patch); and legal ternary projections
     sign(dW) * alpha on free slots, chosen by global top-K of |dW| / alpha or by |dW| > theta * alpha.
"""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_fsa as E  # noqa: E402

MASK_METHOD = ["fsa_free"]       # set from --mask: fsa_free (in-pair free slots) or fsa_full (any base zero)


@torch.no_grad()
def gptq_quantize(T, alpha, free, Hinv_u):
    """GPTQ over columns with a legality mask: T [out, in] target update, alpha [out, 1], free [out, in] bool,
    Hinv_u upper Cholesky factor of H^-1. Each column becomes alpha * {-1,0,+1} on free slots, 0 elsewhere; its
    rounding error is pushed onto the not-yet-quantized columns (OBS update). Returns the codes [out, in]."""
    T = T.clone()
    codes = torch.zeros_like(T)
    a_ = alpha.float().clamp_min(1e-12)
    for i in range(T.shape[1]):
        w = T[:, i]
        c = torch.clamp(torch.round(w / a_[:, 0]), -1, 1) * free[:, i]
        codes[:, i] = c
        err = (w - c * a_[:, 0]) / Hinv_u[i, i]
        if i + 1 < T.shape[1]:
            T[:, i + 1:] -= err[:, None] * Hinv_u[i, i + 1:][None, :]
    return codes


@torch.no_grad()
def gptq_fold(model, tok, facts, layers, projs, scope, lora_dw, bos, dev, a, evaluate, at_file):
    """Second-order fold: H = X^T X of each patched linear's base inputs over the fact training prompts
    (--gptq-formats formats per fact, batch 1: no padding) plus 32 general-text windows. One MoE layer at a time,
    so only that layer's Hessians are resident."""
    E.detach_all(model)
    E.attach(model, layers, MASK_METHOD[0], 0, "ternary", projs, **scope)
    lins = dict(E.patched_linears(model))
    names = list(lins)
    seqs = [E.encode(tok, E.fmt(t, f_), bos) for f_ in facts for t in E.TRAIN_AUG_T[:a.gptq_formats]]
    at = json.load(open(at_file))[:-510]
    anc = [[bos] + at[i:i + 95] for i in range(0, 32 * 95, 95)]
    lnames = [n for n, _ in E.moe_layers(model)]
    t0 = time.time()
    nnz = unseen = fallback = 0
    calib = 0
    for li in layers:
        pref = lnames[li] + "."
        group = [n for n in names if n.startswith(pref)]
        H = {n: None for n in group}
        cnt = {n: 0 for n in group}
        hooks = []
        for n in group:
            def f(mod, inp, out, n=n):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                if x.shape[0] == 0:
                    return
                h = x.T @ x
                H[n] = h if H[n] is None else H[n] + h
                cnt[n] += x.shape[0]
            hooks.append(lins[n].register_forward_hook(f))
        E.set_enabled(False)
        try:
            for sq in seqs + anc:
                E.logits_of(model, torch.tensor([sq], device=dev))
        finally:
            E.set_enabled(True)
            for h in hooks:
                h.remove()
        calib += sum(cnt.values())
        for n in group:
            pm = lins[n].parametrizations.weight[0]
            dw = lora_dw[n].to(pm.M.device)
            codes = None
            if H[n] is None:
                unseen += 1
            else:
                Hm = H[n] / cnt[n]
                dmean = float(Hm.diagonal().mean())
                if math.isfinite(dmean) and dmean > 0:
                    Hm += a.gptq_damp * dmean * torch.eye(Hm.shape[0], device=Hm.device)
                    for jit in (0.0, 1e-2, 1e-1):
                        try:
                            L = torch.linalg.cholesky(Hm + jit * dmean * torch.eye(Hm.shape[0], device=Hm.device))
                            Hinv_u = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
                            codes = gptq_quantize(dw, pm.alpha, pm.mask, Hinv_u)
                            break
                        except RuntimeError:
                            continue
                if codes is None:
                    fallback += 1
            if codes is None:                              # unrouted / zero-energy / failed factorization: round
                codes = torch.clamp(torch.round(dw / pm.alpha.float().clamp_min(1e-12)), -1, 1) * pm.mask
            pm.M.copy_(codes)
            nnz += int((codes != 0).sum())
            H[n] = None
        del H
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"gptq layer {li} done ({time.time()-t0:.0f}s)", flush=True)
    r = evaluate("gptq_fold")
    st_ = E.fsa_storage(nnz, sum(int(lins[n].parametrizations.weight[0].mask.sum()) for n in names), "ternary")
    out = dict(r, slots=nnz, bytes_sparse=st_["bytes_sparse"], unseen_modules=unseen, fallback_modules=fallback,
               calib_tokens=calib, seconds=round(time.time() - t0))
    print("GPTQ", json.dumps(out), flush=True)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    E.eda.add_argument(ap)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", default="250000,1000000,4000000")
    ap.add_argument("--theta", default="0.5,0.25")
    ap.add_argument("--val-facts", type=int, default=60)
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--mask", choices=["free", "any_zero", "pair8_forced"], default="free",
                    help="writable set: in-pair free slots (pair8-legal) or any zero of the base weight")
    ap.add_argument("--gptq", action="store_true", help="also run the second-order (GPTQ/OBS) ternary fold")
    ap.add_argument("--gptq-damp", type=float, default=0.01, help="H damping, fraction of mean diag(H) (> 0)")
    ap.add_argument("--gptq-formats", type=int, default=4, help="training formats per fact in the calibration set")
    ap.add_argument("--only-gptq", action="store_true", help="skip the plain/multiscale projections")
    ap.add_argument("--ms-block", type=int, default=16, help="multi-scale fold: input columns per exponent block")
    ap.add_argument("--ms-levels", default="1,2,3,4", help="multi-scale fold: numbers of exponent levels to try")
    a = ap.parse_args()
    E.eda.announce(a)
    MASK_METHOD[0] = {"any_zero": "fsa_full", "pair8_forced": "fsa_forced"}.get(a.mask, "fsa_free")
    if not (math.isfinite(a.gptq_damp) and a.gptq_damp > 0):
        raise SystemExit("--gptq-damp must be finite and > 0")
    st = torch.load(a.patch, map_location="cpu", weights_only=False)
    meta = st["meta"]
    E.eda.require_backend(st, E.eda.selected(a), "source patch")
    assert meta["method"] == "lora"
    sys.path.insert(0, "/workspace/malleable/mlf")
    from ops.eval.bench_saves import build_model
    from tokenizers import Tokenizer
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
    rngv = random.Random(meta["seed"] + 3)
    vidx = sorted(rngv.sample(range(len(facts)), a.val_facts)) if 0 < a.val_facts < len(facts) else None
    items = E.build_items(tok, facts, E.VAL_T, bos, idxs=vidx)
    et = json.load(open(E.DATA / "eval-wikitext.json"))
    big = torch.tensor([[bos] + et[i * 511:i * 511 + 511] for i in range(a.eval_windows)], device=dev)
    base_big = E.nll_windows(model, big, dev)
    t0 = time.time()

    def evaluate(tag):
        g = E.gen_eval(model, tok, items, dev, 0, 2, 1, conflicts)
        t = E.tf_eval(model, items, dev, 1)
        d = torch.tensor(E.nll_windows(model, big, dev)) - torch.tensor(base_big)
        r = dict(gen=g["recall"], tf=t["exact_match"], confusion=g.get("confusion"),
                 all_people_correct=g.get("all_people_correct"), dnll=float(d.mean()),
                 dnll_se=float(d.std() / math.sqrt(len(d))))
        print("EVAL", tag, json.dumps(r), f"({time.time()-t0:.0f}s)", flush=True)
        return r

    res = dict(eda_backend=E.eda.selected(a), patch=a.patch, n_val_items=len(items), mask=a.mask)
    res["base"] = evaluate("base")
    layers, projs = meta["layers"], tuple(meta["projs"])
    scope = dict(shared=bool(meta.get("shared_expert") or meta.get("shared_only")),
                 routed=not meta.get("shared_only", False))
    E.attach(model, layers, "lora", meta["rank"], "ternary", projs, **scope)
    E.load_patch(model, st)
    res["lora"] = evaluate("lora")
    lora_dw = {n: lin.parametrizations.weight[0].delta_w().detach().float().cpu()
               for n, lin in E.patched_linears(model)}
    E.detach_all(model)

    # energy split and |dW|/alpha on free slots
    E.attach(model, layers, MASK_METHOD[0], 0, "bf16", projs, **scope)
    lins = dict(E.patched_linears(model))
    tot = dict(free=0.0, occupied=0.0, structural=0.0)
    ratios = []
    for n, lin in lins.items():
        pm = lin.parametrizations.weight[0]
        W = lin.parametrizations.weight.original
        dw = lora_dw[n].to(W.device)
        occ = W != 0
        free = pm.mask
        tot["free"] += float((dw[free] ** 2).sum())
        tot["occupied"] += float((dw[occ] ** 2).sum())
        tot["structural"] += float((dw[~occ & ~free] ** 2).sum())
        r = (dw.abs() / pm.alpha.float())[free]
        if r.numel():
            ratios.append(r[torch.randint(0, r.numel(), (min(r.numel(), 20000),), device=r.device)].cpu())
    if not ratios or sum(tot.values()) <= 0:
        raise SystemExit("no free slots or zero LoRA update: nothing to project")
    rs = torch.cat(ratios)
    S = sum(tot.values())
    res["energy_frac"] = {k: v / S for k, v in tot.items()}
    res["free_ratio_quantiles"] = dict(zip(["50", "90", "99", "99.9", "max_sample"],
                                           torch.quantile(rs, torch.tensor([0.5, 0.9, 0.99, 0.999])).tolist()
                                           + [float(rs.max())]))
    res["free_frac_ratio_ge"] = {str(t): float((rs >= t).float().mean()) for t in (0.1, 0.25, 0.5, 1.0)}
    print("ENERGY", json.dumps({k: res[k] for k in ("energy_frac", "free_ratio_quantiles", "free_frac_ratio_ge")}),
          flush=True)

    # continuous dW on free slots only (fmt bf16 -> delta = M * alpha, unquantized)
    for n, lin in lins.items():
        pm = lin.parametrizations.weight[0]
        pm.M.copy_((lora_dw[n].to(pm.M.device) / pm.alpha.float()) * pm.mask)
    res["free_continuous"] = evaluate("free_continuous")
    E.detach_all(model)

    if a.only_gptq:
        E.detach_all(model)
        res["gptq"] = gptq_fold(model, tok, facts, layers, projs, scope, lora_dw, bos, dev, a, evaluate,
                                at_file=E.DATA / "anchor-wikitext.json")
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("RESULT", json.dumps(res), flush=True)
        return
    # legal ternary projections
    E.attach(model, layers, MASK_METHOD[0], 0, "ternary", projs, **scope)
    lins = dict(E.patched_linears(model))
    names = list(lins)

    def ratio(n, pm):
        """|dW| / alpha on free slots with a nonzero dW (else 0), flattened."""
        dw = lora_dw[n].to(pm.M.device)
        return ((dw.abs() / pm.alpha.float()) * (pm.mask & (dw != 0))).flatten(), torch.sign(dw).flatten()

    def set_codes(sel):                                    # sel: {module name: flat indices}
        for n in names:
            pm = lins[n].parametrizations.weight[0]
            pm.M.zero_()
            if n in sel and len(sel[n]):
                _, sg = ratio(n, pm)
                pm.M.view(-1)[sel[n]] = sg[sel[n]]
        return E.patch_stats([lins[n].parametrizations.weight[0] for n in names])[0]

    res["ternary_topk"] = {}
    n_cand = sum(int((ratio(n, lins[n].parametrizations.weight[0])[0] > 0).sum()) for n in names)
    for K in [int(k) for k in a.topk.split(",") if k]:
        if K < 0:
            raise SystemExit("--topk values must be >= 0")
        K = min(K, n_cand)
        if K == 0:
            res["ternary_topk"]["0"] = dict(evaluate("ternary_top0") if not set_codes({}) else {}, slots=0,
                                            ratio_threshold=None)
            continue
        best_v = best_i = None                             # exact running global top-K over free, nonzero-dW slots
        for mi_, n in enumerate(names):
            r, _ = ratio(n, lins[n].parametrizations.weight[0])
            npos = int((r > 0).sum())
            if not npos:
                continue
            v, ix = torch.topk(r, min(K, npos))
            ii = torch.stack([torch.full_like(ix, mi_), ix], 1)
            if best_v is not None:
                v, ii = torch.cat([best_v, v]), torch.cat([best_i, ii])
            top = torch.topk(v, min(K, v.numel())).indices
            best_v, best_i = v[top], ii[top]
        sel = {names[mi_]: best_i[best_i[:, 0] == mi_, 1] for mi_ in best_i[:, 0].unique().tolist()}
        n_on = set_codes(sel)
        assert n_on <= K
        r = evaluate(f"ternary_top{K}")
        res["ternary_topk"][str(K)] = dict(r, slots=n_on, ratio_threshold=float(best_v.min()))
    res["ternary_theta"] = {}
    for th in [float(t) for t in a.theta.split(",") if t]:
        sel = {}
        for n in names:
            r, _ = ratio(n, lins[n].parametrizations.weight[0])
            sel[n] = (r > th).nonzero().squeeze(1)
        n_on = set_codes(sel)
        r = evaluate(f"ternary_theta{th}")
        res["ternary_theta"][str(th)] = dict(r, slots=n_on)
    if a.gptq:
        res["gptq"] = gptq_fold(model, tok, facts, layers, projs, scope, lora_dw, bos, dev, a, evaluate, at_file=E.DATA
                                / "anchor-wikitext.json")
    # (d) multi-scale fold: per (row, block of input columns) a patch-only exponent e in {0..L-1}; slot values
    # c * alpha_row * 2^-e with c ternary on free slots; e minimizes the x-weighted block error
    #   sum_j E[x_j^2] (dW_j - c_j alpha 2^-e)^2   (diagonal activation statistics from 32 anchor windows)
    E.detach_all(model)
    E.attach(model, layers, MASK_METHOD[0], 0, "bf16", projs, **scope)
    lins = dict(E.patched_linears(model))
    mods_ = [lins[n].parametrizations.weight[0] for n in names]
    at = json.load(open(E.DATA / "anchor-wikitext.json"))[:-510]
    anc = torch.tensor([[bos] + at[i:i + 95] for i in range(0, 32 * 95, 95)], device=dev)
    den = E.collect_col_energy(model, mods_, anc)
    res["multiscale"] = {}
    blk = a.ms_block
    for nlev in [int(x) for x in a.ms_levels.split(",") if x]:
        nnz = nblk = 0
        werr = wtot = 0.0
        for n, pm in zip(names, mods_):
            dw = lora_dw[n].to(pm.M.device)
            o, i_ = dw.shape
            u = dw / pm.alpha.float().clamp_min(1e-12)    # in units of alpha_row
            wcol = pm.col_energy[None, :].expand(o, i_)
            best_err = best_c = None
            for e in range(nlev):
                sc = 2.0 ** -e
                c = torch.clamp(torch.round(u / sc), -1, 1) * pm.mask
                err = (wcol * (u - c * sc) ** 2 * pm.alpha.float() ** 2)
                eb = err.reshape(o, i_ // blk, blk).sum(-1)
                if best_err is None:
                    best_err, best_c = eb, (c * sc).reshape(o, i_ // blk, blk)
                else:
                    take = eb < best_err
                    best_err = torch.where(take, eb, best_err)
                    best_c = torch.where(take[..., None], (c * sc).reshape(o, i_ // blk, blk), best_c)
            vals = best_c.reshape(o, i_)
            pm.M.copy_(vals)
            nnz += int((vals != 0).sum())
            nblk += int((vals.reshape(o, i_ // blk, blk) != 0).any(-1).sum())
            werr += float(best_err.sum())
            wtot += float((wcol * dw ** 2).sum())
        r = evaluate(f"multiscale_L{nlev}")
        st_ = E.fsa_storage(nnz, sum(int(m.mask.sum()) for m in mods_), "ternary")
        extra = nblk * math.ceil(math.log2(nlev)) / 8 if nlev > 1 else 0.0
        res["multiscale"][str(nlev)] = dict(r, slots=nnz, patched_blocks=nblk, bytes_sparse=st_["bytes_sparse"],
                                            bytes_exponents=extra, x_weighted_residual_frac=werr / max(wtot, 1e-30))
        print("MULTISCALE", nlev, json.dumps(res["multiscale"][str(nlev)]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
