"""Can legal free slots reproduce what a trained LoRA does? (MEMIT/AlphaEdit-style masked least squares, no training)

For each patched MoE layer L (one at a time, so routing into L is identical with and without the adapter):
  * run the training fact sequences (prompt + answer, --formats phrasings per fact) and general-text windows with
    the LoRA enabled on layer L only, recording every routed expert's output y_lora; then with it disabled,
    recording the down_proj input h (post-relu^2 key, 2304-d) and output y_base, row-aligned per expert;
  * target per expert: T = y_lora - y_base on fact tokens (the adapter's effect), 0 on general tokens;
  * per down_proj row r: solve over that row's free slots F_r only (legal positions; values continuous)
        min_w ||H_f[:, F_r] w - T[:, r]||^2 + lam * ||H_g[:, F_r] w||^2 + eps ||w||^2
    (lam: AlphaEdit-like preservation of general-text keys, as a soft constraint; eps scales with the trace);
  * report the fraction of target energy explained on fact tokens and the general-token energy of the fit vs
    the LoRA's, then evaluate the model with (1) the continuous legal fit and (2) a ternary GPTQ fold of it
    (H = H_f^T H_f + lam H_g^T H_g) on the same protocol (generated + TF recall on validation phrasings,
    confusion, 40 x 512 dNLL). Up projections are left untouched (down-only legal edit).
"""
import argparse, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_fsa as E  # noqa: E402
import lora_project as LP  # noqa: E402


@torch.no_grad()
def Wbase_alpha(Wbase):
    """alpha_row = mean |nonzero| of each base row (the patch's ternary step at scale 1)."""
    nz = Wbase != 0
    return (Wbase.float().abs() * nz).sum(1) / nz.sum(1).clamp_min(1)


@torch.no_grad()
def masked_ls(Hf, T, Hg, free, lam, eps_rel=1e-3, chunk=64, box=None, iters=200):
    """Hf [n_f, i], T [n_f, o], Hg [n_g, i], free [o, i] bool -> W [o, i] (zero off free). Row-wise ridge solves
    restricted to each row's free columns, batched by padding rows to the same support size. box: optional [o]
    per-row bound; then the same objective is minimized subject to |w| <= box[r] by projected gradient (step 1/L,
    L = largest eigenvalue of the row system), started from the clipped unconstrained solution."""
    o, i = free.shape
    G = Hf.T @ Hf + lam * (Hg.T @ Hg)                     # [i, i]
    b = Hf.T @ T                                           # [i, o]
    eps = eps_rel * float(G.diagonal().mean()) + 1e-12
    W = torch.zeros(o, i, device=Hf.device)
    cnt = free.sum(1)
    kmax = int(cnt.max()) if o else 0
    if kmax == 0:
        return W
    for r0 in range(0, o, chunk):
        rows = torch.arange(r0, min(o, r0 + chunk), device=Hf.device)
        fr = free[rows]
        # padded column index per row; padding columns get identity rows (solution 0 there)
        idx = torch.zeros(len(rows), kmax, dtype=torch.long, device=Hf.device)
        valid = torch.zeros(len(rows), kmax, dtype=torch.bool, device=Hf.device)
        for k, r in enumerate(rows.tolist()):
            cols = fr[k].nonzero().squeeze(1)
            idx[k, :len(cols)] = cols
            valid[k, :len(cols)] = True
        A = G[idx[:, :, None], idx[:, None, :]]            # [R, k, k]
        vv = valid[:, :, None] & valid[:, None, :]
        A = torch.where(vv, A, torch.zeros_like(A))
        A = A + torch.diag_embed(torch.where(valid, torch.full_like(valid, eps, dtype=A.dtype),
                                             torch.ones_like(valid, dtype=A.dtype)))
        rhs = torch.where(valid, b[idx, rows[:, None]], torch.zeros(1, device=Hf.device))
        sol = torch.linalg.solve(A, rhs[..., None])[..., 0]
        sol = torch.where(valid, sol, torch.zeros_like(sol))
        if box is not None:
            bnd = box[rows][:, None]
            Lc = torch.linalg.eigvalsh(A)[:, -1:].clamp_min(1e-30)
            x = torch.maximum(torch.minimum(sol, bnd), -bnd)
            for _ in range(iters):
                g_ = (A @ x[..., None])[..., 0] - rhs
                x = torch.maximum(torch.minimum(x - g_ / Lc, bnd), -bnd)
            sol = torch.where(valid, x, torch.zeros_like(x))
        W[rows[:, None].expand_as(idx)[valid], idx[valid]] = sol[valid]
    return W


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    E.eda.add_argument(ap)
    ap.add_argument("--patch", required=True, help="trained LoRA patch (e2e_fsa.py --method lora)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--formats", type=int, default=4, help="training phrasings per fact in the fit")
    ap.add_argument("--general-windows", type=int, default=24)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of general-text key preservation")
    ap.add_argument("--val-facts", type=int, default=60)
    ap.add_argument("--eval-windows", type=int, default=40)
    ap.add_argument("--gptq-damp", type=float, default=0.01)
    ap.add_argument("--mask", choices=["free", "any_zero", "pair8_forced"], default="free")
    ap.add_argument("--scale-mults", type=lambda x: [float(v) for v in x.split(",")], default=[1.0, 0.5, 0.25],
                    help="patch-scale multipliers for the ternary step (patch values +-s*alpha_row)")
    a = ap.parse_args()
    E.eda.announce(a)
    if not a.scale_mults or not all(math.isfinite(m_) and 0 < m_ <= 4 for m_ in a.scale_mults):
        raise SystemExit("--scale-mults must be finite values in (0, 4]")
    st = torch.load(a.patch, map_location="cpu", weights_only=False)
    meta = st["meta"]
    E.eda.require_backend(st, E.eda.selected(a), "source patch")
    assert meta["method"] == "lora" and not meta.get("shared_expert") and not meta.get("shared_only")
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

    seqs = [E.encode(tok, E.fmt(t, f_), bos) for f_ in facts for t in E.TRAIN_AUG_T[:a.formats]]
    at = json.load(open(E.DATA / "anchor-wikitext.json"))[:-510]
    gen_seqs = [[bos] + at[i:i + 95] for i in range(0, a.general_windows * 95, 95)]
    layers, projs = meta["layers"], tuple(meta["projs"])
    E.attach(model, layers, "lora", meta["rank"], "ternary", projs)
    E.load_patch(model, st)
    res = dict(eda_backend=E.eda.selected(a), patch=a.patch, lam=a.lam, formats=a.formats, mask=a.mask)
    res["lora"] = evaluate("lora")
    lmods = dict(E.patched_linears(model))
    lnames = [n for n, _ in E.moe_layers(model)]
    fits = {}                                              # down_proj name -> legal W (cpu)
    stats = defaultdict(float)
    for li in layers:
        pref = lnames[li] + "."
        moe = E.moe_layers(model)[li][1]
        rec = {}

        def run(seq_list, lora_on_layer):
            E.set_enabled(True)                            # which layers are active is set by zeroing B
            store = defaultdict(lambda: dict(y=[], h=[]))
            hooks = []
            for ei, e in enumerate(moe.experts):
                if e is None:
                    continue
                hooks.append(e.register_forward_hook(
                    lambda mod, inp, out, ei=ei: store[ei]["y"].append(out.reshape(-1, out.shape[-1]).float())))
                hooks.append(e.down_proj.register_forward_hook(
                    lambda mod, inp, out, ei=ei: store[ei]["h"].append(inp[0].reshape(-1, inp[0].shape[-1]).float())))
            try:
                for sq in seq_list:
                    E.logits_of(model, torch.tensor([sq], device=dev))
            finally:
                for h in hooks:
                    h.remove()
            return {k: dict(y=torch.cat(v["y"]) if v["y"] else None, h=torch.cat(v["h"]) if v["h"] else None)
                    for k, v in store.items()}

        # LoRAParam has no per-module switch: emulate one by zeroing other layers' B temporarily
        allB = {n: lin.parametrizations.weight[0].B.detach().clone() for n, lin in lmods.items()}
        try:
            for n in allB:
                if not n.startswith(pref):
                    lmods[n].parametrizations.weight[0].B.zero_()
            on_f = run(seqs, True)
            on_g = run(gen_seqs, True)
            for n in allB:
                if n.startswith(pref):
                    lmods[n].parametrizations.weight[0].B.zero_()
            off_f = run(seqs, False)
            off_g = run(gen_seqs, False)
        finally:
            for n, B in allB.items():
                lmods[n].parametrizations.weight[0].B.copy_(B)
        for ei, e in enumerate(moe.experts):
            if e is None or ei not in off_f or off_f[ei]["h"] is None:
                continue
            if on_f.get(ei, {}).get("y") is None or on_f[ei]["y"].shape[0] != off_f[ei]["y"].shape[0]:
                stats["skipped_experts"] += 1
                continue
            Hf, T = off_f[ei]["h"], on_f[ei]["y"] - off_f[ei]["y"]
            Hg = off_g[ei]["h"] if ei in off_g and off_g[ei]["h"] is not None else torch.zeros(0, Hf.shape[1],
                                                                                             device=dev)
            Wbase = e.down_proj.parametrizations.weight.original
            pr = (Wbase.reshape(-1, 4, 2) != 0).any(-1, keepdim=True).expand(-1, -1, 2).reshape(Wbase.shape)
            free = ((Wbase == 0) if a.mask == "any_zero" else E.pair8_forced_mask(Wbase)
                    if a.mask == "pair8_forced" else (Wbase == 0) & pr)
            W = masked_ls(Hf, T, Hg, free, a.lam)
            Wb = masked_ls(Hf, T, Hg, free, a.lam, box=Wbase_alpha(Wbase))
            fit = Hf @ W.T
            stats["target_energy"] += float(T.pow(2).sum())
            stats["residual_energy"] += float((fit - T).pow(2).sum())
            if on_g.get(ei, {}).get("y") is not None and off_g[ei]["y"] is not None and \
                    on_g[ei]["y"].shape == off_g[ei]["y"].shape:
                stats["lora_general_energy"] += float((on_g[ei]["y"] - off_g[ei]["y"]).pow(2).sum())
            stats["fit_general_energy"] += float((Hg @ W.T).pow(2).sum()) if Hg.shape[0] else 0.0
            stats["fit_experts"] += 1
            stats["box_residual_energy"] += float((Hf @ Wb.T - T).pow(2).sum())
            fits[f"{pref}experts.{ei}.down_proj"] = (W.cpu(), (Hf.T @ Hf + a.lam * Hg.T @ Hg).cpu(), Wb.cpu())
        del on_f, on_g, off_f, off_g
        torch.cuda.empty_cache()
        print(f"layer {li}: {int(stats['fit_experts'])} experts fit ({time.time()-t0:.0f}s)", flush=True)
    res["fit"] = dict(stats, explained_frac=1 - stats["residual_energy"] / max(stats["target_energy"], 1e-30),
                      box_explained_frac=1 - stats["box_residual_energy"] / max(stats["target_energy"], 1e-30),
                      general_ratio_fit_vs_lora=stats["fit_general_energy"] / max(stats["lora_general_energy"], 1e-30))
    print("FIT", json.dumps(res["fit"]), flush=True)

    # ladder: unbounded continuous -> box-constrained continuous (|w| <= alpha) -> ternary GPTQ at scale s*alpha
    E.detach_all(model)
    meth = {"any_zero": "fsa_full", "pair8_forced": "fsa_forced"}.get(a.mask, "fsa_free")
    variants = [("legal_cont", "bf16", None, 1.0), ("legal_box", "bf16", "box", 1.0)] + \
        [(f"legal_tern_s{m_:g}", "ternary", None, m_) for m_ in a.scale_mults]
    for tag, fmt_, which, mult in variants:
        E.attach(model, layers, meth, 0, fmt_, ("down_proj",))
        nnz = fallback = 0
        for n, lin in E.patched_linears(model):
            pm = lin.parametrizations.weight[0]
            pm.M.zero_()
            if fmt_ == "ternary":
                with torch.no_grad():
                    pm.alpha.mul_(mult)                    # patch-only scale for EVERY module (export consistency)
            if n not in fits:
                continue
            W, H, Wb = fits[n]
            W = (Wb if which == "box" else W).to(dev)
            if fmt_ == "bf16":
                pm.M.copy_(W / pm.alpha.float().clamp_min(1e-12) * pm.mask)
                continue
            Hm = H.to(dev) / max(float(H.diagonal().mean()), 1e-30)
            Hm += a.gptq_damp * torch.eye(Hm.shape[0], device=dev)
            try:
                Hu = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
                codes = LP.gptq_quantize(W, pm.alpha, pm.mask, Hu)
            except RuntimeError:
                fallback += 1
                codes = torch.clamp(torch.round(W / pm.alpha.float().clamp_min(1e-12)), -1, 1) * pm.mask
            pm.M.copy_(codes)
            nnz += int((codes != 0).sum())
        r = evaluate(tag)
        if fmt_ == "ternary":
            r.update(slots=nnz, gptq_fallback_modules=fallback, scale_mult=mult)
            # save as an e2e_fsa patch (down_proj only) so it can seed refinement with --load-patch
            pmeta = dict(method=meth, fmt="ternary", layers=layers, projs=["down_proj"], rank=16, seed=meta["seed"],
                         n_experts=0, selection=None, fp4_step=0.25, patch_scale="row", scale_block=0,
                         people=meta.get("people", 1), facts_per_person=meta.get("facts_per_person", len(E.ATTRS)),
                         shared_attrs=meta.get("shared_attrs", 0), values=meta.get("values", "disjoint"), budget=0,
                         shared_expert=False, shared_only=False, sparse_state=False, occupied_flips=False,
                         slot_frac=1.0, patch_scale_mult=mult, fold_mode=None, fold_every=None, fold_thr=None,
                         fold_mask=None, fold_mask_frac=None, no_pareto=False, export=str(E.EXPORT),
                         alpha="mean |nonzero base weight| of the row", source=f"ls_diag {tag} lam={a.lam}")
            pf = Path(a.out).with_name(Path(a.out).stem + f"_{tag}.pt")
            r["patch_file"] = str(pf)
            r["patch_file_bytes"] = E.save_patch(model, pf, pmeta)
        res[tag] = r
        E.detach_all(model)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
