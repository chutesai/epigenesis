"""FSA on REAL lambda experts: can the in-active-pair reserve (free 0->+-1 flips) carry new knowledge,
and at what cost to the base expert function?  Base = the real decoded ternary expert (frozen);
reserve = the ~14% in-active-pair zeros, trained as trits at the frozen per-row scale.

Task mirrors the synthetic assay but with a REAL expert as the base MLP: a frozen random head turns the
expert's 384-d output into V-way logits; N facts are random input->label pairs. We report:
  acquisition K_B (bits the reserve learned), interference (how much the base function moves when the
  reserve is ON), and exact-revoke (reserve->0 restores the base bit-exactly)."""
import argparse, json, math, time
import torch
import torch.nn.functional as F


def relu2(x): return F.relu(x).pow(2)
def ste(q, w): return (q - w).detach() + w


def expert_out(x, E, use_reserve):
    def proj(xin, t):
        W = E[t]["W"]
        if use_reserve:
            r = E[t]["res_master"] * E[t]["reserve"]
            if E[t]["ternary"]:
                s = E[t]["row_scale"]
                q = torch.sign(r) * (r.abs() > 0.5 * s).float() * s   # ternary flip at frozen row scale
                W = W + ste(q, r)
            else:
                W = W + r                                            # fp reserve (max capacity)
        return xin @ W.t()
    h = relu2(proj(x, "up_proj"))
    return proj(h, "down_proj")


def facts(n, d, V, seed, device):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g); x /= x.norm(dim=1, keepdim=True)
    y = torch.randint(0, V, (n,), generator=g)
    return x.to(device), y.to(device)


def kfacts(logits, y, V):
    ce = F.cross_entropy(logits, y, reduction="sum").item()
    K = y.numel() * math.log2(V) - ce / math.log(2)
    acc = (logits.argmax(1) == y).float().mean().item()
    return K, acc


def run(dump_path, key, n_b, V, steps, lr, seed, device, reserve_mode="full", ternary=True):
    torch.manual_seed(seed)
    raw = torch.load(dump_path, map_location=device)
    rawent = raw[key]
    ent = {("up_proj" if "up_proj" in k else "down_proj"): v for k, v in rawent.items()}
    d_in = ent["up_proj"]["W"].shape[1]        # 384
    d_out = ent["down_proj"]["W"].shape[0]     # 384
    E = {}
    for t in ("up_proj", "down_proj"):
        W = ent[t]["W"].float().to(device)
        # reserve: 'free' = in-active-pair zeros (stays pair8, 0 extra storage);
        #          'full' = ALL structural zeros (leaves pair8 -> sidecar, costs storage, ~5x bigger)
        resmask = ent[t]["reserve"].to(device).float() if reserve_mode == "free" else (W == 0).float()
        E[t] = dict(W=W, reserve=resmask, ternary=ternary,
                    row_scale=ent[t]["row_scale"].float().to(device),
                    res_master=torch.zeros_like(W, requires_grad=True))
    head = (torch.randn(V, d_out, generator=torch.Generator().manual_seed(seed + 1)) / math.sqrt(d_out)).to(device)
    kB, yB = facts(n_b, d_in, V, seed * 3 + 2, device)
    probe, _ = facts(4096, d_in, V, seed * 3 + 9, device)      # held-out inputs for interference

    base_probe = expert_out(probe, E, use_reserve=False).detach()   # base function on held-out inputs
    opt = torch.optim.AdamW([E["up_proj"]["res_master"], E["down_proj"]["res_master"]], lr=lr, betas=(0.9, 0.95))
    for s in range(steps):
        idx = torch.randint(0, n_b, (min(16384, n_b),), device=device)
        logits = expert_out(kB[idx], E, use_reserve=True) @ head.t()
        loss = F.cross_entropy(logits, yB[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        KB, accB = kfacts(expert_out(kB, E, use_reserve=True) @ head.t(), yB, V)
        # interference: relative change of the base function on held-out inputs when reserve is ON
        on_probe = expert_out(probe, E, use_reserve=True)
        interf = (on_probe - base_probe).norm() / base_probe.norm().clamp_min(1e-9)
        # mechanistic grounding: reserve perturbation norm relative to base weight, per layer
        def relnorm(t):
            r = E[t]["res_master"] * E[t]["reserve"]
            if E[t]["ternary"]:
                s = E[t]["row_scale"]; q = torch.sign(r) * (r.abs() > 0.5 * s).float() * s
            else:
                q = r
            return (q.norm() / E[t]["W"].norm().clamp_min(1e-9)).item()
        relU, relD = relnorm("up_proj"), relnorm("down_proj")
        # interference on a STRUCTURED (low-rank) input distribution (proxy for real routed inputs)
        kdim = 32
        Pb = torch.randn(d_in, kdim, device=device); Pb /= Pb.norm(dim=0, keepdim=True)
        lowx = torch.randn(4096, kdim, device=device) @ Pb.t(); lowx /= lowx.norm(dim=1, keepdim=True)
        bl = expert_out(lowx, E, use_reserve=False)
        interf_lowrank = ((expert_out(lowx, E, use_reserve=True) - bl).norm() / bl.norm().clamp_min(1e-9)).item()
        # exact revoke: reserve OFF must equal the original base bit-for-bit
        revoke_ok = torch.equal(expert_out(probe, E, use_reserve=False), base_probe)
        filled = int((torch.sign(E["up_proj"]["res_master"]) * (E["up_proj"]["res_master"].abs() > 0.5 * E["up_proj"]["row_scale"]).float()
                      != 0).sum() + (torch.sign(E["down_proj"]["res_master"]) * (E["down_proj"]["res_master"].abs() > 0.5 * E["down_proj"]["row_scale"]).float() != 0).sum())
        res_total = int(E["up_proj"]["reserve"].sum() + E["down_proj"]["reserve"].sum())
    return dict(key=key, n_b=n_b, KB=KB, accB=accB, interference=float(interf),
                interf_lowrank=interf_lowrank, relnorm_up=relU, relnorm_down=relD,
                revoke_exact=bool(revoke_ok), reserve_slots=res_total, reserve_filled=filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--key", default="moe0.expert0")
    ap.add_argument("--facts-b", default="100,500,2000,8000,32000")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reserve", choices=["free", "full"], default="full")
    ap.add_argument("--fp", action="store_true", help="fp reserve instead of ternary (max capacity)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for nb in [int(x) for x in a.facts_b.split(",")]:
        t = time.time()
        r = run(a.dump, a.key, nb, a.vocab, a.steps, a.lr, a.seed, dev, reserve_mode=a.reserve, ternary=not a.fp)
        r["reserve_mode"] = a.reserve; r["ternary"] = not a.fp
        r["secs"] = time.time() - t; out.append(r); print(json.dumps(r), flush=True)
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
