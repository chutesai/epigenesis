"""Knowledge-capacity assay for extreme weight formats — the binary vs pair8 question.

Implements a matched-byte QAT capacity-crossover:

  * Synthetic fact storage. N random isotropic keys -> uniform labels in {0..V-1}.
    An MLP "expert" memorizes (key -> label). Retained knowledge, evaluated on the
    DECODED (quantized) weights over all keys, is the variational lower bound
        K_facts = N*log2(V) - sum_i CE_i / ln2        (bits)
    k = K/P (per weight), kappa = K/B (per STORED bit). kappa is the primary metric.

  * QAT, from scratch. Forward uses the quantized weights via straight-through;
    masters are fp32; SR-STE decays masters outside the kept support each step.
    Formats are measured on their exported/decoded codes, never the masters.

  * Formats (one arm per GPU):
      bf16        reference ceiling (no quant)
      ternary     dense ternary + SR-STE                       ~1.585 b/w + row scale
      pair8       ternary, <=2 of 4 aligned pairs active /8 + SR-STE   1.125 b/w + row scale   [BASELINE]
      v1          signed binary {-1,+1} + per-row scale          1.000 b/w + row scale
      v2g4/v2g8/v2g16   {0,1} occupancy + ONE sign per block of g + shared row magnitude
                                                                 1 + 1/g b/w + row scale
      v2g64_s8    {0,1} occupancy + per-block signed 8-bit scale, g=64   1 + 8/64 = 1.125 b/w
      v3          two signed planes a1*b1 + a2*b2 (2 row scales)  2.000 b/w + 2 row scales
      twobit      symmetric 4-level codebook + per-row scale (V3 control)  2.000 b/w + row scale

  pair8 vs v2g8 are byte-identical by construction (9 packed bits / 8 weights + one row
  scale): the decisive pair. Everything else is compared on kappa = K/B.

Instrumentation (the deciding statistics both reviewers asked for):
  * per-block minority-sign ENERGY  r_min = min(sum_+ w^2, sum_- w^2)/sum w^2, by layer and g
  * block-sign churn, support churn, scale "collapse", recall, learning curve.

Run ONE arm (format) per process, pinned to one GPU; launch_8.sh fans 8 arms over 8 GPUs.
TOY/PROXY: an isolated MLP fact assay, not a full-model or LLM-quality claim (per both reviews).
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- quantizers
# Each returns (W_q, keep_mask, info). W_q is the decoded tensor used in the forward
# pass; keep_mask marks masters that SR-STE should NOT decay (the kept support);
# info carries per-tensor stored-bits and diagnostics. Row = output dim; the per-row
# scale is over the input dim, matching "row scale" in the reviews.

def _row_absmean(W, mask=None):
    A = W.abs() if mask is None else (W.abs() * mask)
    d = (mask.sum(1, keepdim=True) if mask is not None else torch.full((W.shape[0], 1), W.shape[1], device=W.device))
    return (A.sum(1, keepdim=True) / d.clamp_min(1)).clamp_min(1e-12)

def q_bf16(W):
    return W.to(torch.bfloat16).float(), torch.ones_like(W), {"bpw": 16.0}

def q_ternary(W):
    thr = 0.7 * W.abs().mean(1, keepdim=True)
    m = (W.abs() > thr).float()
    s = _row_absmean(W, m)
    t = torch.sign(W) * m
    return s * t, m, {"bpw": math.log2(3)}

def q_v1(W):                                   # signed binary {-1,+1}, no zero
    s = _row_absmean(W)
    return s * torch.sign(W), torch.ones_like(W), {"bpw": 1.0}

def _pairs8(W):
    # group the input dim into blocks of 8 = 4 adjacent pairs; keep the top-2 pairs by
    # energy; ternarize within kept pairs. per-row absmean scale over kept weights.
    O, I = W.shape
    g = 8; P = I // g
    Wb = W[:, :P * g].reshape(O, P, 4, 2)            # (O, nblk, 4 pairs, 2)
    pe = Wb.pow(2).sum(-1)                            # pair energy (O, nblk, 4)
    topv, topi = pe.topk(2, dim=-1)                  # keep 2 of 4 pairs
    pair_keep = torch.zeros_like(pe).scatter_(-1, topi, 1.0)   # (O,nblk,4)
    wkeep = pair_keep.unsqueeze(-1).expand_as(Wb)    # (O,nblk,4,2)
    thr = 0.7 * (Wb.abs() * wkeep).sum((1, 2, 3), keepdim=True) / wkeep.sum((1, 2, 3), keepdim=True).clamp_min(1)
    tern = (Wb.abs() > thr).float() * wkeep
    mask = torch.zeros_like(W)
    mask[:, :P * g] = (tern * torch.ones_like(Wb)).reshape(O, P * g)  # carried-nonzero mask
    keep = torch.zeros_like(W)
    keep[:, :P * g] = wkeep.reshape(O, P * g)        # SR-STE keeps whole active pairs
    s = _row_absmean(W, mask)
    Wq = torch.zeros_like(W)
    Wq[:, :P * g] = (s.view(O, 1, 1, 1) * torch.sign(Wb) * tern).reshape(O, P * g)
    # 9 packed bits / 8 weights (417 states < 512=2^9) + one row scale
    return Wq, keep, {"bpw": 9.0 / 8.0}

def _v2(W, g, scale="row"):
    # {0,1} occupancy (top-50% energy within block) + ONE sign per block + magnitude.
    # scale="row": shared per-row absmean; scale="s8": per-block signed 8-bit scale.
    O, I = W.shape
    nb = I // g
    Wb = W[:, :nb * g].reshape(O, nb, g)
    k = max(1, g // 2)
    thr = Wb.abs().kthvalue(g - k + 1, dim=-1, keepdim=True).values       # top-k by |w|
    u = (Wb.abs() >= thr).float()                                          # occupancy
    sel = Wb * u
    sgn = torch.sign(sel.sum(-1, keepdim=True))                            # block sign = sign of net signed energy
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    if scale == "row":
        rmask = torch.zeros_like(W); rmask[:, :nb * g] = u.reshape(O, nb * g)
        a = _row_absmean(W, rmask).unsqueeze(1)                           # (O,1,1) per row
        bpw = 1.0 + 1.0 / g                                              # occ bit + per-block sign bit
    else:  # per-block signed 8-bit scale (abs magnitude 8-bit; sign carried by sgn)
        amag = (sel.abs().sum(-1, keepdim=True) / u.sum(-1, keepdim=True).clamp_min(1))
        amax = amag.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        a = (torch.round(amag / amax * 127) / 127 * amax)                 # 8-bit magnitude per block
        bpw = 1.0 + 8.0 / g
    Wq = torch.zeros_like(W)
    Wq[:, :nb * g] = (a * sgn * u).reshape(O, nb * g)
    keep = torch.zeros_like(W); keep[:, :nb * g] = u.reshape(O, nb * g)
    # minority-sign energy per block (the deciding statistic), on the TARGET weights
    pos = (sel.clamp_min(0)).pow(2).sum(-1); neg = (sel.clamp_max(0)).pow(2).sum(-1)
    tot = (pos + neg).clamp_min(1e-20)
    r_min = (torch.minimum(pos, neg) / tot).mean().item()
    return Wq, keep, {"bpw": bpw, "r_min_energy": r_min, "block_sign": sgn.reshape(O, nb)}

def q_v2g4(W):  return _v2(W, 4)
def q_v2g8(W):  return _v2(W, 8)
def q_v2g16(W): return _v2(W, 16)
def q_v2g64_s8(W): return _v2(W, 64, scale="s8")

def q_v3(W):                                   # two signed planes (ABC-style); 2 row scales
    s1 = _row_absmean(W); b1 = torch.sign(W)
    r = W - s1 * b1
    s2 = _row_absmean(r); b2 = torch.sign(r)
    return s1 * b1 + s2 * b2, torch.ones_like(W), {"bpw": 2.0, "scales": 2}

def q_twobit(W):                               # symmetric 4-level {-3,-1,+1,+3}*s, no zero, row scale
    s = W.abs().amax(1, keepdim=True).clamp_min(1e-12) / 3.0
    lv = torch.clamp(2 * torch.round((W / s - 1) / 2) + 1, -3, 3)           # nearest odd in {-3,-1,1,3}
    return s * lv, torch.ones_like(W), {"bpw": 2.0}

def q_pow2(W):                                 # {0,+-1,+-2,+-4}*s: multiplier-free (shift+add), HAS zero
    s = _row_absmean(W)
    a = (W / s).abs()
    lvl = torch.where(a < 0.5, torch.zeros_like(a),
          torch.where(a < 1.5, torch.ones_like(a),
          torch.where(a < 3.0, 2 * torch.ones_like(a), 4 * torch.ones_like(a))))
    keep = (lvl > 0).float()
    return s * torch.sign(W) * lvl, keep, {"bpw": math.log2(7)}             # 7 states -> 2.807 b/w ceiling

QUANT = {"bf16": q_bf16, "ternary": q_ternary, "pair8": _pairs8,
         "v1": q_v1, "v2g4": q_v2g4, "v2g8": q_v2g8, "v2g16": q_v2g16,
         "v2g64_s8": q_v2g64_s8, "v3": q_v3, "twobit": q_twobit, "pow2": q_pow2}

SRSTE_FORMATS = {"ternary", "pair8", "v2g4", "v2g8", "v2g16", "v2g64_s8", "pow2"}   # have a drop set
ROWSCALE_BITS = 16  # fp16 per-row scale (amortized over the input dim); v3 carries 2, pair8/v2/ternary 1


def stored_bits(fmt, weight_shapes):
    """exact exported bits: payload b/w * P + per-row scale bits, summed over quantized tensors."""
    bpw = {"bf16": 16.0, "ternary": math.log2(3), "pair8": 9.0 / 8.0, "v1": 1.0,
           "v2g4": 1.25, "v2g8": 1.125, "v2g16": 1.0 + 1 / 16, "v2g64_s8": 1.125,
           "v3": 2.0, "twobit": 2.0, "pow2": math.log2(7)}[fmt]
    nscale = 2 if fmt == "v3" else (0 if fmt == "bf16" else 1)
    B = 0.0
    for (O, I) in weight_shapes:
        B += bpw * O * I + nscale * ROWSCALE_BITS * O
        if fmt == "v2g64_s8":                 # per-block 8-bit magnitude replaces row scale cost already in bpw
            B += 0
    return B


# ---------------------------------------------------------------- QAT linear
class QATLinear(torch.nn.Module):
    def __init__(self, din, dout, fmt):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dout, din).normal_(0, 1.0 / math.sqrt(din)))
        self.bias = torch.nn.Parameter(torch.zeros(dout))
        self.fmt = fmt; self.last_info = {}

    def forward(self, x):
        Wq, keep, info = QUANT[self.fmt](self.weight)
        self.last_keep = keep; self.last_info = info
        Wst = (Wq - self.weight).detach() + self.weight            # straight-through
        return F.linear(x, Wst, self.bias)


class Expert(torch.nn.Module):
    def __init__(self, din, hid, vocab, fmt):
        super().__init__()
        self.l1 = QATLinear(din, hid, fmt)
        self.l2 = QATLinear(hid, hid, fmt)
        self.l3 = QATLinear(hid, vocab, fmt)

    def forward(self, x):
        x = F.gelu(self.l1(x)); x = F.gelu(self.l2(x)); return self.l3(x)

    def qlayers(self): return [self.l1, self.l2, self.l3]


# ---------------------------------------------------------------- the assay
def kfacts(model, keys, labels, V, bs=8192):
    model.eval(); ce = 0.0; correct = 0
    with torch.no_grad():
        for i in range(0, keys.shape[0], bs):
            lg = model(keys[i:i + bs])
            ce += F.cross_entropy(lg, labels[i:i + bs], reduction="sum").item()
            correct += (lg.argmax(1) == labels[i:i + bs]).sum().item()
    N = keys.shape[0]
    K = N * math.log2(V) - ce / math.log(2)     # bits retained (variational LB)
    return K, correct / N, ce / N


def run_arm(fmt, n_facts, seed, din, hid, V, steps, lr, srste_lambda, device):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    keys = torch.randn(n_facts, din, generator=gen); keys /= keys.norm(dim=1, keepdim=True)
    labels = torch.randint(0, V, (n_facts,), generator=gen)
    keys = keys.to(device); labels = labels.to(device)
    model = Expert(din, hid, V, fmt).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    P = sum(l.weight.numel() for l in model.qlayers())
    B = stored_bits(fmt, [tuple(l.weight.shape) for l in model.qlayers()])
    curve = []; bs = min(n_facts, 16384)
    churn_prev = {}
    for step in range(steps):
        idx = torch.randint(0, n_facts, (bs,), device=device)
        loss = F.cross_entropy(model(keys[idx]), labels[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if fmt in SRSTE_FORMATS and srste_lambda > 0:          # SR-STE: decay masters outside kept support
            with torch.no_grad():
                for l in model.qlayers():
                    drop = 1.0 - l.last_keep
                    l.weight.add_(l.weight * drop, alpha=-lr * srste_lambda)
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            K, rec, ce = kfacts(model, keys, labels, V)
            curve.append((step, K, rec, ce)); model.train()
    # final diagnostics on decoded weights
    K, rec, ce = kfacts(model, keys, labels, V)
    rmin = [];
    for l in model.qlayers():
        _, _, info = QUANT[fmt](l.weight)
        if "r_min_energy" in info: rmin.append(info["r_min_energy"])
    return {"fmt": fmt, "n_facts": n_facts, "seed": seed, "steps": steps, "lr": lr,
            "srste_lambda": srste_lambda, "P": P, "B_bits": B, "B_bytes": B / 8,
            "K_facts_bits": K, "k_per_weight": K / P, "kappa_per_bit": K / B,
            "recall": rec, "ce_nats": ce, "bpw_eff": B / P,
            "minority_sign_energy": (sum(rmin) / len(rmin) if rmin else None),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmt", required=True, choices=list(QUANT))
    ap.add_argument("--din", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--facts", type=str, default="4000,8000,16000,32000,64000")  # capacity sweep
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--srste-lambda", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for nf in [int(x) for x in a.facts.split(",")]:
        for sd in [int(x) for x in a.seeds.split(",")]:
            t = time.time()
            r = run_arm(a.fmt, nf, sd, a.din, a.hid, a.vocab, a.steps, a.lr, a.srste_lambda, device)
            r["secs"] = time.time() - t; r["device"] = str(device)
            out.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
