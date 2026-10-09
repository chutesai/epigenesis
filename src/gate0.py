"""GATE 0 — does the TERNARY reserve carry knowledge, through the real pair8 quantizer?

Both reviewers' #1 experiment. The dense-fp32 mechanism (freeslot.py) says nothing about whether a
*ternary* reserve, placed under the pair8 2-of-4 constraint with FROZEN base trits and scales, actually
memorizes anything. This decides whether Free-Slot Adaptation exists for the format the whole idea rests on.

Faithful construction (respects <=2-of-4): each aligned block of 8 input columns = 4 pairs.
  pair 0 = BASE slot (trained in phase 1, then frozen).   pair 1 = RESERVE slot (trained in phase 2).
  pairs 2,3 = unused (zero).   -> base 25%, reserve 25%; base+reserve = 2 of 4 pairs (in-codebook).
This is exactly why the in-codebook reserve is NOT (1-rho)N: a 2-pair base leaves no room at all.

Arms (one per GPU), same saturate-then-adapt protocol as freeslot.py:
  ternary_shared : base+reserve ternary; reserve uses the FROZEN base per-row scale (strict pair8).
  ternary_own    : reserve gets its OWN per-row scale (a sidecar 'scale override'); otherwise ternary.
  fp32           : same pair partition, NO quantization -> the diagnostic ceiling.
  frozen         : base ternary, reserve never opened -> no-capacity control.
Verdict: ternary_* K_B >= ~1/2 fp32 K_B  => the ternary reserve carries knowledge (Gate 0 PASS).
         ternary_* K_B -> frozen          => FSA does not exist for pair8 (Gate 0 FAIL).
"""
import argparse, json, math, time
import torch
import torch.nn.functional as F
import kappa_binary as kb


def facts(n, din, V, seed, device):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n, din, generator=g); k /= k.norm(dim=1, keepdim=True)
    y = torch.randint(0, V, (n,), generator=g)
    return k.to(device), y.to(device)


def ste(q, w):  # straight-through: value of q, gradient of w
    return (q - w).detach() + w


class PairMLP(torch.nn.Module):
    def __init__(self, dims, device, quant='ternary', own_scale=False, tern_frac=0.5):
        super().__init__()
        self.lin = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        for l in self.lin:
            torch.nn.init.normal_(l.weight, 0.0, 1.0 / math.sqrt(l.weight.shape[1])); torch.nn.init.zeros_(l.bias)
        self.to(device)
        self.quant, self.own_scale, self.frac = quant, own_scale, tern_frac
        self.base_m, self.res_m = [], []
        for l in self.lin:
            O, I = l.weight.shape; P = I // 8
            bm = torch.zeros(O, I, device=device); rm = torch.zeros(O, I, device=device)
            idx = torch.arange(P * 8, device=device).reshape(P, 8)
            bm[:, idx[:, 0:2].reshape(-1)] = 1.0            # pair 0 = base
            rm[:, idx[:, 2:4].reshape(-1)] = 1.0            # pair 1 = reserve
            self.base_m.append(bm); self.res_m.append(rm)
        self.use_reserve = False
        self.base_q = [None] * len(self.lin); self.base_scale = [None] * len(self.lin)

    def _scale(self, i, mask):
        return kb._row_absmean(self.lin[i].weight.detach(), mask)

    def _tern(self, w, s):
        t = torch.sign(w) * (w.abs() > self.frac * s).float()
        return s * t

    def _quant_base(self, i):
        w = self.lin[i].weight * self.base_m[i]
        if self.quant == 'fp32':
            return w
        return ste(self._tern(w, self._scale(i, self.base_m[i])), w)

    def freeze_base(self):
        with torch.no_grad():
            for i in range(len(self.lin)):
                self.base_scale[i] = self._scale(i, self.base_m[i])
                self.base_q[i] = (self.lin[i].weight * self.base_m[i]).clone() if self.quant == 'fp32' \
                    else self._tern(self.lin[i].weight * self.base_m[i], self.base_scale[i])

    def _W(self, i):
        if not self.use_reserve:                      # phase 1 (base trainable) OR base-only eval
            return self._quant_base(i)
        r = self.lin[i].weight * self.res_m[i]        # phase 2: frozen base + trainable reserve
        if self.quant == 'fp32':
            return self.base_q[i] + r
        s = self._scale(i, self.res_m[i]) if self.own_scale else self.base_scale[i]
        return self.base_q[i] + ste(self._tern(r, s), r)

    def forward(self, x):
        h = x
        for i in range(len(self.lin) - 1):
            h = F.gelu(F.linear(h, self._W(i), self.lin[i].bias))
        return F.linear(h, self._W(len(self.lin) - 1), self.lin[-1].bias)

    def qlayers(self):
        return list(self.lin)

    def freeze_grads_base(self):                      # phase 2: zero base-pair + bias grads
        for i, l in enumerate(self.lin):
            if l.weight.grad is not None:
                l.weight.grad.mul_(self.res_m[i])      # keep only reserve-pair grads
            if l.bias.grad is not None:
                l.bias.grad.zero_()

    def reserve_bits(self):                           # sidecar cost: filled reserve trits + address + scale override
        filled = 0
        with torch.no_grad():
            for i in range(len(self.lin)):
                s = self.base_scale[i] if self.base_scale[i] is not None else self._scale(i, self.res_m[i])
                filled += int(((self._tern(self.lin[i].weight * self.res_m[i], s)).abs() > 0).sum().item())
        bits_trit = filled * math.log2(3)             # a pre-assigned ternary pair slot
        return dict(reserve_filled=filled, trit_bits=bits_trit)


def train(model, keys, labels, steps, lr, device, phase2=False, bs=16384):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    for s in range(steps):
        idx = torch.randint(0, keys.shape[0], (min(bs, keys.shape[0]),), device=device)
        loss = F.cross_entropy(model(keys[idx]), labels[idx])
        opt.zero_grad(); loss.backward()
        if phase2: model.freeze_grads_base()
        opt.step()


def run(arm, n_a, n_b, seed, din, hid, V, steps, lr, device):
    torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    quant = 'fp32' if arm == 'fp32' else 'ternary'
    own = (arm == 'ternary_own')
    model = PairMLP([din, hid, hid, V], device, quant=quant, own_scale=own)
    kA, yA = facts(n_a, din, V, seed * 2 + 1, device)
    kB, yB = facts(n_b, din, V, seed * 2 + 2, device)

    model.use_reserve = False                         # phase 1: learn base in pair 0
    train(model, kA, yA, steps, lr, device)
    model.freeze_base()
    KA_before, recA_before, _ = kb.kfacts(model, kA, yA, V)

    if arm != 'frozen':                               # phase 2: learn reserve in pair 1
        model.use_reserve = True
        train(model, kB, yB, steps, lr, device, phase2=True)

    model.use_reserve = False; KA_mask, recA_mask, _ = kb.kfacts(model, kA, yA, V)   # base only
    model.use_reserve = (arm != 'frozen')
    KB_after, recB_after, _ = kb.kfacts(model, kB, yB, V)
    rb = model.reserve_bits()
    return dict(arm=arm, n_a=n_a, n_b=n_b, seed=seed, quant=quant, own_scale=own,
                KA_before=KA_before, recA_before=recA_before,
                KA_mask=KA_mask, recA_mask=recA_mask, retention_mask=KA_mask / max(KA_before, 1),
                KB_after=KB_after, recB_after=recB_after, **rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['ternary_shared', 'ternary_own', 'fp32', 'frozen'])
    ap.add_argument('--facts-a', type=int, default=128000)
    ap.add_argument('--facts-b', default='500,2000,8000,32000,128000,256000')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--din', type=int, default=256); ap.add_argument('--hid', type=int, default=512)
    ap.add_argument('--vocab', type=int, default=1024); ap.add_argument('--steps', type=int, default=40000)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = []
    for nb in [int(x) for x in a.facts_b.split(',')]:
        for sd in [int(s) for s in a.seeds.split(',')]:
            t = time.time()
            res = run(a.arm, a.facts_a, nb, sd, a.din, a.hid, a.vocab, a.steps, a.lr, dev)
            res['secs'] = time.time() - t; out.append(res); print(json.dumps(res), flush=True)
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
