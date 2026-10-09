"""PackNet-style continual learning in the enforced-sparsity free slots (owner's idea).

Question: if ~50% of weight slots are structurally free (pair8's zeros), can we FREEZE the
learned (active) weights and pour NEW knowledge into the free slots — acquiring task B while
barely forgetting task A, with no RL / rehearsal? (arXiv 1711.05769, mapped onto our sparsity.)

Protocol (two disjoint synthetic fact sets A, B; same assay as kappa_binary):
  phase 1: learn A.  phase 2: learn B.  Measure K_A before vs after phase 2 (RETENTION /
  forgetting) and K_B after (ACQUISITION). Dense weights + a fixed per-tensor mask isolate the
  MECHANISM (format-independent); a pair8 follow-up tests the actual format.

Modes (one per GPU):
  packnet        phase1: train only the A-slots (frac_a of weights); phase2: FREEZE A-slots,
                 train only the free B-slots.  -> should retain A ~fully and still learn B.
  fullft         phase1: dense A; phase2: dense B, nothing frozen. -> catastrophic-forgetting baseline.
  frozen_noslots phase1: A-slots; phase2: A-slots FROZEN, no free slots opened. -> B can't be learned
                 (control: shows the free slots are what enables acquisition).
  joint          train A+B together. -> upper bound on both.
frac_a sweeps the base/free split (0.25/0.5/0.75).
"""
import argparse, json, math, time
import torch
import torch.nn.functional as F
import kappa_binary as kb


class MaskedMLP(torch.nn.Module):
    def __init__(self, din, hid, V, gen, frac_a, device):
        super().__init__()
        self.l = torch.nn.ModuleList([torch.nn.Linear(din, hid), torch.nn.Linear(hid, hid), torch.nn.Linear(hid, V)])
        self.maskA = []
        for lin in self.l:
            torch.nn.init.normal_(lin.weight, 0.0, 1.0 / math.sqrt(lin.weight.shape[1])); torch.nn.init.zeros_(lin.bias)
            m = (torch.rand(lin.weight.shape, generator=gen) < frac_a).to(device)   # True = A/base slot
            self.maskA.append(m)
        self.to(device)
        self.live = [torch.ones_like(m) for m in self.maskA]   # which weights contribute in forward
        self.lora = None; self.use_lora = False

    def set_live(self, which):   # 'A' | 'B' | 'all'
        for i, mA in enumerate(self.maskA):
            if which == 'A':   self.live[i] = mA
            elif which == 'all': self.live[i] = torch.ones_like(mA)
            elif which == 'AorB': self.live[i] = torch.ones_like(mA)

    def add_lora(self, r, device, alpha=2.0):   # standard LoRA: W + (alpha/r)*B@A, A~Kaiming, B=0
        self.lora = []; self.lora_scale = alpha * 1.0  # effective scale on the unit-ish B@A (alpha here acts like alpha with r folded into A's init)
        for j, lin in enumerate(self.l):
            A = torch.nn.Parameter(torch.randn(r, lin.weight.shape[1], device=device) / math.sqrt(lin.weight.shape[1]))
            B = torch.nn.Parameter(torch.zeros(lin.weight.shape[0], r, device=device))
            self.register_parameter(f'loA{j}', A); self.register_parameter(f'loB{j}', B)
            self.lora.append((A, B))
        self.use_lora = True
    def lora_params(self):
        return [p for AB in self.lora for p in AB]

    def _W(self, i, lin):
        w = lin.weight * self.live[i]
        if self.use_lora and self.lora is not None:
            w = w + getattr(self, 'lora_scale', 2.0) * (self.lora[i][1] @ self.lora[i][0])
        return w
    def forward(self, x):
        x = F.gelu(F.linear(x, self._W(0, self.l[0]), self.l[0].bias))
        x = F.gelu(F.linear(x, self._W(1, self.l[1]), self.l[1].bias))
        return F.linear(x, self._W(2, self.l[2]), self.l[2].bias)
    def qlayers(self): return list(self.l)
    def zero_nonA(self):  # hold B-slots at exactly 0 during phase 1
        with torch.no_grad():
            for i, lin in enumerate(self.l): lin.weight.mul_(self.maskA[i])
    def freeze_grads(self, which, base_scale=0.0):  # 'A': freeze base (A-slots+biases); base_scale>0 => SOFT-freeze
        for i, lin in enumerate(self.l):
            if lin.weight.grad is not None:
                if which == 'A':   # keep B-slot grads at full; scale A-slot grads by base_scale (0 = hard freeze)
                    keep = (~self.maskA[i]).to(lin.weight.dtype) + base_scale * self.maskA[i].to(lin.weight.dtype)
                    lin.weight.grad.mul_(keep)
                elif which == 'B': lin.weight.grad.mul_(self.maskA[i])
            if which == 'A' and lin.bias.grad is not None:
                lin.bias.grad.mul_(base_scale)   # biases are shared base state -> scale with the base

    def reserve_stats(self, bits_per_weight=1.088):
        """Honest de-sparsification cost: how many reserve (Z0) slots became non-zero, and the bits that costs."""
        filled = 0; total = 0
        with torch.no_grad():
            for i, lin in enumerate(self.l):
                Z = ~self.maskA[i]
                filled += int(((lin.weight[Z].abs() > 1e-6)).sum().item())
                total += int(lin.weight.numel())
        return dict(reserve_filled=filled, desparsify_bits=filled * bits_per_weight,
                    mask_bits_arbitrary=total, mask_bits_structural=0)


def facts(n, din, V, seed, device):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n, din, generator=g); k /= k.norm(dim=1, keepdim=True)
    y = torch.randint(0, V, (n,), generator=g)
    return k.to(device), y.to(device)


def train(model, keys, labels, steps, lr, freeze, device, bs=16384):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    for s in range(steps):
        idx = torch.randint(0, keys.shape[0], (min(bs, keys.shape[0]),), device=device)
        loss = F.cross_entropy(model(keys[idx]), labels[idx])
        opt.zero_grad(); loss.backward()
        if freeze: model.freeze_grads(freeze)
        opt.step()


BASE_SLOT_MODES = ('packnet', 'frozen_noslots', 'soft', 'softfreeze')  # need a real reserve (A-slots + held-zero B-slots)


def run(mode, frac_a, n_a, n_b, seed, din, hid, V, steps, lr, device, r=16, lam=1.0, base_scale=0.05):
    torch.manual_seed(seed)                       # global seed: weight init + minibatch sampling
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed + 7)
    model = MaskedMLP(din, hid, V, gen, frac_a if mode in BASE_SLOT_MODES else 1.0, device)
    kA, yA = facts(n_a, din, V, seed * 2 + 1, device)
    kB, yB = facts(n_b, din, V, seed * 2 + 2, device)
    P = sum(l.weight.numel() for l in model.qlayers())

    if mode == 'joint':
        k = torch.cat([kA, kB]); y = torch.cat([yA, yB]); model.set_live('all')
        train(model, k, y, steps, lr, None, device)
        KA, rA, _ = kb.kfacts(model, kA, yA, V); KB, rB, _ = kb.kfacts(model, kB, yB, V)
        return dict(mode=mode, frac_a=frac_a, KA_after=KA, KB_after=KB, recA=rA, recB=rB, KA_before=KA, retention=1.0, P=P)

    # phase 1: learn A
    model.set_live('A' if mode in BASE_SLOT_MODES else 'all')
    if mode in BASE_SLOT_MODES: model.zero_nonA()
    train(model, kA, yA, steps, lr, None, device)
    KA_before, recA_before, _ = kb.kfacts(model, kA, yA, V)

    # phase 2: learn B
    if mode == 'packnet':
        model.set_live('all')          # open the free slots
        train(model, kB, yB, steps, lr, 'A', device)   # freeze A-slots -> only B-slots move
    elif mode == 'fullft':
        model.set_live('all')
        train(model, kB, yB, steps, lr, None, device)  # nothing frozen -> overwrites A
    elif mode == 'frozen_noslots':
        model.set_live('A')            # no free slots opened; A frozen
        train(model, kB, yB, steps, lr, 'A', device)   # grads only on B-slots, but B-slots not live -> ~no learning
    elif mode == 'softfreeze':         # MIDDLE GROUND 1: base gets a tiny LR -> it can refine (two-way cross-pollination)
        model.set_live('all')
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        model.train()
        for s in range(steps):
            ib = torch.randint(0, kB.shape[0], (min(16384, kB.shape[0]),), device=device)
            loss = F.cross_entropy(model(kB[ib]), yB[ib])
            opt.zero_grad(); loss.backward(); model.freeze_grads('A', base_scale=base_scale); opt.step()
    elif mode == 'soft':               # MIDDLE GROUND 2: frozen base + consistency penalty on A -> run UNMASKED, no task-ID
        model.set_live('all')
        with torch.no_grad():
            model.set_live('A'); teachA = model(kA).detach(); model.set_live('all')  # frozen base-only logits on A
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        model.train()
        for s in range(steps):
            ib = torch.randint(0, kB.shape[0], (min(16384, kB.shape[0]),), device=device)
            ia = torch.randint(0, kA.shape[0], (min(16384, kA.shape[0]),), device=device)
            lossB = F.cross_entropy(model(kB[ib]), yB[ib])
            lossC = F.kl_div(F.log_softmax(model(kA[ia]), 1), F.softmax(teachA[ia], 1), reduction='batchmean')
            loss = lossB + lam * lossC
            opt.zero_grad(); loss.backward(); model.freeze_grads('A', base_scale=0.0); opt.step()
    elif mode == 'lora':
        model.add_lora(r, device); model.set_live('all')   # base frozen; train only the rank-r adapter on B
        opt = torch.optim.AdamW(model.lora_params(), lr=lr, betas=(0.9, 0.95))
        model.train()
        for s in range(steps):
            idx = torch.randint(0, kB.shape[0], (min(16384, kB.shape[0]),), device=device)
            loss = F.cross_entropy(model(kB[idx]), yB[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # A-retention two ways: with the per-domain "mask off" (true frozen retention) and the domain on (perturbed)
    if mode == 'lora':
        model.use_lora = False; model.set_live('all'); KA_mask, recA_mask, _ = kb.kfacts(model, kA, yA, V)  # base only (adapter off)
        model.use_lora = True;  KA_all, recA_all, _ = kb.kfacts(model, kA, yA, V)   # base + adapter on
        KB_after, recB_after, _ = kb.kfacts(model, kB, yB, V)
        extra_bits = r * sum(lin.weight.shape[0] + lin.weight.shape[1] for lin in model.l) * 16
    else:
        model.set_live('A');   KA_mask, recA_mask, _ = kb.kfacts(model, kA, yA, V)
        model.set_live('all'); KA_all, recA_all, _ = kb.kfacts(model, kA, yA, V)
        KB_after, recB_after, _ = kb.kfacts(model, kB, yB, V)   # B evaluated with all slots live
        extra_bits = 0
    rstats = model.reserve_stats() if mode in BASE_SLOT_MODES else {}  # honest de-sparsification cost (bits written into Z0)
    return dict(mode=mode, frac_a=frac_a, n_a=n_a, n_b=n_b, seed=seed, P=P, lora_rank=(r if mode == 'lora' else 0),
                KA_before=KA_before, recA_before=recA_before, extra_storage_bits=extra_bits,
                KA_mask=KA_mask, recA_mask=recA_mask, retention_mask=KA_mask / max(KA_before, 1),
                KA_all=KA_all, recA_all=recA_all, retention_all=KA_all / max(KA_before, 1),
                KB_after=KB_after, recB_after=recB_after, **rstats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['packnet', 'fullft', 'frozen_noslots', 'joint', 'lora', 'soft', 'softfreeze'])
    ap.add_argument('--frac-a', type=float, default=0.5)
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--lam', type=float, default=1.0)          # soft: consistency-penalty weight
    ap.add_argument('--base-scale', type=float, default=0.05)  # softfreeze: base LR fraction
    ap.add_argument('--facts-a', type=int, default=128000)   # base load (push to saturation ceiling)
    ap.add_argument('--facts-b', default='16000,32000,64000,128000,256000')  # sweep the reserve to its ceiling
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--din', type=int, default=256); ap.add_argument('--hid', type=int, default=512)
    ap.add_argument('--vocab', type=int, default=1024); ap.add_argument('--steps', type=int, default=40000)  # enough to reach the ceiling
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = []
    for nb in [int(x) for x in a.facts_b.split(',')]:
        for sd in [int(s) for s in a.seeds.split(',')]:
            t = time.time()
            res = run(a.mode, a.frac_a, a.facts_a, nb, sd, a.din, a.hid, a.vocab, a.steps, a.lr, dev,
                      r=a.lora_rank, lam=a.lam, base_scale=a.base_scale)
            res['secs'] = time.time() - t; out.append(res)
            print(json.dumps(res), flush=True)
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
