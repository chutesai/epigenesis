# FSA on the real lambda 8B — results and mathematical foundations

Free-Slot Adaptation tested on **real decoded experts** of the lambda 8B (parallax) MoE, not a synthetic
toy. Each expert is a pair8-2:4 ternary MLP (`up_proj` [2304,384] + relu² + `down_proj` [384,2304]) plus a
low-rank carry. We freeze an expert and train new knowledge into its structural-zero reserve. Harness:
[`real_fsa.py`](real_fsa.py); decode: [`decode_dump.py`](decode_dump.py).

## The real reserve (measured on the deployed model)
Per expert (mean over 8 experts): **35.5% active** ternary weights · **13.2% free reserve** (zeros *inside*
active pairs — flippable 0→±1 in-codebook, **zero extra storage**) · **~51% forced-zero pairs** (need a
sidecar — extra storage, leaves pair8). "Full reserve" = all ~65% zeros (sidecar).

## Results (random-fact memorization probe; 3 experts, consistent)

| reserve | trainable params m | extra storage | facts @ ~100% acc | interference I | exact revoke |
|---|---:|---|---:|---:|:--:|
| **free, ternary** | ~232 k | **none** | ~500 | **~1.8 (bounded)** | ✓ bit-exact |
| full, ternary | ~1.14 M | sidecar | ~3–8 k | ~6.7 (bounded) | ✓ |
| free, fp | ~232 k | fp sidecar | ~32 k | 6→100+ (grows) | ✓ |
| full, fp | ~1.14 M | fp sidecar | ~128 k+ | 3→46+ (grows) | ✓ |

**Verdict: the mechanism is validated and lawful.** It *definitely learns* (acc 1.00 at appropriate load,
every run), it *revokes exactly* (100%, a theorem), and both the capacity and the interference obey clean
laws. It is *not* free — ternary is a weak memory, and interference is real.

## The real benefit vs LoRA — CONTROLLED (both methods drift-penalized, matched knowledge)
Earlier drafts here claimed "70–500×". That was an artifact: my LoRA was trained to zero cross-entropy
(inflating its edit), and FSA was compared at lower confidence. Independent adversarial review caught it. The
**controlled** experiment adds a drift penalty `CE + λ·‖f_adapted−f_base‖²` to **both** methods, sweeps λ to
trace each method's interference-vs-knowledge frontier, uses real `α/r` LoRA, and compares at **matched
knowledge bits** ([`fig_controlled_pareto.png`](fig_controlled_pareto.png)). At ~2000 facts (KB≈20k):

| method | min interference | at KB | storage |
|---|---:|---:|---:|
| **FSA full ternary** | **3.0** | 17.5k | 186 KB |
| FSA full fp (high-rank, unclipped control) | 8.1 | 19.2k | 221 KB |
| LoRA r64 (best-interference LoRA) | 22.9 | 19.6k | 672 KB |
| LoRA r16 (storage-matched) | 59.0 | 19.4k | 168 KB |

**Controlled result: ~8–20× lower interference at matched knowledge** — ~20× at matched storage (3.0 vs 59),
~8× vs the best-interference LoRA (which also needs 3.6× more storage). This **survives** giving LoRA the
same drift penalty, so it is **not** the over-training artifact.

**Mechanism (corrected — the `r²` is NOT from relu²).** Interference tracks the weight-perturbation norm
`r = ‖Δ‖/‖W‖`: FSA reaches `r≈1.3` vs LoRA's `r≈9–10` for the same facts. The gap is ~half from the **ternary
magnitude clip** (ternary-FSA 3.0 vs fp-FSA 8.1) and ~half from **high-rank sparse spreading** (fp-FSA 8.1 vs
LoRA 22.9). A φ=identity control shows the gap survives (FSA 5.9 vs LoRA 35) — so it's a weight-norm effect,
not a nonlinearity trick; the `O(r²)` term is the `ΔD·ΔU` cross-term, present for any activation (relu² only
amplifies it).

**Still honest caveats:** synthetic random-fact task + a random readout head (both favor FSA); ISOLATED
expert; "interference" is weight-level output drift, not measured forgetting; the free tier is small (~500
facts) and is **not** 0-storage (a reversible trit diff is ~46 KB); end-to-end impact after routing + RMSNorm
is **untested**. The controlled ~8–20× is a real, mechanism-understood *weight-level* benefit; whether it
survives to tokens is the next (full-model) experiment.

## Mathematical foundations

### 1. Exact revoke — a theorem (confirmed 100%)
Let support `S` (active) and reserve `R` be disjoint, `S∩R=∅`, with all shared state (scales, carry, biases)
frozen. The masked base `f(x; reserve off)` is a function of `W[S]` and shared state alone — none of which
training `W[R]` changes. Hence `f` after adaptation **=** `f` before, bit-for-bit. Observed: `revoke_exact =
True` in all 48+ runs. (Caveat: this preserves the *masked* base; the *enabled* function changes — see §3.)

### 2. Capacity law — linear in params, with a precision-set rate
Memorizing `N` random facts of `log₂V` bits each needs `N·log₂V ≤ ρ·m`, where `m` = trainable reserve
params and `ρ` = **realized knowledge-bits per param** (an associative-memory / knowledge-capacity
constant; cf. Allen-Zhu & Li 2024's ~2 bits/param). Fitting the real-expert data:

- **fp reserve:  ρ ≈ 1.1–1.4 bits/param** (free 320 kbit/232 kp = 1.38; full 1.28 Mbit/1.14 Mp = 1.12) —
  right at the known memorization ceiling, scaled down because the base is frozen and the reserve is a subset.
- **ternary reserve:  ρ ≈ 0.03–0.07 bits/param** — only ~**3–4% of its raw log₂3 = 1.58 bits/param**.

So a **ternary reserve holds ~20–40× fewer facts per slot than fp.** The deficit is the quantitative price
of staying low-bit: a reserve trit is pinned to the base row's single frozen scale `α` and 3 levels, which
cannot supply the precise magnitudes arbitrary memorization needs. (This is the reviewers' Gate-0 concern,
now a measured number, and the reason the "free" tier is small.) Capacity is ~linear in `m`, so the full
reserve (≈5× the slots) holds ≈4–8× more.

### 3. Interference law — quadratic in the reserve/base norm ratio
Define `r = ‖Δ‖_F / ‖W‖_F`, the reserve perturbation relative to the base weight (per layer; we report the
max of up/down). For the expert `f(x) = D·φ(U x)` with `φ = relu²`, a reserve `Δ=(ΔU,ΔD)` gives
`Δf ≈ ΔD·φ(Ux) + D·[φ'(Ux)⊙(ΔU x)] + D·[φ''·(ΔU x)²/2]`. On isotropic `x`, `‖ΔUx‖/‖Ux‖ ≈ r`, and because
`φ=relu²` is **quadratic**, the last term contributes an `r²` relative change. Fitting the real data:

> **I ≈ a·r + b·r²,  with a ≈ 2.5, b ≈ 4**  (the `b·r²` term from relu² dominates for `r > 1`).

Fit check (full-fp, widest range): `I/r²` ≈ 4.0–4.6 across `r = 1.0→4.8`; `I = 3.3, 10.3, 21.5, 45.8` at
`r = 0.66, 1.60, 2.33, 3.27`. The law explains everything:
- **Ternary bounds `r`** (the trit magnitude is pinned to `α`): `r` saturates at ~0.65 (free) / ~1.39 (full),
  so **interference saturates** (~1.8 / ~6.7) no matter how hard you train.
- **fp lets `r` grow** with how much you cram in, so **interference blows up quadratically** (up to 100+).

### 4. Structured inputs do NOT dilute interference (a negative result)
We measured interference on a random isotropic probe *and* on a random low-rank (k=32) input subspace:
**`I_lowrank ≈ I_random` to 3 digits.** A reserve trained on isotropic facts is itself isotropic, so it
perturbs every subspace equally — input low-dimensionality gives no free pass. **The real-model dilution
must therefore come only from routing and the residual stream, not input geometry:** the end-to-end relative
logit change is bounded by `I · p_route · (‖expert_out‖ / ‖residual‖)`, where this expert is 1 of 12-of-128
active and its output is one summand on a much larger residual. In an 8B with `d_model=1152`, that factor is
small — but it is **untested**, and it is the next experiment (adapt a reserve and measure perplexity
*through the full model* on real text).

## What this means
- **Free ternary reserve** (0 storage, `r` bounded → bounded interference ~1.8): a **~500-fact free
  personalization tier** baked into the deployed model — modest but genuinely free, and exactly revocable.
- **More capacity** (full / fp) buys facts at the cost of **storage *and* quadratically-rising interference**,
  so you *must* use the mask (reserve off for base) — which, by §1, is exact.
- The headline the data supports: **FSA on the real model learns (ρ·m bits), revokes exactly, and interferes
  by `~r²`.** A lawful, measured tradeoff — not a free lunch, and not vaporware.

## Honest limits
Synthetic fact task + a random readout head, on **isolated** experts. Not real knowledge, not end-to-end.
The capacity/revoke/interference *laws* are on real weights; their *end-to-end* consequence (does adapting a
few experts help a real task without hurting perplexity) is the open question these laws now let us predict
and test.
