"""Honest mechanism-paper figures from the capacity sweep. Usage: python make_figs2.py <capacity_dir>
NOTE: all of this is the DENSE fp32 mechanism assay. FSA storage/efficiency assumes the fp32
reserve ternarizes losslessly (Gate 0, untested) and a STRUCTURAL (~0-bit) mask. Captions say so."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = sys.argv[1] if len(sys.argv) > 1 else "../experiments/capacity"
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
BPW = 1.088                     # pair8 info-bits/weight (what a trit would cost)
LAYER_IO = (256 + 512) + (512 + 512) + (512 + 1024)   # sum(d_in+d_out) over the 3 MLP layers = 3328

def load(name):
    p = f"{RD}/{name}.json"
    return sorted(json.load(open(p)), key=lambda r: r["n_b"]) if os.path.exists(p) else []

def xk(rows): return [r["n_b"] / 1000 for r in rows]
def yk(rows): return [r["KB_after"] / 1000 for r in rows]

fsa = load("fs_packnet")
LORA = {"r=16": ("ft_r16_2e-4", "#fca5a5"), "r=64": ("ft_r64_1e-4", "#f59e0b"),
        "r=256": ("ft_r256_1e-4", "#b45309"), "r=1024": ("ft_r1024_1e-4", "#7c2d12")}
BLUE = "#2563eb"

# ---- fig2: FSA reserve capacity curve (the ceiling + overflow) ----
if fsa:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(xk(fsa), yk(fsa), "o-", color=BLUE, lw=2.3, ms=7)
    pk = max(fsa, key=lambda r: r["KB_after"])
    ax.annotate(f"observed max ≈ {pk['KB_after']/1000:.0f} kbits", (pk["n_b"]/1000, pk["KB_after"]/1000),
                textcoords="offset points", xytext=(-10, 10), color=BLUE, fontsize=9)
    ax.set_xlabel("new facts poured into the reserve  (thousands)")
    ax.set_ylabel("new knowledge acquired  (kbits)")
    ax.set_title("FSA reserve capacity — fills, then overflows", fontsize=11, loc="left")
    ax.set_xscale("log")
    fig.tight_layout(); fig.savefig("fig2_reserve.png", dpi=160); fig.savefig("fig2_reserve.pdf")
    print("fig2 written")

# ---- fig3: FSA vs tuned LoRA — (a) capacity, (b) knowledge per stored bit ----
if fsa:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 3.7))
    axL.plot(xk(fsa), yk(fsa), "o-", color=BLUE, lw=2.4, ms=6, label="FSA (reserve, ρ=0.5)", zorder=5)
    for lbl, (fn, c) in LORA.items():
        r = load(fn)
        if r: axL.plot(xk(r), yk(r), "s--", color=c, lw=1.8, ms=4, label=f"LoRA {lbl}")
    axL.set_xscale("log"); axL.set_xlabel("new facts  (thousands)"); axL.set_ylabel("knowledge acquired  (kbits)")
    axL.set_title("Capacity — LoRA scales with rank;\nFSA reaches an observed maximum", fontsize=10.5, loc="left")
    axL.legend(frameon=False, fontsize=8.5, loc="upper left")

    # Honest: count BOTH methods in the representation ACTUALLY trained (fp32). The ranking reverses.
    def eff_fsa(r): return r["KB_after"] / max(r.get("reserve_filled", 1) * 32, 1)      # fp32 reserve weights
    def eff_lora(r, rank): return r["KB_after"] / (rank * LAYER_IO * 32)                 # fp32 adapter params
    axR.plot(xk(fsa), [eff_fsa(r) for r in fsa], "o-", color=BLUE, lw=2.4, ms=6,
             label="FSA (fp32 reserve, as trained)", zorder=5)
    for lbl, (fn, c) in LORA.items():
        r = load(fn); rank = int(lbl.split("=")[1])
        if r: axR.plot(xk(r), [eff_lora(x, rank) for x in r], "s--", color=c, lw=1.8, ms=4, label=f"LoRA {lbl}")
    axR.set_xscale("log"); axR.set_xlabel("new facts  (thousands)")
    axR.set_ylabel("knowledge per fp32 bit trained")
    axR.set_title("Storage efficiency, fp32-vs-fp32 — LoRA is ahead", fontsize=10.5, loc="left")
    axR.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.5, -0.02, "Counted in the representation actually trained (fp32), LoRA is more efficient. "
             "A ternary-reserve win is a PREDICTION contingent on the untested Gate 0 — genuinely open "
             "(the format assay measures from-scratch QAT, not a frozen-base reserve).",
             ha="center", fontsize=8, color="#6b7280")
    fig.tight_layout(); fig.savefig("fig3_vs_lora.png", dpi=160, bbox_inches="tight"); fig.savefig("fig3_vs_lora.pdf", bbox_inches="tight")
    print("fig3 written")

# ---- fig4: middle-ground `soft` — graceful degradation of the always-on model ----
soft = load("mg_soft")
if soft:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xf = [r["n_b"] / 1000 for r in soft]; ret = [r.get("retention_all", 0) for r in soft]
    ax.plot(xf, ret, "o-", color="#16a34a", lw=2.3, ms=7)
    ax.set_xscale("log")
    ax.axhline(1.0, color="#9ca3af", lw=0.8, ls=":"); ax.axhline(0.0, color="#9ca3af", lw=0.8, ls=":")
    ax.set_xlabel("new facts added to the reserve  (thousands)")
    ax.set_ylabel("base retained, UNMASKED  (fraction)")
    ax.set_title("Always-on (soft): base retention falls with load (no mask/task-ID)", fontsize=9.5, loc="left")
    ax.set_ylim(-0.2, 1.1)
    fig.tight_layout(); fig.savefig("fig4_soft.png", dpi=160); fig.savefig("fig4_soft.pdf")
    print("fig4 written")
