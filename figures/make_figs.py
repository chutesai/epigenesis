"""Generate the paper figures from the experiment JSONs. Usage: python make_figs.py <results_dir>"""
import json, sys, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = sys.argv[1] if len(sys.argv) > 1 else "../experiments/saturate_then_adapt"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
C = {"fsa": "#2563eb", "packnet": "#2563eb", "fullft": "#dc2626", "joint": "#6b7280",
     "lora_r4": "#f59e0b", "lora_r16": "#d97706", "lora_r64": "#b45309", "lora_r256": "#92400e"}

def load(name):
    for p in (f"{RD}/{name}.json", f"{RD}/sat_{name}.json", f"{RD}/cmp_{name}.json"):
        if os.path.exists(p):
            return sorted(json.load(open(p)), key=lambda r: r["n_b"])
    return []

# ---- Fig 1: the headline — A-retention as the reserve fills, at a SATURATED base ----
fsa, ft = load("packnet") or load("fsa"), load("fullft")
if fsa and ft:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    xb = [r["n_b"] / 1000 for r in fsa]
    ax.plot(xb, [r["retention_mask"] for r in fsa], "o-", color=C["fsa"], lw=2.2, ms=7, label="Free-Slot Adaptation (ours)")
    ax.plot([r["n_b"] / 1000 for r in ft], [max(r["retention_mask"], -0.3) for r in ft], "s--", color=C["fullft"], lw=2.2, ms=7, label="full fine-tune")
    ax.axhline(1.0, color="#9ca3af", lw=0.8, ls=":"); ax.text(xb[-1], 1.01, "perfect", color="#6b7280", fontsize=8, ha="right")
    ax.axhline(0.0, color="#9ca3af", lw=0.8, ls=":"); ax.text(xb[-1], 0.02, "random (base destroyed)", color="#6b7280", fontsize=8, ha="right")
    ax.set_xlabel("new knowledge poured into the reserve  (thousands of facts)")
    ax.set_ylabel("base knowledge retained  (fraction)")
    ax.set_title("Base is SATURATED, then we adapt", fontsize=11, loc="left")
    ax.set_ylim(-0.35, 1.12); ax.legend(frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig("fig1_retention.pdf"); fig.savefig("fig1_retention.png", dpi=160)
    print("fig1_retention written")

# ---- Fig 2: reserve acquisition (how much the free slots hold) ----
if fsa:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([r["n_b"] / 1000 for r in fsa], [r["KB_after"] / 1000 for r in fsa], "o-", color=C["fsa"], lw=2.2, ms=7)
    ax.set_xlabel("new facts presented to the reserve  (thousands)")
    ax.set_ylabel("new knowledge acquired  (kbits)")
    ax.set_title("Reserve capacity (free-slot ceiling)", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig("fig2_reserve.pdf"); fig.savefig("fig2_reserve.png", dpi=160)
    print("fig2_reserve written")

# ---- Fig 3: FSA vs LoRA — acquisition vs extra storage (built when cmp_* exists) ----
loras = [n for n in ("lora_r4", "lora_r16", "lora_r64", "lora_r256") if load(n)]
if loras and fsa:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    # pick the reserve point nearest the ceiling for each method
    def peak(rows): return max(rows, key=lambda r: r["KB_after"])
    fp = peak(fsa)
    ax.scatter([0], [fp["KB_after"] / 1000], color=C["fsa"], s=90, zorder=5, label="Free-Slot (0 extra storage)")
    ax.annotate("FSA", (0, fp["KB_after"] / 1000), textcoords="offset points", xytext=(6, 4), color=C["fsa"])
    for n in loras:
        rows = load(n); p = peak(rows)
        xs = p.get("extra_storage_bits", 0) / 8 / 1e6  # MB
        ax.scatter([xs], [p["KB_after"] / 1000], color=C.get(n, "#b45309"), s=70, zorder=5)
        ax.annotate(n.replace("lora_", "LoRA "), (xs, p["KB_after"] / 1000), textcoords="offset points", xytext=(6, -2), fontsize=8)
    ax.set_xlabel("extra storage added  (MB)")
    ax.set_ylabel("new knowledge acquired  (kbits)")
    ax.set_title("Free-Slot vs LoRA: new knowledge per extra byte", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig("fig3_vs_lora.pdf"); fig.savefig("fig3_vs_lora.png", dpi=160)
    print("fig3_vs_lora written")
else:
    print("fig3 (vs LoRA) skipped — cmp_* results not present yet")
