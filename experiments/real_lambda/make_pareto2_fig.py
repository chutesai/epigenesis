"""Controlled interference-vs-knowledge Pareto (both methods drift-penalized) on a real lambda expert."""
import json, glob, os, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
STYLE = {"fsa_full_tern": ("FSA full (ternary)", "#1e3a8a", "s"),
         "fsa_free_tern": ("FSA free (0-ish storage)", "#2563eb", "o"),
         "fsa_full_fp": ("FSA full (fp, high-rank control)", "#0891b2", "v"),
         "lora_r16a16": ("LoRA r16 (α/r)", "#ea580c", "^"),
         "lora_r64a16": ("LoRA r64 (α/r)", "#991b1b", "D")}

by = defaultdict(list)
for f in glob.glob(os.path.join(R, "pz_*.json")):
    for r in json.load(open(f)):
        by[r["method"]].append(r)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for m, (lbl, c, mk) in STYLE.items():
    pts = sorted([r for r in by.get(m, []) if r["accB"] >= 0.5], key=lambda r: r["KB"])
    if not pts: continue
    ax.plot([p["KB"] / 1000 for p in pts], [p["interference"] for p in pts],
            "-", color=c, marker=mk, ms=5, lw=1.7, label=lbl)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("knowledge acquired (kbits)  — drift penalty λ swept along each curve")
ax.set_ylabel("interference (min achievable at that KB)")
ax.set_title("Controlled Pareto: FSA vs LoRA on a real lambda expert\n(both drift-penalized; lower-right = better)",
             fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig_controlled_pareto.png"), dpi=160)
print("fig_controlled_pareto written")
