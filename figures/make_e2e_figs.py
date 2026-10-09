"""Real-model (lambda 8B) figures for the README and paper.

Usage: python make_e2e_figs.py [data_dir]   (default ../experiments/end_to_end/data)
Writes fig_e2e_frontier, fig_e2e_storage, fig_chaos_floor, fig_e2e_trajectory as .pdf and .png.
"""
import json
import math
import os
import sys

import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "../experiments/end_to_end/data"

C_LORA, C_TERN, C_FP4, C_CONT = "#3b6fb6", "#d1495b", "#2a9d8f", "#8d99ae"


def load(name):
    with open(os.path.join(D, name + ".json")) as f:
        return json.load(f)


def point(name):
    d = load(name)
    p = d["patched"]
    return p["test"]["exact_match"] * 100, p["big"]["dnll"], p["big"]["dnll_se"], d


# Comparable protocol only: answer-only loss + 12 augmented formats + 40x512 held-out dNLL (v6, v7).
ARMS = [
    ("v6_lora_L1415_loc100", "LoRA r16 + penalty", C_LORA, "o"),
    ("v6_lora_L1415_noreg", "LoRA r16, no penalty", C_LORA, "o"),
    ("v6_lora_L28dn_loc100", "LoRA r16, L28 down", C_LORA, "o"),
    ("v7_fsa_all_bop_t07", "FSA ternary, Bop τ0.7", C_TERN, "D"),
    ("v7_fsa_all_bop_t09", "FSA ternary, Bop τ0.9", C_TERN, "D"),
    ("v6_fsa_all_wd03", "FSA ternary, learn-then-prune", C_TERN, "s"),
    ("v6_fsa_all_wd01", "FSA ternary, learn-then-prune (wd 0.1)", C_TERN, "s"),
    ("v7_fsa_all_wd03_block", "FSA ternary + block scales", C_TERN, "^"),
    ("v6_fsa_L28_netcost", "FSA ternary, L28", C_TERN, "v"),
    ("v7_fsa_all_fp4_loc10_aug", "FSA fp4 + penalty", C_FP4, "P"),
    ("v6_fsa_L28_bf16", "FSA continuous, L28", C_CONT, "X"),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


OFF = {"v7_fsa_all_bop_t07": (-8, 8), "v7_fsa_all_bop_t09": (8, 6), "v6_lora_L1415_loc100": (-10, 9),
       "v7_fsa_all_fp4_loc10_aug": (8, -12), "v6_fsa_all_wd03": (8, -4), "v6_fsa_all_wd01": (8, 2),
       "v7_fsa_all_wd03_block": (8, -10), "v6_lora_L1415_noreg": (-6, 8)}
HA = {"v7_fsa_all_bop_t07": "right", "v6_lora_L1415_loc100": "left", "v6_lora_L1415_noreg": "right"}


def frontier():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, label, c, m in ARMS:
        y, x, se, _ = point(name)
        x = max(x, 1e-4)
        big = name in ("v6_lora_L1415_loc100", "v7_fsa_all_bop_t07", "v6_fsa_all_wd03", "v7_fsa_all_fp4_loc10_aug")
        ax.errorbar(x, y, xerr=se, fmt=m, color=c, ms=9 if big else 6, mec="white", mew=0.8,
                    elinewidth=0.8, capsize=2, alpha=1 if big else 0.75, zorder=3)
        ax.annotate(label, (x, y), xytext=OFF.get(name, (6, 3)), textcoords="offset points",
                    ha=HA.get(name, "left"), fontsize=7.5, color=c, fontweight="bold" if big else "normal")
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 2.0)
    ax.set_ylim(30, 104)
    ax.set_xlabel("damage to general text: ΔNLL on 20k held-out tokens (nats, log scale)")
    ax.set_ylabel("held-out recall (%)")
    ax.set_title("Recall vs damage on the real 8B model (40 facts, held-out phrasings)", loc="left", fontsize=11)
    ax.text(0.99, 0.02, "upper-left is better", transform=ax.transAxes, ha="right", fontsize=8, color="#555")
    style(ax)
    fig.tight_layout()
    save(fig, "fig_e2e_frontier")


def storage():
    rows = []
    for name, label, c, m in ARMS:
        if name == "v6_lora_L1415_noreg":
            continue
        if name == "v6_lora_L1415_loc100":
            label = "LoRA r16"
        y, x, se, d = point(name)
        if d["lora_params"]:
            mb = d["lora_params"] * 2 / 1e6  # bf16
        else:
            n, S = d["flipped"], d["reserve_slots"]
            vb = 1 if d["fmt"] == "ternary" else (4 if d["fmt"] == "fp4" else 16)
            p = n / S
            h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0
            mb = (S * h + n * vb) / 8 / 1e6  # entropy-coded positions + values
        rows.append((name, label, c, m, mb, y, x))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    soff = {"v7_fsa_all_bop_t09": (6, 6), "v7_fsa_all_bop_t07": (6, -11), "v6_lora_L1415_loc100": (-6, 8),
            "v7_fsa_all_fp4_loc10_aug": (0, -14)}
    sha = {"v6_lora_L1415_loc100": "right", "v7_fsa_all_fp4_loc10_aug": "center"}
    for name, label, c, m, mb, y, x in rows:
        ax.scatter(mb, y, marker=m, color=c, s=70, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(label, (mb, y), xytext=soff.get(name, (6, 2)), textcoords="offset points",
                    ha=sha.get(name, "left"), fontsize=7.5, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("patch size (MB; FSA = entropy-coded slot positions + values, LoRA = bf16)")
    ax.set_ylabel("held-out recall (%)")
    ax.set_title("What each patch costs to store", loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    save(fig, "fig_e2e_storage")


def chaos_floor():
    rows = load("floor_diag")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = {2: "#6a4c93", 15: "#1982c4", 28: "#8ac926"}
    for layer in (2, 15, 28):
        pts = sorted((r["k"], r["kl"]) for r in rows if r["layer"] == layer and r["projs"] == "down_proj")
        a.plot([k for k, _ in pts], [v for _, v in pts], "-o", color=cols[layer], ms=4, label=f"MoE layer {layer}")
    a.set_xscale("log")
    a.set_ylim(0, 0.032)
    a.set_xlabel("random ternary flips in free slots (down_proj)")
    a.set_ylabel("KL to the unpatched model (nats/token)")
    a.set_title("One flip already costs most of the KL", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    style(a)
    eps, kl = [], []
    with open(os.path.join(D, "eps_diag.txt")) as f:
        for line in f:
            if line.startswith("EPS"):
                _, e, j = line.split(" ", 2)
                eps.append(float(e))
                kl.append(json.loads(j)["kl"])
    b.plot(eps, kl, "-o", color="#1982c4", ms=4)
    b.set_xscale("log")
    b.set_xlabel("single-weight change (× row scale α)")
    b.set_ylabel("KL to the unpatched model")
    b.set_title("Below 0.01α nothing changes", loc="left", fontsize=10)
    style(b)
    fig.tight_layout()
    save(fig, "fig_chaos_floor")


def trajectory():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for name, label, c, ls in (("v6_fsa_all_wd03", "learn-then-prune", C_TERN, "-"),
                               ("v7_fsa_all_bop_t07", "Bop τ0.7", "#b5179e", "-"),
                               ("v6_lora_L1415_loc100", "LoRA + penalty", C_LORA, "--")):
        t = load(name)["trajectory"]
        s = [r["step"] for r in t]
        axes[0].plot(s, [r["val_em"] * 100 for r in t], ls, color=c, marker="o", ms=3, label=label)
        axes[1].plot(s, [r["val_dnll"] for r in t], ls, color=c, marker="o", ms=3)
        if r_nnz := [r["nnz"] for r in t if r["nnz"]]:
            axes[2].plot(s[-len(r_nnz):], [n / 1e6 for n in r_nnz], ls, color=c, marker="o", ms=3)
    axes[0].set_ylabel("validation recall (%)")
    axes[1].set_ylabel("validation ΔNLL (symlog)")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].axhline(0, color="#999", lw=0.6)
    axes[2].set_ylabel("filled slots (millions)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("training step")
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Learn first, then shed", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_e2e_trajectory")


def save(fig, stem):
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=170)
    plt.close(fig)
    print("wrote", stem)


if __name__ == "__main__":
    frontier()
    storage()
    chaos_floor()
    trajectory()
