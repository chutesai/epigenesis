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
    ("v6_lora_L28dn_loc100", "LoRA r16, layer 28 down", C_LORA, "o"),
    ("v8_fsa_all_bop_t07_wd03", "FSA ternary, Bop τ0.7 (repeat run)", C_TERN, "D"),
    ("v7_fsa_all_bop_t07", "FSA ternary, Bop τ0.7", C_TERN, "D"),
    ("v9_bop_t07_loc10", "FSA ternary, Bop + penalty", C_TERN, "*"),
    ("v8_fsa_all_wd05", "FSA ternary, learn-then-prune (decay 0.5)", C_TERN, "s"),
    ("v7_fsa_all_bop_t09", "FSA ternary, Bop τ0.9", C_TERN, "D"),
    ("v6_fsa_all_wd03", "FSA ternary, learn-then-prune (decay 0.3)", C_TERN, "s"),
    ("v6_fsa_all_wd01", "FSA ternary, learn-then-prune (decay 0.1)", C_TERN, "s"),
    ("v7_fsa_all_wd03_block", "FSA ternary + block scales", C_TERN, "^"),
    ("v6_fsa_L28_netcost", "FSA ternary, layer 28", C_TERN, "v"),
    ("v7_fsa_all_fp4_loc10_aug", "FSA fp4 + penalty", C_FP4, "P"),
    ("v6_fsa_L28_bf16", "FSA continuous, layer 28", C_CONT, "X"),
]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


# Legend-driven styling: one entry per method family; minor ablations are drawn muted and grouped.
SERIES = [
    # (label, [run names], color, marker, size, zorder)
    ("FSA ternary, Bop + local penalty", ["v9_bop_t07_loc10"], "#c1121f", "*", 340, 6),
    ("FSA ternary, Bop (τ 0.7, two runs)", ["v7_fsa_all_bop_t07", "v8_fsa_all_bop_t07_wd03"], "#e5383b", "D", 95, 5),
    ("FSA ternary, Bop (τ 0.9)", ["v7_fsa_all_bop_t09"], "#f4a3a8", "D", 80, 4),
    ("FSA ternary, learn-then-prune (decay 0.3)", ["v6_fsa_all_wd03"], "#f77f00", "s", 95, 4),
    ("FSA ternary, learn-then-prune (decay 0.5)", ["v8_fsa_all_wd05"], "#fcbf49", "s", 95, 4),
    ("FSA fp4 slots + local penalty", ["v7_fsa_all_fp4_loc10_aug"], "#2a9d8f", "P", 150, 4),
    ("LoRA r16 + local penalty", ["v6_lora_L1415_loc100"], "#1d4ed8", "o", 150, 5),
    ("LoRA r16, no penalty", ["v6_lora_L1415_noreg"], "#93c5fd", "o", 95, 3),
    ("other ablations (block scales, layer 28, continuous, decay 0.1)",
     ["v7_fsa_all_wd03_block", "v6_fsa_L28_netcost", "v6_fsa_L28_bf16", "v6_lora_L28dn_loc100", "v6_fsa_all_wd01"],
     "#b8bcc6", "o", 40, 2),
]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5, "legend.fontsize": 10})


def legend_right(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0,
              handletextpad=0.6, labelspacing=0.9)


def frontier():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for label, names, c, m, sz, z in SERIES:
        xs, ys, es = [], [], []
        for n in names:
            y, x, se, _ = point(n)
            xs.append(max(x, 1e-4)); ys.append(y); es.append(se)
        ax.errorbar(xs, ys, xerr=es, fmt="none", ecolor=c, elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=z - 1)
        ax.scatter(xs, ys, s=sz, marker=m, color=c, edgecolor="white", linewidth=0.9, zorder=z, label=label)
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 2.0)
    ax.set_ylim(30, 104)
    ax.set_xlabel("damage to general text: ΔNLL on 20k held-out tokens (nats, log scale; bars = paired SE)")
    ax.set_ylabel("held-out recall (%)")
    ax.set_title("Recall vs damage on the real 8B model (40 facts, held-out phrasings)", loc="left")
    ax.text(0.985, 0.03, "upper-left is better", transform=ax.transAxes, ha="right", va="bottom",
            color="#777", fontsize=10)
    style(ax)
    legend_right(ax)
    fig.tight_layout()
    save(fig, "fig_e2e_frontier")


def patch_mb(d):
    if d["lora_params"]:
        return d["lora_params"] * 2 / 1e6  # bf16
    n, S = d["flipped"], d["reserve_slots"]
    vb = 1 if d["fmt"] == "ternary" else (4 if d["fmt"] == "fp4" else 16)
    p = n / S
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0
    mb = (S * h + n * vb) / 8 / 1e6  # entropy-coded positions + values
    if d.get("patch_scale") == "block":
        # add only the learned fp32 block scales: patch_bytes = value/index bits + 32 bits per scale
        base_bits = d["flipped"] * (math.log2(max(S, 2)) + vb) if d.get("budget") else S * (math.log2(3) if vb == 1 else vb)
        mb += (d["patch_bytes"] * 8 - base_bits) / 8 / 1e6
    return mb


def storage():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for label, names, c, m, sz, z in SERIES:
        if label.startswith("LoRA r16, no penalty"):
            continue                                     # same size as the penalized LoRA
        pts = [(patch_mb(point(n)[3]), point(n)[0]) for n in names]
        ax.scatter([a for a, _ in pts], [b for _, b in pts], s=sz, marker=m, color=c, edgecolor="white",
                   linewidth=0.9, zorder=z, label=label.replace("LoRA r16 + local penalty", "LoRA r16 (with or without penalty)"))
    ax.set_xscale("log")
    ax.set_xlabel("patch size (MB, log scale; FSA = entropy-coded slot positions + values, LoRA = bf16 factors)")
    ax.set_ylabel("held-out recall (%)")
    ax.set_title("What each patch costs to store", loc="left")
    ax.text(0.985, 0.03, "upper-left is better", transform=ax.transAxes, ha="right", va="bottom",
            color="#777", fontsize=10)
    style(ax)
    legend_right(ax)
    fig.tight_layout()
    save(fig, "fig_e2e_storage")


def chaos_floor():
    rows = load("floor_diag")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = {2: "#6a4c93", 15: "#1982c4", 28: "#8ac926"}
    for layer in (2, 15, 28):
        pts = sorted((r["k"], r["kl"]) for r in rows if r["layer"] == layer and r["projs"] == "down_proj")
        a.plot([k for k, _ in pts], [v for _, v in pts], "-o", color=cols[layer], ms=4, label=f"MoE layer {layer}")
    a.set_xscale("log")
    a.set_ylim(0, 0.032)
    a.set_xlabel("random ternary flips in free down-projection slots")
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
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
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print("wrote", stem)


if __name__ == "__main__":
    frontier()
    storage()
    chaos_floor()
    trajectory()
