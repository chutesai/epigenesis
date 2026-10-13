"""Progress figures for the top of the README (forced-zero mask results, 2026-10-13).

Every number here is copied from experiments/epigenesis/LEADERBOARD.md (test = held-out phrasings; val =
validation only). Seeds and item counts are printed on each figure. Nothing is fitted or interpolated.

Usage: python make_story_figs.py          (writes fig_story_*.png/.pdf next to this file)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# categorical palette (validated for colour-vision deficiency in adjacent order): FSA, LoRA, dense fp, dense ternary
C_FSA, C_LORA, C_DFP, C_DTERN = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 10, "legend.frameon": False, "figure.facecolor": "white",
})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def footnote(fig, text, y=0.012):
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=8.5, color=INK3, wrap=True)


def save(fig, stem):
    fig.savefig(os.path.join(HERE, stem + ".pdf"))
    fig.savefig(os.path.join(HERE, stem + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------------------------------------
# Figure 1: held-out recall vs values written, three fact tasks (test GEN)
# ---------------------------------------------------------------------------------------------
# (label, [(x = values written, y = test GEN %, seed tag)])  x: FSA = filled ternary slots, LoRA = adapter parameters
TASKS = [
    ("40 facts, one person\n(80 test items)",
     {"fsa": [(175e3, 100.0), (188e3, 100.0)],                               # F40_bopck_s0/s1
      "lora16": [(22.0e6, 93.75), (22.0e6, 92.5), (22.0e6, 90.0)],          # F40_lora16_s0/s1/s2
      "lora_small": [(1.4e6, 93.75), (5.5e6, 96.25)]},                       # F40_lora1_s1, F40_lora4_s1
     "FSA 2 seeds · LoRA r16 3 seeds · LoRA r1/r4 1 seed each"),
    ("2 people × 50 conflicting facts\n(200 test items)",
     {"fsa": [(745e3, 97.0), (745e3, 99.5), (745e3, 99.5)],                  # C_bopck_sub_s0/s1/s2, 733k–757k slots
      "lora16": [(22.0e6, 94.5), (22.0e6, 95.5), (22.0e6, 95.0)]},          # s1_lora16, D_lora16_sub_s1/s2
     "3 seeds each · FSA slots 733k–757k (drawn at the midpoint)"),
    ("3 people × 100 conflicting facts\n(600 test items)",
     {"fsa": [(0.81e6, 91.3), (1.60e6, 99.2), (2.13e6, 99.8)],               # GL_CTRL_s1 (1M), GL_B4M_s1 (4M), GL_B16M_s1 (16M)
      "lora16": [(44.0e6, 96.2)]},                                           # R4_lora16_rand4_s1
     "seed 1 only · MoE layers 3,16,22,29 · final step · budgets 1M / 4M / 16M"),
]


def fig_recall_vs_size():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
    for ax, (title, series, sub) in zip(axes, TASKS):
        style(ax)
        ax.set_xscale("log")
        ax.set_xlim(8e4, 1.5e8)
        ax.set_ylim(86, 101.5)
        ax.set_yticks([90, 95, 100])
        ax.axhline(100, color=GRID, lw=1, zorder=0)
        for key, c, m, s in (("lora_small", C_LORA, "o", 70), ("lora16", C_LORA, "o", 110), ("fsa", C_FSA, "D", 110)):
            if key not in series:
                continue
            xs = [p[0] for p in series[key]]
            ys = [p[1] for p in series[key]]
            ax.scatter(xs, ys, s=s, marker=m, color=c, edgecolor="white", linewidth=1.5, zorder=5)
        ax.set_title(title, loc="left", fontsize=11.5)
        ax.text(0.01, 0.03, sub, transform=ax.transAxes, fontsize=8.5, color=INK3, ha="left", va="bottom")
        ax.set_xlabel("values written (log scale)")
    axes[0].set_ylabel("held-out recall, free generation (% exact)")
    # direct labels, sparing
    axes[0].annotate("FSA 100% / 100%", (182e3, 100), xytext=(0, -16), textcoords="offset points", ha="center",
                     fontsize=9, color=INK2)
    axes[0].annotate("LoRA r16\n90–93.75%", (22e6, 92), xytext=(-8, -30), textcoords="offset points", ha="right",
                     fontsize=9, color=INK2)
    axes[0].annotate("r1", (1.4e6, 93.75), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK3)
    axes[0].annotate("r4", (5.5e6, 96.25), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK3)
    axes[1].annotate("FSA 97–99.5%", (745e3, 99.5), xytext=(0, 9), textcoords="offset points", ha="center",
                     fontsize=9, color=INK2)
    axes[1].annotate("LoRA r16 94.5–95.5%", (22e6, 95.0), xytext=(0, -20), textcoords="offset points", ha="center",
                     fontsize=9, color=INK2)
    axes[2].annotate("FSA, 4M budget\n99.2% · 5.8 MB", (1.60e6, 99.2), xytext=(-12, -6), textcoords="offset points",
                     ha="right", va="top", fontsize=9, color=INK2)
    axes[2].annotate("FSA, 16M budget\n99.8% · 7.7 MB", (2.13e6, 99.8), xytext=(12, 2), textcoords="offset points",
                     ha="left", va="top", fontsize=9, color=INK2)
    axes[2].annotate("FSA, 1M budget\n91.3% · 2.95 MB", (0.81e6, 91.3), xytext=(0, 10), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color=INK2)
    axes[2].annotate("LoRA r16\n96.2% · 88 MB bf16", (44e6, 96.2), xytext=(0, -14), textcoords="offset points",
                     ha="center", va="top", fontsize=9, color=INK2)
    handles = [Line2D([], [], marker="D", color=C_FSA, ls="", ms=9, label="FSA: ternary writes into the forced zeros (frozen base)"),
               Line2D([], [], marker="o", color=C_LORA, ls="", ms=9, label="LoRA adapter on the same expert matrices")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.935), ncol=2)
    fig.suptitle("Same facts, fewer values written: the best tested FSA recipe matches or beats LoRA r16 on three fact tasks", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", y=0.985)
    footnote(fig, "Test = held-out phrasings, greedy generation, whole answer must match. x: FSA = filled ternary slots, LoRA = adapter "
                  "parameters (bf16 bytes = 2 × params; int4 factors would be 4× smaller). Lambda 8B ternary MoE, 725B export. "
                  "3p×100 (MoE layers 3,16,22,29): one seed each; every final patch fails the +0.003 code-damage gate (FSA +0.004 / +0.011 / +0.007 at 1M / 4M / 16M, "
                  "LoRA +0.008). Recipes are tuned per method (rehearsal, penalty, steps differ), so this compares tuned recipes rather than isolating the method.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    save(fig, "fig_story_recall_vs_size")


# ---------------------------------------------------------------------------------------------
# Figure 2: generalisation to question forms never trained on (40-fact patches, layers 14,15)
# ---------------------------------------------------------------------------------------------
PROBES = [  # (label, FSA seeds, LoRA seeds, n items)
    ("unseen question\nphrasing", [90.0, 88.3], [65.0, 61.7], 120),
    ("indirect question\n(\"the person whose X is Y,\nwhat is her Z?\")", [50.8, 44.2], [22.5, 30.0], 120),
    ("pronoun after a mention\n(\"…Tamsin… what is her Z?\")", [75.0], [47.5], 40),
]


def fig_generalisation():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    style(ax)
    w, gap = 0.34, 0.04
    for i, (label, fsa, lora, n) in enumerate(PROBES):
        for j, (vals, c, name) in enumerate(((fsa, C_FSA, "FSA"), (lora, C_LORA, "LoRA r16"))):
            x = i + (j - 0.5) * (w + gap)
            mean = sum(vals) / len(vals)
            ax.bar(x, mean, width=w, color=c, zorder=3, label=name if i == 0 else None)
            if len(vals) > 1:
                ax.scatter([x] * len(vals), vals, s=38, color="white", edgecolor=c, linewidth=1.5, zorder=4)
            txt = " / ".join(f"{v:g}" for v in vals) + "%"
            ax.text(x, max(vals) + 2.2, txt, ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(PROBES)))
    ax.set_xticklabels([f"{p[0]}\nn = {p[3]} per seed" + ("" if len(p[1]) > 1 else " · one seed") for p in PROBES],
                       fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("correct answers, free generation (%)")
    ax.set_title("Facts written into the zeros answer question forms they were never trained on", loc="left")
    ax.legend(loc="upper right")
    footnote(fig, "40-fact patches on MoE layers 14,15 (the FSA patches are the ones that score 100% / 100% on the standard test). Bars = mean "
                  "over seeds, rings = individual seeds (2 per method; pronoun probe 1 per method). Held-out free generation; "
                  "the three phrasings per fact are not independent items. Both methods are near 0% on questions that need two "
                  "facts in one answer under the strict scorer. Tuned recipes per method (rehearsal and penalty strengths differ).")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_story_generalisation")


# ---------------------------------------------------------------------------------------------
# Figure 3: closing the gap on 3 people × 100 conflicting facts (test GEN, seed 1 unless noted)
# ---------------------------------------------------------------------------------------------
STEPS = [  # (x label, FSA test GEN or None, LoRA test GEN or None, lever text)
    ("2 adjacent MoE layers\n(14, 15)", 18.2, 65.7, None),
    ("4 spread MoE layers\n(3, 16, 22, 29),\nslots frozen at step 200", 93.2, 96.2, "spread placement +\nfrozen slot choice"),
    ("slot budget\n1M → 4M", 99.2, None, "budget"),
    ("slot budget\n4M → 16M", 99.8, None, None),
]


def fig_gap_ladder():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    style(ax)
    xs = list(range(len(STEPS)))
    fsa = [s[1] for s in STEPS]
    lora = [s[2] for s in STEPS]
    ax.plot(xs, fsa, color=C_FSA, lw=2.5, marker="D", ms=9, mec="white", mew=1.5, zorder=5, label="FSA (in-pair write set at step 1, forced zeros after)")
    lx = [x for x, v in zip(xs, lora) if v is not None]
    ly = [v for v in lora if v is not None]
    ax.plot(lx, ly, color=C_LORA, lw=2.5, marker="o", ms=9, mec="white", mew=1.5, zorder=5, label="LoRA r16, same layers")
    for x, v in zip(xs, fsa):
        off = {0: (14, -8), 1: (12, -4), 2: (0, -16), 3: (0, 11)}[x]
        txt = f"FSA {v:g}%" + ("\n(in-pair write set)" if x == 0 else "")
        ax.annotate(txt, (x, v), xytext=off, textcoords="offset points",
                    ha={0: "left", 1: "left", 2: "center", 3: "center"}[x], va={0: "top", 1: "top", 2: "top", 3: "bottom"}[x],
                    fontsize=10, color=INK2, fontweight="bold")
    for x, v in zip(lx, ly):
        off = (-12, 4) if x == 1 else (0, 11)
        ax.annotate(f"LoRA {v:g}%", (x, v), xytext=off, textcoords="offset points", ha="right" if x == 1 else "center",
                    va="bottom", fontsize=10, color=INK2)
    for i, s in enumerate(STEPS):
        if s[3]:
            ax.text(i - 0.5, 30, s[3], ha="center", va="center", fontsize=9.5, color=INK2, style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in STEPS], fontsize=9.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("held-out recall, free generation (% exact)")
    ax.set_title("3 people × 100 conflicting facts: successive recipes took FSA from 18% to 99.8% test recall", loc="left")
    ax.legend(loc="lower right")
    footnote(fig, "One seed per point (seed 1 for the 4-layer runs). Step 1 (2 layers): FSA used the earlier in-pair write set, LoRA "
                  "r16 + penalty, both single-seed test runs; the steps do not isolate write set, placement, freezing and training length from one another. "
                  "The 1M-budget recipe rerun on the current code scored 91.3% (vs 93.2%). "
                  "Patches: 5.8 MB at 4M (code damage +0.011), 7.7 MB at 16M (2.13M slots written, code +0.007); both fail the +0.003 gate, "
                  "as LoRA's +0.008 does. Every point is final-step test scoring.")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_story_gap_ladder")


# ---------------------------------------------------------------------------------------------
# Figure 4: damage to general text, three held-out panels (ΔNLL, nats/token)
# ---------------------------------------------------------------------------------------------
DAMAGE = [  # (label, colour, [WikiText, FineWeb-Edu, Python code])
    ("FSA, ternary in the forced zeros\n(40 facts, seed 0 / seed 1)", C_FSA,
     [[-0.0006, -0.0007, +0.0003], [+0.0019, -0.0007, -0.0001]]),
    ("LoRA r16, same layers\n(40 facts, seed 0)", C_LORA, [[+0.0014, +0.0002, +0.0009]]),
    ("dense fp low-rank, all slots\n(1 person × 100 facts, seed 0)", C_DFP, [[+0.028, +0.014, +0.013]]),
    ("dense ternary low-rank\n(1 person × 100 facts, seed 0)", C_DTERN, [[+2.04, +1.69, +0.76]]),
]
PANELS = ["WikiText", "FineWeb-Edu", "Python code"]


def fig_damage():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    style(ax)
    n = len(DAMAGE)
    w, gap = 0.19, 0.02
    for j, (label, c, runs) in enumerate(DAMAGE):
        mean = [sum(r[k] for r in runs) / len(runs) for k in range(3)]
        xs = [i + (j - (n - 1) / 2) * (w + gap) for i in range(3)]
        ax.bar(xs, mean, width=w, color=c, zorder=3, label=label)
        if len(runs) > 1:
            for r in runs:
                ax.scatter(xs, r, s=30, color="white", edgecolor=c, linewidth=1.3, zorder=4)
        for x, v in zip(xs, mean):
            if abs(v) >= 0.003:  # bars inside the clean band are left unlabelled; the README table has their values
                ax.text(x, v, f"{v:+.3g}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_yscale("symlog", linthresh=0.003, linscale=1.2)
    ax.set_ylim(-0.003, 4)
    ax.axhline(0, color=INK3, lw=1, zorder=2)
    ax.axhline(0.003, color=INK3, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(-0.47, 0.0033, "+0.003 'clean' gate", ha="left", va="bottom", fontsize=8.5, color=INK3)
    ax.set_yticks([0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0", "+0.003", "+0.01", "+0.03", "+0.1", "+0.3", "+1", "+3"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(PANELS)
    ax.set_ylabel("damage to general text: ΔNLL vs base (nats/token, symlog)")
    ax.set_title("Small forced-zero patches leave general text within noise;\ndense ternary low-rank writes cost +0.8 to +2.0 nats per token",
                 loc="left")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.115), ncol=4, fontsize=9, handlelength=1.2, columnspacing=1.4)
    footnote(fig, "ΔNLL on held-out windows (40×512 WikiText; FineWeb-Edu and Python code panels), paired SE ≈ 0.002, so FSA and LoRA "
                  "on 40 facts (MoE layers 14,15, Bop / LoRA recipes) are within noise of zero. The dense cells are the interim 2×2 factorial "
                  "(1 person × 100 facts, matched Adam recipe, val-selected LR, one seed; sparse cells still running): dense ternary reaches "
                  "100% test recall at +2.0 nats; dense fp was LR-limited at 200 steps (17.5% recall). Larger forced-zero patches are not free: "
                  "the 3p×100 patches are +0.011 (4M) and +0.007 (16M) on code, above the gate.")
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "fig_story_damage")


if __name__ == "__main__":
    fig_recall_vs_size()
    fig_generalisation()
    fig_gap_ladder()
    fig_damage()
