"""Figures for the revised RagaSeq manuscript (all text uses 'raga')."""
import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = "../figures"
import os; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "legend.frameon": False})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- Figure 1: workflow ------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.0, 3.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
    steps = [("1  Pitch + tonic", "CompMusic feature files\n(.pitch, .tonicFine)"),
             ("2  Window", "central 150 s,\nfive 30-s windows"),
             ("3  Symbolize", "10 Hz bins, cents above Sa,\n12 states, 3-frame smoothing"),
             ("4  Profile", "3-note patterns weighted\nby held time (RagaSeq-D)"),
             ("5  Compare", "Jensen–Shannon distance;\nSmith–Waterman alignment"),
             ("6  Test", "recognition, recording-level\npermutation, shuffle controls")]
    w, h, gap = 3.0, 1.15, 0.35
    for i, (t, s) in enumerate(steps):
        r, c = divmod(i, 3)
        x = 0.1 + c * (w + gap); y = 3.25 - r * 1.6
        last = i == 5
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#fdf1ea" if last else "#f3f6fb", ec=ORANGE if last else BLUE, lw=1))
        ax.text(x + 0.12, y + h - 0.25, t, ha="left", va="center", fontsize=8.5, weight="bold", color=INK)
        ax.text(x + 0.12, y + 0.4, s, ha="left", va="center", fontsize=7.0, color=INK2, linespacing=1.3)
        if c < 2:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2), (x + w + gap - 0.03, y + h / 2), arrowstyle="-|>",
                                         mutation_scale=9, color=MUTED, lw=1))
    ax.text(0.1, 1.05, "Example symbol string (one window):", fontsize=7.5, color=INK2)
    ax.text(0.1, 0.65, "S R G m P m G R S N S R G …", fontsize=8.5, family="DejaVu Sans Mono", color=INK)
    ax.text(5.3, 1.05, "Borrowed from sequence biology:", fontsize=7.5, color=INK2)
    ax.text(5.3, 0.65, "k-mer profiles, local alignment, shuffle null models", fontsize=7.5, color=INK)
    save(fig, "fig1_workflow")


# ---------- Figure 2: Smith–Waterman illustration -------------------------------------------
def fig_sw():
    A = "S R G m P m G R S".split(); B = "D R G m P m n D N".split()
    fig, ax = plt.subplots(figsize=(6.4, 1.5)); ax.axis("off"); ax.set_xlim(-1.4, 13.5); ax.set_ylim(-0.3, 2.2)
    for row, seq, y in [("A", A, 1.5), ("B", B, 0.5)]:
        ax.text(-1.2, y, row, fontsize=9, weight="bold", va="center")
        for i, s in enumerate(seq):
            match = 1 <= i <= 5
            ax.add_patch(Rectangle((i - 0.4, y - 0.32), 0.8, 0.64, fc="#e8f0fb" if match else "white",
                                   ec=BLUE if match else GRID, lw=1))
            ax.text(i, y, s, ha="center", va="center", fontsize=9, family="DejaVu Sans Mono")
    ax.add_patch(Rectangle((0.52, 0.12), 4.96, 1.76, fill=False, ec=BLUE, lw=1.6, ls="-"))
    ax.text(9.0, 1.0, "best local match: R G m P m\nscore 5 × (+2) = 10\nnormalized 10 / (2 × 9) = 0.56",
            fontsize=7.5, color=INK2, va="center")
    ax.text(-1.2, -0.15, "Scoring: match +2, mismatch −1, gap −2", fontsize=7, color=MUTED)
    save(fig, "fig2_smith_waterman")


# ---------- Figure 3: CDFs (development set) --------------------------------------------------
def fig_cdf(pdf):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, col, lab in [(axs[0], "smith_waterman_norm_mean", "Normalized Smith–Waterman similarity"),
                         (axs[1], "kmer_js_distance_mean", "3-mer Jensen–Shannon distance")]:
        for same, c, name in [(False, MUTED, "Cross-raga pairs (1,125)"), (True, BLUE, "Same-raga pairs (100)")]:
            v = np.sort(pdf.loc[pdf.same_raga == same, col].to_numpy())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2, label=name)
        ax.set_xlabel(lab); ax.set_ylabel("Cumulative proportion"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig3_cdf")


# ---------- Figure 4: per-raga within similarity -------------------------------------------------
def fig_per_raag(pdf):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    ragas = sorted(set(pdf.raga_a))
    for ax, col, lab in [(axs[0], "smith_waterman_norm_mean", "Smith–Waterman similarity"),
                         (axs[1], "kmer_js_distance_mean", "3-mer Jensen–Shannon distance")]:
        cross = pdf.loc[~pdf.same_raga, col].mean()
        for i, r in enumerate(ragas):
            v = pdf.loc[pdf.same_raga & (pdf.raga_a == r), col]
            ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), s=10,
                       color=BLUE, alpha=0.45, lw=0)
            ax.scatter([v.mean()], [i], s=46, color=BLUE, ec="white", lw=1.2, zorder=3)
        ax.axvline(cross, color=ORANGE, ls="--", lw=1.4, label="cross-raga mean")
        ax.set_xlabel(lab); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_yticks(range(len(ragas))); axs[0].set_yticklabels(ragas); axs[0].invert_yaxis()
    axs[1].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig4_per_raag")


# ---------- Figure 5: sensitivity -------------------------------------------------------------
def fig_sensitivity():
    sw = [("SW-1", .91), ("SW-2", .75), ("SW-3", 1.46), ("SW-4", 1.09), ("SW-5", .91), ("SW-6", .91), ("SW-7", 1.38)]
    km = [("k = 2", -1.38), ("k = 3", -1.43), ("k = 4", -1.37)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [7, 3]})
    for ax, data, prim, lab in [(axs[0], sw, "SW-4", "Cohen's d (Smith–Waterman)"), (axs[1], km, "k = 3", "Cohen's d (3-mer distance)")]:
        x = np.arange(len(data))
        ax.bar(x, [d for _, d in data], width=0.62, color=[BLUE if n == prim else "#9cc0ec" for n, _ in data])
        ax.set_xticks(x); ax.set_xticklabels([n + ("\n(primary)" if n == prim else "") for n, _ in data], fontsize=7.5)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel(lab); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for xi, (_, d) in zip(x, data):
            ax.text(xi, d + (0.04 if d > 0 else -0.04), f"{d:+.2f}", ha="center", va="bottom" if d > 0 else "top", fontsize=7, color=INK2)
    axs[0].set_ylim(0, 1.7); axs[1].set_ylim(-1.7, 0)
    fig.tight_layout(); save(fig, "fig5_sensitivity")


if __name__ == "__main__":
    pdf = pd.read_csv("dev_pairwise_script1.csv")
    fig_workflow(); fig_sw(); fig_cdf(pdf); fig_per_raag(pdf); fig_sensitivity()
    print("ok")


# ---------- Figure 6: shuffle controls ----------------------------------------------------------
def fig_shuffle():
    import RagaSeq_Shuffle_Control as S
    rng = np.random.default_rng(3)
    notes = np.array([0, 2, 4, 5, 7, 5, 4, 2, 0, 11, 0, 2, 4]); durs = np.array([1.2, .4, .6, .3, 1.5, .3, .5, .3, 1.0, .4, .8, .3, .7])
    fn, fd = S.full_shuffle(notes, durs, rng); dn, dd = S.dyad_shuffle(notes, durs, rng)
    SG = "S r R g G m M P d D n N".split()
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_axes([0.02, 0.62, 0.96, 0.36]); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.4)
    ax.text(0, 3.25, "(a)", weight="bold", fontsize=9, va="top")
    for row, (lab, nn, dd_, c) in enumerate([("Intact", notes, durs, BLUE), ("Full shuffle", fn, fd, ORANGE), ("Dyad shuffle", dn, dd, AQUA)]):
        y = 2.55 - row * 1.0
        ax.text(0.0, y + 0.18, lab, fontsize=8, va="center", weight="bold", color=INK)
        x = 1.55; scale = 0.95
        for n_, d_ in zip(nn, dd_):
            w = d_ * scale
            ax.add_patch(Rectangle((x, y), w - 0.04, 0.38, fc=c, ec="white", lw=0.8, alpha=0.85))
            if w > 0.25: ax.text(x + (w - 0.04) / 2, y + 0.19, SG[n_], ha="center", va="center", fontsize=7, color="white", weight="bold")
            x += w
    ax.text(1.55, 0.18, "Bar length = held time. Both shuffles keep the total time on each note; the dyad shuffle also keeps every note-to-note transition.",
            fontsize=7, color=INK2)
    r = json.load(open("results.json"))
    sr = pd.read_csv("shuffle_replicates.csv"); ex = pd.read_csv("results_expanded/47_shuffle_replicates_expanded.csv")
    ex["set"] = "Expanded (30 raags, 218)"
    allr = pd.concat([sr, ex]); allr = allr[allr.measure == "RagaSeq-D accuracy"]
    ax2 = fig.add_axes([0.1, 0.08, 0.86, 0.46])
    ax2.text(-0.11, 1.04, "(b)", transform=ax2.transAxes, weight="bold", fontsize=9)
    sets = ["Development (10 raags x 5)", "Held-out test (14 raags x 5)", "Expanded (30 raags, 218)"]
    names = ["Development\n10 ragas", "Held-out test\n14 ragas", "All 30 ragas\n218 recordings"]
    for i, s in enumerate(sets):
        g = allr[allr.set == s]
        intact = g[g.condition == "intact"].value.iloc[0]
        for j, (cond, col, lab) in enumerate([("full_shuffle", ORANGE, "Full shuffle (content kept)"), ("dyad_shuffle", AQUA, "Dyad shuffle (transitions kept)")]):
            v = g[g.condition == cond].value.to_numpy() * 100
            xs = i + (j - 0.5) * 0.3 + rng.uniform(-0.09, 0.09, len(v))
            ax2.scatter(xs, v + rng.uniform(-0.25, 0.25, len(v)), s=4, color=col, alpha=0.25, lw=0, label=lab if i == 0 else None)
            ax2.plot([i + (j - 0.5) * 0.3 - 0.12, i + (j - 0.5) * 0.3 + 0.12], [v.mean()] * 2, color=col, lw=2.2)
        ax2.scatter([i + 0.32], [intact * 100], marker="D", s=46, color=BLUE, zorder=4, label="Intact" if i == 0 else None, ec="white", lw=0.8)
        ax2.text(i + 0.4, intact * 100, f"{intact*100:.0f}%", va="center", fontsize=7.5, color=INK)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(names); ax2.set_ylabel("Recognition accuracy (%)")
    ax2.set_xlim(-0.5, 2.7); ax2.grid(axis="y", color=GRID, lw=0.6); ax2.set_axisbelow(True)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=ORANGE, ms=5, label="Full shuffle (content kept)"),
          Line2D([], [], marker="o", ls="", color=AQUA, ms=5, label="Dyad shuffle (transitions kept)"),
          Line2D([], [], marker="D", ls="", color=BLUE, ms=6, label="Intact recordings"),
          Line2D([], [], color=INK2, lw=2, label="Mean of shuffled replicates")]
    ax2.legend(handles=hs, loc="lower left", fontsize=7.5)
    save(fig, "fig6_shuffle")


# ---------- Figure 7: per-raga accuracy on 30 ragas ----------------------------------------
def fig_per_raag30():
    d = pd.read_csv("results_expanded/44_per_raag_accuracy.csv")
    d = d.sort_values(["ragaseq_d", "tdms"], ascending=[True, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    y = np.arange(len(d))
    for col, c, lab, off, mk in [("ragaseq_d", BLUE, "RagaSeq-D", 0.0, "o"), ("tdms", ORANGE, "TDMS", 0.22, "s"), ("pcd", AQUA, "Pitch-class distribution", -0.22, "^")]:
        ax.scatter(d[col] * 100, y + off, s=26 if col == "ragaseq_d" else 18, color=c, marker=mk, label=lab, zorder=3, ec="white", lw=0.6)
    for yi, r in d.iterrows():
        ax.plot([0, 100], [yi, yi], color=GRID, lw=0.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{r} ({n})" for r, n in zip(d.true_raag, d.recordings)], fontsize=7.5)
    ax.set_xlabel("Recordings recognized (%)"); ax.set_xlim(-3, 103); ax.set_ylim(-0.7, len(d) - 0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); save(fig, "fig7_per_raag_30")


if __name__ == "__main__":
    fig_shuffle(); fig_per_raag30()
