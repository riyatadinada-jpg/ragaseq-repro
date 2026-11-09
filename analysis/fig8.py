"""Figure 8: content-only baseline vs order-aware profiles vs shuffles, and shuffle penalty by audio length."""
import json, numpy as np, pandas as pd, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "legend.frameon": False})
V = json.load(open("v4.json"))
sets = V["sets"]
labels = [s["label"] for s in sets]
series = [("k = 1: notes and held time only", "k1", "#9cc0ec"), ("k = 2: + transitions", "k2", AQUA),
          ("k = 3: RagaSeq-D (intact)", "k3", BLUE), ("k = 3, dyad-shuffled", "dyad", YELLOW), ("k = 3, fully shuffled", "full", ORANGE)]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [2.5, 1]})
x = np.arange(len(sets)); w = 0.16
for i, (name, key, c) in enumerate(series):
    a.bar(x + (i - 2) * w, [100 * s[key] for s in sets], w, color=c, label=name, edgecolor="white", lw=0.4)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.6); a.set_ylabel("Recognition accuracy (%)"); a.set_ylim(0, 100)
a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
a.set_title("(a) Content-only baseline, order-aware profiles and shuffles", fontsize=9, loc="left", pad=34)
L = pd.DataFrame(V["length"])
b.plot(L.seconds, 100 * L.k1, "o-", color="#9cc0ec", lw=2, label="k = 1 (content only)")
b.plot(L.seconds, 100 * L.k3_intact, "o-", color=BLUE, lw=2, label="k = 3 intact")
b.fill_between(L.seconds, 100 * L.k3_full_shuffle_lo, 100 * L.k3_full_shuffle_hi, color=ORANGE, alpha=0.25, lw=0)
b.plot(L.seconds, 100 * L.k3_full_shuffle_mean, "o-", color=ORANGE, lw=2, label="k = 3 fully shuffled")
b.set_xticks(L.seconds); b.set_xlabel("Audio analysed per recording (s)"); b.set_ylim(40, 100)
b.grid(color=GRID, lw=0.6); b.set_axisbelow(True); b.legend(fontsize=7, loc="lower right")
b.set_title(f"(b) Shuffle penalty vs audio length\n(CompMusic, {int(L.n[0])} recordings)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig("../figures/fig8_content_vs_order.png", bbox_inches="tight", facecolor="white")
print("fig8 ok")
