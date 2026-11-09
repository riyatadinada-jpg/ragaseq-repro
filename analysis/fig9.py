import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Serif"})
rows = [("Archive retrieval", BLUE, "Archivist or listener\nsubmits a recording", "Ranked recordings by\nother singers, with the\nshared 3-note patterns", "Nearest recording same raga:\n69% (permuted labels 3%);\nTDMS 78%, best of four"),
        ("Label review", ORANGE, "Archive checks every\nrecording against\nits stated raga", "Short list of recordings\nwhose label ranks outside\nthe top 3, with patterns", "Planted errors: recall 94%,\nprecision 55%, fewest false\nflags of four methods"),
        ("Listener study\n(future work)", YELLOW, "Researcher selects\nreal performances", "Intact, dyad-shuffled\nand fully shuffled\nversions, same notes", "Proposed: do listeners\nand brain responses\nuse the order?")]
fig, ax = plt.subplots(figsize=(11.2, 4.4)); ax.set_xlim(0, 11.2); ax.set_ylim(0, len(rows) * 1.4 + 0.5); ax.axis("off")
heads = [(2.5, "Input"), (4.5, "RagaSeq-D"), (6.65, "Output"), (9.2, "Evidence in this study")]
for x, h in heads: ax.text(x, len(rows) * 1.4 + 0.25, h, ha="center", fontsize=9.5, weight="bold", color=INK)
for i, (name, c, a, o, e) in enumerate(rows):
    y = (len(rows) - 1 - i) * 1.4 + 0.15
    ax.text(0.05, y + 0.55, name, fontsize=9, weight="bold", color=c, va="center")
    mid = "shuffle the\n12-state note\nstring" if i == len(rows) - 1 else "12-state\nnote string\n→ 3-note profile"
    for x0, w, t, fc in [(1.7, 1.6, a, "white"), (3.75, 1.5, mid, c + "22"), (5.7, 1.9, o, "white"), (8.05, 2.3, e, "#f4f3ef")]:
        ax.add_patch(FancyBboxPatch((x0, y), w, 1.1, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=c, lw=1.2))
        ax.text(x0 + w / 2, y + 0.55, t, ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.25)
    for x0, x1 in [(3.3, 3.75), (5.25, 5.7), (7.6, 8.05)]:
        ax.add_patch(FancyArrowPatch((x0, y + 0.55), (x1, y + 0.55), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
fig.savefig("../figures/fig9_practical_uses.png", dpi=300, bbox_inches="tight", facecolor="white"); print("ok")
