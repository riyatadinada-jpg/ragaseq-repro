"""Holm correction over the nine length-by-k McNemar comparisons (59_content_vs_order_by_length.csv);
adds len_holm to review_analyses.json (run after review_analyses.py)."""
import json, numpy as np, pandas as pd
d = pd.read_csv("results_expanded/59_content_vs_order_by_length.csv"); rows = []
for _, r in d.iterrows():
    for c in ["k2_vs_k1", "k3_vs_k1", "k3_vs_k2"]: rows.append({"seconds": int(r.seconds), "comparison": c, "p": float(r[c + "_p"])})
p = np.array([r["p"] for r in rows]); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
for rank, i in enumerate(o): run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
for r, a in zip(rows, adj): r["p_holm"] = float(a)
pd.DataFrame(rows).to_csv("results_expanded/59b_length_holm.csv", index=False)
res = json.load(open("review_analyses.json")); res["len_holm"] = rows; json.dump(res, open("review_analyses.json", "w"), indent=1)
print(pd.DataFrame(rows).to_string())
