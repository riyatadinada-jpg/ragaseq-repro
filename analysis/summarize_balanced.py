"""Summarize the balanced 29 x 5 draws (50_balanced_29x5.csv) into balanced.json for make_v4data.py."""
import json, numpy as np, pandas as pd
d = pd.read_csv("results_expanded/50_balanced_29x5.csv"); out = {}
it = d[d.analysis == "intact"]
for m, g in it.groupby("method"):
    v = g.accuracy.to_numpy(); out[m] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
rq = it[it.method == "RagaSeq-D"].set_index("draw").accuracy
for c in ["full_shuffle", "dyad_shuffle"]:
    g = d[d.analysis == c]; v = g.accuracy.to_numpy()
    out[c] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "max": float(v.max()),
              "frac_ge_intact_mean": float((v >= out["RagaSeq-D"]["mean"]).mean()), "n": int(len(v)),
              "frac_ge_same_draw": float((g.accuracy.to_numpy() >= rq.loc[g.draw].to_numpy()).mean())}
json.dump(out, open("balanced.json", "w"), indent=1); print(json.dumps(out, indent=1))
