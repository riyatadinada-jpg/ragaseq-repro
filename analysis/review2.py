"""Second review: allied-group detail and leave-one-group-out; label review with 100 runs, CIs and paired tests;
composition of the long-recording subset."""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
import common as C
R, D = C.R, C.D
OUT = "results_expanded/"; res = {}
P = pd.read_csv(OUT + "73_allied_predictions.csv")
NOTES = {"Purvi": ("S r G M P d N", "exact (Shree, Puriya Dhanashri); Basant adds shuddh Ma"),
         "Bhairav": ("S r G m P d N", "near (Ahir Bhairav uses komal Ni)"),
         "Kalyan": ("S R G m M P D N", "near (all use shuddh notes with tivra Ma; Marubihag and Yaman Kalyan use shuddh Ma sparingly)"),
         "Malhar": ("S R g G m P D n N", "near (both Ni in all three; komal Ga only in Miyan Malhar)"),
         "Bageshri": ("S R g G m P D n", "near (Bageshri komal Ga, Rageshri shuddh Ga)")}
rows = []
def mc(a, b):
    g, l = int((a & ~b).sum()), int((~a & b).sum()); return g, l, (binomtest(g, g + l, .5).pvalue if g + l else 1.0)
for secs in (150, 600):
    S = P[P.seconds == secs]
    for grp, g in S.groupby("group", sort=False):
        key = [k for k in NOTES if grp.startswith(k)][0]
        gn, ln, p = mc(g.k3.values, g.k1.values)
        rows.append({"seconds": secs, "group": grp, "notes": NOTES[key][0], "match": NOTES[key][1], "n": len(g), "ragas": g.raga.nunique(),
                     "k1": g.k1.mean(), "k3": g.k3.mean(), "k3_only": gn, "k1_only": ln, "p": p})
    for grp in S.group.unique():  # leave one group out
        g = S[S.group != grp]; gn, ln, p = mc(g.k3.values, g.k1.values)
        rows.append({"seconds": secs, "group": "Pooled without " + grp.split(" (")[0], "n": len(g), "k1": g.k1.mean(), "k3": g.k3.mean(), "k3_only": gn, "k1_only": ln, "p": p})
    gn, ln, p = mc(S.k3.values, S.k1.values)
    chance = S.groupby("group").apply(lambda g: len(g) / g.raga.nunique(), include_groups=False).sum() / len(S)
    rows.append({"seconds": secs, "group": "All groups pooled", "n": len(S), "chance_weighted": chance, "k1": S.k1.mean(), "k3": S.k3.mean(), "k3_only": gn, "k1_only": ln, "p": p})
A = pd.DataFrame(rows); A.to_csv(OUT + "76_allied_detail.csv", index=False); res["allied_detail"] = A.to_dict("records")
print(A.round(3).to_string(), flush=True)
# composition of long subset
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv"); lg = sel[sel.duration_s >= 610]
comp = pd.DataFrame({"all_218": sel.raag.value_counts(), "long_194": lg.raag.value_counts()}).fillna(0).astype(int)
comp["dropped"] = comp.all_218 - comp.long_194; comp.to_csv(OUT + "77_long_subset_composition.csv")
res["long_subset"] = {"n_all": len(sel), "n_long": len(lg), "ragas_all": int(sel.raag.nunique()), "ragas_long": int(lg.raag.nunique()),
                      "performers_all": int(sel.performer.nunique()), "performers_long": int(lg.performer.nunique()),
                      "median_min_all": float(sel.duration_s.median() / 60), "median_min_long": float(lg.duration_s.median() / 60),
                      "dropped_by_raga": comp[comp.dropped > 0].dropped.to_dict()}
print(res["long_subset"], flush=True)
# label review: 100 runs, same planted errors for every method
recs = C.load(sel.recording_id); lab = np.array(sel.raag); perf = np.array([p.replace("Kumar Gandharva", "Kumar Gandharv") for p in sel.performer]); n = len(lab)
bf = [D.baseline_features(r) for r in recs]
MS = {"RagaSeq-D (k = 3)": C.dist_matrix([D.duration_profile(r["runs"], 3) for r in recs], R.js_distance_dict),
      "Notes only (k = 1)": C.dist_matrix([D.duration_profile(r["runs"], 1) for r in recs], R.js_distance_dict),
      "Pitch-class distribution": C.dist_matrix([b[0] for b in bf], D.jsd_vec), "TDMS": C.dist_matrix([b[1] for b in bf], D.sym_kl)}
def ranks(M, labels):
    rk = []
    for i in range(n):
        ok = np.where(perf != perf[i])[0]; rd = M[np.ix_(ok, ok)][np.triu_indices(len(ok), 1)]
        s = -(M[i, ok] - rd.mean()) / rd.std(ddof=1); L = labels[ok]; m = {g: s[L == g].mean() for g in set(L)}
        order = sorted(m, key=lambda g: -m[g]); rk.append(order.index(labels[i]) if labels[i] in order else 99)
    return np.array(rk)
rng = np.random.default_rng(2026); raags = sorted(set(lab)); lc = []
for rep in range(100):
    lab2 = lab.copy(); bad = rng.choice(n, 22, replace=False)
    for b in bad: lab2[b] = rng.choice([g for g in raags if g != lab[b]])
    isbad = np.zeros(n, bool); isbad[bad] = True
    for name, M in MS.items():
        flag = ranks(M, lab2) >= 3; tp, fp = int((flag & isbad).sum()), int((flag & ~isbad).sum())
        lc.append({"replicate": rep + 1, "method": name, "flagged": int(flag.sum()), "tp": tp, "fp": fp, "recall": tp / 22, "precision": tp / max(1, flag.sum()), "fpr": fp / (n - 22)})
LC = pd.DataFrame(lc); LC.to_csv(OUT + "78_label_review_100.csv", index=False)
summ = []
for m, g in LC.groupby("method", sort=False):
    row = {"method": m}
    for c in ["flagged", "recall", "precision", "fpr"]:
        row[c] = g[c].mean(); row[c + "_lo"] = g[c].quantile(.025); row[c + "_hi"] = g[c].quantile(.975)
    summ.append(row)
W = LC.pivot(index="replicate", columns="method", values="precision"); pair = []
for m in [x for x in MS if x != "RagaSeq-D (k = 3)"]:
    d = W["RagaSeq-D (k = 3)"] - W[m]
    pair.append({"comparison": f"RagaSeq-D vs {m}", "mean_precision_difference": d.mean(), "runs_RagaSeqD_higher": int((d > 0).sum()), "runs_lower": int((d < 0).sum()), "wilcoxon_p": wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 0 else 1.0})
res["label_review_100"] = summ; res["label_review_pairs"] = pair
print(pd.DataFrame(summ).round(3).to_string()); print(pd.DataFrame(pair).round(4).to_string())
pd.DataFrame(summ).to_csv(OUT + "79_label_review_summary.csv", index=False); pd.DataFrame(pair).to_csv(OUT + "80_label_review_pairs.csv", index=False)
json.dump(res, open("review2.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); print("ALL DONE")
