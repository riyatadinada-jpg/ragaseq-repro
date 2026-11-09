"""Analyses requested in review.
A. DAP-Lab confirmation of the duration effect: alaps >= 180 s, central 60/120/180 s (2/4/6 windows of 30 s), k = 1, 2, 3, McNemar.
B. Allied ragas: groups with (nearly) the same note set fixed in advance from standard scale descriptions;
   recognition restricted to the ragas of each group; k = 1, 2, 3, PCD, TDMS at 150 s; k = 1, 3 at 600 s.
C. Label checking: recall, precision, false-positive rate for RagaSeq-D, k = 1, PCD, TDMS (22 planted errors, 20 repeats).
D. Retrieval: same-raga rate of nearest 1 / 5 recordings by other singers, four methods; null by 1,000 label permutations."""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest
import common as C, run_dap_meta as DM
R, D = C.R, C.D
OUT = "results_expanded/"; res = {}
def mcn(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum()); return n01, n10, (binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0)
def kmat(recs, k): return C.dist_matrix([D.duration_profile(r["runs"], k) for r in recs], R.js_distance_dict)
def bmats(recs):
    bf = [D.baseline_features(r) for r in recs]
    return C.dist_matrix([b[0] for b in bf], D.jsd_vec), C.dist_matrix([b[1] for b in bf], D.sym_kl)
def corr(M, lab, perf): return np.array(D.recognise(M, list(lab), list(perf), 0, 0)["pred"]) == np.array(lab)
# A
al = DM.M[(DM.M.form == "alap") & (DM.M.duration_s >= 180)]
rows = []
for secs, win in [(60, 2), (120, 4), (180, 6)]:
    recs = C.load(list(al.id), central_seconds=secs, windows=win); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
    c = {k: corr(kmat(recs, k), lab, perf) for k in (1, 2, 3)}; row = {"seconds": secs, "n": len(recs), **{f"k{k}": c[k].mean() for k in c}}
    for a, b in [(2, 1), (3, 1), (3, 2)]:
        g, l, p = mcn(c[a], c[b]); row.update({f"k{a}_vs_k{b}_gain": g, f"k{a}_vs_k{b}_loss": l, f"k{a}_vs_k{b}_p": p})
    rows.append(row); print("A", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
res["dap_length"] = rows; pd.DataFrame(rows).to_csv(OUT + "71_dap_length_confirmation.csv", index=False)
# B
GROUPS = {"Purvi group (komal Re, komal Dha, tivra Ma)": ["Shree", "Puriya Dhanashri", "Basant"],
          "Bhairav group (komal Re, komal Dha, shuddh Ma)": ["Bhairav", "Ahir Bhairav"],
          "Kalyan–Bihag group (shuddh notes with tivra Ma)": ["Bihag", "Marubihag", "Kedar", "Yaman Kalyan"],
          "Malhar–Desh group (both Ni)": ["Desh", "Gaud Malhar", "Miyan Malhar"],
          "Bageshri–Rageshri (komal Ni)": ["Bageshri", "Rageshri"]}
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv"); brows, recs_all = [], []
for secs, win in [(150, 5), (600, 20)]:
    pooled = {m: [] for m in ["k1", "k2", "k3", "PCD", "TDMS"]}; nall = 0; chance = []
    for g, ragas in GROUPS.items():
        s = sel[sel.raag.isin(ragas)]
        if secs == 600: s = s[s.duration_s >= 610]
        recs = C.load(list(s.recording_id), central_seconds=secs, windows=win); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
        c = {f"k{k}": corr(kmat(recs, k), lab, perf) for k in ((1, 2, 3) if secs == 150 else (1, 3))}
        if secs == 150: P, T = bmats(recs); c["PCD"] = corr(P, lab, perf); c["TDMS"] = corr(T, lab, perf)
        row = {"seconds": secs, "group": g, "ragas": ", ".join(ragas), "n": len(recs), "chance": 1 / len(ragas), **{m: v.mean() for m, v in c.items()}}
        brows.append(row); nall += len(recs); chance += [1 / len(ragas)] * len(recs)
        for m, v in c.items(): pooled[m].append(v)
        for i, r in enumerate(recs): recs_all.append({"seconds": secs, "group": g, "recording_id": r["recording_id"], "raga": lab[i], **{m: bool(v[i]) for m, v in c.items()}})
    P = {m: np.concatenate(v) for m, v in pooled.items() if v}
    row = {"seconds": secs, "group": "All groups pooled", "n": nall, "chance": float(np.mean(chance)), **{m: v.mean() for m, v in P.items()}}
    for a, b in [("k3", "k1"), ("k2", "k1"), ("k3", "PCD"), ("k3", "TDMS")]:
        if a in P and b in P: g_, l_, p_ = mcn(P[a], P[b]); row.update({f"{a}_vs_{b}": f"{g_}:{l_}", f"{a}_vs_{b}_p": p_})
    brows.append(row); print("B", secs, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
res["allied"] = brows; pd.DataFrame(brows).to_csv(OUT + "72_allied_ragas.csv", index=False); pd.DataFrame(recs_all).to_csv(OUT + "73_allied_predictions.csv", index=False)
# C + D
recs = C.load(sel.recording_id); lab = np.array(sel.raag); perf = np.array([p.replace("Kumar Gandharva", "Kumar Gandharv") for p in sel.performer]); n = len(lab)
P, T = bmats(recs); MS = {"RagaSeq-D (k = 3)": kmat(recs, 3), "Notes only (k = 1)": kmat(recs, 1), "Pitch-class distribution": P, "TDMS": T}
def ranks(M, labels):
    rk = []
    for i in range(n):
        ok = np.where(perf != perf[i])[0]; rd = M[np.ix_(ok, ok)][np.triu_indices(len(ok), 1)]
        s = -(M[i, ok] - rd.mean()) / rd.std(ddof=1); L = labels[ok]; m = {g: s[L == g].mean() for g in set(L)}
        order = sorted(m, key=lambda g: -m[g]); rk.append(order.index(labels[i]) if labels[i] in order else 99)
    return np.array(rk)
raags = sorted(set(lab)); lc = []
for name, M in MS.items():
    rng = np.random.default_rng(1234)
    for rep in range(20):
        lab2 = lab.copy(); bad = rng.choice(n, 22, replace=False)
        for b in bad: lab2[b] = rng.choice([g for g in raags if g != lab[b]])
        flag = ranks(M, lab2) >= 3; isbad = np.zeros(n, bool); isbad[bad] = True
        tp, fp = int((flag & isbad).sum()), int((flag & ~isbad).sum())
        lc.append({"method": name, "replicate": rep + 1, "planted_errors": 22, "flagged": int(flag.sum()), "true_positives": tp, "false_positives": fp,
                   "recall": tp / 22, "precision": tp / max(1, flag.sum()), "false_positive_rate": fp / (n - 22)})
LC = pd.DataFrame(lc); LC.to_csv(OUT + "74_label_check_all_methods.csv", index=False)
res["label_check"] = LC.groupby("method", sort=False)[["flagged", "true_positives", "false_positives", "recall", "precision", "false_positive_rate"]].mean().reset_index().to_dict("records")
print("C", pd.DataFrame(res["label_check"]).round(3).to_string(), flush=True)
def retrieval(M, labels, k):
    h = []
    for i in range(n):
        ok = np.where(perf != perf[i])[0]; nn = ok[np.argsort(M[i, ok])[:k]]; h.append((labels[nn] == labels[i]).mean())
    return float(np.mean(h))
rng = np.random.default_rng(1234); null1, null5 = [], []
M0 = MS["RagaSeq-D (k = 3)"]
for _ in range(1000):
    pl = rng.permutation(lab); null1.append(retrieval(M0, pl, 1)); null5.append(retrieval(M0, pl, 5))
rt = [{"method": m, "same_raga_nearest_1": retrieval(M, lab, 1), "same_raga_nearest_5": retrieval(M, lab, 5)} for m, M in MS.items()]
res["retrieval"] = rt; res["retrieval_null"] = {"nearest_1_mean": float(np.mean(null1)), "nearest_1_975": float(np.percentile(null1, 97.5)),
                                                "nearest_5_mean": float(np.mean(null5)), "nearest_5_975": float(np.percentile(null5, 97.5))}
pd.DataFrame(rt).to_csv(OUT + "75_retrieval_all_methods.csv", index=False)
print("D", pd.DataFrame(rt).round(3).to_string(), res["retrieval_null"], flush=True)
json.dump(res, open("review_analyses.json", "w"), indent=1, default=float); print("ALL DONE")
