"""Practical demonstrations on the CompMusic expanded set (218 recordings, central 150 s, RagaSeq-D k = 3).
1 Archive search: for each recording, are its nearest recordings (other performers) the same raga?
2 Label checking: corrupt 10% of labels at random (20 repeats); flag recordings whose stated raga is not
  among the top 3 of the model; count wrong labels caught and correct labels flagged.
3 Student-feedback example: one DAP-Lab alap, top ragas and the 3-note patterns that pull it toward the wrong raga."""
import json, numpy as np, pandas as pd
import common as C
R, D = C.R, C.D
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
recs = C.load(sel.recording_id); lab = np.array(sel.raag); perf = np.array([p.replace("Kumar Gandharva", "Kumar Gandharv") for p in sel.performer])
prof = [D.duration_profile(r["runs"], 3) for r in recs]
M = C.dist_matrix(prof, R.js_distance_dict); n = len(lab)
out = {}
# 1 search
for k in (1, 3, 5):
    hits = []
    for i in range(n):
        ok = np.where(perf != perf[i])[0]; nn = ok[np.argsort(M[i, ok])[:k]]
        hits.append((lab[nn] == lab[i]).mean())
    out[f"precision_at_{k}"] = float(np.mean(hits))
# 2 label check
def ranks(labels):
    rk = []
    for i in range(n):
        ok = np.where(perf != perf[i])[0]; rd = M[np.ix_(ok, ok)][np.triu_indices(len(ok), 1)]
        s = -(M[i, ok] - rd.mean()) / rd.std(ddof=1); L = labels[ok]
        m = {g: s[L == g].mean() for g in set(L)}; order = sorted(m, key=lambda g: -m[g])
        rk.append(order.index(labels[i]) if labels[i] in order else 99)
    return np.array(rk)
rng = np.random.default_rng(1234); raags = sorted(set(lab)); res = []
for rep in range(20):
    lab2 = lab.copy(); bad = rng.choice(n, 22, replace=False)
    for b in bad: lab2[b] = rng.choice([g for g in raags if g != lab[b]])
    rk = ranks(lab2); flag = rk >= 3; isbad = np.zeros(n, bool); isbad[bad] = True
    res.append({"caught": int((flag & isbad).sum()), "false_flags": int((flag & ~isbad).sum()), "flagged": int(flag.sum())})
R2 = pd.DataFrame(res); out["label_check"] = {"wrong_labels": 22, "caught_mean": R2.caught.mean(), "false_flags_mean": R2.false_flags.mean(),
    "flagged_mean": R2.flagged.mean(), "caught_min": int(R2.caught.min()), "caught_max": int(R2.caught.max())}
rk0 = ranks(lab); out["label_check_clean_flagged"] = int((rk0 >= 3).sum())
print(json.dumps(out, indent=1)); json.dump(out, open("practical_demo.json", "w"), indent=1)
pd.DataFrame(res).to_csv("results_expanded/60_label_check_simulation.csv", index=False)

# 3 student-feedback example
import run_dap_meta as DM
SAR = "S r R g G m M P d D n N".split()
q = pd.read_csv("results_dap/66_dap_external_predictions.csv")
ex = q[(q.raag == "Shree") & (q["RagaSeq-D"] == "Todi")].iloc[0]
qr = C.load([ex.id])[0]; pq = D.duration_profile(qr["runs"], 3)
def mean_prof(g):
    ps = [prof[i] for i in np.where(lab == g)[0]]; keys = set().union(*ps)
    return {u: np.mean([p.get(u, 0) for p in ps]) for u in keys}
ms, mt = mean_prof("Shree"), mean_prof("Todi")
name = lambda u: " ".join(SAR[int(x) % 12] for x in (u if isinstance(u, (tuple, list)) else str(u).split()))
pull = sorted(pq, key=lambda u: -pq[u] * (mt.get(u, 0) - ms.get(u, 0)))[:3]
miss = sorted(ms, key=lambda u: -(ms[u] - pq.get(u, 0)) * (ms[u] - mt.get(u, 0)))[:3]
out["feedback_example"] = {"singer": ex.singer, "file": ex.file, "true": "Shree", "predicted": "Todi",
    "patterns_toward_todi": [(name(u), round(100 * pq[u], 1), round(100 * ms.get(u, 0), 1), round(100 * mt.get(u, 0), 1)) for u in pull],
    "shree_patterns_underused": [(name(u), round(100 * pq.get(u, 0), 1), round(100 * ms[u], 1)) for u in miss]}
print(json.dumps(out["feedback_example"], indent=1)); json.dump(out, open("practical_demo.json", "w"), indent=1)
