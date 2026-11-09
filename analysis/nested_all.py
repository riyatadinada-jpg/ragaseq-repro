"""Content vs order, with a content-only baseline.
A. k = 1 (held time per note, no order), k = 2, k = 3 on every set; exact McNemar on the same recordings.
B. Balanced 29 x 5 draws: k = 1, 2, 3 on the same 200 draws.
C. Sparsity test: does full-shuffled k = 3 approach the k = 1 baseline as more audio is analysed?
   Same recordings (expanded set, >= 600 s), 150 / 300 / 600 s (5 / 10 / 20 windows of 30 s), 20 shuffles each.
"""
import json, time, numpy as np, pandas as pd
from scipy.stats import binomtest
import common as C, run_dap_meta as DM
import RagaSeq_Shuffle_Control as S
R, D = C.R, C.D
OUT = "results_expanded/"
def mat(recs, k): return C.dist_matrix([D.duration_profile(r["runs"], k) for r in recs], R.js_distance_dict)
def corr(M, lab, perf): return np.array(D.recognise(M, lab, perf, 0, 0)["pred"]) == np.array(lab)
def mcn(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0)
t0 = time.time(); rows = []
dev = pd.read_csv("RagaSeq_10x5_Selection.csv"); tst = pd.read_csv("RagaSeq_TestSet_14x5_Selection.csv")
exp = pd.read_csv("RagaSeq_Expanded_Selection.csv")
sets = {"CompMusic development (10 x 5)": list(dev.recording_id), "CompMusic held-out (14 x 5)": list(tst.recording_id),
        "CompMusic expanded (30 raags, 218)": list(exp.recording_id)}
for f in ["alap", "pakad"]:
    sets[f"DAP-Lab {f} ({(DM.M.form == f).sum()})"] = list(DM.M[DM.M.form == f].id)
store = {}
for name, ids in sets.items():
    recs = C.load(ids); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
    c = {k: corr(mat(recs, k), lab, perf) for k in (1, 2, 3)}
    row = {"set": name, "n": len(ids), **{f"acc_k{k}": c[k].mean() for k in (1, 2, 3)}}
    for a, b in [(2, 1), (3, 1), (3, 2)]:
        n01, n10, p = mcn(c[a], c[b]); row.update({f"k{a}_vs_k{b}_gain": n01, f"k{a}_vs_k{b}_loss": n10, f"k{a}_vs_k{b}_p": p})
    rows.append(row); print(name, {k: round(v, 3) for k, v in row.items() if k.startswith("acc")}, round(time.time() - t0), flush=True)
pd.DataFrame(rows).to_csv(OUT + "56_content_vs_order_nested.csv", index=False)

# B balanced
sel = exp; cnt = sel.raag.value_counts(); keep = cnt[cnt >= 5].index
sel = sel[sel.raag.isin(keep)].reset_index(drop=True)
recs = C.load(sel.recording_id); lab = np.array(sel.raag); perf = list(sel.performer)
rng = np.random.default_rng(1234)
draws = [np.sort(np.concatenate([rng.choice(np.where(lab == r)[0], 5, replace=False) for r in sorted(keep)])) for _ in range(200)]
Ms = {k: mat(recs, k) for k in (1, 2, 3)}
br = []
for i, idx in enumerate(draws):
    for k, M in Ms.items():
        br.append({"draw": i, "k": k, "accuracy": D.recognise(M[np.ix_(idx, idx)], list(lab[idx]), [perf[j] for j in idx], 0, 0)["accuracy"]})
B = pd.DataFrame(br); B.to_csv(OUT + "57_balanced_nested.csv", index=False)
W = B.pivot(index="draw", columns="k", values="accuracy")
bal = {f"k{k}": W[k].mean() for k in (1, 2, 3)}
bal.update({"k3_minus_k1_mean": (W[3] - W[1]).mean(), "draws_k3_gt_k1": float((W[3] > W[1]).mean()),
            "k2_minus_k1_mean": (W[2] - W[1]).mean(), "draws_k2_gt_k1": float((W[2] > W[1]).mean())})
print("balanced", {k: round(v, 3) for k, v in bal.items()}, round(time.time() - t0), flush=True)

# C sparsity
long_ids = list(exp[exp.duration_s >= 610].recording_id); lr = []
for secs, win in [(150, 5), (300, 10), (600, 20)]:
    recs = C.load(long_ids, central_seconds=secs, windows=win); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
    k1 = corr(mat(recs, 1), lab, perf).mean(); k3 = corr(mat(recs, 3), lab, perf).mean()
    rs = np.random.default_rng(1234); sh = []
    for rep in range(20):
        sh.append(S.evaluate([[S.full_shuffle(n, d, rs) for n, d in r["runs"]] for r in recs], lab, perf, R, D, 3, False)["RagaSeq-D accuracy"])
    lr.append({"seconds": secs, "n": len(recs), "k1": k1, "k3_intact": k3, "k3_full_shuffle_mean": float(np.mean(sh)),
               "k3_full_shuffle_lo": float(np.percentile(sh, 2.5)), "k3_full_shuffle_hi": float(np.percentile(sh, 97.5))})
    print(lr[-1], round(time.time() - t0), flush=True)
pd.DataFrame(lr).to_csv(OUT + "58_shuffle_sparsity_by_length.csv", index=False)
json.dump({"nested": rows, "balanced": bal, "length": lr}, open("nested.json", "w"), default=float)
print("ALL DONE")
