"""Balanced robustness check on the expanded CompMusic set.
29 ragas with >= 5 performers (all but Khamaj); in each draw, 5 recordings per raga are sampled at random
(145 recordings) and performer-aware leave-one-out recognition is repeated. 200 draws for intact data.
Shuffle control: 100 shuffled replicates of all recordings; each is evaluated on 5 random balanced draws.
Distances are computed once on all recordings and subset per draw, so draws differ only in which recordings are used.
"""
import sys, time
import numpy as np, pandas as pd
import common as C
import RagaSeq_Shuffle_Control as S
R, D = C.R, C.D
OUT = "results_expanded/"
NDRAW, NREP, DPR = 200, 100, 5
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
cnt = sel.raag.value_counts(); keep = cnt[cnt >= 5].index
sel = sel[sel.raag.isin(keep)].reset_index(drop=True)
recs = C.load(sel.recording_id); lab = np.array(sel.raag); perf = list(sel.performer)
rng = np.random.default_rng(1234)
draws = [np.sort(np.concatenate([rng.choice(np.where(lab == r)[0], 5, replace=False) for r in sorted(keep)])) for _ in range(NDRAW)]


def acc(M, idx):
    res = D.recognise(M[np.ix_(idx, idx)], list(lab[idx]), [perf[i] for i in idx], 0, 0)
    return res["accuracy"]


t0 = time.time()
prof = [D.duration_profile(r["runs"], 3) for r in recs]; bf = [D.baseline_features(r) for r in recs]
mats = {"RagaSeq-D": C.dist_matrix(prof, R.js_distance_dict), "Pitch-class distribution": C.dist_matrix([b[0] for b in bf], D.jsd_vec),
        "TDMS": C.dist_matrix([b[1] for b in bf], D.sym_kl)}
rows = [{"analysis": "intact", "method": m, "draw": i, "accuracy": acc(M, idx)} for m, M in mats.items() for i, idx in enumerate(draws)]
print("intact done", round(time.time() - t0), "s", flush=True)
runs = [r["runs"] for r in recs]
for cond, fn in S.SHUFFLES.items():
    for rep in range(NREP):
        sp = [D.duration_profile([fn(n, d, rng) for n, d in rec], 3) for rec in runs]
        M = C.dist_matrix(sp, R.js_distance_dict)
        for j in rng.choice(NDRAW, DPR, replace=False):
            rows.append({"analysis": cond, "method": "RagaSeq-D", "draw": int(j), "replicate": rep, "accuracy": acc(M, draws[j])})
    print(cond, "done", round(time.time() - t0), "s", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT + "50_balanced_29x5.csv", index=False)
print(df.groupby(["analysis", "method"]).accuracy.describe(percentiles=[.025, .5, .975]).round(3).to_string())
