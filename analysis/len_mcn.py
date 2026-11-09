import numpy as np, pandas as pd, json
from scipy.stats import binomtest
import common as C
R, D = C.R, C.D
exp = pd.read_csv("RagaSeq_Expanded_Selection.csv"); ids = list(exp[exp.duration_s >= 610].recording_id)
out = []
for secs, win in [(150, 5), (300, 10), (600, 20)]:
    recs = C.load(ids, central_seconds=secs, windows=win); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
    c = {}
    for k in (1, 2, 3):
        M = C.dist_matrix([D.duration_profile(r["runs"], k) for r in recs], R.js_distance_dict)
        c[k] = np.array(D.recognise(M, lab, perf, 0, 0)["pred"]) == np.array(lab)
    row = {"seconds": secs, **{f"k{k}": c[k].mean() for k in c}}
    for a, b in [(2, 1), (3, 1), (3, 2)]:
        n01, n10 = int((c[a] & ~c[b]).sum()), int((~c[a] & c[b]).sum())
        row.update({f"k{a}_vs_k{b}_gain": n01, f"k{a}_vs_k{b}_loss": n10, f"k{a}_vs_k{b}_p": binomtest(n01, n01 + n10, .5).pvalue})
    out.append(row); print(row, flush=True)
pd.DataFrame(out).to_csv("results_expanded/59_content_vs_order_by_length.csv", index=False); json.dump(out, open("len_mcn.json", "w"))
