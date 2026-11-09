"""Table 3, 'Original RagaSeq' row: recognition from Script 1's window-averaged Smith-Waterman similarity and
3-mer Jensen-Shannon distance, combined as z(distance) - z(similarity) over all recording pairs, with the same
performer-aware leave-one-out decision rule as RagaSeq-D.
usage: python original_ragaseq_recognition.py <selection.csv> <Script 1 output folder>"""
import sys, numpy as np, pandas as pd
import common as C
sel, out = sys.argv[1], sys.argv[2]
s = pd.read_csv(sel); perf = dict(zip(s.recording_id, s.performer)) if "performer" in s else {}
p = pd.read_csv(out + "/03_pairwise_comparisons.csv")
ids = list(s.recording_id); idx = {r: i for i, r in enumerate(ids)}; n = len(ids)
lab = [C.MD[r]["raag"] for r in ids]; pf = [perf.get(r, "") for r in ids]
def mat(col):
    M = np.zeros((n, n))
    for a, b, v in zip(p.recording_a, p.recording_b, p[col]): M[idx[a], idx[b]] = M[idx[b], idx[a]] = v
    return M
sw, jd = mat("smith_waterman_norm_mean"), mat("kmer_js_distance_mean"); iu = np.triu_indices(n, 1)
z = lambda M: (M - M[iu].mean()) / M[iu].std()
Dm = z(jd) - z(sw); np.fill_diagonal(Dm, 0)
r = C.D.recognise(Dm, lab, pf, 2000, 1234); r.pop("pred")
pd.DataFrame([{"method": "Original RagaSeq (Smith-Waterman + 3-mer)", **r}]).to_csv(out + "/27_original_ragaseq_recognition.csv", index=False)
print(f"Original RagaSeq: {r['correct']}/{r['n']} ({r['accuracy']:.1%}); top-3 {r['top3_correct']}")
