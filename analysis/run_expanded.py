"""Expanded RagaSeq analysis: all 30 Hindustani raags, one recording per performer per raag.

Same front end and same recognition procedure as Scripts 1 and 3 (unchanged code). Adds:
  A. 30-way performer-aware recognition for RagaSeq-D, pitch-class distribution and TDMS
  B. pattern-length sweep (k = 1..4): k = 1 is held time per note (no order), k = 2 adds
     note-to-note transitions, k = 3 is RagaSeq-D
  C. shuffle control (full and dyad-preserving) on the expanded set
  D. pitch-alphabet sensitivity: 12 equal-tempered states vs 12 just-intonation states
     vs 24 quarter-tone states
  E. paired comparisons between methods (exact McNemar test)
  F. characteristic 3-note patterns for all 30 raags
"""
import json, sys, time
import numpy as np, pandas as pd
from scipy.stats import binomtest
import common as C
import RagaSeq_Shuffle_Control as S

R, D = C.R, C.D
OUT = "results_expanded"
import os; os.makedirs(OUT, exist_ok=True)
PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

sel = pd.read_csv("RagaSeq_Expanded_Selection.csv", dtype={"recording_id": str})
t0 = time.time()
recs = C.load(sel.recording_id)
labels = list(sel.raag); performers = list(sel.performer); groups = list(sel.raag_group)
n = len(recs)
print(f"{n} recordings, {len(set(labels))} raags, loaded in {time.time()-t0:.0f} s", flush=True)


def recog(M, perm=PERM):
    return D.recognise(M, labels, performers, perm, 1234)


def summarise(name, res, extra=None):
    pred = np.array(res["pred"]); true = np.array(labels); g = np.array(groups)
    row = {"method": name, "correct": res["correct"], "n": res["n"], "accuracy": res["accuracy"],
           "ci95_low": res["ci95_low"], "ci95_high": res["ci95_high"], "top3": res["top3_correct"] / res["n"],
           "permutation_p": res["permutation_p"]}
    for grp in ["development raag", "earlier test raag", "new raag"]:
        m = g == grp
        row[f"acc_{grp.replace(' ', '_')}"] = float((pred[m] == true[m]).mean())
    if extra: row.update(extra)
    return row


# ---- A. recognition, three methods -------------------------------------------------
prof = [D.duration_profile(r["runs"], 3) for r in recs]
bf = [D.baseline_features(r) for r in recs]
mats = {"RagaSeq-D": C.dist_matrix(prof, R.js_distance_dict),
        "Pitch-class distribution (Chordia & Rae 2007)": C.dist_matrix([b[0] for b in bf], D.jsd_vec),
        "TDMS (Gulati et al. 2016b)": C.dist_matrix([b[1] for b in bf], D.sym_kl)}
summ, preds = [], {"recording_id": list(sel.recording_id), "performer": performers, "true_raag": labels,
                   "raag_group": groups}
for name, M in mats.items():
    res = recog(M)
    preds[name] = res["pred"]
    sa, cr = C.same_cross(M, labels)
    d = R.cohens_d(sa, cr)
    summ.append(summarise(name, res, {"cohens_d_same_vs_cross": d}))
    print(f"A {name}: {res['correct']}/{n} = {res['accuracy']:.1%} [{res['ci95_low']:.1%}-{res['ci95_high']:.1%}] "
          f"top3 {res['top3_correct']/n:.1%} d={d:.2f}", flush=True)
pd.DataFrame(summ).to_csv(f"{OUT}/41_recognition_summary.csv", index=False)
pdf = pd.DataFrame(preds); pdf.to_csv(f"{OUT}/42_recognition_predictions.csv", index=False)
pd.crosstab(pdf.true_raag, pdf["RagaSeq-D"]).to_csv(f"{OUT}/43_confusion_ragaseq_d.csv")
per_raag = pdf.assign(ok=pdf["RagaSeq-D"] == pdf.true_raag, ok_tdms=pdf["TDMS (Gulati et al. 2016b)"] == pdf.true_raag,
                      ok_pcd=pdf["Pitch-class distribution (Chordia & Rae 2007)"] == pdf.true_raag)
per_raag.groupby(["true_raag", "raag_group"]).agg(recordings=("ok", "size"), ragaseq_d=("ok", "mean"),
    tdms=("ok_tdms", "mean"), pcd=("ok_pcd", "mean")).reset_index().to_csv(f"{OUT}/44_per_raag_accuracy.csv", index=False)

# ---- E. paired comparisons (exact McNemar) -------------------------------------------
true = np.array(labels); rows = []
names = list(mats)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = np.array(preds[names[i]]) == true; b = np.array(preds[names[j]]) == true
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        rows.append({"method_a": names[i], "method_b": names[j], "a_right_b_wrong": n01, "a_wrong_b_right": n10,
                     "both_right": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()), "mcnemar_exact_p": p})
pd.DataFrame(rows).to_csv(f"{OUT}/45_method_comparison_mcnemar.csv", index=False)
print("E", pd.DataFrame(rows)[["method_a", "method_b", "a_right_b_wrong", "a_wrong_b_right", "mcnemar_exact_p"]].to_string(), flush=True)

# ---- B. pattern-length sweep ---------------------------------------------------------
rows = []
for k in [1, 2, 3, 4]:
    pk = [D.duration_profile(r["runs"], k) for r in recs]
    M = C.dist_matrix(pk, R.js_distance_dict)
    res = recog(M, 0); sa, cr = C.same_cross(M, labels)
    rows.append(summarise(f"k = {k}", res, {"k": k, "cohens_d_same_vs_cross": R.cohens_d(sa, cr),
                                            "distinct_patterns": len(set().union(*pk))}))
    print(f"B k={k}: {res['accuracy']:.1%} d={rows[-1]['cohens_d_same_vs_cross']:.2f}", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/46_pattern_length_sweep.csv", index=False)

# ---- C. shuffle control ----------------------------------------------------------------
runs = [r["runs"] for r in recs]
rng = np.random.default_rng(1234); rows = []
intact = S.evaluate(runs, labels, performers, R, D, 3, False)
rows += [dict(condition="intact", replicate=0, measure=m, value=v) for m, v in intact.items()]
for cond, fn in S.SHUFFLES.items():
    for rep in range(1, REPS + 1):
        sh = [[fn(nt, du, rng) for nt, du in rec] for rec in runs]
        res = S.evaluate(sh, labels, performers, R, D, 3, False)
        rows += [dict(condition=cond, replicate=rep, measure=m, value=v) for m, v in res.items()]
    print(f"C {cond} done {time.time()-t0:.0f} s", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/47_shuffle_replicates_expanded.csv", index=False)

# ---- D. pitch alphabet -------------------------------------------------------------------
JI = np.array([0, 112, 204, 316, 386, 498, 590, 702, 814, 884, 996, 1088], float)
def q_ji(cents):
    f = np.mod(cents, 1200.0)[:, None]
    dd = np.abs(f - JI[None, :]); dd = np.minimum(dd, 1200 - dd)
    return dd.argmin(1)
def q_24(cents):
    return np.mod(np.rint(np.mod(cents, 1200.0) / 50.0).astype(int), 24)
orig_q = R.quantize_12
rows = []
for name, q in [("12 equal-tempered states (paper)", orig_q), ("12 just-intonation states", q_ji),
                ("24 quarter-tone states", q_24)]:
    R.quantize_12 = q
    rr = C.load(sel.recording_id)
    pk = [D.duration_profile(r["runs"], 3) for r in rr]
    M = C.dist_matrix(pk, R.js_distance_dict)
    res = recog(M, 0); sa, cr = C.same_cross(M, labels)
    rows.append(summarise(name, res, {"cohens_d_same_vs_cross": R.cohens_d(sa, cr)}))
    print(f"D {name}: {res['accuracy']:.1%}", flush=True)
R.quantize_12 = orig_q
pd.DataFrame(rows).to_csv(f"{OUT}/48_pitch_alphabet_sensitivity.csv", index=False)

# ---- F. characteristic patterns ----------------------------------------------------------
D.characteristic_patterns(prof, labels).to_csv(f"{OUT}/49_characteristic_patterns_30raags.csv", index=False)
json.dump({"n_recordings": n, "n_raags": len(set(labels)), "n_performers": len(set(performers)),
           "permutations": PERM, "shuffle_replicates": REPS, "seed": 1234,
           "selection": "one recording per performer per raag; recording used in the paper if any, else the longest; >=150 s",
           "front_end": "central 150 s, five 30-s windows, 10 Hz median bins, 12 tonic-relative states, 3-frame median"},
          open(f"{OUT}/run_summary_expanded.json", "w"), indent=2)
print("ALL DONE", round(time.time() - t0), "s")
