#!/usr/bin/env python3
"""RagaSeq expanded analysis: all 30 Hindustani ragas in the CompMusic Raga Recognition Dataset,
one recording per performer per raga.

Uses the unchanged front end and recognition code of Scripts 1 and 3, and the shuffles of
Script 4 (place all four scripts in one folder). Analyses:
  A. 30-way performer-aware recognition: RagaSeq-D, pitch-class distribution, TDMS
  B. pattern-length sweep (k = 1..4): k = 1 is held time per note (no order), k = 2 adds
     note-to-note transitions, k = 3 is RagaSeq-D
  C. shuffle control (full and dyad-preserving) on the expanded set
  D. pitch-alphabet sensitivity: 12 equal-tempered vs 12 just-intonation vs 24 quarter-tone states
  E. paired comparisons between methods (exact McNemar test)
  F. characteristic 3-note patterns for all 30 ragas

Example
-------
python RagaSeq_Expanded_Analysis.py PATH/TO/RagaDataset/Hindustani --out results_expanded

The folder must contain _info_/path_mbid_ragaid.json, _info_/ragaId_to_ragaName_mapping.json
and features/ (as extracted from Zenodo record 7278506). The selection used in the paper is
written to RagaSeq_Expanded_Selection.csv (and read from it if it already exists).
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
import RagaSeq_10Raga_Analysis_FINAL_Annotated as R
import RagaSeq_Duration_Recognition as D
import RagaSeq_Shuffle_Control as S

RAAG_NAMES = {
    "Bāgēśrī": "Bageshri", "Bhūp": "Bhoop", "Dēś": "Desh", "Yaman kalyāṇ": "Yaman Kalyan",
    "Rāgēśrī": "Rageshri", "Mārvā": "Marwa", "Miyān malhār": "Miyan Malhar",
    "Ahira bhairav": "Ahir Bhairav", "Mārūbihāg": "Marubihag", "Śuddh sāraṅg": "Shuddh Sarang",
    "Darbāri": "Darbari Kanada", "Bihāg": "Bihag", "Tōḍī": "Todi", "Alahaiyā bilāval": "Alhaiya Bilawal",
    "Bairāgi": "Bairagi", "Kēdār": "Kedar", "Gauḍ malhār": "Gaud Malhar", "Śrī": "Shree",
    "Madhuvanti": "Madhuvanti", "Puriyā dhanaśrī": "Puriya Dhanashri", "Bilāsakhānī tōḍī": "Bilaskhani Todi",
    "Ābhōgī": "Abhogi", "Basant": "Basant", "Khamāj": "Khamaj", "Jōg": "Jog", "Bhairav": "Bhairav",
    "Mālkauns": "Malkauns", "Madhukauns": "Madhukauns", "Lalit": "Lalit", "Hamsadhvāni": "Hamsadhwani",
}
SAME_PERFORMER = {"Kumar_Gandharv": "Kumar_Gandharva"}   # one singer, two spellings in the dataset
DEV_RAAGS = ["Bhairav", "Darbari Kanada", "Desh", "Hamsadhwani", "Jog", "Kedar", "Madhuvanti", "Malkauns",
             "Marwa", "Yaman Kalyan"]


# ===== STEP 1: find every recording and its pitch, tonic, raga and performer =====
def find_recordings(root):
    """recording_id -> row dict with the same keys Scripts 1 and 3 use (pitch, tonicFine, tonic, raga)."""
    meta = json.load(open(root / "_info_" / "path_mbid_ragaid.json", encoding="utf-8"))
    names = json.load(open(root / "_info_" / "ragaId_to_ragaName_mapping.json", encoding="utf-8"))
    files = {}
    for p in (root / "features").rglob("*"):
        if p.is_file() and not p.name.startswith("._"):
            m = re.search(r"([0-9a-f]{8}-[0-9a-f-]{27})\.(pitch|tonicFine|tonic)$", p.name)
            if m:
                files.setdefault(m.group(1), {})[m.group(2)] = str(p)
    rows = {}
    for rid, v in meta.items():
        f = files.get(rid, {})
        if "pitch" not in f or not ("tonicFine" in f or "tonic" in f):
            continue
        perf = v["path"].split("audio/")[1].split("/")[1]
        rows[rid] = {"recording_id": rid, "raga": RAAG_NAMES[names[v["ragaid"]]], "raag_id": v["ragaid"],
                     "performer": SAME_PERFORMER.get(perf, perf).replace("_", " "),
                     "track": v["path"].split("/")[-1].rsplit("_", 1)[0].replace("_", " "),
                     "pitch": f["pitch"], "tonicFine": f.get("tonicFine"), "tonic": f.get("tonic")}
    return rows


def build_selection(rows, published):
    """One recording per performer per raga: the one used in the paper if any, else the longest (>= 150 s)."""
    out = []
    for r in rows.values():
        t, _ = R.load_pitch(r["pitch"])
        out.append({**{k: r[k] for k in ["recording_id", "raga", "raag_id", "performer", "track"]},
                    "duration_s": float(t[-1] - t[0]), "in_published_set": r["recording_id"] in published})
    df = pd.DataFrame(out)
    df = df[df.duration_s >= 150]
    sel = (df.sort_values(["in_published_set", "duration_s"], ascending=False)
             .groupby(["raga", "performer"], as_index=False).head(1)
             .sort_values(["raga", "performer"]).reset_index(drop=True))
    return sel.rename(columns={"raga": "raag"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hindustani_dir")
    ap.add_argument("--out", default="results_expanded")
    ap.add_argument("--selection", default="RagaSeq_Expanded_Selection.csv")
    ap.add_argument("--published", nargs="*", default=["RagaSeq_10x5_Selection_with_performers.csv",
                                                        "RagaSeq_TestSet_14x5_Selection.csv"])
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--replicates", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.hindustani_dir).resolve()
    OUT, PERM, REPS = args.out, args.permutations, args.replicates
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows = find_recordings(root)
    if os.path.exists(args.selection):
        sel = pd.read_csv(args.selection, dtype={"recording_id": str})
    else:
        published = set().union(*[set(pd.read_csv(p, dtype=str).recording_id) for p in args.published if os.path.exists(p)])
        sel = build_selection(rows, published)
        sel.to_csv(args.selection, index=False)
    sel["raag_group"] = np.where(sel.raag.isin(DEV_RAAGS), "development raag", "other raag")
    run(sel, rows, OUT, PERM, REPS, t0)


class _Args:
    central_seconds, windows, sample_hz = 150.0, 5, 10.0


def load_all(sel, rows):
    recs = []
    for rid in sel.recording_id:
        r = rows[rid]
        rec = D.load_recording({"pitch": r["pitch"], "tonicFine": r["tonicFine"], "tonic": r["tonic"],
                                "raga": r["raga"], "recording_id": rid}, _Args())
        recs.append(rec)
    return recs


def dist_matrix(items, fn):
    n = len(items); M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = fn(items[i], items[j])
    return M


def same_cross(M, labels):
    iu = np.triu_indices(len(labels), 1)
    same = np.array([labels[i] == labels[j] for i, j in zip(*iu)])
    return M[iu][same], M[iu][~same]


def run(sel, rows, OUT, PERM, REPS, t0):
    recs = load_all(sel, rows)
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
        for grp in ["development raag", "other raag"]:
            m = g == grp
            row[f"acc_{grp.replace(' ', '_')}"] = float((pred[m] == true[m]).mean())
        if extra: row.update(extra)
        return row


    # ---- A. recognition, three methods -------------------------------------------------
    prof = [D.duration_profile(r["runs"], 3) for r in recs]
    bf = [D.baseline_features(r) for r in recs]
    mats = {"RagaSeq-D": dist_matrix(prof, R.js_distance_dict),
            "Pitch-class distribution (Chordia & Rae 2007)": dist_matrix([b[0] for b in bf], D.jsd_vec),
            "TDMS (Gulati et al. 2016b)": dist_matrix([b[1] for b in bf], D.sym_kl)}
    summ, preds = [], {"recording_id": list(sel.recording_id), "performer": performers, "true_raag": labels,
                       "raag_group": groups}
    for name, M in mats.items():
        res = recog(M)
        preds[name] = res["pred"]
        sa, cr = same_cross(M, labels)
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
        M = dist_matrix(pk, R.js_distance_dict)
        res = recog(M, 0); sa, cr = same_cross(M, labels)
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
        rr = load_all(sel, rows)
        pk = [D.duration_profile(r["runs"], 3) for r in rr]
        M = dist_matrix(pk, R.js_distance_dict)
        res = recog(M, 0); sa, cr = same_cross(M, labels)
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


if __name__ == "__main__":
    main()
