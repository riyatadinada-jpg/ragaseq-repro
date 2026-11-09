"""Independent replication on the DAP-Lab Hindustani vocal dataset (Clayton, Rao et al.; OSF qjkzs).

11 singers x 9 ragas, studio recordings, alap and pakad; pitch in cents re singer tonic at 10 ms,
extracted with a different pipeline (source separation + their own pitch tracking).
A. Within-dataset performer-aware leave-one-out recognition (alap; pakad), three methods.
B. Shuffle controls (full, dyad) within the dataset.
C. External test: alaps of the 6 shared ragas recognized against the 218 CompMusic references (30 candidates).
"""
import gzip, os, sys, time, json
import numpy as np, pandas as pd
from scipy.stats import beta, binomtest
import common as C
import RagaSeq_Shuffle_Control as S
R, D = C.R, C.D
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SRC = os.environ.get("DAP_FILE", os.path.join(os.environ.get("RAGASEQ_DATA", "../data"), "dap_pitch.csv.gz"))
TON = "/tmp/dap/00_data/02_singer_specific_tonic/"
OUT = "results_dap/"; os.makedirs(OUT, exist_ok=True)
ABBR = {"Bag": "Bageshri", "Bahar": "Bahar", "Bilas": "Bilaskhani Todi", "Jaun": "Jaunpuri", "Kedar": "Kedar",
        "Marwa": "Marwa", "MM": "Miyan Malhar", "Nand": "Nand", "Shree": "Shree"}

df = pd.read_csv(SRC)
df["pitch"] = pd.to_numeric(df["pitch"], errors="coerce"); df["time"] = pd.to_numeric(df["time"], errors="coerce")
recs, meta = [], []
for fn, g in df.groupby("filename", sort=True):
    parts = fn.replace(".csv", "").split("_"); singer, kind, raag = parts[0], parts[1], parts[-1]
    tonic = float(open(TON + f"{singer}_tonic.txt").read().split()[0])
    g = g.sort_values("time")
    t = g.time.to_numpy(float); c = g.pitch.to_numpy(float)
    hz = np.where(np.isfinite(c) & (c > -2400), tonic * 2 ** (c / 1200.0), 0.0)
    key = "DAP:" + fn
    C.PK[key] = (t, hz, tonic, {"full_start": t[0], "full_end": t[-1]})
    C.MD[key] = {"raag": ABBR.get(raag, raag), "performer": singer}
    meta.append({"id": key, "file": fn, "singer": singer, "form": "pakad" if kind.lower().startswith("pakad") else "alap",
                 "raag": ABBR.get(raag, raag), "duration_s": t[-1] - t[0], "voiced": float((hz > 0).mean())})
M = pd.DataFrame(meta); M.to_csv(OUT + "61_dap_recordings.csv", index=False)
print(M.groupby("form").agg(n=("id", "size"), singers=("singer", "nunique"), raags=("raag", "nunique"),
                            median_s=("duration_s", "median"), min_s=("duration_s", "min")))


def load(ids):
    return C.load(list(ids))


def summ(name, res, extra=None):
    return {"method": name, "correct": res["correct"], "n": res["n"], "accuracy": res["accuracy"], "ci95_low": res["ci95_low"],
            "ci95_high": res["ci95_high"], "top3": res["top3_correct"] / res["n"], "permutation_p": res.get("permutation_p"), **(extra or {})}


rows, shrows, preds_all = [], [], []
t0 = time.time()
for form in ["alap", "pakad"]:
    sub = M[M.form == form].reset_index(drop=True)
    recs = load(sub.id); lab = list(sub.raag); perf = list(sub.singer)
    prof = [D.duration_profile(r["runs"], 3) for r in recs]; bf = [D.baseline_features(r) for r in recs]
    mats = {"RagaSeq-D": C.dist_matrix(prof, R.js_distance_dict),
            "Pitch-class distribution": C.dist_matrix([b[0] for b in bf], D.jsd_vec),
            "TDMS": C.dist_matrix([b[1] for b in bf], D.sym_kl)}
    pr = {"id": list(sub.id), "form": form, "true": lab}
    for name, Mx in mats.items():
        res = D.recognise(Mx, lab, perf, 2000, 1234); pr[name] = res["pred"]
        sa, cr = C.same_cross(Mx, lab)
        rows.append(summ(name, res, {"form": form, "cohens_d": R.cohens_d(sa, cr)}))
        print(form, name, f"{res['correct']}/{res['n']} = {res['accuracy']:.1%}", flush=True)
    for k in [1, 2, 4]:
        Mx = C.dist_matrix([D.duration_profile(r["runs"], k) for r in recs], R.js_distance_dict)
        res = D.recognise(Mx, lab, perf, 0, 0); rows.append(summ(f"RagaSeq-D k = {k}", res, {"form": form}))
    preds_all.append(pd.DataFrame(pr))
    # McNemar
    true = np.array(lab)
    for b in ["TDMS", "Pitch-class distribution"]:
        A = np.array(pr["RagaSeq-D"]) == true; Bm = np.array(pr[b]) == true
        n01, n10 = int((A & ~Bm).sum()), int((~A & Bm).sum())
        rows.append({"method": f"McNemar RagaSeq-D vs {b}", "form": form, "correct": n01, "n": n10,
                     "permutation_p": binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0})
    # shuffles
    runs = [r["runs"] for r in recs]; rng = np.random.default_rng(1234)
    intact = S.evaluate(runs, lab, perf, R, D, 3, False)
    shrows += [{"form": form, "condition": "intact", "replicate": 0, "measure": m, "value": v} for m, v in intact.items()]
    for cond, fn in S.SHUFFLES.items():
        for rep in range(1, REPS + 1):
            res = S.evaluate([[fn(n, d, rng) for n, d in rec] for rec in runs], lab, perf, R, D, 3, False)
            shrows += [{"form": form, "condition": cond, "replicate": rep, "measure": m, "value": v} for m, v in res.items()]
        print(form, cond, "done", round(time.time() - t0), "s", flush=True)
pd.DataFrame(rows).to_csv(OUT + "62_dap_recognition.csv", index=False)
pd.concat(preds_all).to_csv(OUT + "63_dap_predictions.csv", index=False)
sh = pd.DataFrame(shrows); sh.to_csv(OUT + "64_dap_shuffle_replicates.csv", index=False)
for (form, m), g in sh.groupby(["form", "measure"]):
    iv = g[g.condition == "intact"].value.iloc[0]; hb = m != "RagaSeq-D separation (d)"
    s = f"{form} {m}: intact {iv:.3f}"
    for c in S.SHUFFLES:
        v = g[g.condition == c].value.to_numpy(); ge = (v >= iv).sum() if hb else (v <= iv).sum()
        s += f" | {c[:4]} {v.mean():.3f} [{np.percentile(v, 2.5):.3f},{np.percentile(v, 97.5):.3f}] p={(ge + 1) / (len(v) + 1):.4f}"
    print(s)

# C. external: DAP alaps of shared ragas vs CompMusic references
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
shared = sorted(set(M.raag) & set(sel.raag)); print("shared raags:", shared)
Q = M[(M.form == "alap") & M.raag.isin(shared)].reset_index(drop=True)
refs = C.load(sel.recording_id); qrec = load(Q.id); rl = list(sel.raag); ql = list(Q.raag); ragas = sorted(set(rl))


def ext(prq, prr, fn):
    Drr = C.dist_matrix(prr, fn); iu = np.triu_indices(len(prr), 1); rd = Drr[iu]
    out, t3 = [], []
    for q in prq:
        s = -(np.array([fn(q, r) for r in prr]) - rd.mean()) / rd.std(ddof=1)
        m = {g: s[np.array(rl) == g].mean() for g in ragas}; order = sorted(m, key=lambda g: -m[g])
        out.append(order[0]); t3.append(order[:3])
    return out, t3


ext_rows = []
pq = [D.duration_profile(r["runs"], 3) for r in qrec]; prr = [D.duration_profile(r["runs"], 3) for r in refs]
bfq = [D.baseline_features(r) for r in qrec]; bfr = [D.baseline_features(r) for r in refs]
for name, a, b, fn in [("RagaSeq-D", pq, prr, R.js_distance_dict), ("Pitch-class distribution", [x[0] for x in bfq], [x[0] for x in bfr], D.jsd_vec),
                       ("TDMS", [x[1] for x in bfq], [x[1] for x in bfr], D.sym_kl)]:
    p, t3 = ext(a, b, fn); k = sum(x == y for x, y in zip(p, ql)); n = len(ql)
    ext_rows.append({"method": name, "correct": k, "n": n, "accuracy": k / n, "top3": np.mean([y in t for y, t in zip(ql, t3)]),
                     "ci95_low": beta.ppf(.025, k, n - k + 1) if k else 0, "ci95_high": beta.ppf(.975, k + 1, n - k) if k < n else 1})
    Q[name] = p
    print("external", name, f"{k}/{n} = {k/n:.1%}")
pd.DataFrame(ext_rows).to_csv(OUT + "65_dap_external.csv", index=False); Q.to_csv(OUT + "66_dap_external_predictions.csv", index=False)
print("ALL DONE", round(time.time() - t0), "s")
