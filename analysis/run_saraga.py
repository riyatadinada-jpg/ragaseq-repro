"""External test on Saraga 1.5 Hindustani.

Queries: Saraga recordings whose single raga is one of the 30 CompMusic ragas, that are not in the
CompMusic Raga Recognition Dataset (by recording MBID), with a vocal lead and >= 150 s of pitch.
References: the 218-recording expanded CompMusic set, fixed. For each query, references by the same
lead performer are removed. Same front end, distances and decision rule as Script 3.
Shuffle control: every recording (queries and references) shuffled within windows, as before.
"""
import json, os, re, sys, time, unicodedata
import numpy as np, pandas as pd
from scipy.stats import beta, binomtest
import common as C, packdata as P
import RagaSeq_Shuffle_Control as S
R, D = C.R, C.D
SG = os.environ.get("SARAGA_DIR", os.environ.get("RAGASEQ_DATA", "../data") + "/")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = "results_saraga/"
import os; os.makedirs(OUT, exist_ok=True)


def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z]", "", s)
    return {"kumargandharv": "kumargandharva", "pt": ""}.get(s, s)


# ---------- load Saraga ----------
meta = json.load(open(SG + "saraga_metadata.json", encoding="utf-8"))
sg = {}
import gzip, struct
with gzip.open(SG + "saraga_central600.bin.gz", "rb") as f:
    while True:
        b = f.read(4)
        if not b: break
        h = json.loads(f.read(struct.unpack("<I", b)[0])); n = h["n"]
        hz = np.frombuffer(f.read(4 * n), dtype="<f4").astype(float)
        t0 = np.frombuffer(f.read(8), dtype="<i8")[0]
        dd = np.frombuffer(f.read(4 * (n - 1)), dtype="<i4").astype(np.int64)
        t = np.r_[t0, t0 + np.cumsum(dd)] / 1e7
        if t[0] > h["full_start"]: t = np.r_[h["full_start"], t]; hz = np.r_[0.0, hz]
        if t[-1] < h["full_end"]: t = np.r_[t, h["full_end"]]; hz = np.r_[hz, 0.0]
        sg[h["rel"]] = (t, hz, h["tonic"], h)

cm_all = set(json.load(open(P.INFO / "path_mbid_ragaid.json")))
names = json.load(open(P.INFO / "ragaId_to_ragaName_mapping.json"))
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
raag_ids = dict(zip(sel.raag_id, sel.raag))
rows, excl = [], []
for rel, md in meta.items():
    t, hz, ton, h = sg[rel]
    raags = md.get("raags", [])
    TITLE = {"Raag Bhoopali": "Bhoop", "Raag Kedar": "Kedar", "Raag Gaud Malhar": "Gaud Malhar"}
    if not raags and md.get("title") in TITLE:   # raga missing from metadata but named in the title
        rid = {v: k for k, v in raag_ids.items()}[TITLE[md["title"]]]
        raags = [{"uuid": rid, "common_name": TITLE[md["title"]] + " (from title)"}]
    leads = [a for a in md.get("artists", []) if a.get("lead")]
    lead = leads[0]["artist"]["name"] if leads else ""
    instr = leads[0]["instrument"]["name"] if leads else ""
    dur = h["full_end"] - h["full_start"]
    why = None
    if len({r["uuid"] for r in raags}) != 1: why = "not exactly one raag"
    elif raags[0]["uuid"] not in raag_ids: why = "raag not among the 30"
    elif md.get("mbid") in cm_all and not os.environ.get("ALLOW_OVERLAP"): why = "already in CompMusic dataset"
    elif instr.lower() != "voice": why = "lead not vocal"
    elif not ton or ton <= 0: why = "no tonic"
    elif dur < 150: why = "shorter than 150 s"
    rec = {"saraga_track": rel, "mbid": md.get("mbid"), "title": md.get("title"), "lead": lead, "lead_instrument": instr,
           "raag_uuid": raags[0]["uuid"] if raags else None, "raag_saraga": ", ".join(r.get("common_name", "") for r in raags),
           "raag": raag_ids.get(raags[0]["uuid"]) if len(raags) == 1 else None, "duration_min": round(dur / 60, 1), "excluded": why or ""}
    (excl if why else rows).append(rec)
Q = pd.DataFrame(rows); X = pd.DataFrame(excl)
pd.concat([Q, X]).to_csv(OUT + "51_saraga_tracks.csv", index=False)
print(f"Saraga tracks {len(meta)}; usable queries {len(Q)}; excluded:", X.excluded.value_counts().to_dict() if len(X) else {})
print(Q.raag.value_counts().to_dict() if len(Q) else {})
if len(Q) == 0: sys.exit()

# register queries with the loaders
for r in Q.itertuples():
    t, hz, ton, h = sg[r.saraga_track]
    C.PK["SG:" + r.mbid] = (t, hz, ton, h)
    C.MD["SG:" + r.mbid] = {"raag": r.raag, "performer": r.lead}
qids = ["SG:" + m for m in Q.mbid]
refs = C.load(sel.recording_id); qrec = C.load(qids)
ref_lab = list(sel.raag); ref_perf = [norm(p) for p in sel.performer]
q_lab = list(Q.raag); q_perf = [norm(p) for p in Q.lead]
ragas = sorted(set(ref_lab))


def classify(Dqr, Drr):
    """Dqr: queries x refs distances; Drr: refs x refs. Same rule as Script 3, refs fixed."""
    preds, top3 = [], []
    for i in range(len(q_lab)):
        ok = np.array([q_perf[i] != p for p in ref_perf])
        rr = np.where(ok)[0]
        iu = np.triu_indices(len(rr), 1); rd = Drr[np.ix_(rr, rr)][iu]
        s = -(Dqr[i, rr] - rd.mean()) / rd.std(ddof=1)
        lab = np.array(ref_lab)[rr]
        m = {g: s[lab == g].mean() for g in ragas if (lab == g).any()}
        order = sorted(m, key=lambda g: -m[g])
        preds.append(order[0]); top3.append(q_lab[i] in order[:3])
    return preds, np.array(top3)


def summary(preds, t3):
    k = int(sum(p == t for p, t in zip(preds, q_lab))); n = len(q_lab)
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return {"correct": k, "n": n, "accuracy": k / n, "ci95_low": lo, "ci95_high": hi, "top3": float(t3.mean())}


def dmat(A, B, fn):
    return np.array([[fn(a, b) for b in B] for a in A])


def evaluate(ref_runs, q_runs, k=3):
    pr = [D.duration_profile(r, k) for r in ref_runs]; pq = [D.duration_profile(r, k) for r in q_runs]
    return classify(dmat(pq, pr, R.js_distance_dict), C.dist_matrix(pr, R.js_distance_dict))


t0 = time.time()
res, preds = [], {"saraga_track": list(Q.saraga_track), "lead": list(Q.lead), "true_raag": q_lab}
pr_, t3 = evaluate([r["runs"] for r in refs], [r["runs"] for r in qrec])
res.append({"method": "RagaSeq-D", **summary(pr_, t3)}); preds["RagaSeq-D"] = pr_
bfr = [D.baseline_features(r) for r in refs]; bfq = [D.baseline_features(r) for r in qrec]
for name, j, fn in [("Pitch-class distribution (Chordia & Rae 2007)", 0, D.jsd_vec), ("TDMS (Gulati et al. 2016b)", 1, D.sym_kl)]:
    p, t3_ = classify(dmat([b[j] for b in bfq], [b[j] for b in bfr], fn), C.dist_matrix([b[j] for b in bfr], fn))
    res.append({"method": name, **summary(p, t3_)}); preds[name] = p
for k in [1, 2, 4]:
    p, t3_ = evaluate([r["runs"] for r in refs], [r["runs"] for r in qrec], k)
    res.append({"method": f"RagaSeq-D k = {k}", **summary(p, t3_)})
# chance permutation p for RagaSeq-D (shuffle query labels)
rng = np.random.default_rng(1234); kobs = res[0]["correct"]
ge = sum(sum(p == t for p, t in zip(pr_, rng.permutation(q_lab))) >= kobs for _ in range(10000))
res[0]["permutation_p"] = (ge + 1) / 10001
pd.DataFrame(res).to_csv(OUT + "52_saraga_recognition.csv", index=False)
pd.DataFrame(preds).to_csv(OUT + "53_saraga_predictions.csv", index=False)
true = np.array(q_lab); mcrows = []
for a, b in [("RagaSeq-D", "TDMS (Gulati et al. 2016b)"), ("RagaSeq-D", "Pitch-class distribution (Chordia & Rae 2007)")]:
    A = np.array(preds[a]) == true; Bm = np.array(preds[b]) == true
    n01, n10 = int((A & ~Bm).sum()), int((~A & Bm).sum())
    mcrows.append({"a": a, "b": b, "a_right_b_wrong": n01, "a_wrong_b_right": n10, "p": binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0})
pd.DataFrame(mcrows).to_csv(OUT + "54_saraga_mcnemar.csv", index=False)
print(pd.DataFrame(res).round(3).to_string()); print(pd.DataFrame(mcrows).round(4).to_string())

# shuffle control: shuffle every recording (queries and references)
rows = [{"condition": "intact", "replicate": 0, "accuracy": res[0]["accuracy"]}]
rng = np.random.default_rng(1234)
rr = [r["runs"] for r in refs]; qr = [r["runs"] for r in qrec]
for cond, fn in S.SHUFFLES.items():
    for rep in range(1, REPS + 1):
        sr_ = [[fn(n, d, rng) for n, d in rec] for rec in rr]; sq = [[fn(n, d, rng) for n, d in rec] for rec in qr]
        p, _ = evaluate(sr_, sq)
        rows.append({"condition": cond, "replicate": rep, "accuracy": float(np.mean([a == b for a, b in zip(p, q_lab)]))})
    print(cond, "done", round(time.time() - t0), "s", flush=True)
sh = pd.DataFrame(rows); sh.to_csv(OUT + "55_saraga_shuffle.csv", index=False)
iv = res[0]["accuracy"]
for c in (S.SHUFFLES if REPS else []):
    v = sh[sh.condition == c].accuracy.to_numpy()
    print(c, f"mean {v.mean():.3f} [{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}] best {v.max():.3f} p={((v>=iv).sum()+1)/(len(v)+1):.4f}")
