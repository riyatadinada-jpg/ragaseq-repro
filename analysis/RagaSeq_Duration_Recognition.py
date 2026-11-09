#!/usr/bin/env python3
"""RagaSeq-D: duration-aware RagaSeq and performer-aware held-out raga recognition.

What changes compared with RagaSeq_10Raga_Analysis_FINAL_Annotated.py
---------------------------------------------------------------------
The front end is identical (central 150 s, five 30-s windows, 10 Hz median bins,
12 tonic-relative pitch states, 3-frame median smoothing). Two things change:

  1. When repeated states are merged, every note keeps how long it was held (seconds).
  2. Each recording gets ONE 3-note pattern profile, pooled over all five windows,
     where every pattern is weighted by the total time its three notes are held.
     Recordings are compared with Jensen-Shannon distance between these profiles.

Recognition: performer-aware leave-one-recording-out. Each recording is held out;
every other recording by the same performer is removed from the reference set;
distances are standardized with reference-reference pairs only; the predicted raga
is the one whose reference recordings are, on average, most similar.

Optional --baselines adds two published comparison methods on the same audio interval:
pitch-class distribution (Chordia & Rae, 2007) and time-delayed melody surfaces
(Gulati et al., 2016b; published default settings).

Place this file in the SAME folder as RagaSeq_10Raga_Analysis_FINAL_Annotated.py.

Example
-------
python RagaSeq_Duration_Recognition.py FEATURE_DIRECTORY \
  --selection RagaSeq_10x5_Selection_with_performers.csv --out ragaseq_duration_10x5 --baselines

The selection CSV needs a recording_id column. Add a performer column to enable the
performer-aware exclusion (strongly recommended).

Outputs (in --out)
------------------
21_duration_pairwise.csv       one row per recording pair: RagaSeq-D distance (and baselines)
22_duration_same_vs_cross.csv  same-raga vs cross-raga contrast for RagaSeq-D (Cohen's d, permutation p)
23_recognition_summary.csv     held-out accuracy, 95% CI, top-3, permutation p for each method
24_recognition_predictions.csv one row per held-out recording
25_confusion_ragaseq_d.csv     true raga x predicted raga
26_characteristic_patterns.csv the five most characteristic 3-note patterns of each raga (Appendix A)
run_summary_duration.json      settings and software versions
"""
from __future__ import annotations
import argparse, json, platform, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

import RagaSeq_10Raga_Analysis_FINAL_Annotated as R

# ===== STEP 1: Duration-aware encoding (Section 2.6) =====
# Same front end as the main script, but when repeated states are merged each note keeps
# how long it was held. Each recording then gets one 3-note profile pooled over its five windows,
# with every pattern weighted by the time its notes are held (RagaSeq-D).

# Merge repeated states but keep each note's held duration in seconds.
def runs_with_duration(states, step):
    """Merge repeated states, keeping how long each note was held (seconds)."""
    states = np.asarray(states, int)
    if len(states) == 0:
        return states, np.array([])
    starts = np.r_[0, np.where(states[1:] != states[:-1])[0] + 1]
    lengths = np.diff(np.r_[starts, len(states)])
    return states[starts], lengths * step


# One window: 10 Hz bins -> 12 states -> smoothing -> notes with durations.
def window_runs(t, hz, tonic, w0, w1, sample_hz=10.0, smooth_width=3):
    """Same steps as R.window_sequence, but returns (notes, held durations)."""
    mask = (t >= w0) & (t < w1) & (hz > 0)
    if mask.sum() < 5:
        return np.array([], int), np.array([])
    tt, ff = t[mask], hz[mask]
    step = 1.0 / sample_hz
    edges = np.arange(w0, w1 + step, step)
    vals = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (tt >= a) & (tt < b)
        vals.append(float(np.median(ff[m])) if m.any() else np.nan)
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.array([], int), np.array([])
    states = R.median_smooth_discrete(R.quantize_12(R.cents_relative(vals, tonic)), smooth_width)
    return runs_with_duration(states, step)


# One pooled 3-note profile per recording, weighted by held duration.
# (Equation 6 in the paper.)
def duration_profile(windows, k=3):
    """One profile per recording: k-note patterns pooled over windows, weighted by held time."""
    c = defaultdict(float)
    for notes, durs in windows:
        for i in range(len(notes) - k + 1):
            c[tuple(notes[i:i + k])] += float(durs[i:i + k].sum())
    total = sum(c.values())
    return {key: v / total for key, v in c.items()} if total else {}


# Rank the 3-note patterns that are most characteristic of each raga (Appendix A).
SARGAM = "S r R g G m M P d D n N".split()   # 12 states from Sa; lowercase = komal, M = tivra Ma


def characteristic_patterns(prof, labels, top=5, floor=1e-4):
    """For each raga: patterns present in ALL of its recordings, ranked by
    log(mean share in this raga / mean share in the other ragas)."""
    rows = []
    for raga in sorted(set(labels)):
        ins = [p for p, l in zip(prof, labels) if l == raga]
        oth = [p for p, l in zip(prof, labels) if l != raga]
        cand = []
        for key in set().union(*ins):
            a = np.array([p.get(key, 0.0) for p in ins]); b = np.array([p.get(key, 0.0) for p in oth])
            if (a > 0).sum() < len(ins):
                continue
            lr = float(np.log((a.mean() + floor) / (b.mean() + floor)))
            cand.append((lr, a.mean(), b.mean(), " ".join(SARGAM[s % 12] for s in key)))
        cand.sort(reverse=True)
        for rank, (lr, a_m, b_m, pat) in enumerate(cand[:top], 1):
            rows.append({"raga": raga, "rank": rank, "pattern": pat, "share_in_raga": a_m,
                         "share_in_other_ragas": b_m, "log_ratio": lr, "recordings_with_pattern": len(ins)})
    return pd.DataFrame(rows)


# Read one recording and encode its five windows.
def load_recording(row, args):
    t, hz = R.load_pitch(row["pitch"])
    tonic, tonic_source = R.choose_tonic(row)
    wins = R.fixed_windows(t, args.central_seconds, args.windows)
    runs = [window_runs(t, hz, tonic, a, b, args.sample_hz) for a, b in wins]
    return {"raga": row["raga"], "recording_id": row["recording_id"], "tonic_source": tonic_source,
            "t": t, "hz": hz, "tonic": tonic, "wins": wins, "runs": runs}

# ===== STEP 2: Comparison methods (optional, --baselines) =====
# Pitch-class distribution (Chordia & Rae, 2007) and time-delayed melody surfaces
# (Gulati et al., 2016b), computed on the same audio interval as RagaSeq-D.

# 12-bin pitch-class distribution and 120-bin time-delayed melody surface for one recording.
def baseline_features(rec, eta=120, tau=0.3, alpha=0.75, sigma=2):
    from scipy.ndimage import convolve
    t, hz, tonic = rec["t"], rec["hz"], rec["tonic"]
    s, e = rec["wins"][0][0], rec["wins"][-1][1]
    m = (t >= s) & (t < e)
    t, hz = t[m], hz[m]
    v = hz > 0
    cents = np.full(hz.shape, np.nan)
    cents[v] = 1200 * np.log2(hz[v] / tonic)
    pcd = np.bincount((np.round(cents[v] / 100).astype(int)) % 12, minlength=12).astype(float)
    pcd /= pcd.sum()
    lag = int(round(tau / float(np.median(np.diff(t)))))
    b = np.full(hz.shape, -1)
    b[v] = np.floor(eta * cents[v] / 1200).astype(int) % eta
    a1, a0 = b[lag:], b[:-lag]
    ok = (a1 >= 0) & (a0 >= 0)
    surf = np.zeros((eta, eta))
    np.add.at(surf, (a1[ok], a0[ok]), 1)
    g = np.arange(-4 * sigma, 4 * sigma + 1)
    g1 = np.exp(-g ** 2 / (2 * sigma ** 2))
    kern = np.outer(g1, g1) / np.outer(g1, g1).sum()
    surf = convolve(surf ** alpha, kern, mode="wrap")
    return pcd, surf / surf.sum()


# Jensen-Shannon distance between two distributions.
def jsd_vec(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / b), 0))
    return float(np.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2)))


# Symmetric Kullback-Leibler divergence (used for melody surfaces).
def sym_kl(p, q, eps=1e-12):
    p = p.ravel() + eps; q = q.ravel() + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))

# ===== STEP 3: Held-out recognition (Section 3.4) =====
# Each recording is held out with its label hidden; recordings by the same performer are removed
# from the references; distances are standardized with reference-reference pairs only; the predicted
# raga is the one whose references are most similar on average. Accuracy, 95% CI, top-3 and permutation p.

# Leave-one-recording-out recognition with performer exclusion.
def recognise(D, labels, performers, n_perm, seed):
    """D: distance matrix (lower = more similar). Returns predictions, accuracy, top-3, permutation p."""
    n = len(labels)
    ragas = sorted(set(labels))
    lab = np.array([ragas.index(x) for x in labels])
    ref = np.array([[j != t and (performers[j] != performers[t] or performers[t] == "") for j in range(n)] for t in range(n)])
    S = np.full((n, n), np.nan)
    for t in range(n):
        rr = np.where(ref[t])[0]
        iu = np.triu_indices(len(rr), 1)
        rd = D[np.ix_(rr, rr)][iu]
        S[t, rr] = -(D[t, rr] - rd.mean()) / rd.std(ddof=1)
    def raga_means(L):
        O = np.eye(len(ragas))[L]
        sums = np.where(ref, S, 0) @ O
        cnt = ref.astype(float) @ O
        m = sums / np.where(cnt == 0, np.nan, cnt)
        return np.where(np.isnan(m), -np.inf, m), cnt
    m, cnt = raga_means(lab)
    order = np.argsort(-m, 1)
    pred = order[:, 0]
    k = int((pred == lab).sum())
    top3 = int(sum(lab[t] in order[t, :3] for t in range(n)))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        L = rng.permutation(lab)
        ge += (raga_means(L)[0].argmax(1) == L).mean() >= k / n
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return {"pred": [ragas[p] for p in pred], "correct": k, "n": n, "accuracy": k / n, "ci95_low": lo, "ci95_high": hi,
            "top3_correct": top3, "permutation_p": (ge + 1) / (n_perm + 1), "min_refs_own_raga": int(min(cnt[t, lab[t]] for t in range(n)))}

# ===== STEP 4: Run and write the outputs =====
# Loads the recordings, computes all distances, runs recognition for each method and writes files 21-25.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=".")
    ap.add_argument("--out", default="ragaseq_duration")
    ap.add_argument("--selection", required=True, help="CSV with recording_id (and ideally performer) columns")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--central-seconds", type=float, default=150.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baselines", action="store_true", help="Also run pitch-class distribution and TDMS")
    args = ap.parse_args()

    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve()
    manifest = R.build_manifest(data_dir, args.selection)
    ready = manifest[manifest.analysis_ready & manifest.selected].reset_index(drop=True)
    sel = pd.read_csv(args.selection, dtype=str)
    perf_map = dict(zip(sel.recording_id, sel.performer)) if "performer" in sel.columns else {}
    if not perf_map:
        print("WARNING: no performer column; the held-out test will not exclude same-performer recordings.")
    recs = [load_recording(row, args) for _, row in ready.iterrows()]
    n = len(recs)
    labels = [r["raga"] for r in recs]
    performers = [str(perf_map.get(r["recording_id"], "")) for r in recs]
    print(f"Loaded {n} recordings, {len(set(labels))} ragas")

    prof = [duration_profile(r["runs"], args.k) for r in recs]
    methods = {"RagaSeq-D": lambda i, j: R.js_distance_dict(prof[i], prof[j])}
    if args.baselines:
        bf = [baseline_features(r) for r in recs]
        methods["Pitch-class distribution (Chordia & Rae 2007)"] = lambda i, j: jsd_vec(bf[i][0], bf[j][0])
        methods["TDMS (Gulati et al. 2016b)"] = lambda i, j: sym_kl(bf[i][1], bf[j][1])
    mats = {name: np.full((n, n), np.nan) for name in methods}
    rows = []
    for i, j in combinations(range(n), 2):
        row = {"recording_a": recs[i]["recording_id"], "raga_a": labels[i], "recording_b": recs[j]["recording_id"],
               "raga_b": labels[j], "same_raga": labels[i] == labels[j]}
        for name, fn in methods.items():
            v = fn(i, j); mats[name][i, j] = mats[name][j, i] = v
            row[name] = v
        rows.append(row)
    pairdf = pd.DataFrame(rows)
    pairdf.to_csv(out / "21_duration_pairwise.csv", index=False)

    same = pairdf.loc[pairdf.same_raga, "RagaSeq-D"].to_numpy(); cross = pairdf.loc[~pairdf.same_raga, "RagaSeq-D"].to_numpy()
    obs, perm_p = R.permutation_pvalue(pairdf, "RagaSeq-D", n_perm=args.permutations, seed=args.seed)
    pd.DataFrame([{"metric": "RagaSeq-D distance (lower = more similar)", "same_n": len(same), "cross_n": len(cross),
                   "same_mean": same.mean(), "cross_mean": cross.mean(), "difference": same.mean() - cross.mean(),
                   "cohens_d": R.cohens_d(same, cross), "label_permutation_p": perm_p}]).to_csv(out / "22_duration_same_vs_cross.csv", index=False)

    summ, preds = [], {"recording_id": [r["recording_id"] for r in recs], "performer": performers, "true_raga": labels}
    for name, D in mats.items():
        res = recognise(D, labels, performers, args.permutations, args.seed)
        preds[name] = res.pop("pred")
        summ.append({"method": name, **res})
        print(f"  {name}: {res['correct']}/{n} = {res['accuracy']:.1%} (95% CI {res['ci95_low']:.1%}-{res['ci95_high']:.1%})")
    pd.DataFrame(summ).to_csv(out / "23_recognition_summary.csv", index=False)
    pdf = pd.DataFrame(preds); pdf.to_csv(out / "24_recognition_predictions.csv", index=False)
    pd.crosstab(pdf.true_raga, pdf["RagaSeq-D"]).to_csv(out / "25_confusion_ragaseq_d.csv")
    characteristic_patterns(prof, labels).to_csv(out / "26_characteristic_patterns.csv", index=False)
    (out / "run_summary_duration.json").write_text(json.dumps({
        "n_recordings": n, "ragas": sorted(set(labels)), "k": args.k, "windows": args.windows,
        "central_seconds": args.central_seconds, "sample_hz": args.sample_hz, "permutations": args.permutations,
        "seed": args.seed, "performer_aware": bool(perf_map), "baselines": args.baselines,
        "software": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}}, indent=2))


if __name__ == "__main__":
    main()
