"""Apply the same order test to the established methods.
Each window's 10-Hz median pitch contour (fine pitch, in Hz) is cut into note segments (runs of one 12-state bin).
Full shuffle: segments permuted (each keeps its own fine pitch contour, so intonation and within-note glides survive).
Dyad shuffle: note order from the Altschul-Erickson shuffle; each position gets a random unused segment of that note.
Every frame is kept, so the pitch-class distribution is unchanged by construction.
Methods on the same contours: RagaSeq-D k=1, k=3; pitch-class distribution (PCD); fine pitch distribution (FPD: TDMS
with no delay, 120 bins, same compression and smoothing = the content-only counterpart of TDMS); TDMS (0.3 s delay)."""
import sys, json, time, numpy as np, pandas as pd
from scipy.ndimage import convolve1d, convolve
import common as C, run_dap_meta as DM
import RagaSeq_Shuffle_Control as S
R, D = C.R, C.D
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = "results_expanded/"

def segments(rec, sample_hz=10.0):
    """per window: list of (state, fine-Hz array)"""
    t, hz, tonic = rec["t"], rec["hz"], rec["tonic"]; out = []
    for w0, w1 in rec["wins"]:
        m = (t >= w0) & (t < w1) & (hz > 0)
        if m.sum() < 5: out.append([]); continue
        tt, ff = t[m], hz[m]; step = 1 / sample_hz; edges = np.arange(w0, w1 + step, step); vals = []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (tt >= a) & (tt < b); vals.append(float(np.median(ff[k])) if k.any() else np.nan)
        vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
        if len(vals) < 5: out.append([]); continue
        st = R.median_smooth_discrete(R.quantize_12(R.cents_relative(vals, tonic)), 3)
        starts = np.r_[0, np.where(st[1:] != st[:-1])[0] + 1]; ends = np.r_[starts[1:], len(st)]
        out.append([(int(st[a]), vals[a:b]) for a, b in zip(starts, ends)])
    return out

def shuffle_segs(segs, kind, rng):
    if len(segs) < 3: return segs
    if kind == "full": return [segs[i] for i in rng.permutation(len(segs))]
    notes = np.array([s for s, _ in segs]); new = S.euler_dyad_shuffle(notes, rng)
    pools = {}
    for i in rng.permutation(len(segs)): pools.setdefault(segs[i][0], []).append(segs[i])
    return [pools[int(v)].pop() for v in new]

def features(wsegs, tonic, eta=120, tau_frames=3, alpha=0.75, sigma=2):
    runs, cents_all, b_pairs = [], [], []
    for segs in wsegs:
        if not segs: runs.append((np.array([], int), np.array([]))); continue
        notes = np.array([s for s, _ in segs]); durs = np.array([len(v) for _, v in segs]) * 0.1
        n2, d2 = S.merge_adjacent(notes, durs); runs.append((n2, d2))
        c = 1200 * np.log2(np.concatenate([v for _, v in segs]) / tonic); cents_all.append(c)
        b = np.floor(eta * c / 1200).astype(int) % eta; b_pairs.append((b[tau_frames:], b[:-tau_frames]))
    c = np.concatenate(cents_all)
    pcd = np.bincount(np.round(c / 100).astype(int) % 12, minlength=12).astype(float); pcd /= pcd.sum()
    g = np.arange(-4 * sigma, 4 * sigma + 1); g1 = np.exp(-g ** 2 / (2 * sigma ** 2)); g1 /= g1.sum()
    b = np.floor(eta * c / 1200).astype(int) % eta
    fpd = convolve1d(np.bincount(b, minlength=eta).astype(float) ** alpha, g1, mode="wrap"); fpd /= fpd.sum()
    surf = np.zeros((eta, eta))
    for a1, a0 in b_pairs: np.add.at(surf, (a1, a0), 1)
    surf = convolve(surf ** alpha, np.outer(g1, g1), mode="wrap"); surf /= surf.sum()
    return {"RagaSeq-D k=1": D.duration_profile(runs, 1), "RagaSeq-D k=3": D.duration_profile(runs, 3), "PCD": pcd, "FPD": fpd, "TDMS": surf}

FN = {"RagaSeq-D k=1": R.js_distance_dict, "RagaSeq-D k=3": R.js_distance_dict, "PCD": D.jsd_vec, "FPD": D.sym_kl, "TDMS": D.sym_kl}
def accs(feats, lab, perf):
    out = {}
    for m, fn in FN.items():
        M = C.dist_matrix([f[m] for f in feats], fn); out[m] = D.recognise(M, lab, perf, 0, 0)["accuracy"]
    return out

sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
ALLSETS = [("CompMusic expanded, 150 s", list(sel.recording_id), 150, 5),
        ("CompMusic long, 600 s", list(sel[sel.duration_s >= 610].recording_id), 600, 20),
        ("DAP-Lab alap", list(DM.M[DM.M.form == "alap"].id), 150, 5)]
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
SETS = [x for x in ALLSETS if ONLY is None or x[0].startswith(ONLY)]
OUTF = OUT + (f"70_order_test_{ONLY.split()[0]}.csv" if ONLY else "70_order_test_all_methods.csv")
rows = []; t0 = time.time()
for name, ids, secs, win in SETS:
    recs = C.load(ids, central_seconds=secs, windows=win); lab = [r["raga"] for r in recs]; perf = [r["performer"] for r in recs]
    segs = [segments(r) for r in recs]; ton = [r["tonic"] for r in recs]
    a = accs([features(s, tt) for s, tt in zip(segs, ton)], lab, perf)
    rows += [{"set": name, "condition": "intact", "replicate": 0, "method": m, "accuracy": v} for m, v in a.items()]
    print(name, "intact", {k: round(v, 3) for k, v in a.items()}, round(time.time() - t0), flush=True)
    rng = np.random.default_rng(1234)
    for kind in ["full", "dyad"]:
        for rep in range(1, REPS + 1):
            f = [features([shuffle_segs(w, kind, rng) for w in s], tt) for s, tt in zip(segs, ton)]
            a = accs(f, lab, perf)
            rows += [{"set": name, "condition": kind, "replicate": rep, "method": m, "accuracy": v} for m, v in a.items()]
        g = pd.DataFrame(rows); g = g[(g.set == name) & (g.condition == kind)].groupby("method").accuracy.mean()
        print(name, kind, g.round(3).to_dict(), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(OUTF, index=False)
print("ALL DONE")
