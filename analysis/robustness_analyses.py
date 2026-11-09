"""Script 15. Robustness of the content-order comparison (Section 3.6, Appendix B).
Requires statsmodels (GEE) in addition to NumPy, pandas and SciPy; reads the packed data through common.py (Script 7).
A  formal length x representation model (GEE logistic, clustered by performer) + performer cluster bootstrap of the
   difference-in-differences  DiD = (k3 - k1 at 600 s) - (k3 - k1 at 150 s)
B  section control: is the 600-s order gain about MORE audio or about DIFFERENT sections?
   - four consecutive 150-s blocks of the central 600 s (each 5 x 30 s)
   - 'spread 150 s': 150 s of audio made of twenty 7.5-s windows spread evenly across the central 600 s
     (same sections as 600 s, same amount of audio as 150 s)
C  front-end robustness: tonic error, smoothing, frame rate, gap handling, short-note removal, pitch alphabet.
   For each variant: k1 and k3 accuracy on the 218-recording set at 150 s and on the 194 long recordings at 600 s.
usage: python robustness_analyses.py A   (length x representation model)  -> 81, 82, robust_A.json
       python robustness_analyses.py B   (section control)                 -> 83, robust_B.json
       python robustness_analyses.py C [i,j,...]  (front-end variants)     -> 84_frontend_XX.csv
"""
import sys, json, time, numpy as np, pandas as pd
from scipy.stats import binomtest
import common as C
R, D = C.R, C.D
OUT = "results_expanded/"
sel = pd.read_csv("RagaSeq_Expanded_Selection.csv")
ALL = list(sel.recording_id); LONG = list(sel[sel.duration_s >= 610].recording_id)
MD = C.MD


def q12(c): return R.quantize_12(c)
JI = np.array([0, 112, 204, 316, 386, 498, 590, 702, 814, 884, 996, 1088], float)
def qji(c):
    f = np.mod(c, 1200.0)[:, None]; dd = np.abs(f - JI[None, :]); dd = np.minimum(dd, 1200 - dd); return dd.argmin(1)
def q24(c): return np.mod(np.rint(np.mod(c, 1200.0) / 50.0).astype(int), 24)


def runs_of(states, step):
    return D.runs_with_duration(states, step)


def win_runs(t, hz, tonic, w0, w1, sample_hz=10.0, smooth=3, quant=q12, gap_split=None, min_note=0.0):
    """Generalised front end. Returns a list of (notes, durations) segments for one window."""
    m = (t >= w0) & (t < w1) & (hz > 0)
    if m.sum() < 5: return []
    tt, ff = t[m], hz[m]; step = 1.0 / sample_hz
    edges = np.arange(w0, w1 + step, step); vals = []
    idx = np.searchsorted(tt, edges)
    for a, b in zip(idx[:-1], idx[1:]):
        vals.append(float(np.median(ff[a:b])) if b > a else np.nan)
    vals = np.asarray(vals)
    ok = np.isfinite(vals)
    if ok.sum() < 5: return []
    if gap_split is None:
        pieces = [vals[ok]]
    else:  # split wherever the unvoiced gap is longer than gap_split seconds
        pieces, cur, gap = [], [], 0
        for v in vals:
            if np.isfinite(v):
                if gap * step > gap_split and cur: pieces.append(np.array(cur)); cur = []
                cur.append(v); gap = 0
            else: gap += 1
        if cur: pieces.append(np.array(cur))
    out = []
    for p in pieces:
        if len(p) < 2: continue
        st = R.median_smooth_discrete(quant(R.cents_relative(p, tonic)), smooth)
        n, d = runs_of(st, step)
        if min_note > 0 and len(n):
            keep = d >= min_note
            if keep.sum() == 0: continue
            n, d = runs_of(np.repeat(n[keep], np.maximum(1, np.round(d[keep] / step).astype(int))), step)
        out.append((n, d))
    return out


def central(t, secs):
    return R.fixed_windows(t, secs, 1)[0]


def encode(rids, windows_fn, tonic_err=None, **fe):
    """windows_fn(t) -> list of (w0, w1); returns list of profiles' raw runs (list of segments)"""
    recs = []
    for i, rid in enumerate(rids):
        t, hz = R.load_pitch(rid); tonic = float(C.PK[rid][2])
        if tonic_err is not None: tonic *= 2 ** (tonic_err[i] / 1200)
        segs = []
        for w0, w1 in windows_fn(t): segs += win_runs(t, hz, tonic, w0, w1, **fe)
        recs.append(segs)
    return recs


def correct(segs, rids, k):
    lab = [MD[r]["raag"] for r in rids]; perf = [MD[r]["performer"] for r in rids]
    M = C.dist_matrix([D.duration_profile(s, k) for s in segs], R.js_distance_dict)
    return np.array(D.recognise(M, lab, perf, 0, 0)["pred"]) == np.array(lab)


def std_windows(secs, n):
    return lambda t: R.fixed_windows(t, secs, n)


def block_windows(b):
    def f(t):
        s, e = central(t, 600.0); a = s + 150 * b
        return [(a + 30 * i, a + 30 * (i + 1)) for i in range(5)]
    return f


def spread_windows(t):
    s, e = central(t, 600.0)
    return [(s + 30 * i + 11.25, s + 30 * i + 18.75) for i in range(20)]  # 7.5 s centred in every 30 s


def mcn(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0)


# ------------------------------------------------------------------------------------------------
part = sys.argv[1]
t0 = time.time()
if part == "A":
    rows = []
    for secs, nw in [(150, 5), (300, 10), (600, 20)]:
        segs = encode(LONG, std_windows(secs, nw))
        for k in (1, 3):
            c = correct(segs, LONG, k)
            rows += [{"recording_id": r, "performer": MD[r]["performer"], "raga": MD[r]["raag"], "seconds": secs,
                      "k": k, "correct": int(x)} for r, x in zip(LONG, c)]
        print(secs, round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT + "81_length_by_k_long_records.csv", index=False)
    import statsmodels.api as sm, statsmodels.formula.api as smf
    df["order"] = (df.k == 3).astype(int); df["L300"] = (df.seconds == 300).astype(int); df["L600"] = (df.seconds == 600).astype(int)
    df["log2len"] = np.log2(df.seconds / 150)
    res = {}
    for name, f in [("categorical", "correct ~ (L300 + L600) * order"), ("log2 length", "correct ~ log2len * order")]:
        g = smf.gee(f, groups="performer", data=df, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        tab = pd.DataFrame({"coef": g.params, "se": g.bse, "z": g.tvalues, "p": g.pvalues,
                            "OR": np.exp(g.params), "OR_lo": np.exp(g.conf_int()[0]), "OR_hi": np.exp(g.conf_int()[1])})
        tab.index.name = "term"; tab.to_csv(OUT + f"82_gee_{name.replace(' ', '_')}.csv")
        res[name] = tab.reset_index().to_dict("records"); print(name); print(tab.round(4).to_string(), flush=True)
    # cluster bootstrap by performer of the difference in differences
    w = df.pivot_table(index=["recording_id", "performer"], columns=["seconds", "k"], values="correct").reset_index()
    def did(x):
        g = lambda s: x[(s, 3)].mean() - x[(s, 1)].mean()
        return g(600) - g(150), g(300) - g(150), g(150), g(300), g(600)
    est = did(w); perfs = w.performer.unique(); rng = np.random.default_rng(1234); boots = []
    byp = {p: w[w.performer == p] for p in perfs}
    for _ in range(5000):
        boots.append(did(pd.concat([byp[p] for p in rng.choice(perfs, len(perfs))])))
    B = np.array(boots)
    out = {"did_600_vs_150": est[0], "did_ci": list(np.percentile(B[:, 0], [2.5, 97.5])), "did_p_le0": float((B[:, 0] <= 0).mean()),
           "did_300_vs_150": est[1], "did300_ci": list(np.percentile(B[:, 1], [2.5, 97.5])),
           "gain150": est[2], "gain300": est[3], "gain600": est[4],
           "gain_ci": {s: list(np.percentile(B[:, j], [2.5, 97.5])) for s, j in [(150, 2), (300, 3), (600, 4)]},
           "n_recordings": len(w), "n_performers": len(perfs), "gee": res}
    json.dump(out, open("robust_A.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in out.items() if k != "gee"}, default=float, indent=1))

elif part == "B":
    rows = []; per = {}
    conds = [("central 150 s", std_windows(150, 5)), ("central 600 s", std_windows(600, 20))] + \
            [(f"block {b+1} of 4 (150 s)", block_windows(b)) for b in range(4)] + [("spread 150 s (20 x 7.5 s)", spread_windows)]
    for name, wf in conds:
        segs = encode(LONG, wf); c1, c3 = correct(segs, LONG, 1), correct(segs, LONG, 3)
        g, l, p = mcn(c3, c1); per[name] = (c1, c3)
        rows.append({"condition": name, "audio_s": 600 if "600" in name and "spread" not in name else 150,
                     "k1": c1.mean(), "k3": c3.mean(), "k3_minus_k1": c3.mean() - c1.mean(), "k3_gain": g, "k3_loss": l, "mcnemar_p": p})
        print(rows[-1], round(time.time() - t0), flush=True)
    # mean over the four blocks
    b1 = np.mean([per[f"block {b+1} of 4 (150 s)"][0].mean() for b in range(4)]); b3 = np.mean([per[f"block {b+1} of 4 (150 s)"][1].mean() for b in range(4)])
    rows.append({"condition": "mean of four 150-s blocks", "audio_s": 150, "k1": b1, "k3": b3, "k3_minus_k1": b3 - b1})
    pd.DataFrame(rows).to_csv(OUT + "83_section_control.csv", index=False); json.dump(rows, open("robust_B.json", "w"), indent=1, default=float)

elif part == "C":
    rng = np.random.default_rng(1234)
    V = [("baseline (paper)", {}, None),
         ("tonic +10 cents", {}, 10.0), ("tonic -10 cents", {}, -10.0), ("tonic +25 cents", {}, 25.0), ("tonic -25 cents", {}, -25.0),
         ("tonic random error, SD 15 cents", {}, "r15"), ("tonic random error, SD 30 cents", {}, "r30"),
         ("no smoothing", {"smooth": 1}, None), ("5-frame smoothing", {"smooth": 5}, None),
         ("5 Hz frames", {"sample_hz": 5.0}, None), ("20 Hz frames", {"sample_hz": 20.0}, None),
         ("split at gaps > 0.5 s", {"gap_split": 0.5}, None),
         ("notes < 0.2 s removed", {"min_note": 0.2}, None),
         ("12 just-intonation states", {"quant": qji}, None), ("24 quarter-tone states", {"quant": q24}, None)]
    which = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(len(V))
    for vi in which:
        name, fe, te = V[vi]; row = {"variant": name}
        for setname, rids, secs, nw in [("150 s, 218 recordings", ALL, 150, 5), ("600 s, 194 long recordings", LONG, 600, 20),
                                         ("150 s, 194 long recordings", LONG, 150, 5)]:
            if te is None: err = None
            elif isinstance(te, str): err = np.random.default_rng(1234).normal(0, float(te[1:]), len(rids))
            else: err = np.full(len(rids), te)
            segs = encode(rids, std_windows(secs, nw), err, **fe)
            c1, c3 = correct(segs, rids, 1), correct(segs, rids, 3); g, l, p = mcn(c3, c1)
            tag = setname.split(",")[0].replace(" ", "") + ("_long" if "194" in setname else "")
            row.update({f"{tag}_k1": c1.mean(), f"{tag}_k3": c3.mean(), f"{tag}_gain": g, f"{tag}_loss": l, f"{tag}_p": p})
        print(row, round(time.time() - t0), flush=True)
        pd.DataFrame([row]).to_csv(OUT + f"84_frontend_{vi:02d}.csv", index=False)
    import glob as _g
    parts = sorted(_g.glob(OUT + "84_frontend_[0-9][0-9].csv"))
    if len(parts) == len(V):
        pd.concat([pd.read_csv(p) for p in parts]).to_csv(OUT + "84_frontend_robustness.csv", index=False)
        print("combined", len(parts), "front ends -> 84_frontend_robustness.csv")
