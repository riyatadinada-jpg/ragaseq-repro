#!/usr/bin/env python3
"""RagaSeq primary analysis: tonic-relative symbolic sequences, Smith–Waterman alignment and k-mer Jensen–Shannon distance for Hindustani raga recordings."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, re, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ===== STEP 1: Libraries and fixed settings =====
# Standard scientific libraries. SciPy (Mann-Whitney test) and Numba (faster alignment) are optional.

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None
try:
    from numba import njit
except Exception:
    njit = None

EPS = 1e-12
FEATURE_SUFFIXES = [".pitchSilIntrpPP", ".flatSegNyas", ".taniSegKNN", ".tonicFine", ".pitch", ".tonic"]

CANONICAL_ALIASES = {
    "darbari kanada": "Darbari Kanada",
    "darbari kanada ": "Darbari Kanada",
    "- darbari kanada": "Darbari Kanada",
    "darbari": "Darbari Kanada",
    "bageshree": "Bageshree",
    "bageshri": "Bageshree",
    "bhageshri": "Bageshree",
    "malkauns": "Malkauns",
    "maalkauns": "Malkauns",
    "yaman kalyan": "Yaman Kalyan",
    "yaman": "Yaman Kalyan",
    "bhairav": "Bhairav",
    "madhuvanti": "Madhuvanti",
    "madhuwanti": "Madhuvanti",
    "hamsadhwani": "Hamsadhwani",
    "hansdhwani": "Hamsadhwani",
    "hansadhwani": "Hamsadhwani",
    "kedar": "Kedar",
    "jog": "Jog",
    "desh": "Desh",
    "marwa": "Marwa",
}

# ===== STEP 2: Find and select the recordings =====
# Scans the feature folder, works out each file's recording ID and raga, and keeps only the
# recordings listed in the selection CSV (the locked 50-recording sample).

# Split a feature file name into its recording ID and feature type.
def parse_name(path: Path):
    name = path.name
    if name.startswith("._"):
        return None, None
    for suffix in FEATURE_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix.lstrip(".")
    return None, None

# Pull the recording UUID out of a file name.
def recording_id(stem: str) -> str:
    m = re.search(r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{27,})(?:\(\d+\))?$", stem)
    return m.group(1) if m else stem

# Turn a folder or file name into a standard raga name.
def clean_raga_label(stem: str) -> str:
    s = re.sub(r"^(Raga|Raag)_", "", stem, flags=re.I)
    s = re.sub(r"_[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}(?:\(\d+\))?$", "", s)
    s = s.replace("_", " ").strip(" -_")
    key = re.sub(r"\s+", " ", s).strip().lower()
    return CANONICAL_ALIASES.get(key, s)

# List every recording found, note which feature files exist, and mark the selected ones.
def build_manifest(data_dir: Path, selection_csv: str | None = None) -> pd.DataFrame:
    rows = {}
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        stem, feature = parse_name(p)
        if not stem:
            continue
        rid = recording_id(stem)
        raga = clean_raga_label(stem)
        key = (raga, rid)
        if key not in rows:
            rows[key] = {"raga": raga, "recording_id": rid, "stem": stem}
        rows[key][feature] = str(p)
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    for feature in [s.lstrip(".") for s in FEATURE_SUFFIXES]:
        if feature not in df:
            df[feature] = np.nan
    df["has_pitch"] = df["pitch"].notna()
    df["has_processed_pitch"] = df["pitchSilIntrpPP"].notna()
    df["has_tonic"] = df["tonic"].notna() | df["tonicFine"].notna()
    df["analysis_ready"] = df["has_pitch"] & df["has_tonic"]
    if selection_csv:
        sel = pd.read_csv(selection_csv, dtype=str)
        if "recording_id" not in sel.columns:
            raise ValueError("Selection CSV must contain a recording_id column")
        if sel["recording_id"].isna().any() or (sel["recording_id"].str.strip() == "").any():
            raise ValueError("Selection CSV contains a missing recording_id")
        if sel["recording_id"].duplicated().any():
            dup = sorted(sel.loc[sel["recording_id"].duplicated(), "recording_id"].unique())
            raise ValueError(f"Selection CSV contains duplicate recording_id values: {dup}")
        wanted = set(sel["recording_id"].astype(str))
        discovered = set(df["recording_id"].astype(str))
        missing = sorted(wanted - discovered)
        if missing:
            raise ValueError(f"Selection CSV recording_id values not found in data directory: {missing}")
        df["selected"] = df["recording_id"].astype(str).isin(wanted)
    else:
        df["selected"] = True
    return df.sort_values(["raga", "recording_id"]).reset_index(drop=True)

# ===== STEP 3: Read pitch and tonic =====
# Each recording has a pitch track (time, Hz) and a tonic estimate (Sa) supplied with the dataset.

# Read a single number, e.g. the tonic, from a .tonicFine or .tonic file.
def load_scalar(path):
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return np.nan
    a = np.loadtxt(path, ndmin=1)
    return float(np.ravel(a)[0])

# Read the pitch track: time (s) and frequency (Hz).
def load_pitch(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        if len(a) != 2:
            raise ValueError(f"Unexpected pitch format: {path}")
        a = a.reshape(1, 2)
    return a[:,0].astype(float), a[:,1].astype(float)

# Use the fine tonic estimate when available, otherwise the standard one.
def choose_tonic(row):
    fine = row.get("tonicFine")
    if isinstance(fine, str) and fine:
        return load_scalar(fine), "tonicFine"
    coarse = row.get("tonic")
    if isinstance(coarse, str) and coarse:
        return load_scalar(coarse), "tonic"
    return np.nan, "missing"

# ===== STEP 4: Turn pitch into symbols (Section 2.2) =====
# Hz -> cents above the tonic -> nearest of 12 semitone states (octave folded, Sa = 0)
# -> 3-frame median smoothing -> merge repeats. Done separately for five 30-s windows
# from the central 150 s of each recording.

# Convert Hz to cents above the tonic (Equation 1).
def cents_relative(freq, tonic):
    freq = np.asarray(freq, float)
    out = np.full(freq.shape, np.nan)
    voiced = freq > 0
    out[voiced] = 1200.0*np.log2(freq[voiced]/tonic)
    return out

# Round to the nearest semitone and fold into one octave: 12 states with Sa = 0.
# (Equation 2 in the paper.)
def quantize_12(cents):
    folded = np.mod(cents, 1200.0)
    return np.mod(np.rint(folded/100.0).astype(int), 12)

# Merge repeated states into a single symbol (run compression).
def compress_runs(seq):
    seq = np.asarray(seq, dtype=int)
    if len(seq) == 0:
        return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]

# 3-frame median filter that removes isolated one-frame flickers.
def median_smooth_discrete(seq, width=3):
    seq = np.asarray(seq, dtype=int)
    if width <= 1 or len(seq) < width:
        return seq
    k = width//2
    out = seq.copy()
    for i in range(k, len(seq)-k):
        out[i] = int(np.median(seq[i-k:i+k+1]))
    return out

# Five equal 30-s windows covering the central 150 s.
def fixed_windows(t, central_seconds=150.0, n_windows=5):
    if len(t) < 2:
        return []
    start, end = float(t[0]), float(t[-1])
    duration = end-start
    usable = min(central_seconds, duration)
    c = (start+end)/2.0
    s = max(start, c-usable/2.0)
    e = min(end, s+usable)
    if e-s < usable:
        s = max(start, e-usable)
    return [(s+i*(e-s)/n_windows, s+(i+1)*(e-s)/n_windows) for i in range(n_windows)]

# One window: 10 Hz median bins -> 12 states -> smoothing -> run compression.
def window_sequence(t, hz, tonic, w0, w1, sample_hz=10.0, smooth_width=3):
    mask = (t >= w0) & (t < w1) & (hz > 0)
    if mask.sum() < 5:
        return np.array([], dtype=int), np.nan
    tt, ff = t[mask], hz[mask]
    step = 1.0/sample_hz
    edges = np.arange(w0, w1+step, step)
    vals = []
    for a,b in zip(edges[:-1], edges[1:]):
        m = (tt >= a) & (tt < b)
        if m.any():
            vals.append(float(np.median(ff[m])))
        else:
            vals.append(np.nan)
    vals = np.asarray(vals, float)
    voiced_fraction = float(np.mean(np.isfinite(vals)))
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return np.array([], dtype=int), voiced_fraction
    bins = quantize_12(cents_relative(vals, tonic))
    bins = median_smooth_discrete(bins, smooth_width)
    return compress_runs(bins), voiced_fraction

# Build the symbolic sequences for all five windows of one recording.
def extract_recording(row, sample_hz=10.0, central_seconds=150.0, n_windows=5):
    tonic, tonic_source = choose_tonic(row)
    if not np.isfinite(tonic) or tonic <= 0:
        raise ValueError("Missing valid tonic")
    t, hz = load_pitch(row["pitch"])
    wins = fixed_windows(t, central_seconds=central_seconds, n_windows=n_windows)
    seqs, vf = [], []
    for w0,w1 in wins:
        s,v = window_sequence(t,hz,tonic,w0,w1,sample_hz=sample_hz)
        seqs.append(s); vf.append(v)
    all_voiced = hz[hz>0]
    cents = cents_relative(all_voiced, tonic) if len(all_voiced) else np.array([])
    rec = {
        "raga": row["raga"], "recording_id": row["recording_id"],
        "tonic_hz": tonic, "tonic_source": tonic_source,
        "full_duration_s": float(t[-1]-t[0]) if len(t)>1 else np.nan,
        "analysis_interval_s": min(central_seconds, float(t[-1]-t[0])) if len(t)>1 else np.nan,
        "mean_window_voiced_fraction": float(np.nanmean(vf)) if vf else np.nan,
        "pitch_iqr_cents": float(np.percentile(cents,75)-np.percentile(cents,25)) if len(cents) else np.nan,
        "pitch_range_95_cents": float(np.percentile(cents,97.5)-np.percentile(cents,2.5)) if len(cents) else np.nan,
        "mean_window_sequence_length": float(np.mean([len(s) for s in seqs])) if seqs else np.nan,
        "_sequences": seqs,
        "_windows": wins,
    }
    return rec

# ===== STEP 5: Compare recordings (Section 2.3) =====
# Two complementary measures for every pair of recordings, window by window:
# 3-mer Jensen-Shannon distance (composition) and normalized Smith-Waterman score (local order).

# Relative frequency of every k-note pattern (k = 3) in a sequence.
def kmer_profile(seq, k=3):
    seq = list(map(int, seq))
    c = Counter(tuple(seq[i:i+k]) for i in range(max(0,len(seq)-k+1)))
    total = sum(c.values())
    return {x:n/total for x,n in c.items()} if total else {}

# Jensen-Shannon distance between two pattern profiles (Equation 5).
def js_distance_dict(p,q):
    keys = set(p)|set(q)
    if not keys: return np.nan
    P=np.array([p.get(k,0.0) for k in keys],float); Q=np.array([q.get(k,0.0) for k in keys],float)
    P/=max(P.sum(),EPS); Q/=max(Q.sum(),EPS); M=.5*(P+Q)
    def kl(A,B):
        m=A>0
        return float(np.sum(A[m]*np.log2(A[m]/np.maximum(B[m],EPS))))
    return math.sqrt(max(0.0,.5*kl(P,M)+.5*kl(Q,M)))

# Smith-Waterman local alignment score (+2 match, -1 mismatch, -2 gap).
def _sw_python(a,b,match=2.0,mismatch=-1.0,gap=-2.0):
    prev=np.zeros(len(b)+1,dtype=np.float32); best=0.0
    for ai in a:
        curr=np.zeros(len(b)+1,dtype=np.float32)
        for j in range(1,len(b)+1):
            sc=match if ai==b[j-1] else mismatch
            v=max(0.0,prev[j-1]+sc,prev[j]+gap,curr[j-1]+gap)
            curr[j]=v; best=max(best,v)
        prev=curr
    return best
_sw_fast = njit(cache=True)(_sw_python) if njit is not None else _sw_python

# Best local score, also normalized by match score x shorter length (Equations 3-4).
def smith_waterman_score(a,b,match=2.0,mismatch=-1.0,gap=-2.0):
    a=np.asarray(a,dtype=np.int16); b=np.asarray(b,dtype=np.int16)
    if len(a)==0 or len(b)==0: return np.nan,np.nan
    best=float(_sw_fast(a,b,match,mismatch,gap))
    return best, best/(match*max(1,min(len(a),len(b))))

# Compare every pair of recordings window by window and average the five windows.
def pairwise_table(features,k=3):
    rows=[]
    for A,B in combinations(features,2):
        sws=[]; jss=[]
        for idx,(sa,sb) in enumerate(zip(A["_sequences"],B["_sequences"]),start=1):
            _,sw=smith_waterman_score(sa,sb)
            js=js_distance_dict(kmer_profile(sa,k),kmer_profile(sb,k))
            if np.isfinite(sw): sws.append(sw)
            if np.isfinite(js): jss.append(js)
        rows.append({
            "recording_a":A["recording_id"],"raga_a":A["raga"],
            "recording_b":B["recording_id"],"raga_b":B["raga"],
            "same_raga":A["raga"]==B["raga"],
            "comparison":"Same-raga" if A["raga"]==B["raga"] else "Cross-raga",
            "windows_compared":min(len(sws),len(jss)),
            "smith_waterman_norm_mean":float(np.mean(sws)) if sws else np.nan,
            "smith_waterman_norm_sd":float(np.std(sws,ddof=1)) if len(sws)>1 else np.nan,
            "kmer_js_distance_mean":float(np.mean(jss)) if jss else np.nan,
            "kmer_js_distance_sd":float(np.std(jss,ddof=1)) if len(jss)>1 else np.nan,
        })
    return pd.DataFrame(rows)

# ===== STEP 6: Statistics (Section 2.4) =====
# Same-raga vs cross-raga pairs: Cohen's d, Mann-Whitney U, and a permutation test that
# shuffles raga labels across recordings, so overlapping pairs are not treated as independent.

# Standardized difference between same-raga and cross-raga pairs (Equation 8).
def cohens_d(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); x=x[np.isfinite(x)]; y=y[np.isfinite(y)]
    if len(x)<2 or len(y)<2: return np.nan
    sx=np.var(x,ddof=1); sy=np.var(y,ddof=1)
    sp=math.sqrt(((len(x)-1)*sx+(len(y)-1)*sy)/(len(x)+len(y)-2))
    return float((np.mean(x)-np.mean(y))/sp) if sp>0 else np.nan

# Shuffle raga labels across recordings and recount (Equation 7).
def permutation_pvalue(pairdf,metric,n_perm=5000,seed=1234):
    vals=pairdf[metric].to_numpy(float); valid=np.isfinite(vals); pdf=pairdf.loc[valid].reset_index(drop=True); vals=vals[valid]
    if len(pdf)==0: return np.nan,np.nan
    labels={}
    for _,r in pdf.iterrows():
        labels[r.recording_a]=r.raga_a; labels[r.recording_b]=r.raga_b
    recs=sorted(labels); lv=np.array([labels[r] for r in recs],dtype=object)
    orig=pdf.same_raga.to_numpy(bool)
    if orig.sum()==0 or (~orig).sum()==0: return np.nan,np.nan
    obs=float(np.mean(vals[orig])-np.mean(vals[~orig]))
    rng=np.random.default_rng(seed); ge=0; valid_perm=0
    for _ in range(n_perm):
        pm=dict(zip(recs,rng.permutation(lv)))
        same=np.array([pm[a]==pm[b] for a,b in zip(pdf.recording_a,pdf.recording_b)])
        if same.sum()==0 or (~same).sum()==0: continue
        valid_perm+=1
        stat=float(np.mean(vals[same])-np.mean(vals[~same]))
        if abs(stat)>=abs(obs): ge+=1
    return obs,(ge+1)/(valid_perm+1) if valid_perm else np.nan

# Means, difference, Cohen's d, Mann-Whitney and permutation p (Table 2).
def summary_stats(pairdf,n_perm=5000,seed=1234):
    rows=[]
    for metric,expected in [("smith_waterman_norm_mean","higher_same"),("kmer_js_distance_mean","lower_same")]:
        same=pairdf.loc[pairdf.same_raga,metric].dropna().to_numpy(); cross=pairdf.loc[~pairdf.same_raga,metric].dropna().to_numpy()
        if len(same)==0 or len(cross)==0:
            rows.append({"metric":metric,"same_n":len(same),"cross_n":len(cross),"status":"Insufficient same-raga pairs for inference"})
            continue
        if mannwhitneyu is not None:
            alt="greater" if expected=="higher_same" else "less"
            mw=mannwhitneyu(same,cross,alternative=alt); u=float(mw.statistic); p=float(mw.pvalue)
        else: u=p=np.nan
        obs,pp=permutation_pvalue(pairdf,metric,n_perm=n_perm,seed=seed)
        rows.append({
            "metric":metric,"same_n":len(same),"cross_n":len(cross),
            "same_mean":float(np.mean(same)),"same_median":float(np.median(same)),
            "cross_mean":float(np.mean(cross)),"cross_median":float(np.median(cross)),
            "mean_difference_same_minus_cross":float(np.mean(same)-np.mean(cross)),
            "cohens_d_same_minus_cross":cohens_d(same,cross),"mannwhitney_u":u,
            "mannwhitney_one_sided_p":p,"label_permutation_difference":obs,
            "label_permutation_two_sided_p":pp,
            "status":"Primary contrast; recording-label permutation"})
    return pd.DataFrame(rows)

# How similar the five windows of the same recording are (a consistency check).
def temporal_stability(features):
    rows=[]
    for f in features:
        for i,j in combinations(range(len(f["_sequences"])),2):
            _,s=smith_waterman_score(f["_sequences"][i],f["_sequences"][j])
            if np.isfinite(s): rows.append({"raga":f["raga"],"recording_id":f["recording_id"],"window_a":i+1,"window_b":j+1,"smith_waterman_norm":s})
    return pd.DataFrame(rows)

# ===== STEP 7: Run the analysis and write the outputs =====
# Reads the arguments, runs Steps 2-6, checks the design is complete and balanced, and writes
# the CSV and JSON outputs listed at the top of this file.

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data_dir",nargs="?",default=".")
    ap.add_argument("--out",default="ragaseq_results_v2")
    ap.add_argument("--selection",default=None,help="CSV with recording_id column; only listed recordings are analyzed")
    ap.add_argument("--k",type=int,default=3)
    ap.add_argument("--windows",type=int,default=5)
    ap.add_argument("--central-seconds",type=float,default=150.0)
    ap.add_argument("--sample-hz",type=float,default=10.0)
    ap.add_argument("--permutations",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=1234,help="Random seed for recording-label permutations")
    ap.add_argument("--allow-incomplete",action="store_true",help="Allow analysis to continue after processing failures or with an unbalanced selected set")
    args=ap.parse_args()
    data_dir=Path(args.data_dir).resolve(); out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest=build_manifest(data_dir,args.selection); manifest.to_csv(out/"01_manifest.csv",index=False)
    ready=manifest[manifest.analysis_ready & manifest.selected].copy()
    features=[]; errors=[]
    for _,row in ready.iterrows():
        try: features.append(extract_recording(row,args.sample_hz,args.central_seconds,args.windows))
        except Exception as e: errors.append({"raga":row.raga,"recording_id":row.recording_id,"error":repr(e)})
    recdf=pd.DataFrame([{k:v for k,v in f.items() if not k.startswith("_")} for f in features]); recdf.to_csv(out/"02_recording_features.csv",index=False)
    if errors: pd.DataFrame(errors).to_csv(out/"00_errors.csv",index=False)
    processed_balance=(recdf.groupby("raga").agg(processed_recordings=("recording_id","nunique")).reset_index()
                       if len(recdf) else pd.DataFrame(columns=["raga","processed_recordings"]))
    processed_balanced_min=bool(len(processed_balance)>0 and
                                processed_balance.processed_recordings.nunique()==1 and
                                processed_balance.processed_recordings.min()>=2)
    if (errors or len(features)!=len(ready) or not processed_balanced_min) and not args.allow_incomplete:
        raise RuntimeError("Selected data did not process as a complete balanced set; inspect 00_errors.csv and rerun with --allow-incomplete only for explicitly exploratory work")
    pairdf=pairwise_table(features,args.k); pairdf.to_csv(out/"03_pairwise_comparisons.csv",index=False)
    statdf=summary_stats(pairdf,args.permutations,args.seed); statdf.to_csv(out/"04_statistical_tests.csv",index=False)
    tempdf=temporal_stability(features); tempdf.to_csv(out/"05_temporal_stability.csv",index=False)
    bal=processed_balance.rename(columns={"processed_recordings":"ready_recordings"})
    bal["meets_min_2"] = bal["ready_recordings"]>=2 if len(bal) else []
    bal["meets_preferred_3"] = bal["ready_recordings"]>=3 if len(bal) else []
    bal.to_csv(out/"06_raga_balance.csv",index=False)
    balanced_min = processed_balanced_min
    selection_hash=(hashlib.sha256(Path(args.selection).read_bytes()).hexdigest() if args.selection else None)
    summary={"data_dir":str(data_dir),"n_discovered_recordings":int(len(manifest)),"n_selected_ready":int(len(ready)),"n_processed":int(len(features)),"n_processing_errors":int(len(errors)),"ragas_processed":sorted(recdf.raga.unique().tolist()) if len(recdf) else [],"balanced_for_inference":balanced_min,"inference_mode":"complete balanced" if balanced_min and not errors else "incomplete exploratory","k":args.k,"windows":args.windows,"central_seconds":args.central_seconds,"sample_hz":args.sample_hz,"permutations":args.permutations,"permutation_seed":args.seed,"selection_csv_sha256":selection_hash,"primary_files":".pitch + tonicFine/tonic","software":{"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__},"method_note":"Empty 100-ms bins are removed before symbolic encoding; symbols on either side of an unvoiced gap can therefore become adjacent in the compressed sequence.","caveat":"Primary inference is melodic/pitch-sequence only; no harmonic, rhythmic, EEG, or therapeutic claims."}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
