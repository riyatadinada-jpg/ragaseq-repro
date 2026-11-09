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
M = pd.DataFrame(meta); 
_=(M.groupby("form").agg(n=("id", "size"), singers=("singer", "nunique"), raags=("raag", "nunique"),
                            median_s=("duration_s", "median"), min_s=("duration_s", "min")))


