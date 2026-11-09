"""Shuffle control on the development (10x5) and held-out test (14x5) sets."""
import sys, json, time, numpy as np, pandas as pd
import common as C
import importlib
R, D = C.R, C.D
import RagaSeq_Shuffle_Control as S
importlib.reload(R)  # reload with numba now available
R.load_pitch = lambda rid: (C.PK[rid][0], C.PK[rid][1])
R.choose_tonic = lambda row: (float(C.PK[row["pitch"]][2]), "tonicFine")
D.R = R

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SW_REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
out = []
for name, sel in [("Development (10 raags x 5)", "RagaSeq_10x5_Selection_with_performers.csv"),
                  ("Held-out test (14 raags x 5)", "RagaSeq_TestSet_14x5_Selection.csv")]:
    s = pd.read_csv(sel, dtype=str); perf = dict(zip(s.recording_id, s.performer))
    recs = C.load(s.recording_id)
    labels = [r["raga"] for r in recs]; performers = [perf[r["recording_id"]] for r in recs]
    runs = [r["runs"] for r in recs]
    t0 = time.time()
    intact = S.evaluate(runs, labels, performers, R, D, 3, True)
    out += [dict(set=name, condition="intact", replicate=0, measure=m, value=v) for m, v in intact.items()]
    print(name, "intact", {m: round(v, 3) for m, v in intact.items()}, round(time.time() - t0), "s", flush=True)
    rng = np.random.default_rng(1234)
    for cond, fn in S.SHUFFLES.items():
        for rep in range(1, REPS + 1):
            sh = [[fn(nt, du, rng) for nt, du in rec] for rec in runs]
            res = S.evaluate(sh, labels, performers, R, D, 3, rep <= SW_REPS)
            out += [dict(set=name, condition=cond, replicate=rep, measure=m, value=v) for m, v in res.items()]
        print(name, cond, "done", round(time.time() - t0), "s", flush=True)
pd.DataFrame(out).to_csv("shuffle_replicates.csv", index=False)
