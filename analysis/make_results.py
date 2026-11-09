"""Collect every number the manuscript quotes into results.json (single source for the text)."""
import json, numpy as np, pandas as pd
from scipy.stats import beta

W = ""
res = {}
HB = {"RagaSeq-D accuracy": 1, "RagaSeq-D top-3": 1, "RagaSeq-D separation (d)": 0,
      "SW-4 similarity (d)": 1, "3-mer JSD (d)": 0}


def shuffle_summary(d):
    out = {}
    for m, g in d.groupby("measure"):
        iv = float(g[g.condition == "intact"].value.iloc[0]); out[m] = {"intact": iv}
        for c in ["full_shuffle", "dyad_shuffle"]:
            v = g[g.condition == c].value.to_numpy(float)
            if not len(v): continue
            ge = (v >= iv).sum() if HB[m] else (v <= iv).sum()
            out[m][c] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                         "best": float(v.max() if HB[m] else v.min()), "n": int(len(v)), "p": float((ge + 1) / (len(v) + 1))}
    return out


sr = pd.read_csv(W + "shuffle_replicates.csv")
res["shuffle"] = {s: shuffle_summary(g) for s, g in sr.groupby("set", sort=False)}
res["shuffle"]["Expanded (30 raags, 218)"] = shuffle_summary(pd.read_csv(W + "results_expanded/47_shuffle_replicates_expanded.csv"))

E = W + "results_expanded/"
res["expanded"] = pd.read_csv(E + "41_recognition_summary.csv").to_dict("records")
res["mcnemar"] = pd.read_csv(E + "45_method_comparison_mcnemar.csv").to_dict("records")
res["ksweep"] = pd.read_csv(E + "46_pattern_length_sweep.csv").to_dict("records")
res["alphabet"] = pd.read_csv(E + "48_pitch_alphabet_sensitivity.csv").to_dict("records")
res["per_raag"] = pd.read_csv(E + "44_per_raag_accuracy.csv").to_dict("records")
sel = pd.read_csv(W + "RagaSeq_Expanded_Selection.csv")
res["sel"] = {"n": len(sel), "raags": int(sel.raag.nunique()), "performers": int(sel.performer.nunique()),
              "min_per_raag": int(sel.raag.value_counts().min()), "max_per_raag": int(sel.raag.value_counts().max()),
              "median_dur_min": float(sel.duration_s.median() / 60), "groups": sel.raag_group.value_counts().to_dict(),
              "group_raags": sel.groupby("raag_group").raag.nunique().to_dict()}
json.dump(res, open(W + "results.json", "w"), indent=1, default=float)
print(json.dumps(res["shuffle"], indent=1)[:4000])
