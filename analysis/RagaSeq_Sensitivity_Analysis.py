#!/usr/bin/env python3
"""RagaSeq sensitivity analysis: k-mer length and Smith-Waterman scoring schemes.

Re-runs the primary same-raga vs cross-raga contrast under alternative settings
to show whether the result depends on the parameter choices.

  * 3-mer JSD is recomputed for k = 2, 3, 4          (affects JSD only)
  * Smith-Waterman is recomputed for SW-1 ... SW-7    (affects SW only)

Pitch extraction and symbolic encoding run ONCE and are shared by all
settings, so every setting is compared on identical sequences.

Place this file in the SAME folder as RagaSeq_10Raga_Analysis_FINAL_Annotated.py
(it imports that script's functions, including clean_raga_label).

Example (10 ragas x 5 recordings = 50 recordings, 1,225 pairs)
--------------------------------------------------------------
python RagaSeq_Sensitivity_Analysis.py FEATURE_DIRECTORY \
  --selection RagaSeq_10x5_Selection.csv --out ragaseq_sensitivity_10x5

Outputs (in --out)
------------------
07_sensitivity_pairwise.csv  one row per recording pair, one column per setting
08_sensitivity_summary.csv   one row per setting: means, difference, Cohen's d,
                             Mann-Whitney p, label-permutation p, direction check
sensitivity_run_summary.json settings, counts and software versions

Check: the rows marked is_primary (SW-4 and k=3) should reproduce
04_statistical_tests.csv from the main script exactly.
"""
from __future__ import annotations
import argparse, json, platform, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import RagaSeq_10Raga_Analysis_FINAL_Annotated as R

# Scenario grid (match, mismatch, gap). SW-4 is the primary analysis.

# ===== STEP 1: The ten settings (Section 2.5) =====
# Seven Smith-Waterman scoring schemes (match / mismatch / gap) and three k-mer lengths.
# SW-4 and k = 3 are the settings used in the main analysis.

SW_SCHEMES = {
    "SW-1": (1, -1, -1),
    "SW-2": (1, -1, -2),
    "SW-3": (2, -1, -1),
    "SW-4": (2, -1, -2),
    "SW-5": (2, -1, -3),
    "SW-6": (2, -2, -2),
    "SW-7": (3, -1, -2),
}
PRIMARY_SW = "SW-4"
PRIMARY_K = 3

# ===== STEP 2: Build the symbolic sequences once =====
# Uses the main script's functions, so every setting compares exactly the same sequences.

# Load the selected recordings and build their symbolic sequences with the main script.
def load_features(args):
    """Same discovery, selection, extraction and balance checks as the main script."""
    data_dir = Path(args.data_dir).resolve()
    manifest = R.build_manifest(data_dir, args.selection)
    if manifest.empty:
        raise RuntimeError(f"No feature files found in {data_dir}")
    ready = manifest[manifest.analysis_ready & manifest.selected].copy()
    features, errors = [], []
    for _, row in ready.iterrows():
        try:
            features.append(R.extract_recording(row, args.sample_hz, args.central_seconds, args.windows))
        except Exception as e:
            errors.append({"raga": row.raga, "recording_id": row.recording_id, "error": repr(e)})
    counts = pd.Series([f["raga"] for f in features]).value_counts()
    balanced = len(counts) > 0 and counts.nunique() == 1 and counts.min() >= 2
    if (errors or len(features) != len(ready) or not balanced) and not args.allow_incomplete:
        raise RuntimeError("Selected data did not process as a complete balanced set; "
                           "check the main script's 00_errors.csv / 01_manifest.csv first")
    return data_dir, manifest, features, errors

# ===== STEP 3: Compare every pair under every setting =====
# One column per setting: normalized Smith-Waterman score for SW-1 to SW-7, Jensen-Shannon distance for k = 2, 3, 4.

# Score every recording pair under all ten settings.
def pairwise_all_settings(features, kmers):
    """One row per recording pair; a column per SW scheme and per k."""
    # Pre-compute k-mer profiles once per recording, window and k.
    profiles = {k: [[R.kmer_profile(s, k) for s in f["_sequences"]] for f in features] for k in kmers}
    rows = []
    for (ia, A), (ib, B) in combinations(enumerate(features), 2):
        row = {"recording_a": A["recording_id"], "raga_a": A["raga"],
               "recording_b": B["recording_id"], "raga_b": B["raga"],
               "same_raga": A["raga"] == B["raga"]}
        win_pairs = list(zip(A["_sequences"], B["_sequences"]))
        for name, (m, mm, g) in SW_SCHEMES.items():
            vals = []
            for sa, sb in win_pairs:
                _, sw = R.smith_waterman_score(sa, sb, match=m, mismatch=mm, gap=g)
                if np.isfinite(sw):
                    vals.append(sw)
            row[f"sw_{name}"] = float(np.mean(vals)) if vals else np.nan
        for k in kmers:
            vals = []
            for w in range(min(len(profiles[k][ia]), len(profiles[k][ib]))):
                js = R.js_distance_dict(profiles[k][ia][w], profiles[k][ib][w])
                if np.isfinite(js):
                    vals.append(js)
            row[f"jsd_k{k}"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

# ===== STEP 4: Statistics for each setting (Table 3) =====
# Same statistics as the main analysis, plus a check that the effect is in the expected direction.

# Summarize one setting: means, Cohen's d, Mann-Whitney p, permutation p, direction.
def summarise(pairdf, column, metric, setting, params, expected, n_perm, seed, is_primary):
    same = pairdf.loc[pairdf.same_raga, column].dropna().to_numpy()
    cross = pairdf.loc[~pairdf.same_raga, column].dropna().to_numpy()
    if R.mannwhitneyu is not None:
        alt = "greater" if expected == "higher_same" else "less"
        mw_p = float(R.mannwhitneyu(same, cross, alternative=alt).pvalue)
    else:
        mw_p = np.nan
    obs, perm_p = R.permutation_pvalue(pairdf, column, n_perm=n_perm, seed=seed)
    diff = float(np.mean(same) - np.mean(cross))
    return {
        "metric": metric, "setting": setting, "parameters": params, "is_primary": is_primary,
        "same_n": len(same), "cross_n": len(cross),
        "same_mean": float(np.mean(same)), "cross_mean": float(np.mean(cross)),
        "mean_difference_same_minus_cross": diff,
        "cohens_d_same_minus_cross": R.cohens_d(same, cross),
        "mannwhitney_one_sided_p": mw_p,
        "label_permutation_two_sided_p": perm_p,
        "expected_direction": "same > cross" if expected == "higher_same" else "same < cross",
        "direction_as_expected": bool(diff > 0) if expected == "higher_same" else bool(diff < 0),
    }

# ===== STEP 5: Run and write the outputs =====
# Writes 07_sensitivity_pairwise.csv and 08_sensitivity_summary.csv.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=".")
    ap.add_argument("--out", default="ragaseq_sensitivity")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--kmers", default="2,3,4", help="Comma-separated k values")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--central-seconds", type=float, default=150.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    kmers = [int(k) for k in args.kmers.split(",") if k.strip()]

    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    data_dir, manifest, features, errors = load_features(args)
    print(f"Processed {len(features)} recordings; computing {len(SW_SCHEMES)} SW schemes and k = {kmers} ...")

    pairdf = pairwise_all_settings(features, kmers)
    pairdf.to_csv(out / "07_sensitivity_pairwise.csv", index=False)

    rows = []
    for name, (m, mm, g) in SW_SCHEMES.items():
        rows.append(summarise(pairdf, f"sw_{name}", "smith_waterman_norm", name,
                              f"match {m:+d}, mismatch {mm:+d}, gap {g:+d}", "higher_same",
                              args.permutations, args.seed, name == PRIMARY_SW))
        print(f"  {name} done")
    for k in kmers:
        rows.append(summarise(pairdf, f"jsd_k{k}", "kmer_js_distance", f"k={k}",
                              f"k-mer length {k}", "lower_same",
                              args.permutations, args.seed, k == PRIMARY_K))
        print(f"  k={k} done")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "08_sensitivity_summary.csv", index=False)

    run = {"data_dir": str(data_dir), "n_recordings": len(features), "n_pairs": int(len(pairdf)),
           "n_processing_errors": len(errors), "sw_schemes": SW_SCHEMES, "kmers": kmers,
           "primary": {"sw": PRIMARY_SW, "k": PRIMARY_K},
           "settings_in_expected_direction": f"{int(summary.direction_as_expected.sum())}/{len(summary)}",
           "windows": args.windows, "central_seconds": args.central_seconds, "sample_hz": args.sample_hz,
           "permutations": args.permutations, "permutation_seed": args.seed,
           "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__}}
    (out / "sensitivity_run_summary.json").write_text(json.dumps(run, indent=2))

    cols = ["setting", "parameters", "same_mean", "cross_mean", "mean_difference_same_minus_cross",
            "cohens_d_same_minus_cross", "label_permutation_two_sided_p", "direction_as_expected"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(4).to_string(index=False))
    print(f"\nSettings in expected direction: {run['settings_in_expected_direction']}")


if __name__ == "__main__":
    main()
