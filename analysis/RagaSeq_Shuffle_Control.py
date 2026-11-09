#!/usr/bin/env python3
"""RagaSeq shuffle control: does melodic ORDER carry raga information, beyond note content?

The question
------------
RagaSeq-D and the original RagaSeq measures separate ragas well, but ragas also differ in
WHICH notes they use and how long each is held. This script destroys note order while
keeping note content, re-runs the same analyses, and asks how much performance is lost.

Two shuffles, applied independently to every 30-s window of every recording
--------------------------------------------------------------------------
1. FULL shuffle (order destroyed)
   The window's notes are randomly permuted, each note keeping its held duration.
   Adjacent repeats created by the shuffle are merged (durations summed), so total held
   time per pitch class is preserved EXACTLY. Only order changes.

2. DYAD shuffle (order beyond note pairs destroyed)
   The Altschul-Erickson (1985) dinucleotide-preserving shuffle from bioinformatics:
   a random Eulerian path that keeps every 2-note transition count exactly, so the
   sequence keeps its note content AND its note-to-note transitions (the pitch-class
   dyad information of Chordia & Rae, 2007). Held durations are reshuffled among
   occurrences of the same note. What it destroys is order beyond adjacent pairs,
   i.e. exactly the extra information a 3-note pattern adds over a 2-note transition.

Reading the result
------------------
  intact >> full shuffle   -> note order carries raga information (not just note content)
  intact >> dyad shuffle   -> 3-note patterns add information beyond 2-note transitions
  intact ~= shuffled       -> the separation comes from note content / transitions only

Analyses re-run for every shuffle replicate
-------------------------------------------
  * RagaSeq-D recognition (performer-aware leave-one-out, same code as Script 3)
  * RagaSeq-D same-raga vs cross-raga separation (Cohen's d)
  * optional (--include-alignment): original measures, normalized Smith-Waterman (SW-4)
    and 3-mer Jensen-Shannon distance, same- vs cross-raga Cohen's d (H1 and H2).
    Smith-Waterman is slow without Numba (pip install numba).

Place this file in the SAME folder as Scripts 1 and 3:
  RagaSeq_10Raga_Analysis_FINAL_Annotated.py and RagaSeq_Duration_Recognition.py

Examples
--------
  python RagaSeq_Shuffle_Control.py --self-test
  python RagaSeq_Shuffle_Control.py FEATURE_DIR \
      --selection RagaSeq_10x5_Selection_with_performers.csv \
      --include-alignment --out results_shuffle_development
  python RagaSeq_Shuffle_Control.py FEATURE_DIR \
      --selection RagaSeq_TestSet_14x5_Selection.csv --out results_shuffle_test

Outputs (in --out)
------------------
  31_shuffle_replicates.csv   one row per condition x replicate x measure
  32_shuffle_summary.csv      intact value, shuffled mean and 95% range, empirical p
  run_summary_shuffle.json    settings and software versions
"""
from __future__ import annotations
import argparse, json, platform, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1  Shuffles (no dependency on the other scripts, so --self-test runs anywhere)
# ---------------------------------------------------------------------------

def merge_adjacent(notes, durs):
    """Merge neighbouring identical notes, summing their held durations."""
    notes = np.asarray(notes, int); durs = np.asarray(durs, float)
    if len(notes) == 0:
        return notes, durs
    keep = np.r_[True, notes[1:] != notes[:-1]]
    group = np.cumsum(keep) - 1
    return notes[keep], np.bincount(group, weights=durs)


def full_shuffle(notes, durs, rng):
    """Permute notes (each keeps its duration), then merge adjacent repeats.
    Preserves total held time of every pitch class exactly; destroys order."""
    notes = np.asarray(notes, int); durs = np.asarray(durs, float)
    if len(notes) < 2:
        return notes.copy(), durs.copy()
    perm = rng.permutation(len(notes))
    return merge_adjacent(notes[perm], durs[perm])


def euler_dyad_shuffle(notes, rng, max_tries=10000):
    """Altschul-Erickson shuffle: random sequence with identical 2-note transition counts,
    same first and last note. Works on run-compressed sequences (no self-transitions)."""
    s = [int(x) for x in notes]
    n = len(s)
    if n < 3:
        return np.array(s, int)
    out = defaultdict(list)
    for a, b in zip(s[:-1], s[1:]):
        out[a].append(b)
    last = s[-1]
    verts = list(out.keys())
    for _ in range(max_tries):
        # choose a random last-exit edge for every vertex except the final one
        last_edge = {v: out[v][int(rng.integers(len(out[v])))] for v in verts if v != last}
        # the last-exit edges must form a tree that leads to the final vertex
        ok = True
        for v in last_edge:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = last_edge[u]
            if not ok:
                break
        if ok:
            break
    else:
        raise RuntimeError("Euler shuffle failed to find a valid last-edge tree")
    order = {}
    for v in verts:
        edges = list(out[v])
        if v in last_edge:
            edges.remove(last_edge[v])
            rng.shuffle(edges)
            edges.append(last_edge[v])
        else:
            rng.shuffle(edges)
        order[v] = edges
    pos = {v: 0 for v in verts}
    res, u = [s[0]], s[0]
    for _ in range(n - 1):
        nxt = order[u][pos[u]]
        pos[u] += 1
        res.append(nxt)
        u = nxt
    return np.array(res, int)


def dyad_shuffle(notes, durs, rng):
    """Dyad-preserving note order; durations reshuffled among occurrences of the same note."""
    notes = np.asarray(notes, int); durs = np.asarray(durs, float)
    if len(notes) < 3:
        return notes.copy(), durs.copy()
    new_notes = euler_dyad_shuffle(notes, rng)
    pools = {}
    for v in np.unique(notes):
        d = durs[notes == v].copy()
        rng.shuffle(d)
        pools[int(v)] = list(d)
    new_durs = np.array([pools[int(v)].pop() for v in new_notes], float)
    return new_notes, new_durs


SHUFFLES = {"full_shuffle": full_shuffle, "dyad_shuffle": dyad_shuffle}


def self_test(seed=1234):
    """Check that the shuffles preserve what they claim to preserve."""
    rng = np.random.default_rng(seed)
    for trial in range(500):
        n = int(rng.integers(3, 200))
        raw = rng.integers(0, 12, n * 2)
        notes = raw[np.r_[True, raw[1:] != raw[:-1]]][:n]
        durs = rng.uniform(0.1, 2.0, len(notes)).round(1)
        # full shuffle: identical held time per pitch class, no adjacent repeats
        fn, fd = full_shuffle(notes, durs, rng)
        for v in range(12):
            assert np.isclose(fd[fn == v].sum(), durs[notes == v].sum())
        assert not np.any(fn[1:] == fn[:-1])
        # dyad shuffle: identical transition counts, endpoints, durations per note
        dn, dd = dyad_shuffle(notes, durs, rng)
        def dy(x): return sorted(zip(x[:-1].tolist(), x[1:].tolist()))
        assert dy(dn) == dy(notes)
        assert dn[0] == notes[0] and dn[-1] == notes[-1]
        for v in range(12):
            assert np.allclose(sorted(dd[dn == v]), sorted(durs[notes == v]))
    # the dyad shuffle should actually change order
    notes = np.array([0, 2, 4, 2, 0, 4, 7, 4, 2, 7, 0, 2, 7, 4, 0])
    seen = {tuple(euler_dyad_shuffle(notes, rng)) for _ in range(200)}
    assert len(seen) > 1
    print("Self-test passed: full shuffle preserves held time per note; "
          "dyad shuffle preserves every 2-note transition count.")


# ---------------------------------------------------------------------------
# Step 2  Measures on a set of (possibly shuffled) recordings
# ---------------------------------------------------------------------------

def distance_matrix(profiles, js):
    n = len(profiles)
    M = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        M[i, j] = M[j, i] = js(profiles[i], profiles[j])
    return M


def same_cross_d(M, labels, cohens_d):
    n = len(labels)
    same, cross = [], []
    for i, j in combinations(range(n), 2):
        (same if labels[i] == labels[j] else cross).append(M[i, j])
    return cohens_d(same, cross)


def alignment_measures(runs_per_rec, labels, R, k=3):
    """Original RagaSeq measures (SW-4 and 3-mer JSD), window by window, averaged."""
    n = len(runs_per_rec)
    seqs = [[np.asarray(nt, int) for nt, _ in rec] for rec in runs_per_rec]
    profs = [[R.kmer_profile(s, k) for s in rec] for rec in seqs]
    sw = np.full((n, n), np.nan); jd = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        a, b = [], []
        for w in range(min(len(seqs[i]), len(seqs[j]))):
            _, s = R.smith_waterman_score(seqs[i][w], seqs[j][w])
            d = R.js_distance_dict(profs[i][w], profs[j][w])
            if np.isfinite(s): a.append(s)
            if np.isfinite(d): b.append(d)
        sw[i, j] = sw[j, i] = np.mean(a) if a else np.nan
        jd[i, j] = jd[j, i] = np.mean(b) if b else np.nan
    return {"SW-4 similarity (d)": same_cross_d(sw, labels, R.cohens_d),
            "3-mer JSD (d)": same_cross_d(jd, labels, R.cohens_d)}


def evaluate(runs_per_rec, labels, performers, R, D, k, include_alignment):
    prof = [D.duration_profile(r, k) for r in runs_per_rec]
    M = distance_matrix(prof, R.js_distance_dict)
    res = D.recognise(M, labels, performers, 0, 0)
    out = {"RagaSeq-D accuracy": res["accuracy"],
           "RagaSeq-D top-3": res["top3_correct"] / res["n"],
           "RagaSeq-D separation (d)": same_cross_d(M, labels, R.cohens_d)}
    if include_alignment:
        out.update(alignment_measures(runs_per_rec, labels, R, k))
    return out


# Direction in which "better" lies, for the empirical p-value
HIGHER_IS_BETTER = {"RagaSeq-D accuracy": True, "RagaSeq-D top-3": True,
                    "RagaSeq-D separation (d)": False,   # distance: more negative = better
                    "SW-4 similarity (d)": True,
                    "3-mer JSD (d)": False}


# ---------------------------------------------------------------------------
# Step 3  Run
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=".")
    ap.add_argument("--selection", help="CSV with recording_id and performer columns")
    ap.add_argument("--out", default="ragaseq_shuffle")
    ap.add_argument("--replicates", type=int, default=100, help="shuffles per condition")
    ap.add_argument("--include-alignment", action="store_true",
                    help="also re-run Smith-Waterman and 3-mer JSD (slow without numba)")
    ap.add_argument("--alignment-replicates", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--central-seconds", type=float, default=150.0)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.seed)
        return
    if not args.selection:
        ap.error("--selection is required (or use --self-test)")

    import RagaSeq_10Raga_Analysis_FINAL_Annotated as R
    import RagaSeq_Duration_Recognition as D
    self_test(args.seed)

    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = R.build_manifest(Path(args.data_dir).resolve(), args.selection)
    ready = manifest[manifest.analysis_ready & manifest.selected].reset_index(drop=True)
    sel = pd.read_csv(args.selection, dtype=str)
    perf_map = dict(zip(sel.recording_id, sel.performer)) if "performer" in sel.columns else {}
    if not perf_map:
        print("WARNING: no performer column; recognition will not exclude same-performer recordings.")

    recs = [D.load_recording(row, args) for _, row in ready.iterrows()]
    labels = [r["raga"] for r in recs]
    performers = [str(perf_map.get(r["recording_id"], "")) for r in recs]
    intact_runs = [r["runs"] for r in recs]
    print(f"Loaded {len(recs)} recordings, {len(set(labels))} raags")

    rows = []
    intact = evaluate(intact_runs, labels, performers, R, D, args.k, args.include_alignment)
    for m, v in intact.items():
        rows.append({"condition": "intact", "replicate": 0, "measure": m, "value": v})
    print("Intact:", {m: round(v, 3) for m, v in intact.items()})
    print("(Check: RagaSeq-D accuracy should match 23_recognition_summary.csv from Script 3.)")

    rng = np.random.default_rng(args.seed)
    for cond, fn in SHUFFLES.items():
        for rep in range(1, args.replicates + 1):
            shuf = [[fn(nt, du, rng) for nt, du in rec] for rec in intact_runs]
            with_sw = args.include_alignment and rep <= args.alignment_replicates
            res = evaluate(shuf, labels, performers, R, D, args.k, with_sw)
            for m, v in res.items():
                rows.append({"condition": cond, "replicate": rep, "measure": m, "value": v})
            if rep % 10 == 0:
                print(f"  {cond}: {rep}/{args.replicates}")

    reps = pd.DataFrame(rows)
    reps.to_csv(out / "31_shuffle_replicates.csv", index=False)

    summ = []
    for m, iv in intact.items():
        for cond in SHUFFLES:
            v = reps[(reps.condition == cond) & (reps.measure == m)].value.to_numpy(float)
            if len(v) == 0:
                continue
            at_least = (v >= iv) if HIGHER_IS_BETTER[m] else (v <= iv)
            summ.append({"measure": m, "condition": cond, "intact": iv,
                         "shuffled_mean": v.mean(),
                         "shuffled_2.5pct": np.percentile(v, 2.5),
                         "shuffled_97.5pct": np.percentile(v, 97.5),
                         "intact_minus_shuffled_mean": iv - v.mean(),
                         "n_replicates": len(v),
                         "empirical_p_one_sided": (at_least.sum() + 1) / (len(v) + 1)})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "32_shuffle_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False))

    (out / "run_summary_shuffle.json").write_text(json.dumps({
        "n_recordings": len(recs), "ragas": sorted(set(labels)), "replicates": args.replicates,
        "alignment_replicates": args.alignment_replicates if args.include_alignment else 0,
        "k": args.k, "windows": args.windows, "central_seconds": args.central_seconds,
        "sample_hz": args.sample_hz, "seed": args.seed, "performer_aware": bool(perf_map),
        "shuffles": {"full_shuffle": "notes permuted within window, durations attached, "
                                     "adjacent repeats merged",
                     "dyad_shuffle": "Altschul-Erickson Eulerian shuffle preserving 2-note "
                                     "transition counts; durations reshuffled within note"},
        "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                     "numpy": np.__version__, "pandas": pd.__version__}}, indent=2))


if __name__ == "__main__":
    main()
