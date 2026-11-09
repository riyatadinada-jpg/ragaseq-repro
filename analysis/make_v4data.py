import json, numpy as np, pandas as pd
R = json.load(open("results.json")); N = json.load(open("nested.json")); B = json.load(open("balanced.json"))
acc = "RagaSeq-D accuracy"; S = R["shuffle"]
nested = {r["set"]: r for r in N["nested"]}
sh = pd.read_csv("results_dap/64_dap_shuffle_replicates.csv"); sh = sh[sh.measure == acc]
def dap(form, c):
    v = sh[(sh.form == form) & (sh.condition == c)].value.to_numpy()
    return dict(mean=float(v.mean()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), best=float(v.max()), n=len(v),
                p=float(((v >= sh[(sh.form == form) & (sh.condition == "intact")].value.iloc[0]).sum() + 1) / (len(v) + 1)))
sets = []
for lab, key, s in [("CompMusic\ndevelopment\n(10 ragas, 50)", "CompMusic development (10 x 5)", S["Development (10 raags x 5)"]),
                    ("CompMusic\nheld-out\n(14 ragas, 70)", "CompMusic held-out (14 x 5)", S["Held-out test (14 raags x 5)"]),
                    ("CompMusic\nexpanded\n(30 ragas, 218)", "CompMusic expanded (30 raags, 218)", S["Expanded (30 raags, 218)"])]:
    n = nested[key]; sets.append(dict(label=lab, key=key, k1=n["acc_k1"], k2=n["acc_k2"], k3=n["acc_k3"],
        full=s[acc]["full_shuffle"]["mean"], dyad=s[acc]["dyad_shuffle"]["mean"], full_ci=[s[acc]["full_shuffle"]["lo"], s[acc]["full_shuffle"]["hi"]],
        dyad_ci=[s[acc]["dyad_shuffle"]["lo"], s[acc]["dyad_shuffle"]["hi"]], full_p=s[acc]["full_shuffle"]["p"], dyad_p=s[acc]["dyad_shuffle"]["p"], nested=n))
sets.append(dict(label="CompMusic\nbalanced\n(29 ragas × 5)", key="balanced", k1=N["balanced"]["k1"], k2=N["balanced"]["k2"], k3=N["balanced"]["k3"],
    full=B["full_shuffle"]["mean"], dyad=B["dyad_shuffle"]["mean"], full_ci=[B["full_shuffle"]["lo"], B["full_shuffle"]["hi"]],
    dyad_ci=[B["dyad_shuffle"]["lo"], B["dyad_shuffle"]["hi"]], nested=N["balanced"]))
for f, lab in [("alap", "DAP-Lab alap\n(9 ragas, 197)"), ("pakad", "DAP-Lab pakad\n(9 ragas, 102)")]:
    key = [k for k in nested if k.startswith("DAP-Lab " + f)][0]; n = nested[key]; fu, dy = dap(f, "full_shuffle"), dap(f, "dyad_shuffle")
    sets.append(dict(label=lab, key=key, k1=n["acc_k1"], k2=n["acc_k2"], k3=n["acc_k3"], full=fu["mean"], dyad=dy["mean"],
        full_ci=[fu["lo"], fu["hi"]], dyad_ci=[dy["lo"], dy["hi"]], full_p=fu["p"], dyad_p=dy["p"], full_best=fu["best"], dyad_best=dy["best"], nested=n))
dr = pd.read_csv("results_dap/62_dap_recognition.csv"); dm = pd.read_csv("results_dap/61_dap_recordings.csv")
sg = pd.read_csv("results_saraga/52_saraga_recognition.csv"); sgm = pd.read_csv("results_saraga/54_saraga_mcnemar.csv")
ex = pd.read_csv("results_dap/65_dap_external.csv")
bal = {k: B[k] for k in B}
out = dict(sets=sets, length=N["length"], balanced=bal, balanced_nested=N["balanced"], dap_rec=dr.to_dict("records"),
           dap_meta=dict(n=len(dm), alap=int((dm.form == "alap").sum()), pakad=int((dm.form == "pakad").sum()), singers=int(dm.singer.nunique()),
                         alap_median=float(dm[dm.form == "alap"].duration_s.median()), pakad_median=float(dm[dm.form == "pakad"].duration_s.median())),
           saraga=sg.to_dict("records"), len_mcn=json.load(open("len_mcn.json")), saraga_mc=sgm.to_dict("records"), dap_ext=ex.to_dict("records"))
import math
clean = lambda o: {k: clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, list) else (None if isinstance(o, float) and math.isnan(o) else o)
json.dump(clean(json.loads(json.dumps(out, default=float))), open("v4.json", "w"), indent=1); print("v4.json ok")
import os
parts = [p for p in ["results_expanded/70a_150s.csv", "results_expanded/70_order_test_CompMusic.csv", "results_expanded/70_order_test_DAP-Lab.csv"] if os.path.exists(p)]
if parts: pd.concat([pd.read_csv(p) for p in parts]).to_csv("results_expanded/70_order_test_all_methods.csv", index=False)
f = "results_expanded/70_order_test_all_methods.csv"
if os.path.exists(f):
    T = pd.read_csv(f); ns = {"CompMusic expanded, 150 s": 218, "CompMusic long, 600 s": 194, "DAP-Lab alap": 197}
    summ, intact = [], []
    for (s, m), g in T.groupby(["set", "method"], sort=False):
        row = {"set": s, "method": m, "n": ns.get(s), "intact": float(g[g.condition == "intact"].accuracy.iloc[0])}
        for c in ["full", "dyad"]:
            v = g[g.condition == c].accuracy.to_numpy()
            if len(v): row.update({f"{c}_mean": float(v.mean()), f"{c}_lo": float(np.percentile(v, 2.5)), f"{c}_hi": float(np.percentile(v, 97.5)), f"{c}_reps": len(v),
                                   f"{c}_p": float(((v >= row["intact"]).sum() + 1) / (len(v) + 1))})
        summ.append(row); intact.append({"set": s, "method": m, "n": ns.get(s), "accuracy": row["intact"]})
    out["allm"] = {"summary": summ, "intact": intact}
    json.dump(clean(json.loads(json.dumps(out, default=float))), open("v4.json", "w"), indent=1); print("allm added", len(summ))
