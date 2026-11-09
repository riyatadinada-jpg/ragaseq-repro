"""Compare regenerated results with the values reported in the paper (expected_results.json).
Deterministic results must match exactly; shuffle means within 0.01 (they depend on the number of replicates)."""
import json, sys, pandas as pd
E = json.load(open("expected_results.json")); bad = []; ok = 0
def chk(name, got, exp, tol=1e-4):
    global ok
    if isinstance(exp, list):
        for g, e in zip(got, exp): chk(name, g, e, tol); return
    if abs(float(got) - float(exp)) <= tol: ok += 1
    else: bad.append(f"{name}: got {got}, expected {exp}")
r = pd.read_csv("results_expanded/41_recognition_summary.csv")
for m, c in zip(r.method, r.correct): chk("Table 5 " + m, c, E["table5"][m.split(" ")[0]])
n = json.load(open("nested.json"))
for x in n["nested"]: chk("Table 6 " + x["set"], [x["acc_k1"], x["acc_k3"]], E["table6"][x["set"]])
l = pd.read_csv("results_expanded/59_content_vs_order_by_length.csv")
for s, a, b in zip(l.seconds, l.k1, l.k3): chk(f"Length {s}s", [a, b], E["length"][str(s)])
for x in json.load(open("review_analyses.json"))["allied"]:
    if x["group"].startswith("All"): chk(f"Table 7 pooled {x['seconds']}s", [x["k1"], x["k3"]], E["allied_pooled"][str(x["seconds"])])
d = pd.read_csv("results_dap/62_dap_recognition.csv")
for f, m, c in zip(d.form, d.method, d.correct): chk(f"Table 8 {f} {m}", c, E["dap"][f"{f}|{m}"])
s = pd.read_csv("results_saraga/52_saraga_recognition.csv")
for m, c in zip(s.method, s.correct): chk(f"Table 9 Saraga {m}", c, E["saraga"][m])
for x in json.load(open("v4.json"))["allm"]["summary"]:
    k = f"{x['set']}|{x['method']}"; chk("Table 10 intact " + k, x["intact"], E["order_test_intact"][k])
    if "full_mean" in x and k in E["order_test_full_mean"]: chk("Table 10 shuffled " + k, x["full_mean"], E["order_test_full_mean"][k], 0.01)
g = pd.read_csv("results_expanded/82_gee_categorical.csv"); chk("Table A3 OR order x 600 s", g[g.term == "L600:order"].OR.iloc[0], E["gee_L600_order_OR"], 1e-3)
b = pd.read_csv("results_expanded/83_section_control.csv")
for c, x, y in zip(b.condition, b.k1, b.k3): chk("Table A3 " + c, [x, y], E["section"][c])
f = pd.read_csv("results_expanded/84_frontend_robustness.csv")
for v, x, y in zip(f.variant, f["600s_long_k1"], f["600s_long_k3"]): chk("Table A2 " + v, [x, y], E["frontend_600"][v])
R = json.load(open("results.json"))
for k, v in R["shuffle"].items(): chk("Table 4 full-shuffle mean " + k, v["RagaSeq-D accuracy"]["full_shuffle"]["mean"], E["shuffle_full_mean"][k], 0.01)
t2 = pd.read_csv("results_script1_dev/04_statistical_tests.csv")
chk("Table 2 SW", [t2.same_mean[0], t2.cross_mean[0]], [E["table2"]["sw_same"], E["table2"]["sw_cross"]], 1e-3)
chk("Table 2 JSD", [t2.same_mean[1], t2.cross_mean[1]], [E["table2"]["jsd_same"], E["table2"]["jsd_cross"]], 1e-3)
for setn, folder, f1 in [("dev", "results_ragaseqd_dev", "results_script1_dev"), ("test", "results_ragaseqd_test", "results_script1_test")]:
    t = pd.read_csv(folder + "/23_recognition_summary.csv")
    for m, c in zip(t.method, t.correct): chk(f"Table 3 {setn} {m}", c, E["table3"][setn][m.split(" ")[0]])
    o = pd.read_csv(f1 + "/27_original_ragaseq_recognition.csv"); chk(f"Table 3 {setn} original", o.correct[0], E["table3"][setn]["Original"])
print(f"{ok} values match the paper.")
if bad: print(f"{len(bad)} differ:"); [print("  " + b) for b in bad]; sys.exit(1)
