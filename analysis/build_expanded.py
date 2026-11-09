"""Expanded selection: every Hindustani raag in the dataset, one recording per performer per raag
(the recording already used in the paper when there is one, otherwise the longest), recordings with >= 150 s of audio."""
import pandas as pd, numpy as np, common as C
rows = []
for rid, m in C.MD.items():
    if rid not in C.PK: continue
    h = C.PK[rid][3]
    rows.append(dict(m, duration_s=h["full_end"] - h["full_start"]))
df = pd.DataFrame(rows)
print("available", len(df), "short(<150s)", (df.duration_s < 150).sum())
df = df[df.duration_s >= 150]
dev = set(pd.read_csv("RagaSeq_10x5_Selection_with_performers.csv").recording_id)
test = set(pd.read_csv("RagaSeq_TestSet_14x5_Selection.csv").recording_id)
df["pub"] = df.recording_id.isin(dev | test)
sel = df.sort_values(["pub", "duration_s"], ascending=False).groupby(["raag", "performer"], as_index=False).head(1)
devraags = {C.MD[r]["raag"] for r in dev}; testraags = {C.MD[r]["raag"] for r in test}
sel["raag_group"] = np.where(sel.raag.isin(devraags), "development raag",
                    np.where(sel.raag.isin(testraags), "earlier test raag", "new raag"))
sel["in_published_set"] = sel.recording_id.isin(dev | test)
sel = sel.sort_values(["raag", "performer"]).reset_index(drop=True)
sel[["recording_id","raag","raag_id","performer","track","duration_s","raag_group","in_published_set","rel"]].to_csv("RagaSeq_Expanded_Selection.csv", index=False)
print(len(sel), "recordings,", sel.raag.nunique(), "raags,", sel.performer.nunique(), "performers")
print(sel.groupby("raag_group").agg(raags=("raag","nunique"), recs=("recording_id","size")))
print(sel.raag.value_counts().sort_values().to_string())
print("published recordings retained:", sel.in_published_set.sum(), "of", len(dev|test))
