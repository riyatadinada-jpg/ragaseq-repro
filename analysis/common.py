"""Shared helpers: plug the packed data into the unchanged RagaSeq scripts."""
import numpy as np
import packdata as P
import RagaSeq_10Raga_Analysis_FINAL_Annotated as R
import RagaSeq_Duration_Recognition as D

MD = P.metadata()
PK = P.load_pack()

# The scripts call R.load_pitch(row["pitch"]) and R.choose_tonic(row); serve both from the pack.
R.load_pitch = lambda rid: (PK[rid][0], PK[rid][1])
R.choose_tonic = lambda row: (float(PK[row["pitch"]][2]), "tonicFine")
D.R = R


class Args:
    def __init__(self, central_seconds=150.0, windows=5, sample_hz=10.0):
        self.central_seconds = central_seconds; self.windows = windows; self.sample_hz = sample_hz


def load(rids, central_seconds=150.0, windows=5):
    a = Args(central_seconds, windows)
    out = []
    for rid in rids:
        row = {"pitch": rid, "raga": MD[rid]["raag"], "recording_id": rid}
        rec = D.load_recording(row, a)
        rec["performer"] = MD[rid]["performer"]
        out.append(rec)
    return out


def dist_matrix(items, fn):
    n = len(items)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = fn(items[i], items[j])
    return M


def same_cross(M, labels):
    n = len(labels)
    iu = np.triu_indices(n, 1)
    same = np.array([labels[i] == labels[j] for i, j in zip(*iu)])
    v = M[iu]
    return v[same], v[~same]
