import os
"""Read ragaseq_central600.bin.gz (made by prep_ragaseq.py) and serve recordings to the
RagaSeq scripts without the original 2 GB of pitch files.

The packed file keeps the central 600 s (+1 s) of every pitch track. To make the window
positions exactly the same as with the full file, two unvoiced (0 Hz) sentinel frames are
added at the original first and last time stamps: the scripts locate the central 150 s from
t[0] and t[-1] and ignore 0-Hz frames, so the result is unchanged.
"""
import gzip, json, re, struct
from pathlib import Path
import numpy as np

INFO = Path(os.environ.get("RAGASEQ_DATA", "../data")) / "RagaDataset" / "Hindustani" / "_info_"
PACK = Path(os.environ.get("RAGASEQ_DATA", "../data")) / "ragaseq_central600.bin.gz"

# English names used in the paper (dataset names have diacritics)
RAAG_NAMES = {
    "Bāgēśrī": "Bageshri", "Bhūp": "Bhoop", "Dēś": "Desh", "Yaman kalyāṇ": "Yaman Kalyan",
    "Rāgēśrī": "Rageshri", "Mārvā": "Marwa", "Miyān malhār": "Miyan Malhar",
    "Ahira bhairav": "Ahir Bhairav", "Mārūbihāg": "Marubihag", "Śuddh sāraṅg": "Shuddh Sarang",
    "Darbāri": "Darbari Kanada", "Bihāg": "Bihag", "Tōḍī": "Todi", "Alahaiyā bilāval": "Alhaiya Bilawal",
    "Bairāgi": "Bairagi", "Kēdār": "Kedar", "Gauḍ malhār": "Gaud Malhar", "Śrī": "Shree",
    "Madhuvanti": "Madhuvanti", "Puriyā dhanaśrī": "Puriya Dhanashri", "Bilāsakhānī tōḍī": "Bilaskhani Todi",
    "Ābhōgī": "Abhogi", "Basant": "Basant", "Khamāj": "Khamaj", "Jōg": "Jog", "Bhairav": "Bhairav",
    "Mālkauns": "Malkauns", "Madhukauns": "Madhukauns", "Lalit": "Lalit", "Hamsadhvāni": "Hamsadhwani",
}
SAME_PERFORMER = {"Kumar_Gandharv": "Kumar_Gandharva"}   # one singer, two spellings in the dataset


def _clean(rel):
    return re.sub(r"[:&?*\"<>|']", "_", rel)


def metadata():
    """recording_id -> dict(raag_id, raga, performer, rel) for all 300 Hindustani recordings."""
    m = json.load(open(INFO / "path_mbid_ragaid.json"))
    names = json.load(open(INFO / "ragaId_to_ragaName_mapping.json"))
    out = {}
    for rid, v in m.items():
        rel = v["path"].split("audio/")[1]
        perf = rel.split("/")[1]
        out[rid] = {"recording_id": rid, "raag_id": v["ragaid"], "raag": RAAG_NAMES[names[v["ragaid"]]],
                    "performer": SAME_PERFORMER.get(perf, perf).replace("_", " "),
                    "rel": _clean(rel), "track": rel.split("/")[-1].rsplit("_", 1)[0].replace("_", " ")}
    return out


def load_pack(path=PACK):
    """recording_id -> (t, hz, tonic, header). t and hz include the two 0-Hz sentinels."""
    recs = {}
    with gzip.open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b:
                break
            h = json.loads(f.read(struct.unpack("<I", b)[0]))
            n = h["n"]
            hz = np.frombuffer(f.read(4 * n), dtype="<f4").astype(float)
            if h["regular"]:
                t = h["t0"] + h["hop"] * np.arange(n)
            else:
                t0 = np.frombuffer(f.read(8), dtype="<i8")[0]
                d = np.frombuffer(f.read(4 * (n - 1)), dtype="<i4").astype(np.int64)
                t = np.r_[t0, t0 + np.cumsum(d)] / 1e7
            rid = re.search(r"([0-9a-f]{8}-[0-9a-f-]{27})$", h["rel"]).group(1)
            if t[0] > h["full_start"]:
                t = np.r_[h["full_start"], t]; hz = np.r_[0.0, hz]
            if t[-1] < h["full_end"]:
                t = np.r_[t, h["full_end"]]; hz = np.r_[hz, 0.0]
            tonic = h["tonicFine"] if h["tonicFine"] else h["tonic"]
            recs[rid] = (t, hz, tonic, h)
    return recs
