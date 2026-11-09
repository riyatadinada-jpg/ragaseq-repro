#!/usr/bin/env python3
"""Pack the central 600 s of every Hindustani pitch track into one small file.

Run from the mirdata folder (double-click run_prep.bat). Pure Python, no packages needed.
Writes ragaseq_central600.bin.gz and prep_log.txt next to this script.
Each record: 4-byte header length, JSON header, float32 Hz values, and (if the hop is
irregular) the first time and then time steps, in units of 1e-7 s (int64 + int32 deltas). Only the central 600 s (+1 s margin) is kept, which contains
the central 150 s and 300 s used in the paper.
"""
import gzip, io, json, os, struct, sys, time, zipfile
from array import array

HERE = os.environ.get("RAGASEQ_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
FEAT = os.path.join(HERE, "RagaDataset", "Hindustani", "features")
OUT = os.path.join(HERE, "ragaseq_central600.bin.gz")
LOG = os.path.join(HERE, "prep_log.txt")
HALF = 300.0


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def read_pitch_lines(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".pitch")][0]
            return io.TextIOWrapper(z.open(name), encoding="utf-8").read().splitlines()
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def scalar(path):
    try:
        with open(path, encoding="utf-8") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def main():
    open(LOG, "w").close()
    t_start = time.time()
    items = []
    for root, _, files in os.walk(FEAT):
        for fn in files:
            if fn.endswith(".pitch") or (fn.endswith(".zip") and not fn.startswith("._")):
                items.append(os.path.join(root, fn))
    items.sort()
    log(f"Found {len(items)} pitch files")
    with gzip.open(OUT, "wb", compresslevel=6) as out:
        for n, path in enumerate(items, 1):
            try:
                stem = path[: -len(".pitch")] if path.endswith(".pitch") else path[: -len(".zip")]
                rel = os.path.relpath(stem, FEAT).replace("\\", "/")
                lines = read_pitch_lines(path)
                t = array("d"); hz = array("d")
                for ln in lines:
                    p = ln.split()
                    if len(p) >= 2:
                        t.append(float(p[0])); hz.append(float(p[1]))
                start, end = t[0], t[-1]
                c = (start + end) / 2.0
                lo, hi = c - HALF - 1.0, c + HALF + 1.0
                i0 = next(i for i in range(len(t)) if t[i] >= lo)
                i1 = max(i for i in range(i0, len(t)) if t[i] <= hi) + 1
                ts = t[i0:i1]; hs = array("f", hz[i0:i1])
                hops = [ts[i + 1] - ts[i] for i in range(0, len(ts) - 1, max(1, len(ts) // 2000))]
                hop = sorted(hops)[len(hops) // 2] if hops else 0.0
                regular = all(abs((ts[i] - ts[0]) - i * hop) < 1e-4 for i in range(0, len(ts), max(1, len(ts) // 5000)))
                regular = regular and abs((ts[-1] - ts[0]) - (len(ts) - 1) * hop) < 1e-4
                head = {"rel": rel, "full_start": start, "full_end": end, "n": len(ts),
                        "t0": ts[0], "hop": hop, "regular": regular,
                        "tonicFine": scalar(stem + ".tonicFine"), "tonic": scalar(stem + ".tonic")}
                hb = json.dumps(head).encode("utf-8")
                out.write(struct.pack("<I", len(hb))); out.write(hb)
                out.write(hs.tobytes())
                if not regular:
                    ticks = [round(x * 1e7) for x in ts]
                    head_d = array("q", [ticks[0]]) 
                    out.write(head_d.tobytes())
                    out.write(array("i", [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]).tobytes())
                log(f"{n}/{len(items)} ok {rel[:60]} n={len(ts)} regular={regular}")
            except Exception as e:
                log(f"{n}/{len(items)} ERROR {path}: {e!r}")
    log(f"DONE in {time.time() - t_start:.0f} s; size {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
