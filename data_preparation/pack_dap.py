#!/usr/bin/env python3
"""Keep only filename, time and pitch from the DAP-Lab 2D master files (OSF qjkzs).
Double-click run_pack_dap.bat. Pure Python. Writes dap_pitch.csv.gz (filename,time,pitch_cents)."""
import csv, gzip, io, os, zipfile
HERE = os.environ.get("RAGASEQ_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ZIP = os.path.join(HERE, "qjkzs-osfstorage-03_master_files_2d-archive.zip")
OUT = os.path.join(HERE, "dap_pitch.csv.gz")
LOG = os.path.join(HERE, "dap_log.txt")
def log(m):
    open(LOG, "a", encoding="utf-8").write(m + "\n"); print(m, flush=True)
open(LOG, "w").close()
with zipfile.ZipFile(ZIP) as z, gzip.open(OUT, "wt", newline="", encoding="utf-8") as o:
    w = csv.writer(o); w.writerow(["singer_file", "filename", "time", "pitch"])
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    log(f"{len(names)} csv files: " + ", ".join(names))
    for n in names:
        rows = 0
        with z.open(n) as f:
            r = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            head = next(r); it, ip, ifn = head.index("time"), head.index("pitch"), head.index("filename")
            for row in r:
                w.writerow([os.path.basename(n), row[ifn], row[it], row[ip]]); rows += 1
        log(f"{n}: {rows} rows")
log(f"DONE {os.path.getsize(OUT)/1e6:.1f} MB")
