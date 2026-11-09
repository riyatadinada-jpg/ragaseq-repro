#!/usr/bin/env python3
"""Fetch only the files RagaSeq needs from Saraga 1.5 Hindustani (Zenodo record 4301737).

Double-click run_saraga.bat (in the same folder). Pure Python, no packages needed.
Instead of downloading the whole 4.1 GB zip (mostly audio), this reads the zip's table of contents
over the internet and downloads only the pitch (.pitch.txt), tonic (.ctonic.txt) and metadata (.json)
files, several at a time, checking each file's CRC. It writes:
  saraga_central600.bin.gz   central 600 s (+1 s) of every pitch track
  saraga_metadata.json       every track's metadata (raag, artists, recording MBID)
  saraga_log.txt             progress log
Re-running skips files already fetched (kept in the saraga_parts folder).
"""
import gzip, io, json, os, struct, sys, time, urllib.request, zlib
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.environ.get("RAGASEQ_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
URL = "https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip?download=1"
PARTS = os.path.join(HERE, "saraga_parts")
OUT = os.path.join(HERE, "saraga_central600.bin.gz")
META = os.path.join(HERE, "saraga_metadata.json")
LOG = os.path.join(HERE, "saraga_log.txt")
HALF = 300.0
WANT = (".pitch.txt", ".ctonic.txt", ".json")


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")
    print(m, flush=True)


_final_url = None


def rng(start, end):
    """Bytes start..end inclusive, with retries."""
    global _final_url
    for attempt in range(30):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "ragaseq-fetch", "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"server did not honour range request (status {r.status})")
                data = r.read()
            if len(data) != end - start + 1:
                raise RuntimeError("short read")
            return data
        except RuntimeError as e:
            if "range" in str(e):
                raise
            time.sleep(min(60, 5 * (attempt + 1)))
        except Exception:
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"failed to fetch bytes {start}-{end}")


def size():
    req = urllib.request.Request(URL, headers={"User-Agent": "ragaseq-fetch", "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        cr = r.headers.get("Content-Range", "")
        return int(cr.split("/")[-1])


def central_directory(total, get=rng):
    tail = get(max(0, total - 70000), total - 1)
    base = max(0, total - 70000)
    e = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[e + 12:e + 20])
    n_entries = struct.unpack("<H", tail[e + 10:e + 12])[0]
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_entries == 0xFFFF:  # zip64
        loc = tail.rfind(b"PK\x06\x07")
        z64off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        rec = tail[z64off - base:] if z64off >= base else get(z64off, z64off + 55)
        n_entries, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
    cd = get(cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, crc, csize, usize, nlen, xlen, clen, off) = (struct.unpack("<H", cd[p + 10:p + 12])[0],
            *struct.unpack("<III", cd[p + 16:p + 28]), *struct.unpack("<HHH", cd[p + 28:p + 34]), struct.unpack("<I", cd[p + 42:p + 46])[0])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[q:q + 4]); body = extra[q + 4:q + 4 + hl]; r = 0
            if hid == 1:
                if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
                if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
                if off == 0xFFFFFFFF: off = struct.unpack("<Q", body[r:r + 8])[0]; r += 8
            q += 4 + hl
        out.append({"name": name, "method": method, "crc": crc, "csize": csize, "usize": usize, "off": off})
        p += 46 + nlen + xlen + clen
    return out


def fetch_member(m, get=rng):
    head = get(m["off"], m["off"] + 29)
    nlen, xlen = struct.unpack("<HH", head[26:30])
    start = m["off"] + 30 + nlen + xlen
    raw = get(start, start + m["csize"] - 1) if m["csize"] else b""
    data = zlib.decompress(raw, -15) if m["method"] == 8 else raw
    if zlib.crc32(data) & 0xFFFFFFFF != m["crc"]:
        raise RuntimeError("CRC mismatch for " + m["name"])
    return data


def pack(files):
    """files: dict name -> bytes for the wanted members."""
    meta = {}
    pitches = sorted(n for n in files if n.endswith(".pitch.txt"))
    log(f"Packing {len(pitches)} pitch files")
    with gzip.open(OUT, "wb", compresslevel=6) as out:
        for k, pn in enumerate(pitches, 1):
            stem = pn[: -len(".pitch.txt")]
            try:
                md = json.loads(files[stem + ".json"].decode("utf-8")) if stem + ".json" in files else {}
                ton = float(files[stem + ".ctonic.txt"].decode().split()[0]) if stem + ".ctonic.txt" in files else None
                t = array("d"); hz = array("d")
                for ln in files[pn].decode("utf-8").splitlines():
                    p = ln.split()
                    if len(p) >= 2:
                        t.append(float(p[0])); hz.append(float(p[1]))
                start, end = t[0], t[-1]; c = (start + end) / 2
                lo, hi = c - HALF - 1, c + HALF + 1
                i0 = next(i for i in range(len(t)) if t[i] >= lo)
                i1 = max(i for i in range(i0, len(t)) if t[i] <= hi) + 1
                ts = t[i0:i1]; hs = array("f", hz[i0:i1])
                head = {"rel": stem, "mbid": md.get("mbid"), "full_start": start, "full_end": end, "n": len(ts),
                        "t0": ts[0], "hop": 0.0, "regular": False, "tonicFine": ton, "tonic": ton}
                hb = json.dumps(head).encode("utf-8")
                out.write(struct.pack("<I", len(hb))); out.write(hb); out.write(hs.tobytes())
                ticks = [round(x * 1e7) for x in ts]
                out.write(array("q", [ticks[0]]).tobytes())
                out.write(array("i", [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]).tobytes())
                meta[stem] = md
            except Exception as e:
                log(f"  ERROR packing {stem}: {e!r}")
    json.dump(meta, open(META, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"DONE: {len(meta)} tracks packed; {os.path.getsize(OUT) / 1e6:.1f} MB")


def main():
    open(LOG, "w").close()
    os.makedirs(PARTS, exist_ok=True)
    total = size(); log(f"Zip on Zenodo: {total / 1e9:.2f} GB; reading its table of contents ...")
    members = [m for m in central_directory(total) if m["name"].endswith(WANT) and "__MACOSX" not in m["name"] and "/._" not in m["name"]]
    # keep only per-track files: those whose stem also has a .pitch.txt
    stems = {m["name"][:-len(".pitch.txt")] for m in members if m["name"].endswith(".pitch.txt")}
    members = [m for m in members if any(m["name"] == s + ext for s in stems for ext in WANT)]
    mb = sum(m["csize"] for m in members) / 1e6
    log(f"Need {len(members)} files ({mb:.0f} MB) for {len(stems)} tracks instead of the whole zip")
    files, todo = {}, []
    for m in members:
        p = os.path.join(PARTS, f"{m['crc']:08x}_{m['usize']}")
        if os.path.exists(p) and os.path.getsize(p) == m["usize"]:
            files[m["name"]] = open(p, "rb").read()
        else:
            todo.append((m, p))
    log(f"{len(files)} already fetched; fetching {len(todo)} with 6 parallel connections ...")
    done_mb, t0 = 0.0, time.time()
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(fetch_member, m): (m, p) for m, p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            m, p = futs[f]
            try:
                data = f.result(); open(p, "wb").write(data); files[m["name"]] = data
                done_mb += m["csize"] / 1e6
            except Exception as e:
                log(f"  FAILED {m['name']}: {e!r} (run again to retry)")
            if i % 10 == 0 or i == len(todo):
                rate = done_mb / max(1, time.time() - t0)
                log(f"  {i}/{len(todo)} files, {done_mb:.0f} of {mb:.0f} MB, {rate * 1000:.0f} KB/s")
    missing = [m["name"] for m in members if m["name"] not in files]
    if missing:
        log(f"{len(missing)} files still missing; run again to resume. Packing what is available ...")
    pack(files)


if __name__ == "__main__":
    main()
