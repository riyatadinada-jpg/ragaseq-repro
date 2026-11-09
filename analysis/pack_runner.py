"""Run Scripts 1-3 unchanged on the packed pitch data (no need for the 2 GB feature folder).
usage: python pack_runner.py <script.py> [script arguments...]
The data_dir argument of the script is ignored; recordings come from --selection and the pack (common.py)."""
import importlib, sys
import pandas as pd
import common as C

def manifest(data_dir, selection=None):
    sel = pd.read_csv(selection, dtype=str)
    return pd.DataFrame([{"raga": C.MD[r]["raag"], "recording_id": r, "pitch": r, "stem": r,
                          "analysis_ready": True, "selected": True} for r in sel.recording_id])

C.R.build_manifest = manifest
mod = importlib.import_module(sys.argv[1].removesuffix(".py"))
sys.argv = [sys.argv[1]] + sys.argv[2:]
mod.main()
