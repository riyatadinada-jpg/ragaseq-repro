# Reproducibility package

Code for *How Much Does Melodic Order Add to Raga Recognition? A Matched Analysis of Hindustani Vocal Collections*.
One command regenerates every result, table and figure from public pitch data and checks them against the values in the paper.

## Data (public; no audio needed)

| Collection | Source | Files used |
|---|---|---|
| CompMusic Indian Art Music Raga Recognition Dataset (Hindustani) | Zenodo, https://doi.org/10.5281/zenodo.7278506 (CC BY 4.0) | `.pitch`, `.tonicFine` (`.tonic` fallback), `_info_` metadata |
| Saraga 1.5 Hindustani | Zenodo, https://zenodo.org/records/4301737 | pitch, tonic and metadata files |
| DAP-Lab Hindustani vocal dataset | OSF, https://osf.io/qjkzs/ (Ethical Use Declaration) | 2D master files (pitch contours) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.13
pip install -r requirements.txt
```

**Step 0 – prepare the data (once).** Put the downloaded files in `data/` and run the three scripts in `data_preparation/`
(or set `RAGASEQ_DATA` to another folder). They keep only the central 600 s of each pitch track:

| Script | Input in `data/` | Output in `data/` |
|---|---|---|
| `prep_ragaseq.py` | `RagaDataset/Hindustani/features/` and `_info_/` from the CompMusic features archive | `ragaseq_central600.bin.gz` |
| `get_saraga.py` | downloads Saraga pitch/tonic/metadata only | `saraga_central600.bin.gz`, `saraga_metadata.json` |
| `pack_dap.py` | `qjkzs-osfstorage-03_master_files_2d-archive.zip` from OSF | `dap_pitch.csv.gz` |

## Run

```bash
./run_all.sh            # full run, about 3 hours on 2 CPU cores
QUICK=1 ./run_all.sh    # smoke test, about 45 minutes (few shuffles; shuffle-based values will differ)
FROM=7 ./run_all.sh     # resume at a numbered step
```

The last step, `analysis/verify_results.py`, compares the regenerated numbers with those reported in the paper
(`analysis/expected_results.json`): deterministic results must match exactly, shuffle means within 0.01.
Random seed: 1234 throughout. Figures are written to `figures/`, results to `analysis/results_*`.

## Where each table and figure comes from

| Paper | Script(s) | Output |
|---|---|---|
| Table 2, Figures 2–3 | `RagaSeq_10Raga_Analysis_FINAL_Annotated.py` (Script 1) via `pack_runner.py`; `figures.py` | `results_script1_dev/04_statistical_tests.csv` |
| Table 3 | `RagaSeq_Duration_Recognition.py` (Script 3) via `pack_runner.py`; `original_ragaseq_recognition.py` | `results_ragaseqd_*/23_recognition_summary.csv`, `results_script1_*/27_*.csv` |
| Table 4, Figure 4 | `run_shuffle.py`, `run_expanded.py` (Script 4 functions) | `shuffle_replicates.csv`, `results_expanded/47_*.csv` |
| Table 5, Appendix A | `build_expanded.py`, `run_expanded.py` | `results_expanded/41_*.csv`, `49_*.csv` |
| Table 6, Figure 5 | `nested_all.py`, `run_balanced.py`, `summarize_balanced.py`, `len_mcn.py`, `holm_length.py`, `fig8.py` | `results_expanded/56–59*.csv` |
| Table 7 | `review_analyses.py`, `review2.py` | `results_expanded/72–73*.csv`, `76–80*.csv` |
| Tables 8–9 | `run_dap.py`, `run_saraga.py` | `results_dap/`, `results_saraga/` |
| Table 10 | `tdms_shuffle.py` | `results_expanded/70*.csv` |
| Figure 6 | `review_analyses.py`, `review2.py`, `practical_demo.py`, `fig9.py` | `results_expanded/74–75*.csv` |
| Tables A2–A3 | `robustness_analyses.py` | `results_expanded/81–84*.csv` |
| Sensitivity (Supplementary File S4) | `RagaSeq_Sensitivity_Analysis.py` (Script 2) via `pack_runner.py` | `results_sensitivity/` |

`common.py` and `packdata.py` serve the packed data to the scripts unchanged; `make_results.py` and `make_v4data.py`
collect the numbers quoted in the text. Selection files (development, held-out and expanded sets) are in `selections/`.

## License

MIT (see LICENSE).
