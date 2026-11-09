#!/usr/bin/env bash
# Regenerate every result, table and figure of the paper from the public pitch data, then check them
# against the reported values (analysis/verify_results.py).
#   RAGASEQ_DATA  folder with the packed data made by data_preparation/ (default: ./data)
#   QUICK=1       smoke test with very few shuffles/permutations (shuffle-based values will differ)
#   FROM=N        resume at step N (steps are numbered in the output)
# Full run: about 3 hours on 2 CPU cores. Seed 1234 throughout.
set -euo pipefail
cd "$(dirname "$0")"
export RAGASEQ_DATA="$(cd "${RAGASEQ_DATA:-data}" && pwd)"
if [ "${QUICK:-0}" = 1 ]; then SH=3; SWR=1; PERM=20; EXPP=20; REPS=3; DREPS=3; SREPS=3; TR=2; TR150=5
else SH=1000; SWR=50; PERM=10000; EXPP=2000; REPS=200; DREPS=200; SREPS=200; TR=20; TR150=50; fi
cd analysis && cp ../selections/*.csv .
STEP=0
step() { STEP=$((STEP + 1)); echo; echo "=== [$STEP] $* ==="; }
run() { if [ "$STEP" -ge "${FROM:-1}" ]; then "$@"; fi; }

step "Scripts 1-3 on the development and held-out sets (Tables 2-3, Figures 2-3, S4)"
run python pack_runner.py RagaSeq_10Raga_Analysis_FINAL_Annotated.py . --selection RagaSeq_10x5_Selection.csv --out results_script1_dev --permutations $PERM
run python pack_runner.py RagaSeq_10Raga_Analysis_FINAL_Annotated.py . --selection RagaSeq_TestSet_14x5_Selection.csv --out results_script1_test --permutations $PERM
run cp results_script1_dev/03_pairwise_comparisons.csv dev_pairwise_script1.csv
run python original_ragaseq_recognition.py RagaSeq_10x5_Selection_with_performers.csv results_script1_dev
run python original_ragaseq_recognition.py RagaSeq_TestSet_14x5_Selection.csv results_script1_test
run python pack_runner.py RagaSeq_Sensitivity_Analysis.py . --selection RagaSeq_10x5_Selection.csv --out results_sensitivity --permutations $PERM
run python pack_runner.py RagaSeq_Duration_Recognition.py . --selection RagaSeq_10x5_Selection_with_performers.csv --out results_ragaseqd_dev --baselines --permutations $PERM
run python pack_runner.py RagaSeq_Duration_Recognition.py . --selection RagaSeq_TestSet_14x5_Selection.csv --out results_ragaseqd_test --baselines --permutations $PERM
step "Shuffle controls, development and held-out (Table 4)"; run python run_shuffle.py $SH $SWR
step "30 ragas (Table 5, Appendix A)"; run python build_expanded.py; run python run_expanded.py $EXPP $REPS
step "Balanced draws"; run python run_balanced.py; run python summarize_balanced.py
step "Content-only baseline and audio length (Table 6, Figure 5)"; run python nested_all.py; run python len_mcn.py
step "DAP-Lab and cross-collection (Tables 8-9)"; run python run_dap.py $DREPS; run python run_saraga.py $SREPS
step "Order test on all methods (Table 10)"
run python tdms_shuffle.py $TR150 "CompMusic expanded"; run mv results_expanded/70_order_test_CompMusic.csv results_expanded/70a_150s.csv
run python tdms_shuffle.py $TR "CompMusic long"; run python tdms_shuffle.py $TR "DAP-Lab"
step "Allied ragas, Holm correction, applications (Table 7, Figure 6)"
run python review_analyses.py; run python holm_length.py; run python review2.py; run python practical_demo.py
step "Robustness (Appendix B)"; run python robustness_analyses.py A; run python robustness_analyses.py B; run python robustness_analyses.py C
step "Collect results and draw figures"; run python make_results.py; run python make_v4data.py
run mkdir -p ../figures; run python figures.py; run python fig8.py; run python fig9.py
step "Check against the paper"; run python verify_results.py
