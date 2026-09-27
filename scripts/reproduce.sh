#!/usr/bin/env bash
# Reproduce every reported result from the artifacts shipped in this repository.
#
#     bash scripts/reproduce.sh
#
# Needs only requirements.txt: no GPU, no Qdrant, no corpus database, no model downloads.
# Two steps below do use the corpus database if CORPUS_DB points at one; both skip cleanly
# if it does not, and their outputs are shipped either way.
#
# Rebuilding the artifacts themselves (first-stage retrieval and cross-encoder inference) is
# a separate route that does need the corpus. See TECHNICAL_APPENDIX.md, "Path A".

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
failed=()

run() {
    local label="$1"; shift
    printf '\n\033[1m=== %s\033[0m\n' "$label"
    if "$@"; then
        printf '\033[32m    ok\033[0m\n'
    else
        printf '\033[31m    FAILED: %s\033[0m\n' "$label"
        failed+=("$label")
    fi
}

if [ -n "${CORPUS_DB:-}" ] && [ -f "${CORPUS_DB}" ]; then
    echo "Corpus database: ${CORPUS_DB}"
    run "Corpus composition (Figure 1 input)" \
        "$PY" evaluation/corpus_stats.py --db "$CORPUS_DB" --out results/corpus_stats.json
else
    echo "CORPUS_DB not set or not found - using the shipped results/corpus_stats.json."
    echo "See corpus/CORPUS_BUILD.md to obtain and verify the index."
fi

# The headline check: 32 reported numbers, recomputed and compared.
run "Verify reported numbers"            "$PY" evaluation/verify_reported_numbers.py

# Analysis tables. ce_experiment_metrics must precede ce_experiment_metrics2, which reads
# summaries_v1_v3.json from it; everything else is independent.
run "Metric audit (final performance)"   "$PY" evaluation/ce_metrics_audit.py
run "Strict dual-evidence coverage"      "$PY" evaluation/dual_evidence_strict.py
run "Score and signal analysis"          "$PY" evaluation/score_signal_analysis.py
run "CE variants 1 and 3"                "$PY" evaluation/ce_experiment_metrics.py
run "CE variants 2 and 4, comparison"    "$PY" evaluation/ce_experiment_metrics2.py
run "RQ1/RQ3 same-budget + statistics"   "$PY" evaluation/rq1_rq3_same_budget.py

# Figures.
run "Report figures"                     "$PY" evaluation/make_report_figures.py
run "Signal diagnostic figures"          "$PY" evaluation/score_signal_figures.py

printf '\n\033[1m=== Summary\033[0m\n'
if [ ${#failed[@]} -eq 0 ]; then
    echo "All steps completed."
    echo
    echo "Read results/verification_report.csv for the number-by-number comparison"
    echo "against the report. Figures are in figures/."
else
    printf '\033[31m%d step(s) failed:\033[0m\n' "${#failed[@]}"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi
