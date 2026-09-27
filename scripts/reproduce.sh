#!/usr/bin/env bash
# Reproduce the reported results, part by part.
#
#     bash scripts/reproduce.sh              # run all nine parts
#     bash scripts/reproduce.sh --list       # what the parts are
#     bash scripts/reproduce.sh performance  # run one part
#     bash scripts/reproduce.sh 3 5 7        # run parts by number
#
# This is a thin wrapper around evaluation/reproduce.py, which is where the parts and their
# checks are defined. Each part re-runs the scripts that produce its outputs and then checks
# those outputs against the values the report prints; a part passes only when every one of
# its checks passes. Exit status is 0 only if every selected part passed.
#
# Needs only requirements.txt: no GPU, no Qdrant, no model downloads. Part 1 regenerates its
# numbers from the corpus database if CORPUS_DB points at one, and otherwise checks the
# shipped copy of its output without regenerating it, which it says in the output.

set -uo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" evaluation/reproduce.py "${@:---all}"
