#!/usr/bin/env python3
"""Single place where every path in this package is resolved.

All paths resolve relative to the repository root (the parent of this file's directory),
so the package runs from any checkout location without editing. Each can be overridden
with an environment variable, which is how you point the harness at a corpus database or
a rebuilt cache that lives outside the repository.

    REPO_ROOT       repository root                       (default: parent of evaluation/)
    CORPUS_DB       frozen corpus SQLite index            (default: corpus/chunk_index.sqlite3)
    CANDIDATE_CACHE first-stage candidate cache (parquet) (default: data/candidate_cache.parquet)
    BENCHMARK_DIR   scenarios + gold evidence             (default: benchmark/)
    RESULTS_DIR     where analyses write their output     (default: results/)
    FIGURES_DIR     where figures are written             (default: figures/)

The corpus database is NOT shipped in this repository (it is 165 MB and is rebuildable
from the sources listed in corpus/CORPUS_BUILD.md). Scripts that need it will tell you
to set CORPUS_DB. Scripts that only need the frozen candidate cache run out of the box.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[1]))

BENCHMARK_DIR = Path(os.environ.get("BENCHMARK_DIR", REPO_ROOT / "benchmark"))
SCENARIOS_PATH = BENCHMARK_DIR / "scenarios_all_208.jsonl"
GOLD_PATH = BENCHMARK_DIR / "gold_evidence_208.jsonl"

DATA_DIR = REPO_ROOT / "data"
CANDIDATE_CACHE = Path(os.environ.get("CANDIDATE_CACHE", DATA_DIR / "candidate_cache.parquet"))
CE_OUTPUT_BASELINE = DATA_DIR / "reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json"
CE_OUTPUT_VARIANT2_DEV = DATA_DIR / "variant2_ce_output_DEV.json"
CE_OUTPUT_VARIANT2_TEST = DATA_DIR / "variant2_ce_output_TEST.json"

CORPUS_DB = Path(os.environ.get("CORPUS_DB", REPO_ROOT / "corpus" / "chunk_index.sqlite3"))

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))
CONFIG_DIR = REPO_ROOT / "config"


def require_corpus_db() -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not CORPUS_DB.exists():
        raise SystemExit(
            f"Corpus database not found at {CORPUS_DB}.\n"
            "The 165 MB corpus index is not shipped in this repository. Either rebuild it "
            "(see corpus/CORPUS_BUILD.md) or point the harness at an existing copy:\n"
            "    export CORPUS_DB=/path/to/chunk_index.sqlite3"
        )
    return CORPUS_DB


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{what} not found at {path}")
    return path
