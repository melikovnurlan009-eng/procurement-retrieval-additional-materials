"""Shared configuration and reproducibility-manifest machinery for the analysis harness.

Every analysis script imports AnalysisConfig from here rather than hardcoding paths or
weights, so a later run against a corrected/larger benchmark only needs different
--scenarios/--gold/--split flags, not code changes (per the harness's core requirement).

IMPORTANT: this module does not import chunk_retrieval.py itself, to guarantee every
script imports it from the SAME place (CODE_DIR, added to sys.path by common.py) rather
than risking an accidental import of the stale procurement-kg-rag/chunk_retrieval.py copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------- paths
# Every path resolves through evaluation/paths.py so this package runs from any checkout
# without editing. Override with the CORPUS_DB / BENCHMARK_DIR / CANDIDATE_CACHE
# environment variables; see paths.py for the full list.
from evaluation.paths import (  # noqa: E402
    BENCHMARK_DIR as _BENCH_DIR,
    CE_OUTPUT_BASELINE as _CE_OUTPUT,
    CANDIDATE_CACHE as _CANDIDATE_CACHE,
    CORPUS_DB as _CORPUS_DB,
    GOLD_PATH as _GOLD,
    RESULTS_DIR as _RESULTS_DIR,
    SCENARIOS_PATH as _SCENARIOS,
)

# CODE_DIR is the directory holding the active retriever. It is the ONLY chunk_retrieval.py
# this harness ever imports; common.py puts it on sys.path and assert_active_retriever()
# checks at runtime that the module actually loaded came from here.
CODE_DIR = Path(__file__).resolve().parents[1] / "src"
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = _RESULTS_DIR

DEFAULT_VERSION = "v2b_sum2"
DEFAULT_DB = _CORPUS_DB
DEFAULT_COLLECTION = f"chunks__bge_m3__{DEFAULT_VERSION}"
DEFAULT_BENCH_DIR = _BENCH_DIR
DEFAULT_SCENARIOS = _SCENARIOS
DEFAULT_GOLD = _GOLD
DEFAULT_QRELS = _BENCH_DIR / "qrels_provisional.jsonl"   # optional; not used by the reported results

# Cross-encoder output the harness REUSES rather than recomputing: top-75/lane,
# bge-reranker-v2-m3, the frozen baseline rerank shipped in data/.
DEFAULT_CE_OUTPUT = _CE_OUTPUT

# ---------------------------------------------------------------------------------- weights
# Nominal production defaults, mirrored from chunk_retrieval.py's own module/class
# constants (BM25_WEIGHT, DENSE_WEIGHT, AUTH_BLEND) - kept here ONLY as the default value
# for --alpha/--beta flags; the actual _fuse()/_authority_norm() math is never
# reimplemented here, only re-driven with different weight arguments over cached raw
# scores (see evaluation/common.py's refuse()).
DEFAULT_BETA = 0.40   # BM25 share of the lexical+dense fusion (chunk_retrieval.BM25_WEIGHT)
DEFAULT_ALPHA = 0.30  # authority share of the final blend (chunk_retrieval.AUTH_BLEND)
DEFAULT_CANDIDATES = 300     # per channel, per lane - matches this session's validated eval depth
DEFAULT_RERANK_DEPTH = 75    # matches the reused CE output's own depth
DEFAULT_LANE_BUDGET = (5, 5)  # (K_leg, K_other) - chunk_retrieval.search_two_lanes()'s own default
RRF_K = 60  # conventional RRF constant; not otherwise defined anywhere in this repo

LEGISLATION_CLASSES = ("PRIMARY_LEGISLATION", "SECONDARY_LEGISLATION")

CHUNK_LENGTH_BINS = [(0, 99), (100, 199), (200, 399), (400, 799), (800, float("inf"))]
CHUNK_LENGTH_BIN_LABELS = ["<100", "100-199", "200-399", "400-799", "800+"]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class AnalysisConfig:
    """Resolved configuration for one analysis run, plus everything needed to write a
    full reproducibility manifest (Section 16 of the spec)."""
    scenarios_path: Path = DEFAULT_SCENARIOS
    gold_path: Path = DEFAULT_GOLD
    qrels_path: Path = DEFAULT_QRELS
    split: str = "ALL"  # DEV | TEST | ALL
    db_path: Path = DEFAULT_DB
    collection: str = DEFAULT_COLLECTION
    ce_output_path: Path = DEFAULT_CE_OUTPUT
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    candidates: int = DEFAULT_CANDIDATES
    rerank_depth: int = DEFAULT_RERANK_DEPTH
    lane_budget: tuple = DEFAULT_LANE_BUDGET
    use_graph: bool = True
    use_jurisdiction: bool = True
    keyword_expansion: bool = False
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    random_seed: int = 20260920
    results_dir: Path = RESULTS_ROOT / "current60_exploratory"
    frozen_config_path: Path | None = None  # if set in TEST mode, tuning is refused

    def __post_init__(self):
        for f in ("scenarios_path", "gold_path", "qrels_path", "db_path", "ce_output_path", "results_dir"):
            setattr(self, f, Path(getattr(self, f)))
        if self.split == "TEST" and self.frozen_config_path is None:
            raise ValueError(
                "split=TEST requires --frozen-config pointing at a config.json saved from a "
                "prior DEV run (alpha/beta/lane_budget/etc must come from that file, not be "
                "re-chosen here) - refusing to run TEST without one, per the no-retuning rule."
            )
        if self.frozen_config_path is not None and self.frozen_config_path.exists():
            frozen = json.loads(Path(self.frozen_config_path).read_text())
            for k in ("alpha", "beta", "candidates", "rerank_depth", "lane_budget", "use_graph", "use_jurisdiction"):
                if k in frozen:
                    setattr(self, k, tuple(frozen[k]) if k == "lane_budget" else frozen[k])

    def manifest(self) -> dict[str, Any]:
        import sqlite3
        m: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(CODE_DIR),
            "code_dir": str(CODE_DIR),
            "retriever_module": str(CODE_DIR / "chunk_retrieval.py"),
            "corpus_sqlite_path": str(self.db_path),
            "corpus_sqlite_sha256": sha256_file(self.db_path),
            "qdrant_collection": self.collection,
            "scenarios_path": str(self.scenarios_path),
            "scenarios_sha256": sha256_file(self.scenarios_path),
            "gold_path": str(self.gold_path),
            "gold_sha256": sha256_file(self.gold_path),
            "split": self.split,
            "alpha": self.alpha,
            "beta": self.beta,
            "candidates_per_channel_per_lane": self.candidates,
            "rerank_depth": self.rerank_depth,
            "lane_budget_K_leg_K_other": list(self.lane_budget),
            "use_graph": self.use_graph,
            "use_jurisdiction": self.use_jurisdiction,
            "keyword_expansion": self.keyword_expansion,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "random_seed": self.random_seed,
            "frozen_config_path": str(self.frozen_config_path) if self.frozen_config_path else None,
        }
        if self.db_path.exists():
            con = sqlite3.connect(self.db_path)
            try:
                m["corpus_total_chunks"] = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                m["corpus_total_documents"] = con.execute(
                    "SELECT COUNT(DISTINCT document_id) FROM chunks").fetchone()[0]
            except Exception as e:
                m["corpus_count_error"] = str(e)
            finally:
                con.close()
        return m

    def write_manifest(self, out_dir: Path | None = None) -> Path:
        d = out_dir or self.results_dir
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.json"
        p.write_text(json.dumps(self.manifest(), indent=2))
        return p


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Every analysis script's CLI (Section 17's --scenarios/--gold/--split contract)."""
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    ap.add_argument("--split", choices=["DEV", "TEST", "ALL"], default="ALL")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    ap.add_argument("--rerank-depth", type=int, default=DEFAULT_RERANK_DEPTH)
    ap.add_argument("--lane-budget", type=int, nargs=2, default=list(DEFAULT_LANE_BUDGET),
                     metavar=("K_LEG", "K_OTHER"))
    ap.add_argument("--use-graph", type=int, choices=[0, 1], default=1,
                     help="1 (default) = graph expansion on, matching production. 0 = graph off, "
                          "for the graph ablation cache (Section 15).")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--frozen-config", type=Path, default=None,
                     help="Required when --split TEST: a config.json saved from a prior DEV run. "
                          "TEST mode refuses to run without one (no retuning on TEST).")
    ap.add_argument("--cache", type=Path, default=_CANDIDATE_CACHE,
                     help="Path to a candidate cache built by build_candidate_cache.py "
                          "(default: the frozen cache shipped in data/)")


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        scenarios_path=args.scenarios, gold_path=args.gold, qrels_path=args.qrels,
        split=args.split, db_path=args.db, collection=args.collection,
        alpha=args.alpha, beta=args.beta, candidates=args.candidates,
        rerank_depth=args.rerank_depth, lane_budget=tuple(args.lane_budget),
        results_dir=args.results_dir, frozen_config_path=args.frozen_config,
        use_graph=bool(getattr(args, "use_graph", 1)),
    )
