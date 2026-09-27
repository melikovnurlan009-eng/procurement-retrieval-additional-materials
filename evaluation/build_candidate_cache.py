#!/usr/bin/env python3
"""Stage 1: build the reusable first-stage candidate cache - THE ONE real retrieval pass
(FTS5 lexical + BGE-M3 dense, via chunk_retrieval.search_two_lanes) that every downstream
analysis reuses instead of re-embedding or re-querying the corpus. Writes one row per
(scenario, lane, candidate) to a parquet file, carrying the raw bm25/dense/authority values
as well as the normalised and fused scores.

Run this FIRST. Every analysis in this repository reads the cache, and where a different
alpha/beta is needed it recomputes fused/blended/final scores from the cached RAW values
through chunk_retrieval's own _fuse()/_authority_norm() (via evaluation.common.refuse_pool)
- never by re-embedding. That is what lets the same-budget ablations in rq1_rq3_same_budget.py
run without a GPU.

Requires the corpus SQLite index and a running Qdrant instance; this is the only stage that
does. See TECHNICAL_APPENDIX.md, "Path A: rebuild from source".

Usage:
    python evaluation/build_candidate_cache.py --candidates 300 --use-graph 0 \
        --db $CORPUS_DB --cache data/candidate_cache.parquet
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from evaluation.config import add_common_args, config_from_args, AnalysisConfig
from evaluation.common import (
    load_scenarios, load_gold, load_qrels, load_ce_output, essential_targets,
    all_gold_chunk_ids, wrong_regime_chunk_ids, category_of, assert_active_retriever,
)
import chunk_retrieval as _cr
from chunk_retrieval import ChunkRetriever


def chunk_static_fields(con: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    """Fields chunk_retrieval's own result rows do NOT carry (token count, corpus_role,
    chunking_method), fetched once per scenario/lane in a single batched query."""
    if not chunk_ids:
        return {}
    q = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT chunk_id, est_tokens, char_count, corpus_role, chunking_method, parent_node_id "
        f"FROM chunks WHERE chunk_id IN ({q})", chunk_ids,
    ).fetchall()
    return {r[0]: {"est_tokens": r[1], "char_count": r[2], "corpus_role": r[3],
                   "chunking_method": r[4], "parent_node_id": r[5]} for r in rows}


def build(cfg: AnalysisConfig, out_path: Path) -> pd.DataFrame:
    assert_active_retriever()
    scenarios = load_scenarios(cfg.scenarios_path, split="ALL")  # cache holds ALL; filter downstream
    gold = load_gold(cfg.gold_path)
    qrels = load_qrels(cfg.qrels_path)
    ce = load_ce_output(cfg.ce_output_path)

    retriever = ChunkRetriever(cfg.db_path, collection=cfg.collection)
    # Instance-level override only (never mutates ChunkRetriever's own class defaults, so
    # this has zero effect on any other caller of the module) - lets the cache reflect a
    # DEV-selected alpha/beta rather than always the class defaults, which build() silently
    # ignored before 2026-09-20 (cfg.alpha/cfg.beta were accepted on the CLI but never
    # actually applied to the retriever instance doing the real retrieval).
    retriever.BM25_WEIGHT = cfg.beta
    retriever.DENSE_WEIGHT = 1.0 - cfg.beta
    retriever.AUTH_BLEND = cfg.alpha
    con = sqlite3.connect(cfg.db_path)

    rows: list[dict] = []
    t0 = time.time()
    for i, sc in enumerate(scenarios, 1):
        sid = sc["scenario_id"]
        g = gold.get(sid, {})
        req_targets = essential_targets(g)
        gold_all = all_gold_chunk_ids(g)
        wrong_regime_ids = wrong_regime_chunk_ids(g)
        ce_rec = ce.get(sid, {})
        ce_by_lane = {
            "legislation": {r["chunk_id"]: r for r in ce_rec.get("legislation_lane_top75", [])},
            "other": {r["chunk_id"]: r for r in ce_rec.get("other_lane_top75", [])},
        }

        # Request a top_k deep enough to capture the FULL scored pool per lane (union of
        # lexical+dense candidates, typically well under 2*candidates after de-dup), so this
        # cache holds every candidate the retriever actually scored, not just its production
        # top-5 output.
        deep_k = cfg.candidates * 2
        leg, oth, trace = retriever.search_two_lanes(
            sc["query"], top_k_legislation=deep_k, top_k_other=deep_k,
            candidates=cfg.candidates, use_graph=cfg.use_graph,
        )

        for lane, lane_rows in (("legislation", leg), ("other", oth)):
            chunk_ids = [r["chunk_id"] for r in lane_rows]
            static = chunk_static_fields(con, chunk_ids)
            ce_map = ce_by_lane[lane]
            targets_flat = {cid for ids in req_targets.values() for cid in ids}
            reqs_by_chunk: dict[str, list[str]] = {}
            for rid, ids in req_targets.items():
                for cid in ids:
                    reqs_by_chunk.setdefault(cid, []).append(rid)

            for pre_rank, r in enumerate(lane_rows, 1):
                cid = r["chunk_id"]
                s = static.get(cid, {})
                ce_row = ce_map.get(cid)
                q = (r.get("citation") or "") + " " + sid
                rows.append({
                    "scenario_id": sid, "split": sc.get("split"), "suite": category_of(sc),
                    "regime_context": sc.get("regime_context"), "regime_confidence": sc.get("regime_confidence"),
                    "lane": lane,
                    "chunk_id": cid, "document_id": r.get("document_id"),
                    "parent_node_id": s.get("parent_node_id"),
                    "citation": r.get("citation"), "retrieval_title": r.get("retrieval_title"),
                    "authority_class": r.get("authority_class"), "legal_regime": r.get("legal_regime"),
                    "jurisdiction": r.get("jurisdiction"), "source_url": r.get("source_url"),
                    "corpus_role": s.get("corpus_role"), "chunking_method": s.get("chunking_method"),
                    "est_tokens": s.get("est_tokens"), "char_count": s.get("char_count"),
                    "bm25_raw": r.get("bm25_score"), "bm25_norm": r.get("bm25_norm"),
                    "dense_raw": r.get("dense_score"), "dense_norm": r.get("dense_norm"),
                    "authority_raw": r.get("authority_weight"), "authority_norm": r.get("authority_norm"),
                    "jurisdiction_weight": r.get("jurisdiction_weight"),
                    "via_graph_only": r.get("via_graph_only"),
                    "fusion_score": r.get("fusion_score"), "final_score": r.get("final_score"),
                    "pre_rerank_rank": pre_rank,
                    "ce_score": (ce_row or {}).get("ce_score"),
                    "ce_rank": (ce_row or {}).get("rerank_position"),
                    "is_gold": cid in gold_all,
                    "is_essential_gold": cid in targets_flat,
                    "requirement_ids_satisfied": json.dumps(reqs_by_chunk.get(cid, [])),
                    "qrel_grade": (qrels.get((sid, cid)) or {}).get("relevance_grade"),
                    "qrel_is_binding": (qrels.get((sid, cid)) or {}).get("is_binding"),
                    "qrel_is_currently_applicable": (qrels.get((sid, cid)) or {}).get("is_currently_applicable"),
                    "is_wrong_regime_gold": cid in wrong_regime_ids,
                })
        if i % 10 == 0 or i == len(scenarios):
            print(f"  [{i}/{len(scenarios)}] {sid} rows={len(rows)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    con.close()
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows ({df['scenario_id'].nunique()} scenarios) to {out_path}", file=sys.stderr)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out_path = args.cache or (cfg.results_dir / "candidate_cache.parquet")
    build(cfg, out_path)
    cfg.write_manifest()


if __name__ == "__main__":
    main()
