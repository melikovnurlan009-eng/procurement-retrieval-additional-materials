#!/usr/bin/env python3
"""Stage 2: fresh bge-reranker-v2-m3 cross-encoder inference over the top-75/lane
first-stage candidates for every benchmark scenario, producing the reranked_top75_output
JSON that load_ce_output() expects. Reads the candidate_cache.parquet built by
build_candidate_cache.py (run that first) and fetches each candidate's chunk text directly
from the frozen corpus DB, because chunk text is deliberately not stored in the parquet.

The shipped output (data/reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json) was
produced by this script over the 208-scenario benchmark. max_length is 512 by deliberate
choice, not by model limit - see TECHNICAL_APPENDIX.md, "Cross-encoder settings".

Run:
    python evaluation/run_ce_rerank.py \
        --cache data/candidate_cache.parquet \
        --scenarios benchmark/scenarios_all_208.jsonl \
        --db $CORPUS_DB \
        --out data/reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json
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

from evaluation.config import DEFAULT_DB
from evaluation.common import load_scenarios

RERANK_DEPTH = 75
MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def fetch_text(con: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    q = ",".join("?" * len(chunk_ids))
    rows = con.execute(f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({q})", chunk_ids).fetchall()
    return {r[0]: r[1] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rerank-depth", type=int, default=RERANK_DEPTH)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device} ...", file=sys.stderr)
    model = CrossEncoder(MODEL_NAME, max_length=512, device=device)
    sigmoid = torch.nn.Sigmoid()

    cache = pd.read_parquet(args.cache)
    scenarios = load_scenarios(args.scenarios, split="ALL")
    query_by_id = {s["scenario_id"]: s["query"] for s in scenarios}
    con = sqlite3.connect(args.db)

    out_scenarios = []
    t0 = time.time()
    n_pairs_total = 0
    for i, sid in enumerate(sorted(cache.scenario_id.unique()), 1):
        query = query_by_id.get(sid)
        if query is None:
            continue
        rec = {"scenario_id": sid, "query": query}
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)].sort_values("pre_rerank_rank")
            top = sub.head(args.rerank_depth)
            chunk_ids = top["chunk_id"].tolist()
            texts = fetch_text(con, chunk_ids)
            pairs = [(query, texts.get(cid, "") or "") for cid in chunk_ids]
            n_pairs_total += len(pairs)
            if pairs:
                raw = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False,
                                     convert_to_numpy=True)
                scores = sigmoid(torch.tensor(raw)).tolist()
            else:
                scores = []
            scored = list(zip(chunk_ids, scores, top["pre_rerank_rank"].tolist(),
                               top["final_score"].tolist(), top["citation"].tolist(),
                               top["authority_class"].tolist(), top["legal_regime"].tolist(),
                               top["document_id"].tolist(), top["source_url"].tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            rec[field] = [
                {
                    "chunk_id": cid, "lane": lane, "ce_score": float(sc),
                    "rerank_position": pos, "pre_rerank_position": pre_rank,
                    "final_score": final_score, "citation": citation,
                    "authority_class": auth, "legal_regime": regime,
                    "document_id": doc, "source_url": url,
                }
                for pos, (cid, sc, pre_rank, final_score, citation, auth, regime, doc, url)
                in enumerate(scored, 1)
            ]
        out_scenarios.append(rec)
        if i % 10 == 0 or i == cache.scenario_id.nunique():
            print(f"  [{i}] {sid} pairs_so_far={n_pairs_total} ({time.time()-t0:.0f}s)", file=sys.stderr)

    con.close()
    out = {
        "version": "final150_2026-09-20", "reranker": MODEL_NAME, "rerank_depth": args.rerank_depth,
        "note": "Fresh CE inference over ALL 150 fixed-benchmark queries (60 current60 + 90 faq90) "
                "against the post-audit corrected gold, top-75/lane candidates from candidate_cache.parquet "
                "(candidates=300/channel/lane). Replaces the earlier 60-query-only CE output.",
        "scenarios": out_scenarios,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out_scenarios)} scenarios, {n_pairs_total} total CE pairs, to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
