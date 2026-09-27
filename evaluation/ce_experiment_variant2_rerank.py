#!/usr/bin/env python3
"""Variant 2 of the cross-encoder experiment: metadata-enriched cross-encoder input.

Instead of feeding the reranker the chunk's raw text alone (the baseline), each candidate is
presented as

    citation. retrieval_title. retrieval_summary. authority_class. legal_regime. <raw text>

No gold feature is used anywhere: every field above is corpus metadata that is available at
real inference time. Everything else is held fixed against the baseline - identical
first-stage candidate pool, identical model, identical rerank depth of 75 per lane,
identical max_length of 512.

Produces the variant-2 CE output JSON in the same schema as run_ce_rerank.py, which
ce_experiment_metrics2.py and ce_metrics_audit.py then read.

    python evaluation/ce_experiment_variant2_rerank.py --split DEV \
        --db $CORPUS_DB --out data/variant2_ce_output_DEV.json

Requires the corpus SQLite index (for chunk text and metadata) and the reranker model.
The two shipped outputs, data/variant2_ce_output_{DEV,TEST}.json, came from this script.
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
import torch
from sentence_transformers import CrossEncoder

from evaluation.common import load_scenarios
from evaluation.paths import CANDIDATE_CACHE, DATA_DIR, SCENARIOS_PATH, require, require_corpus_db

RERANK_DEPTH = 75
MAX_LENGTH = 512   # deliberate; see TECHNICAL_APPENDIX.md, "Cross-encoder settings"


def fetch_meta(con: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    q = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT chunk_id, text, citation, retrieval_title, retrieval_summary, authority_class, "
        f"legal_regime FROM chunks WHERE chunk_id IN ({q})",
        chunk_ids,
    ).fetchall()
    return {
        r[0]: {"text": r[1], "citation": r[2], "title": r[3], "summary": r[4],
               "auth": r[5], "regime": r[6]}
        for r in rows
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["DEV", "TEST", "ALL"], default="DEV")
    ap.add_argument("--cache", type=Path, default=CANDIDATE_CACHE)
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    ap.add_argument("--db", type=Path, default=None, help="corpus SQLite index (default: $CORPUS_DB)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rerank-depth", type=int, default=RERANK_DEPTH)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    db = args.db if args.db else require_corpus_db()
    out = args.out or DATA_DIR / f"variant2_ce_output_{args.split}.json"
    cache = pd.read_parquet(require(args.cache, "candidate cache"))
    scen_by_id = {s["scenario_id"]: s for s in load_scenarios(args.scenarios, args.split)}
    con = sqlite3.connect(db)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading BAAI/bge-reranker-v2-m3 on {device}...", file=sys.stderr)
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=MAX_LENGTH, device=device)
    sigmoid = torch.nn.Sigmoid()

    out_scenarios = []
    t0 = time.time()
    for i, sid in enumerate(scen_by_id, 1):
        query = scen_by_id[sid].get("query") or scen_by_id[sid].get("scenario_text", "")
        rec = {"scenario_id": sid, "query": query}
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            sub = (cache[(cache.scenario_id == sid) & (cache.lane == lane)]
                   .sort_values("pre_rerank_rank").head(args.rerank_depth))
            chunk_ids = sub.chunk_id.tolist()
            meta = fetch_meta(con, chunk_ids)
            pairs = []
            for cid in chunk_ids:
                m = meta.get(cid, {})
                pairs.append((query, f"{m.get('citation') or ''}. {m.get('title') or ''}. "
                                     f"{m.get('summary') or ''}. {m.get('auth') or ''}. "
                                     f"{m.get('regime') or ''}. {m.get('text') or ''}"))
            if pairs:
                raw = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
                scores = sigmoid(torch.tensor(raw)).tolist()
            else:
                scores = []
            scored = sorted(zip(chunk_ids, scores, sub["pre_rerank_rank"].tolist()), key=lambda x: -x[1])
            rec[field] = [
                {"chunk_id": cid, "ce_score": float(sc), "rerank_position": pos, "pre_rerank_position": pre}
                for pos, (cid, sc, pre) in enumerate(scored, 1)
            ]
        out_scenarios.append(rec)
        if i % 10 == 0 or i == len(scen_by_id):
            print(f"  [{i}/{len(scen_by_id)}] {sid} ({time.time()-t0:.0f}s)", file=sys.stderr)
    con.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": f"variant2_metadata_enriched_{args.split}",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "rerank_depth": args.rerank_depth,
        "max_length": MAX_LENGTH,
        "input_format": "citation. retrieval_title. retrieval_summary. authority_class. legal_regime. raw_text",
        "scenarios": out_scenarios,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
