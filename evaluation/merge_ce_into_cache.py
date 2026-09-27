#!/usr/bin/env python3
"""Stage 3: merge a cross-encoder rerank output (run_ce_rerank.py's output) into an
existing candidate_cache.parquet's ce_score/ce_rank columns, in place. This is what makes
the cache self-contained: after the merge, every downstream analysis reads CE scores from
the parquet and never needs the reranker, the corpus DB or a GPU again.

Run:
    python evaluation/merge_ce_into_cache.py \
        --cache data/candidate_cache.parquet \
        --ce-output data/reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ce-output", type=Path, required=True)
    args = ap.parse_args()

    cache = pd.read_parquet(args.cache)
    ce = json.loads(args.ce_output.read_text())

    ce_lookup = {}
    for sc in ce["scenarios"]:
        sid = sc["scenario_id"]
        for field in ("legislation_lane_top75", "other_lane_top75"):
            for row in sc.get(field, []):
                ce_lookup[(sid, row["lane"], row["chunk_id"])] = (row["ce_score"], row["rerank_position"])

    def lookup(row):
        key = (row["scenario_id"], row["lane"], row["chunk_id"])
        return ce_lookup.get(key, (None, None))

    scores, ranks = zip(*cache.apply(lookup, axis=1)) if len(cache) else ((), ())
    cache["ce_score"] = scores
    cache["ce_rank"] = ranks

    n_matched = cache["ce_rank"].notna().sum()
    print(f"matched CE scores for {n_matched}/{len(cache)} cache rows "
          f"({cache[cache.ce_rank.notna()].scenario_id.nunique()} scenarios)")
    cache.to_parquet(args.cache, index=False)
    print(f"wrote back to {args.cache}")


if __name__ == "__main__":
    main()
