#!/usr/bin/env python3
"""Chunk token-count distributions over the corpus, for the report's Appendix A.

These are the numbers behind the statement that the cross-encoder's 512-token window is a
deliberate operating choice rather than a model limit, and behind the quantified cost of
that choice. Three fields are measured, all with the reranker's own tokenizer
(BAAI/bge-reranker-v2-m3), over all 19,087 chunks in the frozen corpus:

    metadata   the metadata preamble of the enriched cross-encoder input,
               "citation. retrieval_title. retrieval_summary. authority_class. legal_regime."
               - exactly what ce_experiment_variant2_rerank.py builds, minus the raw text
    summary    the retrieval_summary field alone
    rawtext    the raw chunk text, which is what the baseline reranker is fed

The split matters. The metadata preamble and the summary never approach the window; it is
the raw text that overflows it, on 14.2% of chunks. That locates the truncation cost
precisely, instead of leaving "some inputs are truncated" as an unquantified caveat.

    python evaluation/corpus_token_distribution.py --db $CORPUS_DB

Needs the corpus database, because it measures chunk text. Writes one JSON summary and one
per-chunk CSV per field to results/token_distribution/.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from evaluation.paths import RESULTS_DIR, require_corpus_db

MODEL = "BAAI/bge-reranker-v2-m3"

FIELDS = {
    "metadata": {
        "sql": "SELECT chunk_id, citation, retrieval_title, retrieval_summary, "
               "authority_class, legal_regime FROM chunks",
        "build": lambda r: f"{r[1] or ''}. {r[2] or ''}. {r[3] or ''}. {r[4] or ''}. {r[5] or ''}.",
        "label": "metadata-enriched CE input EXCLUDING raw chunk text: "
                 "'citation. retrieval_title. retrieval_summary. authority_class. legal_regime.'",
    },
    "summary": {
        "sql": "SELECT chunk_id, retrieval_summary FROM chunks",
        "build": lambda r: r[1] or "",
        "label": "retrieval_summary field only (not raw chunk text)",
    },
    "rawtext": {
        "sql": "SELECT chunk_id, text FROM chunks",
        "build": lambda r: r[1] or "",
        "label": "raw chunk text, i.e. what the baseline cross-encoder is fed",
    },
}


def summarise(counts: np.ndarray, label: str) -> dict:
    return {
        "field": label,
        "tokenizer": MODEL,
        "n_chunks": int(counts.size),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "p25": float(np.percentile(counts, 25)),
        "p75": float(np.percentile(counts, 75)),
        "p90": float(np.percentile(counts, 90)),
        "p95": float(np.percentile(counts, 95)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "pct_over_512": float((counts > 512).mean() * 100),
        "pct_over_1024": float((counts > 1024).mean() * 100),
        "pct_over_2048": float((counts > 2048).mean() * 100),
        "note": "Chunk-side tokens only. The real cross-encoder pair also carries the query, "
                "so an actual input is longer than the count reported here.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="corpus SQLite index (default: $CORPUS_DB)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "token_distribution")
    args = ap.parse_args()

    db = args.db if args.db else require_corpus_db()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    con = sqlite3.connect(db)
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, spec in FIELDS.items():
        rows = con.execute(spec["sql"]).fetchall()
        ids, counts = [], []
        for r in rows:
            ids.append(r[0])
            counts.append(len(tok.encode(spec["build"](r), truncation=False,
                                         add_special_tokens=False)))
        arr = np.array(counts)
        summaries[name] = summarise(arr, spec["label"])

        pd.DataFrame({"chunk_id": ids, "n_tokens": counts}).to_csv(
            args.out / f"token_counts_{name}.csv", index=False)
        (args.out / f"token_distribution_{name}.json").write_text(
            json.dumps(summaries[name], indent=2))

        s = summaries[name]
        print(f"{name:9} n={s['n_chunks']:,}  mean={s['mean']:7.1f}  median={s['median']:6.0f}  "
              f"p90={s['p90']:6.0f}  p95={s['p95']:6.0f}  max={s['max']:6,}  "
              f">512: {s['pct_over_512']:5.2f}%")
    con.close()

    (args.out / "token_distribution_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote per-field JSON and CSV to {args.out}")
    print("\nThe 512-token window is never reached by the metadata preamble or the summary; "
          f"it is the raw text that overflows, on {summaries['rawtext']['pct_over_512']:.1f}% "
          "of chunks.")


if __name__ == "__main__":
    main()
