#!/usr/bin/env python3
"""Corpus composition statistics (report Section 3.1 and Figure 1).

Reads the frozen corpus SQLite index and emits document/chunk counts overall and per
authority class. These are the numbers quoted in Section 3.1 of the report and plotted
in Figure 1.

Usage:
    python evaluation/corpus_stats.py --db <path to chunk_index.sqlite3> --out results/corpus_stats.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def collect(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    total_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    docs_by_class = dict(
        con.execute(
            "SELECT authority_class, COUNT(*) FROM documents GROUP BY authority_class ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    chunks_by_class = dict(
        con.execute(
            "SELECT authority_class, COUNT(*) FROM chunks GROUP BY authority_class ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    chunks_by_regime = dict(
        con.execute(
            "SELECT legal_regime, COUNT(*) FROM chunks GROUP BY legal_regime ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    edges_by_relation = dict(
        con.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation ORDER BY COUNT(*) DESC").fetchall()
    )
    total_edges = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    chunks_by_method = dict(
        con.execute("SELECT chunking_method, COUNT(*) FROM chunks GROUP BY chunking_method "
                    "ORDER BY COUNT(*) DESC").fetchall()
    )
    chunks_by_role = dict(
        con.execute("SELECT corpus_role, COUNT(*) FROM chunks GROUP BY corpus_role "
                    "ORDER BY COUNT(*) DESC").fetchall()
    )
    # Which publishers the corpus was harvested from (Section 3.1 / corpus/CORPUS_BUILD.md).
    docs_by_host = dict(
        con.execute("""
            SELECT CASE
                WHEN source_url LIKE '%legislation.gov.uk%' THEN 'legislation.gov.uk'
                WHEN source_url LIKE '%procurementpathway.civilservice.gov.uk%'
                    THEN 'procurementpathway.civilservice.gov.uk'
                WHEN source_url LIKE '%procurementjourney.scot%' THEN 'procurementjourney.scot'
                WHEN source_url LIKE '%gov.uk%' THEN 'gov.uk (other)'
                WHEN source_url IS NULL OR source_url = '' THEN '(no source_url)'
                ELSE 'other publishers'
            END AS host, COUNT(*) FROM documents GROUP BY host ORDER BY COUNT(*) DESC
        """).fetchall()
    )
    # The build's own provenance record, written into the index at build time.
    index_manifest = {k: json.loads(v) for k, v in
                      con.execute("SELECT key, value FROM index_manifest").fetchall()}

    # Lane split: the two-lane architecture routes primary/secondary legislation to the
    # legislation lane and everything else to the other-evidence lane (Section 4.1).
    legislation_classes = ("PRIMARY_LEGISLATION", "SECONDARY_LEGISLATION")
    q = ",".join("?" * len(legislation_classes))
    docs_legislation = con.execute(
        f"SELECT COUNT(*) FROM documents WHERE authority_class IN ({q})", legislation_classes
    ).fetchone()[0]
    con.close()

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "documents_by_authority_class": docs_by_class,
        "chunks_by_authority_class": chunks_by_class,
        "chunks_by_legal_regime": chunks_by_regime,
        "lane_split_documents": {
            "legislation_lane": docs_legislation,
            "other_evidence_lane": total_docs - docs_legislation,
        },
        "graph_total_edges": total_edges,
        "graph_edges_by_relation": edges_by_relation,
        "chunks_by_chunking_method": chunks_by_method,
        "chunks_by_corpus_role": chunks_by_role,
        "documents_by_source_host": docs_by_host,
        "index_manifest": index_manifest,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True, help="path to chunk_index.sqlite3")
    ap.add_argument("--out", type=Path, default=Path("results/corpus_stats.json"))
    args = ap.parse_args()

    stats = collect(args.db)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
