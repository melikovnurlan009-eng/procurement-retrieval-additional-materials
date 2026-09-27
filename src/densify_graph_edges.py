#!/usr/bin/env python3
"""Make legal edges traversable by resolving them to the granularity that was chunked.

Problem
-------
Reference resolution works at the granularity the legislation itself uses: a cross-reference
to "paragraph 1(1) of Schedule 4" resolves to `UKPGA_2023_54__schedule-4-paragraph-1-1`.
Chunking, however, happens at provision level, so that node has no chunk. Measured on the
merged corpus, 4,747 of 11,927 legal edges (39.8%) pointed at sub-provision nodes that
exist in the source Akoma Ntoso tree but are not retrievable. The reference was CORRECT and
still unusable, which is a large part of why graph expansion measured as marginal.

Approach
--------
`target_id` is never modified: it is the precise legal reference and it is right. A separate
`retrieval_target_id` column records the nearest ANCESTOR that actually has a chunk, found
by dropping trailing path components (`schedule-4-paragraph-1-1` -> `schedule-4-paragraph-1`).
Traversal uses the retrieval target; citation and evidence keep the exact one. This is the
legal-identity / retrieval-identity separation the rest of the pipeline already relies on.

Rollup is conservative by construction:
* Only ancestors are considered, never siblings or descendants, so an edge is never
  redirected to a provision the source did not point into.
* The document prefix must match, so rollup cannot cross instruments.
* Self-loops after rollup are dropped: an edge from section 12 to section 12(3) carries no
  retrieval information once both collapse to the same chunk.
* Edges already pointing at a chunked node keep that node as their retrieval target.

Usage
-----
    python densify_graph_edges.py --db state/chunk_index_merged.sqlite3 --dry-run
    python densify_graph_edges.py --db state/chunk_index_merged.sqlite3 --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"
LEGAL_RELATIONS = ("REFERENCES", "CROSS_REFERS_TO")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ancestors(node_id: str) -> list[str]:
    """Ancestor node ids, nearest first, by dropping trailing path components."""
    parts = node_id.split("__")
    if len(parts) < 2:
        return []
    doc, tail = parts[0], parts[-1]
    bits = tail.split("-")
    return [f"{doc}__{'-'.join(bits[:k])}" for k in range(len(bits) - 1, 0, -1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not (args.apply or args.dry_run):
        ap.error("choose --dry-run or --apply")

    root = Path(__file__).resolve().parent
    db = args.db if args.db.is_absolute() else root / args.db
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    chunked = {r[0] for r in con.execute(
        "SELECT DISTINCT parent_node_id FROM chunks WHERE parent_node_id IS NOT NULL AND parent_node_id<>''")}
    chunk_ids = {r[0] for r in con.execute("SELECT chunk_id FROM chunks")}
    doc_ids = {r[0] for r in con.execute("SELECT document_id FROM documents")}
    reachable = chunked | chunk_ids | doc_ids

    placeholders = ",".join("?" * len(LEGAL_RELATIONS))
    edges = con.execute(
        f"SELECT edge_id, relation, source_id, target_id FROM edges "
        f"WHERE relation IN ({placeholders})", LEGAL_RELATIONS).fetchall()

    stats = collections.Counter()
    updates: list[tuple[str, str, str]] = []
    src_updates: list[tuple[str, str]] = []
    for e in edges:
        tgt = e["target_id"]
        if tgt in reachable:
            stats["already_reachable"] += 1
            updates.append((tgt, "DIRECT", e["edge_id"]))
            continue
        hit = next((a for a in ancestors(tgt) if a in chunked), None)
        if hit is None:
            stats["unresolvable"] += 1
            continue
        # Source and target collapsing to the same provision carries no information.
        src_node = e["source_id"]
        src_roll = src_node if src_node in chunked else next(
            (a for a in ancestors(src_node) if a in chunked), src_node)
        if src_roll == hit:
            stats["self_loop_dropped"] += 1
            continue
        stats["rolled_up"] += 1
        updates.append((hit, "NEAREST_ANCESTOR_ROLLUP", e["edge_id"]))

    # Second pass: make each edge's SOURCE addressable at the granularity traversal uses.
    for e in edges:
        src = e["source_id"]
        if src in chunked or src in chunk_ids or src in doc_ids:
            src_updates.append((src, e["edge_id"])); stats["source_direct"] += 1
            continue
        hit = next((a for a in ancestors(src) if a in chunked), None)
        if hit is None:
            stats["source_unresolvable"] += 1
            continue
        src_updates.append((hit, e["edge_id"])); stats["source_rolled_up"] += 1

    report = {
        "tool_version": TOOL_VERSION, "generated_at": now_iso(), "db": str(db),
        "legal_relations": list(LEGAL_RELATIONS), "legal_edges": len(edges),
        **{k: v for k, v in sorted(stats.items())},
        "traversable_after": stats["already_reachable"] + stats["rolled_up"],
        "traversable_before": stats["already_reachable"],
        "source_addressable_after": stats["source_direct"] + stats["source_rolled_up"],
        "source_addressable_before": stats["source_direct"],
    }
    print(json.dumps(report, indent=2))

    if args.apply:
        cols = {r[1] for r in con.execute("PRAGMA table_info(edges)")}
        if "retrieval_target_id" not in cols:
            con.execute("ALTER TABLE edges ADD COLUMN retrieval_target_id TEXT")
            con.execute("ALTER TABLE edges ADD COLUMN retrieval_resolution TEXT")
        if "retrieval_source_id" not in cols:
            con.execute("ALTER TABLE edges ADD COLUMN retrieval_source_id TEXT")
        con.executemany(
            "UPDATE edges SET retrieval_target_id=?, retrieval_resolution=? WHERE edge_id=?",
            updates)
        con.executemany("UPDATE edges SET retrieval_source_id=? WHERE edge_id=?", src_updates)
        con.execute("CREATE INDEX IF NOT EXISTS idx_edges_rsource ON edges(retrieval_source_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_edges_rtarget ON edges(retrieval_target_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        con.commit()
        print(f"\napplied {len(updates)} retrieval targets")
        (db.parent / "densify_graph_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
