#!/usr/bin/env python3
"""Suppress duplicate ingestions of the same legal instrument.

The defect
----------
Three instruments were ingested twice, through two pipelines that did not know about each
other. The legislation pipeline derives a document id from the URL (`UKPGA_2023_54`) and
parses the Akoma Ntoso tree into provision-level segments. The general web-ingest adapter
hashes the URL into an opaque id and treats the same file as a generic document, producing a
flat block stream. Because the two URLs differ by a `/data.akn` suffix, no id or URL check
caught it.

    legislation.gov.uk/ukpga/2023/54            UKPGA_2023_54   479 chunks, node identity
    legislation.gov.uk/ukpga/2023/54/data.akn   d47c6658...      49 chunks, none

Same statute, two representations. The scraped copy is coarser (up to 3,238 tokens against a
median near 315), carries no `parent_node_id` so cannot participate in the citation graph,
and competes for the same ten result slots.

Approach
--------
Non-destructive. A `superseded_by` column records which canonical document replaces each
duplicate; retrieval excludes superseded chunks. Nothing is deleted, so the decision is
reversible and the raw ingest is untouched.

Canonical selection is by parsed legal identity, not by size: the document whose chunks carry
`parent_node_id` wins, because only that copy can be cited.

Evaluation consequence
----------------------
28 of the 71 duplicate chunks carry relevance judgments, 16 of them relevant and 13 at the
top grade. Those labels are not wrong - the coarse chunks do contain the operative law - but
they refer to units that will no longer be retrievable. Qrels are defined over a collection,
so changing the collection requires re-scoping them. This tool emits `qrels_v1_dedup.jsonl`
with judgments on superseded chunks removed, so that the old and new systems can be compared
on a common ground truth rather than on two different ones.

Usage
-----
    python deduplicate_instruments.py --dry-run
    python deduplicate_instruments.py --apply
"""
from __future__ import annotations

import argparse, collections, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path

TOOL_VERSION = "1.0.0"


def norm(u: str) -> str:
    u = (u or "").lower().split("#")[0].rstrip("/")
    u = re.sub(r"^https?://(www\.)?", "", u)
    return re.sub(r"/(data\.(akn|xml|htm|feed)|contents|made|introduction)$", "", u)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--qrels", default="data/evaluation/qrels_v1.jsonl", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run): ap.error("choose --dry-run or --apply")
    root = Path(__file__).resolve().parent
    con = sqlite3.connect(root / a.db); con.row_factory = sqlite3.Row

    docs = [dict(r) for r in con.execute(
        "SELECT document_id, min(source_url) url, min(source_kind) kind, count(*) n, "
        "sum(CASE WHEN parent_node_id<>'' THEN 1 ELSE 0 END) identified "
        "FROM chunks GROUP BY document_id")]
    groups = collections.defaultdict(list)
    for d in docs:
        if d["url"]: groups[norm(d["url"])].append(d)

    plan, superseded_docs = [], {}
    for url, members in groups.items():
        if len(members) < 2: continue
        # Canonical = the copy with parsed legal identity; size breaks ties.
        members.sort(key=lambda d: (d["identified"] > 0, d["n"]), reverse=True)
        canon, dups = members[0], members[1:]
        for d in dups:
            superseded_docs[d["document_id"]] = canon["document_id"]
            plan.append({"url": url, "canonical": canon["document_id"],
                         "canonical_chunks": canon["n"], "superseded": d["document_id"],
                         "superseded_chunks": d["n"],
                         "superseded_has_identity": bool(d["identified"])})

    ph = ",".join("?" * len(superseded_docs)) or "''"
    dup_chunks = [r[0] for r in con.execute(
        f"SELECT chunk_id FROM chunks WHERE document_id IN ({ph})",
        tuple(superseded_docs))] if superseded_docs else []

    qpath = root / a.qrels
    qrows = [json.loads(l) for l in qpath.open(encoding="utf-8") if l.strip()] if qpath.exists() else []
    affected = [x for x in qrows if x["chunk_id"] in set(dup_chunks)]
    report = {
        "tool_version": TOOL_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "duplicate_groups": len(plan), "superseded_documents": list(superseded_docs),
        "superseded_chunks": len(dup_chunks),
        "plan": plan,
        "qrels_judgments_on_superseded": len(affected),
        "qrels_relevant_lost": sum(1 for x in affected if x["relevance"] >= 2),
        "qrels_queries_touched": len({x["query_id"] for x in affected}),
    }
    print(json.dumps(report, indent=2))

    if a.apply:
        cols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
        if "superseded_by" not in cols:
            con.execute("ALTER TABLE chunks ADD COLUMN superseded_by TEXT")
        con.executemany("UPDATE chunks SET superseded_by=? WHERE document_id=?",
                        [(c, d) for d, c in superseded_docs.items()])
        con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_superseded ON chunks(superseded_by)")
        con.commit()
        keep = [x for x in qrows if x["chunk_id"] not in set(dup_chunks)]
        outq = qpath.with_name("qrels_v1_dedup.jsonl")
        with outq.open("w", encoding="utf-8") as f:
            for x in keep: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with qpath.with_name("qrels_v1_dedup.trec").open("w", encoding="utf-8") as f:
            for x in keep: f.write(f"{x['query_id']} 0 {x['chunk_id']} {x['relevance']}\n")
        (root / "state" / "deduplicate_instruments_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"\napplied: {len(dup_chunks)} chunks marked superseded")
        print(f"re-scoped qrels: {len(qrows)} -> {len(keep)} judgments -> {outq.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
