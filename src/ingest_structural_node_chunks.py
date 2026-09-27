#!/usr/bin/env python3
"""Ingest chunks built directly from a legislation instrument's parsed structural tree.

Supersedes every currently-live chunk for the given document_id(s) with the output of
`chunk_legislation_from_nodes.py` - identity (parent_node_id, citation) comes straight
from the source XML parse, not from an LLM self-report, which is what broke for the
old chunking method (see chunk_legislation_from_nodes.py's docstring).
"""
from __future__ import annotations

import argparse, hashlib, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NORM = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
H = lambda t: hashlib.sha1(NORM(t).encode()).hexdigest()
TAG = "STRUCTURAL_NODE_V1"
MIN_TOKENS = 20

META = {
    "UKPGA_2023_54": dict(authority_class="PRIMARY_LEGISLATION", corpus_role="CORE_PRIMARY_LEGISLATION",
                           legal_regime="PA2023", source_url="https://www.legislation.gov.uk/ukpga/2023/54"),
    "UKSI_2024_692": dict(authority_class="SECONDARY_LEGISLATION", corpus_role="CORE_SECONDARY_LEGISLATION",
                           legal_regime="PR2024", source_url="https://www.legislation.gov.uk/uksi/2024/692"),
    "UKSI_2024_716": dict(authority_class="SECONDARY_LEGISLATION", corpus_role="PA2023_COMMENCEMENT",
                           legal_regime="PA2023", source_url="https://www.legislation.gov.uk/uksi/2024/716"),
    "UKSI_2024_959": dict(authority_class="SECONDARY_LEGISLATION", corpus_role="PA2023_COMMENCEMENT",
                           legal_regime="PA2023", source_url="https://www.legislation.gov.uk/uksi/2024/959"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--chunks", nargs="+", type=Path, required=True, help="output(s) of chunk_legislation_from_nodes.py")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.error("choose --dry-run or --apply")

    con = sqlite3.connect(ROOT / a.db); con.row_factory = sqlite3.Row

    all_rows = []
    supersede_docs = []
    stats = {"kept": 0, "flagged_small": 0, "flagged_dup": 0}
    for f in a.chunks:
        chunks = json.loads(f.read_text(encoding="utf-8"))
        # document_id is the parent_node_id prefix before "__"
        did = chunks[0]["parent_node_id"].split("__", 1)[0]
        meta = META[did]
        supersede_docs.append(did)
        seen_hashes = set()
        for i, c in enumerate(chunks, 1):
            hh = H(c["text"])
            flag = None
            if hh in seen_hashes:
                flag = "duplicate_within_structural_rebuild"; stats["flagged_dup"] += 1
            elif c["est_tokens"] < MIN_TOKENS:
                flag = "too_small_to_retrieve"; stats["flagged_small"] += 1
            seen_hashes.add(hh)
            stats["kept"] = stats.get("kept", 0) + (1 if flag is None else 0)
            base_unit = "regulation" if "reg " in c["citation"] else "section"
            parent_type = "schedule_paragraph" if "schedule-" in c["parent_node_id"] and "paragraph" in c["parent_node_id"] else (
                "schedule" if "schedule-" in c["parent_node_id"] else base_unit)
            embedding_text = f"{c['citation']}\n\n{c['text']}"
            all_rows.append((
                f"{did}__NODEV1__CH_{i:05d}", did, f"{did}__NODEV1", i,
                c["parent_node_id"], parent_type, "LEGISLATION",
                meta["authority_class"], meta["corpus_role"], meta["legal_regime"], "UK",
                c["citation"], None, "[]",
                "", None, "[]", "[]", "[]", "[]",
                c["text"], embedding_text, "[]", meta["source_url"], hh,
                TAG, None, "structural_node_v1", "1.0.0",
                c["char_count"], c["est_tokens"], None, flag,
            ))

    print(f"prepared: {len(all_rows):,} rows across {len(a.chunks)} document(s)")
    print(f"  kept active : {stats.get('kept',0):,}")
    print(f"  flagged     : too_small={stats['flagged_small']:,} dup_within={stats['flagged_dup']:,}")

    if a.dry_run:
        print("\ndry run - nothing written")
        return 0

    superseded_count = 0
    for did in supersede_docs:
        cur = con.execute(
            "UPDATE chunks SET superseded_by=? WHERE document_id=? AND superseded_by IS NULL",
            (TAG, did))
        superseded_count += cur.rowcount
    print(f"superseded {superseded_count} old chunks across {supersede_docs}")

    cols = ("chunk_id,document_id,segment_id,chunk_ordinal,parent_node_id,parent_type,source_kind,"
            "authority_class,corpus_role,legal_regime,jurisdiction,citation,heading,heading_path,"
            "retrieval_title,retrieval_summary,topics,legal_concepts,procurement_stage,"
            "question_intents,text,embedding_text,link_placeholders,source_url,content_sha256,"
            "chunking_method,chunking_model,prompt_version,pipeline_version,char_count,"
            "est_tokens,superseded_by,filtered_out")
    con.executemany(f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({','.join('?'*33)})", all_rows)
    con.executemany(
        "INSERT INTO chunks_fts (chunk_id,text,retrieval_title,retrieval_summary,keywords,citation) "
        "VALUES (?,?,?,?,?,?)",
        [(r[0], r[20], r[14], "", "", r[11]) for r in all_rows if r[32] is None],
    )
    con.commit()

    act = con.execute("SELECT count(*), sum(char_count) FROM chunks "
                       "WHERE superseded_by IS NULL AND filtered_out IS NULL").fetchone()
    print(f"\napplied. active corpus now: {act[0]:,} chunks, {act[1]:,} chars")
    ((ROOT / a.db).parent / "ingest_structural_node_report.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), "documents": supersede_docs,
         "superseded": superseded_count, "inserted": len(all_rows), **stats}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
