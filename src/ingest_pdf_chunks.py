#!/usr/bin/env python3
"""Ingest the re-chunked PDF corpus, retiring what it replaces.

The PDF lane measured 88% INCOMPLETE - the worst of any source, and half the corpus text -
because its chunks were page-sized blobs with no internal structure. This replaces them with
10,028 chunks emitted from page text by the model and verified two ways (per-chunk shingle
fidelity for fabrication, source recall for omission).

Duplicate handling, measured before writing anything:
  2,233  old PDF chunks superseded - the material being replaced
    169  exact duplicates inside the new set, 164 of them across documents where the same
         guidance text is reissued in several PDFs; first occurrence kept, rest flagged
     40  exact duplicates of text already in the corpus from a non-PDF source; flagged, since
         the surviving copy came through a lane that preserved more structure

Every write is reversible: retirement sets `superseded_by`, duplicates set `filtered_out`, and
inserted rows carry `chunking_method='LLM_PDF_TEXT_V2'` so they can be removed as a set.
"""
from __future__ import annotations

import argparse, collections, csv, glob, hashlib, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NORM = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
H = lambda t: hashlib.sha1(NORM(t).encode()).hexdigest()
TAG = "LLM_PDF_TEXT_V2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--chunks-dir", default="data/pdf_chunks_mini", type=Path)
    ap.add_argument("--urls", default="evaluation/pdf_extraction_samples/all_pdf_urls.csv", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.error("choose --dry-run or --apply")
    con = sqlite3.connect(ROOT / a.db); con.row_factory = sqlite3.Row

    # Metadata for the new chunks is inherited from the document they came from, taken from
    # the rows about to be retired so nothing is invented.
    meta = {}
    for r in con.execute("SELECT document_id, min(source_url) url, min(citation) citation, "
                         "min(authority_class) authority_class, min(legal_regime) legal_regime, "
                         "min(jurisdiction) jurisdiction, min(corpus_role) corpus_role "
                         "FROM chunks WHERE lower(source_url) LIKE '%.pdf' GROUP BY document_id"):
        meta[r["document_id"]] = dict(r)

    existing = {H(r["text"]) for r in con.execute(
        "SELECT text FROM chunks WHERE superseded_by IS NULL AND filtered_out IS NULL "
        "AND lower(source_url) NOT LIKE '%.pdf'")}

    retire = [r[0] for r in con.execute(
        "SELECT chunk_id FROM chunks WHERE superseded_by IS NULL AND lower(source_url) LIKE '%.pdf'")]

    rows, seen = [], set()
    stats = collections.Counter()
    for f in sorted(glob.glob(str(ROOT / a.chunks_dir / "*.chunks.json"))):
        did = Path(f).name.replace(".chunks.json", "")
        d = json.load(open(f, encoding="utf-8"))
        m = meta.get(did)
        if not m:
            stats["no_metadata"] += 1
            continue
        for i, c in enumerate(d["data"], 1):
            hh = H(c["text"])
            flag = None
            if hh in seen:
                flag = "duplicate_within_pdf_rechunk"; stats["dup_within"] += 1
            elif hh in existing:
                flag = "duplicate_of_non_pdf_source"; stats["dup_existing"] += 1
            elif c.get("fidelity_failed"):
                flag = "pdf_fidelity_below_threshold"; stats["fidelity_failed"] += 1
            seen.add(hh)
            title = c.get("title") or ""
            prefix = "\n".join(x for x in [m["citation"], title] if x)
            rows.append((
                f"{did}__PDFV2__CH_{i:04d}", did, f"{did}__PDFV2", i, None, None, "PDF",
                m["authority_class"], m["corpus_role"], m["legal_regime"], m["jurisdiction"],
                m["citation"], None, "[]", title, None, "[]", "[]", "[]", "[]",
                c["text"], f"{prefix}\n\n{c['text']}", "[]", m["url"], hh,
                TAG, d.get("model"), "pdf_text_chunks_v1", "2.0.0",
                c["char_count"], c["est_tokens"], None, flag))
            stats["kept" if flag is None else "flagged"] += 1

    print(f"new chunks prepared : {len(rows):,}")
    print(f"  kept active       : {stats['kept']:,}")
    print(f"  flagged           : {stats['flagged']:,}  {dict((k,v) for k,v in stats.items() if k.startswith(('dup','fid')))}")
    print(f"old PDF chunks to retire : {len(retire):,}")
    if a.dry_run:
        print("\ndry run - nothing written")
        return 0

    cols = ("chunk_id,document_id,segment_id,chunk_ordinal,parent_node_id,parent_type,source_kind,"
            "authority_class,corpus_role,legal_regime,jurisdiction,citation,heading,heading_path,"
            "retrieval_title,retrieval_summary,topics,legal_concepts,procurement_stage,"
            "question_intents,text,embedding_text,link_placeholders,source_url,content_sha256,"
            "chunking_method,chunking_model,prompt_version,pipeline_version,char_count,"
            "est_tokens,superseded_by,filtered_out")
    con.executemany(f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({','.join('?'*33)})", rows)
    con.executemany("UPDATE chunks SET superseded_by=? WHERE chunk_id=?",
                    [("pdf_rechunk_v2", c) for c in retire])
    # FTS: drop retired, add new.
    con.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", [(c,) for c in retire])
    con.executemany("INSERT INTO chunks_fts (chunk_id,text,retrieval_title,retrieval_summary,"
                    "keywords,citation) VALUES (?,?,?,?,?,?)",
                    [(r[0], r[20], r[14], "", "", r[11]) for r in rows if r[32] is None])
    con.commit()
    act = con.execute("SELECT count(*), sum(char_count) FROM chunks "
                      "WHERE superseded_by IS NULL AND filtered_out IS NULL").fetchone()
    print(f"\napplied. active corpus: {act[0]:,} chunks, {act[1]:,} chars")
    ((ROOT / a.db).parent / "ingest_pdf_chunks_report.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(), "inserted": len(rows),
         "retired": len(retire), **dict(stats)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
