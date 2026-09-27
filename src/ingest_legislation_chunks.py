#!/usr/bin/env python3
"""Ingest the re-chunked legislation, preserving legal identity.

17 instruments were acquired because reference resolution recorded them as cited-but-absent:
their citations dead-ended, which is a large part of why graph connectivity sat at 26%. Every
chunk here carries a `parent_node_id` derived from the provision it states, so these become
addressable by citation rather than merely searchable - the property that lets an edge point
at them.

Duplicate and quality handling, measured before writing:
    397  identical text within the new set - the same provision reproduced by adjacent
         windows at a boundary; first occurrence kept
      0  duplicates against the existing corpus, and none of the 17 instruments were
         already present, so this is a clean insert
     82  below the fidelity threshold
    771  under 20 tokens - extent, commencement and short-title clauses. Earlier labelling
         found 77% of the sub-50-token band to be LOW_VALUE, so these are flagged rather
         than indexed: they cannot accumulate BM25 weight or carry a discriminative
         embedding, and they answer no question a practitioner asks.

Flagged rows are inserted, not discarded, so the decision is visible and reversible.
"""
from __future__ import annotations

import argparse, collections, glob, hashlib, json, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NORM = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
H = lambda t: hashlib.sha1(NORM(t).encode()).hexdigest()
TAG = "LLM_LEG_TEXT_V2"
MIN_TOKENS = 20

TITLES = {
    "UKPGA_2018_12": "Data Protection Act 2018", "UKPGA_2000_36": "Freedom of Information Act 2000",
    "UKPGA_1996_53": "Housing Grants, Construction and Regeneration Act 1996",
    "UKPGA_2006_35": "Fraud Act 2006", "UKPGA_2002_40": "Enterprise Act 2002",
    "UKPGA_1998_20": "Late Payment of Commercial Debts (Interest) Act 1998",
    "UKSI_2024_716": "Procurement Act 2023 (Commencement No. 3 and Transitional and Saving Provisions) Regulations 2024",
    "UKSI_2024_959": "Procurement Act 2023 (Commencement No. 3) (Amendment) Regulations 2024",
}
UNIT = {"UKPGA": "s", "ASP": "s", "ASC": "s", "UKSI": "reg", "SSI": "reg", "NISR": "reg"}


def cite(did: str, prov: str | None) -> str:
    base = TITLES.get(did) or did.replace("_", " ")
    u = UNIT.get(did.split("_")[0], "s")
    return f"{base} {u} {prov}" if prov else base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--chunks-dir", default="data/legislation_chunks", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.error("choose --dry-run or --apply")
    con = sqlite3.connect(ROOT / a.db); con.row_factory = sqlite3.Row
    existing = {H(r[0]) for r in con.execute(
        "SELECT text FROM chunks WHERE superseded_by IS NULL AND filtered_out IS NULL")}

    rows, seen, stats = [], set(), collections.Counter()
    for f in sorted(glob.glob(str(ROOT / a.chunks_dir / "*.chunks.json"))):
        d = json.load(open(f, encoding="utf-8"))
        did = d["document_id"]
        kind = did.split("_")[0]
        regime = ("PA2023" if did in ("UKSI_2024_716", "UKSI_2024_959") else None)
        role = ("PA2023_COMMENCEMENT" if regime else "CITED_SUPPORTING_LEGISLATION")
        auth = "PRIMARY_LEGISLATION" if kind in ("UKPGA", "ASP", "ASC") else "SECONDARY_LEGISLATION"
        url = (f"https://www.legislation.gov.uk/{kind.lower()}/"
               f"{did.split('_')[1]}/{did.split('_')[2]}")
        for i, c in enumerate(d["data"], 1):
            hh = H(c["text"])
            flag = None
            if hh in seen:
                flag = "duplicate_within_legislation_rechunk"; stats["dup_within"] += 1
            elif hh in existing:
                flag = "duplicate_of_existing_source"; stats["dup_existing"] += 1
            elif c.get("fidelity_failed"):
                flag = "leg_fidelity_below_threshold"; stats["fidelity_failed"] += 1
            elif (c["est_tokens"] or 0) < MIN_TOKENS:
                flag = "too_small_to_retrieve"; stats["too_small"] += 1
            seen.add(hh)
            prov = (c.get("provision") or "").strip() or None
            citation = cite(did, prov)
            title = c.get("title") or ""
            rows.append((
                f"{did}__LEGV2__CH_{i:05d}", did, f"{did}__LEGV2", i,
                c.get("parent_node_id"), "section" if auth == "PRIMARY_LEGISLATION" else "regulation",
                "LEGISLATION", auth, role, regime, "UK", citation, None, "[]",
                title, None, "[]", "[]", "[]", "[]",
                c["text"], f"{citation}\n{title}\n\n{c['text']}", "[]", url, hh,
                TAG, d.get("model"), "legislation_text_chunks_v1", "2.0.0",
                c["char_count"], c["est_tokens"], None, flag))
            stats["kept" if flag is None else "flagged"] += 1

    print(f"prepared      : {len(rows):,}")
    print(f"  kept active : {stats['kept']:,}")
    print(f"  flagged     : {stats['flagged']:,}  "
          f"{ {k: v for k, v in stats.items() if k not in ('kept','flagged')} }")
    if a.dry_run:
        print("\ndry run - nothing written"); return 0

    cols = ("chunk_id,document_id,segment_id,chunk_ordinal,parent_node_id,parent_type,source_kind,"
            "authority_class,corpus_role,legal_regime,jurisdiction,citation,heading,heading_path,"
            "retrieval_title,retrieval_summary,topics,legal_concepts,procurement_stage,"
            "question_intents,text,embedding_text,link_placeholders,source_url,content_sha256,"
            "chunking_method,chunking_model,prompt_version,pipeline_version,char_count,"
            "est_tokens,superseded_by,filtered_out")
    con.executemany(f"INSERT OR REPLACE INTO chunks ({cols}) VALUES ({','.join('?'*33)})", rows)
    con.executemany("INSERT INTO chunks_fts (chunk_id,text,retrieval_title,retrieval_summary,"
                    "keywords,citation) VALUES (?,?,?,?,?,?)",
                    [(r[0], r[20], r[14], "", "", r[11]) for r in rows if r[32] is None])
    con.commit()
    act = con.execute("SELECT count(*), sum(char_count) FROM chunks "
                      "WHERE superseded_by IS NULL AND filtered_out IS NULL").fetchone()
    ident = con.execute("SELECT count(*) FROM chunks WHERE parent_node_id IS NOT NULL "
                        "AND parent_node_id<>'' AND superseded_by IS NULL "
                        "AND filtered_out IS NULL").fetchone()[0]
    print(f"\napplied. active corpus: {act[0]:,} chunks, {act[1]:,} chars")
    print(f"chunks with legal identity: {ident:,} ({100*ident/act[0]:.1f}%)")
    ((ROOT / a.db).parent / "ingest_legislation_report.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "inserted": len(rows), **dict(stats)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
