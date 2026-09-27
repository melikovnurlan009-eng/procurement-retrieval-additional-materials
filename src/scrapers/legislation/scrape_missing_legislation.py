#!/usr/bin/env python3
"""Acquire the cited-but-missing legislation, ranked by how often the corpus cites it.

Why this exists
---------------
Only three instruments were ever parsed from statutory XML, so only 9.5% of the corpus
carries legal node identity. Everything else is text with a URL: findable by search, but
unable to be the target of a citation, and therefore invisible to the knowledge graph.
Reference resolution recorded 701 distinct instruments that the corpus cites and does not
hold. Acquiring the head of that distribution is the single highest-value corpus action
available: the top ten alone are projected to raise the legislation share from 9.5% to
roughly 31%, and each acquired instrument also converts dead-end citations into live edges.

Method
------
This does NOT introduce a second parser. It reuses `group_a_legislation_scraper_v4`, which
already produces the representation the LEGISLATION lane expects, by generating that
scraper's per-instrument configuration from the acquisition list rather than hard-coding it.
Consequently every acquired instrument is chunked, validated and indexed exactly as the
Procurement Act was, and no new format enters the pipeline.

Per instrument the underlying scraper fetches four official representations - latest and
original, each as Akoma Ntoso and CLML - scores them by PARSED PROVISION COVERAGE rather
than raw tag counts, and keeps the strongest. `/contents` endpoints are never used as the
body: they are navigation documents and omit provision text.

Validation for unknown instruments
----------------------------------
The existing scraper hard-fails when provision counts fall below hand-set expectations. That
is correct for three known instruments and wrong here, because the expected size of an
arbitrary Act is not known in advance. Thresholds are therefore set permissively and the
ACTUAL parsed counts are recorded, so implausible results surface as a report line for
inspection rather than as a crash that halts the batch.

Politeness
----------
Sequential, one instrument at a time, with a pause between requests and an identifying
User-Agent. legislation.gov.uk is a public service; this is a research crawl of a few dozen
documents, not a bulk mirror. Already-acquired instruments are skipped so the job resumes.

Usage
-----
    python scrapers/legislation/scrape_missing_legislation.py --top 10 --dry-run
    python scrapers/legislation/scrape_missing_legislation.py --top 10 --apply
"""
from __future__ import annotations

import argparse, collections, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scrapers" / "legislation"))

LEG = re.compile(
    r"legislation\.gov\.uk/(?:id/)?(ukpga|uksi|ukdsi|asp|asc|nisr|ssi)/(\d{4})/([\w\-]+)", re.I)

# Original-variant name differs by instrument class: Acts are "enacted", SIs are "made".
ORIGINAL = {"ukpga": "enacted", "asp": "enacted", "asc": "enacted",
            "uksi": "made", "ukdsi": "made", "nisr": "made", "ssi": "made"}
TOP_TYPE = {"ukpga": "section", "asp": "section", "asc": "section",
            "uksi": "regulation", "ukdsi": "regulation", "nisr": "regulation",
            "ssi": "regulation"}


def targets(acq: Path, held: set[str], top: int) -> list[dict]:
    cites = collections.Counter()
    for line in acq.open(encoding="utf-8"):
        r = json.loads(line)
        m = LEG.search(r["url"])
        if not m:
            continue
        t, y, n = m.group(1).lower(), m.group(2), m.group(3)
        # ukdsi identifiers are ISBN-style draft ids and are not addressable as /type/year/num.
        if t == "ukdsi" or not n.isdigit():
            continue
        cites[(t, y, n)] += r["citations"]
    out = []
    for (t, y, n), c in cites.most_common():
        doc_id = f"{t.upper()}_{y}_{n}"
        if doc_id in held:
            continue
        out.append({
            "key": doc_id, "document_id": doc_id, "citations": c,
            "title": f"{t.upper()} {y}/{n}", "type": t, "year": int(y), "number": int(n),
            "root_url": f"https://www.legislation.gov.uk/{t}/{y}/{n}",
            "contents_url": f"https://www.legislation.gov.uk/{t}/{y}/{n}/contents",
            "original_variant": ORIGINAL.get(t, "made"),
            "expected_top_type": TOP_TYPE.get(t, "section"),
            # Permissive: real counts are recorded and reviewed, not asserted in advance.
            "expected_min_top": 1, "expected_min_nodes": 1,
            "expect_structured_refs": False,
            "corpus_role": "CITED_SUPPORTING_LEGISLATION",
        })
        if len(out) >= top:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acquisition", default="evaluation/acquisition/missing_references.jsonl", type=Path)
    ap.add_argument("--output-dir", default="data/legislation_acquired", type=Path)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.apply or a.dry_run):
        ap.error("choose --dry-run or --apply")

    import sqlite3
    con = sqlite3.connect(ROOT / a.db)
    held = {r[0] for r in con.execute("SELECT DISTINCT document_id FROM chunks")}
    out_root = ROOT / a.output_dir
    already = {p.name.upper() for p in out_root.iterdir()} if out_root.exists() else set()

    tgts = targets(ROOT / a.acquisition, held, a.top)
    tgts = [t for t in tgts if t["key"].upper() not in already]

    print(f"{'#':>3} {'citations':>10}  {'document_id':22s} {'url'}")
    print("-" * 92)
    for i, t in enumerate(tgts, 1):
        print(f"{i:3d} {t['citations']:10d}  {t['document_id']:22s} {t['root_url']}")
    print(f"\n{len(tgts)} instruments to acquire "
          f"(pause {a.pause}s, ~4 requests each -> ~{len(tgts)*4*a.pause/60:.1f} min of requests)")

    if a.dry_run:
        (ROOT / "evaluation" / "acquisition" / "scrape_plan.json").write_text(
            json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                        "targets": tgts}, indent=2), encoding="utf-8")
        print("plan written to evaluation/acquisition/scrape_plan.json")
        return 0

    import group_a_legislation_scraper_v4 as V4
    out_root.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for i, t in enumerate(tgts, 1):
        V4.SOURCES[t["key"]] = t
        d = out_root / t["key"].lower()
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i}/{len(tgts)}] {t['document_id']}  ({t['citations']} citations)", flush=True)
        try:
            doc, summary = V4.InstrumentScraper(
                source_key=t["key"], output_dir=d, timeout=a.timeout, pause=a.pause).run()
            ok.append({"document_id": t["document_id"], "citations": t["citations"],
                       "summary": summary})
            print(f"    ok: {json.dumps(summary)[:160]}", flush=True)
        except Exception as exc:
            failed.append({"document_id": t["document_id"], "error": str(exc)[:220]})
            print(f"    FAILED: {str(exc)[:160]}", flush=True)
        time.sleep(a.pause)

    rep = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "requested": len(tgts), "acquired": len(ok), "failed": len(failed),
           "ok": ok, "failures": failed}
    (ROOT / "evaluation" / "acquisition" / "scrape_report.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nacquired {len(ok)}, failed {len(failed)} -> evaluation/acquisition/scrape_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
