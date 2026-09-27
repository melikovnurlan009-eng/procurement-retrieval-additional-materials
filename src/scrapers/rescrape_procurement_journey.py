#!/usr/bin/env python3
"""Re-acquire procurementjourney.scot from HTML pages instead of print-to-PDF renders.

Why
---
200 documents (500 chunks) were ingested from `/print/pdf/node/<id>`, a Drupal print
endpoint that renders the page as a PDF. Arriving as PDF, the material lost every
structural signal the chunker relies on - headings, list items, tables all flatten to bare
lines - and the page's accordion controls survived as body text: the literal string
"Open or close" appears 698 times, and 247 chunks carry an accordion header duplicated
immediately by its panel title.

This fetches the same nodes as HTML, extracts typed blocks, and removes interface chrome.
The canonical URL is recovered by following the redirect from `/node/<id>`, so the output
records where the content actually lives rather than the print endpoint.

Nothing is ingested. Output is staged for inspection, because the corpus already contains a
version of every one of these documents and replacing them is a separate decision.

Chrome removal is conservative and explicit: elements are dropped by tag (nav, header,
footer, script, style, form, button) and by a small blocklist of exact control labels. Text
that merely resembles boilerplate is kept - the aim is to remove interface, not to judge
content.
"""
from __future__ import annotations

import argparse, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import html as LH

UA = ("procurement-kg-rag research crawler (MSc dissertation, University of Manchester); "
      "contact via project repository")
DROP_TAGS = {"nav", "header", "footer", "script", "style", "form", "button", "noscript",
             "aside", "svg", "iframe"}
CHROME_TEXT = {"open or close", "back to top", "skip to main content", "skip to content",
               "expand all", "collapse all", "show all", "hide all", "print this page",
               "toggle navigation", "cookies on procurement journey", "accept cookies",
               "view pdf", "download pdf", "print", "share", "menu", "search",
               "checklist", "quickfire guide", "open or close section"}
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd",
              "blockquote", "pre"]


def clean(s: str) -> str:
    return re.sub(r"[ \t ]+", " ", (s or "").replace("\r", "")).strip()


def extract(doc, url: str) -> dict:
    for el in doc.iter():
        if el.tag in DROP_TAGS:
            el.drop_tree()
    # Prefer the main content region when the theme provides one.
    main = None
    for xp in ('//main', '//*[@role="main"]', '//*[contains(@class,"region-content")]',
               '//*[@id="content"]', '//article'):
        found = doc.xpath(xp)
        if found:
            main = found[0]; break
    root = main if main is not None else doc

    blocks, seen_heading = [], None
    for el in root.iter():
        if el.tag not in BLOCK_TAGS:
            continue
        txt = clean(el.text_content())
        if not txt or txt.lower() in CHROME_TEXT:
            continue
        if len(txt) < 2:
            continue
        # An accordion title repeats as its panel title, and the repeat is not always a
        # heading tag - "Quickfire Guide" arrives twice as <p>. Drop any adjacent repeat of
        # a short block, whatever its tag.
        if blocks and txt == blocks[-1]["text"] and len(txt) < 120:
            continue
        if el.tag.startswith("h"):
            seen_heading = txt
        blocks.append({"block_id": f"B{len(blocks)+1:04d}", "block_type": el.tag,
                       "heading_level": int(el.tag[1]) if el.tag.startswith("h") else None,
                       "text": txt})
    title = clean(doc.xpath("string(//title)")) or (blocks[0]["text"] if blocks else "")
    return {"canonical_url": url, "title": title, "blocks": blocks,
            "block_count": len(blocks),
            "char_count": sum(len(b["text"]) for b in blocks)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--out", default="data/rescrape_procurement_journey", type=Path)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=45)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / a.out; (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "normalized").mkdir(parents=True, exist_ok=True)

    import sqlite3
    con = sqlite3.connect(root / a.db)
    urls = sorted({r[0] for r in con.execute(
        "SELECT DISTINCT source_url FROM chunks WHERE source_url LIKE '%/print/pdf/node/%'")})
    nodes = []
    for u in urls:
        m = re.search(r"/print/pdf/node/(\d+)", u)
        if m:
            nodes.append({"node_id": m.group(1), "old_url": u})
    if a.limit:
        nodes = nodes[: a.limit]
    print(f"nodes to re-fetch: {len(nodes)}", flush=True)

    s = requests.Session(); s.headers.update({"User-Agent": UA})
    ok, fail = [], []
    for i, n in enumerate(nodes, 1):
        page = f"https://www.procurementjourney.scot/node/{n['node_id']}"
        try:
            r = s.get(page, timeout=a.timeout, allow_redirects=True)
            r.raise_for_status()
            (out / "raw" / f"node_{n['node_id']}.html").write_text(r.text, encoding="utf-8")
            rec = extract(LH.fromstring(r.text), r.url)
            rec.update({"node_id": n["node_id"], "requested_url": page,
                        "superseded_url": n["old_url"], "http_status": r.status_code,
                        "fetched_at": datetime.now(timezone.utc).isoformat()})
            (out / "normalized" / f"node_{n['node_id']}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            ok.append({k: rec[k] for k in ("node_id", "canonical_url", "title",
                                           "block_count", "char_count")})
            if i % 20 == 0 or i == len(nodes):
                print(f"  {i}/{len(nodes)}", flush=True)
        except Exception as exc:
            fail.append({"node_id": n["node_id"], "error": str(exc)[:180]})
            print(f"  [{i}] FAILED node {n['node_id']}: {str(exc)[:110]}", flush=True)
        time.sleep(a.pause)

    (out / "rescrape_report.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(nodes), "ok": len(ok), "failed": len(fail),
        "ingested": False, "note": "staged for inspection only",
        "documents": ok, "failures": fail}, indent=2), encoding="utf-8")
    print(f"\nfetched {len(ok)}, failed {len(fail)} -> {out}")
    print("NOT ingested. Inspect data/rescrape_procurement_journey/normalized/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
