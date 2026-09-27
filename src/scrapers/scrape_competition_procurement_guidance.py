#!/usr/bin/env python3
"""
Targeted scraper for four procurement/competition HTML guidance pages.

Pages:
1. Bid-rigging: advice for public sector procurers
2. Procurement Act 2023: exclusion and debarment on competition grounds
3. Competition law: guide for public authorities
4. Bid-rigging Risk in all Procurement: Practice Note (HTML)

Design matches the previous placeholder-link scraper:
- scrape each HTML page independently;
- preserve raw rendered HTML;
- prefer GOV.UK Content API body where available;
- extract substantive HTML only;
- replace every link occurrence with [[LINK_NNNN]];
- store links separately with heading/context;
- preserve headings and update history;
- NO recursive crawling;
- NO PDF downloading;
- NO LLM chunking;
- NO graph-edge inference.

Outputs
-------
<output_dir>/
    corpus_manifest.json
    documents.jsonl
    all_links.jsonl
    all_headings.jsonl
    all_updates.jsonl
    <source_key>/
        document.json
        content_with_link_placeholders.txt
        content_with_link_placeholders.html
        links.jsonl
        headings.jsonl
        updates.jsonl
        raw/
            rendered_page.html
            govuk_content_api.json   (if available)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


VERSION = "1.0.0"

SOURCES = {
    "CMA_BID_RIGGING_PROCURERS": {
        "source_id": "CMA_BID_RIGGING_PROCURERS",
        "url": "https://www.gov.uk/government/publications/bid-rigging-advice-for-public-sector-procurers/bid-rigging-advice-for-public-sector-procurers",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["BID_RIGGING", "COLLUSION", "COMPETITION", "PROCUREMENT"],
    },
    "CMA_PA23_EXCLUSION_DEBARMENT": {
        "source_id": "CMA_PA23_EXCLUSION_DEBARMENT",
        "url": "https://www.gov.uk/government/publications/the-procurement-act-2023-information-note-on-exclusion-and-debarment-on-competition-grounds/exclusion-and-debarment-on-competition-grounds-what-suppliers-and-contractors-need-to-know",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["PROCUREMENT_ACT_2023", "EXCLUSION", "DEBARMENT", "COMPETITION"],
    },
    "CMA_COMPETITION_PUBLIC_AUTHORITIES": {
        "source_id": "CMA_COMPETITION_PUBLIC_AUTHORITIES",
        "url": "https://www.gov.uk/government/publications/competition-law-guide-for-public-authorities/competition-law-guide-for-public-authorities",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["COMPETITION_LAW", "PUBLIC_AUTHORITIES"],
    },
    "PSFA_BID_RIGGING_PRACTICE_NOTE": {
        "source_id": "PSFA_BID_RIGGING_PRACTICE_NOTE",
        "url": "https://www.gov.uk/government/publications/bid-rigging-risk-in-all-procurement/bid-rigging-risk-in-all-procurement-practice-note-html",
        "evidence_class": "OFFICIAL_PRACTICE_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["BID_RIGGING", "FRAUD", "CARTELS", "PROCUREMENT"],
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_ws(s):
    return re.sub(r"[ \t\r\f\v]+", " ", s or "").strip()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class FetchRecord:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    retrieved_at: str
    sha256: str
    byte_count: int


class HtmlGuidanceScraper:
    def __init__(self, source_key: str, output_dir: Path, timeout=60, pause=0.4):
        self.source_key = source_key
        self.source = SOURCES[source_key]
        self.output_dir = output_dir
        self.raw_dir = output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MastersThesisProcurementResearchBot/1.0",
            "Accept-Language": "en-GB,en;q=0.9",
        })
        self.fetch_log = []

    def fetch(self, url, accept):
        r = self.session.get(url, headers={"Accept": accept}, timeout=self.timeout, allow_redirects=True)
        r.raise_for_status()
        data = r.content
        self.fetch_log.append(asdict(FetchRecord(
            requested_url=url,
            final_url=r.url,
            status_code=r.status_code,
            content_type=r.headers.get("content-type", ""),
            retrieved_at=utc_now(),
            sha256=sha256_bytes(data),
            byte_count=len(data),
        )))
        time.sleep(self.pause)
        return data, r.url, r.headers.get("content-type", "")

    def content_api_url(self):
        path = urlparse(self.source["url"]).path
        return "https://www.gov.uk/api/content" + path

    def fetch_api(self):
        api_url = self.content_api_url()
        try:
            data, _, _ = self.fetch(api_url, "application/json,*/*;q=0.5")
            (self.raw_dir / "govuk_content_api.json").write_bytes(data)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            (self.raw_dir / "govuk_content_api_error.json").write_text(
                json.dumps({"attempted_url": api_url, "error": str(e)}, indent=2),
                encoding="utf-8",
            )
            return None

    def get_body_html(self, rendered_soup, api):
        if api:
            body = (api.get("details") or {}).get("body")
            if isinstance(body, str) and clean_ws(BeautifulSoup(body, "html.parser").get_text(" ", strip=True)):
                return body, "GOVUK_CONTENT_API_DETAILS_BODY"

        main = rendered_soup.find("main")
        if not main:
            raise RuntimeError("Could not find GOV.UK main content")

        for c in [
            main.select_one(".govspeak"),
            main.select_one(".gem-c-govspeak"),
            main.select_one("article"),
        ]:
            if c and clean_ws(c.get_text(" ", strip=True)):
                return str(c), "RENDERED_HTML_MAIN_BODY"

        return str(main), "RENDERED_HTML_MAIN_FALLBACK"

    def clean_body(self, soup):
        for t in soup.find_all(["script", "style", "noscript", "form", "button", "input", "select", "textarea"]):
            t.decompose()

        for selector in [
            ".gem-c-print-link",
            ".gem-c-feedback",
            ".gem-c-subscription-links",
            ".subscription-links",
            ".govuk-breadcrumbs",
            ".gem-c-contextual-sidebar",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

    def nearest_heading(self, node):
        h = node.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        return clean_ws(h.get_text(" ", strip=True)) if h else None

    def context(self, a, n=700):
        p = a.find_parent(["p", "li", "td", "th", "div"])
        return clean_ws((p or a).get_text(" ", strip=True))[:n]

    def classify_link(self, url):
        p = urlparse(url)
        host = p.netloc.lower()
        path = p.path.lower()

        if host.endswith("legislation.gov.uk"):
            return "LEGISLATION"
        if host == "assets.publishing.service.gov.uk":
            if path.endswith(".pdf"):
                return "GOVUK_PDF_ATTACHMENT"
            if path.endswith(".odt"):
                return "GOVUK_ODT_ATTACHMENT"
            if path.endswith(".docx"):
                return "GOVUK_DOCX_ATTACHMENT"
            if path.endswith(".xlsx"):
                return "GOVUK_XLSX_ATTACHMENT"
            return "GOVUK_ASSET"
        if host == "submit.forms.service.gov.uk":
            return "GOVUK_FORM"
        if host == "www.gov.uk":
            if path.startswith("/guidance/"):
                return "GOVUK_GUIDANCE"
            if path.startswith("/government/publications/"):
                return "GOVUK_PUBLICATION"
            if path.startswith("/government/collections/"):
                return "GOVUK_COLLECTION"
            if path.startswith("/government/organisations/"):
                return "GOVUK_ORGANISATION"
            return "GOVUK_PAGE"
        if host.endswith("gov.uk"):
            return "OTHER_GOVUK_SERVICE"
        return "EXTERNAL"

    def replace_links(self, soup, source_url):
        out = []
        i = 0

        for a in list(soup.find_all("a", href=True)):
            href = clean_ws(a.get("href"))
            if not href:
                continue

            i += 1
            ph = f"[[LINK_{i:04d}]]"
            abs_url = urljoin(source_url, href)
            defrag, frag = urldefrag(abs_url)
            text = clean_ws(a.get_text(" ", strip=True))

            out.append({
                "source_id": self.source["source_id"],
                "placeholder": ph,
                "occurrence": i,
                "anchor_text": text,
                "raw_href": href,
                "absolute_url": abs_url,
                "defragmented_url": defrag,
                "fragment": frag or None,
                "link_class": self.classify_link(abs_url),
                "source_heading": self.nearest_heading(a),
                "source_context": self.context(a),
                "followed": False,
                "target_document_id": None,
                "resolution_status": "UNRESOLVED",
                "edge_status": "NOT_CREATED",
            })

            a.replace_with(NavigableString((text + " " if text else "") + ph))

        return out

    def structured_text(self, soup):
        lines = []

        for n in soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "tr", "blockquote",
        ]):
            text = clean_ws(n.get_text(" ", strip=True))
            if not text:
                continue

            if re.fullmatch(r"h[1-6]", n.name or ""):
                lines += ["", "#" * int(n.name[1]) + " " + text, ""]
            elif n.name == "li":
                lines.append("- " + text)
            elif n.name == "tr":
                cells = [
                    clean_ws(c.get_text(" ", strip=True))
                    for c in n.find_all(["th", "td"], recursive=False)
                ]
                if any(cells):
                    lines.append(" | ".join([c for c in cells if c]))
            elif n.name == "blockquote":
                lines.append("> " + text)
            else:
                lines += [text, ""]

        return "\n".join(lines).strip()

    def headings(self, soup):
        out = []
        stack = []

        for i, h in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]), 1):
            lvl = int(h.name[1])
            text = clean_ws(h.get_text(" ", strip=True))
            if not text:
                continue
            stack = [x for x in stack if x[0] < lvl]
            stack.append((lvl, text))
            out.append({
                "source_id": self.source["source_id"],
                "ordinal": i,
                "heading_level": lvl,
                "heading": text,
                "heading_path": [x[1] for x in stack],
            })
        return out

    def updates(self, soup):
        out = []

        for h in soup.find_all(["h2", "h3"]):
            if "updates to this page" in clean_ws(h.get_text(" ", strip=True)).lower():
                lvl = int(h.name[1])
                parts = []

                for s in h.next_siblings:
                    if isinstance(s, Tag) and s.name in {"h1", "h2", "h3", "h4", "h5", "h6"} and int(s.name[1]) <= lvl:
                        break
                    if isinstance(s, Tag):
                        t = clean_ws(s.get_text(" ", strip=True))
                        if t:
                            parts.append(t)

                txt = " ".join(parts)

                for m in re.finditer(
                    r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+(.*?)(?=(?:\d{1,2}\s+[A-Za-z]+\s+\d{4})|$)",
                    txt
                ):
                    out.append({
                        "source_id": self.source["source_id"],
                        "date_text": m.group(1),
                        "note": clean_ws(m.group(2)),
                    })
                break

        return out

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raw, final_url, _ = self.fetch(
            self.source["url"],
            "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"
        )
        (self.raw_dir / "rendered_page.html").write_bytes(raw)

        api = self.fetch_api()
        rendered = BeautifulSoup(raw, "html.parser")
        body_html, method = self.get_body_html(rendered, api)
        body = BeautifulSoup(body_html, "html.parser")
        self.clean_body(body)

        links = self.replace_links(body, final_url)
        text = self.structured_text(body)
        html = str(body)
        heads = self.headings(body)
        ups = self.updates(rendered)

        (self.output_dir / "content_with_link_placeholders.txt").write_text(text, encoding="utf-8")
        (self.output_dir / "content_with_link_placeholders.html").write_text(html, encoding="utf-8")
        write_jsonl(self.output_dir / "links.jsonl", links)
        write_jsonl(self.output_dir / "headings.jsonl", heads)
        write_jsonl(self.output_dir / "updates.jsonl", ups)

        title = (api or {}).get("title")
        if not title:
            h1 = rendered.find("h1")
            title = clean_ws(h1.get_text(" ", strip=True)) if h1 else ""

        status_text = clean_ws(rendered.get_text(" ", strip=True)).lower()

        doc = {
            "source_key": self.source_key,
            "source_id": self.source["source_id"],
            "canonical_url": self.source["url"],
            "final_url": final_url,
            "title": title,
            "source_type": "GOVUK_HTML_GUIDANCE",
            "evidence_class": self.source["evidence_class"],
            "authority_class": self.source["authority_class"],
            "retrieval_lane": self.source["retrieval_lane"],
            "topic": self.source["topic"],
            "document_status": (
                "WITHDRAWN"
                if "this guidance was withdrawn" in status_text or "[withdrawn]" in status_text
                else "CURRENT"
            ),
            "content_id": (api or {}).get("content_id"),
            "document_type": (api or {}).get("document_type"),
            "schema_name": (api or {}).get("schema_name"),
            "published_at": (api or {}).get("first_published_at"),
            "public_updated_at": (api or {}).get("public_updated_at"),
            "publishing_app": (api or {}).get("publishing_app"),
            "locale": (api or {}).get("locale"),
            "extraction_method": method,
            "content_text_sha256": sha256_text(text),
            "content_html_sha256": sha256_text(html),
            "content_char_count": len(text),
            "link_count": len(links),
            "heading_count": len(heads),
            "update_event_count": len(ups),
            "chunking_status": "NOT_CHUNKED",
            "link_resolution_status": "NOT_FOLLOWED",
            "graph_edge_status": "NOT_CREATED",
        }

        (self.output_dir / "document.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        manifest = {
            "scraper_version": VERSION,
            "generated_at": utc_now(),
            "source_key": self.source_key,
            "source_id": self.source["source_id"],
            "source_url": self.source["url"],
            "scope_policy": {
                "scrape_current_html_page_only": True,
                "preserve_raw_rendered_html": True,
                "prefer_govuk_content_api_body": True,
                "replace_links_with_placeholders": True,
                "store_links_separately": True,
                "follow_links": False,
                "download_linked_assets": False,
                "chunk_content": False,
                "infer_edges": False,
            },
            "placeholder_policy": {
                "format": "[[LINK_NNNN]]",
                "same_url_can_have_multiple_placeholders": True,
                "placeholder_represents_occurrence_not_unique_url": True,
            },
            "fetch_log": self.fetch_log,
        }

        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return doc, links, heads, ups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-dir",
        default="data/competition_procurement_guidance"
    )
    ap.add_argument(
        "--source",
        choices=["ALL"] + sorted(SOURCES.keys()),
        default="ALL"
    )
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.4)
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    selected = sorted(SOURCES.keys()) if args.source == "ALL" else [args.source]

    docs = []
    all_links = []
    all_headings = []
    all_updates = []
    failures = []

    for key in selected:
        out_dir = root / key.lower()
        try:
            doc, links, heads, ups = HtmlGuidanceScraper(
                key,
                out_dir,
                timeout=args.timeout,
                pause=args.pause
            ).run()
            docs.append(doc)
            all_links.extend(links)
            all_headings.extend(heads)
            all_updates.extend(ups)
            print(f"Completed {key}: {out_dir}")
        except Exception as e:
            failures.append({"source_key": key, "error": str(e)})
            print(f"ERROR {key}: {e}", file=sys.stderr)

    write_jsonl(root / "documents.jsonl", docs)
    write_jsonl(root / "all_links.jsonl", all_links)
    write_jsonl(root / "all_headings.jsonl", all_headings)
    write_jsonl(root / "all_updates.jsonl", all_updates)

    corpus_manifest = {
        "scraper_version": VERSION,
        "generated_at": utc_now(),
        "requested_sources": selected,
        "successful_documents": len(docs),
        "failure_count": len(failures),
        "failures": failures,
        "pipeline": (
            "raw HTML -> substantive HTML -> link placeholders -> "
            "LLM chunking later -> target scraping/resolution later -> graph edges later"
        ),
        "sources": SOURCES,
    }

    (root / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
