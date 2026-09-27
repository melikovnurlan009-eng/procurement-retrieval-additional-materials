#!/usr/bin/env python3
"""
Professional / practitioner procurement corpus scraper.

This scraper implements the same acquisition pattern used in the previous
thesis scrapers:

HTML
  -> preserve raw HTML
  -> extract substantive page content
  -> replace hyperlink occurrences with [[LINK_NNNN]]
  -> store links separately with context
  -> NO semantic chunking
  -> NO graph-edge inference

PDF
  -> preserve raw PDF
  -> extract page text
  -> extract PDF hyperlink annotations separately
  -> NO semantic chunking
  -> NO graph-edge inference

SPECIAL CASE
------------
For the Mills & Reeve "Part 3: Award of public contracts and procedures" page,
the user explicitly requested the PDFs linked from that page too. Therefore
this scraper follows only PDF links found in that page's substantive body and
stores each PDF as an independently addressable child document.

It does NOT recursively follow ordinary HTML links.

Outputs
-------
<output_dir>/
    corpus_manifest.json
    documents.jsonl
    all_links.jsonl
    failures.jsonl

    <source_key>/
        document.json
        content_with_link_placeholders.txt      # HTML sources
        content_with_link_placeholders.html     # HTML sources
        links.jsonl
        headings.jsonl
        updates.jsonl
        raw/rendered_page.html

    <pdf_source_key>/
        document.json
        pdf_pages.jsonl
        pdf_links.jsonl
        full_text.txt
        raw/source.pdf

    procurementportal_part3/linked_pdfs/<child_id>/
        document.json
        pdf_pages.jsonl
        pdf_links.jsonl
        full_text.txt
        raw/source.pdf

The link placeholders in HTML remain suitable for the later workflow:
LLM semantic chunking -> placeholder-to-link join -> target resolution ->
typed graph-edge creation.
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

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


VERSION = "1.0.0"

SOURCES = {
    "TRANSITIONAL_SAVING_ARRANGEMENTS_PDF": {
        "source_id": "TRANSITIONAL_SAVING_ARRANGEMENTS_PDF",
        "url": "https://assets.publishing.service.gov.uk/media/68c92f0b5e09a4a59af0bf1c/Guidance_-_Transitional_and_Saving_Arrangements_FINAL_v4_pdf_links__.docx.pdf",
        "format": "PDF",
        "provider": "Cabinet Office / GOV.UK",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "evidence_class": "OFFICIAL_PA23_TECHNICAL_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["TRANSITIONAL_ARRANGEMENTS", "SAVINGS", "PA2023", "PCR2015_LEGACY"],
    },

    "PROCUREMENTPORTAL_PRE_PROCUREMENT": {
        "source_id": "PROCUREMENTPORTAL_PRE_PROCUREMENT",
        "url": "https://www.procurementportal.com/procurement-law-and-guidance/procurement-act-2023-faqs/pre-procurement-considerations-under-the-act/",
        "format": "HTML",
        "provider": "Mills & Reeve",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "PRE_PROCUREMENT", "PLANNING", "MARKET_ENGAGEMENT"],
    },

    "PROCUREMENTPORTAL_PART3": {
        "source_id": "PROCUREMENTPORTAL_PART3",
        "url": "https://www.procurementportal.com/procurement-law-and-guidance/navigating-the-procurement-act-2023/part-3-award-of-public-contracts-and-procedures/",
        "format": "HTML",
        "provider": "Mills & Reeve",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "AWARD", "PROCEDURES", "STANDSTILL", "FRAMEWORKS", "DIRECT_AWARD"],
        "follow_linked_pdfs": True,
    },

    "PROCUREMENTPORTAL_PHRASEBOOK_PDF": {
        "source_id": "PROCUREMENTPORTAL_PHRASEBOOK_PDF",
        "url": "https://www.procurementportal.com/media/jqxapevh/procurement-act-2023-legal-phrasebook.pdf",
        "format": "PDF",
        "provider": "Mills & Reeve",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_REFERENCE_TOOL",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "TERMINOLOGY", "PCR2015_LEGACY"],
    },

    "BURGES_SALMON_AWARD_STAGE": {
        "source_id": "BURGES_SALMON_AWARD_STAGE",
        "url": "https://www.burges-salmon.com/our-thinking/preparing-for-change-award-stage/",
        "fallback_urls": [
            "https://www.burges-salmon.com/news-and-insight/legal-updates/preparing-for-change-award-stage"
        ],
        "format": "HTML",
        "provider": "Burges Salmon",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "AWARD", "MAT", "AWARD_CRITERIA"],
    },

    "ANTHONY_COLLINS_FRAMEWORKS": {
        "source_id": "ANTHONY_COLLINS_FRAMEWORKS",
        "url": "https://www.anthonycollins.com/insights/ebriefings/procurement-act-guidance-on-frameworks-is-out-buyers-beware/",
        "format": "HTML",
        "provider": "Anthony Collins",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "FRAMEWORKS", "OPEN_FRAMEWORKS", "CALL_OFFS"],
    },

    "ADDLESHAW_FIRST_PA23_CASE": {
        "source_id": "ADDLESHAW_FIRST_PA23_CASE",
        "url": "https://www.addleshawgoddard.com/en/insights/insights-briefings/2026/dispute-resolution/procurement-first-case-procurement-act-2023-suggests-shift-balance-power/",
        "format": "HTML",
        "provider": "Addleshaw Goddard",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_CASE_ANALYSIS",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "REMEDIES", "AUTOMATIC_SUSPENSION", "SECTION_102", "CASE_LAW"],
    },

    "ADDLESHAW_BID_RIGGING_AI": {
        "source_id": "ADDLESHAW_BID_RIGGING_AI",
        "url": "https://www.addleshawgoddard.com/en/insights/insights-briefings/2025/competition/detecting-bid-rigging-through-ai-and-public-procurement-law-implications/",
        "format": "HTML",
        "provider": "Addleshaw Goddard",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["BID_RIGGING", "AI", "COMPETITION", "DEBARMENT", "PROCUREMENT"],
    },

    "PINSENT_FLEXIBLE_PROCEDURES": {
        "source_id": "PINSENT_FLEXIBLE_PROCEDURES",
        "url": "https://www.pinsentmasons.com/en-gb/out-law/analysis/new-uk-procurement-act-procedures-offer-greater-flexibility",
        "format": "HTML",
        "provider": "Pinsent Masons / Out-Law",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "COMPETITIVE_FLEXIBLE_PROCEDURE", "OPEN_PROCEDURE", "FRAMEWORKS"],
    },

    "PINSENT_JUDICIAL_REVIEW": {
        "source_id": "PINSENT_JUDICIAL_REVIEW",
        "url": "https://www.pinsentmasons.com/out-law/analysis/judicial-review-uk-procurement-act-challenge",
        "format": "HTML",
        "provider": "Pinsent Masons / Out-Law",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_CASE_ANALYSIS",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "JUDICIAL_REVIEW", "REMEDIES", "DEBARMENT", "CHALLENGE"],
    },

    "PINSENT_CONTRACTOR_RISKS": {
        "source_id": "PINSENT_CONTRACTOR_RISKS",
        "url": "https://www.pinsentmasons.com/out-law/analysis/plenty-to-cheer-about-contractors-uk-procurement-act",
        "format": "HTML",
        "provider": "Pinsent Masons / Out-Law",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "SUPPLIERS", "PERFORMANCE", "KPI", "EXCLUSION", "DEBARMENT"],
    },

    "TROWERS_PIPELINE_NOTICES": {
        "source_id": "TROWERS_PIPELINE_NOTICES",
        "url": "https://www.trowers.com/insights/2025/june/procurement-act-2023-pipeline-notices--what-do-contracting-authorities-need-to-know",
        "format": "HTML",
        "provider": "Trowers & Hamlins",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "PIPELINE_NOTICES", "SECTION_93", "TRANSPARENCY"],
    },

    "TROWERS_FRAMEWORKS_GOLD_STANDARD": {
        "source_id": "TROWERS_FRAMEWORKS_GOLD_STANDARD",
        "url": "https://www.trowers.com/insights/2025/february/procurement-act-2023-frameworks-and-gold-standard",
        "format": "HTML",
        "provider": "Trowers & Hamlins",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "FRAMEWORKS", "GOLD_STANDARD", "SME", "CALL_OFFS"],
    },

    "TROWERS_THRESHOLD_CHANGES": {
        "source_id": "TROWERS_THRESHOLD_CHANGES",
        "url": "https://www.trowers.com/insights/2025/december/procurement-act-2023-threshold-changes",
        "format": "HTML",
        "provider": "Trowers & Hamlins",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "THRESHOLDS", "2026", "TEMPORAL_LAW"],
    },

    "TROWERS_ONE_YEAR_REVIEW": {
        "source_id": "TROWERS_ONE_YEAR_REVIEW",
        "url": "https://www.trowers.com/insights/2026/february/almost-a-year-on-how-has-the-procurement-act-2023-faired",
        "format": "HTML",
        "provider": "Trowers & Hamlins",
        "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
        "evidence_class": "PROFESSIONAL_INTERPRETATION",
        "retrieval_lane": "COMMENTARY",
        "topic": ["PA2023", "IMPLEMENTATION", "EARLY_CASES", "PRACTICE"],
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


def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180].strip("_") or "file"


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


class BaseScraper:
    def __init__(self, timeout=60, pause=0.4):
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MastersThesisProcurementResearchBot/1.0",
            "Accept-Language": "en-GB,en;q=0.9",
        })
        self.fetch_log = []

    def fetch(self, url, accept):
        r = self.session.get(
            url,
            headers={"Accept": accept},
            timeout=self.timeout,
            allow_redirects=True,
        )
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


class PDFScraper(BaseScraper):
    def scrape(
        self,
        source: dict[str, Any],
        output_dir: Path,
        parent_source_id: str | None = None,
        parent_placeholder: str | None = None,
    ):
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for PDF extraction. "
                "Install requirements_professional_sources.txt"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        data, final_url, content_type = self.fetch(
            source["url"],
            "application/pdf,*/*;q=0.5"
        )

        (raw_dir / "source.pdf").write_bytes(data)

        pdf = fitz.open(stream=data, filetype="pdf")
        pages = []
        pdf_links = []
        all_text = []

        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            text_clean = text.strip()
            all_text.append(text_clean)

            pages.append({
                "source_id": source["source_id"],
                "page_number": page_no,
                "text": text_clean,
                "text_sha256": sha256_text(text_clean),
                "char_count": len(text_clean),
            })

            link_index = 0
            for link in page.get_links():
                uri = link.get("uri")
                if not uri:
                    continue
                link_index += 1
                pdf_links.append({
                    "source_id": source["source_id"],
                    "page_number": page_no,
                    "link_occurrence_on_page": link_index,
                    "absolute_url": uri,
                    "link_class": classify_link(uri),
                    "followed": False,
                    "target_document_id": None,
                    "resolution_status": "UNRESOLVED",
                    "edge_status": "NOT_CREATED",
                })

        full_text = "\n\n".join(all_text).strip()
        (output_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
        write_jsonl(output_dir / "pdf_pages.jsonl", pages)
        write_jsonl(output_dir / "pdf_links.jsonl", pdf_links)

        doc = {
            "source_id": source["source_id"],
            "canonical_url": source["url"],
            "final_url": final_url,
            "provider": source.get("provider"),
            "source_type": "PDF",
            "authority_class": source.get("authority_class"),
            "evidence_class": source.get("evidence_class"),
            "retrieval_lane": source.get("retrieval_lane"),
            "topic": source.get("topic", []),
            "parent_source_id": parent_source_id,
            "parent_placeholder": parent_placeholder,
            "content_type": content_type,
            "file_sha256": sha256_bytes(data),
            "page_count": len(pages),
            "pdf_link_count": len(pdf_links),
            "full_text_sha256": sha256_text(full_text),
            "chunking_status": "NOT_CHUNKED",
            "graph_edge_status": "NOT_CREATED",
            "fetch_log": self.fetch_log,
        }

        (output_dir / "document.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return doc, pdf_links


class HTMLScraper(BaseScraper):
    def __init__(self, source: dict[str, Any], timeout=60, pause=0.4):
        super().__init__(timeout=timeout, pause=pause)
        self.source = source

    def fetch_with_fallbacks(self):
        candidates = [self.source["url"]] + self.source.get("fallback_urls", [])
        errors = []

        for url in candidates:
            try:
                return self.fetch(
                    url,
                    "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"
                )
            except Exception as exc:
                errors.append({"url": url, "error": str(exc)})

        raise RuntimeError(
            "All URL candidates failed: " + json.dumps(errors)
        )

    def get_content(self, soup):
        host = urlparse(self.source["url"]).netloc.lower()

        selectors_by_host = {
            "www.procurementportal.com": [
                "main article", "article", "main",
                ".content", ".main-content"
            ],
            "www.burges-salmon.com": [
                "main article", "article", "main",
                ".article-content", ".content"
            ],
            "www.anthonycollins.com": [
                "main article", "article", "main",
                ".article-content", ".content"
            ],
            "www.addleshawgoddard.com": [
                "main article", "article", "main",
                ".article-content", ".content"
            ],
            "www.pinsentmasons.com": [
                "main article", "article", "main",
                ".article-content", ".content"
            ],
            "www.trowers.com": [
                "main article", "article", "main",
                ".article-content", ".content"
            ],
        }

        for selector in selectors_by_host.get(host, ["main article", "article", "main"]):
            c = soup.select_one(selector)
            if c and len(clean_ws(c.get_text(" ", strip=True))) > 300:
                return c

        main = soup.find("main")
        if main:
            return main

        return soup

    def clean(self, soup):
        for t in soup.find_all([
            "script", "style", "noscript", "form", "button",
            "input", "select", "textarea", "svg"
        ]):
            t.decompose()

        # Remove common non-content widgets only when nested inside selected body.
        for selector in [
            "nav",
            ".breadcrumbs", ".breadcrumb",
            ".cookie", ".cookies",
            ".social-share", ".share",
            ".newsletter", ".signup",
            ".related-content", ".related",
            ".footer", "footer",
        ]:
            for t in soup.select(selector):
                t.decompose()

    def classify_and_replace_links(self, soup, source_url):
        links = []
        i = 0

        for a in list(soup.find_all("a", href=True)):
            href = clean_ws(a.get("href"))
            if not href:
                continue

            i += 1
            placeholder = f"[[LINK_{i:04d}]]"
            absolute_url = urljoin(source_url, href)
            defrag, fragment = urldefrag(absolute_url)
            anchor_text = clean_ws(a.get_text(" ", strip=True))

            heading = a.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
            container = a.find_parent(["p", "li", "td", "th", "div", "section"])

            links.append({
                "source_id": self.source["source_id"],
                "placeholder": placeholder,
                "occurrence": i,
                "anchor_text": anchor_text,
                "raw_href": href,
                "absolute_url": absolute_url,
                "defragmented_url": defrag,
                "fragment": fragment or None,
                "link_class": classify_link(absolute_url),
                "source_heading": (
                    clean_ws(heading.get_text(" ", strip=True))
                    if heading else None
                ),
                "source_context": (
                    clean_ws((container or a).get_text(" ", strip=True))[:900]
                ),
                "followed": False,
                "target_document_id": None,
                "resolution_status": "UNRESOLVED",
                "edge_status": "NOT_CREATED",
            })

            a.replace_with(
                NavigableString(
                    (anchor_text + " " if anchor_text else "") + placeholder
                )
            )

        return links

    def structured_text(self, soup):
        lines = []

        for n in soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "tr", "blockquote"
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

        text = "\n".join(lines).strip()
        if len(clean_ws(text)) < 300:
            text = soup.get_text("\n", strip=True)
        return text

    def headings(self, soup):
        result = []
        stack = []

        for i, h in enumerate(
            soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]),
            start=1
        ):
            level = int(h.name[1])
            text = clean_ws(h.get_text(" ", strip=True))
            if not text:
                continue
            stack = [x for x in stack if x[0] < level]
            stack.append((level, text))
            result.append({
                "source_id": self.source["source_id"],
                "ordinal": i,
                "heading_level": level,
                "heading": text,
                "heading_path": [x[1] for x in stack],
            })
        return result

    def scrape(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)

        raw, final_url, content_type = self.fetch_with_fallbacks()
        (raw_dir / "rendered_page.html").write_bytes(raw)

        rendered = BeautifulSoup(raw, "html.parser")
        body = self.get_content(rendered)

        # Copy body into a separate soup before mutation.
        body_soup = BeautifulSoup(str(body), "html.parser")
        self.clean(body_soup)

        links = self.classify_and_replace_links(
            body_soup,
            final_url
        )
        text = self.structured_text(body_soup)
        html = str(body_soup)
        heads = self.headings(body_soup)

        title_el = rendered.find("h1")
        title = clean_ws(
            title_el.get_text(" ", strip=True)
        ) if title_el else None

        (output_dir / "content_with_link_placeholders.txt").write_text(
            text,
            encoding="utf-8"
        )
        (output_dir / "content_with_link_placeholders.html").write_text(
            html,
            encoding="utf-8"
        )
        write_jsonl(output_dir / "links.jsonl", links)
        write_jsonl(output_dir / "headings.jsonl", heads)
        write_jsonl(output_dir / "updates.jsonl", [])

        doc = {
            "source_id": self.source["source_id"],
            "canonical_url": self.source["url"],
            "final_url": final_url,
            "title": title,
            "provider": self.source.get("provider"),
            "source_type": "HTML",
            "authority_class": self.source.get("authority_class"),
            "evidence_class": self.source.get("evidence_class"),
            "retrieval_lane": self.source.get("retrieval_lane"),
            "topic": self.source.get("topic", []),
            "content_type": content_type,
            "content_text_sha256": sha256_text(text),
            "content_html_sha256": sha256_text(html),
            "content_char_count": len(text),
            "link_count": len(links),
            "heading_count": len(heads),
            "chunking_status": "NOT_CHUNKED",
            "link_resolution_status": "NOT_FOLLOWED",
            "graph_edge_status": "NOT_CREATED",
            "fetch_log": self.fetch_log,
        }

        (output_dir / "document.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        return doc, links


def classify_link(url):
    p = urlparse(url)
    host = p.netloc.lower()
    path = p.path.lower()

    if path.endswith(".pdf"):
        return "PDF"
    if path.endswith(".docx"):
        return "DOCX"
    if path.endswith(".xlsx"):
        return "XLSX"
    if host.endswith("legislation.gov.uk"):
        return "LEGISLATION"
    if host == "www.gov.uk":
        return "GOVUK"
    if "procurementportal.com" in host:
        return "PROCUREMENT_PORTAL"
    if host:
        return "EXTERNAL_WEB"
    return "OTHER"


def scrape_linked_pdfs_from_part3(
    parent_doc,
    parent_links,
    parent_output_dir,
    timeout,
    pause,
):
    children = []

    linked_pdf_root = parent_output_dir / "linked_pdfs"
    linked_pdf_root.mkdir(exist_ok=True)

    seen_urls = set()

    for link in parent_links:
        if link.get("link_class") != "PDF":
            continue

        pdf_url = link["absolute_url"]

        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)

        child_id = (
            "PROCUREMENTPORTAL_PART3_PDF_"
            + sha256_text(pdf_url)[:12].upper()
        )

        child_source = {
            "source_id": child_id,
            "url": pdf_url,
            "provider": "Mills & Reeve",
            "authority_class": "NON_AUTHORITATIVE_PROFESSIONAL",
            "evidence_class": "PROFESSIONAL_SUPPORTING_DOCUMENT",
            "retrieval_lane": "COMMENTARY",
            "topic": parent_doc.get("topic", []),
        }

        child_dir = linked_pdf_root / child_id.lower()

        try:
            pdf_scraper = PDFScraper(
                timeout=timeout,
                pause=pause
            )
            child_doc, child_links = pdf_scraper.scrape(
                child_source,
                child_dir,
                parent_source_id=parent_doc["source_id"],
                parent_placeholder=link["placeholder"],
            )

            children.append({
                "document": child_doc,
                "links": child_links,
            })

            # Parent link is now followed in the acquisition phase, but no
            # semantic graph relation is inferred.
            link["followed"] = True
            link["target_document_id"] = child_id
            link["resolution_status"] = "SCRAPED_PDF_CHILD"
            link["edge_status"] = "NOT_CREATED"

        except Exception as exc:
            children.append({
                "error": str(exc),
                "url": pdf_url,
                "parent_placeholder": link["placeholder"],
            })

    # Rewrite parent links after child PDF acquisition status updates.
    write_jsonl(
        parent_output_dir / "links.jsonl",
        parent_links
    )

    return children


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output-dir",
        default="data/professional_procurement_sources"
    )
    ap.add_argument(
        "--source",
        choices=["ALL"] + sorted(SOURCES),
        default="ALL"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.4
    )

    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    selected = (
        sorted(SOURCES)
        if args.source == "ALL"
        else [args.source]
    )

    documents = []
    all_links = []
    failures = []
    child_pdf_documents = []

    for key in selected:
        source = SOURCES[key]
        source_dir = root / key.lower()

        try:
            if source["format"] == "PDF":
                scraper = PDFScraper(
                    timeout=args.timeout,
                    pause=args.pause
                )
                doc, links = scraper.scrape(
                    source,
                    source_dir
                )

            else:
                scraper = HTMLScraper(
                    source,
                    timeout=args.timeout,
                    pause=args.pause
                )
                doc, links = scraper.scrape(
                    source_dir
                )

            documents.append(doc)
            all_links.extend(
                [{**x, "source_key": key} for x in links]
            )

            if source.get("follow_linked_pdfs"):
                children = scrape_linked_pdfs_from_part3(
                    parent_doc=doc,
                    parent_links=links,
                    parent_output_dir=source_dir,
                    timeout=args.timeout,
                    pause=args.pause,
                )

                for child in children:
                    if "document" in child:
                        child_pdf_documents.append(
                            child["document"]
                        )
                        all_links.extend(
                            child.get("links", [])
                        )
                    else:
                        failures.append({
                            "source_key": key,
                            "kind": "LINKED_PDF",
                            **child,
                        })

            print(f"Completed {key}")

        except Exception as exc:
            failures.append({
                "source_key": key,
                "source_url": source["url"],
                "error": str(exc),
            })
            print(
                f"ERROR {key}: {exc}",
                file=sys.stderr
            )

    documents.extend(child_pdf_documents)

    write_jsonl(
        root / "documents.jsonl",
        documents
    )
    write_jsonl(
        root / "all_links.jsonl",
        all_links
    )
    write_jsonl(
        root / "failures.jsonl",
        failures
    )

    manifest = {
        "scraper_version": VERSION,
        "generated_at": utc_now(),
        "selected_sources": selected,
        "successful_documents": len(documents),
        "linked_pdf_documents": len(child_pdf_documents),
        "failure_count": len(failures),
        "sources": SOURCES,
        "method": {
            "html_link_placeholders": True,
            "html_links_stored_separately": True,
            "pdf_page_text_extraction": True,
            "pdf_link_annotations_stored_separately": True,
            "recursive_html_crawl": False,
            "part3_follow_linked_pdfs_only": True,
            "semantic_chunking": False,
            "graph_edge_inference": False,
        },
    }

    (root / "corpus_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
