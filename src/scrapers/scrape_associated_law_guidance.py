#!/usr/bin/env python3
"""
Targeted associated-law scraper for the procurement thesis corpus.

Purpose
-------
Scrape a tightly bounded set of procurement-adjacent official guidance pages
while preserving every hyperlink occurrence as a stable placeholder.

Pipeline
--------
source HTML page
  -> preserve raw rendered HTML
  -> extract substantive content
  -> replace each <a> occurrence with [[LINK_NNNN]]
  -> store links separately with source heading/context
  -> NO recursive crawling
  -> NO automatic target scraping
  -> NO LLM chunking
  -> NO graph-edge inference

This is designed so that later:
1. the LLM chunks placeholder-bearing text;
2. placeholders remain inside chunks;
3. links.jsonl maps placeholders back to URLs;
4. selected linked targets are scraped independently;
5. canonical target IDs are resolved;
6. graph edges are created with provenance.

Sources
-------
The default configuration covers:

Competition / bid-rigging:
- CMA bid-rigging advice for public sector procurers
- CMA PA23 exclusion/debarment on competition grounds
- CMA competition-law guide for public authorities
- PSFA bid-rigging risk in all procurement practice note

National security:
- NSUP current guidance
- PA23 Exclusions Annex 2: National Security Grounds

Information disclosure:
- ICO FOIA section 43 commercial interests
- ICO outsourcing FOIA/EIR obligations
- ICO EIR regulation 12(5)(e) commercial/industrial information

Data protection:
- ICO controller/processor contracts and liabilities

Fraud:
- PSFA Full Fraud Risk Assessment publication page
  (kept because it is the official source URL; substantive child links are
   preserved in links.jsonl and are NOT followed automatically)

Outputs
-------
<output_dir>/
    corpus_manifest.json
    documents.jsonl
    all_links.jsonl
    all_headings.jsonl
    all_updates.jsonl
    failures.jsonl
    <source_key>/
        document.json
        content_with_link_placeholders.txt
        content_with_link_placeholders.html
        links.jsonl
        headings.jsonl
        updates.jsonl
        raw/
            rendered_page.html
            govuk_content_api.json        (GOV.UK when available)
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
    # ------------------------------------------------------------------
    # COMPETITION / BID-RIGGING
    # ------------------------------------------------------------------
    "CMA_BID_RIGGING_PROCURERS": {
        "source_id": "CMA_BID_RIGGING_PROCURERS",
        "url": "https://www.gov.uk/government/publications/bid-rigging-advice-for-public-sector-procurers/bid-rigging-advice-for-public-sector-procurers",
        "publisher": "Competition and Markets Authority",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["BID_RIGGING", "COLLUSION", "COMPETITION", "PROCUREMENT"],
    },
    "CMA_PA23_EXCLUSION_DEBARMENT": {
        "source_id": "CMA_PA23_EXCLUSION_DEBARMENT",
        "url": "https://www.gov.uk/government/publications/the-procurement-act-2023-information-note-on-exclusion-and-debarment-on-competition-grounds/exclusion-and-debarment-on-competition-grounds-what-suppliers-and-contractors-need-to-know",
        "publisher": "Competition and Markets Authority",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["PROCUREMENT_ACT_2023", "EXCLUSION", "DEBARMENT", "COMPETITION"],
    },
    "CMA_COMPETITION_PUBLIC_AUTHORITIES": {
        "source_id": "CMA_COMPETITION_PUBLIC_AUTHORITIES",
        "url": "https://www.gov.uk/government/publications/competition-law-guide-for-public-authorities/competition-law-guide-for-public-authorities",
        "publisher": "Competition and Markets Authority",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["COMPETITION_LAW", "PUBLIC_AUTHORITIES"],
    },
    "PSFA_BID_RIGGING_PRACTICE_NOTE": {
        "source_id": "PSFA_BID_RIGGING_PRACTICE_NOTE",
        "url": "https://www.gov.uk/government/publications/bid-rigging-risk-in-all-procurement/bid-rigging-risk-in-all-procurement-practice-note-html",
        "publisher": "Public Sector Fraud Authority",
        "evidence_class": "OFFICIAL_PRACTICE_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["BID_RIGGING", "FRAUD", "CARTELS", "PROCUREMENT"],
    },

    # ------------------------------------------------------------------
    # NATIONAL SECURITY
    # ------------------------------------------------------------------
    "NSUP": {
        "source_id": "GOVUK_NSUP",
        "url": "https://www.gov.uk/guidance/the-national-security-unit-for-procurement",
        "publisher": "Cabinet Office",
        "evidence_class": "OFFICIAL_SPECIALIST_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["NATIONAL_SECURITY", "DEBARMENT", "EXCLUSION", "TERMINATION"],
    },
    "PA23_NATIONAL_SECURITY_EXCLUSIONS_ANNEX": {
        "source_id": "PA23_NATIONAL_SECURITY_EXCLUSIONS_ANNEX",
        "url": "https://www.gov.uk/government/publications/procurement-act-2023-guidance-documents-procure-phase/guidance-exclusions-annex-2-national-security-grounds-html",
        "publisher": "Cabinet Office",
        "evidence_class": "OFFICIAL_PA23_TECHNICAL_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["PROCUREMENT_ACT_2023", "NATIONAL_SECURITY", "EXCLUSION"],
    },

    # ------------------------------------------------------------------
    # INFORMATION DISCLOSURE / FOIA / EIR
    # ------------------------------------------------------------------
    "ICO_FOIA_SECTION_43": {
        "source_id": "ICO_FOIA_SECTION_43",
        "url": "https://ico.org.uk/for-organisations/foi/freedom-of-information-and-environmental-information-regulations/section-43-commercial-interests/",
        "publisher": "Information Commissioner's Office",
        "evidence_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "authority_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["FOIA", "COMMERCIAL_INTERESTS", "DISCLOSURE", "PROCUREMENT"],
    },
    "ICO_OUTSOURCING_FOIA_EIR": {
        "source_id": "ICO_OUTSOURCING_FOIA_EIR",
        "url": "https://ico.org.uk/for-organisations/foi/freedom-of-information-and-environmental-information-regulations/outsourcing-foia-and-eir-obligations/",
        "publisher": "Information Commissioner's Office",
        "evidence_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "authority_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["FOIA", "EIR", "OUTSOURCING", "CONTRACTS", "PROCUREMENT"],
    },
    "ICO_EIR_12_5_E": {
        "source_id": "ICO_EIR_12_5_E",
        "url": "https://ico.org.uk/for-organisations/foi/freedom-of-information-and-environmental-information-regulations/regulation-12-5-e-commercial-or-industrial-information/",
        "publisher": "Information Commissioner's Office",
        "evidence_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "authority_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["EIR", "COMMERCIAL_INFORMATION", "DISCLOSURE", "PROCUREMENT"],
    },

    # ------------------------------------------------------------------
    # DATA PROTECTION
    # ------------------------------------------------------------------
    "ICO_CONTROLLER_PROCESSOR_CONTRACTS": {
        "source_id": "ICO_CONTROLLER_PROCESSOR_CONTRACTS",
        "url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/contracts-and-liabilities-between-controllers-and-processors-multi/",
        "publisher": "Information Commissioner's Office",
        "evidence_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "authority_class": "OFFICIAL_REGULATOR_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["UK_GDPR", "CONTROLLER", "PROCESSOR", "CONTRACTS", "PROCUREMENT"],
    },

    # ------------------------------------------------------------------
    # FRAUD - OPTIONAL SUPPORTING SOURCE
    # ------------------------------------------------------------------
    "PSFA_FULL_FRAUD_RISK_ASSESSMENT": {
        "source_id": "PSFA_FULL_FRAUD_RISK_ASSESSMENT",
        "url": "https://www.gov.uk/government/publications/full-fraud-risk-assessment-practice-note",
        "publisher": "Public Sector Fraud Authority",
        "evidence_class": "OFFICIAL_PRACTICE_GUIDANCE",
        "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
        "retrieval_lane": "GUIDANCE",
        "topic": ["FRAUD_RISK", "PUBLIC_SECTOR", "PROCUREMENT_SUPPORTING"],
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


class AssociatedLawHtmlScraper:
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
            "User-Agent": (
                "MastersThesisProcurementResearchBot/1.0 "
                "(academic reproducible corpus acquisition)"
            ),
            "Accept-Language": "en-GB,en;q=0.9",
        })
        self.fetch_log = []

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GOV.UK API
    # ------------------------------------------------------------------

    def is_govuk(self):
        return urlparse(self.source["url"]).netloc == "www.gov.uk"

    def content_api_url(self):
        return "https://www.gov.uk/api/content" + urlparse(self.source["url"]).path

    def fetch_api(self):
        if not self.is_govuk():
            return None

        api_url = self.content_api_url()

        try:
            data, _, _ = self.fetch(
                api_url,
                "application/json,*/*;q=0.5"
            )
            (self.raw_dir / "govuk_content_api.json").write_bytes(data)
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            (self.raw_dir / "govuk_content_api_error.json").write_text(
                json.dumps({
                    "attempted_url": api_url,
                    "error": str(e),
                }, indent=2),
                encoding="utf-8",
            )
            return None

    # ------------------------------------------------------------------
    # Body extraction
    # ------------------------------------------------------------------

    def get_body_html(self, rendered_soup, api):
        # GOV.UK: API body is preferred.
        if api:
            body = (api.get("details") or {}).get("body")
            if isinstance(body, str):
                body_text = clean_ws(
                    BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
                )
                if body_text:
                    return body, "GOVUK_CONTENT_API_DETAILS_BODY"

        # Generic rendered-page fallback.
        main = (
            rendered_soup.find("main")
            or rendered_soup.find("article")
            or rendered_soup
        )

        if self.is_govuk():
            for c in [
                main.select_one(".govspeak"),
                main.select_one(".gem-c-govspeak"),
                main.select_one("article"),
            ]:
                if c and clean_ws(c.get_text(" ", strip=True)):
                    return str(c), "RENDERED_GOVUK_MAIN_BODY"

        # ICO: prefer the central article/content region.
        host = urlparse(self.source["url"]).netloc.lower()
        if "ico.org.uk" in host:
            selectors = [
                "main article",
                "article",
                ".article-content",
                ".content",
                "main",
            ]
            for selector in selectors:
                c = rendered_soup.select_one(selector)
                if c and clean_ws(c.get_text(" ", strip=True)):
                    return str(c), "RENDERED_ICO_CONTENT"

        if clean_ws(main.get_text(" ", strip=True)):
            return str(main), "RENDERED_MAIN_FALLBACK"

        raise RuntimeError("Could not identify substantive page content")

    def clean_body(self, soup):
        for t in soup.find_all([
            "script", "style", "noscript", "form", "button", "input",
            "select", "textarea", "svg",
        ]):
            t.decompose()

        selectors = [
            ".gem-c-print-link",
            ".gem-c-feedback",
            ".gem-c-subscription-links",
            ".subscription-links",
            ".govuk-breadcrumbs",
            ".gem-c-contextual-sidebar",

            # Common ICO/site chrome patterns; removed only if nested in the
            # selected content region.
            ".breadcrumb",
            ".breadcrumbs",
            ".social-share",
            ".share",
            ".cookie",
            ".feedback",
        ]

        for selector in selectors:
            for tag in soup.select(selector):
                tag.decompose()

    # ------------------------------------------------------------------
    # Link placeholders
    # ------------------------------------------------------------------

    def nearest_heading(self, node):
        h = node.find_previous([
            "h1", "h2", "h3", "h4", "h5", "h6"
        ])
        return clean_ws(h.get_text(" ", strip=True)) if h else None

    def context(self, a, n=800):
        p = a.find_parent(["p", "li", "td", "th", "div", "section"])
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

        if "ico.org.uk" in host:
            return "ICO_INTERNAL"

        return "EXTERNAL"

    def replace_links(self, soup, source_url):
        out = []
        occurrence = 0

        for a in list(soup.find_all("a", href=True)):
            href = clean_ws(a.get("href"))
            if not href:
                continue

            occurrence += 1
            placeholder = f"[[LINK_{occurrence:04d}]]"

            absolute_url = urljoin(source_url, href)
            defragmented_url, fragment = urldefrag(absolute_url)
            anchor_text = clean_ws(a.get_text(" ", strip=True))

            out.append({
                "source_key": self.source_key,
                "source_id": self.source["source_id"],
                "placeholder": placeholder,
                "occurrence": occurrence,
                "anchor_text": anchor_text,
                "raw_href": href,
                "absolute_url": absolute_url,
                "defragmented_url": defragmented_url,
                "fragment": fragment or None,
                "link_class": self.classify_link(absolute_url),
                "source_heading": self.nearest_heading(a),
                "source_context": self.context(a),
                "followed": False,
                "target_document_id": None,
                "resolution_status": "UNRESOLVED",
                "edge_status": "NOT_CREATED",
            })

            replacement = (
                f"{anchor_text} {placeholder}"
                if anchor_text
                else placeholder
            )
            a.replace_with(NavigableString(replacement))

        return out

    # ------------------------------------------------------------------
    # Structured text / headings
    # ------------------------------------------------------------------

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
                lines += [
                    "",
                    "#" * int(n.name[1]) + " " + text,
                    "",
                ]

            elif n.name == "li":
                lines.append("- " + text)

            elif n.name == "tr":
                cells = [
                    clean_ws(c.get_text(" ", strip=True))
                    for c in n.find_all(
                        ["th", "td"],
                        recursive=False
                    )
                ]
                if any(cells):
                    lines.append(
                        " | ".join([c for c in cells if c])
                    )

            elif n.name == "blockquote":
                lines.append("> " + text)

            else:
                lines += [text, ""]

        text = "\n".join(lines).strip()

        if len(clean_ws(text)) < 300:
            text = soup.get_text("\n", strip=True)

        return text

    def headings(self, soup):
        out = []
        stack = []

        for i, h in enumerate(
            soup.find_all([
                "h1", "h2", "h3", "h4", "h5", "h6"
            ]),
            1,
        ):
            level = int(h.name[1])
            text = clean_ws(h.get_text(" ", strip=True))
            if not text:
                continue

            stack = [x for x in stack if x[0] < level]
            stack.append((level, text))

            out.append({
                "source_key": self.source_key,
                "source_id": self.source["source_id"],
                "ordinal": i,
                "heading_level": level,
                "heading": text,
                "heading_path": [x[1] for x in stack],
            })

        return out

    # ------------------------------------------------------------------
    # Update history
    # ------------------------------------------------------------------

    def updates(self, soup):
        out = []

        # GOV.UK style
        for h in soup.find_all(["h2", "h3"]):
            heading_text = clean_ws(
                h.get_text(" ", strip=True)
            ).lower()

            if "updates to this page" not in heading_text:
                continue

            level = int(h.name[1])
            parts = []

            for s in h.next_siblings:
                if (
                    isinstance(s, Tag)
                    and s.name in {
                        "h1", "h2", "h3", "h4", "h5", "h6"
                    }
                    and int(s.name[1]) <= level
                ):
                    break

                if isinstance(s, Tag):
                    text = clean_ws(
                        s.get_text(" ", strip=True)
                    )
                    if text:
                        parts.append(text)

            combined = " ".join(parts)

            for m in re.finditer(
                r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+"
                r"(.*?)(?=(?:\d{1,2}\s+[A-Za-z]+\s+\d{4})|$)",
                combined,
            ):
                out.append({
                    "source_key": self.source_key,
                    "source_id": self.source["source_id"],
                    "date_text": m.group(1),
                    "note": clean_ws(m.group(2)),
                })

            break

        return out

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        raw, final_url, content_type = self.fetch(
            self.source["url"],
            "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"
        )
        (self.raw_dir / "rendered_page.html").write_bytes(raw)

        api = self.fetch_api()

        rendered = BeautifulSoup(raw, "html.parser")
        body_html, extraction_method = self.get_body_html(
            rendered,
            api
        )

        body = BeautifulSoup(body_html, "html.parser")
        self.clean_body(body)

        links = self.replace_links(
            body,
            final_url
        )
        text = self.structured_text(body)
        html = str(body)
        heads = self.headings(body)
        ups = self.updates(rendered)

        (self.output_dir / "content_with_link_placeholders.txt").write_text(
            text,
            encoding="utf-8"
        )

        (self.output_dir / "content_with_link_placeholders.html").write_text(
            html,
            encoding="utf-8"
        )

        write_jsonl(
            self.output_dir / "links.jsonl",
            links
        )
        write_jsonl(
            self.output_dir / "headings.jsonl",
            heads
        )
        write_jsonl(
            self.output_dir / "updates.jsonl",
            ups
        )

        title = None
        if api:
            title = api.get("title")

        if not title:
            h1 = rendered.find("h1")
            if h1:
                title = clean_ws(
                    h1.get_text(" ", strip=True)
                )

        rendered_text_lower = clean_ws(
            rendered.get_text(" ", strip=True)
        ).lower()

        document_status = (
            "WITHDRAWN"
            if (
                "this guidance was withdrawn" in rendered_text_lower
                or "[withdrawn]" in rendered_text_lower
            )
            else "CURRENT_OR_UNMARKED"
        )

        doc = {
            "source_key": self.source_key,
            "source_id": self.source["source_id"],
            "canonical_url": self.source["url"],
            "final_url": final_url,
            "title": title,
            "publisher": self.source["publisher"],
            "source_type": "OFFICIAL_HTML_GUIDANCE",
            "evidence_class": self.source["evidence_class"],
            "authority_class": self.source["authority_class"],
            "retrieval_lane": self.source["retrieval_lane"],
            "topic": self.source["topic"],
            "document_status": document_status,
            "content_id": (api or {}).get("content_id"),
            "document_type": (api or {}).get("document_type"),
            "schema_name": (api or {}).get("schema_name"),
            "published_at": (api or {}).get("first_published_at"),
            "public_updated_at": (api or {}).get("public_updated_at"),
            "publishing_app": (api or {}).get("publishing_app"),
            "locale": (api or {}).get("locale"),
            "extraction_method": extraction_method,
            "content_type": content_type,
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
            json.dumps(
                doc,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        manifest = {
            "scraper_version": VERSION,
            "generated_at": utc_now(),
            "source_key": self.source_key,
            "source_id": self.source["source_id"],
            "source_url": self.source["url"],
            "scope_policy": {
                "scrape_configured_page_only": True,
                "preserve_raw_rendered_html": True,
                "prefer_govuk_content_api_body": True,
                "extract_substantive_html": True,
                "replace_links_with_placeholders": True,
                "store_links_separately": True,
                "follow_links": False,
                "download_linked_assets": False,
                "chunk_content": False,
                "infer_edges": False,
            },
            "placeholder_policy": {
                "format": "[[LINK_NNNN]]",
                "numbering": "first occurrence order within source page",
                "same_url_can_have_multiple_placeholders": True,
                "placeholder_represents_occurrence_not_unique_url": True,
            },
            "fetch_log": self.fetch_log,
        }

        (self.output_dir / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return doc, links, heads, ups


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/associated_law_guidance"
    )

    parser.add_argument(
        "--source",
        choices=["ALL"] + sorted(SOURCES.keys()),
        default="ALL"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=0.4
    )

    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = (
        sorted(SOURCES.keys())
        if args.source == "ALL"
        else [args.source]
    )

    docs = []
    all_links = []
    all_headings = []
    all_updates = []
    failures = []

    for key in selected:
        out_dir = root / key.lower()

        try:
            doc, links, heads, ups = AssociatedLawHtmlScraper(
                key,
                out_dir,
                timeout=args.timeout,
                pause=args.pause
            ).run()

            docs.append(doc)
            all_links.extend(links)
            all_headings.extend(heads)
            all_updates.extend(ups)

            print(
                f"Completed {key}: {out_dir}"
            )

        except Exception as e:
            failure = {
                "source_key": key,
                "source_url": SOURCES[key]["url"],
                "error": str(e),
            }
            failures.append(failure)

            print(
                f"ERROR {key}: {e}",
                file=sys.stderr
            )

    write_jsonl(
        root / "documents.jsonl",
        docs
    )

    write_jsonl(
        root / "all_links.jsonl",
        all_links
    )

    write_jsonl(
        root / "all_headings.jsonl",
        all_headings
    )

    write_jsonl(
        root / "all_updates.jsonl",
        all_updates
    )

    write_jsonl(
        root / "failures.jsonl",
        failures
    )

    corpus_manifest = {
        "scraper_version": VERSION,
        "generated_at": utc_now(),
        "requested_sources": selected,
        "successful_documents": len(docs),
        "failure_count": len(failures),
        "pipeline": (
            "raw HTML -> substantive HTML -> link placeholders -> "
            "LLM semantic chunking later -> target scraping/resolution later -> "
            "graph edges later"
        ),
        "sources": SOURCES,
    }

    (root / "corpus_manifest.json").write_text(
        json.dumps(
            corpus_manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
