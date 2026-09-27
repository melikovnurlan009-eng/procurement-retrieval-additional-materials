#!/usr/bin/env python3
"""
Scrape the CURRENT Procurement Compliance & Oversight GOV.UK guidance page.

Source:
https://www.gov.uk/guidance/procurement-compliance-oversight

Designed for the thesis pipeline:

    GOV.UK page
      -> preserve raw HTML
      -> extract substantive main HTML
      -> replace hyperlinks with stable placeholders
      -> save clean text for later LLM semantic chunking
      -> store every link separately
      -> later scrape/resolve selected links
      -> later create graph edges

This script deliberately DOES NOT:
- recursively crawl linked pages;
- download linked PDFs/assets;
- chunk the content;
- infer graph edges;
- merge referenced documents into this page.

It preserves the page as one source document and captures the link evidence
needed for a later reference-resolution/graph stage.

Outputs
-------
<output_dir>/
    manifest.json
    document.json
    content_with_link_placeholders.txt
    content_with_link_placeholders.html
    links.jsonl
    headings.jsonl
    updates.jsonl
    raw/
        rendered_page.html
        govuk_content_api.json

Placeholder example
-------------------
Original HTML:
    <a href="/guidance/example">Procurement Compliance Service</a>

Normalized text:
    Procurement Compliance Service [[LINK_0007]]

links.jsonl:
    {
      "placeholder": "[[LINK_0007]]",
      "anchor_text": "Procurement Compliance Service",
      "absolute_url": "https://www.gov.uk/guidance/example",
      ...
    }

Later, your LLM can chunk the normalized content while retaining placeholders.
Afterwards you can map placeholders back to URLs and create typed edges.
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

SOURCE_URL = "https://www.gov.uk/guidance/procurement-compliance-oversight"
SOURCE_ID = "GOVUK_PROCUREMENT_COMPLIANCE_OVERSIGHT"
CONTENT_API_URL = "https://www.gov.uk/api/content/guidance/procurement-compliance-oversight"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_ws(text: str | None) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    """
    Preserve paragraph/list separation while normalizing repeated spaces.
    """
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


class ProcurementComplianceOversightScraper:
    def __init__(
        self,
        output_dir: Path,
        timeout: int = 60,
        pause: float = 0.4,
    ):
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
        self.fetch_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def fetch(self, url: str, accept: str) -> tuple[bytes, str, str]:
        r = self.session.get(
            url,
            headers={"Accept": accept},
            timeout=self.timeout,
            allow_redirects=True,
        )
        r.raise_for_status()

        data = r.content
        content_type = r.headers.get("content-type", "")

        self.fetch_log.append(asdict(FetchRecord(
            requested_url=url,
            final_url=r.url,
            status_code=r.status_code,
            content_type=content_type,
            retrieved_at=utc_now(),
            sha256=sha256_bytes(data),
            byte_count=len(data),
        )))

        time.sleep(self.pause)
        return data, r.url, content_type

    # ------------------------------------------------------------------
    # Main acquisition
    # ------------------------------------------------------------------

    def run(self) -> Path:
        rendered_bytes, final_url, rendered_ctype = self.fetch(
            SOURCE_URL,
            "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
        )
        (self.raw_dir / "rendered_page.html").write_bytes(rendered_bytes)

        api_payload = self.fetch_content_api()

        rendered_soup = BeautifulSoup(rendered_bytes, "html.parser")
        title = self.extract_title(rendered_soup, api_payload)
        metadata = self.extract_metadata(rendered_soup, api_payload)

        # Prefer the GOV.UK Content API body because it excludes navigation,
        # cookie banners, global footer, etc. Fall back to rendered <main>.
        body_html, extraction_method = self.obtain_substantive_body(
            rendered_soup,
            api_payload,
        )

        body_soup = BeautifulSoup(body_html, "html.parser")
        self.clean_substantive_html(body_soup)

        # Capture links BEFORE replacement.
        links = self.replace_links_with_placeholders(
            body_soup=body_soup,
            source_url=final_url,
        )

        # Headings refer to the placeholder-bearing HTML.
        headings = self.extract_headings(body_soup)

        normalized_html = self.serialize_body(body_soup)
        normalized_text = self.html_to_structured_text(body_soup)

        # Save the exact input intended for the later LLM chunker.
        (self.output_dir / "content_with_link_placeholders.html").write_text(
            normalized_html,
            encoding="utf-8",
        )
        (self.output_dir / "content_with_link_placeholders.txt").write_text(
            normalized_text,
            encoding="utf-8",
        )
        write_jsonl(self.output_dir / "links.jsonl", links)
        write_jsonl(self.output_dir / "headings.jsonl", headings)

        updates = self.extract_updates(rendered_soup)
        write_jsonl(self.output_dir / "updates.jsonl", updates)

        document = {
            "source_id": SOURCE_ID,
            "canonical_url": SOURCE_URL,
            "final_url": final_url,
            "title": title,
            "source_type": "GOVUK_GUIDANCE",
            "evidence_class": "OFFICIAL_OVERSIGHT_GUIDANCE",
            "authority_class": "OFFICIAL_GOVERNMENT_GUIDANCE",
            "retrieval_lane": "GUIDANCE",
            "document_status": self.infer_status(rendered_soup),
            "published_at": metadata.get("published_at"),
            "public_updated_at": metadata.get("public_updated_at"),
            "content_id": metadata.get("content_id"),
            "document_type": metadata.get("document_type"),
            "schema_name": metadata.get("schema_name"),
            "publishing_app": metadata.get("publishing_app"),
            "locale": metadata.get("locale"),
            "extraction_method": extraction_method,
            "content_text_sha256": sha256_text(normalized_text),
            "content_html_sha256": sha256_text(normalized_html),
            "content_char_count": len(normalized_text),
            "link_count": len(links),
            "heading_count": len(headings),
            "update_event_count": len(updates),
            "chunking_status": "NOT_CHUNKED",
            "link_resolution_status": "NOT_FOLLOWED",
            "graph_edge_status": "NOT_CREATED",
        }

        (self.output_dir / "document.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest = {
            "scraper_version": VERSION,
            "generated_at": utc_now(),
            "source_id": SOURCE_ID,
            "source_url": SOURCE_URL,
            "scope_policy": {
                "scrape_current_page_only": True,
                "preserve_raw_rendered_html": True,
                "prefer_govuk_content_api_body": True,
                "preserve_full_substantive_html": True,
                "replace_links_with_placeholders": True,
                "store_links_separately": True,
                "follow_links": False,
                "download_linked_assets": False,
                "chunk_content": False,
                "infer_edges": False,
            },
            "placeholder_policy": {
                "format": "[[LINK_NNNN]]",
                "numbering": "first occurrence order",
                "same_url_can_have_multiple_placeholders": True,
                "reason": (
                    "A link placeholder identifies an occurrence in the source text. "
                    "Occurrences are not deduplicated because the same URL may be cited "
                    "from different contexts and later produce different provenance."
                ),
            },
            "recommended_next_stage": [
                "Give content_with_link_placeholders.txt or .html to the LLM chunker.",
                "Require each generated chunk to retain every link placeholder present in its source span.",
                "Use links.jsonl to resolve each placeholder after chunking.",
                "Scrape selected linked documents independently as their own source documents.",
                "Resolve canonical target IDs.",
                "Create graph edges only after target resolution.",
            ],
            "fetch_log": self.fetch_log,
        }

        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(json.dumps({
            "source_id": SOURCE_ID,
            "title": title,
            "status": document["document_status"],
            "published_at": document["published_at"],
            "public_updated_at": document["public_updated_at"],
            "content_chars": len(normalized_text),
            "links": len(links),
            "headings": len(headings),
            "updates": len(updates),
            "output_dir": str(self.output_dir),
        }, indent=2, ensure_ascii=False))

        return self.output_dir

    # ------------------------------------------------------------------
    # GOV.UK Content API
    # ------------------------------------------------------------------

    def fetch_content_api(self) -> dict[str, Any] | None:
        try:
            data, _, _ = self.fetch(
                CONTENT_API_URL,
                "application/json,*/*;q=0.5",
            )
            (self.raw_dir / "govuk_content_api.json").write_bytes(data)
            return json.loads(data.decode("utf-8"))
        except Exception as exc:
            # Rendered HTML remains a complete fallback.
            error = {
                "error": str(exc),
                "attempted_url": CONTENT_API_URL,
                "retrieved_at": utc_now(),
            }
            (self.raw_dir / "govuk_content_api_error.json").write_text(
                json.dumps(error, indent=2),
                encoding="utf-8",
            )
            return None

    def obtain_substantive_body(
        self,
        rendered_soup: BeautifulSoup,
        api_payload: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if api_payload:
            details = api_payload.get("details") or {}
            body = details.get("body")
            if isinstance(body, str) and clean_ws(
                BeautifulSoup(body, "html.parser").get_text(" ", strip=True)
            ):
                return body, "GOVUK_CONTENT_API_DETAILS_BODY"

        main = rendered_soup.find("main")
        if main is None:
            raise RuntimeError("Could not find GOV.UK main content.")

        # Prefer the govspeak/content section within <main>.
        candidates = [
            main.select_one(".govspeak"),
            main.select_one(".gem-c-govspeak"),
            main.select_one("article"),
        ]
        for candidate in candidates:
            if candidate and clean_ws(candidate.get_text(" ", strip=True)):
                return str(candidate), "RENDERED_HTML_MAIN_BODY"

        return str(main), "RENDERED_HTML_MAIN_FALLBACK"

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_substantive_html(self, soup: BeautifulSoup) -> None:
        # Only non-content mechanics are removed. Lists, tables, headings,
        # blockquotes, emphasis, images, etc. are preserved.
        for tag in soup.find_all([
            "script", "style", "noscript", "form", "button", "input",
            "select", "textarea",
        ]):
            tag.decompose()

        # Remove print/subscription/feedback controls if the rendered fallback
        # happens to include them.
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

    # ------------------------------------------------------------------
    # Link placeholders
    # ------------------------------------------------------------------

    def replace_links_with_placeholders(
        self,
        body_soup: BeautifulSoup,
        source_url: str,
    ) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        occurrence = 0

        for anchor in list(body_soup.find_all("a", href=True)):
            raw_href = clean_ws(anchor.get("href"))
            if not raw_href:
                continue

            occurrence += 1
            placeholder = f"[[LINK_{occurrence:04d}]]"

            absolute_url = urljoin(source_url, raw_href)
            defragmented_url, fragment = urldefrag(absolute_url)
            anchor_text = clean_ws(anchor.get_text(" ", strip=True))

            parent_heading = self.nearest_previous_heading(anchor)
            surrounding_text = self.surrounding_context(anchor)

            link_record = {
                "source_id": SOURCE_ID,
                "placeholder": placeholder,
                "occurrence": occurrence,
                "anchor_text": anchor_text,
                "raw_href": raw_href,
                "absolute_url": absolute_url,
                "defragmented_url": defragmented_url,
                "fragment": fragment or None,
                "link_class": self.classify_link(absolute_url),
                "source_heading": parent_heading,
                "source_context": surrounding_text,
                "followed": False,
                "target_document_id": None,
                "resolution_status": "UNRESOLVED",
                "edge_status": "NOT_CREATED",
            }
            links.append(link_record)

            # Retain readable anchor text and append a deterministic placeholder.
            replacement_text = (
                f"{anchor_text} {placeholder}" if anchor_text else placeholder
            )
            anchor.replace_with(NavigableString(replacement_text))

        return links

    def classify_link(self, url: str) -> str:
        p = urlparse(url)
        host = p.netloc.lower()
        path = p.path.lower()

        if not host and url.startswith("#"):
            return "INTERNAL_FRAGMENT"
        if host.endswith("legislation.gov.uk"):
            return "LEGISLATION"
        if host == "assets.publishing.service.gov.uk":
            if path.endswith(".pdf"):
                return "GOVUK_PDF_ATTACHMENT"
            return "GOVUK_ASSET"
        if host == "www.gov.uk":
            if path.startswith("/guidance/"):
                return "GOVUK_GUIDANCE"
            if path.startswith("/government/publications/"):
                return "GOVUK_PUBLICATION"
            if path.startswith("/government/collections/"):
                return "GOVUK_COLLECTION"
            return "GOVUK_PAGE"
        if host.endswith(".gov.uk") or host.endswith("gov.uk"):
            return "OTHER_GOVUK_SERVICE"
        return "EXTERNAL"

    def nearest_previous_heading(self, node: Tag) -> str | None:
        heading = node.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        return clean_ws(heading.get_text(" ", strip=True)) if heading else None

    def surrounding_context(self, anchor: Tag, max_chars: int = 600) -> str:
        container = anchor.find_parent(["p", "li", "td", "th", "div"])
        if container is None:
            return clean_ws(anchor.get_text(" ", strip=True))
        text = clean_ws(container.get_text(" ", strip=True))
        return text[:max_chars]

    # ------------------------------------------------------------------
    # Structured text and headings
    # ------------------------------------------------------------------

    def serialize_body(self, soup: BeautifulSoup) -> str:
        # Pretty printing can introduce formatting noise. str() retains source
        # structure adequately for later DOM-aware LLM preprocessing.
        return str(soup)

    def html_to_structured_text(self, soup: BeautifulSoup) -> str:
        """
        Produce an LLM-friendly text representation while preserving:
        - headings;
        - paragraphs;
        - bullets;
        - table rows;
        - link placeholders already inserted into text.
        """
        lines: list[str] = []

        # Avoid nested duplicate extraction by handling selected block tags.
        for node in soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "tr", "blockquote",
        ]):
            text = clean_ws(node.get_text(" ", strip=True))
            if not text:
                continue

            if node.name and re.fullmatch(r"h[1-6]", node.name):
                level = int(node.name[1])
                lines.append("")
                lines.append("#" * level + " " + text)
                lines.append("")
            elif node.name == "li":
                lines.append("- " + text)
            elif node.name == "tr":
                cells = [
                    clean_ws(cell.get_text(" ", strip=True))
                    for cell in node.find_all(["th", "td"], recursive=False)
                ]
                cells = [x for x in cells if x]
                if cells:
                    lines.append(" | ".join(cells))
            elif node.name == "blockquote":
                lines.append("> " + text)
            else:
                lines.append(text)
                lines.append("")

        text = "\n".join(lines)

        # Fallback if block extraction unexpectedly produces little content.
        if len(clean_ws(text)) < 500:
            text = soup.get_text("\n", strip=True)

        return normalize_text(text)

    def extract_headings(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        headings = []
        stack: list[tuple[int, str]] = []

        for ordinal, heading in enumerate(
            soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]),
            start=1,
        ):
            level = int(heading.name[1])
            text = clean_ws(heading.get_text(" ", strip=True))
            if not text:
                continue

            stack = [x for x in stack if x[0] < level]
            stack.append((level, text))

            headings.append({
                "source_id": SOURCE_ID,
                "ordinal": ordinal,
                "heading_level": level,
                "heading": text,
                "heading_path": [x[1] for x in stack],
            })

        return headings

    # ------------------------------------------------------------------
    # Metadata/status/updates
    # ------------------------------------------------------------------

    def extract_title(
        self,
        soup: BeautifulSoup,
        api_payload: dict[str, Any] | None,
    ) -> str:
        if api_payload and api_payload.get("title"):
            return clean_ws(api_payload["title"])
        h1 = soup.find("h1")
        return clean_ws(h1.get_text(" ", strip=True)) if h1 else ""

    def extract_metadata(
        self,
        soup: BeautifulSoup,
        api_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if api_payload:
            return {
                "content_id": api_payload.get("content_id"),
                "document_type": api_payload.get("document_type"),
                "schema_name": api_payload.get("schema_name"),
                "publishing_app": api_payload.get("publishing_app"),
                "locale": api_payload.get("locale"),
                "published_at": api_payload.get("first_published_at"),
                "public_updated_at": api_payload.get("public_updated_at"),
            }

        text = clean_ws(soup.get_text(" ", strip=True))
        published = re.search(
            r"Published:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            text,
            re.I,
        )
        updated = re.search(
            r"Last updated:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            text,
            re.I,
        )
        return {
            "content_id": None,
            "document_type": "guidance",
            "schema_name": None,
            "publishing_app": None,
            "locale": "en",
            "published_at": published.group(1) if published else None,
            "public_updated_at": updated.group(1) if updated else None,
        }

    def infer_status(self, soup: BeautifulSoup) -> str:
        text = clean_ws(soup.get_text(" ", strip=True)).lower()
        if "[withdrawn]" in text or "this guidance was withdrawn" in text:
            return "WITHDRAWN"
        return "CURRENT"

    def extract_updates(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        """
        Best-effort extraction of the GOV.UK update-history section from
        rendered HTML. The raw HTML remains authoritative if markup changes.
        """
        updates: list[dict[str, Any]] = []

        heading = None
        for h in soup.find_all(["h2", "h3"]):
            if "updates to this page" in clean_ws(h.get_text(" ", strip=True)).lower():
                heading = h
                break

        if not heading:
            return updates

        container = heading.find_parent()
        candidates = []

        # GOV.UK commonly renders change notes in list/article structures.
        if container:
            candidates.extend(container.find_all(["li", "article"], recursive=True))

        seen = set()
        for node in candidates:
            text = clean_ws(node.get_text(" ", strip=True))
            if not text:
                continue
            m = re.search(
                r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+(.*)",
                text,
            )
            if not m:
                continue
            date_text = m.group(1)
            note = clean_ws(m.group(2))
            key = (date_text, note)
            if key in seen:
                continue
            seen.add(key)
            updates.append({
                "source_id": SOURCE_ID,
                "date_text": date_text,
                "note": note,
            })

        return updates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/procurement_compliance_oversight",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pause", type=float, default=0.4)
    args = parser.parse_args()

    try:
        out = ProcurementComplianceOversightScraper(
            output_dir=Path(args.output_dir),
            timeout=args.timeout,
            pause=args.pause,
        ).run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Completed: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
