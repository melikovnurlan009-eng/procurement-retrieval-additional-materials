#!/usr/bin/env python3
"""
Group A legislation scraper for legislation.gov.uk.

Purpose
-------
Scrape ONE selected Group A legislation item fully, without crawling into any
other legislation. The scraper records:

- full Akoma Ntoso XML snapshot
- legal hierarchy/provision nodes
- exact provision text
- explicit reference links found inside the selected item
- annotations/notes, including commencement/amendment/modification/extent notes
  where they are present in the source XML
- document-level metadata and provenance
- optional "changes affecting this legislation" page as metadata ABOUT the
  selected item only; it never follows links to affecting Acts/SIs

Allowed Group A sources
-----------------------
PA2023  - Procurement Act 2023
PR2024  - Procurement Regulations 2024
PCR2015 - Public Contracts Regulations 2015

Important design rule
---------------------
References to external legislation are STORED as references only.
They are NEVER followed or scraped by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from lxml import etree


VERSION = "1.0.0"

SOURCES = {
    "PA2023": {
        "document_id": "UKPGA_2023_54",
        "title": "Procurement Act 2023",
        "type": "ukpga",
        "year": "2023",
        "number": "54",
        "root_url": "https://www.legislation.gov.uk/ukpga/2023/54",
        "contents_url": "https://www.legislation.gov.uk/ukpga/2023/54/contents",
    },
    "PR2024": {
        "document_id": "UKSI_2024_692",
        "title": "Procurement Regulations 2024",
        "type": "uksi",
        "year": "2024",
        "number": "692",
        "root_url": "https://www.legislation.gov.uk/uksi/2024/692",
        "contents_url": "https://www.legislation.gov.uk/uksi/2024/692/contents/made",
    },
    "PCR2015": {
        "document_id": "UKSI_2015_102",
        "title": "Public Contracts Regulations 2015",
        "type": "uksi",
        "year": "2015",
        "number": "102",
        "root_url": "https://www.legislation.gov.uk/uksi/2015/102",
        "contents_url": "https://www.legislation.gov.uk/uksi/2015/102/contents/made",
    },
}

# Structural element names commonly used by Akoma Ntoso legislation.
STRUCTURAL_NAMES = {
    "part",
    "chapter",
    "title",
    "section",
    "subsection",
    "article",
    "rule",
    "regulation",
    "paragraph",
    "subparagraph",
    "schedule",
    "attachment",
    "division",
    "hcontainer",
    "blockList",
}

NOTE_NAMES = {
    "note",
    "authorialNote",
    "editorialNote",
    "remark",
    "comment",
    "annotation",
}

REFERENCE_NAMES = {
    "ref",
    "rref",
    "term",
    "def",
}

# UK revised-legislation annotation categories.
ANNOTATION_PATTERNS = {
    "TEXTUAL_AMENDMENT": re.compile(r"(^|\b)F\d+\b|textual amendment", re.I),
    "MODIFICATION": re.compile(r"(^|\b)C\d+\b|modification|modified|applied|excluded|extended|restricted", re.I),
    "COMMENCEMENT": re.compile(r"(^|\b)I\d+\b|commencement|coming into force|in force", re.I),
    "EXTENT": re.compile(r"(^|\b)E\d+\b|extent", re.I),
    "MARGINAL_CITATION": re.compile(r"(^|\b)M\d+\b", re.I),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def localname(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def clean_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def element_text(el: etree._Element) -> str:
    return clean_ws(" ".join(el.itertext()))


def first_child_text(el: etree._Element, child_names: set[str]) -> str:
    for child in el:
        if localname(child.tag) in child_names:
            value = element_text(child)
            if value:
                return value
    return ""


def get_attr(el: etree._Element, names: Iterable[str]) -> str | None:
    for name in names:
        if name in el.attrib:
            return el.attrib[name]
    # Also accept namespaced attributes by local name.
    for key, value in el.attrib.items():
        if localname(key) in names:
            return value
    return None


def nearest_structural_ancestor(el: etree._Element) -> etree._Element | None:
    cur = el.getparent()
    while cur is not None:
        if localname(cur.tag) in STRUCTURAL_NAMES and get_attr(cur, {"eId", "id"}):
            return cur
        cur = cur.getparent()
    return None


def classify_annotation(text: str, attrs: dict[str, Any]) -> str:
    evidence = " ".join([text] + [str(v) for v in attrs.values() if v])
    for label, pattern in ANNOTATION_PATTERNS.items():
        if pattern.search(evidence):
            return label
    return "OTHER_ANNOTATION"


def looks_external_legislation_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "legislation.gov.uk" in host and bool(
        re.search(r"/(ukpga|uksi|asp|ssi|nia|nisr|anaw|mwa|ukla|ukcm|ukci)/", url)
    )


@dataclass
class FetchRecord:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    retrieved_at: str
    sha256: str
    byte_count: int


class LegislationScraper:
    def __init__(
        self,
        source_key: str,
        output_dir: Path,
        timeout: int = 45,
        pause: float = 0.5,
        include_changes_page: bool = True,
    ):
        if source_key not in SOURCES:
            raise ValueError(f"Unsupported source {source_key!r}. Allowed: {', '.join(SOURCES)}")
        self.source_key = source_key
        self.source = SOURCES[source_key]
        self.output_dir = output_dir
        self.timeout = timeout
        self.pause = pause
        self.include_changes_page = include_changes_page

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "MastersThesisProcurementResearchBot/1.0 "
                    "(research scraper; polite single-document retrieval)"
                ),
                "Accept": "*/*",
            }
        )
        self.fetch_log: list[dict[str, Any]] = []

    def fetch(self, url: str, accept: str | None = None) -> bytes:
        headers = {"Accept": accept} if accept else {}
        response = self.session.get(
            url,
            headers=headers,
            timeout=self.timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        data = response.content
        rec = FetchRecord(
            requested_url=url,
            final_url=response.url,
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
            retrieved_at=now_iso(),
            sha256=sha256_bytes(data),
            byte_count=len(data),
        )
        self.fetch_log.append(asdict(rec))
        time.sleep(self.pause)
        return data

    def get_akn(self) -> tuple[bytes, str]:
        """
        Fetch exactly ONE complete AKN representation of the selected legislation.

        Preferred endpoint:
            <root>/data.akn

        Fallbacks are representations of the SAME selected legislation only.
        No discovered external reference is ever fetched.
        """
        candidates = [
            f"{self.source['root_url']}/data.akn",
            f"{self.source['contents_url'].rstrip('/')}/data.akn",
        ]

        errors = []
        for url in candidates:
            try:
                data = self.fetch(
                    url,
                    accept="application/xml,text/xml,application/akn+xml;q=0.9,*/*;q=0.1",
                )
                # Basic XML sanity check.
                etree.fromstring(data)
                return data, url
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError("Could not retrieve AKN XML:\n" + "\n".join(errors))

    def parse_akn(self, data: bytes) -> dict[str, Any]:
        parser = etree.XMLParser(
            recover=True,
            remove_comments=False,
            huge_tree=True,
            resolve_entities=False,
            no_network=True,
        )
        root = etree.fromstring(data, parser=parser)

        nodes: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []

        # Build structural node records.
        for el in root.iter():
            lname = localname(el.tag)
            eid = get_attr(el, {"eId", "id"})
            if lname not in STRUCTURAL_NAMES or not eid:
                continue

            parent = nearest_structural_ancestor(el)
            parent_eid = get_attr(parent, {"eId", "id"}) if parent is not None else None

            num = first_child_text(el, {"num"})
            heading = first_child_text(el, {"heading", "subheading"})

            # Full provision text is preserved. This deliberately does not chunk.
            text = element_text(el)

            nodes.append(
                {
                    "document_id": self.source["document_id"],
                    "node_id": f"{self.source['document_id']}::{eid}",
                    "eId": eid,
                    "element_type": lname,
                    "parent_eId": parent_eid,
                    "number": num,
                    "heading": heading,
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )

        # Explicit references are recorded but NEVER followed.
        for el in root.iter():
            lname = localname(el.tag)
            if lname not in REFERENCE_NAMES:
                continue

            href = get_attr(el, {"href", "refersTo", "showAs"})
            if not href:
                continue

            anc = nearest_structural_ancestor(el)
            source_eid = get_attr(anc, {"eId", "id"}) if anc is not None else None
            display_text = element_text(el)

            absolute_url = href
            if href.startswith("/"):
                absolute_url = urljoin("https://www.legislation.gov.uk", href)
            elif href.startswith("#"):
                absolute_url = href
            elif not urlparse(href).scheme:
                absolute_url = urljoin(self.source["root_url"] + "/", href)

            references.append(
                {
                    "document_id": self.source["document_id"],
                    "source_eId": source_eid,
                    "display_text": display_text,
                    "raw_href": href,
                    "resolved_href": absolute_url,
                    "is_internal_fragment": absolute_url.startswith("#"),
                    "is_external_legislation_reference": looks_external_legislation_url(
                        absolute_url
                    ),
                    "followed": False,
                }
            )

        # Capture note/annotation-like elements generically.
        # This is intentionally schema-tolerant because legislation.gov.uk AKN can
        # contain annotation metadata in namespaced or UK-specific elements.
        for el in root.iter():
            lname = localname(el.tag)
            attrs = {localname(k): v for k, v in el.attrib.items()}
            marker = " ".join(str(v) for v in attrs.values())

            is_note_like = lname in NOTE_NAMES
            is_annotation_marked = bool(
                re.search(r"\b[FICEM]\d+\b", marker, flags=re.I)
            )

            if not (is_note_like or is_annotation_marked):
                continue

            text = element_text(el)
            if not text and not attrs:
                continue

            anc = nearest_structural_ancestor(el)
            source_eid = get_attr(anc, {"eId", "id"}) if anc is not None else None

            annotations.append(
                {
                    "document_id": self.source["document_id"],
                    "source_eId": source_eid,
                    "xml_element_type": lname,
                    "annotation_type": classify_annotation(text, attrs),
                    "text": text,
                    "attributes": attrs,
                }
            )

        # De-duplicate records while preserving order.
        def dedupe(records: list[dict[str, Any]], key_func):
            seen = set()
            out = []
            for record in records:
                key = key_func(record)
                if key in seen:
                    continue
                seen.add(key)
                out.append(record)
            return out

        references = dedupe(
            references,
            lambda r: (
                r["source_eId"],
                r["raw_href"],
                r["display_text"],
            ),
        )
        annotations = dedupe(
            annotations,
            lambda r: (
                r["source_eId"],
                r["annotation_type"],
                r["text"],
                json.dumps(r["attributes"], sort_keys=True),
            ),
        )

        return {
            "nodes": nodes,
            "references": references,
            "annotations": annotations,
        }

    def scrape_changes_affecting(self) -> dict[str, Any] | None:
        """
        Fetch the selected item's "changes affecting" page as metadata ABOUT
        this legislation only.

        It extracts rows/links but does not follow any affecting legislation.
        """
        if not self.include_changes_page:
            return None

        t = self.source["type"]
        year = self.source["year"]
        number = self.source["number"]
        url = f"https://www.legislation.gov.uk/changes/affected/{t}/{year}/{number}"

        try:
            html = self.fetch(url, accept="text/html,*/*;q=0.5")
        except Exception as exc:
            return {
                "url": url,
                "status": "FETCH_FAILED",
                "error": str(exc),
                "rows": [],
                "links": [],
            }

        soup = BeautifulSoup(html, "html.parser")

        rows: list[dict[str, Any]] = []
        for tr in soup.find_all("tr"):
            cells = [clean_ws(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if not cells:
                continue

            links = []
            for a in tr.find_all("a", href=True):
                href = urljoin(url, a["href"])
                links.append(
                    {
                        "text": clean_ws(a.get_text(" ", strip=True)),
                        "href": href,
                        "followed": False,
                    }
                )

            rows.append(
                {
                    "cells": cells,
                    "links": links,
                    "row_text": " | ".join(cells),
                }
            )

        all_links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            if "legislation.gov.uk" not in urlparse(href).netloc:
                continue
            all_links.append(
                {
                    "text": clean_ws(a.get_text(" ", strip=True)),
                    "href": href,
                    "followed": False,
                }
            )

        # Unique links.
        seen = set()
        unique_links = []
        for item in all_links:
            key = item["href"]
            if key not in seen:
                seen.add(key)
                unique_links.append(item)

        return {
            "url": url,
            "status": "PARSED",
            "rows": rows,
            "links": unique_links,
            "important_rule": (
                "This page was used only as metadata about the selected legislation. "
                "No affecting legislation link was followed."
            ),
        }

    def run(self) -> Path:
        out = self.output_dir / self.source["document_id"]
        raw_dir = out / "raw"
        processed_dir = out / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        akn_data, akn_url = self.get_akn()
        (raw_dir / "legislation.akn.xml").write_bytes(akn_data)

        parsed = self.parse_akn(akn_data)
        changes = self.scrape_changes_affecting()

        # JSONL outputs.
        self.write_jsonl(processed_dir / "nodes.jsonl", parsed["nodes"])
        self.write_jsonl(processed_dir / "references.jsonl", parsed["references"])
        self.write_jsonl(processed_dir / "annotations.jsonl", parsed["annotations"])

        if changes is not None:
            (processed_dir / "changes_affecting.json").write_text(
                json.dumps(changes, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        manifest = {
            "scraper_name": "group_a_legislation_scraper",
            "scraper_version": VERSION,
            "generated_at": now_iso(),
            "source_key": self.source_key,
            "document_id": self.source["document_id"],
            "title": self.source["title"],
            "root_url": self.source["root_url"],
            "contents_url": self.source["contents_url"],
            "akn_requested_from": akn_url,
            "scope_policy": {
                "full_ingestion_of_selected_document": True,
                "follow_internal_or_external_reference_links": False,
                "scrape_other_legislation": False,
                "store_reference_links": True,
                "capture_annotations": True,
                "capture_changes_affecting_metadata": self.include_changes_page,
            },
            "counts": {
                "nodes": len(parsed["nodes"]),
                "references": len(parsed["references"]),
                "annotations": len(parsed["annotations"]),
                "changes_rows": len(changes["rows"]) if changes and "rows" in changes else 0,
            },
            "fetch_log": self.fetch_log,
        }

        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return out

    @staticmethod
    def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Scrape exactly one Group A legislation item from legislation.gov.uk "
            "without crawling referenced legislation."
        )
    )
    ap.add_argument(
        "--source",
        required=True,
        choices=sorted(SOURCES),
        help="PA2023, PR2024, or PCR2015",
    )
    ap.add_argument(
        "--output-dir",
        default="data/group_a_legislation",
        help="Base output directory",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout in seconds",
    )
    ap.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="Polite delay between requests",
    )
    ap.add_argument(
        "--no-changes-page",
        action="store_true",
        help="Do not fetch the selected legislation's changes-affecting metadata page",
    )

    args = ap.parse_args()

    scraper = LegislationScraper(
        source_key=args.source,
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
        pause=args.pause,
        include_changes_page=not args.no_changes_page,
    )

    try:
        output = scraper.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Completed: {args.source}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
