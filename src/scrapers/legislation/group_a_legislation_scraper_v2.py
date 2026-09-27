#!/usr/bin/env python3
"""
Complete Group A scraper for legislation.gov.uk
===============================================

Supported instruments only:
  PA2023   Procurement Act 2023              UKPGA_2023_54
  PR2024   Procurement Regulations 2024      UKSI_2024_692
  PCR2015  Public Contracts Regulations 2015 UKSI_2015_102

Scope rule
----------
The scraper may retrieve multiple REPRESENTATIONS/VERSIONS of the selected
instrument (latest/revised + enacted/made) and metadata ABOUT that instrument,
but it NEVER follows a reference into another Act/SI.

Important correction from v1
----------------------------
Do NOT use:
    /contents/made/data.akn
as the full-document source.  That is a representation of the CONTENTS view,
not necessarily the full instrument body.

For SIs the original full-document candidates are:
    /made/data.akn
    /made/data.xml

For Acts:
    /enacted/data.akn
    /enacted/data.xml

The scraper also tries the root /data.akn and /data.xml as latest/revised
representations.  It validates candidates by actual provision coverage and
selects the strongest full-document representation rather than accepting the
first XML response that happens to parse.

Outputs
-------
<output>/<document_id>/
    manifest.json
    raw/
        latest_*.xml
        original_*.xml
        changes_affecting.csv
    processed/
        nodes.jsonl                  canonical searchable node set
        nodes_latest.jsonl
        nodes_original.jsonl
        references.jsonl             explicit body links/references
        external_reference_candidates.jsonl
        annotations.jsonl
        legal_effects.jsonl          changes-affecting metadata
        cross_references.jsonl       internal typed CROSS_REFERS_TO edges
        cross_reference_review.jsonl
        extraction_summary.json

Cross references
----------------
Uses BOTH:
  1. structured inline links where a source provision can be identified; and
  2. deterministic regex over the provision text.

PA2023 patterns include section/subsection/Part/Schedule/paragraph references.
PR2024 and PCR2015 patterns additionally include regulation/regulations/reg.
and local paragraph references.

The regex stage emits only INTERNAL references to the currently selected
instrument.  External legislation references are stored, never followed.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from lxml import etree

VERSION = "2.0.0"

SOURCES = {
    "PA2023": {
        "document_id": "UKPGA_2023_54",
        "title": "Procurement Act 2023",
        "type": "ukpga",
        "year": "2023",
        "number": "54",
        "root_url": "https://www.legislation.gov.uk/ukpga/2023/54",
        "contents_url": "https://www.legislation.gov.uk/ukpga/2023/54/contents",
        "original_segment": "enacted",
        "primary_kind": "section",
        "expected_min_primary": 127,
    },
    "PR2024": {
        "document_id": "UKSI_2024_692",
        "title": "Procurement Regulations 2024",
        "type": "uksi",
        "year": "2024",
        "number": "692",
        "root_url": "https://www.legislation.gov.uk/uksi/2024/692",
        "contents_url": "https://www.legislation.gov.uk/uksi/2024/692/contents/made",
        "original_segment": "made",
        "primary_kind": "regulation",
        "expected_min_primary": 50,
    },
    "PCR2015": {
        "document_id": "UKSI_2015_102",
        "title": "Public Contracts Regulations 2015",
        "type": "uksi",
        "year": "2015",
        "number": "102",
        "root_url": "https://www.legislation.gov.uk/uksi/2015/102",
        "contents_url": "https://www.legislation.gov.uk/uksi/2015/102/contents/made",
        "original_segment": "made",
        "primary_kind": "regulation",
        "expected_min_primary": 122,
    },
}

# AKN structural tags.
AKN_STRUCTURAL = {
    "part", "chapter", "title", "section", "subsection", "article", "rule",
    "regulation", "paragraph", "subparagraph", "schedule", "attachment",
    "division", "hcontainer", "blockList",
}

# Common CLML structure tags used by legislation.gov.uk data.xml.
CLML_STRUCTURAL = {
    "Part", "Chapter", "P1group", "P1", "P2", "P3", "P4", "P5", "P6",
    "Schedule", "ScheduleBody", "Schedules", "Appendix", "BlockAmendment",
}

REFERENCE_TAGS = {"ref", "rref", "Citation", "CitationSubRef"}
ANNOTATION_TAGS = {
    "note", "authorialNote", "editorialNote", "remark", "comment", "annotation",
    "Commentary", "CommentaryRef", "Footnote", "FootnoteRef",
}

EFFECT_CLASS_PATTERNS = [
    ("COMMENCED_BY", re.compile(r"\bcoming into force\b|\bcommencement\b", re.I)),
    ("INSERTED_BY", re.compile(r"\binserted\b|\bwords inserted\b", re.I)),
    ("SUBSTITUTED_BY", re.compile(r"\bsubstituted\b|\bwords substituted\b", re.I)),
    ("OMITTED_BY", re.compile(r"\bomitted\b|\bwords omitted\b", re.I)),
    ("REPEALED_BY", re.compile(r"\brepealed\b|\brevoked\b", re.I)),
    ("EXCLUDED_BY", re.compile(r"\bexcluded\b", re.I)),
    ("RESTRICTED_BY", re.compile(r"\brestricted\b", re.I)),
    ("MODIFIED_BY", re.compile(r"\bmodified\b|\bmodifications?\b|\bapplied\b|\bextended\b", re.I)),
    ("AMENDED_BY", re.compile(r"\bamended\b|\bamendment\b", re.I)),
]

INSTRUMENT_PATH_RE = re.compile(
    r"/(?P<typ>ukpga|uksi|asp|ssi|nia|nisr|anaw|mwa|wsi|ukla|ukcm|ukci)/"
    r"(?P<year>\d{4})/(?P<number>[^/]+)",
    re.I,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_ws(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def localname(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def element_text(el: etree._Element) -> str:
    return clean_ws(" ".join(el.itertext()))


def get_attr(el: etree._Element, names: Iterable[str]) -> str | None:
    names = set(names)
    for k, v in el.attrib.items():
        if k in names or localname(k) in names:
            return v
    return None


def first_descendant_text(el: etree._Element, names: set[str]) -> str:
    for d in el.iterdescendants():
        if localname(d.tag) in names:
            txt = element_text(d)
            if txt:
                return txt
    return ""


def strip_number(s: str) -> str:
    s = clean_ws(s)
    # Retain alphanumerics such as 5A, 40A.
    m = re.search(r"([0-9]+[A-Za-z]?|[IVXLCDM]+)$", s, re.I)
    if m:
        return m.group(1)
    return re.sub(r"[^0-9A-Za-z]", "", s)


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def is_same_instrument_url(url: str, source: dict[str, Any]) -> bool:
    m = INSTRUMENT_PATH_RE.search(urlparse(url).path)
    if not m:
        return False
    return (
        m.group("typ").lower() == source["type"].lower()
        and m.group("year") == source["year"]
        and m.group("number") == source["number"]
    )


def classify_effect(text: str) -> str:
    for label, pat in EFFECT_CLASS_PATTERNS:
        if pat.search(text or ""):
            return label
    return "LEGAL_EFFECT_OTHER"


@dataclass
class FetchRecord:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    retrieved_at: str
    sha256: str
    byte_count: int


class CompleteLegislationScraper:
    def __init__(
        self,
        source_key: str,
        output_dir: Path,
        timeout: int = 60,
        pause: float = 0.75,
    ):
        if source_key not in SOURCES:
            raise ValueError(f"Unsupported source {source_key}")
        self.source_key = source_key
        self.source = SOURCES[source_key]
        self.output_dir = output_dir
        self.timeout = timeout
        self.pause = pause
        self.fetch_log: list[dict[str, Any]] = []

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "MastersThesisProcurementResearchBot/2.0 "
                "(single-instrument academic research scraper)"
            ),
            "Accept": "*/*",
        })

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def fetch(self, url: str, accept: str | None = None) -> tuple[bytes, str]:
        headers = {"Accept": accept} if accept else {}
        r = self.session.get(
            url, headers=headers, timeout=self.timeout, allow_redirects=True
        )
        r.raise_for_status()
        data = r.content
        self.fetch_log.append(asdict(FetchRecord(
            requested_url=url,
            final_url=r.url,
            status_code=r.status_code,
            content_type=r.headers.get("content-type", ""),
            retrieved_at=now_iso(),
            sha256=sha256_bytes(data),
            byte_count=len(data),
        )))
        time.sleep(self.pause)
        return data, r.url

    # ------------------------------------------------------------------
    # Representation discovery and validation
    # ------------------------------------------------------------------

    def representation_candidates(self, version: str) -> list[tuple[str, str]]:
        root = self.source["root_url"].rstrip("/")
        original = self.source["original_segment"]

        if version == "latest":
            # Full instrument only.  Deliberately no /contents/... candidates.
            return [
                ("akn", f"{root}/data.akn"),
                ("clml", f"{root}/data.xml"),
            ]
        if version == "original":
            return [
                ("akn", f"{root}/{original}/data.akn"),
                ("clml", f"{root}/{original}/data.xml"),
            ]
        raise ValueError(version)

    def parse_xml(self, data: bytes) -> etree._Element:
        parser = etree.XMLParser(
            recover=True,
            huge_tree=True,
            resolve_entities=False,
            no_network=True,
            remove_comments=False,
        )
        return etree.fromstring(data, parser=parser)

    def detect_format(self, root: etree._Element) -> str:
        names = [localname(x.tag) for x in list(root.iter())[:200]]
        if any(n in {"akomaNtoso", "act", "doc"} for n in names) and any(
            n in AKN_STRUCTURAL for n in names
        ):
            return "akn"
        if any(n in {"Legislation", "Primary", "Secondary", "EURetained"} for n in names):
            return "clml"
        if any(n in CLML_STRUCTURAL for n in names):
            return "clml"
        # Last resort: inspect eId usage.
        if any(get_attr(x, {"eId"}) for x in list(root.iter())[:500]):
            return "akn"
        return "unknown"

    def score_representation(self, data: bytes, hinted_format: str) -> dict[str, Any]:
        try:
            root = self.parse_xml(data)
        except Exception as exc:
            return {"valid": False, "error": str(exc), "score": -1}

        fmt = self.detect_format(root)
        parsed = self.parse_document(data, forced_format=fmt if fmt != "unknown" else hinted_format)
        primary = sum(
            1 for n in parsed["nodes"]
            if n.get("element_type") == self.source["primary_kind"]
        )
        total = len(parsed["nodes"])
        text_chars = sum(len(n.get("text", "")) for n in parsed["nodes"])
        expected = self.source["expected_min_primary"]

        # Fullness validation: provision count is the main signal.
        coverage = primary / expected if expected else 0
        score = primary * 100000 + total * 100 + min(text_chars, 10_000_000)

        return {
            "valid": True,
            "format": fmt if fmt != "unknown" else hinted_format,
            "primary_count": primary,
            "total_nodes": total,
            "text_chars": text_chars,
            "expected_min_primary": expected,
            "primary_coverage_ratio": round(coverage, 4),
            "score": score,
            "parsed": parsed,
        }

    def retrieve_best_representation(self, version: str, raw_dir: Path) -> dict[str, Any] | None:
        attempts = []
        successful = []

        for hinted_format, url in self.representation_candidates(version):
            try:
                data, final_url = self.fetch(
                    url,
                    accept="application/xml,text/xml,application/akn+xml;q=0.9,*/*;q=0.1",
                )
                quality = self.score_representation(data, hinted_format)
                attempt = {
                    "url": url,
                    "final_url": final_url,
                    "hinted_format": hinted_format,
                    **{k: v for k, v in quality.items() if k != "parsed"},
                    "sha256": sha256_bytes(data),
                    "byte_count": len(data),
                }
                attempts.append(attempt)
                if quality.get("valid"):
                    suffix = "akn.xml" if quality["format"] == "akn" else "clml.xml"
                    raw_path = raw_dir / f"{version}_{suffix}"
                    raw_path.write_bytes(data)
                    successful.append({
                        "data": data,
                        "url": url,
                        "final_url": final_url,
                        "raw_path": str(raw_path),
                        **quality,
                    })
            except Exception as exc:
                attempts.append({
                    "url": url,
                    "hinted_format": hinted_format,
                    "valid": False,
                    "error": str(exc),
                })

        if not successful:
            return {"version": version, "selected": None, "attempts": attempts}

        selected = max(successful, key=lambda x: x["score"])
        return {
            "version": version,
            "selected": selected,
            "attempts": attempts,
        }

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    def parse_document(self, data: bytes, forced_format: str | None = None) -> dict[str, Any]:
        root = self.parse_xml(data)
        fmt = forced_format or self.detect_format(root)
        if fmt == "akn":
            return self.parse_akn(root)
        if fmt == "clml":
            return self.parse_clml(root)
        raise ValueError("Could not identify XML format")

    def parse_akn(self, root: etree._Element) -> dict[str, Any]:
        nodes = []
        references = []
        annotations = []

        # Only body-like elements are eligible as structural nodes.
        for el in root.iter():
            lname = localname(el.tag)
            eid = get_attr(el, {"eId", "id"})
            if lname not in AKN_STRUCTURAL or not eid:
                continue

            etype = lname
            if self.source["primary_kind"] == "regulation" and lname in {"article", "section"}:
                # UK SI AKN serialisations may use article-like structures.
                etype = "regulation"

            parent = self.nearest_akn_structural(el)
            parent_eid = get_attr(parent, {"eId", "id"}) if parent is not None else None
            num = self.direct_child_text(el, {"num"})
            heading = self.direct_child_text(el, {"heading", "subheading"})
            text = element_text(el)

            nodes.append(self.node_record(
                eid=eid, etype=etype, parent_eid=parent_eid,
                number=num, heading=heading, text=text, xml_tag=lname
            ))

        node_ids = {n["eId"] for n in nodes}

        # Inline/body references only. Metadata references with no structural
        # source are preserved separately but not promoted to graph edges.
        for el in root.iter():
            lname = localname(el.tag)
            if lname not in {"ref", "rref"}:
                continue
            href = get_attr(el, {"href", "refersTo"})
            if not href:
                continue
            source_ancestor = self.nearest_akn_structural(el)
            source_eid = get_attr(source_ancestor, {"eId", "id"}) if source_ancestor is not None else None
            display = element_text(el)
            resolved = self.resolve_href(href)

            references.append({
                "document_id": self.source["document_id"],
                "source_eId": source_eid if source_eid in node_ids else None,
                "display_text": display,
                "raw_href": href,
                "resolved_href": resolved,
                "same_instrument": is_same_instrument_url(resolved, self.source) if resolved.startswith("http") else resolved.startswith("#"),
                "followed": False,
                "source_kind": "INLINE_AKN" if source_eid else "AKN_METADATA_REFERENCE",
            })

        for el in root.iter():
            lname = localname(el.tag)
            if lname not in ANNOTATION_TAGS:
                continue
            text = element_text(el)
            if not text:
                continue
            anc = self.nearest_akn_structural(el)
            source_eid = get_attr(anc, {"eId", "id"}) if anc is not None else None
            annotations.append({
                "document_id": self.source["document_id"],
                "source_eId": source_eid,
                "annotation_type": self.classify_annotation(text),
                "text": text,
                "xml_element_type": lname,
                "attributes": {localname(k): v for k, v in el.attrib.items()},
            })

        return {
            "format": "akn",
            "nodes": self.dedupe(nodes, ("node_id",)),
            "references": self.dedupe(references, ("source_eId", "resolved_href", "display_text")),
            "annotations": self.dedupe(annotations, ("source_eId", "annotation_type", "text")),
        }

    def parse_clml(self, root: etree._Element) -> dict[str, Any]:
        nodes = []
        references = []
        annotations = []
        generated_counter = collections.Counter()

        # CLML does not always mirror AKN IDs. We construct deterministic IDs
        # from the hierarchy/number while retaining native IDs where present.
        element_to_eid: dict[int, str] = {}

        for el in root.iter():
            lname = localname(el.tag)
            if lname not in CLML_STRUCTURAL:
                continue

            native_id = get_attr(el, {"id", "Id", "DocumentURI"})
            num = first_descendant_text(el, {"Pnumber", "Number", "TitleNumber"})
            heading = first_descendant_text(el, {"Title", "Subtitle"})

            etype = self.map_clml_type(lname, el)
            if not etype:
                continue

            parent = self.nearest_clml_structural(el)
            parent_eid = element_to_eid.get(id(parent)) if parent is not None else None

            cleaned_num = strip_number(num)
            base = f"{etype}-{cleaned_num}" if cleaned_num else etype
            if parent_eid:
                base = f"{parent_eid}-{base}"
            generated_counter[base] += 1
            eid = native_id or (
                base if generated_counter[base] == 1
                else f"{base}-{generated_counter[base]}"
            )
            eid = safe_id(str(eid))
            element_to_eid[id(el)] = eid

            # For P1, use its own block only. Descendant P2/P3 text is included
            # in the provision text, matching the AKN node model.
            text = element_text(el)
            nodes.append(self.node_record(
                eid=eid, etype=etype, parent_eid=parent_eid,
                number=num, heading=heading, text=text, xml_tag=lname
            ))

        node_ids = {n["eId"] for n in nodes}

        for el in root.iter():
            lname = localname(el.tag)
            if lname not in REFERENCE_TAGS:
                continue
            href = (
                get_attr(el, {"URI", "Uri", "href", "HRef", "Ref"})
                or get_attr(el, {"idref"})
            )
            if not href:
                continue
            anc = self.nearest_clml_structural(el)
            source_eid = element_to_eid.get(id(anc)) if anc is not None else None
            resolved = self.resolve_href(href)
            references.append({
                "document_id": self.source["document_id"],
                "source_eId": source_eid if source_eid in node_ids else None,
                "display_text": element_text(el),
                "raw_href": href,
                "resolved_href": resolved,
                "same_instrument": is_same_instrument_url(resolved, self.source) if resolved.startswith("http") else resolved.startswith("#"),
                "followed": False,
                "source_kind": "INLINE_CLML" if source_eid else "CLML_METADATA_REFERENCE",
            })

        for el in root.iter():
            lname = localname(el.tag)
            if lname not in ANNOTATION_TAGS and not lname.lower().startswith("commentary"):
                continue
            text = element_text(el)
            if not text:
                continue
            anc = self.nearest_clml_structural(el)
            source_eid = element_to_eid.get(id(anc)) if anc is not None else None
            annotations.append({
                "document_id": self.source["document_id"],
                "source_eId": source_eid,
                "annotation_type": self.classify_annotation(text),
                "text": text,
                "xml_element_type": lname,
                "attributes": {localname(k): v for k, v in el.attrib.items()},
            })

        return {
            "format": "clml",
            "nodes": self.dedupe(nodes, ("node_id",)),
            "references": self.dedupe(references, ("source_eId", "resolved_href", "display_text")),
            "annotations": self.dedupe(annotations, ("source_eId", "annotation_type", "text")),
        }

    def map_clml_type(self, lname: str, el: etree._Element) -> str | None:
        if lname == "Part":
            return "part"
        if lname == "Chapter":
            return "chapter"
        if lname == "Schedule":
            return "schedule"
        if lname == "P1":
            return self.source["primary_kind"]
        if lname == "P2":
            return "subsection" if self.source["primary_kind"] == "section" else "paragraph"
        if lname == "P3":
            return "paragraph" if self.source["primary_kind"] == "section" else "subparagraph"
        if lname in {"P4", "P5", "P6"}:
            return "subparagraph"
        # P1group is often a heading/group, not a legal provision.
        return None

    def node_record(
        self, eid: str, etype: str, parent_eid: str | None,
        number: str, heading: str, text: str, xml_tag: str
    ) -> dict[str, Any]:
        return {
            "document_id": self.source["document_id"],
            "node_id": f"{self.source['document_id']}::{eid}",
            "eId": eid,
            "element_type": etype,
            "xml_tag": xml_tag,
            "parent_eId": parent_eid,
            "number": clean_ws(number),
            "heading": clean_ws(heading),
            "text": clean_ws(text),
            "text_sha256": sha256_text(clean_ws(text)),
        }

    def nearest_akn_structural(self, el: etree._Element) -> etree._Element | None:
        cur = el.getparent()
        while cur is not None:
            if localname(cur.tag) in AKN_STRUCTURAL and get_attr(cur, {"eId", "id"}):
                return cur
            cur = cur.getparent()
        return None

    def nearest_clml_structural(self, el: etree._Element) -> etree._Element | None:
        cur = el.getparent()
        while cur is not None:
            if localname(cur.tag) in CLML_STRUCTURAL:
                return cur
            cur = cur.getparent()
        return None

    def direct_child_text(self, el: etree._Element, names: set[str]) -> str:
        for child in el:
            if localname(child.tag) in names:
                return element_text(child)
        return ""

    def resolve_href(self, href: str) -> str:
        if href.startswith("#"):
            return href
        return urljoin(self.source["root_url"] + "/", href)

    def classify_annotation(self, text: str) -> str:
        if re.search(r"\bF\d+\b|textual amendment", text, re.I):
            return "TEXTUAL_AMENDMENT"
        if re.search(r"\bC\d+\b|modification|modified|applied|excluded|restricted", text, re.I):
            return "MODIFICATION"
        if re.search(r"\bI\d+\b|commencement|coming into force|in force", text, re.I):
            return "COMMENCEMENT"
        if re.search(r"\bE\d+\b|extent", text, re.I):
            return "EXTENT"
        return "OTHER_ANNOTATION"

    # ------------------------------------------------------------------
    # Changes affecting: ALL rows, not first 50 only
    # ------------------------------------------------------------------

    def fetch_changes_affecting(self, raw_dir: Path) -> list[dict[str, Any]]:
        t, y, n = self.source["type"], self.source["year"], self.source["number"]
        base = f"https://www.legislation.gov.uk/changes/affected/{t}/{y}/{n}"

        # Prefer CSV because it is the site's structured alternative format.
        csv_url = f"{base}/data.csv?results-count=1000&sort=affected-year-number"
        try:
            data, final = self.fetch(csv_url, accept="text/csv,text/plain,*/*;q=0.5")
            raw_dir.joinpath("changes_affecting.csv").write_bytes(data)
            text = data.decode("utf-8-sig", errors="replace")
            rows = list(csv.DictReader(io.StringIO(text)))
            parsed = self.normalize_change_rows(rows, source_url=final)
            if parsed:
                return parsed
        except Exception:
            pass

        # Fallback: paginate HTML until rows stop.
        all_rows = []
        seen_signatures = set()
        for page in range(1, 200):
            url = f"{base}?results-count=50&sort=affected-year-number&page={page}"
            try:
                data, final = self.fetch(url, accept="text/html,*/*;q=0.5")
            except Exception:
                break
            soup = BeautifulSoup(data, "html.parser")
            page_rows = []
            for tr in soup.find_all("tr"):
                cells = [clean_ws(x.get_text(" ", strip=True)) for x in tr.find_all(["th", "td"])]
                if len(cells) < 8 or cells[0].lower().startswith("sort by"):
                    continue
                sig = tuple(cells)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                links = [
                    {"text": clean_ws(a.get_text(" ", strip=True)),
                     "href": urljoin(final, a.get("href", "")),
                     "followed": False}
                    for a in tr.find_all("a", href=True)
                ]
                page_rows.append({"cells": cells, "links": links})
            if not page_rows:
                break
            all_rows.extend(page_rows)

        return self.normalize_change_html_rows(all_rows, source_url=base)

    def normalize_change_rows(
        self, rows: list[dict[str, str]], source_url: str
    ) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            # Column labels vary slightly. Keep raw row and infer fields.
            vals = [clean_ws(v) for v in row.values()]
            joined = " | ".join(vals)
            if not joined or "Changed Provision" in joined:
                continue
            effect = next((v for v in vals if any(p.search(v) for _, p in EFFECT_CLASS_PATTERNS)), "")
            out.append({
                "document_id": self.source["document_id"],
                "relation": classify_effect(effect or joined),
                "effect_text": effect,
                "raw_columns": row,
                "source_url": source_url,
                "followed_affecting_legislation": False,
            })
        return out

    def normalize_change_html_rows(
        self, rows: list[dict[str, Any]], source_url: str
    ) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            c = row["cells"]
            if len(c) < 8:
                continue
            # Known table layout:
            # 0 title, 1 affected citation, 2 affected provision, 3 effect,
            # 4 affecting title, 5 affecting citation, 6 affecting provision,
            # 7 applied, 8 note (optional)
            affected_provision = c[2] if len(c) > 2 else ""
            effect = c[3] if len(c) > 3 else ""
            affecting_title = c[4] if len(c) > 4 else ""
            affecting_citation = c[5] if len(c) > 5 else ""
            affecting_provision = c[6] if len(c) > 6 else ""

            links = row.get("links", [])
            affected_url = None
            affecting_url = None
            for link in links:
                href = link["href"]
                txt = link["text"]
                if txt == affected_provision or (
                    is_same_instrument_url(href, self.source) and affected_provision
                ):
                    affected_url = href
                elif txt == affecting_provision or (
                    href and not is_same_instrument_url(href, self.source)
                ):
                    affecting_url = href

            out.append({
                "document_id": self.source["document_id"],
                "affected_provision": affected_provision,
                "affected_url": affected_url,
                "relation": classify_effect(effect),
                "effect_text": effect,
                "affecting_title": affecting_title,
                "affecting_citation": affecting_citation,
                "affecting_provision": affecting_provision,
                "affecting_url": affecting_url,
                "applied_to_website_text": c[7] if len(c) > 7 else "",
                "note": c[8] if len(c) > 8 else "",
                "source_url": source_url,
                "followed_affecting_legislation": False,
            })
        return self.dedupe(
            out,
            ("affected_provision", "relation", "affecting_citation", "affecting_provision"),
        )

    # ------------------------------------------------------------------
    # Cross reference extraction
    # ------------------------------------------------------------------

    def build_cross_references(
        self,
        nodes: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        idx = CrossReferenceIndex(self.source, nodes)
        structured, structured_review = idx.from_structured_links(references)
        regex_edges, regex_review = idx.from_regex()

        # Structured link wins if same source/target locator exists.
        combined = structured + regex_edges
        combined.sort(key=lambda e: (
            e.get("source_node_id") or "",
            e.get("target_locator") or "",
            0 if e.get("extraction_method") == "STRUCTURED_LINK" else 1,
        ))
        final = self.dedupe(
            combined, ("source_node_id", "relation", "target_locator")
        )
        return final, structured_review + regex_review

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> Path:
        out = self.output_dir / self.source["document_id"]
        raw_dir = out / "raw"
        proc_dir = out / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        latest = self.retrieve_best_representation("latest", raw_dir)
        original = self.retrieve_best_representation("original", raw_dir)

        if not latest or not latest.get("selected"):
            latest_selected = None
        else:
            latest_selected = latest["selected"]

        if not original or not original.get("selected"):
            original_selected = None
        else:
            original_selected = original["selected"]

        if not latest_selected and not original_selected:
            raise RuntimeError("No full-document representation could be parsed.")

        # Canonical searchable set: prefer latest/revised when it has meaningful
        # provision coverage; otherwise fall back to original/made.
        canonical = latest_selected or original_selected
        expected = self.source["expected_min_primary"]
        if latest_selected and latest_selected["primary_count"] < max(1, int(expected * 0.60)):
            if original_selected and original_selected["primary_count"] > latest_selected["primary_count"]:
                canonical = original_selected

        nodes = canonical["parsed"]["nodes"]
        references = canonical["parsed"]["references"]
        annotations = canonical["parsed"]["annotations"]

        latest_nodes = latest_selected["parsed"]["nodes"] if latest_selected else []
        original_nodes = original_selected["parsed"]["nodes"] if original_selected else []

        self.write_jsonl(proc_dir / "nodes.jsonl", nodes)
        self.write_jsonl(proc_dir / "nodes_latest.jsonl", latest_nodes)
        self.write_jsonl(proc_dir / "nodes_original.jsonl", original_nodes)
        self.write_jsonl(proc_dir / "references.jsonl", references)

        external_refs = [
            r for r in references
            if r.get("resolved_href", "").startswith("http")
            and not is_same_instrument_url(r["resolved_href"], self.source)
        ]
        self.write_jsonl(proc_dir / "external_reference_candidates.jsonl", external_refs)
        self.write_jsonl(proc_dir / "annotations.jsonl", annotations)

        effects = self.fetch_changes_affecting(raw_dir)
        self.write_jsonl(proc_dir / "legal_effects.jsonl", effects)

        crossrefs, review = self.build_cross_references(nodes, references)
        self.write_jsonl(proc_dir / "cross_references.jsonl", crossrefs)
        self.write_jsonl(proc_dir / "cross_reference_review.jsonl", review)

        primary_count = sum(
            1 for n in nodes
            if n.get("element_type") == self.source["primary_kind"]
        )

        summary = {
            "scraper_version": VERSION,
            "source_key": self.source_key,
            "document_id": self.source["document_id"],
            "title": self.source["title"],
            "canonical_representation_url": canonical["final_url"],
            "canonical_representation_format": canonical["format"],
            "canonical_primary_provision_count": primary_count,
            "expected_min_primary_provisions": expected,
            "coverage_ratio_vs_expected_min": round(primary_count / expected, 4) if expected else None,
            "node_count": len(nodes),
            "reference_count": len(references),
            "external_reference_candidate_count": len(external_refs),
            "annotation_count": len(annotations),
            "legal_effect_count": len(effects),
            "cross_reference_count": len(crossrefs),
            "cross_reference_review_count": len(review),
            "latest_representation": self.rep_summary(latest),
            "original_representation": self.rep_summary(original),
        }
        proc_dir.joinpath("extraction_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        manifest = {
            **summary,
            "generated_at": now_iso(),
            "root_url": self.source["root_url"],
            "contents_url_for_human_navigation_only": self.source["contents_url"],
            "scope_policy": {
                "selected_instrument_only": True,
                "full_document_representations_only": True,
                "contents_data_endpoint_used_as_body": False,
                "external_reference_links_followed": False,
                "latest_and_original_versions_may_be_stored": True,
                "changes_affecting_metadata_captured": True,
                "regex_cross_references_enabled": True,
            },
            "fetch_log": self.fetch_log,
        }
        out.joinpath("manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(json.dumps(summary, indent=2))
        return out

    def rep_summary(self, rep: dict[str, Any] | None) -> dict[str, Any] | None:
        if not rep:
            return None
        sel = rep.get("selected")
        return {
            "selected": None if not sel else {
                "url": sel["url"],
                "final_url": sel["final_url"],
                "format": sel["format"],
                "primary_count": sel["primary_count"],
                "total_nodes": sel["total_nodes"],
                "text_chars": sel["text_chars"],
                "coverage_ratio": sel["primary_coverage_ratio"],
                "raw_path": sel["raw_path"],
            },
            "attempts": rep.get("attempts", []),
        }

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def dedupe(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        seen = set()
        out = []
        for row in rows:
            key = tuple(row.get(f) for f in fields)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out


# ======================================================================
# Cross-reference index
# ======================================================================

class CrossReferenceIndex:
    def __init__(self, source: dict[str, Any], nodes: list[dict[str, Any]]):
        self.source = source
        self.doc_id = source["document_id"]
        self.nodes = nodes
        self.by_eid = {n["eId"]: n for n in nodes if n.get("eId")}
        self.children = collections.defaultdict(list)
        for n in nodes:
            if n.get("parent_eId"):
                self.children[n["parent_eId"]].append(n["eId"])

        self.primary = {}
        self.parts = {}
        self.schedules = {}
        for n in nodes:
            typ = n.get("element_type")
            num = strip_number(n.get("number", ""))
            if not num:
                continue
            if typ == source["primary_kind"]:
                self.primary[num.lower()] = n
            elif typ == "part":
                self.parts[num.lower()] = n
            elif typ == "schedule":
                self.schedules[num.lower()] = n

    def most_specific_nodes(self) -> list[dict[str, Any]]:
        eligible = {"section", "regulation", "subsection", "paragraph", "subparagraph"}
        out = []
        for n in self.nodes:
            if n.get("element_type") not in eligible:
                continue
            children = [
                self.by_eid[c] for c in self.children.get(n.get("eId"), [])
                if c in self.by_eid and self.by_eid[c].get("element_type") in eligible
            ]
            if not children and clean_ws(n.get("text")):
                out.append(n)
        return out

    def containing_primary_number(self, node: dict[str, Any]) -> str | None:
        eid = node.get("eId")
        seen = set()
        while eid and eid not in seen:
            seen.add(eid)
            n = self.by_eid.get(eid)
            if not n:
                break
            if n.get("element_type") == self.source["primary_kind"]:
                return strip_number(n.get("number", ""))
            eid = n.get("parent_eId")
        return None

    def resolve_primary(self, number: str, chain: list[str]) -> dict[str, Any]:
        num = strip_number(number)
        base = self.primary.get(num.lower())
        locator_label = "section" if self.source["primary_kind"] == "section" else "regulation"
        locator = f"{locator_label} {num}" + "".join(f"({x})" for x in chain)

        target = base
        resolution = "EXACT_PRIMARY_NODE" if base else "UNRESOLVED"

        # Try common deterministic EID forms first.
        if base and chain:
            candidates = []
            base_eid = base["eId"]
            candidates.append(base_eid + "".join("-" + x for x in chain))
            candidates.append(base_eid + "".join("__" + x for x in chain))
            for cand in candidates:
                if cand in self.by_eid:
                    target = self.by_eid[cand]
                    resolution = "EXACT_NODE"
                    break
            else:
                resolution = "NEAREST_ANCESTOR_NODE"

        path_kind = "section" if self.source["primary_kind"] == "section" else "regulation"
        url = self.source["root_url"] + f"/{path_kind}/{num}" + "".join("/" + x for x in chain)
        return self.target(locator, target, resolution, url)

    def resolve_part(self, number: str) -> dict[str, Any]:
        num = strip_number(number)
        node = self.parts.get(num.lower())
        return self.target(
            f"Part {num}", node,
            "EXACT_NODE" if node else "LOCATOR_ONLY",
            self.source["root_url"] + f"/part/{num}",
        )

    def resolve_schedule(self, number: str) -> dict[str, Any]:
        num = strip_number(number)
        node = self.schedules.get(num.lower())
        return self.target(
            f"Schedule {num}", node,
            "EXACT_NODE" if node else "LOCATOR_ONLY",
            self.source["root_url"] + f"/schedule/{num}",
        )

    def target(
        self, locator: str, node: dict[str, Any] | None,
        resolution: str, url: str
    ) -> dict[str, Any]:
        return {
            "target_document_id": self.doc_id,
            "target_locator": locator,
            "target_node_id": node.get("node_id") if node else None,
            "target_eId": node.get("eId") if node else None,
            "target_resolution": resolution,
            "target_url": url,
        }

    def edge(
        self, source: dict[str, Any], target: dict[str, Any],
        matched: str, evidence: str, method: str, pattern: str,
        confidence: float
    ) -> dict[str, Any]:
        return {
            "document_id": self.doc_id,
            "source_node_id": source.get("node_id"),
            "source_eId": source.get("eId"),
            "source_element_type": source.get("element_type"),
            "relation": "CROSS_REFERS_TO",
            **target,
            "matched_text": clean_ws(matched),
            "evidence_text": clean_ws(evidence),
            "extraction_method": method,
            "pattern_type": pattern,
            "confidence": confidence,
        }

    def evidence_window(self, text: str, start: int, end: int, radius: int = 150) -> str:
        return clean_ws(text[max(0, start-radius): min(len(text), end+radius)])

    def parse_chain(self, chain: str) -> list[str]:
        return re.findall(r"\(([0-9A-Za-z]+)\)", chain or "")

    def from_regex(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        edges = []
        review = []

        if self.source["primary_kind"] == "section":
            explicit = re.compile(
                r"\b(?:section|s\.)\s*(\d+[A-Za-z]?)(?P<chain>(?:\([0-9A-Za-z]+\))*)",
                re.I,
            )
            list_re = re.compile(
                r"\bsections?\s+(?P<body>\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*"
                r"(?:\s*(?:,|and|or)\s*\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*)+)",
                re.I,
            )
            local_re = re.compile(
                r"\bsubsections?\s+(?P<body>\([0-9A-Za-z]+\)(?:\([0-9A-Za-z]+\))*"
                r"(?:\s*(?:,|and|or)\s*\([0-9A-Za-z]+\)(?:\([0-9A-Za-z]+\))*)*)",
                re.I,
            )
            local_pattern_name = "LOCAL_SUBSECTION"
        else:
            explicit = re.compile(
                r"\b(?:regulation|reg\.)\s*(\d+[A-Za-z]?)(?P<chain>(?:\([0-9A-Za-z]+\))*)",
                re.I,
            )
            list_re = re.compile(
                r"\bregulations?\s+(?P<body>\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*"
                r"(?:\s*(?:,|and|or)\s*\d+[A-Za-z]?(?:\([0-9A-Za-z]+\))*)+)",
                re.I,
            )
            # Within an SI, "paragraph (2)" generally means paragraph (2)
            # of the current regulation.
            local_re = re.compile(
                r"\bparagraphs?\s+(?P<body>\([0-9A-Za-z]+\)(?:\([0-9A-Za-z]+\))*"
                r"(?:\s*(?:,|and|or)\s*\([0-9A-Za-z]+\)(?:\([0-9A-Za-z]+\))*)*)",
                re.I,
            )
            local_pattern_name = "LOCAL_PARAGRAPH"

        schedule_para = re.compile(
            r"\bparagraphs?\s+(?P<p>\d+[A-Za-z]?)(?P<chain>(?:\([0-9A-Za-z]+\))*)"
            r"\s+of\s+Schedule\s+(?P<s>\d+[A-Za-z]?)",
            re.I,
        )
        schedule_re = re.compile(r"\bSchedule\s+(\d+[A-Za-z]?)\b", re.I)
        part_re = re.compile(r"\bPart\s+(\d+[A-Za-z]?)\b", re.I)

        for source in self.most_specific_nodes():
            text = clean_ws(source.get("text"))
            if not text:
                continue
            occupied = []

            def overlaps(span):
                a, b = span
                return any(not (b <= x or a >= y) for x, y in occupied)

            for m in list_re.finditer(text):
                occupied.append(m.span())
                for item in re.split(r"\s*(?:,|and|or)\s*", m.group("body")):
                    mm = re.fullmatch(r"(\d+[A-Za-z]?)(.*)", item.strip())
                    if not mm:
                        continue
                    target = self.resolve_primary(mm.group(1), self.parse_chain(mm.group(2)))
                    edges.append(self.edge(
                        source, target, m.group(0),
                        self.evidence_window(text, *m.span()),
                        "REGEX", self.source["primary_kind"].upper(), 0.99
                    ))

            for m in explicit.finditer(text):
                if overlaps(m.span()):
                    continue
                occupied.append(m.span())
                target = self.resolve_primary(m.group(1), self.parse_chain(m.group("chain")))
                edges.append(self.edge(
                    source, target, m.group(0),
                    self.evidence_window(text, *m.span()),
                    "REGEX", self.source["primary_kind"].upper(), 0.99
                ))

            current_num = self.containing_primary_number(source)
            if current_num:
                for m in local_re.finditer(text):
                    if overlaps(m.span()):
                        continue
                    occupied.append(m.span())
                    items = re.split(r"\s*(?:,|and|or)\s*", m.group("body"))
                    for item in items:
                        chain = self.parse_chain(item)
                        if not chain:
                            continue
                        target = self.resolve_primary(current_num, chain)
                        edges.append(self.edge(
                            source, target, m.group(0),
                            self.evidence_window(text, *m.span()),
                            "REGEX", local_pattern_name, 0.97
                        ))

            for m in schedule_para.finditer(text):
                if overlaps(m.span()):
                    continue
                occupied.append(m.span())
                # Preserve the exact schedule paragraph locator even if a
                # dedicated node is unavailable.
                snum = strip_number(m.group("s"))
                pnum = strip_number(m.group("p"))
                chain = self.parse_chain(m.group("chain"))
                sched = self.resolve_schedule(snum)
                sched.update({
                    "target_locator": f"Schedule {snum} paragraph {pnum}" + "".join(f"({x})" for x in chain),
                    "target_resolution": "NEAREST_SCHEDULE_NODE" if sched["target_node_id"] else "LOCATOR_ONLY",
                    "target_url": self.source["root_url"] + f"/schedule/{snum}/paragraph/{pnum}" + "".join("/"+x for x in chain),
                })
                edges.append(self.edge(
                    source, sched, m.group(0),
                    self.evidence_window(text, *m.span()),
                    "REGEX", "SCHEDULE_PARAGRAPH", 0.99
                ))

            for m in schedule_re.finditer(text):
                if overlaps(m.span()):
                    continue
                target = self.resolve_schedule(m.group(1))
                edges.append(self.edge(
                    source, target, m.group(0),
                    self.evidence_window(text, *m.span()),
                    "REGEX", "SCHEDULE", 0.97
                ))

            for m in part_re.finditer(text):
                if overlaps(m.span()):
                    continue
                target = self.resolve_part(m.group(1))
                edges.append(self.edge(
                    source, target, m.group(0),
                    self.evidence_window(text, *m.span()),
                    "REGEX", "PART", 0.97
                ))

        # Remove direct self-link and dedupe.
        deduped = []
        seen = set()
        for e in edges:
            if e.get("source_node_id") and e.get("source_node_id") == e.get("target_node_id"):
                continue
            key = (
                e.get("source_node_id"), e.get("relation"),
                e.get("target_locator"), e.get("matched_text"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
            if e["target_resolution"] in {"UNRESOLVED", "LOCATOR_ONLY"}:
                review.append({**e, "review_reason": "TARGET_NOT_EXACTLY_RESOLVED"})
        return deduped, review

    def from_structured_links(
        self, refs: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        edges = []
        review = []

        kind = "section" if self.source["primary_kind"] == "section" else "regulation"
        path_re = re.compile(
            rf"/{kind}/(?P<num>[0-9A-Za-z]+)(?P<rest>(?:/[0-9A-Za-z]+)*)/?$",
            re.I,
        )

        for ref in refs:
            if not ref.get("same_instrument"):
                continue
            href = ref.get("resolved_href") or ""
            source_eid = ref.get("source_eId")
            source = self.by_eid.get(source_eid) if source_eid else None

            if href.startswith("#"):
                target_eid = href[1:]
                target_node = self.by_eid.get(target_eid)
                if not target_node:
                    continue
                target = self.target(
                    target_node.get("number") or target_eid,
                    target_node, "EXACT_NODE", href
                )
            else:
                m = path_re.search(urlparse(href).path)
                if not m:
                    continue
                chain = [x for x in m.group("rest").split("/") if x]
                target = self.resolve_primary(m.group("num"), chain)

            if not source:
                review.append({
                    "document_id": self.doc_id,
                    "source_eId": source_eid,
                    "relation": "CROSS_REFERS_TO",
                    **target,
                    "matched_text": ref.get("display_text"),
                    "extraction_method": "STRUCTURED_LINK",
                    "confidence": 1.0,
                    "review_reason": "NO_SOURCE_PROVISION_FOR_STRUCTURED_REFERENCE",
                })
                continue

            edges.append(self.edge(
                source, target,
                ref.get("display_text") or href,
                ref.get("display_text") or href,
                "STRUCTURED_LINK", "INLINE_REFERENCE", 1.0
            ))
        return edges, review


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--output-dir", default="data/group_a_legislation_v2")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pause", type=float, default=0.75)
    args = ap.parse_args()

    scraper = CompleteLegislationScraper(
        source_key=args.source,
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
        pause=args.pause,
    )

    try:
        out = scraper.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Completed {args.source}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
