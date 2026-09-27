#!/usr/bin/env python3
"""
group_a_legislation_scraper_v4.py

Robust full-instrument scraper for:
- Procurement Act 2023 (PA2023)
- Procurement Regulations 2024 (PR2024)
- Public Contracts Regulations 2015 (PCR2015)

V4 design
=========
This version is intentionally based on the successful V2 design principles:

1. Fetch multiple official legislation.gov.uk representations.
2. Detect Akoma Ntoso (AKN) vs Crown Legislation Markup Language (CLML).
3. Parse each representation with a representation-specific parser.
4. Score candidates using *parsed provision coverage*, not raw tag counts.
5. Select the strongest valid candidate.
6. Extract:
   - full hierarchy
   - exact text
   - structured references
   - external references
   - annotations / commentary where available
   - legal-effect candidates
   - regex cross-reference candidates
7. Preserve both current/latest and original/enacted/made representations.
8. Fail validation loudly if provision counts are implausible.
9. Produce full_text files for downstream inspection.
10. Do NOT LLM-chunk and do NOT infer semantic graph relations.

Important
=========
Never use /contents/.../data.akn or /contents/.../data.xml as the body.

The /contents endpoints are useful for navigation but may not contain the full
instrument text.

Dependencies:
    requests
    lxml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from lxml import etree


VERSION = "4.0.0"


SOURCES = {
    "PA2023": {
        "document_id": "UKPGA_2023_54",
        "title": "Procurement Act 2023",
        "type": "ukpga",
        "year": 2023,
        "number": 54,
        "root_url": "https://www.legislation.gov.uk/ukpga/2023/54",
        "contents_url": "https://www.legislation.gov.uk/ukpga/2023/54/contents",
        "original_variant": "enacted",
        "expected_top_type": "section",
        "expected_min_top": 127,
        "expected_min_nodes": 1000,
        "expect_structured_refs": True,
        "corpus_role": "CORE_PRIMARY_LEGISLATION",
    },
    "PR2024": {
        "document_id": "UKSI_2024_692",
        "title": "Procurement Regulations 2024",
        "type": "uksi",
        "year": 2024,
        "number": 692,
        "root_url": "https://www.legislation.gov.uk/uksi/2024/692",
        "contents_url": "https://www.legislation.gov.uk/uksi/2024/692/contents/made",
        "original_variant": "made",
        "expected_top_type": "regulation",
        "expected_min_top": 50,
        "expected_min_nodes": 500,
        "expect_structured_refs": True,
        "corpus_role": "CORE_SECONDARY_LEGISLATION",
    },
    "PCR2015": {
        "document_id": "UKSI_2015_102",
        "title": "Public Contracts Regulations 2015",
        "type": "uksi",
        "year": 2015,
        "number": 102,
        "root_url": "https://www.legislation.gov.uk/uksi/2015/102",
        "contents_url": "https://www.legislation.gov.uk/uksi/2015/102/contents/made",
        "original_variant": "made",
        "expected_top_type": "regulation",
        "expected_min_top": 120,
        "expected_min_nodes": 1000,
        # V2 may legitimately have sparse/zero structured refs for some PCR
        # representations, so do not make this a hard failure.
        "expect_structured_refs": False,
        "corpus_role": "LEGACY_PROCUREMENT_LEGISLATION",
    },
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def lname(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def attr_local(el: etree._Element, names: set[str]) -> Optional[str]:
    wanted = {x.lower() for x in names}
    for k, v in el.attrib.items():
        if lname(k).lower() in wanted:
            return v
    return None


def text_of(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return clean_ws(" ".join(el.itertext()))


def first_desc_text(el: etree._Element, names: set[str]) -> Optional[str]:
    wanted = {x.lower() for x in names}
    for x in el.iter():
        if lname(x.tag).lower() in wanted:
            t = text_of(x)
            if t:
                return t
    return None


def safe(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "")
    return s.strip("_")[:180] or "x"


# ---------------------------------------------------------------------------
# Candidate fetching
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    label: str
    variant: str
    url: str
    local_file: str
    status_code: Optional[int] = None
    final_url: Optional[str] = None
    content_type: Optional[str] = None
    byte_count: int = 0
    sha256: Optional[str] = None
    xml_ok: bool = False
    representation: Optional[str] = None
    parsed_ok: bool = False
    node_count: int = 0
    top_count: int = 0
    reference_count: int = 0
    text_chars: int = 0
    score: int = 0
    error: Optional[str] = None


def candidate_specs(src: dict[str, Any]) -> list[tuple[str, str, str]]:
    root = src["root_url"].rstrip("/")
    orig = src["original_variant"]

    # Current/latest representation
    out = [
        ("latest_akn", "latest", f"{root}/data.akn"),
        ("latest_clml", "latest", f"{root}/data.xml"),
        # Original representation
        ("original_akn", "original", f"{root}/{orig}/data.akn"),
        ("original_clml", "original", f"{root}/{orig}/data.xml"),
    ]

    # Exact de-duplication while preserving order
    seen = set()
    deduped = []
    for item in out:
        if item[2] not in seen:
            seen.add(item[2])
            deduped.append(item)
    return deduped


# ---------------------------------------------------------------------------
# Representation detection
# ---------------------------------------------------------------------------

def detect_representation(root: etree._Element) -> str:
    root_name = lname(root.tag).lower()
    ns_values = " ".join(
        x for x in [root.nsmap.get(None, ""), *[v for v in root.nsmap.values() if v]]
        if x
    ).lower()

    # AKN usually has akomaNtoso / act / doc with akn namespace.
    if "akomantoso" in root_name or "akomantoso" in ns_values or "/akn/" in ns_values:
        return "AKN"

    # UK CLML typically includes Legislation namespace and elements like P1,
    # P1group, Body, Primary, Secondary, Schedule.
    sample_names = {lname(x.tag).lower() for x in list(root.iter())[:5000] if isinstance(x.tag, str)}
    if (
        any(x in sample_names for x in {"p1", "p1group", "p2", "p3"})
        or "legislation" in ns_values
    ):
        return "CLML"

    # Last-resort structure checks.
    if any(x in sample_names for x in {"section", "article", "subsection", "paragraph"}):
        return "AKN"

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Common extraction model
# ---------------------------------------------------------------------------

class ParseResult:
    def __init__(self):
        self.nodes: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.legal_effects: list[dict[str, Any]] = []
        self.external_reference_candidates: list[dict[str, Any]] = []


class BaseParser:
    def __init__(self, source_key: str, src: dict[str, Any], variant: str):
        self.source_key = source_key
        self.src = src
        self.variant = variant
        self.result = ParseResult()
        self.ordinal = 0

    def make_node_id(
        self,
        node_type: str,
        number: Optional[str],
        eid: Optional[str],
        path: list[int],
    ) -> str:
        base = self.src["document_id"]

        if eid:
            return f"{base}__{safe(eid)}"

        suffix = safe(number) if number else "_".join(map(str, path))
        return f"{base}__{node_type}__{suffix}"

    def add_node(
        self,
        *,
        node_type: str,
        number: Optional[str],
        heading: Optional[str],
        text: str,
        eid: Optional[str],
        parent_id: Optional[str],
        depth: int,
        path: list[int],
        source_element: str,
    ) -> str:
        self.ordinal += 1
        node_id = self.make_node_id(node_type, number, eid, path)

        self.result.nodes.append({
            "node_id": node_id,
            "document_id": self.src["document_id"],
            "source_key": self.source_key,
            "variant": self.variant,
            "node_type": node_type,
            "number": number,
            "heading": heading,
            "eid": eid,
            "parent_node_id": parent_id,
            "depth": depth,
            "ordinal": self.ordinal,
            "path_index": path,
            "source_element": source_element,
            "text": text,
            "text_sha256": sha256_text(text),
            "source_url": self.src["root_url"],
            "corpus_role": self.src["corpus_role"],
            "chunking_status": "NOT_CHUNKED",
        })

        return node_id

    def add_reference(
        self,
        *,
        source_node_id: str,
        source_eid: Optional[str],
        anchor_text: str,
        href: str,
        element_name: str,
        method: str,
    ) -> None:
        abs_url = urljoin(self.src["root_url"], href)
        defrag, frag = urldefrag(abs_url)

        ref_id = (
            f"{source_node_id}__REF_"
            f"{len([r for r in self.result.references if r['source_node_id']==source_node_id])+1:04d}"
        )

        row = {
            "reference_id": ref_id,
            "source_document_id": self.src["document_id"],
            "source_node_id": source_node_id,
            "source_eid": source_eid,
            "anchor_text": anchor_text,
            "raw_href": href,
            "absolute_url": abs_url,
            "defragmented_url": defrag,
            "fragment": frag or None,
            "element_name": element_name,
            "extraction_method": method,
            "resolution_status": "UNRESOLVED",
            "target_document_id": None,
            "target_node_id": None,
            "edge_status": "NOT_CREATED",
        }
        self.result.references.append(row)

        # Keep obvious external-legislation refs as a separate resolution queue.
        if "legislation.gov.uk" in urlparse(abs_url).netloc and self.src["root_url"] not in abs_url:
            self.result.external_reference_candidates.append({
                **row,
                "candidate_type": "EXTERNAL_LEGISLATION_REFERENCE",
            })


# ---------------------------------------------------------------------------
# AKN parser
# ---------------------------------------------------------------------------

class AKNParser(BaseParser):
    STRUCTURAL = {
        "part", "chapter", "section", "article", "subsection",
        "paragraph", "subparagraph", "point", "subpoint",
        "schedule", "hcontainer", "rule",
    }

    WRAPPERS = {
        "akomaNtoso", "act", "doc", "body", "mainBody", "preface",
        "conclusions", "attachments", "attachment", "component",
        "components", "portion", "level",
    }

    def normalized_type(self, el: etree._Element) -> Optional[str]:
        n = lname(el.tag).lower()

        if n not in self.STRUCTURAL:
            return None

        # UK SIs can encode regulations as article or section-like nodes in AKN.
        if self.src["type"] == "uksi" and n in {"article", "section"}:
            return "regulation"

        return n

    def number(self, el: etree._Element) -> Optional[str]:
        for child in el:
            if lname(child.tag).lower() == "num":
                t = text_of(child)
                return t or None
        return None

    def heading(self, el: etree._Element) -> Optional[str]:
        for child in el:
            if lname(child.tag).lower() in {"heading", "subheading", "title"}:
                t = text_of(child)
                return t or None
        return None

    def eid(self, el: etree._Element) -> Optional[str]:
        return attr_local(el, {"eid", "id"})

    def nearest_structural_parent(self, el: etree._Element) -> Optional[etree._Element]:
        p = el.getparent()
        while p is not None:
            if self.normalized_type(p):
                return p
            p = p.getparent()
        return None

    def structural_roots(self, root: etree._Element) -> list[etree._Element]:
        roots = []
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if self.normalized_type(el) and self.nearest_structural_parent(el) is None:
                roots.append(el)
        return roots

    def immediate_structural_children(self, el: etree._Element) -> list[etree._Element]:
        out = []
        for cand in el.iterdescendants():
            if not self.normalized_type(cand):
                continue
            if self.nearest_structural_parent(cand) is el:
                out.append(cand)
        return out

    def extract_refs_for_node(self, el: etree._Element, node_id: str, eid: Optional[str]) -> None:
        for x in el.iter():
            if not isinstance(x.tag, str):
                continue
            ename = lname(x.tag)
            low = ename.lower()

            href = attr_local(x, {"href", "ref", "refersTo", "uri"})
            if not href:
                continue

            # AKN structured refs are especially common in ref/rref elements,
            # but accept href-bearing elements generally.
            if low in {"ref", "rref", "link", "a"} or href:
                self.add_reference(
                    source_node_id=node_id,
                    source_eid=eid,
                    anchor_text=text_of(x),
                    href=href,
                    element_name=ename,
                    method="AKN_STRUCTURED_REFERENCE",
                )

    def extract_annotations(self, root: etree._Element) -> None:
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            n = lname(el.tag).lower()
            if n in {"note", "authorialnote", "editorialnote", "remark"}:
                txt = text_of(el)
                if txt:
                    self.result.annotations.append({
                        "document_id": self.src["document_id"],
                        "variant": self.variant,
                        "element_name": lname(el.tag),
                        "eid": attr_local(el, {"eid", "id"}),
                        "text": txt,
                        "text_sha256": sha256_text(txt),
                    })

    def parse(self, root: etree._Element) -> ParseResult:
        def walk(el: etree._Element, parent_id: Optional[str], depth: int, path: list[int]) -> None:
            nt = self.normalized_type(el)
            if not nt:
                return

            number = self.number(el)
            heading = self.heading(el)
            eid = self.eid(el)
            txt = text_of(el)

            node_id = self.add_node(
                node_type=nt,
                number=number,
                heading=heading,
                text=txt,
                eid=eid,
                parent_id=parent_id,
                depth=depth,
                path=path,
                source_element=lname(el.tag),
            )

            self.extract_refs_for_node(el, node_id, eid)

            for i, child in enumerate(self.immediate_structural_children(el), start=1):
                walk(child, node_id, depth + 1, path + [i])

        for i, root_el in enumerate(self.structural_roots(root), start=1):
            walk(root_el, None, 0, [i])

        self.extract_annotations(root)
        return self.result


# ---------------------------------------------------------------------------
# CLML parser
# ---------------------------------------------------------------------------

class CLMLParser(BaseParser):
    """
    Crown Legislation Markup Language parser.

    Important hierarchy:
      P1      -> section / regulation
      P2      -> subsection / paragraph
      P3      -> paragraph / subparagraph
      P4/P5/P6 -> deeper points
      Part / Chapter / Schedule are structural containers.
      P1group is a grouping wrapper, NOT itself a numbered provision.
    """

    STRUCTURAL_NAMES = {
        "part", "chapter", "schedule", "p1", "p2", "p3", "p4", "p5", "p6"
    }

    def normalized_type(self, el: etree._Element) -> Optional[str]:
        n = lname(el.tag).lower()

        if n == "part":
            return "part"
        if n == "chapter":
            return "chapter"
        if n == "schedule":
            return "schedule"

        if n == "p1":
            return "section" if self.src["type"] == "ukpga" else "regulation"

        if n == "p2":
            return "subsection" if self.src["type"] == "ukpga" else "paragraph"

        if n == "p3":
            return "paragraph"
        if n == "p4":
            return "subparagraph"
        if n == "p5":
            return "point"
        if n == "p6":
            return "subpoint"

        return None

    def number(self, el: etree._Element) -> Optional[str]:
        # CLML often places Pnumber as an immediate child, sometimes nested in
        # P1para/P2para. Search shallow-first.
        for child in el:
            if lname(child.tag).lower() in {"pnumber", "number"}:
                t = text_of(child)
                if t:
                    return t

        for x in el.iterdescendants():
            if lname(x.tag).lower() in {"pnumber", "number"}:
                # Avoid stealing a descendant provision's number.
                p = x.getparent()
                nearest_structural = None
                while p is not None and p is not el:
                    if self.normalized_type(p):
                        nearest_structural = p
                        break
                    p = p.getparent()
                if nearest_structural is None:
                    t = text_of(x)
                    if t:
                        return t
        return None

    def heading(self, el: etree._Element) -> Optional[str]:
        for child in el:
            if lname(child.tag).lower() in {
                "title", "heading", "number", "pnumber"
            }:
                if lname(child.tag).lower() in {"number", "pnumber"}:
                    continue
                t = text_of(child)
                if t:
                    return t

        for x in el.iterdescendants():
            if lname(x.tag).lower() in {"title", "heading"}:
                p = x.getparent()
                nearest_structural = None
                while p is not None and p is not el:
                    if self.normalized_type(p):
                        nearest_structural = p
                        break
                    p = p.getparent()
                if nearest_structural is None:
                    t = text_of(x)
                    if t:
                        return t
        return None

    def eid(self, el: etree._Element) -> Optional[str]:
        return attr_local(el, {"id", "eid"})

    def nearest_structural_parent(self, el: etree._Element) -> Optional[etree._Element]:
        p = el.getparent()
        while p is not None:
            if self.normalized_type(p):
                return p
            p = p.getparent()
        return None

    def structural_roots(self, root: etree._Element) -> list[etree._Element]:
        roots = []
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            if self.normalized_type(el) and self.nearest_structural_parent(el) is None:
                roots.append(el)
        return roots

    def immediate_structural_children(self, el: etree._Element) -> list[etree._Element]:
        out = []
        for cand in el.iterdescendants():
            if not self.normalized_type(cand):
                continue
            if self.nearest_structural_parent(cand) is el:
                out.append(cand)
        return out

    def extract_refs_for_node(self, el: etree._Element, node_id: str, eid: Optional[str]) -> None:
        for x in el.iter():
            if not isinstance(x.tag, str):
                continue

            href = attr_local(x, {
                "URI", "Uri", "uri",
                "href", "HRef", "Href",
                "ref", "Ref",
            })

            if not href:
                continue

            self.add_reference(
                source_node_id=node_id,
                source_eid=eid,
                anchor_text=text_of(x),
                href=href,
                element_name=lname(x.tag),
                method="CLML_STRUCTURED_REFERENCE",
            )

    def extract_annotations_and_effects(self, root: etree._Element) -> None:
        for el in root.iter():
            if not isinstance(el.tag, str):
                continue
            n = lname(el.tag).lower()
            txt = text_of(el)

            if n in {
                "commentary", "commentaryref", "commentaryitem",
                "margincommentary", "footnote", "note"
            } and txt:
                self.result.annotations.append({
                    "document_id": self.src["document_id"],
                    "variant": self.variant,
                    "element_name": lname(el.tag),
                    "id": attr_local(el, {"id", "eid"}),
                    "text": txt,
                    "text_sha256": sha256_text(txt),
                })

            # Keep legal effect metadata conservatively as raw candidates.
            # Different CLML exports can encode these differently.
            if n in {
                "changes", "change", "effect", "effects",
                "textualamendment", "commencement", "modification"
            }:
                attrs = {lname(k): v for k, v in el.attrib.items()}
                if txt or attrs:
                    self.result.legal_effects.append({
                        "document_id": self.src["document_id"],
                        "variant": self.variant,
                        "element_name": lname(el.tag),
                        "attributes": attrs,
                        "text": txt or None,
                    })

    def parse(self, root: etree._Element) -> ParseResult:
        def walk(el: etree._Element, parent_id: Optional[str], depth: int, path: list[int]) -> None:
            nt = self.normalized_type(el)
            if not nt:
                return

            number = self.number(el)
            heading = self.heading(el)
            eid = self.eid(el)
            txt = text_of(el)

            node_id = self.add_node(
                node_type=nt,
                number=number,
                heading=heading,
                text=txt,
                eid=eid,
                parent_id=parent_id,
                depth=depth,
                path=path,
                source_element=lname(el.tag),
            )

            self.extract_refs_for_node(el, node_id, eid)

            for i, child in enumerate(self.immediate_structural_children(el), start=1):
                walk(child, node_id, depth + 1, path + [i])

        for i, root_el in enumerate(self.structural_roots(root), start=1):
            walk(root_el, None, 0, [i])

        self.extract_annotations_and_effects(root)
        return self.result


# ---------------------------------------------------------------------------
# Candidate parse + scoring
# ---------------------------------------------------------------------------

def parse_candidate_bytes(
    *,
    source_key: str,
    src: dict[str, Any],
    variant: str,
    data: bytes,
) -> tuple[str, ParseResult, etree._Element]:
    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
        remove_blank_text=False,
        resolve_entities=False,
    )
    root = etree.fromstring(data, parser=parser)

    rep = detect_representation(root)

    if rep == "AKN":
        result = AKNParser(source_key, src, variant).parse(root)
    elif rep == "CLML":
        result = CLMLParser(source_key, src, variant).parse(root)
    else:
        raise RuntimeError(
            f"Could not confidently detect AKN/CLML; root={lname(root.tag)}"
        )

    return rep, result, root


def score_parse(src: dict[str, Any], result: ParseResult) -> tuple[int, int, int]:
    expected = src["expected_top_type"]
    top_count = sum(1 for n in result.nodes if n["node_type"] == expected)
    node_count = len(result.nodes)
    ref_count = len(result.references)

    # Strong preference for correct top-level provision count, then hierarchy,
    # then references.
    score = (
        top_count * 1_000_000
        + node_count * 1_000
        + min(ref_count, 10000) * 10
    )
    return score, top_count, node_count


# ---------------------------------------------------------------------------
# Regex cross-reference extraction
# ---------------------------------------------------------------------------

LEGAL_REF_PATTERNS = [
    # Sections
    (
        "SECTION",
        re.compile(
            r"\b(?:section|sections|s\.)\s+"
            r"(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?(?:\([ivx]+\))?)",
            re.I,
        ),
    ),
    # Regulations
    (
        "REGULATION",
        re.compile(
            r"\b(?:regulation|regulations|reg\.)\s+"
            r"(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?(?:\([ivx]+\))?)",
            re.I,
        ),
    ),
    (
        "SCHEDULE",
        re.compile(
            r"\bSchedule\s+(\d+[A-Z]?)",
            re.I,
        ),
    ),
    (
        "PART",
        re.compile(
            r"\bPart\s+(\d+[A-Z]?)",
            re.I,
        ),
    ),
]


def regex_reference_candidates(
    source_key: str,
    src: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for node in nodes:
        txt = node.get("text") or ""
        for ref_type, pat in LEGAL_REF_PATTERNS:
            for m in pat.finditer(txt):
                rows.append({
                    "source_document_id": src["document_id"],
                    "source_node_id": node["node_id"],
                    "source_key": source_key,
                    "reference_type": ref_type,
                    "locator": m.group(1),
                    "matched_text": m.group(0),
                    "start_char": m.start(),
                    "end_char": m.end(),
                    "extraction_method": "REGEX_LEGAL_REFERENCE",
                    "resolution_status": "UNRESOLVED",
                    "target_document_id": None,
                    "target_node_id": None,
                    "edge_status": "NOT_CREATED",
                    "confidence": 0.75,
                })

    return rows


# ---------------------------------------------------------------------------
# Full text
# ---------------------------------------------------------------------------

def build_full_text(src: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    """
    Readable reconstruction using top-level provisions only, avoiding duplicate
    repetition of every nested node because top-level node text already
    contains descendants.
    """
    expected = src["expected_top_type"]
    lines = [src["title"], src["root_url"], ""]

    for n in nodes:
        if n["node_type"] not in {"part", "chapter", "schedule", expected}:
            continue

        if n["node_type"] == expected:
            label = f"{expected.title()} {n.get('number') or ''}".strip()
            if n.get("heading"):
                label += f" — {n['heading']}"
            lines += [label, n["text"], ""]
        else:
            label = n["node_type"].upper()
            if n.get("number"):
                label += f" {n['number']}"
            if n.get("heading"):
                label += f" — {n['heading']}"
            lines += [label, ""]

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(
    source_key: str,
    src: dict[str, Any],
    result: ParseResult,
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = []
    counts = Counter(n["node_type"] for n in result.nodes)

    top_count = counts.get(src["expected_top_type"], 0)
    node_count = len(result.nodes)
    ref_count = len(result.references)

    if top_count < src["expected_min_top"]:
        reasons.append(
            f"top-level {src['expected_top_type']} count {top_count} "
            f"< expected minimum {src['expected_min_top']}"
        )

    if node_count < src["expected_min_nodes"]:
        reasons.append(
            f"node count {node_count} < expected minimum {src['expected_min_nodes']}"
        )

    if src.get("expect_structured_refs") and ref_count == 0:
        reasons.append(
            "structured reference count is zero although references are expected"
        )

    metrics = {
        "top_level_type": src["expected_top_type"],
        "top_level_count": top_count,
        "expected_min_top": src["expected_min_top"],
        "node_count": node_count,
        "expected_min_nodes": src["expected_min_nodes"],
        "reference_count": ref_count,
        "annotation_count": len(result.annotations),
        "legal_effect_count": len(result.legal_effects),
        "external_reference_candidate_count": len(
            result.external_reference_candidates
        ),
        "node_type_counts": dict(counts),
    }

    return (len(reasons) == 0), reasons, metrics


# ---------------------------------------------------------------------------
# Scraper orchestration
# ---------------------------------------------------------------------------

class InstrumentScraper:
    def __init__(
        self,
        source_key: str,
        output_dir: Path,
        timeout: int = 90,
        pause: float = 0.35,
    ):
        self.source_key = source_key
        self.src = SOURCES[source_key]
        self.output_dir = output_dir
        self.raw_dir = output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.pause = pause

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "MastersThesisProcurementResearchBot/4.0 "
                "(academic reproducible corpus acquisition)"
            ),
            "Accept-Language": "en-GB,en;q=0.9",
        })

    def fetch_and_parse_candidates(self) -> list[tuple[Candidate, Optional[ParseResult]]]:
        out = []

        for label, variant, url in candidate_specs(self.src):
            local_file = f"{label}.xml"
            cand = Candidate(
                label=label,
                variant=variant,
                url=url,
                local_file=local_file,
            )
            result = None

            try:
                r = self.session.get(
                    url,
                    headers={
                        "Accept": (
                            "application/akn+xml,application/xml,text/xml,"
                            "*/*;q=0.2"
                        )
                    },
                    timeout=self.timeout,
                    allow_redirects=True,
                )

                cand.status_code = r.status_code
                cand.final_url = r.url
                cand.content_type = r.headers.get("content-type", "")
                cand.byte_count = len(r.content)
                cand.sha256 = sha256_bytes(r.content)

                if r.status_code != 200:
                    cand.error = f"HTTP {r.status_code}"
                    out.append((cand, None))
                    time.sleep(self.pause)
                    continue

                path = self.raw_dir / local_file
                path.write_bytes(r.content)

                try:
                    rep, result, root = parse_candidate_bytes(
                        source_key=self.source_key,
                        src=self.src,
                        variant=variant,
                        data=r.content,
                    )
                    cand.xml_ok = True
                    cand.representation = rep
                    cand.parsed_ok = True
                    cand.reference_count = len(result.references)
                    cand.text_chars = len(text_of(root))
                    cand.score, cand.top_count, cand.node_count = score_parse(
                        self.src, result
                    )
                except Exception as exc:
                    cand.error = f"parse failed: {exc}"

            except Exception as exc:
                cand.error = str(exc)

            out.append((cand, result))
            time.sleep(self.pause)

        (self.raw_dir / "candidate_report.json").write_text(
            json.dumps(
                [asdict(c) for c, _ in out],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return out

    def choose_best_for_variant(
        self,
        parsed_candidates: list[tuple[Candidate, Optional[ParseResult]]],
        variant: str,
    ) -> tuple[Candidate, ParseResult]:
        valid = [
            (c, r)
            for c, r in parsed_candidates
            if c.variant == variant and c.parsed_ok and r is not None
        ]

        if not valid:
            raise RuntimeError(
                f"No parseable {variant} representation for {self.source_key}"
            )

        valid.sort(
            key=lambda cr: (
                cr[0].score,
                cr[0].top_count,
                cr[0].node_count,
                cr[0].reference_count,
                cr[0].byte_count,
            ),
            reverse=True,
        )

        return valid[0][0], valid[0][1]

    def save_variant(
        self,
        variant: str,
        cand: Candidate,
        result: ParseResult,
    ) -> dict[str, Any]:
        suffix = "latest" if variant == "latest" else "original"

        # Preserve selected source representation.
        selected_raw = self.raw_dir / f"selected_{suffix}.xml"
        shutil.copy2(self.raw_dir / cand.local_file, selected_raw)

        nodes_path = self.output_dir / f"nodes_{suffix}.jsonl"
        refs_path = self.output_dir / f"references_{suffix}.jsonl"
        ann_path = self.output_dir / f"annotations_{suffix}.jsonl"
        effects_path = self.output_dir / f"legal_effects_{suffix}.jsonl"
        external_path = (
            self.output_dir / f"external_reference_candidates_{suffix}.jsonl"
        )
        regex_path = self.output_dir / f"regex_reference_candidates_{suffix}.jsonl"
        text_path = self.output_dir / f"full_text_{suffix}.txt"

        write_jsonl(nodes_path, result.nodes)
        write_jsonl(refs_path, result.references)
        write_jsonl(ann_path, result.annotations)
        write_jsonl(effects_path, result.legal_effects)
        write_jsonl(external_path, result.external_reference_candidates)

        regex_rows = regex_reference_candidates(
            self.source_key, self.src, result.nodes
        )
        write_jsonl(regex_path, regex_rows)

        full_text = build_full_text(self.src, result.nodes)
        text_path.write_text(full_text, encoding="utf-8")

        passed, reasons, metrics = validate_result(
            self.source_key, self.src, result
        )

        variant_summary = {
            "variant": variant,
            "selected_candidate": asdict(cand),
            "validation_passed": passed,
            "validation_reasons": reasons,
            **metrics,
            "regex_reference_candidate_count": len(regex_rows),
            "full_text_char_count": len(full_text),
            "files": {
                "nodes": nodes_path.name,
                "references": refs_path.name,
                "annotations": ann_path.name,
                "legal_effects": effects_path.name,
                "external_reference_candidates": external_path.name,
                "regex_reference_candidates": regex_path.name,
                "full_text": text_path.name,
                "selected_raw": str(selected_raw.relative_to(self.output_dir)),
            },
        }

        return variant_summary

    def run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        parsed = self.fetch_and_parse_candidates()

        latest_cand, latest_result = self.choose_best_for_variant(parsed, "latest")
        original_cand, original_result = self.choose_best_for_variant(parsed, "original")

        latest_summary = self.save_variant(
            "latest", latest_cand, latest_result
        )
        original_summary = self.save_variant(
            "original", original_cand, original_result
        )

        # Strict validation: latest must pass for PA2023/PR2024.
        # PCR2015 latest consolidated representation can differ significantly
        # from the original due to amendments/repeals, therefore accept the
        # instrument if original passes and preserve both.
        hard_fail_reasons = []

        if self.source_key in {"PA2023", "PR2024"}:
            if not latest_summary["validation_passed"]:
                hard_fail_reasons.append(
                    "latest representation failed: "
                    + "; ".join(latest_summary["validation_reasons"])
                )

        if self.source_key == "PCR2015":
            if not original_summary["validation_passed"]:
                hard_fail_reasons.append(
                    "original/made representation failed: "
                    + "; ".join(original_summary["validation_reasons"])
                )

        # For all instruments, at least one representation must pass.
        if (
            not latest_summary["validation_passed"]
            and not original_summary["validation_passed"]
        ):
            hard_fail_reasons.append(
                "both latest and original representations failed validation"
            )

        # Canonical dataset:
        # PA/PR -> latest
        # PCR -> original for complete legacy structure, while latest is
        # retained separately for consolidated/current-status analysis.
        canonical_variant = (
            "original"
            if self.source_key == "PCR2015"
            else "latest"
        )

        canonical_result = (
            original_result
            if canonical_variant == "original"
            else latest_result
        )

        # Canonical compatibility files.
        write_jsonl(self.output_dir / "nodes.jsonl", canonical_result.nodes)
        write_jsonl(
            self.output_dir / "references.jsonl",
            canonical_result.references
        )
        write_jsonl(
            self.output_dir / "annotations.jsonl",
            canonical_result.annotations
        )
        write_jsonl(
            self.output_dir / "legal_effects.jsonl",
            canonical_result.legal_effects
        )
        write_jsonl(
            self.output_dir / "external_reference_candidates.jsonl",
            canonical_result.external_reference_candidates
        )

        canonical_regex = regex_reference_candidates(
            self.source_key,
            self.src,
            canonical_result.nodes,
        )
        write_jsonl(
            self.output_dir / "regex_reference_candidates.jsonl",
            canonical_regex
        )

        canonical_full_text = build_full_text(
            self.src,
            canonical_result.nodes,
        )
        (self.output_dir / "full_text.txt").write_text(
            canonical_full_text,
            encoding="utf-8",
        )

        summary = {
            "source_key": self.source_key,
            "document_id": self.src["document_id"],
            "title": self.src["title"],
            "scraper_version": VERSION,
            "generated_at": utc_now(),
            "canonical_variant": canonical_variant,
            "latest": latest_summary,
            "original": original_summary,
            "hard_validation_passed": len(hard_fail_reasons) == 0,
            "hard_validation_reasons": hard_fail_reasons,
        }

        (self.output_dir / "extraction_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        document = {
            "document_id": self.src["document_id"],
            "source_key": self.source_key,
            "title": self.src["title"],
            "root_url": self.src["root_url"],
            "contents_url": self.src["contents_url"],
            "legislation_type": self.src["type"],
            "year": self.src["year"],
            "number": self.src["number"],
            "corpus_role": self.src["corpus_role"],
            "canonical_variant": canonical_variant,
            "ingestion_scope": (
                "FULL_LEGACY"
                if self.source_key == "PCR2015"
                else "FULL"
            ),
            "authority_class": (
                "PRIMARY_LEGISLATION"
                if self.src["type"] == "ukpga"
                else "SECONDARY_LEGISLATION"
            ),
            "chunking_status": "NOT_CHUNKED",
            "graph_edge_status": "NOT_CREATED",
            "scraper_version": VERSION,
            "scraped_at": utc_now(),
        }

        (self.output_dir / "document.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if hard_fail_reasons:
            raise RuntimeError(
                "VALIDATION_FAILED: " + " | ".join(hard_fail_reasons)
            )

        return document, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["ALL", "PA2023", "PR2024", "PCR2015"],
        default="ALL",
    )
    ap.add_argument(
        "--output-dir",
        default="data/group_a_legislation_v4",
    )
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--pause", type=float, default=0.35)
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    selected = (
        ["PA2023", "PR2024", "PCR2015"]
        if args.source == "ALL"
        else [args.source]
    )

    documents = []
    summaries = []
    failures = []

    for key in selected:
        out_dir = root / key.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc, summary = InstrumentScraper(
                source_key=key,
                output_dir=out_dir,
                timeout=args.timeout,
                pause=args.pause,
            ).run()

            documents.append(doc)
            summaries.append(summary)

            canonical = summary[summary["canonical_variant"]]

            print(
                f"[OK] {key} "
                f"canonical={summary['canonical_variant']} "
                f"top={canonical['top_level_count']} "
                f"nodes={canonical['node_count']} "
                f"refs={canonical['reference_count']}"
            )

        except Exception as exc:
            failure = {
                "source_key": key,
                "document_id": SOURCES[key]["document_id"],
                "root_url": SOURCES[key]["root_url"],
                "error": str(exc),
                "status": "FAILED_VALIDATION_OR_EXTRACTION",
            }
            failures.append(failure)

            print(
                f"[FAILED] {key}: {exc}",
                file=sys.stderr,
            )

    write_jsonl(root / "documents.jsonl", documents)
    write_jsonl(root / "extraction_summaries.jsonl", summaries)
    write_jsonl(root / "failures.jsonl", failures)

    manifest = {
        "scraper_version": VERSION,
        "generated_at": utc_now(),
        "requested_sources": selected,
        "successful_sources": len(documents),
        "failed_sources": len(failures),
        "methodology": {
            "multiple_official_representations": True,
            "representation_detection": ["AKN", "CLML"],
            "representation_specific_parsing": True,
            "candidate_scoring_after_parsing": True,
            "latest_and_original_preserved": True,
            "contents_endpoint_used_as_body": False,
            "structured_references_extracted": True,
            "regex_reference_candidates_extracted": True,
            "annotations_preserved": True,
            "legal_effect_candidates_preserved": True,
            "full_text_generated": True,
            "llm_chunking": False,
            "semantic_graph_edges_inferred": False,
            "fail_fast_validation": True,
        },
        "sources": SOURCES,
    }

    (root / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Non-zero exit if anything failed.
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
