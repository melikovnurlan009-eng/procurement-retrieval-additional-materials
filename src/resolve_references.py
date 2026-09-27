#!/usr/bin/env python3
"""Reference resolution for the procurement legal knowledge graph.

Turns reference CANDIDATES into graph edges only where the target can be identified
with evidence. The graph must express explicit legal relationships, so precision is
preferred over edge count and every unresolved case is retained for review rather than
guessed into an edge.

Candidate families
------------------
STRUCTURED_REFERENCE  reference elements in the legislation XML (internal eId hrefs,
                      legislation.gov.uk URLs, opaque CLML keys)
REGEX_REFERENCE       textual citations - "section 23(3)(a)", "regulation 5", "Schedule 7"
PLACEHOLDER_LINK      hyperlinks captured from guidance/commentary pages

Resolution ladder (only the first four create production edges)
--------------------------------------------------------------
EXACT_INTERNAL        provision located inside the citing instrument
EXACT_URL             legislation.gov.uk URL parsed to a provision present in the corpus
EXACT_DOCUMENT        link resolved to a corpus document
NEAREST_ANCESTOR      deepest existing ancestor of a deeper locator (e.g. s.83B(5)(b)(i)
                      resolves to s.83B when subparagraph nodes are absent)
DOCUMENT_ONLY         target document known, provision not present
TARGET_NOT_IN_CORPUS  identifiable legislation outside thesis scope -> acquisition candidate
AMBIGUOUS             locator type conflicts with instrument type, or context is unsafe
NOT_A_LEGAL_REFERENCE mailto, anchors, non-legal web links
UNRESOLVED            opaque or uninterpretable

The critical safeguard
----------------------
An unqualified "section 18" inside PA2023 normally means PA2023 s.18 - but consequential
amendment provisions cite OTHER Acts ("section 18 of the Freedom of Information Act
2000", "for section 5 substitute"). Resolving those internally would fabricate
cross-references between unrelated statutes. Every internal resolution therefore
inspects the surrounding text and refuses when an external instrument is named.

Usage
-----
    python resolve_references.py --corpus-dir data/search_corpus
    python resolve_references.py --corpus-dir data/search_corpus --sample-for-review 300
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse

RESOLVER_VERSION = "1.0.0"

EDGE_CREATING = {"EXACT_INTERNAL", "EXACT_CROSS_DOCUMENT", "EXACT_URL", "EXACT_DOCUMENT", "NEAREST_ANCESTOR"}

# legislation.gov.uk path -> canonical document id prefix
DOC_TYPE_MAP = {"ukpga": "UKPGA", "uksi": "UKSI", "ukdsi": "UKDSI", "asp": "ASP", "nisr": "NISR"}

# Instruments in the corpus, by canonical id.
IN_SCOPE_DOCS = {"UKPGA_2023_54", "UKSI_2024_692", "UKSI_2015_102"}

LEG_URL_RE = re.compile(
    r"legislation\.gov\.uk/(?P<type>[a-z]+)/(?P<year>\d{4})/(?P<num>\d+)"
    r"(?:/(?P<unit>section|regulation|schedule|article|part|paragraph)/(?P<locator>[\w.\-]+))?",
    re.I,
)

# "section 18 of the Freedom of Information Act 2000" / "in the 2015 Regulations"
EXTERNAL_QUALIFIER_RE = re.compile(
    r"\bof\s+(the\s+)?[A-Z][\w'’\-]*(\s+[\w'’\-]+){0,7}\s+(Act|Regulations|Order|Rules)\b"
    r"|\bof\s+that\s+(Act|Order|Regulations)\b",
    re.I,
)
# Amendment framing: the locator belongs to the instrument BEING AMENDED, not this one.
AMENDMENT_CONTEXT_RE = re.compile(
    r"\b(amend(?:ment|ed|s)?|insert|substitute|omit|repeal|revoke)\b", re.I
)

UNIT_BY_DOC_PREFIX = {"UKPGA": "section", "UKSI": "regulation", "UKDSI": "regulation"}

# Instruments the corpus contains, recognised by how they are named in running text.
# Used to convert a correctly-detected external citation into a real cross-document
# edge when the named instrument IS in scope, instead of discarding it.
NAMED_INSTRUMENT_PATTERNS = [
    (re.compile(r"\b(procurement act 2023|PA\s?2023|the 2023 Act)\b", re.I), "UKPGA_2023_54"),
    (re.compile(r"\b(procurement regulations 2024|PR\s?2024|the 2024 Regulations)\b", re.I), "UKSI_2024_692"),
    (re.compile(r"\b(public contracts regulations 2015|PCR\s?2015|the 2015 Regulations)\b", re.I), "UKSI_2015_102"),
]

# An SI's unqualified "section N" almost always cites the Act it is made under.
# Only asserted where the enabling relationship is a matter of record.
ENABLING_ACT = {"UKSI_2024_692": "UKPGA_2023_54"}


def named_instrument(text: str) -> str | None:
    for pattern, doc_id in NAMED_INSTRUMENT_PATTERNS:
        if pattern.search(text or ""):
            return doc_id
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_doc_id(doc_type: str, year: str, num: str) -> str:
    return f"{DOC_TYPE_MAP.get(doc_type.lower(), doc_type.upper())}_{year}_{num}"


def split_locator(locator: str) -> list[str]:
    """'83B(5)(b)(i)' -> ['83B','5','b','i']; '5' -> ['5']."""
    locator = (locator or "").strip().rstrip(".")
    if not locator:
        return []
    head = re.match(r"^([0-9]+[A-Za-z]*)", locator)
    parts = [head.group(1)] if head else []
    parts += re.findall(r"\(([^)]+)\)", locator)
    return [p.strip() for p in parts if p.strip()]


class Corpus:
    """Node universe and URL index used for target lookup."""

    def __init__(self, root: Path, corpus_dir: Path):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_text: dict[str, str] = {}
        for path in sorted((root / "data" / "group_a_legislation_v4").glob("*/nodes_*.jsonl")):
            for row in read_jsonl(path):
                nid = row.get("node_id")
                if nid and nid not in self.nodes:
                    self.nodes[nid] = row
                    self.node_text[nid] = row.get("text") or ""
        self.documents: set[str] = {n.split("__", 1)[0] for n in self.nodes}

        self.url_to_doc: dict[str, str] = {}
        for line in (corpus_dir / "parent_segments.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            seg = json.loads(line)
            url = seg.get("source_url")
            if url:
                self.url_to_doc[urldefrag(url).url.rstrip("/").lower()] = seg["document_id"]

    def node_id_for(self, doc_id: str, unit: str, parts: list[str]) -> tuple[str | None, str]:
        """Deepest existing node for a locator, plus how it was found."""
        if not parts:
            return None, "NO_LOCATOR"
        candidate = f"{doc_id}__{unit}-" + "-".join(parts)
        if candidate in self.nodes:
            return candidate, "EXACT"
        for depth in range(len(parts) - 1, 0, -1):
            anc = f"{doc_id}__{unit}-" + "-".join(parts[:depth])
            if anc in self.nodes:
                return anc, "ANCESTOR"
        return None, "ABSENT"

    def context_window(self, node_id: str, start: int | None, end: int | None, width: int = 140) -> str:
        text = self.node_text.get(node_id, "")
        if not text or start is None:
            return ""
        return text[max(0, start - width) : (end or start) + width]


def resolve_regex_reference(raw: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    src_doc = raw.get("source_document_id") or ""
    ref_type = (raw.get("reference_type") or "").upper()
    locator = raw.get("locator") or ""
    node_id = raw.get("source_node_id")
    ctx = corpus.context_window(node_id, raw.get("start_char"), raw.get("end_char"))

    # Safeguard: text after the citation naming another instrument means the locator is
    # not ours to resolve internally.
    after = ""
    if ctx and raw.get("matched_text"):
        idx = ctx.find(raw["matched_text"])
        after = ctx[idx + len(raw["matched_text"]) : idx + len(raw["matched_text"]) + 90] if idx >= 0 else ""
    if after and EXTERNAL_QUALIFIER_RE.search(after):
        # The citation names another instrument. If that instrument is in the corpus,
        # this is a genuine cross-document reference worth an edge; only otherwise is
        # it out of scope.
        other = named_instrument(after)
        if other and other != src_doc:
            unit_o = UNIT_BY_DOC_PREFIX.get(other.split("_", 1)[0], "section")
            tgt, how = corpus.node_id_for(other, unit_o, split_locator(locator))
            if tgt and how == "EXACT":
                return {"status": "EXACT_CROSS_DOCUMENT", "target_node_id": tgt,
                        "target_document_id": other, "reason": "citation names an in-corpus instrument",
                        "evidence": after.strip()[:120], "confidence": 0.9}
            if tgt:
                return {"status": "NEAREST_ANCESTOR", "target_node_id": tgt, "target_document_id": other,
                        "evidence": after.strip()[:120], "confidence": 0.7}
            return {"status": "DOCUMENT_ONLY", "target_document_id": other,
                    "evidence": after.strip()[:120], "confidence": 0.5}
        return {
            "status": "TARGET_NOT_IN_CORPUS",
            "reason": "citation qualified by another instrument",
            "evidence": after.strip()[:120],
            "confidence": 0.9,
        }
    if ctx and AMENDMENT_CONTEXT_RE.search(ctx) and EXTERNAL_QUALIFIER_RE.search(ctx):
        return {
            "status": "AMBIGUOUS",
            "reason": "amendment context citing another instrument",
            "evidence": ctx.strip()[:120],
            "confidence": 0.5,
        }

    prefix = src_doc.split("_", 1)[0]
    expected_unit = UNIT_BY_DOC_PREFIX.get(prefix)
    unit = {"SECTION": "section", "REGULATION": "regulation", "SCHEDULE": "schedule", "PART": "part"}.get(ref_type)
    if not unit:
        return {"status": "UNRESOLVED", "reason": f"unknown reference_type {ref_type}", "confidence": 0.0}

    # An Act citing "regulation 5" (or an SI citing "section 5") is pointing outside
    # itself; resolving internally would invent a relationship.
    if unit in {"section", "regulation"} and expected_unit and unit != expected_unit:
        # An SI has regulations, not sections, so "section 45(7)" points outside this
        # instrument - normally at the Act it was made under. Resolve there when the
        # enabling relationship is known, otherwise leave it out of scope rather than
        # inventing an internal link.
        parent = ENABLING_ACT.get(src_doc)
        if unit == "section" and parent:
            tgt, how = corpus.node_id_for(parent, "section", split_locator(locator))
            if tgt and how == "EXACT":
                return {"status": "EXACT_CROSS_DOCUMENT", "target_node_id": tgt, "target_document_id": parent,
                        "reason": f"unqualified section cited in SI resolved to enabling Act {parent}",
                        "confidence": 0.8}
            if tgt:
                return {"status": "NEAREST_ANCESTOR", "target_node_id": tgt, "target_document_id": parent,
                        "reason": "resolved to enabling Act ancestor", "confidence": 0.65}
        return {
            "status": "EXTERNAL_INSTRUMENT_REFERENCE",
            "reason": f"{unit} cited inside {prefix} whose provisions are '{expected_unit}'",
            "confidence": 0.5,
        }

    parts = split_locator(locator)
    target, how = corpus.node_id_for(src_doc, unit, parts)
    if target and how == "EXACT":
        return {"status": "EXACT_INTERNAL", "target_node_id": target, "target_document_id": src_doc, "confidence": 0.95}
    if target and how == "ANCESTOR":
        return {
            "status": "NEAREST_ANCESTOR",
            "target_node_id": target,
            "target_document_id": src_doc,
            "reason": f"deeper locator {locator} not present as a node",
            "confidence": 0.75,
        }
    return {"status": "DOCUMENT_ONLY", "target_document_id": src_doc, "reason": f"{unit} {locator} absent", "confidence": 0.4}


def resolve_url(url: str, corpus: Corpus) -> dict[str, Any]:
    if not url:
        return {"status": "UNRESOLVED", "reason": "no url", "confidence": 0.0}
    low = url.lower()
    if low.startswith("mailto:") or low.startswith("tel:"):
        return {"status": "NOT_A_LEGAL_REFERENCE", "reason": "mailto/tel", "confidence": 1.0}

    m = LEG_URL_RE.search(url)
    if m:
        doc_id = canonical_doc_id(m.group("type"), m.group("year"), m.group("num"))
        if doc_id not in corpus.documents:
            return {
                "status": "TARGET_NOT_IN_CORPUS",
                "target_document_id": doc_id,
                "reason": "legislation outside thesis scope",
                "acquisition_candidate": True,
                "confidence": 0.9,
            }
        unit, loc = m.group("unit"), m.group("locator")
        if unit and loc:
            target, how = corpus.node_id_for(doc_id, unit.lower(), split_locator(loc))
            if target and how == "EXACT":
                return {"status": "EXACT_URL", "target_node_id": target, "target_document_id": doc_id, "confidence": 0.98}
            if target:
                return {"status": "NEAREST_ANCESTOR", "target_node_id": target, "target_document_id": doc_id, "confidence": 0.75}
            return {"status": "DOCUMENT_ONLY", "target_document_id": doc_id, "confidence": 0.5}
        return {"status": "EXACT_DOCUMENT", "target_document_id": doc_id, "confidence": 0.9}

    key = urldefrag(url).url.rstrip("/").lower()
    if key in corpus.url_to_doc:
        return {"status": "EXACT_DOCUMENT", "target_document_id": corpus.url_to_doc[key], "confidence": 0.9}
    host = urlparse(url).netloc.lower()
    if host:
        return {"status": "TARGET_NOT_IN_CORPUS", "reason": f"external web target ({host})", "confidence": 0.6}
    return {"status": "UNRESOLVED", "reason": "unparseable url", "confidence": 0.0}


def resolve_structured(raw: dict[str, Any], corpus: Corpus) -> dict[str, Any]:
    href = (raw.get("raw_href") or "").strip()
    src_doc = raw.get("source_document_id") or ""
    if href.startswith("http"):
        return resolve_url(raw.get("absolute_url") or href, corpus)
    if href.startswith("key-"):
        # Opaque CLML internal keys carry no target identity on their own.
        return {"status": "UNRESOLVED", "reason": "opaque CLML key", "confidence": 0.0}
    m = re.match(r"^(section|regulation|schedule|part|article)[-_]?([\w\-.]*)$", href, re.I)
    if m:
        unit = m.group(1).lower()
        parts = [p for p in re.split(r"[-.]", m.group(2)) if p]
        if not parts:
            return {"status": "DOCUMENT_ONLY", "target_document_id": src_doc, "reason": f"bare {unit} href", "confidence": 0.4}
        target, how = corpus.node_id_for(src_doc, unit, parts)
        if target and how == "EXACT":
            return {"status": "EXACT_INTERNAL", "target_node_id": target, "target_document_id": src_doc, "confidence": 0.95}
        if target:
            return {"status": "NEAREST_ANCESTOR", "target_node_id": target, "target_document_id": src_doc, "confidence": 0.75}
    return {"status": "UNRESOLVED", "reason": f"uninterpretable href '{href[:40]}'", "confidence": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", default="data/search_corpus", type=Path)
    ap.add_argument("--sample-for-review", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else root / args.corpus_dir
    corpus = Corpus(root, corpus_dir)
    candidates = read_jsonl(corpus_dir / "unresolved_references.jsonl")

    resolved, edges, unresolved, acquisitions = [], [], [], []
    edge_keys: set[tuple] = set()

    for cand in candidates:
        ctype = cand.get("candidate_type")
        raw = cand.get("raw") or {}
        if ctype == "REGEX_REFERENCE":
            res = resolve_regex_reference(raw, corpus)
            source_id = raw.get("source_node_id")
            relation = "CROSS_REFERS_TO"
            evidence = raw.get("matched_text")
        elif ctype == "STRUCTURED_REFERENCE":
            res = resolve_structured(raw, corpus)
            source_id = raw.get("source_node_id")
            relation = "CROSS_REFERS_TO"
            evidence = raw.get("anchor_text") or raw.get("raw_href")
        elif ctype == "PLACEHOLDER_LINK":
            res = resolve_url(cand.get("url") or "", corpus)
            source_id = cand.get("source_chunk_id")
            relation = "REFERENCES"
            evidence = cand.get("anchor_text")
        else:
            res = {"status": "UNRESOLVED", "reason": f"unknown candidate_type {ctype}", "confidence": 0.0}
            source_id, relation, evidence = None, "REFERENCES", None

        row = {
            "candidate_type": ctype,
            "source_id": source_id,
            "source_document_id": raw.get("source_document_id") or cand.get("source_document_id"),
            "evidence_text": evidence,
            "extraction_method": raw.get("extraction_method") or ctype,
            "resolver_version": RESOLVER_VERSION,
            **res,
        }
        resolved.append(row)

        if res.get("acquisition_candidate"):
            acquisitions.append({"target_document_id": res.get("target_document_id"), "evidence": evidence})

        target = res.get("target_node_id") or res.get("target_document_id")
        if res["status"] in EDGE_CREATING and source_id and target:
            key = (source_id, relation, target)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges.append(
                {
                    "edge_id": f"E{len(edges) + 1:06d}",
                    "source_id": source_id,
                    "relation": relation,
                    "target_id": target,
                    "evidence_method": row["extraction_method"],
                    "evidence_text": evidence,
                    "resolution_status": res["status"],
                    "confidence": res.get("confidence"),
                    "resolver_version": RESOLVER_VERSION,
                }
            )
        else:
            unresolved.append(row)

    status_counts = collections.Counter(r["status"] for r in resolved)
    by_type = collections.defaultdict(collections.Counter)
    for r in resolved:
        by_type[r["candidate_type"]][r["status"]] += 1

    report = {
        "resolver_version": RESOLVER_VERSION,
        "generated_at": now_iso(),
        "candidates": len(candidates),
        "resolved_rows": len(resolved),
        "edges_created": len(edges),
        "edge_yield_pct": round(100 * len(edges) / max(1, len(candidates)), 1),
        "held_for_review": len(unresolved),
        "status_distribution": dict(status_counts.most_common()),
        "status_by_candidate_type": {k: dict(v.most_common()) for k, v in by_type.items()},
        "edge_relations": dict(collections.Counter(e["relation"] for e in edges)),
        "confidence_distribution": dict(collections.Counter(str(e["confidence"]) for e in edges).most_common()),
        "acquisition_candidates": dict(
            collections.Counter(a["target_document_id"] for a in acquisitions if a["target_document_id"]).most_common(20)
        ),
        "edge_creating_statuses": sorted(EDGE_CREATING),
    }

    with (corpus_dir / "edges_v2.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (corpus_dir / "unresolved_references_v2.jsonl").open("w", encoding="utf-8") as f:
        for r in unresolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (corpus_dir / "reference_resolution_all.jsonl").open("w", encoding="utf-8") as f:
        for r in resolved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (corpus_dir / "reference_resolution_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Stratified manual-review sample: edge precision must be measured, not assumed.
    if args.sample_for_review:
        import random

        rng = random.Random(args.seed)
        strata: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in resolved:
            strata[(r["candidate_type"], r["status"])].append(r)
        per = max(1, args.sample_for_review // max(1, len(strata)))
        sample = []
        for key in sorted(strata, key=str):
            pool = strata[key][:]
            rng.shuffle(pool)
            for r in pool[:per]:
                sample.append({**r, "MANUAL_VERDICT": "", "MANUAL_NOTES": ""})
        with (corpus_dir / "reference_resolution_review_sample.jsonl").open("w", encoding="utf-8") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["review_sample"] = len(sample)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
