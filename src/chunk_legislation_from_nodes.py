#!/usr/bin/env python3
"""Chunk core-legislation instruments directly from their parsed structural tree.

`chunk_legislation_text.py` flattens the Act to prose and asks an LLM to both split it
and self-report which provision each piece states. That self-report is what broke: UK
statutes number a provision once ("1.—") then never repeat it before the following
subsections ("(1)", "(2)", ...), and the model conflates the repeated sub-paragraph
number with the provision identity - confirmed live in UKSI_2024_716, where 9 chunks
covering nine different regulations are all mislabeled "reg 1".

For the three core instruments the acquisition pipeline already parsed this correctly:
`processed/nodes.jsonl` is a full structural tree (part/chapter/section or regulation/
subsection/paragraph/schedule) with a stable eId straight from the source XML, and the
graph's edges already address chunks by exactly this id shape
(`UKPGA_2023_54__schedule-3-paragraph-1`). Walking that tree removes the LLM from the
identity question entirely: parent_node_id and citation are read off the parse, not
guessed from prose.

The unit chosen is the node whose own `text` field is already self-contained: for PA2023
and PR2024 that is `element_type in ("section", "regulation")`, which conveniently covers
both true top-level sections/regulations AND a schedule's own numbered paragraphs (the
parser reuses that same tag one level down inside a schedule). A schedule that has no
such children (a table, not numbered paragraphs - PR2024 Sch 3's CPV code list) has
nothing to select and is left for a separate row-splitting pass; it is not fabricated
into paragraphs it does not have.

PCR2015's tree is NOT used here: its regulation nodes carry no body text (mean ~1 token)
and its schedule nodes are far too small for what they cite, so the underlying parse is
incomplete and would only be trusted for citation numbering, never for the paragraph
text itself. Kept out of scope for this script.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"\w+")

INSTRUMENT_NAMES = {
    "UKPGA_2023_54": "Procurement Act 2023",
    "UKSI_2024_692": "Procurement Regulations 2024",
}
UNIT_WORD = {
    "UKPGA_2023_54": "s",
    "UKSI_2024_692": "reg",
}


def strip_prefix(eid: str, document_id: str) -> str:
    """eId is a full slug like 'http-www-legislation-gov-uk-ukpga-2023-54-section-1'.

    The part after the document's own slug is the addressable suffix already used by
    the graph ('section-1', 'schedule-3-paragraph-1', ...).
    """
    parts = eid.split("-")
    # Find where the document-specific suffix starts: after the fixed
    # "http-www-legislation-gov-uk-<type>-<year>-<num>" prefix. Locate it by matching
    # against the known numeric/year tokens rather than assuming a fixed word count,
    # since ukpga vs uksi vary in slug length.
    doc_tokens = document_id.lower().split("_")  # e.g. ['ukpga','2023','54']
    joined = "-".join(parts)
    marker = "-".join(doc_tokens)
    idx = joined.find(marker)
    if idx == -1:
        return eid
    rest = joined[idx + len(marker):].lstrip("-")
    return rest


def citation_for(document_id: str, node: dict, suffix: str) -> str:
    name = INSTRUMENT_NAMES[document_id]
    unit = UNIT_WORD[document_id]
    if suffix.startswith("schedule-"):
        m = re.match(r"schedule-([\w]+)(?:-paragraph-([\w]+))?", suffix)
        if m:
            sch, para = m.group(1), m.group(2)
            if para:
                return f"{name} Sch {sch} para {para}"
            return f"{name} Sch {sch}"
    num = node.get("number", "")
    return f"{name} {unit} {num}"


def est_tokens(text: str) -> int:
    return int(len(WORD.findall(text)) * 1.3)


def _normalize_node(n: dict) -> dict:
    """Accept either the acquisition pipeline's original field names
    (`element_type`/`eId`) or the v4 scraper's renamed equivalents
    (`node_type`/`eid`), so this script works against either scraper's output
    without the caller needing to know which one produced a given nodes.jsonl.
    Also guards against a null `eId`/`eid` (present on a small number of v4
    output nodes, e.g. some subparagraphs/subsections that the parser did not
    assign one to) by falling back to the always-present `node_id`, which is
    unique but not necessarily in the same slug shape as a real eId -- fine
    here since this fallback only ever applies to nodes this script does not
    select as chunks (schedule-child detection tolerates an unmatched id)."""
    if "element_type" not in n and "node_type" in n:
        n["element_type"] = n["node_type"]
    if not n.get("eId") and n.get("eid"):
        n["eId"] = n["eid"]
    if not n.get("eId"):
        n["eId"] = n.get("node_id", "")
    return n


def build(document_id: str, nodes_path: Path) -> list[dict]:
    nodes = [_normalize_node(json.loads(l)) for l in nodes_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # A schedule node is only selected itself when it has no section/regulation children
    # (a plain table, e.g. PR2024 Sch 3) - otherwise its paragraphs are the real units and
    # the schedule's own node would duplicate their text.
    schedule_eids_with_children = set()
    for n in nodes:
        if n["element_type"] in ("section", "regulation"):
            for s in nodes:
                if s["element_type"] == "schedule" and n["eId"].startswith(s["eId"] + "-"):
                    schedule_eids_with_children.add(s["eId"])

    selected = [
        n for n in nodes
        if n["element_type"] in ("section", "regulation")
        or (n["element_type"] == "schedule" and n["eId"] not in schedule_eids_with_children)
    ]

    out = []
    seen_parent_ids = set()
    for n in selected:
        text = (n.get("text") or "").strip()
        if not text:
            continue
        suffix = strip_prefix(n["eId"], document_id)
        parent_node_id = f"{document_id}__{suffix}"
        if parent_node_id in seen_parent_ids:
            continue  # a node can appear twice in the tree walk order; keep the first
        seen_parent_ids.add(parent_node_id)
        citation = citation_for(document_id, n, suffix)
        out.append({
            "parent_node_id": parent_node_id,
            "citation": citation,
            "element_type": n["element_type"],
            "text": text,
            "est_tokens": est_tokens(text),
            "char_count": len(text),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, choices=list(INSTRUMENT_NAMES))
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    chunks = build(a.doc, a.nodes)
    toks = sorted(c["est_tokens"] for c in chunks)
    n = len(toks)
    print(f"{a.doc}: {n} chunks, mean {sum(toks)//n} tok, p50 {toks[n//2]}, p90 {toks[int(.9*n)]}, max {toks[-1]}")
    over = sum(1 for t in toks if t > 1000)
    print(f"  >1000 tok: {over} ({100*over/n:.1f}%)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
