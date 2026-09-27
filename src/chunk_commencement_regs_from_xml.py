#!/usr/bin/env python3
"""Chunk UKSI_2024_716 and UKSI_2024_959 directly from their own source XML.

`chunk_legislation_text.py` produced these two instruments' currently-live chunks by
flattening to prose and asking an LLM to self-report which regulation each chunk states.
Confirmed broken: UKSI_2024_716 has 9 chunks covering nine different regulations all
mislabeled "reg 1", because the source text file starts mid-provision (its first
character is "(4)") and the model conflates a repeated sub-paragraph number "(1)/(2)/(3)"
with the regulation number, which is stated once and not repeated.

Reparsing the raw AKN/CLML XML with the acquisition pipeline's own parser
(group_a_legislation_scraper_v4.parse_candidate_bytes) sidesteps this - identity comes
from the XML, not a self-report - but the two instruments parse into different shapes:

UKSI_2024_959 (CLML): clean `regulation`-type nodes, each carrying its own full text and
a populated `number` field. Used directly.

UKSI_2024_716 (AKN): the parser's `regulation`-type nodes are a dead end here - their
text is a bare cross-reference mention ("reg. 5(1)"), not the provision. The real body
text sits on `hcontainer` nodes at depth 0, each opening with "<heading> <N>. <body>".
The heading and number are recovered from that opening text with a regex rather than
trusted from the node's own (empty) `number` field.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scrapers" / "legislation"))
from group_a_legislation_scraper_v4 import parse_candidate_bytes  # noqa: E402

WORD = re.compile(r"\w+")
HEADING_NUM = re.compile(r"^(?P<heading>.*?)\s*(?P<num>\d+[A-Z]?)\.\s*(?P<body>\(?\d.*)$", re.S)

INSTRUMENTS = {
    "UKSI_2024_716": dict(
        title="Procurement Act 2023 (Commencement No. 3 and Transitional and Saving Provisions) Regulations 2024",
        root_url="https://www.legislation.gov.uk/uksi/2024/716",
    ),
    "UKSI_2024_959": dict(
        title="Procurement Act 2023 (Commencement No. 3) (Amendment) Regulations 2024",
        root_url="https://www.legislation.gov.uk/uksi/2024/959",
    ),
}


def est_tokens(text: str) -> int:
    return int(len(WORD.findall(text)) * 1.3)


def build(document_id: str, xml_path: Path) -> list[dict]:
    info = INSTRUMENTS[document_id]
    kind, _, num = document_id.lower().split("_")
    src = dict(document_id=document_id, title=info["title"], type=kind, year=int(document_id.split("_")[1]),
               number=int(num), root_url=info["root_url"], contents_url=info["root_url"] + "/contents",
               expected_top_type="regulation", corpus_role="PA2023_COMMENCEMENT")
    data = xml_path.read_bytes()
    rep, result, _ = parse_candidate_bytes(source_key=document_id, src=src, variant="latest", data=data)

    out = []
    if rep == "CLML":
        regs = [n for n in result.nodes if n["node_type"] == "regulation" and n.get("number")]
        for n in regs:
            text = (n["text"] or "").strip()
            if not text:
                continue
            num_str = n["number"]
            out.append(dict(
                parent_node_id=f"{document_id}__regulation-{num_str}",
                citation=f"{info['title']} reg {num_str}",
                text=text, est_tokens=est_tokens(text), char_count=len(text),
            ))
    else:  # AKN
        candidates = [n for n in result.nodes if n["node_type"] == "hcontainer" and n.get("depth") == 0]
        for n in candidates:
            text = (n["text"] or "").strip()
            m = HEADING_NUM.match(text)
            if not m:
                continue  # signature block, schedule preamble etc - not a numbered regulation
            num_str = m.group("num")
            out.append(dict(
                parent_node_id=f"{document_id}__regulation-{num_str}",
                citation=f"{info['title']} reg {num_str}",
                text=text, est_tokens=est_tokens(text), char_count=len(text),
            ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, choices=list(INSTRUMENTS))
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    chunks = build(a.doc, a.xml)
    print(f"{a.doc}: {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c['citation']:70s} ({c['est_tokens']} tok)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
