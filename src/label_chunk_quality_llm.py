#!/usr/bin/env python3
"""Batch LLM labelling of chunk quality, judged as RETRIEVAL UNITS.

The structural detectors test whether a boundary is syntactically damaged. They cannot see
that a chunk is a navigation footer, or that an applicability clause has been severed from
the rule it governs - both were found by reading, not by measurement. This asks a model the
question the detectors cannot: would this chunk function as evidence if a search returned it?

Chunks are sent in batches so the model sees them in context and labels consistently, and
because per-chunk calls over a corpus of this size are not worth the cost.

Labels
------
GOOD        Self-contained and substantive. States a rule, definition, procedure or
            explanation that could stand alone as evidence in an answer.
INCOMPLETE  Real content, but severed. An applicability clause without its rule, a list
            without its stem, a sentence cut mid-clause, a section fragment that refers to
            something absent. Would mislead or frustrate a reader on its own.
LOW_VALUE   No retrieval value regardless of completeness: navigation, boilerplate advice
            ("seek legal advice"), page furniture, extraction garbage, tables of contents,
            or text too thin to answer anything.

The distinction that matters is INCOMPLETE versus LOW_VALUE: the first is a chunking failure
worth fixing by re-segmentation, the second is an ingestion failure worth fixing by filtering.
They have different remedies, so they are labelled separately rather than as one "bad" class.

Usage
-----
    python label_chunk_quality_llm.py --sample 800 --batch 25
"""
from __future__ import annotations

import argparse, collections, json, os, random, re, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

SYSTEM = """You assess whether chunks of UK public procurement material would work as RETRIEVAL UNITS in a legal search system.

A user asks a procurement law question; the system returns chunks as evidence. Judge each chunk on whether it could serve that purpose ON ITS OWN.

Assign exactly one label per chunk:

GOOD - self-contained and substantive. States a rule, condition, definition, procedure or explanation that a reader could rely on without needing an adjacent chunk. Legislation stating an obligation, guidance explaining a process, a complete definition.

INCOMPLETE - contains real content but has been cut. Signs: opens with "This section applies..." and never says what the section requires; a list with no introductory stem; a stem with no list; starts or ends mid-sentence; refers to "the following" or "such cases" that are not present; a fragment of a provision whose operative part is missing.

LOW_VALUE - would not help any query even if complete. Navigation menus, breadcrumbs, "Additional support and guidance / seek legal advice" footers, copyright notices, contents listings, headings alone, extraction garbage (words split across lines, spreadsheet formulas, layout artifacts), or content too thin to answer anything.

Guidance:
- Judge the chunk as retrieved, not the document it came from. A page about exclusion grounds whose chunk contains only a footer is LOW_VALUE.
- Length alone decides nothing. A 20-token provision stating a complete rule is GOOD. A 3,000-token chunk of boilerplate is LOW_VALUE.
- Legal formatting is normal. Numbered subsections, lettered paragraphs and enumerated conditions are how statute is written, not damage.
- If a chunk is both cut and worthless, label LOW_VALUE.

Return one entry per chunk, in the order given, each with the chunk_ref, the label, and a short reason (under 15 words)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_ref": {"type": "integer"},
                    "label": {"type": "string", "enum": ["GOOD", "INCOMPLETE", "LOW_VALUE"]},
                    "reason": {"type": "string"},
                },
                "required": ["chunk_ref", "label", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--sample", type=int, default=800)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--chars", type=int, default=1100)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", default="evaluation/chunk_quality_llm", type=Path)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = root / a.out; out.mkdir(parents=True, exist_ok=True)

    import sqlite3
    con = sqlite3.connect(root / a.db); con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT chunk_id, citation, retrieval_title, authority_class, source_kind, "
        "jurisdiction, est_tokens, parent_node_id, source_url, text "
        "FROM chunks WHERE superseded_by IS NULL")]

    # Stratify by token band so small and large chunks are both represented, rather than
    # letting the corpus's own skew decide what gets labelled.
    def band(t):
        for lo, hi, n in ((0,50,"0-50"),(50,200,"50-200"),(200,500,"200-500"),
                          (500,1000,"500-1k"),(1000,3000,"1k-3k"),(3000,10**9,"3k+")):
            if lo <= (t or 0) < hi: return n
        return "?"
    by = collections.defaultdict(list)
    for r in rows: by[band(r["est_tokens"])].append(r)
    random.seed(a.seed)
    per = max(1, a.sample // len(by))
    sample = []
    for k, v in by.items():
        random.shuffle(v); sample += v[:per]
    random.shuffle(sample)
    print(f"corpus {len(rows)} chunks -> labelling {len(sample)} "
          f"({per} per band across {len(by)} bands), batch={a.batch}", flush=True)

    done = {}
    outf = out / "labels.jsonl"
    if outf.exists():
        for l in outf.open(encoding="utf-8"):
            x = json.loads(l); done[x["chunk_id"]] = x
    todo = [r for r in sample if r["chunk_id"] not in done]
    print(f"already labelled {len(done)}, to do {len(todo)}", flush=True)

    from openai import OpenAI
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    fails = 0
    for bi, b in enumerate(batches, 1):
        payload = [{"chunk_ref": i,
                    "citation": r["citation"], "title": r["retrieval_title"],
                    "authority_class": r["authority_class"], "tokens": r["est_tokens"],
                    "text": (r["text"] or "")[: a.chars]} for i, r in enumerate(b)]
        # Rate limits are expected on batched calls of this size; without backoff a 429
        # silently drops 25 chunks and the sample quietly becomes non-random.
        resp = None
        for attempt in range(5):
            try:
                resp = cl.chat.completions.create(
                    model=a.model, temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "chunk_labels", "schema": SCHEMA,
                                                     "strict": True}})
                break
            except Exception as exc:
                if "429" not in str(exc) or attempt == 4:
                    if attempt == 4: print(f"  batch {bi} gave up after retries", flush=True)
                    break
                wait = 2 ** attempt * 5
                print(f"  batch {bi} rate-limited, retrying in {wait}s", flush=True)
                time.sleep(wait)
        try:
            if resp is None:
                raise RuntimeError("no response")
            got = {x["chunk_ref"]: x for x in json.loads(resp.choices[0].message.content)["labels"]}
            with outf.open("a", encoding="utf-8") as f:
                for i, r in enumerate(b):
                    g = got.get(i)
                    if not g: continue
                    f.write(json.dumps({
                        "chunk_id": r["chunk_id"], "label": g["label"], "reason": g["reason"],
                        "citation": r["citation"], "authority_class": r["authority_class"],
                        "source_kind": r["source_kind"], "jurisdiction": r["jurisdiction"],
                        "est_tokens": r["est_tokens"], "band": band(r["est_tokens"]),
                        "has_legal_identity": bool(r["parent_node_id"]),
                        "domain": re.sub(r"https?://(www\.)?([^/]+).*", r"\2", r["source_url"] or "") or "?",
                        "labeller": f"llm:{a.model}", "labelled_at": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
        except Exception as exc:
            fails += 1
            print(f"  batch {bi} FAILED: {str(exc)[:120]}", flush=True)
        if bi % 5 == 0 or bi == len(batches):
            print(f"  {bi}/{len(batches)} batches ({fails} failures)", flush=True)
        time.sleep(0.2)

    labs = [json.loads(l) for l in outf.open(encoding="utf-8")]
    def dist(rs):
        c = collections.Counter(x["label"] for x in rs); n = len(rs)
        return {k: f"{c.get(k,0)} ({100*c.get(k,0)/n:.0f}%)" for k in ("GOOD","INCOMPLETE","LOW_VALUE")}
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": a.model,
           "labelled": len(labs), "overall": dist(labs),
           "by_band": {k: dist(v) for k, v in sorted(collections.defaultdict(
               list, {b: [x for x in labs if x["band"] == b] for b in {y["band"] for y in labs}}).items())},
           "by_domain": {k: dist([x for x in labs if x["domain"] == k])
                         for k in collections.Counter(x["domain"] for x in labs)},
           "by_authority": {k: dist([x for x in labs if x["authority_class"] == k])
                            for k in collections.Counter(x["authority_class"] for x in labs)}}
    (out / "summary.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\nOVERALL:", rep["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
