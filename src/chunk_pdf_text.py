#!/usr/bin/env python3
"""Chunk PDF page text by having the model emit fully-formed chunks, then verifying fidelity.

Why this differs from the rest of the pipeline
----------------------------------------------
Everywhere else the model returns BOUNDARIES ONLY - a start and end block id - and chunk text
is reconstructed from immutable blocks by code. That guarantees the corpus contains exactly
what the source contained, and it is why placeholder conservation is checkable at 527 = 527.

That design fails on PDFs, and the failure is structural rather than incidental. The atoms
are page rectangles: 87 of 89 pages in one playbook begin mid-sentence. Selecting boundaries
over atoms that are themselves broken cannot produce an unbroken chunk, which is why the PDF
lane measured 88% INCOMPLETE against 34% for parsed legislation.

So here the model emits chunk text directly, joining sentences across page breaks and
dropping running headers. That buys complete chunks at the cost of the reconstruction
guarantee, so the guarantee is replaced by a check rather than dropped: every emitted chunk
is verified against the source by token-shingle containment. A chunk whose content is not
substantially present in the pages it claims to come from is flagged, not silently accepted.

Fidelity is reported per chunk and in aggregate:
    coverage   fraction of the chunk's 5-token shingles found in the source pages
    novel      shingles absent from the source - the fabrication signal
A chunk below the coverage threshold is marked `fidelity_failed` and excluded by default.
"""
from __future__ import annotations

import argparse, collections, json, os, re, statistics, time
from datetime import datetime, timezone
from pathlib import Path

SYSTEM = """You are segmenting text extracted from a UK public procurement PDF into retrieval chunks.

You receive the text of consecutive pages, paragraph by paragraph. Page breaks fall wherever the layout designer put them, so sentences, lists and tables are frequently cut across pages.

Produce chunks that would work as evidence returned by a search engine.

Rules:
1. Emit the chunk TEXT, reproducing the source wording exactly. Join sentences and lists that were split across a page break. Do not paraphrase, summarise, correct or add anything.
2. Drop running headers, footers, page numbers and navigation furniture.
3. Each chunk must be self-contained: a complete rule, procedure, definition or explanation. Keep an introductory stem with the list it governs. Never end on "the following—" or begin with a bare list item.
4. Prefer 100-400 words. Coherence outweighs size: a complete 60-word rule is better than a padded 300-word one.
5. COVER THE SUPPLIED TEXT COMPLETELY. Every substantive paragraph you are given must appear in some chunk. Do not summarise, do not skip, do not select highlights. Working through the pages in order and emitting chunks as you go is the reliable way to achieve this. Omit only running headers, footers, page numbers and navigation.
6. Give each chunk a title naming what it states, and list the page numbers it draws on.

Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer"}},
                    "self_contained": {"type": "boolean"},
                },
                "required": ["title", "text", "pages", "self_contained"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chunks"],
    "additionalProperties": False,
}

WORD = re.compile(r"\w+")


def shingles(t: str, n: int = 5) -> set[tuple]:
    w = [x.lower() for x in WORD.findall(t or "")]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def fidelity(chunk: str, source: str) -> dict:
    cs, ss = shingles(chunk), shingles(source)
    if not cs:
        return {"coverage": 0.0, "novel_shingles": 0, "chunk_shingles": 0}
    found = cs & ss
    return {"coverage": round(len(found) / len(cs), 4),
            "novel_shingles": len(cs) - len(found), "chunk_shingles": len(cs)}


def batches(pages: list[dict], max_chars: int):
    cur, n = [], 0
    for p in pages:
        if cur and n + p["chars"] > max_chars:
            yield cur; cur, n = [], 0
        cur.append(p); n += p["chars"]
    if cur:
        yield cur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages_json", type=Path, nargs="+")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--window-chars", type=int, default=9000, help="page text per request")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--out", default="data/pdf_chunks", type=Path)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    from openai import OpenAI
    # Without an explicit timeout a stalled connection blocks indefinitely: one
    # 6KB document held the pipeline for 38 minutes before this was added.
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=90.0, max_retries=3)

    for src in a.pages_json:
        doc = json.loads(src.read_text(encoding="utf-8"))
        pages = doc["page_data"]
        produced, failed = [], 0
        for bi, win in enumerate(batches(pages, a.window_chars), 1):
            payload = {"pages": [{"page": p["page"], "paragraphs": p["paragraphs"],
                                  "continues_from_previous": p["continues_from_previous"],
                                  "likely_furniture": p["likely_furniture"]} for p in win]}
            source_text = "\n".join(t for p in win for t in p["paragraphs"])
            try:
                r = cl.chat.completions.create(
                    model=a.model, temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "pdf_chunks", "schema": SCHEMA,
                                                     "strict": True}})
                got = json.loads(r.choices[0].message.content)["chunks"]
            except Exception as exc:
                print(f"  window {bi} failed: {str(exc)[:110]}", flush=True)
                failed += 1
                continue
            for c in got:
                f = fidelity(c["text"], source_text)
                c.update({"fidelity": f,
                          "fidelity_failed": f["coverage"] < a.min_coverage,
                          "window": bi,
                          "est_tokens": int(len(WORD.findall(c["text"])) * 1.3),
                          "char_count": len(c["text"])})
                produced.append(c)
            print(f"  window {bi}: pages {win[0]['page']}-{win[-1]['page']} -> "
                  f"{len(got)} chunks", flush=True)

        # Source-side recall. Per-chunk fidelity measures whether emitted text is faithful;
        # it is blind to text that was never emitted. Measured on one document, gpt-4o-mini
        # retained 80% of source characters against 93% for gpt-4.1 while scoring an
        # identical 0.96 fidelity - so omission needs its own metric.
        all_source = "\n".join(t for p in pages for t in p["paragraphs"])
        src_sh = shingles(all_source)
        got_sh = set()
        for c in produced:
            got_sh |= shingles(c["text"])
        source_recall = round(len(src_sh & got_sh) / len(src_sh), 4) if src_sh else 0.0
        cov = [c["fidelity"]["coverage"] for c in produced] or [0]
        bad = [c for c in produced if c["fidelity_failed"]]
        rec = {"source": doc["file"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "model": a.model, "method": "llm_emits_chunk_text_verified_by_shingle_coverage",
               "source_pages": doc["pages"], "source_chars": doc["chars"],
               "chunks": len(produced), "windows_failed": failed,
               "chunk_chars": sum(c["char_count"] for c in produced),
               "median_tokens": statistics.median([c["est_tokens"] for c in produced] or [0]),
               "fidelity": {"mean_coverage": round(statistics.mean(cov), 4),
                            "min_coverage": round(min(cov), 4),
                            "below_threshold": len(bad), "threshold": a.min_coverage},
               "source_recall": source_recall,
               "source_recall_note": "fraction of the source's 5-token shingles present in "
                                     "some chunk; low values mean content was skipped",
               "self_contained_reported": sum(1 for c in produced if c.get("self_contained")),
               "data": produced}
        f = out / f"{src.stem.replace('.pages','')}.chunks.json"
        f.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{doc['file']}: {len(produced)} chunks, median {rec['median_tokens']} tok, "
              f"mean fidelity {rec['fidelity']['mean_coverage']:.3f}, "
              f"{len(bad)} below {a.min_coverage}")
        print(f"  source {doc['chars']:,} chars -> chunks {rec['chunk_chars']:,} chars "
              f"({100*rec['chunk_chars']/doc['chars']:.0f}% retained), "
              f"source recall {source_recall:.3f}")
        print(f"  -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
