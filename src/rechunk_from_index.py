#!/usr/bin/env python3
"""Re-chunk material still on the original boundary-selection lane, using text emission.

Why these sources
-----------------
The old lane holds 22.6% of active chunks but supplies 64.6% of the results that outrank a
gold provision - 2.9x over-represented in blocking. It is also the only lane still emitting
chunks above 1,000 tokens (16% of it), and a large chunk matches many queries weakly, which
is how it comes to occupy ranks two through eight.

    gov.uk              318 docs, 1,371 chunks, median 487 tok, only 37% in the 50-400 band
    legislation.gov.uk    3 docs,   796 chunks, median 171 tok, 61% in band

Source text is reconstructed by concatenating each document's existing chunks in order, so
nothing needs re-fetching; the text is already exact.

Trade-off on legislation, stated because it is real: the three core instruments were
boundary-chunked, which reconstructs text from immutable blocks and guarantees the corpus
contains exactly what the statute contains. Text emission replaces that guarantee with a
measurement (fidelity + source recall). `parent_node_id` is preserved by carrying the
provision number through, so legal identity and graph participation survive.
"""
from __future__ import annotations

import argparse, collections, json, os, re, sqlite3, statistics, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"\w+")
PROV = re.compile(r"(?m)^\s*(?:(Section|Regulation|Article|Schedule)\s+)?(\d+[A-Z]?)\s+(?=[A-Z(])")

SYSTEM_LEG = open(ROOT / "chunk_legislation_text.py", encoding="utf-8").read()
SYSTEM_LEG = SYSTEM_LEG[SYSTEM_LEG.index('SYSTEM = """') + 12:SYSTEM_LEG.index('"""\n\nSCHEMA')]

SYSTEM_GUIDANCE = """You are segmenting UK public procurement guidance into retrieval chunks.

You receive the text of one document. It may carry extraction artifacts: headings run into body text, lists flattened, navigation furniture left in.

Produce chunks that would work as evidence returned by a search engine.

Rules:
1. Emit the chunk TEXT, reproducing the source wording exactly. You may repair extraction artifacts - restore a break, separate a heading from body text. Do NOT paraphrase, summarise or add anything.
2. Each chunk must stand alone: a complete rule, procedure, definition or explanation. Keep an introductory stem with the list it governs. Never end on "the following—" or begin with a bare list item.
3. Prefer 100-400 words. A complete 60-word rule beats a padded 300-word one.
4. Cover the supplied text completely. Omit only navigation, contents listings, copyright notices and "seek legal advice" footers.
5. Give each chunk a title naming what it states.

Return JSON only."""

SCHEMA = {"type": "object", "properties": {"chunks": {"type": "array", "items": {
    "type": "object", "properties": {
        "title": {"type": "string"}, "text": {"type": "string"},
        "provision": {"type": "string"}, "self_contained": {"type": "boolean"}},
    "required": ["title", "text", "provision", "self_contained"],
    "additionalProperties": False}}}, "required": ["chunks"], "additionalProperties": False}


def shingles(t, n=5):
    w = [x.lower() for x in WORD.findall(t or "")]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def windows(text, target, hard_max=None):
    hard_max = hard_max or int(target * 1.6)
    marks = sorted({m.start() for m in PROV.finditer(text)} | {0, len(text)})
    out, start = [], 0
    while start < len(text):
        limit = start + hard_max
        nxt = [m for m in marks if start < m <= limit]
        if nxt:
            end = max([m for m in nxt if m - start >= target] or [nxt[-1]])
        else:
            seg = text[start:limit]; br = seg.rfind("\n\n")
            end = start + (br if br > target // 2 else len(seg))
        end = min(max(end, start + 1), len(text))
        if text[start:end].strip():
            out.append(text[start:end])
        start = end
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="state/chunk_index_merged.sqlite3", type=Path)
    ap.add_argument("--like", required=True, help="source_url LIKE pattern")
    ap.add_argument("--kind", choices=["legislation", "guidance"], required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--window-chars", type=int, default=9000)
    ap.add_argument("--min-coverage", type=float, default=0.80)
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / a.db); con.row_factory = sqlite3.Row
    from openai import OpenAI
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300.0, max_retries=4)
    SYSTEM = SYSTEM_LEG if a.kind == "legislation" else SYSTEM_GUIDANCE

    docs = [r[0] for r in con.execute(
        "SELECT DISTINCT document_id FROM chunks WHERE chunking_method='LLM_SEMANTIC_BOUNDARY_V1' "
        "AND superseded_by IS NULL AND filtered_out IS NULL AND source_url LIKE ?", (a.like,))]
    print(f"documents to re-chunk: {len(docs)}", flush=True)
    for di, did in enumerate(docs, 1):
        dest = out / f"{did}.chunks.json"
        if dest.exists():
            continue
        parts = [dict(r) for r in con.execute(
            "SELECT text, citation, chunk_ordinal FROM chunks WHERE document_id=? "
            "AND chunking_method='LLM_SEMANTIC_BOUNDARY_V1' AND superseded_by IS NULL "
            "AND filtered_out IS NULL ORDER BY chunk_ordinal", (did,))]
        if not parts:
            continue
        text = "\n\n".join(p["text"] for p in parts if p["text"])
        wins = windows(text, a.window_chars)
        produced, failed = [], []
        for wi, w in enumerate(wins, 1):
            try:
                r = cl.chat.completions.create(
                    model=a.model, temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(
                                  {"document_id": did, "text": w}, ensure_ascii=False)}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "rechunk", "schema": SCHEMA, "strict": True}})
                got = json.loads(r.choices[0].message.content)["chunks"]
            except Exception as exc:
                failed.append(wi)
                print(f"  {did[:14]} window {wi} failed: {str(exc)[:80]}", flush=True)
                continue
            ws = shingles(w)
            for c in got:
                cs = shingles(c["text"])
                cov = len(cs & ws) / len(cs) if cs else 0.0
                num = re.sub(r"[^0-9A-Za-z]", "", c.get("provision") or "")
                c.update({"fidelity_coverage": round(cov, 4),
                          "fidelity_failed": cov < a.min_coverage, "window": wi,
                          "parent_node_id": (f"{did}__section-{num}"
                                             if a.kind == "legislation" and num else None),
                          "est_tokens": int(len(WORD.findall(c["text"])) * 1.3),
                          "char_count": len(c["text"])})
                produced.append(c)
        ss = shingles(text); gs = set()
        for c in produced: gs |= shingles(c["text"])
        rec = {"document_id": did, "citation": parts[0]["citation"], "kind": a.kind,
               "generated_at": datetime.now(timezone.utc).isoformat(), "model": a.model,
               "source_chars": len(text), "windows": len(wins), "chunks": len(produced),
               "chunk_chars": sum(c["char_count"] for c in produced),
               "median_tokens": statistics.median([c["est_tokens"] for c in produced] or [0]),
               "source_recall": round(len(ss & gs) / len(ss), 4) if ss else 0.0,
               "mean_fidelity": round(statistics.mean(
                   [c["fidelity_coverage"] for c in produced] or [0]), 4),
               "failed_windows": failed, "complete": not failed, "data": produced}
        dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{di}/{len(docs)}] {did[:16]} {len(produced):5d} ch  "
              f"med {rec['median_tokens']:.0f} tok  recall {rec['source_recall']:.3f}  "
              f"fid {rec['mean_fidelity']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
