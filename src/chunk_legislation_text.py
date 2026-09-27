#!/usr/bin/env python3
"""Chunk acquired legislation: the model emits fully-formed chunks from provision text.

Same method as the PDF lane - the model returns chunk TEXT rather than block ids - and for
the same reason it is verified rather than trusted: per-chunk shingle fidelity catches
fabrication, source recall catches omission.

Legal identity is preserved despite text emission. Windows are cut at provision boundaries
detected in the source, and the model is asked to report which provision each chunk states,
so `parent_node_id` can be reconstructed as `{document}__section-{n}`. That is what makes a
chunk addressable by a citation and therefore a participant in the graph; without it these
instruments would be searchable but invisible to reference resolution, which is the whole
reason they were acquired.

Source selection is explicit because the stored copies are not uniform: three instruments
were re-acquired after the scraper selected an annotation feed instead of the statute, so
`full_text_reacquired.txt` is preferred where present, then `full_text_original.txt`.
"""
from __future__ import annotations

import argparse, json, os, re, statistics, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORD = re.compile(r"\w+")
# "Section 12", "Regulation 3", or a bare provision number opening a line.
PROV = re.compile(r"(?m)^\s*(?:(Section|Regulation|Article|Schedule)\s+)?(\d+[A-Z]?)\s+(?=[A-Z(])")

SYSTEM = """You are segmenting UK legislation into retrieval chunks for a legal search system.

You receive statutory text, which may carry extraction artifacts: numbering rendered inconsistently, lines joined or split oddly, headings run into body text.

Produce chunks that a search engine could return as evidence.

Rules:
1. Emit the chunk TEXT, reproducing the statutory wording exactly. You may repair extraction artifacts - restore a line break, separate a heading that has run into the text, fix spacing. You must NOT paraphrase, summarise, modernise, correct the law, or add explanation. The words of the statute are the words of the statute.
2. Keep a provision together with what it governs. An applicability clause ("This section applies where—") must stay with the rule it introduces. A stem ("the following conditions—") must stay with its list. Never end a chunk on a stem, never begin one on a bare lettered item.
3. One provision per chunk where the provision is a workable size. Split a long provision at subsection boundaries, never mid-subsection. Merge consecutive very short provisions (extent, commencement, short title) rather than emitting them alone.
4. Report the provision number each chunk states, as it appears in the text ("57", "12A", "Sch 5 para 3"). If a chunk spans several, report the first.
5. Cover the supplied text completely. Every substantive provision must appear in some chunk. Omit only running headers, page furniture and repeated citation fragments.
6. Give each chunk a title naming the rule it states.

Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"},
            "text": {"type": "string"},
            "provision": {"type": "string"},
            "self_contained": {"type": "boolean"},
        }, "required": ["title", "text", "provision", "self_contained"],
            "additionalProperties": False}}
    },
    "required": ["chunks"], "additionalProperties": False,
}


def shingles(t, n=5):
    w = [x.lower() for x in WORD.findall(t or "")]
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def source_for(d: Path) -> tuple[Path, str] | tuple[None, None]:
    for name, tag in (("full_text_reacquired.txt", "reacquired"),
                      ("full_text_original.txt", "original"),
                      ("full_text.txt", "latest")):
        p = d / name
        if p.exists() and p.stat().st_size > 1000:
            return p, tag
    return None, None


def windows(text: str, target: int, hard_max: int | None = None):
    """Cut at provision boundaries, but never exceed a hard size limit.

    The first version cut ONLY at detected provision marks, so an instrument whose numbering
    the regex did not match produced enormous windows: the Freedom of Information Act became
    4 windows of ~52,500 characters against a 9,000 target. The model's reply then exceeded
    its output limit and returned truncated JSON, which surfaced as a parse error and silently
    dropped the whole window - 86% of that Act's text was lost while fidelity still read 0.963,
    because fidelity can only measure text that was emitted.

    A provision boundary is preferred; a hard cut at a paragraph break is taken when none is
    available within the limit.
    """
    hard_max = hard_max or int(target * 1.6)
    marks = sorted({m.start() for m in PROV.finditer(text)} | {0, len(text)})
    out, start = [], 0
    while start < len(text):
        limit = start + hard_max
        nxt = [m for m in marks if start < m <= limit]
        if nxt:
            # Largest provision boundary that fits, so windows stay near the target.
            end = max([m for m in nxt if m - start >= target] or [nxt[-1]])
        else:
            # No provision boundary in range: fall back to the last paragraph break.
            seg = text[start:limit]
            br = seg.rfind("\n\n")
            end = start + (br if br > target // 2 else len(seg))
        end = min(max(end, start + 1), len(text))
        chunk = text[start:end]
        if chunk.strip():
            out.append(chunk)
        start = end
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/legislation_acquired", type=Path)
    ap.add_argument("--out", default="data/legislation_chunks", type=Path)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--window-chars", type=int, default=9000)
    ap.add_argument("--min-coverage", type=float, default=0.80)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    from openai import OpenAI
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300.0, max_retries=4)

    dirs = sorted(p for p in (ROOT / a.dir).iterdir() if p.is_dir())
    if a.only:
        keep = {x.lower() for x in a.only}
        dirs = [p for p in dirs if p.name.lower() in keep]
    for d in dirs:
        did = d.name.upper()
        dest = out / f"{did}.chunks.json"
        if dest.exists():
            print(f"{did}: already done"); continue
        src, tag = source_for(d)
        if not src:
            print(f"{did}: no usable source"); continue
        text = src.read_text(encoding="utf-8", errors="replace")
        wins = windows(text, a.window_chars)
        print(f"{did}: {len(text):,} chars ({tag}) -> {len(wins)} windows", flush=True)
        produced = []
        failed_windows = []
        for wi, w in enumerate(wins, 1):
            try:
                r = cl.chat.completions.create(
                    model=a.model, temperature=0,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(
                                  {"document_id": did, "text": w}, ensure_ascii=False)}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "legislation_chunks", "schema": SCHEMA, "strict": True}})
                got = json.loads(r.choices[0].message.content)["chunks"]
            except Exception as exc:
                print(f"   window {wi} failed: {str(exc)[:100]} - retrying once", flush=True)
                try:
                    time.sleep(10)
                    r = cl.chat.completions.create(
                        model=a.model, temperature=0,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": json.dumps(
                                      {"document_id": did, "text": w}, ensure_ascii=False)}],
                        response_format={"type": "json_schema", "json_schema": {
                            "name": "legislation_chunks", "schema": SCHEMA, "strict": True}})
                    got = json.loads(r.choices[0].message.content)["chunks"]
                except Exception as exc2:
                    failed_windows.append(wi)
                    print(f"   window {wi} FAILED AFTER RETRY: {str(exc2)[:90]}", flush=True)
                    continue
            ws = shingles(w)
            for c in got:
                cs = shingles(c["text"])
                cov = len(cs & ws) / len(cs) if cs else 0.0
                num = re.sub(r"[^0-9A-Za-z]", "", c.get("provision") or "")
                c.update({"fidelity_coverage": round(cov, 4),
                          "fidelity_failed": cov < a.min_coverage,
                          "window": wi,
                          "parent_node_id": f"{did}__section-{num}" if num else None,
                          "est_tokens": int(len(WORD.findall(c["text"])) * 1.3),
                          "char_count": len(c["text"])})
                produced.append(c)
            if wi % 10 == 0:
                print(f"   {wi}/{len(wins)} windows, {len(produced)} chunks", flush=True)
        src_sh = shingles(text)
        got_sh = set()
        for c in produced: got_sh |= shingles(c["text"])
        rec = {"document_id": did, "source_file": src.name, "source_variant": tag,
               "generated_at": datetime.now(timezone.utc).isoformat(), "model": a.model,
               "source_chars": len(text), "windows": len(wins), "chunks": len(produced),
               "chunk_chars": sum(c["char_count"] for c in produced),
               "median_tokens": statistics.median([c["est_tokens"] for c in produced] or [0]),
               "source_recall": round(len(src_sh & got_sh) / len(src_sh), 4) if src_sh else 0.0,
               "mean_fidelity": round(statistics.mean(
                   [c["fidelity_coverage"] for c in produced] or [0]), 4),
               "fidelity_failed": sum(1 for c in produced if c["fidelity_failed"]),
               "with_parent_node_id": sum(1 for c in produced if c["parent_node_id"]),
               "failed_windows": failed_windows,
               "complete": not failed_windows,
               "data": produced}
        dest.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   -> {len(produced)} chunks, median {rec['median_tokens']} tok, "
              f"recall {rec['source_recall']}, fidelity {rec['mean_fidelity']}, "
              f"node_id on {rec['with_parent_node_id']}"
              + (f"  [INCOMPLETE: {len(failed_windows)} windows lost]" if failed_windows else "")
              + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
