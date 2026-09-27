#!/usr/bin/env python3
"""Chunk quality evaluation for the semantic search corpus.

The chunking claim under test is that an LLM choosing boundaries over immutable blocks
produces retrievable, semantically coherent units - better than fixed-size splitting.
That claim needs measurement, not assertion, so this tool has two tiers:

TIER 1  structural (deterministic, free, runs over every chunk)
    Detects boundary defects that are objectively checkable without judging meaning:
    mid-sentence starts/ends, list stems severed from their lists, orphaned list items,
    heading-only chunks, dangling legal connectors ("unless", "provided that"), size
    outliers, metadata gaps, unfaithful summaries (lexical grounding), duplicate text.

TIER 2  LLM judge (sampled, costed, optional)
    Scores a stratified sample against a rubric: boundary correctness, coherence,
    self-containedness, title/summary faithfulness, retrieval usefulness.

Both write versioned artifacts so results are reproducible and comparable across runs.

Why lexical grounding rather than "does the summary look right": retrieval_summary is
LLM-generated metadata, never evidence. A summary asserting content absent from the
chunk is a fabrication risk in the retrieval layer, and content-word containment
detects that cheaply and deterministically.

Usage
-----
    python evaluate_chunk_quality.py structural --corpus-dir data/search_corpus
    python evaluate_chunk_quality.py judge --corpus-dir data/search_corpus \
        --model gpt-4o-mini --sample 120
    python evaluate_chunk_quality.py compare --corpus-dir data/search_corpus \
        --other-corpus-dir data/search_corpus_full
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATOR_VERSION = "1.0.0"
JUDGE_PROMPT_VERSION = "chunk_quality_rubric_v1"

# Preferred band from the chunking prompt: roughly 200-800 tokens, up to ~1200 acceptable.
TARGET_MIN_TOKENS = 200
TARGET_MAX_TOKENS = 800
HARD_MAX_TOKENS = 1200

PLACEHOLDER_RE = re.compile(r"\[\[LINK_\d{4,}\]\]")
SENTENCE_END = tuple(".!?\"')]”’")
# Legal connectives that must not be the last thing in a chunk: the rule they govern
# lives in the next block, so a split here severs a condition from its consequence.
# Only trailing connectives that genuinely signal an unfinished enumeration or
# condition. A chunk ending "... in section 56, or" has been cut mid-list.
DANGLING_LEGAL_CONNECTIVE_RE = re.compile(
    r"(,\s*(or|and)|\b(unless|provided that|subject to|except where|if))\s*$", re.I
)
# Deliberately EXCLUDES "12. " numbered paragraphs: in GOV.UK guidance those are
# numbered paragraph headings and a perfectly good place to start a chunk. Only
# bracketed/lettered/bulleted markers indicate an item whose stem may be elsewhere.
LIST_START = re.compile(r"^\s*(\(\s*[a-z0-9ivx]{1,4}\s*\)|[-•*]\s|[a-z]\)\s)", re.I)
STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","with","by","is","are","be","as","at","that",
    "this","it","from","its","which","may","must","shall","not","any","all","such","under","where",
    "if","when","has","have","was","were","will","can","should","would","there","their","they","been",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def est_tokens(text: str) -> int:
    """Cheap, stable token proxy. Absolute accuracy is unnecessary; comparability is not."""
    return max(1, len(text) // 4)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower()) if w not in STOPWORDS}


def grounding(summary: str, text: str) -> float:
    """Fraction of the summary's content words that actually occur in the chunk text."""
    s = content_words(summary)
    if not s:
        return 0.0
    return len(s & content_words(text)) / len(s)


# --------------------------------------------------------------------------- tier 1
def assess_chunk(chunk: dict[str, Any], next_chunk: dict[str, Any] | None) -> dict[str, Any]:
    """Score one chunk.

    Two severities, deliberately separated:

    DEFECTS are objectively wrong for legal retrieval - a rule severed from its
    enumeration, a list stem parted from its list, an empty or duplicated unit.

    ADVISORIES are descriptive signals that are NOT automatically faults. A 150-token
    chunk is undersized against the prompt's band but is often exactly right for a short
    statutory provision, and counting it as a defect would penalise precisely the
    behaviour we want on legislation. Reporting them apart keeps the headline number
    honest.

    Sentence truncation is judged ACROSS the boundary, not from punctuation alone:
    bullet lists legitimately end without a full stop, so the reliable signal is that
    this chunk ends unterminated AND the next chunk resumes in lower case.
    """
    text = (chunk.get("text") or "").strip()
    defects: list[str] = []
    advisories: list[str] = []
    tokens = est_tokens(text)

    if not text:
        defects.append("EMPTY_TEXT")

    if text:
        tail = text.rstrip()
        head_next = (next_chunk.get("text") or "").strip() if next_chunk else ""

        # Severed sentence: unterminated here AND resumed lower-case next.
        if not tail.endswith(SENTENCE_END) and not tail.endswith((";", ",")):
            if head_next[:1].islower() and not LIST_START.match(head_next):
                defects.append("SEVERED_SENTENCE")
            else:
                advisories.append("UNTERMINATED_TAIL")

        # A trailing "or"/"and"/"unless" means the enumeration or condition continues.
        if DANGLING_LEGAL_CONNECTIVE_RE.search(tail[-40:]):
            defects.append("SEVERED_ENUMERATION")

        # A chunk ending on a colon has been parted from the list it introduces.
        if tail.endswith(":"):
            defects.append("SEVERED_LIST_STEM")

        # A chunk opening on a list marker whose stem sits in the previous chunk.
        if LIST_START.match(text) and chunk.get("chunk_ordinal", 1) > 1:
            defects.append("ORPHAN_LIST_ITEM")

        if tokens < 15:
            defects.append("TOO_SMALL_TO_RETRIEVE")
        elif tokens < TARGET_MIN_TOKENS:
            advisories.append("BELOW_PREFERRED_BAND")
        if tokens > HARD_MAX_TOKENS:
            advisories.append("ABOVE_PREFERRED_BAND")

    title = (chunk.get("retrieval_title") or "").strip()
    summary = (chunk.get("retrieval_summary") or "").strip()
    if not title:
        defects.append("NO_TITLE")
    if not summary:
        defects.append("NO_SUMMARY")
    if not (chunk.get("topics") or []):
        advisories.append("NO_TOPICS")
    if not (chunk.get("legal_concepts") or []):
        advisories.append("NO_LEGAL_CONCEPTS")

    # Grounding is advisory: summaries legitimately paraphrase, so low overlap flags a
    # candidate for the LLM judge rather than proving fabrication.
    ground = grounding(summary, text) if summary and text else 0.0
    if summary and text and ground < 0.15:
        advisories.append("SUMMARY_LOW_GROUNDING")

    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "segment_id": chunk.get("segment_id"),
        "source_kind": chunk.get("source_kind"),
        "authority_class": chunk.get("authority_class"),
        "legal_regime": chunk.get("legal_regime"),
        "chunking_method": chunk.get("chunking_method"),
        "chars": len(text),
        "est_tokens": tokens,
        "in_target_band": TARGET_MIN_TOKENS <= tokens <= TARGET_MAX_TOKENS,
        "summary_grounding": round(ground, 3),
        "link_count": len(chunk.get("link_placeholders") or []) + len(chunk.get("links") or []),
        "defects": defects,
        "advisories": advisories,
        "defect_count": len(defects),
    }


def structural(corpus_dir: Path, out_prefix: str) -> dict[str, Any]:
    chunks = read_jsonl(corpus_dir / "chunks.jsonl")
    if not chunks:
        raise SystemExit(f"no chunks.jsonl under {corpus_dir}")

    by_segment: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in chunks:
        by_segment[c["segment_id"]].append(c)
    for v in by_segment.values():
        v.sort(key=lambda c: c["chunk_ordinal"])

    rows = []
    for cs in by_segment.values():
        for i, c in enumerate(cs):
            rows.append(assess_chunk(c, cs[i + 1] if i + 1 < len(cs) else None))

    # duplicate evidence text wastes context and inflates apparent coverage
    norm = collections.Counter(re.sub(r"\s+", " ", (c.get("text") or "")).strip().lower() for c in chunks)
    dupes = {t: n for t, n in norm.items() if n > 1 and t}
    dup_chunks = sum(n for n in dupes.values()) - len(dupes)

    tokens = [r["est_tokens"] for r in rows]
    flag_counts = collections.Counter(f for r in rows for f in r["defects"])
    adv_counts = collections.Counter(a for r in rows for a in r["advisories"])
    clean = sum(1 for r in rows if not r["defects"])
    grounded = [r["summary_grounding"] for r in rows if r["summary_grounding"] > 0]

    by_method = collections.defaultdict(lambda: {"n": 0, "defects": 0, "in_band": 0})
    for r in rows:
        m = r.get("chunking_method") or "UNKNOWN"
        by_method[m]["n"] += 1
        by_method[m]["defects"] += r["defect_count"]
        by_method[m]["in_band"] += 1 if r["in_target_band"] else 0

    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "generated_at": now_iso(),
        "corpus_dir": str(corpus_dir),
        "chunks": len(rows),
        "size": {
            "mean_tokens": round(statistics.mean(tokens), 1),
            "median_tokens": statistics.median(tokens),
            "p10": sorted(tokens)[len(tokens) // 10],
            "p90": sorted(tokens)[int(len(tokens) * 0.9)],
            "max_tokens": max(tokens),
            "in_target_band_pct": round(100 * sum(r["in_target_band"] for r in rows) / len(rows), 1),
            "oversized_pct": round(100 * flag_counts["OVERSIZED"] / len(rows), 1),
            "undersized_pct": round(100 * flag_counts["UNDERSIZED"] / len(rows), 1),
        },
        "boundary_defects": {
            k: flag_counts[k]
            for k in ("SEVERED_SENTENCE","SEVERED_ENUMERATION","SEVERED_LIST_STEM",
                      "ORPHAN_LIST_ITEM","TOO_SMALL_TO_RETRIEVE","EMPTY_TEXT")
        },
        "metadata_defects": {
            k: flag_counts[k] for k in ("NO_TITLE", "NO_SUMMARY")
        },
        "summary_grounding": {
            "mean": round(statistics.mean(grounded), 3) if grounded else None,
            "median": round(statistics.median(grounded), 3) if grounded else None,
            "below_0_15": sum(1 for g in grounded if g < 0.15),
        },
        "duplicates": {"duplicate_text_groups": len(dupes), "redundant_chunks": dup_chunks},
        "clean_chunks": clean,
        "clean_pct": round(100 * clean / len(rows), 1),
        "defects_per_chunk": round(sum(r["defect_count"] for r in rows) / len(rows), 2),
        "by_chunking_method": {
            m: {
                "chunks": v["n"],
                "defects_per_chunk": round(v["defects"] / v["n"], 2),
                "in_target_band_pct": round(100 * v["in_band"] / v["n"], 1),
            }
            for m, v in by_method.items()
        },
        "defect_totals": dict(flag_counts.most_common()),
        "advisory_totals": dict(adv_counts.most_common()),
    }

    (corpus_dir / f"{out_prefix}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (corpus_dir / f"{out_prefix}_per_chunk.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return report


# --------------------------------------------------------------------------- tier 2
JUDGE_SYSTEM = """You audit chunk quality for a UK public-procurement legal retrieval system.
You are given ONE chunk of exact source text plus the retrieval metadata generated for it.
Judge only what is present. Do not rewrite the text.

Score each dimension 1-5 (5 best):
- boundary_correctness: does the chunk begin and end at a defensible semantic boundary,
  keeping a legal rule with its conditions, exceptions and governed lists?
- semantic_coherence: is it about one coherent thing?
- self_containedness: could a reader use this as evidence without the neighbouring chunks?
- title_accuracy: does retrieval_title describe THIS text?
- summary_faithfulness: is retrieval_summary fully supported by THIS text, inventing nothing?
- retrieval_usefulness: would this chunk be a good search result for a real procurement question?

Also list concrete issues, and answer whether the chunk was split at a harmful point."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "boundary_correctness": {"type": "integer"},
        "semantic_coherence": {"type": "integer"},
        "self_containedness": {"type": "integer"},
        "title_accuracy": {"type": "integer"},
        "summary_faithfulness": {"type": "integer"},
        "retrieval_usefulness": {"type": "integer"},
        "harmful_split": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "boundary_correctness", "semantic_coherence", "self_containedness",
        "title_accuracy", "summary_faithfulness", "retrieval_usefulness",
        "harmful_split", "issues", "rationale",
    ],
    "additionalProperties": False,
}

DIMENSIONS = [
    "boundary_correctness", "semantic_coherence", "self_containedness",
    "title_accuracy", "summary_faithfulness", "retrieval_usefulness",
]


def stratified_sample(chunks: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Sample across source kind and authority so no stratum dominates the estimate."""
    import random

    rng = random.Random(seed)
    strata: dict[tuple, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in chunks:
        if not (c.get("text") or "").strip():
            continue
        strata[(c.get("source_kind"), c.get("authority_class"))].append(c)
    keys = sorted(strata, key=lambda k: str(k))
    out: list[dict[str, Any]] = []
    per = max(1, n // max(1, len(keys)))
    for k in keys:
        pool = strata[k]
        rng.shuffle(pool)
        out.extend(pool[:per])
    rng.shuffle(out)
    return out[:n]


def judge(corpus_dir: Path, model: str, n: int, seed: int, pause: float) -> dict[str, Any]:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    chunks = read_jsonl(corpus_dir / "chunks.jsonl")
    sample = stratified_sample(chunks, n, seed)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out_path = corpus_dir / "chunk_quality_judge.jsonl"
    done = {r["chunk_id"] for r in read_jsonl(out_path)}

    rows = read_jsonl(out_path)
    for i, c in enumerate(sample, 1):
        if c["chunk_id"] in done:
            continue
        payload = {
            "retrieval_title": c.get("retrieval_title"),
            "retrieval_summary": c.get("retrieval_summary"),
            "topics": c.get("topics"),
            "legal_concepts": c.get("legal_concepts"),
            "citation": c.get("citation"),
            "authority_class": c.get("authority_class"),
            "text": (c.get("text") or "")[:12000],
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "chunk_quality", "schema": JUDGE_SCHEMA, "strict": True},
                },
            )
            verdict = json.loads(resp.choices[0].message.content)
            verdict.update(
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c.get("document_id"),
                    "source_kind": c.get("source_kind"),
                    "authority_class": c.get("authority_class"),
                    "chunking_method": c.get("chunking_method"),
                    "judge_model": model,
                    "prompt_version": JUDGE_PROMPT_VERSION,
                    "judged_at": now_iso(),
                }
            )
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(verdict, ensure_ascii=False) + "\n")
            rows.append(verdict)
            print(f"[{i}/{len(sample)}] {c['chunk_id'][:60]} boundary={verdict['boundary_correctness']}")
        except Exception as e:  # keep going; a single bad call must not lose the run
            print(f"[{i}/{len(sample)}] ERROR {c['chunk_id'][:50]}: {e}", file=sys.stderr)
        time.sleep(pause)

    report = summarize_judge(rows, model, n, seed)
    (corpus_dir / "chunk_quality_judge_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def summarize_judge(rows: list[dict[str, Any]], model: str, n: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"judged": 0}
    def mean(key: str, subset=None) -> float:
        vals = [r[key] for r in (subset or rows) if isinstance(r.get(key), int)]
        return round(statistics.mean(vals), 2) if vals else 0.0

    by_kind: dict[str, dict[str, Any]] = {}
    for kind in sorted({r.get("source_kind") for r in rows if r.get("source_kind")}):
        sub = [r for r in rows if r.get("source_kind") == kind]
        by_kind[kind] = {"n": len(sub), **{d: mean(d, sub) for d in DIMENSIONS}}

    issues = collections.Counter(i.strip().lower()[:80] for r in rows for i in (r.get("issues") or []))
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "judge_model": model,
        "sample_requested": n,
        "judged": len(rows),
        "seed": seed,
        "means": {d: mean(d) for d in DIMENSIONS},
        "overall_mean": round(statistics.mean([mean(d) for d in DIMENSIONS]), 2),
        "harmful_split_pct": round(100 * sum(1 for r in rows if r.get("harmful_split")) / len(rows), 1),
        "by_source_kind": by_kind,
        "top_issues": issues.most_common(15),
    }


def compare(a: Path, b: Path) -> dict[str, Any]:
    ra = json.loads((a / "chunk_quality_report.json").read_text())
    rb = json.loads((b / "chunk_quality_report.json").read_text())
    keys = ["clean_pct", "defects_per_chunk"]
    return {
        "a": {"corpus": str(a), **{k: ra.get(k) for k in keys}, "size": ra.get("size")},
        "b": {"corpus": str(b), **{k: rb.get(k) for k in keys}, "size": rb.get("size")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["structural", "judge", "compare"])
    ap.add_argument("--corpus-dir", default="data/search_corpus", type=Path)
    ap.add_argument("--other-corpus-dir", type=Path)
    ap.add_argument("--out-prefix", default="chunk_quality")
    ap.add_argument("--model", default=os.getenv("CHUNK_JUDGE_MODEL", "gpt-4o-mini"))
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--pause", type=float, default=0.15)
    args = ap.parse_args()

    if args.stage == "structural":
        print(json.dumps(structural(args.corpus_dir, args.out_prefix), indent=2, ensure_ascii=False))
    elif args.stage == "judge":
        print(json.dumps(judge(args.corpus_dir, args.model, args.sample, args.seed, args.pause), indent=2))
    else:
        if not args.other_corpus_dir:
            raise SystemExit("compare requires --other-corpus-dir")
        print(json.dumps(compare(args.corpus_dir, args.other_corpus_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
