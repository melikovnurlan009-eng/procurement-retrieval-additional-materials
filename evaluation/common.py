"""Shared loading, scoring-recompute, and metric functions for every analysis script.

Reuses chunk_retrieval.ChunkRetriever's OWN _normalize/_fuse/_authority_norm methods for
every weight sweep (rather than reimplementing the maths), by instantiating one retriever
and temporarily overriding its BM25_WEIGHT/DENSE_WEIGHT/AUTH_BLEND class-level defaults on
that instance only - chunk_retrieval.py's own module-level defaults are never mutated, so
running an alpha/beta sweep here has zero effect on anything else that imports the module
(e.g. a concurrently running production query).
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from evaluation.config import CODE_DIR, LEGISLATION_CLASSES, CHUNK_LENGTH_BINS, CHUNK_LENGTH_BIN_LABELS

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import chunk_retrieval as _cr  # noqa: E402  (the ONE active retriever module, verified in config.py)
from chunk_retrieval import ChunkRetriever  # noqa: E402

ACTIVE_RETRIEVER_MODULE_FILE = _cr.__file__  # asserted against in sanity_checks.assert_active_retriever


# ---------------------------------------------------------------------------------- I/O
def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_scenarios(path: Path, split: str = "ALL") -> list[dict]:
    rows = load_jsonl(path)
    if split == "ALL":
        return rows
    want = split.lower()
    return [r for r in rows if str(r.get("split", "")).lower() == want]


def load_gold(path: Path) -> dict[str, dict]:
    return {r["scenario_id"]: r for r in load_jsonl(path)}


def load_qrels(path: Path) -> dict[tuple[str, str], dict]:
    """(scenario_id, chunk_id) -> judgment row (relevance_grade, is_binding, is_currently_applicable, ...)."""
    if not Path(path).exists():
        return {}
    out = {}
    for r in load_jsonl(path):
        out[(r["scenario_id"], r["chunk_id"])] = r
    return out


def load_ce_output(path: Path) -> dict[str, dict]:
    """scenario_id -> {"legislation_lane_top75": [...], "other_lane_top75": [...]}"""
    if not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text())
    return {s["scenario_id"]: s for s in raw.get("scenarios", [])}


# ---------------------------------------------------------------------------------- gold targets
ACCEPTED_STATUSES = ("MATCHED", "FUZZY_MATCHED")


def essential_targets(gold_record: dict) -> dict[str, set[str]]:
    """requirement_id -> set of essential-evidence chunk_ids (status MATCHED/FUZZY_MATCHED only,
    matching the shipped scorer's own acceptance set)."""
    out = {}
    for req in gold_record.get("requirements", []):
        ids = set()
        for it in req.get("essential_evidence", []):
            res = it.get("resolution") or {}
            if res.get("status") in ACCEPTED_STATUSES and res.get("chunk_id"):
                ids.add(res["chunk_id"])
        if ids:
            out[req["requirement_id"]] = ids
    return out


def all_gold_chunk_ids(gold_record: dict, buckets=("essential_evidence", "strong_supporting_evidence")) -> set[str]:
    out = set()
    for req in gold_record.get("requirements", []):
        for bucket in buckets:
            for it in req.get(bucket, []) or []:
                res = it.get("resolution") or {}
                if res.get("status") in ACCEPTED_STATUSES and res.get("chunk_id"):
                    out.add(res["chunk_id"])
    return out


def wrong_regime_chunk_ids(gold_record: dict) -> set[str]:
    """Bug fixed 2026-09-20: items moved into wrong_regime_evidence during the gold audit
    (e.g. TEST019's PCR2015 reg.87) often carry their chunk_id under `acceptable_chunk_ids`
    (added by a later mechanical re-resolution pass) rather than `resolution.chunk_id` (which
    only ever existed on items that were originally essential_evidence/strong_supporting -
    relocated acceptable_alternatives items were never given one). Checking only
    `resolution.chunk_id` silently returned zero wrong-regime chunks for every scenario, even
    ones with real, populated wrong_regime_evidence - caught when wrong_regime_analysis.py
    reported 0 scenarios with any wrong-regime evidence despite TEST019/DEV007/DEV011
    visibly having it. Now checks both fields."""
    out = set()
    for req in gold_record.get("requirements", []):
        for it in req.get("wrong_regime_evidence", []) or []:
            res = it.get("resolution") or {}
            if res.get("chunk_id"):
                out.add(res["chunk_id"])
            for cid in it.get("acceptable_chunk_ids", []) or []:
                out.add(cid)
    return out


# ---------------------------------------------------------------------------------- category labels
# Original benchmark "suite" values are NEVER altered on disk - this is an output-only
# rename for readability in tables/figures, per the harness spec's explicit instruction.
CATEGORY_DISPLAY_NAMES = {
    "exact_anchor": "direct legal anchor",
    "semantic": "semantic / practitioner phrasing",
    "vocabulary_mismatch": "vocabulary mismatch",
    "graph_multi_hop": "cross-reference / multi-instrument",
    "applicability": "applicability / transition",
    "practical_guidance": "practical guidance / multi-evidence",
    "authority": "authority / source-role sensitive",
    "compound": "compound / multi-requirement",
}


def category_of(scenario: dict) -> str:
    return scenario.get("suite", "unknown")


def display_category(scenario: dict) -> str:
    return CATEGORY_DISPLAY_NAMES.get(category_of(scenario), category_of(scenario))


# ---------------------------------------------------------------------------------- chunk length
def length_bin(token_count: int | None) -> str:
    if token_count is None:
        return "unknown"
    for (lo, hi), label in zip(CHUNK_LENGTH_BINS, CHUNK_LENGTH_BIN_LABELS):
        if lo <= token_count <= hi:
            return label
    return "unknown"


# ---------------------------------------------------------------------------------- rescoring
@contextmanager
def weighted_retriever(beta: float = _cr.ChunkRetriever.BM25_WEIGHT,
                        alpha: float = _cr.ChunkRetriever.AUTH_BLEND,
                        db_path: Path | None = None, collection: str | None = None):
    """A ChunkRetriever instance with BM25_WEIGHT/DENSE_WEIGHT/AUTH_BLEND overridden on the
    INSTANCE only (Python resolves instance attrs before class attrs), so
    chunk_retrieval.ChunkRetriever's own class-level defaults - and therefore every other
    caller of the module - are never mutated. Used to re-drive _fuse()/_authority_norm()
    over cached raw scores at arbitrary alpha/beta without re-embedding or re-querying FTS5.
    """
    r = ChunkRetriever(db_path or _cr.DEFAULT_DB, collection=collection or _cr.DEFAULT_COLLECTION)
    r.BM25_WEIGHT = beta
    r.DENSE_WEIGHT = 1.0 - beta
    r.AUTH_BLEND = alpha
    try:
        yield r
    finally:
        pass  # instance is discarded; class defaults were never touched


def refuse_pool(retriever: ChunkRetriever, bm25_raw: dict[str, float], dense_raw: dict[str, float],
                 pool: set[str], meta: dict[str, dict], jurisdiction_weights: dict | None = None
                 ) -> dict[str, dict[str, float]]:
    """Recompute fused/authority/final scores for one lane's pool at the retriever's current
    alpha/beta, using chunk_retrieval's OWN _fuse/_authority_norm (not reimplemented here).
    Returns chunk_id -> {bm25_norm, dense_norm, authority_norm, fused, final_score}.
    """
    fused, bm25_n, dense_n = retriever._fuse(bm25_raw, dense_raw, pool)  # noqa: SLF001 (intentional reuse)
    authority_n = retriever._authority_norm(pool, meta)  # noqa: SLF001
    jw = jurisdiction_weights or _cr.JURISDICTION_WEIGHTS
    out = {}
    for cid in pool:
        m = meta.get(cid) or {}
        juris_w = jw.get(m.get("jurisdiction"), _cr.DEFAULT_JURISDICTION_WEIGHT)
        blended = fused.get(cid, 0.0) * (1 - retriever.AUTH_BLEND) + authority_n.get(cid, 0.5) * retriever.AUTH_BLEND
        out[cid] = {
            "bm25_norm": bm25_n.get(cid, 0.0), "dense_norm": dense_n.get(cid, 0.0),
            "authority_norm": authority_n.get(cid, 0.5), "fused_score": fused.get(cid, 0.0),
            "final_score": blended * juris_w, "jurisdiction_weight": juris_w,
        }
    return out


# ---------------------------------------------------------------------------------- metrics
def requirement_recall_at_k(ranked_leg: list[str], ranked_oth: list[str], targets: dict[str, set[str]],
                             k_leg: int, k_oth: int) -> float:
    """Fraction of requirements satisfied: target-chunk present in leg[:k_leg] OR oth[:k_oth],
    each lane sliced INDEPENDENTLY (never merged-then-sliced) - the pooling rule established
    for this project. Returns 0.0 for a scenario with no resolvable targets (caller should
    exclude such scenarios from the denominator, not silently count a 0)."""
    if not targets:
        return None
    topL, topO = set(ranked_leg[:k_leg]), set(ranked_oth[:k_oth])
    sat = sum(1 for ids in targets.values() if (ids & topL) or (ids & topO))
    return sat / len(targets)


def complete_coverage(ranked_leg: list[str], ranked_oth: list[str], targets: dict[str, set[str]],
                       k_leg: int, k_oth: int) -> float | None:
    """1.0 iff EVERY requirement is satisfied within budget, else 0.0 (a binary per-query
    outcome, used for McNemar's test in statistical_analysis.py)."""
    if not targets:
        return None
    topL, topO = set(ranked_leg[:k_leg]), set(ranked_oth[:k_oth])
    return 1.0 if all((ids & topL) or (ids & topO) for ids in targets.values()) else 0.0


def candidate_recall_at_k(ranked: list[str], gold_ids: set[str], k: int) -> float | None:
    if not gold_ids:
        return None
    top = set(ranked[:k])
    return len(top & gold_ids) / len(gold_ids)


def hit_at_k(ranked: list[str], gold_ids: set[str], k: int) -> float | None:
    if not gold_ids:
        return None
    return 1.0 if set(ranked[:k]) & gold_ids else 0.0


def mrr(ranked: list[str], gold_ids: set[str]) -> float | None:
    if not gold_ids:
        return None
    for i, cid in enumerate(ranked, 1):
        if cid in gold_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: list[str], grades: dict[str, float], k: int) -> float:
    return sum((2 ** grades.get(cid, 0.0) - 1) / math.log2(i + 1) for i, cid in enumerate(ranked[:k], 1))


def ndcg_at_k(ranked: list[str], grades: dict[str, float], k: int) -> float | None:
    ideal = sorted(grades.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))
    if idcg == 0:
        return None
    return dcg_at_k(ranked, grades, k) / idcg


def global_requirement_recall_at_k(ranked_pooled: list[str], targets: dict[str, set[str]], k: int) -> float | None:
    """Same acceptance rule as requirement_recall_at_k, but for a CONVENTIONAL (non-two-lane)
    baseline: one globally-ranked list pooled across both lanes, top-k taken as a single
    budget (not k_leg/k_oth sliced independently). Used only for RQ1's 'conventional lexical/
    dense/hybrid retrieval' comparison points (configs 1-6) - the two-lane system itself is
    always scored with requirement_recall_at_k's lane-independent OR-pool rule, never this."""
    if not targets:
        return None
    top = set(ranked_pooled[:k])
    sat = sum(1 for ids in targets.values() if ids & top)
    return sat / len(targets)


def global_complete_coverage(ranked_pooled: list[str], targets: dict[str, set[str]], k: int) -> float | None:
    if not targets:
        return None
    top = set(ranked_pooled[:k])
    return 1.0 if all(ids & top for ids in targets.values()) else 0.0


# ---------------------------------------------------------------------------------- rrf
def rrf_scores(rank_lists: list[list[str]], k: int = _cr.RETRIEVAL_VERSION and 60) -> dict[str, float]:
    """Standard RRF: sum of 1/(k+rank) across the given rank lists (1-indexed). k=60 is the
    conventional constant; this repository defines no other standard value anywhere, so 60
    is used and stated explicitly (per the spec's instruction)."""
    out: dict[str, float] = defaultdict(float)
    for lst in rank_lists:
        for i, cid in enumerate(lst, 1):
            out[cid] += 1.0 / (k + i)
    return dict(out)


def assert_active_retriever():
    """Sanity check importable from any script: fails loudly if chunk_retrieval resolved to
    some other copy on sys.path instead of this repository's src/chunk_retrieval.py."""
    from evaluation.config import CODE_DIR
    assert Path(ACTIVE_RETRIEVER_MODULE_FILE).resolve().parent == CODE_DIR.resolve(), (
        f"chunk_retrieval imported from unexpected path: {ACTIVE_RETRIEVER_MODULE_FILE} "
        f"- expected {CODE_DIR / 'chunk_retrieval.py'}"
    )
    assert hasattr(ChunkRetriever, "search_two_lanes"), "active retriever missing search_two_lanes - wrong file?"
    assert ChunkRetriever.BM25_WEIGHT == 0.40 and ChunkRetriever.AUTH_BLEND == 0.30, (
        "active retriever's own default weights have changed since this harness was written "
        f"(BM25_WEIGHT={ChunkRetriever.BM25_WEIGHT}, AUTH_BLEND={ChunkRetriever.AUTH_BLEND}) - "
        "update evaluation/config.py's DEFAULT_BETA/DEFAULT_ALPHA to match before trusting results."
    )
