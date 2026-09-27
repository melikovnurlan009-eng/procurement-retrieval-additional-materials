#!/usr/bin/env python3
"""Hybrid + graph-expanded + authority-aware retrieval over the chunk corpus.

Pipeline
--------
    query -> BM25 (SQLite FTS5)      \\
             dense (BGE-M3 / Qdrant)  ) -> normalized weighted-sum fusion -> legal anchors
                                     /                                       -> bounded graph expansion
                                                                            -> authority rerank
                                                                            -> evidence bundle

Every stage is separately switchable so the thesis arms can be evaluated in isolation:

    A  dense only              --no-lexical --no-graph --no-rerank
    B  lexical only            --no-dense --no-graph --no-rerank
    C  hybrid                  --no-graph --no-rerank
    D  hybrid + rerank         --no-graph
    E  hybrid + graph          --no-rerank
    F  hybrid + graph + rerank  (default)

Design decisions that matter
----------------------------
Fusion is a normalized weighted sum, not RRF. BM25 and cosine similarity are put on a
comparable footing by z-scoring each channel's raw scores against its own candidate
pool for this query (removing the query-dependent scale) and squashing the result
through tanh into (0, 1) (bounding outliers without a hard clip). The two channels then
combine as 0.4*bm25 + 0.6*dense. Every candidate is scored on BOTH channels - a chunk
that only entered the pool via one channel gets its other score computed directly
(a real BM25/cosine value, not a placeholder), so no result is ever missing a score.

Graph expansion is bounded by hop count AND relation type AND per-hop fan-out. Legal
graphs are dense - PA2023 alone has thousands of cross-references - so unbounded
traversal returns the whole statute and destroys precision. It always excludes
PCR2015/EU targets (superseded law), since per-query legacy-intent detection proved
unreliable and was removed.

Authority is applied as an explicit multiplicative prior, never learned from the query.
Professional commentary must not outrank primary legislation because its phrasing
happens to match the question more closely.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

RETRIEVAL_VERSION = "1.0.0"
DEFAULT_DB = "state/chunk_index_merged.sqlite3"
DEFAULT_COLLECTION = "chunks__bge_m3__v1"

# Explicit, configurable, and logged with every run - never hardcoded silently.
AUTHORITY_WEIGHTS: dict[str, float] = {
    "PRIMARY_LEGISLATION": 1.00,
    "SECONDARY_LEGISLATION": 0.97,
    "OFFICIAL_TECHNICAL_GUIDANCE": 0.90,
    "OFFICIAL_GOVERNMENT_GUIDANCE": 0.88,
    "OFFICIAL_REGULATOR_GUIDANCE": 0.86,
    "PROCUREMENT_POLICY": 0.84,
    "OFFICIAL_WORKFLOW": 0.78,
    "OFFICIAL_TRAINING": 0.74,
    "PROFESSIONAL_INTERPRETATION": 0.62,
    "PROFESSIONAL_CASE_ANALYSIS": 0.62,
    "INDUSTRY_PRACTICE": 0.55,
    # Emitted by the professional-sources scraper; mapped explicitly so practitioner
    # commentary is weighted by a stated policy rather than falling through to a default.
    "NON_AUTHORITATIVE_PROFESSIONAL": 0.62,
    "OFFICIAL_SPECIALIST_GUIDANCE": 0.86,
    "OFFICIAL_PA23_TECHNICAL_GUIDANCE": 0.90,
    "OFFICIAL_PRACTICE_GUIDANCE": 0.84,
}

# Statute competes against guidance in one fused, one authority-weighted ranking, and
# structurally loses even when it is the right answer: guidance echoes a query's own
# phrasing while statute uses defined terms, so cosine and the cross-encoder both reward
# guidance's surface overlap. Measured on the 300-query anchor set: restricting the pool
# to legislation-only recovers gold at rank 1 for 26% of the queries production otherwise
# gets wrong (59/230) - not a fix for the majority, but a real, cheap one where legislation
# is competitive within its own class yet never reaches the top of the mixed pool.
LEGISLATION_CLASSES = ("PRIMARY_LEGISLATION", "SECONDARY_LEGISLATION")
DEFAULT_AUTHORITY_WEIGHT = 0.60

# Instrument aliases, expanded before matching. Legislation never cites itself, so the only
# string identifying PCR 2015 reg 72 as PCR 2015 is its `citation` field - "Public Contracts
# Regulations 2015 reg 72". A practitioner writes "PCR 2015". BM25 tokenises the abbreviation
# and the full name as unrelated terms, so the most discriminating term in the query matches
# nothing, and the abbreviation actively steers retrieval toward the 509 guidance chunks that
# DISCUSS PCR 2015 rather than the regulations themselves. Measured: with the abbreviation the
# gold provision was absent from the top 40 of both channels; with the full name it ranked 1.
#
# The alias is APPENDED rather than substituted, so the original wording still contributes and
# a query that already uses the full name is unaffected.
INSTRUMENT_ALIASES = [
    (re.compile(r"\bPCR\s?-?\s?2015\b", re.I), "Public Contracts Regulations 2015"),
    (re.compile(r"\bPA\s?-?\s?2023\b", re.I), "Procurement Act 2023"),
    (re.compile(r"\bPR\s?-?\s?2024\b", re.I), "Procurement Regulations 2024"),
    (re.compile(r"\bUCR\s?-?\s?2016\b", re.I), "Utilities Contracts Regulations 2016"),
    (re.compile(r"\bCCR\s?-?\s?2016\b", re.I), "Concession Contracts Regulations 2016"),
    (re.compile(r"\bDSPCR\s?-?\s?2011\b", re.I), "Defence and Security Public Contracts Regulations 2011"),
    (re.compile(r"\bFOIA\b", re.I), "Freedom of Information Act 2000"),
    (re.compile(r"\bDPA\s?-?\s?2018\b", re.I), "Data Protection Act 2018"),
]


def expand_instrument_aliases(query: str) -> str:
    """Append the full instrument name wherever an abbreviation appears."""
    extra = []
    for rx, full in INSTRUMENT_ALIASES:
        if rx.search(query or "") and full.lower() not in (query or "").lower():
            extra.append(full)
    return f"{query} {' '.join(extra)}" if extra else query


# Regime intent was removed here: a regex classifier deciding "this query wants PCR2015
# vs PA2023" from the query string alone proved unreliable in practice (e.g. it read
# "awarded under the old procurement regulations, just before the new Procurement Act
# rules came into force" as a CURRENT-law query, because the trigger phrases were narrow
# literal strings - "old procurement regime", not "old procurement regulations"). Regime
# is still shown on every result (`legal_regime`) as metadata; it no longer reweights the
# ranking. Graph expansion still excludes PCR2015/EU (see `expand()` call sites) as a
# fixed default, since that guard does not depend on per-query intent detection.

# Jurisdiction prior. The corpus retains 213 EU chunks (Official Journal directives) that
# informed the pre-Brexit regime. They are kept because legacy and transitional questions
# can turn on them, but they must not outrank domestic law on a UK question: measured, an
# EU directive ranked first on "can we split up a contract to avoid the procurement rules"
# and the resulting answer inverted the section 12 anti-avoidance position. A multiplicative
# demotion is used rather than a hard filter, for the same reason PCR2015 is demoted rather
# than removed - the material stays reachable when it is genuinely what the query is about.
JURISDICTION_WEIGHTS: dict[str, float] = {"EU": 0.35}
DEFAULT_JURISDICTION_WEIGHT = 1.0

# HAS_CHUNK is deliberately EXCLUDED by default. It is a structural relation
# (document -> its own chunks), so expanding it returns same-document siblings rather
# than legally connected provisions: measured, it flooded the graph channel and cut
# anchor recall from 0.618 to 0.510. The legal relations are the ones worth traversing.
EXPANSION_RELATIONS = ("CROSS_REFERS_TO", "REFERENCES")
EXPANSION_RELATIONS_WITH_STRUCTURE = ("CROSS_REFERS_TO", "REFERENCES", "HAS_CHUNK")


@dataclass
class Trace:
    """Everything needed to explain and debug a single query."""
    query: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    lexical_hits: list[dict[str, Any]] = field(default_factory=list)
    dense_hits: list[dict[str, Any]] = field(default_factory=list)
    fused: list[dict[str, Any]] = field(default_factory=list)
    anchors: list[str] = field(default_factory=list)
    edges_traversed: list[dict[str, Any]] = field(default_factory=list)
    graph_added: list[dict[str, Any]] = field(default_factory=list)
    final: list[dict[str, Any]] = field(default_factory=list)
    # Raw per-channel scores and the exact pool they were normalized over, for every candidate
    # (not just each channel's own top-k) - kept so a chunk absent from the result set (e.g. a
    # gold label the pipeline missed) can be scored with score_against_pool() using the SAME
    # normalization statistics as everything that WAS returned, rather than a separately-scaled
    # number that isn't comparable to the visible ranking.
    bm25_raw: dict[str, float] = field(default_factory=dict)
    dense_raw: dict[str, float] = field(default_factory=dict)
    pool: list = field(default_factory=list)  # list, not set, so asdict()/json stays serializable
    pool_meta: dict[str, dict[str, Any]] = field(default_factory=dict)


# Function words carry no retrieval signal but dominate an OR query: measured on this
# corpus, "for" matches 588 of 743 chunks and "can" 236, while the terms that actually
# discriminate - "cartel" (26), "rigging" (18) - are swamped. IDF alone does not rescue
# this, because every chunk still enters the candidate set.
FTS_STOPWORDS = {
    "a","an","and","are","as","at","be","been","but","by","can","could","do","does","for",
    "from","had","has","have","how","i","if","in","into","is","it","its","may","might","must",
    "of","on","or","should","so","such","than","that","the","their","them","then","there",
    "these","they","this","to","under","was","we","were","what","when","where","which","who",
    "why","will","with","would","you","your","about","any","all",
}


PREFIX_MIN_EXACT_DOCS = 5  # a term's exact form must match fewer docs than this to widen to prefix


def fts_query(text: str, vocab: dict[str, int] | None = None) -> str:
    """FTS5 MATCH string over content terms only.

    Quoting every term keeps punctuation from being read as FTS syntax. Stopwords are
    dropped; if a query is nothing but stopwords the original terms are used rather than
    returning an empty match.

    Each term is prefix-matched ("term"* - valid immediately after a quoted string in FTS5
    grammar) ONLY when its exact form is sparse: absent from `vocab`, or present in fewer
    than PREFIX_MIN_EXACT_DOCS documents. A common word ("procurement", thousands of exact
    hits) stays exact; a genuinely sparse or inflected one still gets the wider net. This
    replaced blanket prefix-matching-every-term after it measurably HURT recall rather than
    helping it: at the benchmark's candidates=50 depth, prefix-matching every term dropped
    gold-chunk candidate-pool membership from 4/14 to 3/14 on a 5-query sample, because
    widening a common word crowds the fixed-size pool with tangential matches
    ("procuring"/"procured" documents having nothing to do with the query) without rescuing
    anything the exact form would have missed. `vocab` (chunk_id-agnostic term -> doc-count,
    from ChunkRetriever._load_vocab) is optional so this stays callable without a retriever;
    omitted, every term is prefix-matched (the old, since-reverted blanket behaviour).
    """
    terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", text) if len(t) > 1]
    content = [t for t in terms if t.lower() not in FTS_STOPWORDS]
    use = content or terms

    def clause(t: str) -> str:
        sparse = vocab is None or vocab.get(t.lower(), 0) < PREFIX_MIN_EXACT_DOCS
        return f'"{t}"*' if sparse else f'"{t}"'

    return " OR ".join(clause(t) for t in use) or '""'


# Fuzzy matching: FTS5 has no native edit-distance operator (and this SQLite build has no
# loadable-extension support, so spellfix1 is not reachable either - see the venv/sqlite probe
# behind this change). rapidfuzz against the index's own vocabulary is the lightweight
# alternative: no new service, no reproducibility burden beyond one pure-C-extension pip
# package, and the vocabulary here is ~20k terms - small enough that a per-query fuzzy lookup
# costs single-digit milliseconds. Elasticsearch was the other option on the table; it was not
# used because it would add a whole second search engine (a JVM service to install, run and
# keep in sync) to what has otherwise been an entirely SQLite-based, reproducible pipeline, for
# a capability this gives us without a new moving part.
FUZZY_MIN_TERM_LEN = 4
FUZZY_SCORE_CUTOFF = 72
FUZZY_MAX_VARIANTS_PER_TERM = 3


class ChunkRetriever:
    def __init__(self, db_path: Path, collection: str = DEFAULT_COLLECTION,
                 qdrant_url: str | None = None, model_name: str | None = None):
        # SQLite connections are bound to the thread that created them, so a retriever
        # shared by a threaded server must hand out one connection per thread rather than
        # reusing a single handle. Disabling the check instead would silently permit
        # cross-thread use of one connection.
        self._db_path = db_path
        self._local = threading.local()
        self.collection = collection
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.model_name = model_name or os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
        self._model = None
        self._client = None
        self._vocab: dict[str, int] | None = None

    @property
    def con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self._db_path)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    def _load_vocab(self) -> dict[str, int]:
        """Distinct FTS5 terms -> document frequency, cached for the life of this retriever.
        `fts5vocab` is a metadata-only virtual table (no data duplication) built lazily so a
        retriever opened against a read-only DB copy degrades to no fuzzy matching instead of
        raising.
        """
        if self._vocab is not None:
            return self._vocab
        try:
            self.con.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab('chunks_fts', 'row')"
            )
            rows = self.con.execute("SELECT term, doc FROM chunks_fts_vocab").fetchall()
            self._vocab = {r["term"]: r["doc"] for r in rows}
        except sqlite3.OperationalError:
            self._vocab = {}
        return self._vocab

    def fuzzy_variants(self, query: str) -> dict[str, list[tuple[str, float]]]:
        """Original query term (lowercased) -> [(vocabulary variant, similarity 0-1), ...],
        for terms that have NEITHER an exact NOR a prefix match anywhere in the vocabulary -
        i.e. genuinely absent, not just rare. A word already reachable by fts_query()'s own
        prefix match is left alone here; fuzzy is only for the words that would otherwise
        contribute nothing at all.
        """
        vocab = self._load_vocab()
        if not vocab:
            return {}
        terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", query)
                 if len(t) >= FUZZY_MIN_TERM_LEN and t.lower() not in FTS_STOPWORDS]
        out: dict[str, list[tuple[str, float]]] = {}
        for t in terms:
            tl = t.lower()
            if tl in vocab or any(v.startswith(tl) for v in vocab):
                continue
            from rapidfuzz import process, fuzz
            matches = process.extract(tl, vocab.keys(), scorer=fuzz.ratio,
                                       limit=FUZZY_MAX_VARIANTS_PER_TERM, score_cutoff=FUZZY_SCORE_CUTOFF)
            if matches:
                out[tl] = [(m[0], m[1] / 100.0) for m in matches]
        return out

    # ---------------------------------------------------------------- channels
    @staticmethod
    def _lane_sql(lane: str | None) -> tuple[str, list[str]]:
        """SQL restriction for a retrieval lane: 'legislation' (primary/secondary legislation
        only) or 'other' (everything else); None = whole corpus. Applied at RETRIEVAL time so
        each lane gets its own full top-k - a post-hoc split of one shared pool starves the
        thinner lane (guidance echoes query phrasing and crowds statute out of the candidate
        slots before any lane logic runs)."""
        if lane is None:
            return "", []
        q = ",".join("?" * len(LEGISLATION_CLASSES))
        if lane == "legislation":
            return f" AND c.authority_class IN ({q})", list(LEGISLATION_CLASSES)
        return f" AND (c.authority_class IS NULL OR c.authority_class NOT IN ({q}))", list(LEGISLATION_CLASSES)

    def _bm25_raw(self, match_string: str, k: int | None = None,
                  chunk_ids: list[str] | None = None, lane: str | None = None) -> dict[str, float]:
        """One FTS5 MATCH -> {chunk_id: raw bm25 score (higher is better)}. Shared by every
        lexical lookup - a top-k scan (k set, chunk_ids None), an exact-ids lookup (chunk_ids
        set, k None, used to backfill a candidate found via another channel), or both bounded
        together. A chunk_ids filter that resolves to nothing returns {} without querying.
        `lane` restricts the scan to one retrieval lane (see _lane_sql).
        """
        if chunk_ids is not None and not chunk_ids:
            return {}
        where_ids, params = "", [match_string]
        if chunk_ids is not None:
            q = ",".join("?" * len(chunk_ids))
            where_ids = f" AND c.chunk_id IN ({q})"
            params += list(chunk_ids)
        lane_sql, lane_params = self._lane_sql(lane)
        params += lane_params
        sql = ("SELECT c.chunk_id, bm25(chunks_fts) AS score FROM chunks_fts "
               "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
               f"WHERE chunks_fts MATCH ?{where_ids}{lane_sql} ORDER BY score")
        if k is not None:
            sql += " LIMIT ?"
            params.append(k)
        try:
            rows = self.con.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return {}
        # bm25() returns lower-is-better; invert so higher is better everywhere.
        return {r["chunk_id"]: -r["score"] for r in rows}

    def _blended_bm25(self, query: str, k: int | None = None,
                       chunk_ids: list[str] | None = None, lane: str | None = None) -> dict[str, float]:
        """Exact/prefix bm25 PLUS a separate, similarity-discounted bm25 pass per fuzzy-expanded
        term, summed per chunk. Fuzzy runs as its OWN MATCH query per original term (not folded
        into one big OR) specifically so each term's contribution can be scaled by how close its
        best vocabulary match actually was - a barely-over-the-cutoff guess (e.g. "tendor" ->
        "tendon" at 0.83) counts for less than a near-exact one ("procurment" -> "procurement"
        at ~0.95), and neither ever counts as much as a term the query actually contained.
        A fixed overfetch (k*8, floor 400) when k is set gives fuzzy's additive contribution
        room to reorder the top-k before truncation; exact chunk_ids lookups are unbounded.
        """
        scan_k = None if chunk_ids is not None else max((k or 0) * 8, 400)
        combined = dict(self._bm25_raw(fts_query(query, self._load_vocab()), k=scan_k, chunk_ids=chunk_ids, lane=lane))
        for term, variants in self.fuzzy_variants(query).items():
            discount = max(ratio for _, ratio in variants)
            fuzzy_match = " OR ".join(f'"{v}"*' for v, _ in variants)
            for cid, s in self._bm25_raw(fuzzy_match, k=scan_k, chunk_ids=chunk_ids, lane=lane).items():
                combined[cid] = combined.get(cid, 0.0) + discount * s
        if chunk_ids is not None:
            return combined
        return dict(sorted(combined.items(), key=lambda kv: -kv[1])[:k]) if k is not None else combined

    def lexical(self, query: str, k: int, lane: str | None = None) -> list[dict[str, Any]]:
        scored = self._blended_bm25(query, k=k, lane=lane)
        return [{"chunk_id": cid, "score": s, "channel": "lexical"} for cid, s in scored.items()]

    def lexical_scores_for(self, query: str, chunk_ids: list[str]) -> dict[str, float]:
        """Exact (blended exact+discounted-fuzzy) BM25 for specific ids, used to backfill a
        candidate that entered the pool via dense or graph but fell outside lexical()'s own
        top-k. An id this returns nothing for has zero term overlap on both passes - its true
        lexical relevance is 0.0, not unknown - callers default absent ids to 0.0 rather than
        treating it as an error. No lane filter: explicit ids already belong to a lane.
        """
        return self._blended_bm25(query, chunk_ids=chunk_ids)

    def _ensure_dense_model(self) -> None:
        if self._model is None:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._client = QdrantClient(url=self.qdrant_url, timeout=60)

    def embed_query(self, query: str):
        self._ensure_dense_model()
        return self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    def dense(self, query: str, k: int, vec=None, lane: str | None = None) -> list[dict[str, Any]]:
        """`lane` = 'legislation' | 'other' | None restricts the vector search to one retrieval
        lane via a payload filter on authority_class (see _lane_sql for why at retrieval time)."""
        try:
            self._ensure_dense_model()
        except ImportError:
            return []
        if vec is None:
            vec = self.embed_query(query)
        flt = None
        if lane is not None:
            from qdrant_client import models
            cond = models.FieldCondition(key="authority_class", match=models.MatchAny(any=list(LEGISLATION_CLASSES)))
            flt = models.Filter(must=[cond]) if lane == "legislation" else models.Filter(must_not=[cond])
        res = self._client.query_points(
            collection_name=self.collection, query=vec.tolist(), limit=k, with_payload=True, query_filter=flt
        ).points
        return [
            {"chunk_id": p.payload.get("chunk_id"), "score": float(p.score), "channel": "dense"}
            for p in res
        ]

    def dense_scores_for(self, chunk_ids: list[str], vec) -> dict[str, float]:
        """Exact cosine similarity for specific ids, used to backfill a candidate that entered
        the pool via lexical or graph but fell outside dense()'s own top-k. Qdrant point ids
        are arbitrary integers, not the chunk_id, so a candidate can only be scored by
        re-querying with a payload filter restricted to it, not by direct id lookup.
        """
        if not chunk_ids:
            return {}
        try:
            self._ensure_dense_model()
        except ImportError:
            return {}
        from qdrant_client import models
        flt = models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=chunk_ids))])
        res = self._client.query_points(
            collection_name=self.collection, query=vec.tolist(),
            query_filter=flt, limit=len(chunk_ids), with_payload=True,
        ).points
        return {p.payload.get("chunk_id"): float(p.score) for p in res}

    # ------------------------------------------------------------------ fusion
    BM25_WEIGHT = 0.40
    DENSE_WEIGHT = 0.60

    AUTH_BLEND = 0.30  # share of final score from authority standing in-pool; see _authority_norm

    @staticmethod
    def _normalize(raw: dict[str, float], stats_from: dict[str, float] | None = None) -> dict[str, float]:
        """Z-score each value in `raw` against a reference pool's mean/std, then squash through
        tanh into (0, 1). By default the reference pool IS `raw` (fit and transform together).

        Passing `stats_from` fits mean/std on a DIFFERENT, usually smaller "core" set and applies
        those frozen statistics to (possibly more than just that set of) `raw` values. This
        matters because z-score normalization, unlike RRF, is NOT invariant to what else is in
        the pool: measured on a real query at the benchmark's candidates=50 depth, adding just 5
        graph-expansion candidates to a 90-item pool shifted an unrelated, already-present gold
        chunk's normalized BM25 score by +63% even though its raw score never changed - enough to
        move its rank. Freezing the statistics to the "core" pool (lexical+dense candidates,
        before graph) and scoring graph-added (or, in score_against_pool, a single folded-in)
        candidates against that same frozen reference stops turning graph on/off, or asking "what
        would this chunk have scored", from silently perturbing everyone else's score.
        """
        basis = stats_from if stats_from is not None else raw
        if not basis:
            return {}
        vals = list(basis.values())
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5 or 1.0
        return {k: (math.tanh((v - mean) / std) + 1) / 2 for k, v in raw.items()}

    def _fuse(self, bm25_raw: dict[str, float], dense_raw: dict[str, float], pool: set[str],
              graph_raw: dict[str, float] | None = None, graph_weight: float = 0.0,
              stats_pool: set[str] | None = None
              ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """0.4*bm25_norm + 0.6*dense_norm (+ graph_weight*graph_norm for graph-added ids), over
        `pool`. `stats_pool`, if given, is the smaller reference set whose bm25/dense values fix
        the normalization mean/std (see _normalize) - `pool` can then safely be a superset (e.g.
        stats_pool plus graph-added ids) without changing any stats_pool member's own score.
        Returns (fused, bm25_norm, dense_norm) so callers can report the normalized per-channel
        components alongside the raw scores.
        """
        basis = stats_pool if stats_pool is not None else pool
        bm25_basis = {c: bm25_raw.get(c, 0.0) for c in basis}
        dense_basis = {c: dense_raw.get(c, 0.0) for c in basis}
        bm25_n = self._normalize({c: bm25_raw.get(c, 0.0) for c in pool}, stats_from=bm25_basis)
        dense_n = self._normalize({c: dense_raw.get(c, 0.0) for c in pool}, stats_from=dense_basis)
        graph_n = self._normalize(graph_raw) if graph_raw else {}
        fused = {}
        for c in pool:
            s = self.BM25_WEIGHT * bm25_n.get(c, 0.0) + self.DENSE_WEIGHT * dense_n.get(c, 0.0)
            if c in graph_n:
                s += graph_weight * graph_n[c]
            fused[c] = s
        return fused, bm25_n, dense_n

    def _authority_norm(self, pool: set[str], meta: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Authority standing relative to this pool, on the same (0,1) tanh scale as bm25/dense -
        so authority can be blended additively (AUTH_BLEND) with a fixed, scale-independent share
        of the final score instead of a flat multiplier. The multiplier alone lost its practical
        power once fusion moved from RRF (whose top-of-list scores are naturally compressed into
        a narrow band, so a 0.55-1.00x nudge could flip close candidates) to this normalized sum
        (whose top-of-list spread can be much wider per query, making the same 0.55-1.00x band a
        proportionally smaller lever) - measured as a real regression (F strict recall@10
        0.411->0.272) before this fix.
        """
        raw = {c: AUTHORITY_WEIGHTS.get((meta.get(c) or {}).get("authority_class"), DEFAULT_AUTHORITY_WEIGHT)
               for c in pool}
        return self._normalize(raw)

    def score_against_pool(self, query: str, chunk_id: str, bm25_raw: dict[str, float],
                            dense_raw: dict[str, float], pool: set[str],
                            pool_meta: dict[str, dict[str, Any]], vec=None) -> dict[str, Any] | None:
        """Fold one extra chunk (e.g. a gold-labelled id absent from the retrieved pool) into
        an already-computed candidate pool, and score it with the SAME normalization statistics
        (the pool's mean/std) that produced every other row's score - so the result is directly
        comparable to the pool's own final_score column, not a separately-scaled number. Returns
        the chunk's row plus the rank it would occupy if inserted into that pool.

        `pool_meta` must already carry `load()` output for every id in `pool` (the caller has
        this from building its own result rows) - re-querying per id here would be one SQL call
        per pool member just to rank one extra chunk.
        """
        meta = self.load([chunk_id]).get(chunk_id)
        if not meta:
            return None
        retrieval_query = expand_instrument_aliases(query)
        vec = vec if vec is not None else self.embed_query(retrieval_query)
        b = dict(bm25_raw)
        if chunk_id not in b:
            b[chunk_id] = self.lexical_scores_for(retrieval_query, [chunk_id]).get(chunk_id, 0.0)
        d = dict(dense_raw)
        if chunk_id not in d:
            d[chunk_id] = self.dense_scores_for([chunk_id], vec).get(chunk_id, 0.0)
        augmented_pool = pool | {chunk_id}
        # stats_pool=pool freezes normalization to the ORIGINAL pool, so every existing row's
        # bm25_norm/dense_norm/final_score computed here is identical to what the real search
        # already returned for it - only the folded-in chunk_id's own score is new.
        fused, bm25_n, dense_n = self._fuse(b, d, augmented_pool, stats_pool=pool)
        auth_raw_pool = {c: AUTHORITY_WEIGHTS.get((pool_meta.get(c) or {}).get("authority_class"), DEFAULT_AUTHORITY_WEIGHT)
                          for c in pool}
        authority = AUTHORITY_WEIGHTS.get(meta.get("authority_class"), DEFAULT_AUTHORITY_WEIGHT)
        authority_n = self._normalize({**auth_raw_pool, chunk_id: authority}, stats_from=auth_raw_pool)
        juris_w = JURISDICTION_WEIGHTS.get(meta.get("jurisdiction"), DEFAULT_JURISDICTION_WEIGHT)
        blended = fused[chunk_id] * (1 - self.AUTH_BLEND) + authority_n.get(chunk_id, 0.5) * self.AUTH_BLEND
        final_score = blended * juris_w
        rank = 1
        for c in pool:
            pm = pool_meta.get(c)
            if not pm:
                continue
            pj = JURISDICTION_WEIGHTS.get(pm.get("jurisdiction"), DEFAULT_JURISDICTION_WEIGHT)
            p_blended = fused.get(c, 0.0) * (1 - self.AUTH_BLEND) + authority_n.get(c, 0.5) * self.AUTH_BLEND
            if p_blended * pj > final_score:
                rank += 1
        return {
            "chunk_id": chunk_id, "was_in_pool": chunk_id in pool,
            "bm25_score": b[chunk_id], "bm25_norm": bm25_n.get(chunk_id),
            "dense_score": d[chunk_id], "dense_norm": dense_n.get(chunk_id),
            "fusion_score": fused[chunk_id], "authority_weight": authority,
            "authority_norm": authority_n.get(chunk_id), "jurisdiction_weight": juris_w,
            "final_score": final_score, "rank_if_inserted": rank, "pool_size": len(pool),
            "authority_class": meta.get("authority_class"), "legal_regime": meta.get("legal_regime"),
            "jurisdiction": meta.get("jurisdiction"), "citation": meta.get("citation"),
            "retrieval_title": meta.get("retrieval_title"), "text": meta.get("text"),
        }

    # ------------------------------------------------------------------- graph
    def anchors_for(self, chunk_ids: list[str]) -> list[str]:
        """Legal identity of retrieved chunks: provision first, else document."""
        if not chunk_ids:
            return []
        q = ",".join("?" * len(chunk_ids))
        rows = self.con.execute(
            f"SELECT chunk_id, parent_node_id, document_id FROM chunks WHERE chunk_id IN ({q})",
            chunk_ids,
        ).fetchall()
        out = []
        for r in rows:
            # The chunk id itself is an anchor: guidance and commentary citations are
            # recorded as CHUNK --REFERENCES--> PROVISION, so anchoring only on the
            # parent provision or document would miss the entire guidance->law layer.
            out.append(r["chunk_id"])
            if r["parent_node_id"]:
                out.append(r["parent_node_id"])
            out.append(r["document_id"])
        return [a for a in dict.fromkeys(out) if a]

    def expand(self, anchors: list[str], hops: int, per_hop: int,
               relations: tuple[str, ...], trace: Trace,
               exclude_regimes: tuple[str, ...] = (),
               exclude_jurisdictions: tuple[str, ...] = ()) -> list[str]:
        """Bounded traversal. Returns chunk ids reachable from the anchors.

        Regime filtering is applied HERE, not only at reranking. Legacy legislation is
        disproportionately central in the citation graph - PCR2015 is the target of 2,376
        cross-references against PA2023's 2,217, because the older instrument accumulated
        internal references over a decade. Expanding into it for a current-law question
        floods the graph channel with repealed provisions, and a downstream score penalty
        cannot recover a result set that has already been crowded out. Demotion is the
        right instrument at ranking time; exclusion is the right one at traversal time.
        """
        frontier = list(anchors)
        seen_nodes = set(anchors)
        node_order: list[str] = []
        reached_chunks: list[str] = []
        rel_clause = ",".join("?" * len(relations))
        for hop in range(1, hops + 1):
            if not frontier:
                break
            q = ",".join("?" * len(frontier))
            # Traverse on retrieval_target_id, falling back to target_id where the
            # densification pass has not run. target_id remains the exact legal reference;
            # retrieval_target_id is that reference rolled up to the granularity that was
            # actually chunked. Before this, 39.8% of legal edges pointed at sub-provision
            # nodes with no chunk, so the traversal silently dropped them.
            # Deduplicate on (source, target, relation). A provision may cite the same
            # target from several subsections, and rolling sources up to provision level
            # collapses those into identical edges: measured, 30.9% of legal edges are
            # duplicates this way, with one pair appearing 26 times. Because the fan-out
            # cap is applied by LIMIT, duplicates consume traversal budget without
            # reaching anything new - only 75.1% of a source's edges lead somewhere
            # distinct. MAX(confidence) keeps the strongest evidence for the pair.
            rows = self.con.execute(
                f"SELECT COALESCE(retrieval_source_id, source_id) AS source_id, relation, "
                f"COALESCE(retrieval_target_id, target_id) AS target_id, "
                f"MIN(target_id) AS legal_target_id, MAX(confidence) AS confidence, "
                f"MIN(resolution_status) AS resolution_status "
                f"FROM edges WHERE COALESCE(retrieval_source_id, source_id) IN ({q}) "
                f"AND relation IN ({rel_clause}) "
                f"GROUP BY source_id, target_id, relation "
                f"ORDER BY confidence DESC LIMIT ?",
                (*frontier, *relations, per_hop),
            ).fetchall()
            next_frontier = []
            for r in rows:
                trace.edges_traversed.append(
                    {"hop": hop, "source": r["source_id"], "relation": r["relation"],
                     "target": r["target_id"], "legal_target": r["legal_target_id"],
                     "confidence": r["confidence"], "status": r["resolution_status"]}
                )
                tgt = r["target_id"]
                if tgt in seen_nodes:
                    continue
                seen_nodes.add(tgt)
                node_order.append(tgt)
                next_frontier.append(tgt)
            frontier = next_frontier
        # Map reached provisions/documents to their chunks, preserving discovery order so
        # the graph channel is RANKED (nearer hops and higher-confidence edges first)
        # rather than an unordered bag.
        ordered_targets = [n for n in node_order if n not in set(anchors)]
        exclude_clause = ""
        params_extra: list[str] = []
        if exclude_regimes:
            exclude_clause += (" AND (legal_regime IS NULL OR legal_regime NOT IN (%s))"
                               % ",".join("?" * len(exclude_regimes)))
            params_extra += list(exclude_regimes)
        if exclude_jurisdictions:
            # Regime exclusion alone leaves EU-jurisdiction chunks reachable: they carry no
            # legal_regime (they predate the PA2023/PCR2015/PR2024 split entirely), so the
            # regime clause's own "legal_regime IS NULL" branch passes them through. The
            # 0.35x jurisdiction demotion in search()'s final scoring does not help here -
            # it discounts a candidate already in the pool, but traversal is what put it
            # there ahead of a chunk that never entered the graph channel at all.
            exclude_clause += (" AND (jurisdiction IS NULL OR jurisdiction NOT IN (%s))"
                               % ",".join("?" * len(exclude_jurisdictions)))
            params_extra += list(exclude_jurisdictions)
        for node in ordered_targets:
            rows = self.con.execute(
                "SELECT chunk_id FROM chunks WHERE (parent_node_id = ? OR document_id = ? "
                "OR chunk_id = ?)" + exclude_clause,
                (node, node, node, *params_extra),
            ).fetchall()
            for r in rows:
                if r["chunk_id"] not in reached_chunks:
                    reached_chunks.append(r["chunk_id"])
        return reached_chunks

    # ---------------------------------------------------------------- metadata
    def load(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        q = ",".join("?" * len(chunk_ids))
        rows = self.con.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({q})", chunk_ids).fetchall()
        # Superseded chunks are duplicate ingestions of an instrument already held in a
        # parsed, provision-level form. They carry no legal identity, are far coarser, and
        # compete for the same result slots. Dropping them here removes them from every
        # channel at once, because every channel's output passes through load().
        out = {}
        for r in rows:
            d = dict(r)
            if d.get("superseded_by"):
                continue
            # Blocks the content filter identified as page furniture rather than content:
            # advice footers, copyright notices, contact blocks, navigation listings.
            # Validated against 900 LLM labels at 0% false-positive on GOOD chunks.
            if d.get("filtered_out"):
                continue
            out[d["chunk_id"]] = d
        return out

    # ---------------------------------------------------------------- pipeline
    def search(self, query: str, top_k: int = 10, candidates: int = 40,
               use_lexical: bool = True, use_dense: bool = True, use_graph: bool = True,
               use_rerank: bool = True, hops: int = 1, per_hop: int = 40,
               graph_weight: float = 0.60, graph_channel_cap: int | None = None,
               use_legislation_lane: bool = True, legislation_lane_top_n: int = 3,
               expand_query: bool | str = False,
               expansion_relations: tuple[str, ...] = EXPANSION_RELATIONS) -> tuple[list[dict[str, Any]], Trace]:
        trace = Trace(query=query)
        # Vocabulary bridge, applied BEFORE matching. The question itself is unchanged;
        # only the string handed to the lexical and dense channels is augmented, because
        # the measured failure is at the first stage rather than in ranking.
        # Applied to the matching string only; `query` itself is what gets logged, traced
        # and evaluated, so the intervention stays confined to the first stage.
        retrieval_query = expand_instrument_aliases(query)
        if expand_query:
            from query_expansion import QueryExpander
            if not hasattr(self, "_expander"):
                self._expander = QueryExpander()
            exp = self._expander.expand(query)
            # Expand only when the question actually uses non-statutory phrasing.
            # Applied unconditionally, expansion raised recall on the practitioner-phrased
            # suite from 0.438 to 0.625 but cut the semantic suite from 0.800 to 0.467:
            # added terminology dilutes a query that already matches the statute. Gating on
            # a detected colloquialism confines the intervention to the case it addresses.
            has_colloquialism = bool(exp.get("colloquialisms_detected"))
            if expand_query == "always" or has_colloquialism:
                retrieval_query = exp.get("expanded_query") or query
            trace.config["query_expansion"] = {
                "statutory_terms": exp.get("statutory_terms"),
                "colloquialisms_detected": exp.get("colloquialisms_detected"),
                "source": exp.get("source"),
            }
        trace.config = {
            "retrieval_version": RETRIEVAL_VERSION, "top_k": top_k, "candidates": candidates,
            "lexical": use_lexical, "dense": use_dense, "graph": use_graph, "rerank": use_rerank,
            "hops": hops, "per_hop": per_hop, "graph_weight": graph_weight,
            "graph_channel_cap": graph_channel_cap,
            "fusion": "normalized_weighted_sum", "bm25_weight": self.BM25_WEIGHT, "dense_weight": self.DENSE_WEIGHT,
            "authority_weights": AUTHORITY_WEIGHTS, "jurisdiction_weights": JURISDICTION_WEIGHTS, "expansion_relations": list(expansion_relations),
            "use_legislation_lane": use_legislation_lane, "legislation_lane_top_n": legislation_lane_top_n,
        }

        if not use_lexical and not use_dense:
            return [], trace
        vec = self.embed_query(retrieval_query) if use_dense else None
        if use_lexical:
            trace.lexical_hits = self.lexical(retrieval_query, candidates)
        if use_dense:
            trace.dense_hits = self.dense(retrieval_query, candidates, vec=vec)
        bm25_raw = {h["chunk_id"]: h["score"] for h in trace.lexical_hits}
        dense_raw = {h["chunk_id"]: h["score"] for h in trace.dense_hits}
        union = set(bm25_raw) | set(dense_raw)

        # Every candidate gets a real score on BOTH channels, not just the one that surfaced it:
        # a chunk found only by dense search still gets an exact BM25 score (0.0 if FTS5 truly
        # has no term overlap - a real value, not a gap), and vice versa via a payload-filtered
        # Qdrant re-query. This is what makes every result row report both scores.
        if use_lexical:
            missing = [c for c in union if c not in bm25_raw]
            bm25_raw.update({c: 0.0 for c in missing})
            bm25_raw.update(self.lexical_scores_for(retrieval_query, missing))
        if use_dense:
            missing = [c for c in union if c not in dense_raw]
            if missing:
                dense_raw.update(self.dense_scores_for(missing, vec))

        fused0, _, _ = self._fuse(bm25_raw, dense_raw, union)
        seeds = sorted(union, key=lambda c: -fused0.get(c, 0.0))[:top_k]
        retrieved = set(union)
        graph_only: set[str] = set()
        graph_raw: dict[str, float] = {}

        if use_graph and seeds:
            trace.anchors = self.anchors_for(seeds)
            # Always excludes repealed PCR2015 provisions and pre-Brexit EU directives from
            # graph traversal - per-query legacy-intent detection proved unreliable (see the
            # module docstring) and was removed, so this is now a fixed, unconditional default.
            exclude = ("PCR2015",)
            exclude_juris = ("EU",)
            trace.config["expansion_excluded_regimes"] = list(exclude)
            trace.config["expansion_excluded_jurisdictions"] = list(exclude_juris)
            added = self.expand(trace.anchors, hops, per_hop, expansion_relations, trace,
                                exclude_regimes=exclude, exclude_jurisdictions=exclude_juris)
            capped = added[:graph_channel_cap] if graph_channel_cap else added
            graph_only = set(capped) - retrieved
            trace.graph_added = [{"chunk_id": c} for c in list(graph_only)[:50]]
            graph_raw = {cid: 1.0 / rank for rank, cid in enumerate(capped, 1)}
            if graph_only:
                if use_lexical:
                    bm25_raw.update({c: 0.0 for c in graph_only})  # graph_only is disjoint from union by construction
                    bm25_raw.update(self.lexical_scores_for(retrieval_query, list(graph_only)))
                if use_dense:
                    dense_raw.update(self.dense_scores_for(list(graph_only), vec))

        full_pool = union | graph_only
        # stats_pool=union freezes bm25/dense normalization to the pre-graph candidates, so
        # turning graph on never perturbs a non-graph candidate's own score (see _normalize).
        fused, bm25_n, dense_n = self._fuse(bm25_raw, dense_raw, full_pool, graph_raw, graph_weight,
                                             stats_pool=union)
        trace.bm25_raw, trace.dense_raw, trace.pool = bm25_raw, dense_raw, sorted(full_pool)

        meta = self.load(list(full_pool))
        trace.pool_meta = meta
        authority_n = self._authority_norm(full_pool, meta) if use_rerank else {}
        results = []
        for cid, base_score in fused.items():
            m = meta.get(cid)
            if not m:
                continue
            authority = AUTHORITY_WEIGHTS.get(m.get("authority_class"), DEFAULT_AUTHORITY_WEIGHT)
            juris_w = JURISDICTION_WEIGHTS.get(m.get("jurisdiction"), DEFAULT_JURISDICTION_WEIGHT)
            # Authority blends in as its own normalized channel (AUTH_BLEND share) rather than a
            # flat multiplier - see _authority_norm for why the multiplier alone lost its power.
            blended = base_score * (1 - self.AUTH_BLEND) + authority_n.get(cid, 0.5) * self.AUTH_BLEND
            final = blended * juris_w if use_rerank else base_score
            results.append(
                {
                    "chunk_id": cid,
                    "final_score": final,
                    "fusion_score": base_score,
                    "bm25_score": bm25_raw.get(cid), "bm25_norm": bm25_n.get(cid),
                    "dense_score": dense_raw.get(cid), "dense_norm": dense_n.get(cid),
                    "authority_weight": authority if use_rerank else None,
                    "authority_norm": authority_n.get(cid) if use_rerank else None,
                    "jurisdiction_weight": juris_w if use_rerank else None,
                    "jurisdiction": m.get("jurisdiction"),
                    "via_graph_only": cid in graph_only,
                    "authority_class": m.get("authority_class"),
                    "legal_regime": m.get("legal_regime"),
                    "citation": m.get("citation"),
                    "retrieval_title": m.get("retrieval_title"),
                    "source_url": m.get("source_url"),
                    "document_id": m.get("document_id"),
                    "text": m.get("text"),
                }
            )
        results.sort(key=lambda r: -r["final_score"])
        if use_legislation_lane:
            lane_check = results[:legislation_lane_top_n]
            if not any(r["authority_class"] in LEGISLATION_CLASSES for r in lane_check):
                lane_candidates = [r for r in results if r["authority_class"] in LEGISLATION_CLASSES]
                if lane_candidates:
                    winner = lane_candidates[0]  # already sorted by final_score desc
                    results.remove(winner)
                    results.insert(0, winner)
                    trace.config["legislation_lane_promoted"] = winner["chunk_id"]
        results = results[:top_k]
        trace.final = [
            {k: v for k, v in r.items() if k != "text"} for r in results
        ]
        return results, trace

    def search_two_lanes(self, query: str, top_k_legislation: int = 5, top_k_other: int = 5,
                          candidates: int = 100, use_graph: bool = True, hops: int = 1,
                          per_hop: int = 40, graph_weight: float = 0.60, graph_channel_cap: int | None = None,
                          expansion_relations: tuple[str, ...] = EXPANSION_RELATIONS,
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Trace]:
        """Legislation and everything-else, ranked and returned as two SEPARATE lists.

        `search()` merges every source into one ranked list and, when legislation falls
        out of the top few places, promotes its single best candidate into that list.
        That is a patch on a single ranking, not a different architecture: guidance and
        statute still compete for the same slots before the patch fires, and only one
        legislation chunk ever gets rescued regardless of how many are actually relevant.

        Measured on the 150-query test set: a real split - each lane weighted independently,
        never compared against the other - gives the legislation lane 58.7% hit@10, ahead of
        every merged configuration tried (RRF 40.7%, dense-only 46.7%). That measurement was
        taken when the split was still post-hoc (one shared pool from the whole corpus,
        partitioned only at the final sort), which starved the thinner lane: guidance echoes
        query phrasing and took most candidate slots and set the normalization baseline before
        statute ever got its own lane. The lanes are now separated at RETRIEVAL time: each runs
        its own lexical and dense search (authority-class filter in FTS and in the Qdrant payload
        filter), gets its own full top-`candidates` per channel, its own normalization statistics
        and its own authority blend. `candidates` is therefore per lane, not shared.

        The two lists are not reconciled into a single ranking here. Presentation
        (leading with law, deduplicating, capping a combined bundle) is the caller's
        job - answer_query.py and chunk_api.py - so that decision stays visible and
        auditable rather than hidden inside retrieval.
        """
        trace = Trace(query=query)
        retrieval_query = expand_instrument_aliases(query)
        trace.config = {
            "retrieval_version": RETRIEVAL_VERSION, "mode": "two_lane",
            "top_k_legislation": top_k_legislation, "top_k_other": top_k_other,
            "candidates": candidates, "graph": use_graph, "hops": hops, "per_hop": per_hop,
            "fusion": "normalized_weighted_sum", "bm25_weight": self.BM25_WEIGHT, "dense_weight": self.DENSE_WEIGHT,
            "graph_weight": graph_weight,
        }

        # Two INDEPENDENT retrievals. Each lane runs its own lexical and dense search with the
        # lane filter applied at retrieval time, gets its own full top-`candidates` from each
        # channel, its own backfill, its own normalization statistics and its own authority
        # blend. Nothing in one lane is ever scored against, normalized with, or displaced by a
        # candidate from the other. (The earlier form retrieved one shared pool from the whole
        # corpus and only partitioned the final sort - guidance, which echoes query phrasing,
        # took most candidate slots and set the normalization baseline before statute ever got
        # a lane of its own.)
        vec = self.embed_query(retrieval_query)
        lanes: dict[str, dict[str, Any]] = {}
        for lane in ("legislation", "other"):
            lex = self.lexical(retrieval_query, candidates, lane=lane)
            den = self.dense(retrieval_query, candidates, vec=vec, lane=lane)
            trace.lexical_hits += lex
            trace.dense_hits += den
            b = {h["chunk_id"]: h["score"] for h in lex}
            d = {h["chunk_id"]: h["score"] for h in den}
            union = set(b) | set(d)
            missing = [c for c in union if c not in b]
            b.update({c: 0.0 for c in missing})
            b.update(self.lexical_scores_for(retrieval_query, missing))
            missing = [c for c in union if c not in d]
            if missing:
                d.update(self.dense_scores_for(missing, vec))
            lanes[lane] = {"bm25": b, "dense": d, "union": union, "graph_only": set()}

        graph_raw: dict[str, float] = {}
        if use_graph:
            # Seeds come from BOTH lanes' preliminary rankings; a reached chunk joins the lane
            # its authority class belongs to (guidance->law citations land in the legislation
            # lane, which is the whole point of the graph channel).
            seeds: list[str] = []
            for lane, L in lanes.items():
                f0, _, _ = self._fuse(L["bm25"], L["dense"], L["union"])
                k = top_k_legislation if lane == "legislation" else top_k_other
                seeds += sorted(L["union"], key=lambda c: -f0.get(c, 0.0))[:k]
            if seeds:
                trace.anchors = self.anchors_for(seeds)
                # Fixed default now that per-query legacy-intent detection has been removed (see
                # the module docstring): always keep repealed PCR2015 / pre-Brexit EU material
                # out of graph traversal.
                exclude = ("PCR2015",)
                exclude_juris = ("EU",)
                trace.config["expansion_excluded_regimes"] = list(exclude)
                trace.config["expansion_excluded_jurisdictions"] = list(exclude_juris)
                added = self.expand(trace.anchors, hops, per_hop, expansion_relations, trace,
                                    exclude_regimes=exclude, exclude_jurisdictions=exclude_juris)
                capped = added[:graph_channel_cap] if graph_channel_cap else added
                retrieved = lanes["legislation"]["union"] | lanes["other"]["union"]
                graph_only = set(capped) - retrieved
                trace.graph_added = [{"chunk_id": c} for c in list(graph_only)[:50]]
                graph_raw = {cid: 1.0 / rank for rank, cid in enumerate(capped, 1)}
                if graph_only:
                    gmeta = self.load(list(graph_only))
                    b_add = self.lexical_scores_for(retrieval_query, list(graph_only))
                    d_add = self.dense_scores_for(list(graph_only), vec)
                    for cid, m in gmeta.items():
                        lane = "legislation" if m.get("authority_class") in LEGISLATION_CLASSES else "other"
                        lanes[lane]["graph_only"].add(cid)
                        lanes[lane]["bm25"][cid] = b_add.get(cid, 0.0)
                        lanes[lane]["dense"][cid] = d_add.get(cid, 0.0)

        trace.bm25_raw, trace.dense_raw, trace.pool, trace.pool_meta = {}, {}, [], {}
        ranked: dict[str, list[dict[str, Any]]] = {}
        for lane, L in lanes.items():
            pool = L["union"] | L["graph_only"]
            # stats_pool=L["union"] freezes this lane's normalization to its own pre-graph
            # candidates, so graph additions never perturb an existing member's score.
            fused, bm25_n, dense_n = self._fuse(L["bm25"], L["dense"], pool, graph_raw, graph_weight,
                                                 stats_pool=L["union"])
            meta = self.load(list(pool))
            authority_n = self._authority_norm(pool, meta)
            rows = []
            for cid, base_score in fused.items():
                m = meta.get(cid)
                if not m:
                    continue
                authority = AUTHORITY_WEIGHTS.get(m.get("authority_class"), DEFAULT_AUTHORITY_WEIGHT)
                juris_w = JURISDICTION_WEIGHTS.get(m.get("jurisdiction"), DEFAULT_JURISDICTION_WEIGHT)
                # Authority blends in as its own normalized channel (AUTH_BLEND share) rather
                # than a flat multiplier - see _authority_norm. Within the legislation lane this
                # only separates primary from secondary legislation; within the other lane it
                # orders the guidance tiers.
                blended = base_score * (1 - self.AUTH_BLEND) + authority_n.get(cid, 0.5) * self.AUTH_BLEND
                rows.append({
                    "chunk_id": cid, "lane": lane, "final_score": blended * juris_w,
                    "fusion_score": base_score,
                    "bm25_score": L["bm25"].get(cid), "bm25_norm": bm25_n.get(cid),
                    "dense_score": L["dense"].get(cid), "dense_norm": dense_n.get(cid),
                    "authority_weight": authority, "authority_norm": authority_n.get(cid),
                    "jurisdiction_weight": juris_w, "jurisdiction": m.get("jurisdiction"),
                    "via_graph_only": cid in L["graph_only"], "authority_class": m.get("authority_class"),
                    "legal_regime": m.get("legal_regime"), "citation": m.get("citation"),
                    "retrieval_title": m.get("retrieval_title"), "source_url": m.get("source_url"),
                    "document_id": m.get("document_id"), "text": m.get("text"),
                })
            rows.sort(key=lambda r: -r["final_score"])
            ranked[lane] = rows
            trace.bm25_raw.update(L["bm25"]); trace.dense_raw.update(L["dense"])
            trace.pool += sorted(pool); trace.pool_meta.update(meta)

        legislation, other = ranked["legislation"][:top_k_legislation], ranked["other"][:top_k_other]
        trace.final = [{k: v for k, v in r.items() if k != "text"} for r in legislation + other]
        return legislation, other, trace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--db", default=DEFAULT_DB, type=Path)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--no-lexical", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--per-hop", type=int, default=40)
    ap.add_argument("--debug", action="store_true", help="print the full retrieval trace")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-text", type=int, default=0, help="chars of evidence text to show")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    db = args.db if args.db.is_absolute() else root / args.db
    r = ChunkRetriever(db, args.collection)
    results, trace = r.search(
        args.query, top_k=args.top_k, candidates=args.candidates,
        use_lexical=not args.no_lexical, use_dense=not args.no_dense,
        use_graph=not args.no_graph, use_rerank=not args.no_rerank,
        hops=args.hops, per_hop=args.per_hop,
    )

    if args.json:
        print(json.dumps({"results": results, "trace": asdict(trace)}, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQUERY: {trace.query}")
    print(f"channels: lexical={len(trace.lexical_hits)} dense={len(trace.dense_hits)} "
          f"pool={len(trace.pool)} anchors={len(trace.anchors)} "
          f"edges={len(trace.edges_traversed)} graph_added={len(trace.graph_added)}")
    print("-" * 100)
    for i, res in enumerate(results, 1):
        tag = " [GRAPH-ONLY]" if res["via_graph_only"] else ""
        print(f"{i:2d}. {res['final_score']:.5f}{tag}  [{res['authority_class']}"
              f"{'/' + res['legal_regime'] if res['legal_regime'] else ''}]")
        print(f"    {res['citation'] or res['document_id']}")
        print(f"    {res['retrieval_title']}")
        if args.show_text:
            print(f"    {(res['text'] or '')[:args.show_text]}...")
    if args.debug:
        print("\n--- TRACE ---")
        print(json.dumps(asdict(trace), indent=2, ensure_ascii=False)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
