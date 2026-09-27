#!/usr/bin/env python3
"""Build the searchable index over the semantic chunk corpus.

Reuses the infrastructure already in this repository rather than introducing new
technology: SQLite FTS5 (as procurement_kg.graph_store already uses, porter tokenizer)
for lexical retrieval, and the local BGE-M3 embeddings + Qdrant that the baseline
system uses for dense retrieval.

Isolation from the frozen baseline
----------------------------------
baseline-v1 indexes the older node-level corpus in collection
kg__bge_m3__text_focused_v1 and state/procurement_kg.sqlite3. Nothing here writes to
either. The chunk corpus gets its own SQLite database and its own Qdrant collection so
the baseline remains reproducible and can serve as a genuine ablation arm.

What is indexed
---------------
lexical   text + retrieval_title + retrieval_summary + topics + legal_concepts +
          citation + heading_path + document title, in one FTS5 table with BM25 ranking
dense     `embedding_text` (source text with placeholders stripped and citation/heading
          prefixed) - never the raw text with [[LINK_NNNN]] tokens in it
metadata  every filterable field, so authority/regime/jurisdiction filtering happens in
          SQL rather than being approximated by the vector score
edges     merged structural + resolved reference edges, indexed both directions, so
          graph expansion is a join rather than a scan

Empty chunks are excluded from both indexes: a chunk with no evidence text cannot be
evidence, and returning one would be a retrieval defect.

Usage
-----
    python build_chunk_index.py lexical --corpus-dir data/search_corpus
    .venv-embed/bin/python build_chunk_index.py dense --corpus-dir data/search_corpus
    python build_chunk_index.py stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_VERSION = "1.0.0"
DEFAULT_DB = "state/chunk_index.sqlite3"
DEFAULT_COLLECTION = "chunks__bge_m3__v1"
DEFAULT_EMBED_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    segment_id TEXT,
    chunk_ordinal INTEGER,
    parent_node_id TEXT,
    parent_type TEXT,
    source_kind TEXT,
    authority_class TEXT,
    corpus_role TEXT,
    legal_regime TEXT,
    jurisdiction TEXT,
    citation TEXT,
    heading TEXT,
    heading_path TEXT,
    retrieval_title TEXT,
    retrieval_summary TEXT,
    topics TEXT,
    legal_concepts TEXT,
    procurement_stage TEXT,
    question_intents TEXT,
    text TEXT NOT NULL,
    embedding_text TEXT,
    link_placeholders TEXT,
    source_url TEXT,
    content_sha256 TEXT,
    chunking_method TEXT,
    chunking_model TEXT,
    prompt_version TEXT,
    pipeline_version TEXT,
    char_count INTEGER,
    est_tokens INTEGER,
    superseded_by TEXT,
    filtered_out TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_superseded ON chunks(superseded_by);
CREATE INDEX IF NOT EXISTS idx_chunks_filtered ON chunks(filtered_out);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_chunks_authority ON chunks(authority_class);
CREATE INDEX IF NOT EXISTS idx_chunks_regime ON chunks(legal_regime);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(source_kind);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    retrieval_title,
    retrieval_summary,
    keywords,
    citation,
    tokenize='porter'
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT,
    source_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_id TEXT NOT NULL,
    evidence_method TEXT,
    evidence_text TEXT,
    resolution_status TEXT,
    confidence REAL,
    origin TEXT
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id, relation);
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_key ON edges(source_id, relation, target_id);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT,
    source_url TEXT,
    authority_class TEXT,
    legal_regime TEXT,
    chunk_count INTEGER
);

CREATE TABLE IF NOT EXISTS index_manifest (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def indexable(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evidence-bearing chunks only. An empty chunk cannot support an answer."""
    return [c for c in chunks if (c.get("text") or "").strip()]


def build_lexical(corpus_dir: Path, db_path: Path, root: Path) -> dict[str, Any]:
    chunks = indexable(read_jsonl(corpus_dir / "chunks.jsonl"))
    if not chunks:
        raise SystemExit(f"no chunks under {corpus_dir}")

    # Prefer resolved reference edges; fall back to the v1 file for structural edges.
    v1 = read_jsonl(corpus_dir / "edges.jsonl")
    v2 = read_jsonl(corpus_dir / "edges_v2.jsonl")
    structural = [e for e in v1 if e.get("relation") in {"CONTAINS", "HAS_CHUNK"}]
    edges = [{**e, "origin": "structural_v1"} for e in structural]
    edges += [{**e, "origin": "reference_v2"} for e in v2]
    if not v2:  # resolver not run yet: keep v1 reference edges so the index still works
        edges += [{**e, "origin": "reference_v1"} for e in v1 if e.get("relation") not in {"CONTAINS", "HAS_CHUNK"}]
    # Guidance/commentary citations of legislation. Without these the guidance layer is
    # disconnected from the provisions it explains, and vocabulary gaps (the Act says
    # "cartel", users say "bid rigging") become unbridgeable.
    edges += [{**e, "origin": "guidance_refs"} for e in read_jsonl(corpus_dir / "edges_guidance_refs.jsonl")]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM chunks")
    con.execute("DELETE FROM chunks_fts")
    con.execute("DELETE FROM edges")
    con.execute("DELETE FROM documents")

    rows, fts_rows = [], []
    for c in chunks:
        text = c["text"]
        rows.append(
            (
                c["chunk_id"], c["document_id"], c.get("segment_id"), c.get("chunk_ordinal"),
                c.get("parent_node_id"), c.get("parent_type"), c.get("source_kind"),
                c.get("authority_class"), c.get("corpus_role"), c.get("legal_regime"),
                c.get("jurisdiction"), c.get("citation"), c.get("heading"),
                json.dumps(c.get("heading_path") or []), c.get("retrieval_title"),
                c.get("retrieval_summary"), json.dumps(c.get("topics") or []),
                json.dumps(c.get("legal_concepts") or []), json.dumps(c.get("procurement_stage") or []),
                json.dumps(c.get("question_intents") or []), text, c.get("embedding_text"),
                json.dumps(c.get("link_placeholders") or []), c.get("source_url"),
                c.get("content_sha256"), c.get("chunking_method"), c.get("chunking_model"),
                c.get("prompt_version"), c.get("pipeline_version"), len(text), max(1, len(text) // 4),
                c.get("superseded_by"), c.get("filtered_out"),
            )
        )
        keywords = " ".join(
            [as_text(c.get("topics")), as_text(c.get("legal_concepts")),
             as_text(c.get("procurement_stage")), as_text(c.get("question_intents")),
             as_text(c.get("heading_path"))]
        )
        fts_rows.append((c["chunk_id"], text, c.get("retrieval_title") or "",
                         c.get("retrieval_summary") or "", keywords, c.get("citation") or ""))

    con.executemany(f"INSERT INTO chunks VALUES ({','.join('?' * 33)})", rows)
    con.executemany(
        "INSERT INTO chunks_fts(chunk_id,text,retrieval_title,retrieval_summary,keywords,citation) "
        "VALUES (?,?,?,?,?,?)",
        fts_rows,
    )
    con.executemany(
        "INSERT OR IGNORE INTO edges(edge_id,source_id,relation,target_id,evidence_method,"
        "evidence_text,resolution_status,confidence,origin) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (e.get("edge_id"), e.get("source_id"), e.get("relation"), e.get("target_id"),
             e.get("evidence_method"), e.get("evidence_text"), e.get("resolution_status"),
             e.get("confidence"), e.get("origin"))
            for e in edges
            if e.get("source_id") and e.get("target_id")
        ],
    )
    con.execute(
        "INSERT INTO documents SELECT document_id, MIN(citation), MIN(source_url), "
        "MIN(authority_class), MIN(legal_regime), COUNT(*) FROM chunks GROUP BY document_id"
    )

    manifest = {
        "index_version": INDEX_VERSION,
        "built_at": now_iso(),
        "corpus_dir": str(corpus_dir),
        "chunks_indexed": len(chunks),
        "chunks_sha256": sha256_file(corpus_dir / "chunks.jsonl"),
        "edges_indexed": con.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "fts_tokenizer": "porter",
        "lexical_fields": "text, retrieval_title, retrieval_summary, keywords, citation",
        "python": sys.version.split()[0],
    }
    con.executemany(
        "INSERT OR REPLACE INTO index_manifest(key,value) VALUES (?,?)",
        [(k, json.dumps(v)) for k, v in manifest.items()],
    )
    con.commit()
    con.close()
    return manifest


def build_dense(corpus_dir: Path, db_path: Path, collection: str, model_name: str,
                batch: int, qdrant_url: str) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    chunks = indexable(read_jsonl(corpus_dir / "chunks.jsonl"))
    texts = [(c.get("embedding_text") or c["text"]) for c in chunks]

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    vectors = model.encode(texts, batch_size=batch, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)

    client = QdrantClient(url=qdrant_url, timeout=120)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    points = []
    for i, (c, vec) in enumerate(zip(chunks, vectors)):
        points.append(
            models.PointStruct(
                id=i,
                vector=vec.tolist(),
                payload={
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "authority_class": c.get("authority_class"),
                    "legal_regime": c.get("legal_regime"),
                    "source_kind": c.get("source_kind"),
                    "citation": c.get("citation"),
                    "retrieval_title": c.get("retrieval_title"),
                },
            )
        )
        if len(points) >= 256:
            client.upsert(collection_name=collection, points=points)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points)

    manifest = {
        "dense_built_at": now_iso(),
        "embedding_model": model_name,
        "embedding_dimensions": dim,
        "embedding_input_field": "embedding_text",
        "qdrant_collection": collection,
        "qdrant_url": qdrant_url,
        "vectors": len(chunks),
        "normalized": True,
        "distance": "COSINE",
    }
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT OR REPLACE INTO index_manifest(key,value) VALUES (?,?)",
        [(k, json.dumps(v)) for k, v in manifest.items()],
    )
    con.commit()
    con.close()
    return manifest


def stats(db_path: Path) -> dict[str, Any]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    out: dict[str, Any] = {"db": str(db_path)}
    out["chunks"] = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    out["documents"] = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    out["edges"] = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    out["by_authority"] = {
        r[0]: r[1] for r in con.execute(
            "SELECT authority_class, COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC")
    }
    out["by_regime"] = {
        str(r[0]): r[1] for r in con.execute(
            "SELECT legal_regime, COUNT(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC")
    }
    out["by_relation"] = {
        r[0]: r[1] for r in con.execute("SELECT relation, COUNT(*) FROM edges GROUP BY 1 ORDER BY 2 DESC")
    }
    out["manifest"] = {r[0]: json.loads(r[1]) for r in con.execute("SELECT key,value FROM index_manifest")}
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["lexical", "dense", "all", "stats"])
    ap.add_argument("--corpus-dir", default="data/search_corpus", type=Path)
    ap.add_argument("--db", default=DEFAULT_DB, type=Path)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else root / args.corpus_dir
    db_path = args.db if args.db.is_absolute() else root / args.db

    if args.stage in {"lexical", "all"}:
        print(json.dumps(build_lexical(corpus_dir, db_path, root), indent=2))
    if args.stage in {"dense", "all"}:
        print(json.dumps(build_dense(corpus_dir, db_path, args.collection, args.model,
                                     args.batch, args.qdrant_url), indent=2))
    if args.stage == "stats":
        print(json.dumps(stats(db_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
