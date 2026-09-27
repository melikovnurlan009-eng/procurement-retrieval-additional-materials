# Corpus: what it is, how it was built, and how to obtain it

The retrieval corpus is a single SQLite file, `chunk_index.sqlite3`, holding 19,087 chunks
drawn from 1,370 source documents, plus a 23,785-edge citation graph over them. It is the
input to every experiment in this repository.

**The file itself is not in this repository.** It is 165 MB, and it is derived entirely from
public sources listed below. Following the University's guidance that a reproducer can be
assumed to have access to the source data, this repository ships the code that builds the
index, the exact provenance record of the build, and a cryptographic hash so that any rebuilt
or supplied copy can be checked against the one the reported results were computed on.

Almost nothing in this repository needs it. The frozen candidate cache in `data/` already
contains every retrieval score for every benchmark query, so all reported numbers, tables and
figures reproduce without the corpus. See `TECHNICAL_APPENDIX.md` for which two steps do need it.

## Identity of the frozen index

| | |
|---|---|
| File | `chunk_index.sqlite3` |
| Size | 173,146,112 bytes |
| SHA-256 | `7d1f9067719fabcf9680c0a4ce632b14440233633642ab7211ba6b3ac17b2156` |
| Built at | 2026-09-18T10:59:25Z |
| Index version | 1.0.0 |
| Chunks indexed | 19,087 |
| Chunk-payload SHA-256 | `0736d85768a8a8f5770f3663aa95288366daaf57ea40502d173353a46aba38c5` |
| Edges indexed | 23,785 |
| Paired Qdrant collection | `chunks__bge_m3__v2b_sum2` (19,087 points, BAAI/bge-m3) |

Verify a copy before using it:

```bash
shasum -a 256 chunk_index.sqlite3
# expect 7d1f9067719fabcf9680c0a4ce632b14440233633642ab7211ba6b3ac17b2156
export CORPUS_DB=/absolute/path/to/chunk_index.sqlite3
```

The build also writes its own provenance into the index, in a table called `index_manifest`.
`evaluation/corpus_stats.py` reads it back out, so `results/corpus_stats.json` carries the
record even though the database does not travel with the repository.

## Composition

Regenerate every number in this section with:

```bash
python evaluation/corpus_stats.py --db "$CORPUS_DB" --out results/corpus_stats.json
```

**Documents by authority class** (report Figure 1; 1,370 documents total):

| Authority class | Documents | Nominal authority weight |
|---|---:|---:|
| OFFICIAL_WORKFLOW | 662 | 0.78 |
| OFFICIAL_GOVERNMENT_GUIDANCE | 552 | 0.88 |
| PROFESSIONAL_INTERPRETATION | 110 | 0.62 |
| NON_AUTHORITATIVE_PROFESSIONAL | 16 | 0.62 |
| PRIMARY_LEGISLATION | 14 | 1.00 |
| SECONDARY_LEGISLATION | 9 | 0.97 |
| OFFICIAL_REGULATOR_GUIDANCE | 6 | 0.86 |
| OFFICIAL_TECHNICAL_GUIDANCE | 1 | 0.90 |

The two legislation classes (23 documents) form the legislation lane; the remaining 1,347
documents form the other-evidence lane. The 23-against-1,347 imbalance is the reason the
architecture partitions the candidate pool rather than ranking one merged list: in a single
ranking, statute is outnumbered roughly sixty to one by material that echoes a query's own
wording more closely.

**Publishers** (by `source_url` host):

| Host | Documents |
|---|---:|
| gov.uk (other than procurementpathway) | 558 |
| procurementjourney.scot | 450 |
| procurementpathway.civilservice.gov.uk | 212 |
| other publishers | 130 |
| legislation.gov.uk | 20 |

The 130 "other publishers" are professional-commentary sites (procurementportal.com 70,
procurementlawyers.org.uk 43, and a long tail of law-firm and regulator pages), plus three
EUR-Lex documents that carry the EU jurisdiction demotion described in the appendix.

The 20 legislation.gov.uk documents produce 8,170 chunks — 43% of the corpus by chunk count
from 1.5% of it by document count, because statute is chunked at provision granularity.

**Chunking methods:**

| Method | Chunks | What it is |
|---|---:|---|
| LLM_PDF_TEXT_V2 | 7,876 | PDF page text re-chunked, with fidelity verification against the source |
| LLM_LEG_TEXT_V2 | 7,579 | Legislation provision text re-chunked |
| LLM_SEMANTIC_BOUNDARY_V1 | 3,268 | HTML guidance grouped at semantic boundaries |
| STRUCTURAL_NODE_V1 | 364 | Chunks emitted directly from a parsed legislation structural tree |

**Citation graph** (23,785 edges): HAS_CHUNK 9,140, CONTAINS 6,489, REFERENCES 6,065,
CROSS_REFERS_TO 2,091. The frozen configuration has graph expansion **off** — this material
is documented because the graph ablation is reported, not because the adopted system uses it.

## Build order

Each stage is a script in `src/`. The stages run in this order; each depends on the previous.

1. **Acquire legislation** — `group_a_legislation_scraper_v4.py` fetches the Procurement Act
   2023, the Procurement Regulations 2024 and the Public Contracts Regulations 2015 as full
   instruments from legislation.gov.uk, parsing the structure rather than scraping rendered text.
2. **Chunk** — four chunkers, one per source shape:
   - `chunk_legislation_from_nodes.py` — core instruments, from the parsed structural tree.
   - `chunk_commencement_regs_from_xml.py` — the two commencement SIs (UKSI 2024/716 and
     2024/959) directly from their own source XML, where the structural tree is not available.
   - `chunk_legislation_text.py` — other acquired legislation, from provision text.
   - `chunk_pdf_text.py` — PDF sources, page text in, verified chunks out.
   - `build_search_corpus.py` — HTML guidance, grouped at semantic boundaries from contiguous
     source blocks, which are carried through unmodified.
3. **Ingest** — `ingest_legislation_chunks.py`, `ingest_structural_node_chunks.py` and
   `ingest_pdf_chunks.py` write chunks into the corpus store, preserving legal identity and
   retiring whatever each batch supersedes.
4. **Resolve references** — `resolve_references.py` turns citation strings inside chunk text
   into graph edges between the chunks they point at.
5. **Densify edges** — `densify_graph_edges.py` re-points edges that resolved to a coarser
   granularity than the corpus was chunked at, so they are actually traversable.
6. **Index** — `build_chunk_index.py` writes `chunk_index.sqlite3`: the `chunks` and
   `documents` tables, the `edges` table, the FTS5 virtual table `chunks_fts`, and the
   `index_manifest` provenance record.

Dense vectors live outside SQLite, in a Qdrant collection built with BAAI/bge-m3 over the same
chunk set.

## What this repository includes, and what it does not

The reported results are computed against the pinned index identified above. Obtain that index
and verify it by hash; the build code in `src/` is included so the construction is inspectable.

The HTML acquisition layer for the non-legislation sources — site-specific scrapers and the
per-source filters that strip page furniture — is not included. It produces the inputs to stage
2, it is specific to page layouts that have since changed, and no reported result depends on
re-running it. The legislation scraper is included, because it is self-contained and because
the legislation lane is where the report's central architectural claim lives.

One consequence of that filtering layer is worth recording for anyone reading the corpus
metadata: Open Government Licence and Crown copyright notices were treated as page footer
furniture and dropped, so no per-document licence field survives into the index.
