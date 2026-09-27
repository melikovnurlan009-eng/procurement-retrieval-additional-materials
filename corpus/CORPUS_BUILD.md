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

The build runs in three stages: acquisition produces a **pre-cleaned** corpus, assessment finds
the chunks that are not fit to retrieve, and repair and filtering produce the **served** corpus.
Each stage's scripts are in `src/`. Chunk counts at each boundary:

```
   sources on the web
        │   acquisition + chunking + ingestion          stage A
        ▼
   22,042 chunks  ·  1,737 documents  ·  19 source domains     ← pre-cleaned
        │   quality assessment, then human verification        stage B
        ▼
   21,521 chunks after repair
        │   filtering                                          stage C
        ▼
   19,087 chunks  ·  1,370 documents                           ← served corpus
```

### Stage A — acquisition, chunking and ingestion (pre-cleaned corpus)

1. **Acquire sources.** `group_a_legislation_scraper_v4.py` fetches the Procurement Act 2023,
   the Procurement Regulations 2024 and the Public Contracts Regulations 2015 as full
   instruments from legislation.gov.uk, parsing the structure rather than scraping rendered
   text. `src/scrapers/` holds the acquisition scripts for the other source families, one per
   publisher; `src/scrapers/README.md` maps each script to its family.
2. **Chunk.** Four chunkers, one per source shape:
   - `chunk_legislation_from_nodes.py` — core instruments, from the parsed structural tree.
   - `chunk_commencement_regs_from_xml.py` — the two commencement SIs (UKSI 2024/716 and
     2024/959) directly from their own source XML, where the structural tree is not available.
   - `chunk_legislation_text.py` — other acquired legislation, from provision text.
   - `chunk_pdf_text.py` — PDF sources, page text in, verified chunks out.
   - `build_search_corpus.py` — HTML guidance, grouped at semantic boundaries from contiguous
     source blocks, which are carried through unmodified.
3. **Ingest.** `ingest_legislation_chunks.py`, `ingest_structural_node_chunks.py` and
   `ingest_pdf_chunks.py` write chunks into the corpus store, preserving legal identity and
   retiring whatever each batch supersedes. `deduplicate_instruments.py` suppresses instruments
   that were ingested twice through different pipelines.

**The output of stage A is the pre-cleaned corpus: 22,042 chunks.** Running the scrapers and
chunkers reproduces this stage, not the served corpus.

### Stage B — quality assessment and human verification

Two criteria decide whether a chunk is fit to retrieve, and they have different remedies, so
they are assessed separately rather than as one "bad" class:

- **Completeness** — does the chunk carry a whole point? A rule severed from its conditions, a
  list without its stem, a sentence cut mid-clause. The remedy is re-segmentation.
- **Cleanliness** — is the chunk evidence at all? Navigation, cookie notices, contents lists,
  contact blocks, boilerplate, extraction garbage. The remedy is filtering.

Three passes, in order:

4. **Structural detectors** — `evaluate_chunk_quality.py`, deterministic, over every chunk with
   its neighbour: `SEVERED_SENTENCE`, `SEVERED_ENUMERATION`, `SEVERED_LIST_STEM`,
   `ORPHAN_LIST_ITEM`, `TOO_SMALL_TO_RETRIEVE`, `EMPTY_TEXT`, plus advisories on size band,
   summary lexical grounding and missing metadata. These catch completeness failures that are
   objectively checkable without judging meaning. 1,085 chunks carried a boundary error.

5. **Model pass** — `label_chunk_quality_llm.py` and the judge tier of
   `evaluate_chunk_quality.py`. The structural detectors cannot see that a chunk is a navigation
   footer, or that an applicability clause has been severed from the rule it governs; this pass
   asks the question they cannot, scoring each chunk on an anchored 1–5 rubric
   (`answer_backbone`, `boundary_correctness`, `semantic_coherence`, `self_containedness`,
   `retrieval_usefulness`, `title_accuracy`, `summary_faithfulness`) and labelling it GOOD,
   INCOMPLETE or LOW_VALUE. It returned 2,886 drop verdicts.

6. **Human verification** — the drop verdicts are one model's opinion, so none was applied on
   its own. Every one was first checked against labels the project already had, and then the
   contested remainder was reviewed by hand:
   - **41** chunks the model wanted dropped are labelled relevant by the benchmark's own gold
     and qrels files. Kept.
   - **17** are targets of legal citation edges; dropping them would break the edges. Kept.
   - **568** items went to hand review under a six-way rubric — KEEP, KEEP_RECHUNK,
     TEMPLATE_ONLY, GRAPH_ONLY, EVAL_ONLY, DISCARD — recorded per chunk in
     `corpus/quality_review/procurement_search_chunk_decisions.json`. Outcome: KEEP 45,
     KEEP_RECHUNK 128, TEMPLATE_ONLY 21, GRAPH_ONLY 4, EVAL_ONLY 44, DISCARD 326. Every DISCARD
     and TEMPLATE decision was re-checked against gold and qrels: **0 collisions**.

   The full working record, including the counts at every step, is
   `corpus/quality_review/CHUNK_QUALITY_WORK_NOTES.md`.

### Stage C — repair, filtering and indexing (served corpus)

7. **Repair.** `rechunk_from_index.py` applies the repair decisions, re-chunking the units the
   assessment flagged. Rule-based repair modified 926 chunks, taking severed sentences from 179
   to 8, severed enumerations from 11 to 0 and severed list stems from 301 to 64. The repaired
   index holds 21,521 chunks.
8. **Filtering.** `content_filters.py` applies per-source page-furniture rules. Nothing is
   deleted: a `filtered_out` column is set, and the retriever's `load()` skips any row that
   carries one, so filtered chunks disappear from every channel while the citation edges, the
   full-text index and the vector collection stay consistent. The decision is reversible with
   one UPDATE.
9. **Resolve references.** `resolve_references.py` turns citation strings inside chunk text
   into graph edges between the chunks they point at.
10. **Densify edges.** `densify_graph_edges.py` re-points edges that resolved to a coarser
    granularity than the corpus was chunked at, so they are actually traversable.
11. **Index.** `build_chunk_index.py` writes `chunk_index.sqlite3`: the `chunks` and `documents`
    tables, the `edges` table, the FTS5 virtual table `chunks_fts`, and the `index_manifest`
    provenance record.

**The output of stage C is the served corpus: 19,087 chunks over 1,370 documents**, which is
what every reported result is computed against.

Dense vectors live outside SQLite, in a Qdrant collection built with BAAI/bge-m3 over the same
chunk set.

## What this repository includes, and what it does not

The reported results are computed against the pinned index identified above. Obtain that index
and verify it by hash; the build code in `src/` is included so the construction is inspectable.

**Stage A reproduces the pre-cleaned corpus, not the served one.** The acquisition scripts in
`src/scrapers/` fetch live pages, and published pages have changed since the corpus was built,
so re-running them acquires today's versions of those sources rather than the versions the
reported results were computed on; some documents are no longer retrievable at their original
URLs at all. Coverage is most of the corpus by document count rather than all of it — the
scripts for a small number of source families, and one intermediate version of the legislation
scraper, are not retained.

**Stage B's hand review is a record, not a computation.** The 568 decisions in
`corpus/quality_review/` were made by the assessor and are shipped so they can be read and
audited per chunk; they are an input to the build, not something a re-run derives.

Together that is why the served corpus is identified by the SHA-256 above rather than by
rebuilding: obtain that index and verify it by hash. The build code is included so every stage
is inspectable, and the quality-review record so the filtering decisions are too.

One consequence of the page-furniture filtering is worth recording for anyone reading the
corpus metadata: Open Government Licence and Crown copyright notices were filtered with the
rest of the footer content, so no per-document licence field survives into the index.
