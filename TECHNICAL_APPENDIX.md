# Technical Appendix

Supporting materials for *source-aware two-lane retrieval over UK public procurement law*.

This appendix is written for someone who does not have the report in front of them. It states
what the system does, every parameter it does it with, the exact order of operations that turns
the source data into the reported results, and what each file in this repository is for.

Everything here describes the **frozen configuration that produced the reported results**. Work
that was run during development but does not appear in the report is not documented here, and
the files that produced it are not in this repository. The study's scope, findings and
limitations are the report's own; this document exists to let you reproduce its numbers.

---

## Contents

1. [What the system does](#1-what-the-system-does)
2. [Exact parameter settings](#2-exact-parameter-settings)
3. [Evaluation metrics, defined precisely](#3-evaluation-metrics-defined-precisely)
4. [End-to-end workflow](#4-end-to-end-workflow)
5. [Every file in this repository](#5-every-file-in-this-repository)
6. [Where each reported number comes from](#6-where-each-reported-number-comes-from)
7. [Environment](#7-environment)

---

## 1. What the system does

The task is evidence retrieval for UK public procurement questions. A practitioner asks
something like *"can we award this contract directly without running a competition?"*, and the
system must return the passages a defensible answer has to rest on.

That is not ordinary passage retrieval, for a structural reason. A complete answer needs
**both** the controlling statutory rule and the official guidance that operationalises it — and
those two live in wildly unequal populations. The corpus holds 23 legislation documents against
1,347 non-legislation documents. Guidance echoes a practitioner's own wording; statute does not.
In a single merged ranking, statute is outnumbered roughly sixty to one by material that *looks*
more relevant to a lexical or dense scorer, and it is pushed off the result list even when it is
the thing the question turns on.

The system's response is **lane separation**. The candidate pool is partitioned by source type
before ranking, and each lane is ranked and cut independently:

- **Legislation lane** — `PRIMARY_LEGISLATION` and `SECONDARY_LEGISLATION` (23 documents).
- **Other-evidence lane** — everything else (1,347 documents).

The two lanes never compete. Each retrieves 300 candidates per channel, each is reranked to
depth 75, and each contributes its own top 25 to the final output of 50. At evaluation the two
lanes are OR-pooled: a requirement is satisfied if any acceptable chunk appears within either
lane's cutoff. A requirement whose evidence is statutory cannot be starved by guidance winning
on surface similarity, because guidance is not in its lane.

Within each lane the pipeline is:

```
query
  ├── lexical channel: SQLite FTS5 bm25() over 5 indexed fields  → 300 candidates
  └── dense channel:   BAAI/bge-m3 cosine via Qdrant             → 300 candidates
                                   │
            per-query normalisation (tanh of z-score, over the query's own pool)
                                   │
            fused    = 0.40·bm25_norm + 0.60·dense_norm
            blended  = 0.90·fused + 0.10·authority_norm
            final    = blended · jurisdiction_weight
                                   │
            cross-encoder rerank, BAAI/bge-reranker-v2-m3, top 75
                                   │
                            top 25 per lane
```

Two source-aware signals act inside that scoring. **Authority weighting** gives each chunk a
prior by publisher class (statute above official guidance above professional commentary).
**Jurisdiction weighting** multiplicatively demotes EU-sourced material to 0.35, because
pre-Brexit EU directives would otherwise outrank domestic law on domestic questions — measured
during development, an EU directive ranked first on a contract-splitting query and inverted the
Procurement Act's anti-avoidance position. It is a demotion rather than a filter so the material
stays reachable when a query is genuinely about it.

---

## 2. Exact parameter settings

Every value below is read from the code in `src/` and the freeze record in
`config/frozen_config.json`. Nothing here is a reconstruction from memory.

### 2.1 Frozen retrieval configuration

| Parameter | Value | Where |
|---|---|---|
| `alpha` (authority blend share) | **0.10** | `config/frozen_config.json` |
| `beta` (BM25 share of fusion) | **0.40** | `config/frozen_config.json` |
| Dense share of fusion | 0.60 (= 1 − beta) | `src/chunk_retrieval.py:433` |
| Graph expansion | **off** | `config/frozen_config.json` |
| Jurisdiction weighting | **on** | `config/frozen_config.json` |
| Keyword expansion | **off** | `config/frozen_config.json` |
| Candidates per channel per lane | 300 | `config/frozen_cache_manifest.json` |
| Cross-encoder rerank depth | 75 per lane | `config/frozen_config.json` |
| Final retention | 25 per lane (50 total) | evaluation cutoff |
| Random seed | 20260920 | `config/frozen_config.json` |

`src/chunk_retrieval.py` carries its own class-level defaults of `BM25_WEIGHT = 0.40`,
`DENSE_WEIGHT = 0.60` and `AUTH_BLEND = 0.30`. The frozen alpha of 0.10 **overrides** that 0.30
default; the analysis harness sets it per-instance and `evaluation/common.py:assert_active_retriever()`
fails loudly if the module's own defaults ever drift from what the harness expects.

**Why alpha is 0.10 and not 0.30.** An earlier freeze used 0.30. On the stratified split the
alpha sweep reverses: requirement recall falls monotonically as alpha rises (0.565 at 0.00,
0.513 at 0.10, 0.469 at 0.20, 0.455 at 0.30), consistently across the whole beta grid. The
aggregate optimum is alpha = 0.00, but 0.00 lets professional commentary occupy 1.49 of the
other lane's top 5; at 0.10 that collapses to 0.12 while recall stays near the optimum. 0.10 is
the resolution of that trade-off, chosen on DEV alone. The full reasoning, in the words it was
frozen with, is in `config/frozen_config.json` under `alpha_dev_justification`.

### 2.2 Scoring formulas

Per-query normalisation, applied to raw BM25 and raw dense scores independently
(`src/chunk_retrieval.py:_normalize`):

```
norm(v) = (tanh((v − mean) / std) + 1) / 2
```

where `mean` and `std` are computed over **that query's own candidate pool**, not over the
corpus. The statistics can be frozen to a reference sub-pool (`stats_from`) so that adding
graph-expansion candidates cannot silently move an unrelated candidate's score — measured
during development, adding 5 graph candidates to a 90-item pool shifted an already-present gold
chunk's normalised BM25 by +63% with no change to its raw score.

```
fused    = 0.40 · bm25_norm + 0.60 · dense_norm
blended  = fused · (1 − alpha) + authority_norm · alpha        # alpha = 0.10
final    = blended · jurisdiction_weight
```

`authority_norm` is the nominal authority weight put through the *same* per-query normalisation,
so authority competes on a scale-free footing rather than as a raw additive bonus.

`verify_reported_numbers.py` reconstructs `final_score` from the cached raw values through these
formulas for all 105,823 DEV candidate rows. **Maximum absolute reconstruction error: 0.0.**

### 2.3 Authority weights

`src/chunk_retrieval.py:59`. Classes present in the corpus are marked ●.

| Authority class | Weight | | Authority class | Weight |
|---|---:|---|---|---:|
| ● PRIMARY_LEGISLATION | 1.00 | | PROCUREMENT_POLICY | 0.84 |
| ● SECONDARY_LEGISLATION | 0.97 | | OFFICIAL_PRACTICE_GUIDANCE | 0.84 |
| ● OFFICIAL_TECHNICAL_GUIDANCE | 0.90 | | OFFICIAL_TRAINING | 0.74 |
| OFFICIAL_PA23_TECHNICAL_GUIDANCE | 0.90 | | ● PROFESSIONAL_INTERPRETATION | 0.62 |
| ● OFFICIAL_GOVERNMENT_GUIDANCE | 0.88 | | PROFESSIONAL_CASE_ANALYSIS | 0.62 |
| ● OFFICIAL_REGULATOR_GUIDANCE | 0.86 | | ● NON_AUTHORITATIVE_PROFESSIONAL | 0.62 |
| OFFICIAL_SPECIALIST_GUIDANCE | 0.86 | | INDUSTRY_PRACTICE | 0.55 |
| ● OFFICIAL_WORKFLOW | 0.78 | | *(unmapped default)* | 0.50 |

Jurisdiction: `EU → 0.35`, everything else `1.00` (`src/chunk_retrieval.py:136`).

**A finding worth flagging, because it is counter-intuitive.** The nominal gap between primary
(1.00) and secondary (0.97) legislation is 0.03. After per-query normalisation the *mean*
realised gap is 0.751 − 0.038 = 0.713 — an amplification of **23.76×**. Per-query z-scoring is
not weight-preserving: because primary legislation is nearly always at the top of its pool's
authority distribution and secondary nearly always below the mean, a nominally negligible
distinction becomes a large one in practice. Reproduce it with
`evaluation/score_signal_analysis.py`; it is report Figure 5.

### 2.4 BM25 / lexical channel

The lexical channel is SQLite FTS5's built-in `bm25()` ranking function.

| Setting | Value | Note |
|---|---|---|
| k1 | **1.2** | SQLite's built-in default. **Not overridable** — FTS5 hardcodes it; there is no pragma or compile option, and the code does not pass column weights either. |
| b | **0.75** | Same: SQLite's hardcoded default. |
| Tokenizer | `porter` | Porter stemmer over the default unicode61 tokenizer. Set at index creation. |
| Indexed fields | 5, equally weighted | `text`, `retrieval_title`, `retrieval_summary`, `keywords`, `citation` |
| Index-side stopwords | none | Everything is indexed. |
| Query-side stopwords | 48 terms | `src/chunk_retrieval.py:174` |

The k1 and b values are stated because a report must state them, not because they were chosen:
they are SQLite's defaults and the implementation cannot change them. Anyone tuning BM25 on this
corpus would have to replace FTS5's ranking function.

The 48-term query-side stopword list exists because FTS5 MATCH is an OR query: measured on this
corpus, `"for"` matched 588 of 743 chunks and `"can"` 236, while the terms that actually
discriminate — `"cartel"` (26), `"rigging"` (18) — were swamped. IDF does not rescue this,
because every chunk still enters the candidate set.

Two lexical fallbacks, both gated on sparsity:

- **Prefix expansion** — a term widens to a prefix match only if its exact form matches fewer
  than `PREFIX_MIN_EXACT_DOCS = 5` documents. Common words never widen.
- **Fuzzy variants** — RapidFuzz against the FTS vocabulary, minimum term length 4, score cutoff
  72, at most 3 variants per term. This catches misspellings without letting a
  barely-over-threshold guess dominate.

### 2.5 Dense channel

| Setting | Value |
|---|---|
| Model | `BAAI/bge-m3` |
| Store | Qdrant, collection `chunks__bge_m3__v2b_sum2` |
| Points | 19,087 (one per chunk) |
| Similarity | cosine |

### 2.6 Cross-encoder reranker

| Setting | Value |
|---|---|
| Model | `BAAI/bge-reranker-v2-m3` |
| `max_length` | **512** |
| Batch size | 32 |
| Device | MPS where available, else CPU |
| Depth | top 75 per lane |
| Score transform | sigmoid over the raw logit |
| Placement | replaces the first-stage order within the reranked head; positions below 75 keep their first-stage order |

**The 512-token limit is a deliberate choice, not a model limit.** `bge-reranker-v2-m3` reports
`max_position_embeddings = 8194` and supports inputs up to 8192 tokens. 512 was set for
throughput, and its effect is measured in two places. Corpus-wide, **14.2% of raw chunk texts
exceed 512 tokens** (mean 305.1, P95 797, max 10,831), while the metadata preamble (mean 103.1,
max 380) and the retrieval summary (mean 71.7, max 351) stay well inside it. Among the 48
essential-gold chunks the reranker left below rank 25 on DEV, 23 exceed the window. Both
measurements are in `results/token_distribution/` and `results/truncation_diagnostic.csv`,
reproduced by part 10.

The baseline reranker sees the chunk's **raw text only**. Variant 2 in the cross-encoder
experiment instead presents
`citation. retrieval_title. retrieval_summary. authority_class. legal_regime. <raw text>`.
No gold feature appears in either input: every field is corpus metadata available at real
inference time.

---

## 3. Evaluation metrics, defined precisely

The authoritative implementation of gold-target extraction is
`evaluation/ce_metrics_audit.py:mandatory_requirements_with_targets()`. All four metrics below
use it.

Let a scenario have mandatory requirements *R*, each with an acceptable target set *T(r)* of
chunk ids. Let *L* and *O* be the legislation-lane and other-lane ranked lists. A requirement is
**satisfied at k-per-lane** when `T(r)` intersects `L[:k] ∪ O[:k]` — the two lanes are OR-pooled,
never merged into one ranking.

| Metric | Definition |
|---|---|
| **RequirementRecall@k-per-lane** | (satisfied requirements) ÷ (total mandatory requirements), summed across scenarios. **Requirement-weighted, not a mean of per-scenario ratios.** |
| **CompleteCoverage@k-per-lane** | Fraction of scenarios where *every* mandatory requirement is satisfied. |
| **DualEvidenceCoverage@k-per-lane** | On mixed-evidence scenarios only: 1 when **all** mandatory legislation-role requirements **and all** mandatory non-legislation-role requirements are satisfied. Strict — one miss on either side scores 0. |
| **CandidateRequirementRecall@75** | The same as RequirementRecall but measured over the full reranked candidate pool. This is the **ceiling** the reranker is working against: 0.916 on DEV (142/155). |

Role sets for DualEvidenceCoverage:

- Legislation: `controlling_rule`, `implementing_detail`
- Non-legislation: `explanatory_guidance`, `procedural_guidance`, `workflow_instruction`,
  `policy_rule`, `regulator_interpretation`, `transition_rule`

**Aggregation matters, and is stated because it changes the answer.** Requirement-weighted
recall on DEV is 0.813; averaging per-scenario ratios instead gives 0.833. The two diverge by up
to 11 percentage points elsewhere in the analysis. Every reported figure is
requirement-weighted, per the definition above.

### Two evaluator corrections

`mandatory_requirements_with_targets()` supersedes the original extractor,
`evaluation/common.py:essential_targets()`, in two respects. Both came out of a from-scratch
metric audit:

1. **`acceptable_chunk_ids` were dropped.** These are alternate chunk representations of the
   same citation. 191 of 402 essential-evidence items carry at least one; 145 valid target
   chunks were being ignored, so retrieving the right provision through a different chunking
   path counted as a miss.
2. **The `mandatory` flag was not applied.** Nine requirements marked `mandatory: false` were
   being scored as if a complete answer required them. Where the key is absent (older records
   predating the distinction) the requirement is treated as mandatory.

Both corrections are gated on the same item's `resolution.status` being `MATCHED` or
`FUZZY_MATCHED`; an unresolved item contributes nothing.

`common.py:essential_targets()` remains in the repository because scripts written before the
audit import it. Section 6 names which reported number comes from which extractor.

### Statistical procedures

| Procedure | Setting |
|---|---|
| Confidence intervals | Cluster bootstrap, resampling **scenarios** (not requirements), 10,000 resamples, seed 20260920 |
| Significance | Paired sign-flip permutation, 10,000 permutations |
| Effect size | Paired Cohen's *d*, computed per scenario |
| Binary per-scenario metrics | Exact McNemar |

Scenarios are the resampling unit because requirements within a scenario are not independent.

---

## 4. End-to-end workflow

There are two routes to the reported results. **Path A** rebuilds everything from the corpus.
**Path B** verifies every reported number from the artifacts shipped here, needs no GPU, no
Qdrant and no corpus database, and takes about a minute.

Path B is the one to run first: it establishes that the shipped artifacts are the ones the
report was written from, before any question of rebuilding arises.

### Path B — reproduce the reported results

The report is not one result, so reproduction is not one script. It is **twelve parts**, each
corresponding to a section, table or figure. A part re-runs the scripts that produce its own
outputs and then checks those outputs against the values the report prints. **A part passes
only when every one of its checks passes** — a script exiting cleanly is not treated as success.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python evaluation/reproduce.py --list          # what the parts are
python evaluation/reproduce.py --all           # run all twelve
python evaluation/reproduce.py rq1 reranking   # run named parts
python evaluation/reproduce.py 3 8             # or by number
```

`bash scripts/reproduce.sh` is a wrapper over the same driver and takes the same arguments.

| # | Part | Reproduces | Scripts it runs |
|---|---|---|---|
| 1 | `corpus` | Section 3.1, Figure 1 | `corpus_stats.py` |
| 2 | `artifacts` | Sections 4 and 5.3, Appendix A | *(none — checks only)* |
| 3 | `rq1` | Section 6.1, Tables 3–4, Figures 4–5 | `rq1_rq3_same_budget.py` |
| 4 | `signals` | Sections 6.1.1, 6.2.1–6.2.2, Figures 6–7 | `score_signal_analysis.py` |
| 5 | `graph` | Section 6.2.3 | `graph_edge_audit.py`, `make_benchmark150.py`, `rq1_ablations.py` (DEV and TEST) |
| 6 | `regime` | Section 6.2.2 | `make_benchmark150.py`, `rq3_regime_constraint.py` |
| 7 | `reranking` | Section 6.3, Table 7, Figure 8 | `ce_experiment_metrics.py`, `ce_experiment_metrics2.py`, `dual_evidence_strict.py`, `test_confirmation.py`, `ce_variants_test.py` |
| 8 | `performance` | Section 6.4, Tables 6 and 8, Figure 9 | `final_performance.py`, `ce_metrics_audit.py` |
| 9 | `ir-metrics` | Appendix A, Figures 10–11 | `ir_metrics.py` |
| 10 | `tokens` | Appendix A, Section 7.3 | `corpus_token_distribution.py` |
| 11 | `figures` | Figures 1 and 4–11 | `make_report_figures.py`, `score_signal_figures.py` |
| 12 | `cross-check` | every headline number, independently | `verify_reported_numbers.py` |

Each part prints four things: the inputs it **reads**, the scripts it **runs**, the artifacts
it **writes**, and a line per **check** giving the reported value, the recomputed value and a
verdict. For every artifact written, the part compares the regenerated bytes against the copy
shipped in the repository and reports `identical to shipped`, `created`, or
`DIFFERS from shipped`. That comparison is the reproducibility statement — the part rebuilt the
artifact from its inputs and got the same file back — and it is what distinguishes a genuine
regeneration from a check against a stale file.

`python evaluation/reproduce.py --chain` prints what every part reads and writes without
running anything, so the flow from benchmark and cache through to tables and figures can be
read first.

A final summary gives one row per part plus the artifact tally, and every check is written to
`results/reproduction_report.csv`. Exit status is 0 only if every selected part passed.

**Current state: 12/12 parts, 281/281 checks, 48 artifacts regenerated and all 48 identical
to the shipped copies.**

```
  1  corpus         Corpus composition                           10/10  PASS
  2  artifacts      Frozen artifact integrity and provenance     23/23  PASS
  3  rq1            RQ1: matched-budget comparison               71/71  PASS
  4  signals        Signal behaviour and authority calibration   15/15  PASS
  5  graph          RQ2: graph edge quality and retrieval effect  19/19  PASS
  6  regime         RQ2: regime-compatibility constraint           8/8  PASS
  7  reranking      RQ3: cross-encoder variants and completeness  34/34  PASS
  8  performance    Final fixed-configuration performance         24/24  PASS
  9  ir-metrics     Secondary IR diagnostics                      11/11  PASS
 10  tokens         Token distributions and the 512-token window  13/13  PASS
 11  figures        Figure regeneration                           11/11  PASS
 12  cross-check    Independent cross-check of headline numbers   34/34  PASS
  ------------------------------------------------------------------------
  12/12 parts reproduced   281/281 checks passed
  48 artifacts regenerated under results/ and figures/ — 48 identical to the shipped copy
```

**Why parts 2 and 12 exist.** Part 2 checks nothing about retrieval quality; it establishes
that the inputs are the ones the frozen run used — the gold file's SHA-256 against the frozen
manifest, the frozen parameters against what the report states, and that every cached
`final_score` reconstructs through the documented formula. Every other number is conditional
on that. Part 12 then recomputes 32 reported numbers from the benchmark and the candidate
cache **directly**, reading no other part's output, so a bug in a producing script cannot hide
behind its own output.

**Independence.** Each part regenerates its own artifacts before checking them, so a part run
alone still means something. Part 11 is the one exception, and says so: several figures plot a
table another part owns, so it wants those parts first, or `--all`.

**The corpus database.** Only parts 1 and 10 need it. With `CORPUS_DB` set they regenerate
their numbers from the index; without it they check the shipped outputs without regenerating
them, marking the step skipped while still running every check.

### Path A — rebuild from source

This is the full chain that produced the artifacts in `data/`.

**Prerequisites**

```bash
pip install -r requirements.txt -r requirements-pipeline.txt
export CORPUS_DB=/path/to/chunk_index.sqlite3     # see corpus/CORPUS_BUILD.md
# plus a running Qdrant holding collection chunks__bge_m3__v2b_sum2
```

**Stage 0 — build the corpus.** Six steps in `src/`, documented in `corpus/CORPUS_BUILD.md`.
The reported results are computed against one pinned index, whose SHA-256 is recorded there.
Obtain that index and verify it by hash rather than rebuilding.

**Stage 1 — first-stage retrieval, once.**

```bash
python evaluation/build_candidate_cache.py \
    --candidates 300 --use-graph 0 --alpha 0.10 --beta 0.40 \
    --db "$CORPUS_DB" --cache data/candidate_cache.parquet
```

This is the only real retrieval pass in the whole project. It runs both channels for all 208
scenarios in both lanes and writes one row per `(scenario, lane, candidate)` — 105,823 DEV rows
and the TEST equivalent — carrying the **raw** BM25, dense and authority values alongside the
normalised and fused scores. Because the raw values are cached, every later analysis can
recompute scores at a different alpha or beta through the retriever's own `_fuse()` and
`_authority_norm()` without re-embedding anything. That is what makes the same-budget ablations
runnable on a laptop.

**Stage 2 — cross-encoder inference.**

```bash
python evaluation/run_ce_rerank.py \
    --cache data/candidate_cache.parquet \
    --scenarios benchmark/scenarios_all_208.jsonl --db "$CORPUS_DB" \
    --out data/reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json
```

Needs the corpus database, because chunk text is deliberately not stored in the parquet.

**Stage 3 — merge the reranker's scores into the cache.**

```bash
python evaluation/merge_ce_into_cache.py \
    --cache data/candidate_cache.parquet \
    --ce-output data/reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json
```

After this merge the cache is self-contained: `ce_score` and `ce_rank` are columns in the
parquet, and nothing downstream touches the reranker, the corpus or a GPU again. **This is the
boundary between Path A and Path B.**

**Stage 4 — the cross-encoder input variant** (for the variant comparison table only):

```bash
python evaluation/ce_experiment_variant2_rerank.py --split DEV  --db "$CORPUS_DB"
python evaluation/ce_experiment_variant2_rerank.py --split TEST --db "$CORPUS_DB"
```

**Stage 5 onwards** is exactly Path B above: `python evaluation/reproduce.py --all`.

### Order dependencies

```
corpus build ──► build_candidate_cache ──► run_ce_rerank ──► merge_ce_into_cache
                                                                    │
                                    ┌───────────────────────────────┴──────────────┐
                                    ▼                                              ▼
                       ce_experiment_variant2_rerank                 verify_reported_numbers
                                    │                                ce_metrics_audit
                                    ▼                                dual_evidence_strict
                          ce_experiment_metrics                      score_signal_analysis
                                    │                                rq1_rq3_same_budget
                                    ▼                                make_report_figures
                          ce_experiment_metrics2                     score_signal_figures
```

Inside Path B there are two ordering constraints, and `evaluation/reproduce.py` encodes both:
`ce_experiment_metrics.py` must run before `ce_experiment_metrics2.py`, which reads
`summaries_v1_v3.json` from it; and `make_report_figures.py` plots `corpus_stats.json` and
Table 5, so it wants parts 1 and 3 first. Everything else is independent and can run in any
order, which is what makes the parts individually meaningful.

---

## 5. Every file in this repository

### Root

| File | What it is |
|---|---|
| `README.md` | Entry point: what this is and the shortest path to verifying it. |
| `TECHNICAL_APPENDIX.md` | This document. |
| `requirements.txt` | Pinned packages for Path B. Enough to reproduce every reported number. |
| `requirements-pipeline.txt` | Additional packages for Path A: reranker, embeddings, Qdrant, corpus build. |
| `.gitignore` | Excludes the corpus index (not distributed here), local virtualenvs and caches, and the per-run manifests the harness writes into `results/`. The frozen manifests live in `config/`. |

### `benchmark/` — the evaluation set

| File | What it is |
|---|---|
| `scenarios_all_208.jsonl` | 208 scenarios: query, context, suite, split. |
| `gold_evidence_218.jsonl` | Requirement-level gold evidence, resolved to corpus chunk ids. 218 records covering the 208 benchmark scenarios plus ten `EXP_GRAPH*` records from the frozen run; the file is shipped unmodified, so it still hashes to the value in `config/frozen_cache_manifest.json`. Every script joins on `scenario_id`, so the ten extra records are never read. Part 2 checks this. |
| `graph_edge_review/edge_classifications.csv` | The 122 sampled edges behind Section 6.2.3, each with its relation, stratum and classification. |
| `graph_edge_review/review_items.jsonl` | The same 122 edges with their endpoints and the underlying provision text. |
| `derived/` | Written by `make_benchmark150.py`, not shipped: the earlier 150-scenario view of the benchmark. |
| `BENCHMARK.md` | Field-by-field schema, composition, split protocol, construction method, and one recorded defect. |

### `config/` — the freeze record

| File | What it is |
|---|---|
| `frozen_config.json` | **The audit trail for "no retuning on TEST".** Every tunable — alpha, beta, graph on/off, jurisdiction, lane budget, rerank depth, the RQ3 threshold — with the DEV-only evidence that justified it, written at freeze time before TEST was ever scored. Paths inside it refer to the machine the freeze ran on and are kept verbatim as provenance. |
| `frozen_cache_manifest.json` | Provenance of the candidate cache: git commit, SHA-256 of the corpus index and both benchmark files, the Qdrant collection, every retrieval parameter, and corpus counts. |

### `corpus/` — the corpus, by description

| File | What it is |
|---|---|
| `CORPUS_BUILD.md` | What the corpus contains, the three build stages from pre-cleaned to served, its SHA-256, and how to obtain and verify a copy. The 165 MB index itself is not shipped. |
| `quality_review/CHUNK_QUALITY_WORK_NOTES.md` | The working record of the quality assessment: what each pass detected, the counts at every step, and how each decision was applied to the index. |
| `quality_review/review_needed.json` | The 568 chunks the assessment could not settle automatically, sent to hand review. |
| `quality_review/procurement_search_chunk_decisions.json` | The hand-review outcome, per chunk, under the six-way rubric, with the rubric itself and the summary counts. |
| `quality_review/decisions_applied.json` | The log of how those decisions were applied to the live index. |
| `quality_review/keep_rechunk_leftovers.json` | The KEEP_RECHUNK items still outstanding when the corpus was frozen. |

### `src/` — the system

The retriever and the corpus-construction chain. These are the code under test, not analysis code.

| File | What it is |
|---|---|
| `chunk_retrieval.py` | **The retriever.** Two-lane search, FTS5 lexical channel, dense channel, per-query normalisation, fusion, authority and jurisdiction weighting, optional graph expansion. Every scoring formula in §2.2 lives here. |
| `group_a_legislation_scraper_v4.py` | Acquires the Procurement Act 2023, Procurement Regulations 2024 and PCR 2015 from legislation.gov.uk as full parsed instruments. The version the corpus was built with. |
| `scrapers/` | The source-acquisition scripts for every other source family, one per publisher, with `scrapers/README.md` listing which script covers which family. They fetch live pages, so a re-run acquires today's versions of those sources rather than the versions the reported results were computed on; the corpus is identified by hash instead. Included so the acquisition is inspectable. |
| `chunk_legislation_from_nodes.py` | Chunks core instruments from their parsed structural tree. |
| `chunk_commencement_regs_from_xml.py` | Chunks the two commencement SIs from source XML, where no structural tree exists. |
| `chunk_legislation_text.py` | Chunks other acquired legislation from provision text. |
| `chunk_pdf_text.py` | Chunks PDF sources, with a fidelity check that the chunks reproduce the source text. |
| `build_search_corpus.py` | Semantic chunker for HTML guidance: groups contiguous source blocks into retrievable chunks and emits their retrieval metadata. |
| `ingest_legislation_chunks.py` | Ingests re-chunked legislation, preserving legal identity. |
| `ingest_structural_node_chunks.py` | Ingests chunks built from a structural tree. |
| `ingest_pdf_chunks.py` | Ingests re-chunked PDF content, retiring what it supersedes. |
| `deduplicate_instruments.py` | Suppresses duplicate ingestions of the same legal instrument, where two pipelines derived different document ids for it. |
| `content_filters.py` | Per-source content filters: keeps the substantive parts of a page and drops navigation, headers and footers. |
| `evaluate_chunk_quality.py` | Chunk quality assessment. Deterministic tier: severed sentences, broken enumerations, orphaned list stems, empty or too-small chunks, plus advisories on size and metadata. Judge tier: an anchored 1-5 rubric over `answer_backbone`, `boundary_correctness`, `semantic_coherence`, `self_containedness`, `retrieval_usefulness`, `title_accuracy` and `summary_faithfulness`. |
| `label_chunk_quality_llm.py` | The judge tier's batch labeller, classifying each chunk GOOD, INCOMPLETE or LOW_VALUE — the completeness and cleanliness criteria that decide whether a chunk is re-segmented or filtered. |
| `rechunk_from_index.py` | Applies the repair decisions back to the index, re-chunking the units the assessment flagged. |
| `resolve_references.py` | Turns citation strings inside chunk text into graph edges. |
| `densify_graph_edges.py` | Re-points edges that resolved coarser than the corpus was chunked, making them traversable. |
| `build_chunk_index.py` | Builds `chunk_index.sqlite3`: chunks, documents, edges, the FTS5 table, and the `index_manifest` provenance record. |

### `evaluation/` — the harness

**Infrastructure**

| File | What it is |
|---|---|
| `paths.py` | Every path in the package, resolved relative to the repository root and overridable by environment variable. Why the package runs from any checkout without editing. |
| `config.py` | `AnalysisConfig` and the shared CLI (`--scenarios/--gold/--split/--alpha/--beta/--cache/...`), plus manifest writing. Scripts take configuration from here rather than hardcoding it. |
| `common.py` | Shared loaders, metric functions, and the weight-sweep machinery. Reuses the retriever's *own* `_normalize`/`_fuse`/`_authority_norm` on a temporarily-overridden instance, so a sweep never reimplements the maths and never mutates module-level defaults. Also holds `assert_active_retriever()`. |
| `__init__.py` | Marks the package. Empty. |

**Path A — building the artifacts** (needs corpus, Qdrant, GPU)

| File | What it is |
|---|---|
| `build_candidate_cache.py` | Stage 1. The one real retrieval pass; writes `data/candidate_cache.parquet`. |
| `run_ce_rerank.py` | Stage 2. Cross-encoder inference at depth 75 per lane. |
| `merge_ce_into_cache.py` | Stage 3. Merges reranker scores into the cache, making it self-contained. |
| `ce_experiment_variant2_rerank.py` | Stage 4. Re-runs the reranker with the metadata-enriched input for the variant comparison. |

**Path B — analysis from shipped artifacts** (no corpus, no GPU)

| File | What it is |
|---|---|
| `reproduce.py` | **Start here.** The part-wise reproduction driver: defines the nine parts, runs each part's scripts, checks each part's outputs against the report, and writes `results/reproduction_report.csv`. `--list`, `--all`, or part names/numbers. |
| `verify_reported_numbers.py` | Part 9. Recomputes 32 reported numbers from the benchmark and the candidate cache directly, reading no other part's output, and compares each to the report. Writes `results/verification_report.csv`. |
| `final_performance.py` | Part 3. Computes Table 5 / Figure 7 for both splits. The single producer of `results/top25_per_lane_final_metrics.csv`; `make_report_figures.py` plots that file rather than recomputing it, so the table and the figure cannot diverge. |
| `ce_metrics_audit.py` | The from-scratch metric recomputation, and home of the authoritative evaluator `mandatory_requirements_with_targets()` plus the lane-ranking helpers most other scripts import. Produces the final performance table and the full audit table. |
| `dual_evidence_strict.py` | Strict DualEvidenceCoverage at three cutoffs for four reranker variants, with per-scenario diagnosis of which side failed. |
| `score_signal_analysis.py` | Signal-by-signal behaviour: ROC AUCs, effective ranges, saturation, authority amplification, jurisdiction effects, reranker rank movement, and the exact score reconstruction check. |
| `ir_metrics.py` | Appendix A's secondary diagnostics: cumulative RequirementRecall@k, Precision@k and NDCG@k at k = 10…75, per lane and OR-pooled, before and after reranking. Produces Figures 10 and 11's data. |
| `graph_edge_audit.py` | Section 6.2.3's stratified edge-quality audit: aggregates the 122 classified edges into strict and permissive rates, overall, per stratum and per relation. |
| `rq1_ablations.py` | The nine-configuration ablation, including the graph-on against graph-off comparison in Section 6.2.3. |
| `rq3_regime_constraint.py` | Section 6.2.2's exploratory regime-compatibility constraint. Refuses to run on TEST without a frozen config, so it cannot be tuned there. |
| `category_boost.py` | The query-category classifier and boost rule that `test_confirmation.py` evaluates in Section 6.3.3. Imported, not run directly. |
| `ce_variants_test.py` | The cross-encoder variants re-measured on TEST, so the DEV selection can be checked against a split nothing was tuned on. |
| `test_confirmation.py` | Section 6.3.1 and 6.3.3's TEST-side confirmation: authority-alpha sensitivity, cross-encoder diagnostics, and the category-boost reflection against an oracle. Reporting only; nothing is selected from TEST. |
| `make_benchmark150.py` | Materialises the earlier 150-scenario benchmark view from the shipped 208-scenario file, for the two experiments that used it. |
| `rq1_rq3_same_budget.py` | The same-budget ablation: six conventional pooled configurations at top-50 against three two-lane configurations at 25+25, so no comparison is confounded by budget. Adds bootstrap CIs, permutation p-values, Cohen's *d* and McNemar. |
| `ce_experiment_metrics.py` | Cross-encoder variants 1 and 3. Writes `summaries_v1_v3.json`. **Run before `ce_experiment_metrics2.py`.** |
| `ce_experiment_metrics2.py` | Variants 2, 4 and the truncation diagnostic, then the variant comparison table. |
| `corpus_stats.py` | Corpus composition from the index. Needs the corpus database. |
| `corpus_token_distribution.py` | Appendix A's chunk token-count distributions for the metadata preamble, the retrieval summary and the raw text. Needs the corpus database. |
| `make_report_figures.py` | Regenerates the report's data-driven figures. |
| `score_signal_figures.py` | Four signal-diagnostic panels supporting the score analysis. |

### `data/` — frozen artifacts (40 MB)

| File | What it is |
|---|---|
| `candidate_cache.parquet` | **The single most important file here.** One row per (scenario, lane, candidate) for all 208 scenarios: raw and normalised BM25, dense and authority values, jurisdiction weight, fused/blended/final scores, first-stage rank, merged cross-encoder score and rank, and gold flags. Everything in Path B reads this. |
| `reranked_top75_output_COMBINED218_bge-reranker-v2-m3.json` | Raw baseline reranker output, top 75 per lane per scenario, with pre- and post-rerank positions. |
| `variant2_ce_output_DEV.json`, `variant2_ce_output_TEST.json` | Metadata-enriched reranker output, same schema. |
| `benchmark150/candidate_cache_graph_off.parquet` | The frozen graph-off cache over the earlier 150-scenario benchmark. Needed by Sections 6.2.2 and 6.2.3. |
| `benchmark150/candidate_cache_graph_on.parquet` | The same retrieval pass with graph expansion **on**. The pair is what makes the graph comparison in Section 6.2.3 a real measurement rather than an estimate. |

### `results/` — generated outputs

Every file here is regenerated by a script in `evaluation/`; none is hand-made.

| File | Produced by | What it holds |
|---|---|---|
| `reproduction_report.csv` | `reproduce.py` | Every check from the last reproduction run: part, report location, reported value, recomputed value, verdict. |
| `graph_edge_audit.json` | `graph_edge_audit.py` | Class counts, the strict and permissive rates, and the best and worst strata. |
| `graph_edge_audit_by_stratum.csv` | `graph_edge_audit.py` | Per-stratum class counts and meaningful rate. |
| `ir_metrics/` | `ir_metrics.py` | Cumulative recall, Precision@k and NDCG@k, per scenario and summarised. |
| `token_distribution/` | `corpus_token_distribution.py` | Per-field token summaries and per-chunk counts. |
| `test_confirmation/` | `test_confirmation.py` | TEST alpha sensitivity, cross-encoder diagnostics, category-boost reflection. |
| `benchmark150/` | `rq1_ablations.py`, `rq3_regime_constraint.py` | The two experiments that ran on the earlier benchmark. |
| `verification_report.csv` | `verify_reported_numbers.py` | 32 reported numbers vs. recomputed, with PASS/FAIL. |
| `corpus_stats.json` | `corpus_stats.py` | Corpus counts by class, regime, host, chunking method; graph edges; the index's own build manifest. |
| `top25_per_lane_final_metrics.csv` | `final_performance.py` | Table 5: final performance on DEV and TEST at 25 per lane. Also the input to Figure 7. |
| `metric_audit_table.csv` | `ce_metrics_audit.py` | Every metric with explicit numerator, denominator, cutoff and lane logic. |
| `audit_key_numbers.json` | `ce_metrics_audit.py` | Scoreable scenario, requirement and essential-chunk counts as structured data (95 / 155 / 279). |
| `dual_evidence_strict.json` | `dual_evidence_strict.py` | Full strict dual-evidence results including the scenario ids that failed on each side. |
| `dual_evidence_strict_summary.csv` | `dual_evidence_strict.py` | The same, tabulated. |
| `table4_dev_ce_variants.csv` | `ce_experiment_metrics2.py` | The cross-encoder variant comparison. |
| `summaries_v1_v3.json`, `summaries_all.json` | `ce_experiment_metrics{,2}.py` | Per-variant metric bundles; the first is the second's input. |
| `truncation_diagnostic.csv` | `ce_experiment_metrics2.py` | The 48 essential-gold chunks the reranker left below rank 25, with untruncated token counts and a truncation flag. 23 exceed 512 tokens. |
| `truncation_summary.json` | `ce_experiment_metrics2.py` | The truncation diagnostic's headline counts: 48 chunks below rank 25, 23 truncated, 47.9%. |
| `score_signal_stats_summary.json` | `score_signal_analysis.py` | All signal statistics: AUCs, Mann-Whitney tests, Cliff's delta, effective ranges, authority amplification, jurisdiction counts, reranker movement. |
| `signal_effective_range.csv` | `score_signal_analysis.py` | Nominal coefficient vs. realised spread per signal. |
| `authority_amplification_analysis.csv` | `score_signal_analysis.py` | Nominal vs. normalised authority per class. Report Figure 5. |
| `authority_rank_movement_by_class.csv` | `score_signal_analysis.py` | Reranker rank movement by authority class. |
| `normalization_saturation_analysis.csv` | `score_signal_analysis.py` | How much of each signal's range sits near the ceiling. |
| `score_distribution_summary.csv` | `score_signal_analysis.py` | Raw and normalised distributions per signal and gold group. |
| `signal_correlation_matrix.csv` | `score_signal_analysis.py` | Correlations among signals, final score, ranks and labels. |
| `source_type_score_analysis.csv` | `score_signal_analysis.py` | Score behaviour by source group. |
| `query_category_score_analysis.csv` | `score_signal_analysis.py` | Score behaviour by query suite. |
| `ce_score_analysis.csv` | `score_signal_analysis.py` | Reranker score distribution by gold group. |
| `stats_summary.json` | `score_signal_analysis.py` | Companion statistics bundle. |
| `rq1_rq3_same_budget/rq1_same_budget_50_baselines.csv` | `rq1_rq3_same_budget.py` | Nine configurations at a matched 50-result budget, DEV and TEST. |
| `rq1_rq3_same_budget/rq1_per_category.csv` | `rq1_rq3_same_budget.py` | The same comparison broken down by query category. |
| `rq1_rq3_same_budget/rq3_ce_onoff_per_category.csv` | `rq1_rq3_same_budget.py` | Reranker on/off at 25 per lane, by category. |
| `rq1_rq3_same_budget/scenario_level_all_configs.csv` | `rq1_rq3_same_budget.py` | Per-scenario results for every configuration — the input to the bootstrap. |
| `rq1_rq3_same_budget/statistical_tests_TEST.json` | `rq1_rq3_same_budget.py` | Bootstrap CIs, permutation p-values, Cohen's *d* and McNemar for the two headline comparisons. |

### `figures/`

| File | Produced by |
|---|---|
| Figure | File | Produced by |
|---|---|---|
| 1 | `figure01_corpus_composition.png` | `make_report_figures.py` |
| 4 | `figure04_configuration_comparison_test.png` | `make_report_figures.py` |
| 5 | `figure05_category_comparison_test.png` | `make_report_figures.py` |
| 6 | `figure06_first_stage_auc.png` | `make_report_figures.py` |
| 7 | `figure07_authority_calibration.png` | `make_report_figures.py` |
| 8 | `figure08_mixed_evidence_dev.png` | `make_report_figures.py` |
| 9 | `figure09_final_performance.png` | `make_report_figures.py` |
| 10 | `figure10_cumulative_recall.png` | `make_report_figures.py` |
| 11 | `figure11_ndcg_by_lane.png` | `make_report_figures.py` |
| — | `signal_diagnostics/fig1_raw_vs_normalized.png` | `score_signal_figures.py` |
| — | `signal_diagnostics/fig2_bm25_gold_vs_nongold.png` | `score_signal_figures.py` |
| — | `signal_diagnostics/fig3_dense_gold_vs_nongold_wrongregime.png` | `score_signal_figures.py` |
| — | `signal_diagnostics/fig4_ce_gold_vs_nongold.png` | `score_signal_figures.py` |

File names carry the report's own figure numbers. The four `signal_diagnostics/` panels are
supporting material for Section 6.2 rather than numbered figures in the report.

Figures 2 (two-lane architecture) and 3 (manual benchmark construction protocol) are
schematic diagrams drawn by hand rather than computed from data, so no script reproduces them.

### `scripts/`

| File | What it is |
|---|---|
| `reproduce.sh` | Wrapper over `evaluation/reproduce.py`, taking the same arguments. `bash scripts/reproduce.sh` runs all nine parts. |

---

## 6. Where each reported number comes from

`results/reproduction_report.csv` is the machine-checked version of this table: run
`python evaluation/reproduce.py --all` and read it. Table and figure numbers below are the
report's own.

| Reported quantity | Value | Artifact | Part |
|---|---|---|---|
| Corpus: documents / chunks / graph edges | 1,370 / 19,087 / 23,785 | `results/corpus_stats.json` | 1 |
| Lane split (documents) | 23 / 1,347 | same | 1 |
| BM25 ROC AUC, essential gold vs rest (DEV) | 0.728 | `results/score_signal_stats_summary.json` | 4 |
| Dense ROC AUC, essential gold vs rest (DEV) | 0.847 | same | 4 |
| **Table 3** BM25 / dense / hybrid pooled @50 (TEST) | 0.345 / 0.496 / 0.475 | `results/rq1_rq3_same_budget/rq1_same_budget_50_baselines.csv` | 3 |
| **Table 3** best pooled @50 → two-lane pre-CE @25+25 | 0.612 → 0.763 | same | 3 |
| **Table 3** two-lane post-CE @25+25 | 0.806 | same | 3 |
| RQ1 difference, CI, p, d | +0.151, [0.080, 0.225], p<0.0001, d=0.444 | `results/rq1_rq3_same_budget/statistical_tests_TEST.json` | 3 |
| RQ1 CompleteCoverage difference | +0.191, [0.106, 0.277], McNemar 19/1 | same | 3 |
| **Table 4** per-category, TEST | 11 categories | `results/rq1_rq3_same_budget/rq1_per_category.csv` | 3 |
| Authority: primary / secondary mean normalised | 0.751 / 0.038 | `results/authority_amplification_analysis.csv` | 4 |
| Authority amplification factor | 23.76× | `results/score_signal_stats_summary.json` | 4 |
| Primary-secondary pairs authority reorders | 943,159 (19.3%) | same | 4 |
| Dense gap vs non-gold / vs wrong-regime | +0.101 / +0.024 | same | 4 |
| EU candidate rows / essential gold among them | 5,088 / 0 | same | 4 |
| Graph audit: strict / not-wrong rate | 59.8% / 82.0% of 122 edges | `results/graph_edge_audit.json` | 5 |
| Graph audit: best / worst strata | 80–100% / 10–20% | `results/graph_edge_audit_by_stratum.csv` | 5 |
| Graph expansion, DEV recall on/off | 0.491 / 0.513 | `results/benchmark150/rq1_ablation_summary.csv` | 5 |
| Regime constraint, TEST cost | −2.4 / −3.2 points (n=63) | `results/benchmark150/rq3_test_frozen_summary.csv` | 6 |
| Candidate-pool ceiling, DEV | 0.916 = 142/155 | `results/metric_audit_table.csv` | 8 |
| Candidate-pool ceiling, TEST | 0.885 | `results/test_confirmation/test_ce_diagnostics.json` | 7 |
| Reranker AUC within the top-75 pool, DEV / TEST | 0.688 / 0.680 | `score_signal_stats_summary.json`, `test_ce_diagnostics.json` | 4, 7 |
| TEST rank movement, into / out of top 25 | +3.37 (+1), 15 / 9 | `results/test_confirmation/test_ce_diagnostics.json` | 7 |
| Harmful demotion, DEV / TEST | 9.1% / 7.3% | `score_signal_stats_summary.json`, `test_ce_diagnostics.json` | 4, 7 |
| **Table 7** four CE variants at the 5+5 budget | 0.593 / 0.641 / 0.652 / 0.655 | `results/table4_dev_ce_variants.csv` | 7 |
| Category boost, TEST: baseline / real / oracle | 0.556 / 0.583 / 0.694 | `results/test_confirmation/test_category_boost_reflection.csv` | 7 |
| **Figure 8** dual evidence pre-CE → post-CE (DEV) | 9/17 → 13/17 | `results/dual_evidence_strict_summary.csv` | 7 |
| **Table 6** reranking on/off at 25/lane (TEST) | 0.763 → 0.806 (+0.043, p=0.215) | `results/rq1_rq3_same_budget/statistical_tests_TEST.json` | 8 |
| **Table 8** DEV: scenarios / requirements | 95 / 155 | `results/top25_per_lane_final_metrics.csv` | 8 |
| **Table 8** DEV recall / coverage / dual evidence | 0.813 / 0.747 / 0.765 (N=17) | same | 8 |
| **Table 8** TEST: scenarios / requirements | 94 / 139 | same | 8 |
| **Table 8** TEST recall / coverage / dual evidence | 0.806 / 0.755 / 0.600 (N=15) | same | 8 |
| **Figure 10** cumulative recall at k=10, DEV / TEST | 0.690 / 0.727 post-CE | `results/ir_metrics/cumulative_requirement_recall_summary.csv` | 9 |
| **Figure 11** NDCG@k, legislation / other lane | 0.54–0.63 / 0.29–0.39 | `results/ir_metrics/precision_ndcg_summary.csv` | 9 |
| Token counts: metadata / summary / raw text | 103.1 / 71.7 / 305.1 mean | `results/token_distribution/token_distribution_summary.json` | 10 |
| Raw chunk text over the 512-token window | 14.2% | same | 10 |
| DEV candidate rows analysed | 105,823 | `results/score_signal_stats_summary.json` | 4 |
| Score reconstruction max abs error | 0.0 | same | 2, 4 |

**Which evaluator produced which number.** Everything above uses the corrected evaluator,
`mandatory_requirements_with_targets()`. The cross-encoder variant comparison (Table 7) is the
exception: it was computed before the metric audit, using `common.py:essential_targets()`, and
is reported as it was computed. Its variants are compared against each other under one
consistent extractor, so the comparison is sound; its absolute levels are not directly
comparable to the corrected numbers, which is why the report uses it for variant selection and
not for headline performance.

### Which benchmark view each part uses

Ten of the twelve parts run against the 208-scenario benchmark shipped in `benchmark/`. Parts 5
and 6 — Sections 6.2.3 and 6.2.2 — run against the 150-scenario view of it, which is the set
those two experiments were computed on, at n = 68 on DEV and n = 63 on TEST.

`evaluation/make_benchmark150.py` reconstructs that view from the shipped file: every scenario
records its collection round in `source_set` and its split under that protocol in
`source_split_150rebalance`, so the set is a filter over the shipped benchmark rather than a
separate file to be trusted on its own. `data/benchmark150/` holds the two frozen caches those
runs used, graph-off and graph-on. Parts 5 and 6 call the reconstruction themselves, so running
either one reproduces its numbers with no extra step.

## 7. Environment

The reported results were produced on:

| | |
|---|---|
| Python | 3.9.6 |
| OS | macOS 14, Apple silicon |
| Cross-encoder device | MPS |
| Random seed | 20260920 (bootstrap, permutation, split construction) |

Package versions are pinned in `requirements.txt` and `requirements-pipeline.txt`. They are
pinned so that a future run can tell an environment difference from a real difference, not
because later versions are known to break.

Determinism: the first-stage retrieval, the scoring formulas and the reranker are deterministic
given the same corpus and models, and the bootstrap and permutation tests are seeded, so Path B
is fully deterministic — two consecutive runs of `scripts/reproduce.sh` leave the working tree
unchanged. Every reported result is computed against the pinned corpus index identified in
`corpus/CORPUS_BUILD.md`.
