# Source-aware two-lane retrieval over UK public procurement law — additional materials

Supporting materials for an MSc dissertation on evidence retrieval for UK public procurement
questions. Everything needed to reproduce the reported results is here.

**Start with [`TECHNICAL_APPENDIX.md`](TECHNICAL_APPENDIX.md).** It describes the system, states
every parameter, gives the end-to-end workflow, and explains what every file in this repository
is for. This README is just the fastest way in.

---

## Reproduce it, part by part

The report is not one result, so reproduction is not one script. It is **nine parts**, each
matching a section, table or figure. A part re-runs the scripts that produce its own outputs
and then checks those outputs against the values the report prints. **A part passes only when
every one of its checks passes** — a script exiting cleanly is not treated as success.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python evaluation/reproduce.py --list          # what the parts are
python evaluation/reproduce.py --all           # run all nine (about three minutes)
python evaluation/reproduce.py performance     # run just one
python evaluation/reproduce.py 3 5 7           # or several, by number
```

Every part prints its steps, then a line per check with the reported value, the recomputed
value and a verdict, then its own verdict. The run ends with:

```
  1  corpus         Corpus composition                           10/10  PASS
  2  artifacts      Frozen artifact integrity and provenance     26/26  PASS
  3  performance    Final retrieval performance                  15/15  PASS
  4  dual-evidence  Strict mixed-evidence completeness             8/8  PASS
  5  signals        Score and signal analysis                    16/16  PASS
  6  ce-variants    Cross-encoder input variants and truncation   14/14  PASS
  7  rq1-rq3        Same-budget ablations and statistical tests   25/25  PASS
  8  figures        Figure regeneration                            9/9  PASS
  9  cross-check    Independent cross-check of headline numbers   34/34  PASS
  ------------------------------------------------------------------------
  9/9 parts reproduced   157/157 checks passed
```

Check-by-check detail goes to `results/reproduction_report.csv`. Exit status is 0 only if
every selected part passed. `bash scripts/reproduce.sh` is a wrapper taking the same
arguments.

No GPU, no vector database, no model downloads. Only part 1 wants the corpus database, and it
says so and checks the shipped copy of its output when that is absent.

## What the system does, in a paragraph

A complete answer to a procurement question needs both the controlling statutory rule and the
official guidance that operationalises it. Those live in very unequal populations: 23 legislation
documents against 1,347 non-legislation documents, with guidance echoing a practitioner's own
wording far more closely than statute does. In one merged ranking, statute loses. The system
therefore **partitions the candidate pool by source type** and ranks each lane independently —
a legislation lane and an other-evidence lane, each contributing its own top 25 — so statute
never has to out-compete guidance on surface similarity. Inside each lane, BM25 and dense
retrieval are fused, an authority prior and a jurisdiction demotion are applied, and a
cross-encoder reranks the top 75.

At a matched 50-result budget on the held-out confirmation set, the best conventional pooled
configuration reaches a requirement recall of 0.612; the two-lane architecture reaches 0.763
(+0.151, 95% CI [0.080, 0.225]).

## Layout

| | |
|---|---|
| [`TECHNICAL_APPENDIX.md`](TECHNICAL_APPENDIX.md) | **The main document.** System, parameters, workflow, every file explained. |
| [`benchmark/`](benchmark/) | 208 scenarios with requirement-level gold evidence, and [`BENCHMARK.md`](benchmark/BENCHMARK.md) documenting the schema and how it was built. |
| [`config/`](config/) | The freeze record: every tunable, with the DEV-only evidence that justified it, written before TEST was scored. |
| [`corpus/`](corpus/) | [`CORPUS_BUILD.md`](corpus/CORPUS_BUILD.md) — what the corpus is, how it was built, its SHA-256, how to obtain it. The 165 MB index itself is not shipped. |
| [`src/`](src/) | The system under test: the retriever, and the corpus-construction chain. |
| [`evaluation/`](evaluation/) | The analysis harness. |
| [`data/`](data/) | Frozen artifacts: the candidate cache and the cross-encoder outputs. This is what makes verification possible without the corpus. |
| [`results/`](results/) | Generated tables and statistics. Every file is produced by a script in `evaluation/`. |
| [`figures/`](figures/) | Generated figures. |
| [`scripts/`](scripts/) | `reproduce.sh`, a wrapper over the part-wise driver. |

## Two routes

**Reproduce** (default) — the nine parts above, recomputing every reported number from the
artifacts here. Needs `requirements.txt` and nothing else.

**Rebuild** — re-run first-stage retrieval and cross-encoder inference from the corpus. Needs
`requirements-pipeline.txt`, the corpus SQLite index (see
[`corpus/CORPUS_BUILD.md`](corpus/CORPUS_BUILD.md)) and a running Qdrant instance. The appendix
gives the exact commands in order.

The boundary between the two is `evaluation/merge_ce_into_cache.py`: once the reranker's scores
are merged into the candidate cache, nothing downstream needs a model, a GPU or the corpus again.

## Paths

Nothing is hardcoded to a machine. Every path resolves relative to the repository root through
`evaluation/paths.py`, and each is overridable by environment variable:

```bash
export CORPUS_DB=/path/to/chunk_index.sqlite3     # only for the two steps that need it
export CANDIDATE_CACHE=/path/to/candidate_cache.parquet
export RESULTS_DIR=/somewhere/else                # to avoid writing into the checkout
```

## Honest notes

The appendix's final section lists known issues in full. The two most worth knowing up front:

- **TEST is a confirmation set, not a pristine held-out set.** The stratified split superseded an
  earlier split by source, so part of TEST comes from material audited earlier in the project.
  This was a deliberate trade for category comparability; the older split is preserved
  per-scenario and can be reconstructed.
- **The corpus build is not bit-reproducible**, because semantic chunking calls an LLM. The index
  is therefore hash-pinned and every reported result is computed against that one pinned copy.
