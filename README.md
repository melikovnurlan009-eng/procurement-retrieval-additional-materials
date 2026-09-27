# Source-aware two-lane retrieval over UK public procurement law — additional materials

Supporting materials for an MSc dissertation on evidence retrieval for UK public procurement
questions. Everything needed to reproduce the reported results is here.

**Start with [`TECHNICAL_APPENDIX.md`](TECHNICAL_APPENDIX.md).** It describes the system, states
every parameter, gives the end-to-end workflow, and explains what every file in this repository
is for. This README is just the fastest way in.

---

## Reproduce it, part by part

The report is not one result, so reproduction is not one script. It is **twelve parts**, each
matching a section, table or figure. A part re-runs the scripts that produce its own outputs
and then checks those outputs against the values the report prints. **A part passes only when
every one of its checks passes** — a script exiting cleanly is not treated as success.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python evaluation/reproduce.py --list          # what the parts are
python evaluation/reproduce.py --all           # run all twelve (about four minutes)
python evaluation/reproduce.py rq1 reranking   # run just some
python evaluation/reproduce.py 3 8             # or by number
```

Every part prints its steps, then a line per check with the reported value, the recomputed
value and a verdict, then its own verdict. The run ends with:

```
  1  corpus         Corpus composition                           10/10  PASS
  2  artifacts      Frozen artifact integrity and provenance     23/23  PASS
  3  rq1            RQ1: matched-budget comparison               33/33  PASS
  4  signals        Signal behaviour and authority calibration   17/17  PASS
  5  graph          RQ2: graph edge quality and retrieval effect  11/11  PASS
  6  regime         RQ2: regime-compatibility constraint           8/8  PASS
  7  reranking      RQ3: cross-encoder variants and completeness  30/30  PASS
  8  performance    Final fixed-configuration performance         20/20  PASS
  9  ir-metrics     Secondary IR diagnostics                      11/11  PASS
 10  tokens         Token distributions and the 512-token window  14/14  PASS
 11  figures        Figure regeneration                           11/11  PASS
 12  cross-check    Independent cross-check of headline numbers   34/34  PASS
  ------------------------------------------------------------------------
  12/12 parts reproduced   222/222 checks passed
```

Check-by-check detail goes to `results/reproduction_report.csv`. Exit status is 0 only if
every selected part passed. `bash scripts/reproduce.sh` is a wrapper taking the same arguments.

No GPU, no vector database, no model downloads. Only parts 1 and 10 want the corpus database,
and both say so and check the shipped copies of their outputs when it is absent.

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
| [`data/`](data/) | Frozen artifacts: the candidate caches and the cross-encoder outputs. This is what makes reproduction possible without the corpus. |
| [`results/`](results/) | Generated tables and statistics. Every file is produced by a script in `evaluation/`. |
| [`figures/`](figures/) | Generated figures. |
| [`scripts/`](scripts/) | `reproduce.sh`, a wrapper over the part-wise driver. |

## Two routes

**Reproduce** (default) — the twelve parts above, recomputing every reported number from the
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
- **Two sections of the report use an earlier, smaller version of the benchmark.** Sections
  6.2.2 and 6.2.3 predate the last expansion round. The report does not say so; the appendix
  does, and parts 5 and 6 reconstruct that view and reproduce both results exactly.
