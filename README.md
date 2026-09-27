# Source-aware two-lane retrieval over UK public procurement law — additional materials

Supporting materials for an MSc dissertation on evidence retrieval for UK public procurement
questions. Everything needed to reproduce the reported results is here.

**Start with [`TECHNICAL_APPENDIX.md`](TECHNICAL_APPENDIX.md).** It describes the system, states
every parameter, gives the end-to-end workflow, and explains what every file in this repository
is for. This README is just the fastest way in.

---

## The one-minute version

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python evaluation/verify_reported_numbers.py
```

That recomputes 32 numbers quoted in the report — the whole final performance table, the
candidate-pool ceiling, both first-stage ROC AUCs, the authority amplification factor, the
reranker's rank movement and harmful demotion rate, and an exact score-reconstruction check —
directly from the artifacts in this repository, and compares each against the reported value.
It writes `results/verification_report.csv` and prints a pass/fail line per check.

Expected output ends with:

```
32/32 checks passed.
```

No GPU, no Qdrant, no corpus database, no model downloads. About a minute.

To regenerate everything else — figures, tables, statistical tests:

```bash
bash scripts/reproduce.sh
```

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
| [`scripts/`](scripts/) | `reproduce.sh`. |

## Two routes

**Verify** (default) — recompute every reported number from the artifacts here. Needs
`requirements.txt` and nothing else.

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
