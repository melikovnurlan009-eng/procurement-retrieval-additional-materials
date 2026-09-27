# Chunk quality assessment, repair and pruning — working record

Dates: 2026-09-17 to 2026-09-18. Corpus: the merged index `code/state/chunk_index_merged.sqlite3`
(22,042 chunks, 1,737 documents, 19 source domains) and Qdrant collection `chunks__bge_m3__merged`.
All numbers below are measured, not estimated. Paths are relative to the additional-materials
bundle unless absolute.

## 1. Starting point

The shipped chunk-quality tooling (`code/evaluate_chunk_quality.py`) had two tiers:

- **Tier 1, structural (deterministic).** Regex/heuristic detectors run over every chunk with its
  next neighbour: `SEVERED_SENTENCE` (chunk ends unterminated AND the next chunk resumes in lower
  case), `SEVERED_ENUMERATION` (ends on `, or` / `, and` / `unless` / `provided that` / `subject to`
  / `except where` / `if`), `SEVERED_LIST_STEM` (ends on `:`), `ORPHAN_LIST_ITEM` (starts on a
  bracketed/lettered/bulleted marker with `chunk_ordinal > 1`), `TOO_SMALL_TO_RETRIEVE` (< 15
  tokens), `EMPTY_TEXT`, plus advisories (size band 200–800 tokens, hard max 1,200; summary
  lexical grounding < 0.15; missing topics/concepts).
- **Tier 2, LLM judge (v1).** A 14-line prompt: six dimensions scored 1–5 with one-line definitions,
  free-text issues, a boolean `harmful_split`. Its only prior run (gpt-4o-mini, 49 chunks of an
  early 744-chunk snapshot) returned 4.56/5 overall, LEGISLATION 4.94–5.0 on every dimension, and
  every "top issue" counted exactly once — the signature of an unanchored rubric (scores drift to
  4–5) and of issues that never aggregate. No judge run had ever covered the merged corpus.

## 2. Redesign of the LLM judge (rubric v3, `code/evaluate_chunk_quality.py`)

Why: the question that matters for a RAG system is whether the chunk *text* carries a substantive,
citable point an answer can be generated from — not whether it is tidy. The v1 rubric never asked
that, and could not see boundaries because it never saw the neighbouring text.

What changed:

- **Primary dimension `answer_backbone`** (1–5): 5 = a complete operative point a practitioner
  could act on (rule with conditions, obligation, deadline/threshold, procedure, definition,
  exception); 3 = real substance but partial; 1 = no operative content (news, opinion, navigation,
  heading, contents list, pointer elsewhere). Kept: `boundary_correctness`, `semantic_coherence`,
  `self_containedness`, `retrieval_usefulness`, and the two metadata checks `title_accuracy`,
  `summary_faithfulness` (now nullable — PDF-lane chunks carry no generated summary).
- **Anchored scale**: each dimension defines what a 1, 3 and 5 look like. Opening posture: "an
  auditor, not a grader; a 5 is rare; never round up; do not reward length, legal vocabulary or
  being on-topic".
- **Hard caps** that override judgment: `answer_backbone <= 2` for commentary/boilerplate;
  `<= 3` if the point's conditions or list sit outside the chunk; `summary_faithfulness <= 2` if the
  summary states any specific absent from the text; `boundary_correctness <= 2` for cuts that
  separate a rule from its conditions or fall mid-sentence/list/table; `retrieval_usefulness <= 2`
  for navigation/boilerplate; `semantic_coherence <= 3` for three or more separately headed topics.
- **Neighbour context**: the tail of the previous chunk and the head of the next (400 chars each)
  are passed, labelled "do not score", so a boundary can actually be judged. The judge is told when
  we truncated the text at 12,000 chars so it does not read our cut as the chunker's.
- **Coded issues**, 22 codes in two severities (MAJOR: `SEVERED_RULE_FROM_CONDITIONS`,
  `SEVERED_SENTENCE`, `SEVERED_LIST_STEM`, `ORPHAN_LIST_ITEM`, `SEVERED_TABLE`, `HEADING_ONLY`,
  `NO_OPERATIVE_CONTENT`, `MIXED_TOPICS`, `UNRESOLVED_CROSS_REFERENCE`, `UNDEFINED_TERM_DEPENDENCY`,
  `SUMMARY_FABRICATED_SPECIFIC`, `SUMMARY_OVERGENERALISES`, `TITLE_WRONG_SUBJECT`,
  `NAVIGATION_OR_BOILERPLATE`, `TOO_SMALL_TO_BE_EVIDENCE`, `DUPLICATED_CONTENT`; MINOR:
  `TITLE_TOO_GENERIC`, `SUMMARY_OMITS_KEY_POINT`, `LEADING_FRAGMENT`, `TRAILING_FRAGMENT`,
  `EXTRACTION_NOISE`, `WRONG_METADATA_TAGS`, `OTHER`). Every issue must carry a verbatim quote of
  at most 25 words ("no quote, no issue"); every score of 3 or below must be explained by an issue.
  The prompt and the JSON schema are generated from one dict so they cannot drift.
- **`content_type`** (LEGAL_RULE / PROCEDURE_OR_GUIDANCE / DEFINITION / COMMENTARY_OR_NEWS /
  NAVIGATION_OR_BOILERPLATE / MIXED) and a **`rechunk` verdict** (KEEP / SPLIT / MERGE / DROP) with a
  40-word reason; SPLIT must supply verbatim `suggested_split_points` (first 6–12 words of each
  new chunk); the prompt states that smaller is not better by default and forbids splitting a rule
  from its exemption list; `answer_backbone <= 2` cannot be KEEP.
- Engineering: judge straight from the SQLite index (all lanes, `--db`), explicit `--chunk-ids`,
  `--dry-run` token/cost counting, a 6-worker thread pool with exponential back-off on rate limits,
  a clean stop on `insufficient_quota` that keeps everything already written, resume keyed on
  (`prompt_version`, `judge_model`) so a rubric change never silently reuses old verdicts, and a
  report with per-dimension score histograms, `pct_scoring_2_or_below`, coded issue counts and
  breakdowns by `source_kind` and `chunking_method`. Versions: `EVALUATOR_VERSION 1.1.0`,
  `JUDGE_PROMPT_VERSION chunk_quality_rubric_v3`.
- Smoke test on three hand-picked chunks showed the known limit of the chosen model: gpt-4.1-mini
  scored a chunk opening mid-sentence ("Schedule 8, paragraph 6(b)). For further information…") as
  boundary 5 even with that exact example written into the cap. Recorded as a caveat: the judge is
  reliable on *what the text says*, weak on *where it was cut*.

## 3. Full-corpus judge run

- Model gpt-4.1-mini (chosen over gpt-4.1 on cost: measured 65.2M input / 9.9M output tokens for
  the corpus; ~$26–42 on mini vs ~$132–210 on 4.1). Six workers, 2h 15m, 22,042/22,042 judged,
  **zero errors**. Outputs: `code/evaluation/chunk_quality_judge/chunk_quality_judge.jsonl` (one
  verdict per chunk) and `chunk_quality_judge_report.json`.
- Means: answer_backbone **3.83**, retrieval_usefulness 3.84, title_accuracy 4.33,
  summary_faithfulness 4.38, self_containedness 4.90, boundary_correctness 4.98,
  semantic_coherence 4.98. `pct_scoring_2_or_below`: answer_backbone **20.1%**, boundary 0.0%.
- answer_backbone distribution: 1: 3,058 · 2: 1,373 · 3: 3,792 · 4: 1,759 · 5: 12,060.
  Tiers: strong (4–5) 13,819 (62.7%), partial (3) 3,792 (17.2%), weak (1–2) 4,431 (20.1%).
- content_type: LEGAL_RULE 12,996 · PROCEDURE_OR_GUIDANCE 5,536 · COMMENTARY_OR_NEWS 2,039 ·
  NAVIGATION_OR_BOILERPLATE 1,267 · DEFINITION 202 · MIXED 2.
- rechunk: KEEP 18,936 · DROP 2,886 · MERGE 172 · SPLIT 48 (all 48 with split points).
- By lane (answer_backbone / %DROP / % with a MAJOR issue): LLM_LEG_TEXT_V2 4.12 / 4.2 / 20;
  STRUCTURAL_NODE_V1 4.06 / 13.0 / 26; LLM_PDF_TEXT_V2 3.81 / 13.8 / 32;
  LLM_SEMANTIC_BOUNDARY_V1 (web guidance) 3.36 / 27.3 / 47.
- By authority (backbone / %DROP): PRIMARY_LEGISLATION 4.13 / 4.7; SECONDARY_LEGISLATION 4.12 / 4.4;
  OFFICIAL_REGULATOR_GUIDANCE 4.28 / 9.2; OFFICIAL_GOVERNMENT_GUIDANCE 3.86 / 12.7;
  OFFICIAL_WORKFLOW 3.06 / 31.8; PROFESSIONAL_INTERPRETATION 3.07 / 32.4.
- Top issues: NO_OPERATIVE_CONTENT 5,948 (27.0%); UNDEFINED_TERM_DEPENDENCY 729;
  TOO_SMALL_TO_BE_EVIDENCE 333; TITLE_WRONG_SUBJECT 244; UNRESOLVED_CROSS_REFERENCE 236;
  minor: TITLE_TOO_GENERIC 4,270, SUMMARY_OMITS_KEY_POINT 2,790.
- Does the junk get served? Measured on the 60-scenario benchmark, top-10 only: corpus rate of
  DROP chunks 13.1% vs lexical 7.7%, dense 3.0%, hybrid 5.5%, two-lane legislation lane 1.1%,
  two-lane other lane 4.5%. The ranker already avoids most of it; the guidance lane still serves
  ~1 in 13 chunks with no backbone.

## 4. Structural tier on the same corpus, and the disagreement

Ran tier 1 over the current 22,042 chunks (never done before on the merged corpus): boundary
defects SEVERED_SENTENCE 179, SEVERED_ENUMERATION 11, SEVERED_LIST_STEM 301, ORPHAN_LIST_ITEM 659,
TOO_SMALL 42 — **1,085 chunks** with at least one boundary defect; rate by lane PDF 7.9%,
web guidance 6.8%, legislation 0.5%, structural-node 0.0%. 88% of all boundary defects are the one
failure "list stem severed from its items".

Cross-check against the judge:

| | structural | judge | overlap |
|---|---|---|---|
| boundary defect flags | 1,085 chunks | 232 chunks with a boundary issue code | 45 |
| SPLIT candidates | 609 chunks > 1,200 tokens (advisory) | 48 SPLIT | 32 |
| MERGE candidates | 42 TOO_SMALL | 172 MERGE | 1 |

Samples of the 917 chunks the structural tier flagged but the judge scored boundary 5 / KEEP, by
trigger: 252 start on "(NN)" — numbered paragraphs and directive recitals, i.e. **structural false
positives** (the `LIST_START` regex treats a numbered paragraph as a list item); 245 start on a
lettered item, 56 on a bullet, 48 on a roman sub-item — true orphans the judge missed; 218 end on
":" (true severed stems, missed); 140 severed sentences (missed); 9 dangling enumerations.
Samples of the 173 the judge wanted split/merged but the structural tier passed: oversize
multi-topic chunks (2,600–3,400 tokens) and tiny lead-ins ("…must have regard to the following
considerations.", 41 tokens) that end on a full stop and so trip no regex.

Decision, after reading the samples: **take the union** of the two, minus the one identified
false-positive class. Structural catches cuts the judge cannot see; the judge catches oversize
topic mixtures and stemless lead-ins the regexes cannot see; the "(NN)" orphan class is excluded.
Fix set: 1,010 chunks (structural boundary flags minus "(NN)"-only orphans ∪ judge SPLIT/MERGE).

Where the low-quality chunks come from (union of content-bad = judge DROP or backbone ≤ 2, and
boundary-bad as above): 4,183 content-bad, 749 boundary-bad, 253 both, 16,857 (76.5%) fine.
Content-bad is a source problem: COMMENTARY_OR_NEWS 85% bad, NAVIGATION_OR_BOILERPLATE 100%,
LEGAL_RULE 7%, PROCEDURE_OR_GUIDANCE 9%; by domain procurementpathway.civilservice.gov.uk 57%,
procurementlawyers.org.uk 44%, procurementjourney.scot 32%, gov.uk 28%, procurementportal.com
29%, assets.publishing.service.gov.uk 18%, legislation.gov.uk 12% (mostly the off-topic Acts —
FOIA s.36 73%, Enterprise Act — and title/signature chunks). Boundary-bad is a lane problem:
the UK–EU TCA PDF alone (2,566 chunks of annex schedules) carries 343 boundary defects.

## 5. Repair (re-chunking) — rule-based, no LLM

Script: `scratchpad/graph_fix/repair_chunks.py` (dry-run + apply; reads the index DB for chunk
order, the structural per-chunk file and the judge file; never touches the live DB). Operations
are decided per boundary between consecutive chunks of one segment, left to right; a merge
re-evaluates the new chunk against the following one so a list keeps merging until it ends or
the 1,200-token cap is reached.

| Operation | Rule | Count |
|---|---|---|
| MOVE_FRAGMENT | chunk ends mid-sentence or on ", or"/"unless" → the tail after the last sentence terminator moves to the head of the next chunk (only if the fragment is < 50% of the chunk and < 300 tokens) | 174 |
| MERGE (list) | chunk ends on ":" or next chunk starts on (a)/(iv)/bullet → join, continue while the list continues, cap 1,200 tokens | 510 |
| MERGE (judge) | judge MERGE → join with the neighbour that completes it: previous if the chunk starts on an item, next if it is a lead-in or < 80 tokens | 67 (44 prev, 23 next) |
| PREFIX_STEM | merge would exceed the cap → copy the stem line onto the head of the item chunk so the items are never read without their governing clause | 40 (60 skipped: stem line empty or > 80 tokens) |
| SPLIT | judge SPLIT at its verbatim split points, each piece ≥ 120 tokens | 21 chunks → 79 pieces (27 SPLIT verdicts had no usable point) |
| merge blocked | size cap, no stem to copy | 7 |

The LLM's role here was to *say* SPLIT/MERGE and where; the cutting and joining was done by the
rules above, deterministically. Merged chunks keep the first constituent's metadata; split pieces
drop the inherited `retrieval_summary` (it described the whole chunk); changed chunks get
`embedding_text` recomposed as citation / heading / retrieval_title + new text (the same layout the
index uses); `repaired_from` and `repair_ops` are recorded on every changed chunk and
`repair_map.jsonl` maps every old id to its new id.

Result: 926 chunks changed, 22,042 → 21,521. Re-scored by the structural tier:
SEVERED_SENTENCE 179 → 8, SEVERED_ENUMERATION 11 → 0, SEVERED_LIST_STEM 301 → 64,
ORPHAN_LIST_ITEM 659 → 307 (250 of the 307 are the "(NN)" false positives; real residue 57),
TOO_SMALL 42 → 40; max chunk 12,912 → 12,655 tokens, median 216 → 217 (no bloat).
Spot checks at the seams read correctly (e.g. "one of the following conditions must be met
(section 78(3)): a. before awarding…" is one chunk again). Known imperfection: in a few
PREFIX_STEM cases the copied "stem" is itself a list item; the rule should require the line to
end on ":".

**Still needs attention: 226 chunks (1.05%)** — 64 unresolved stems (20 already ≥ 1,000 tokens),
57 real orphans, 40 too-small, 8 severed sentences, 18 unapplied judge SPLITs, 42 unapplied judge
MERGEs. Listed with reasons and text in `review_needed.json` → `rechunk_residue`.

## 6. Dropping — with the benchmark's own labels as the guard

The judge returned 2,886 DROP verdicts. Because that is one model's opinion, every DROP was
checked against ground truth the project already has: the standalone benchmark's
`gold_evidence.jsonl` (84 resolved chunk ids), `qrels_provisional.jsonl` (grade ≥ 2), and the
workbench `qrels_silver.jsonl` (grade ≥ 2, DEV and TEST), plus the citation graph.

- **41** DROP chunks are labelled relevant by the benchmark (10 gold targets, 4 silver-relevant,
  32 provisional-relevant) — judge errors, e.g. the "Threshold amounts" guidance chunk. Kept.
- **17** DROP chunks are targets of legal citation edges. Kept (dropping them breaks the edges).
- Remaining 2,828 split by the judge's content type:
  - **Tier 1, dropped: 1,251** NAVIGATION_OR_BOILERPLATE (menus, contents lists, cookie/legal
    notices, contact blocks).
  - **Tier 2, dropped: 1,293** COMMENTARY_OR_NEWS with no rule stated.
  - **Tier 3, held for human review: 284** LEGAL_RULE / PROCEDURE text the judge found empty —
    mostly amendment-instruction paragraphs in off-topic Acts (DPA 2018, Enterprise Act 2002,
    FOIA) and Schedule 6 offence entries. Listed in `review_needed.json` → `drop_tier3_review`.

How the drop was applied to the live index: rows were **not deleted**. `chunks.filtered_out` was
set to `JUDGE_DROP_T1_NAVIGATION_BOILERPLATE` (1,251) / `JUDGE_DROP_T2_COMMENTARY_NEWS` (1,293) —
the same column and mechanism the retriever already uses for page furniture: `ChunkRetriever.load()`
skips any row with `filtered_out` set, and every channel (lexical, dense, graph) passes through
`load()`, so the 2,544 chunks are gone from all results while edges, FTS and the Qdrant
collection stay consistent. Verified: `load()` returns 0 of 6 sampled dropped ids. Reversible with
one UPDATE; a pre-drop backup of the DB is at
`scratchpad/graph_fix/chunk_index_merged.pre_drop_backup.sqlite3`.

## 7. Graph work done alongside (summary)

- Audit of the 27,600 edges: only 5,118 were traversable; 55% of guidance→law and 51% of
  law→law edges had sources no anchor could match (3,146 stale pre-rechunk chunk ids across 217
  re-chunked documents; 758 from documents no longer in the corpus; 1,419 from unchunked
  Part/Schedule nodes; 920 from bare-mention parser nodes; ~830 unchunked targets).
- Rebuilt the guidance→law layer with the shipped `extract_guidance_references.py` on the current
  chunks: 6,164 edges, 3,325 new pairs; traversable edges 5,118 → 8,795; chunk→chunk pairs for
  REFERENCES 2,876 → 6,549; chunks touched by the graph 3.4% → 11.9%. Revival rules measured for the
  rest: push-down of container-level sources recovers ~1,120, roll-down of whole-Schedule targets
  ~740; ~1,700 are correctly dead.
- Rule-based COMMENCED_BY / COMMENCEMENT_PROVISION / AMENDED_BY / REVOKED_BY / SAVED_BY
  candidates from legislation.gov.uk editorial notes: 627 with both ends chunked
  (`scratchpad/graph_fix/build_annotation_candidates.py`), pending the human edge review
  (`graph_edge_review/review_sheet.md`, 122 items in 14 strata).

## 8. Versioned rebuild for the benchmark (in progress at time of writing)

Three versions under `scratchpad/graph_fix/versions/`, each a new SQLite index built with the
shipped `build_chunk_index.py lexical` + `densify_graph_edges.py`, and a benchmark copy whose
gold/qrels chunk ids are remapped through `repair_map.jsonl`:

- **v1g** — original 22,042 chunks + repaired graph; vectors: the live collection.
- **v2a** — repaired 21,521 chunks + repaired graph; collection `chunks__bge_m3__v2a` (unchanged
  chunks copy their live vectors; the 926 changed chunks + 79 split pieces re-embedded with BGE-M3).
- **v2b** — v2a minus tier-1/2 drops: 19,087 chunks; collection `chunks__bge_m3__v2b`.

Configs A–F re-run on each with the unmodified `ChunkRetriever`. The shipped bundle, live DB
contents (other than the `filtered_out` marks above) and live collection are unchanged; baseline-v1
stays frozen.

## 8a. Merge sizes (checked after the question "very long chunks merged?")

478 merged chunks (405 from two constituents, 53 from three, 21 from four to six). Characters:
min 133, median 2,488, p90 4,189, **max 4,795**; tokens max 1,198 — none over the 1,200 cap, 157
above the 800-token preferred band. PREFIX_STEM added 5–302 chars; MOVE_FRAGMENT moved at most 806
chars. The corpus's genuinely long chunks pre-date the repair: 609 chunks over 1,200 tokens (579 of
them LLM_SEMANTIC_BOUNDARY_V1 web-guidance pages, largest 50,622 chars — Procurement Journey
"Module 9: Contract governance"); the repair never created one and only touched 59 of them by
MOVE_FRAGMENT/PREFIX_STEM. Splitting those is a separate decision (the judge asked for 48 splits).

## 8b. Benchmark results (60 scenarios, configs A–F, unmodified ChunkRetriever)

Judge-independent strict essential-evidence recall@10 (cannot be biased by pooling):

| config | v1 shipped | v1g (graph repaired) | v2a (+chunk repair) | v2b (+drops) |
|---|---|---|---|---|
| E hybrid+graph | 0.139 | 0.178 | 0.186 | 0.186 |
| **F two-lane** | **0.264** | **0.394** | **0.411** | **0.411** |
| F scenario-complete@10 | 0.133 | 0.250 | 0.267 | 0.283 |
| A/B/C/D | unchanged (0.050 / 0.108 / 0.125 / 0.214) | | | |

qrels-based nDCG@10 (judged on the v1 candidate pool): A 0.721→0.718→0.711→0.729;
B 0.843→0.843→0.835→0.834; C 0.815→0.809→0.799→0.803; D 0.833→0.838→0.822→0.823;
E 0.780→0.776→0.774→0.778; F 0.810→0.786→0.785→0.781. F requirement-coverage@10 0.769→0.803→
0.794→0.794; F regime-error 0.167→0.117→0.117→0.133; lexical MRR 0.320→0.318→0.325→0.352.

Reading: the graph repair is the large effect (F strict recall +50% relative), the chunk repair adds
a little more (+0.017), the drops help BM25 most (lexical nDCG +0.017, MRR +0.027) and nothing
else measurably. The F nDCG dip is pooling bias, not a regression: unjudged top-10 slots in the F
legislation lane rise from 4.1% (v1) to 20.5% (v1g) / 26.1% (v2a) / 28.2% (v2b) because the
repaired graph surfaces legislation chunks that were never in the judged pool and are scored as
non-relevant by default. Per `JUDGING_PROTOCOL.md` the pool must be expanded and re-judged before
those numbers are comparable (`judge_candidate_pool.py`, ~150 new pairs, gpt-4o-mini).

## 8c. Human review decisions applied (2026-09-18)

The 568 items in `review_needed.json` were reviewed by the assessor
(`procurement_search_chunk_decisions.json`, rubric: KEEP / KEEP_RECHUNK / TEMPLATE_ONLY /
GRAPH_ONLY / EVAL_ONLY / DISCARD). File summary: KEEP 45, KEEP_RECHUNK 128, TEMPLATE_ONLY 21,
GRAPH_ONLY 4, EVAL_ONLY 44, DISCARD 326. Applied to the live index as follows (a merged/split
review id maps to all its constituent live chunks through `repair_map.jsonl`, so applied counts
are slightly higher than the file's):

- DISCARD → `filtered_out = HUMAN_REVIEW_DISCARD` (353 live chunks); TEMPLATE_ONLY →
  `HUMAN_REVIEW_TEMPLATE_ONLY` (22; excluded from evidence search, tagged so a template lane can
  pick them up); EVAL_ONLY → `HUMAN_REVIEW_EVAL_ONLY` (44; rows kept so benchmark ids stay valid,
  excluded from production search). Every DISCARD/TEMPLATE decision was re-checked against
  gold/qrels: 0 collisions.
- GRAPH_ONLY (4): removed from `chunks_fts` and from the live Qdrant collection (vectors backed
  up to `scratchpad/graph_fix/graph_only_vectors_backup.json`), rows kept so `load()` still
  returns them when graph expansion reaches them.
- KEEP / KEEP_RECHUNK: any earlier judge tag cleared (6 chunks un-dropped).
- Caveat recorded: 16 of the filtered chunks are legal-citation edge targets; while filtered,
  graph expansion cannot surface them (the assessor's EVAL_ONLY on the 17 edge-target chunks).
- Live index now: 2,898 of 22,042 rows filtered (1,199 T1 navigation, 1,282 T2 commentary,
  351 human DISCARD, 44 EVAL_ONLY, 22 TEMPLATE_ONLY); collection 22,038 points.
  Backup before this step: `scratchpad/graph_fix/chunk_index_merged.pre_decisions_backup.sqlite3`.
  Log: `chunk_quality_work/decisions_applied.json`.
- KEEP_RECHUNK (128): 36 are > 1,200 tokens and are being split by the LLM oversize job; 92 go
  through a relaxed second rule pass (`repair_pass2.py`); 7 judge-SPLIT items are deferred to the
  LLM splitter.

## 9. Files

- `chunk_quality_work/review_needed.json` — 226 re-chunk residue + 284 tier-3 drop review + the 58
  overruled DROPs + the 2,544 dropped ids.
- `code/evaluation/chunk_quality_judge/` — full judge verdicts and report.
- `scratchpad/graph_fix/corpus/chunk_quality_report.json` and `chunk_quality_per_chunk.jsonl` —
  structural tier on the current corpus; `repaired_corpus/` — the same after repair.
- `scratchpad/graph_fix/repaired/` — `chunks_repaired.jsonl`, `repair_map.jsonl`, `repair_report.json`.
- `scratchpad/graph_fix/drop_lists.json` — tiers, overruled, edge targets.
- `graph_edge_review/` — edge review pack and the 150-chunk one-hop dump.

Uncommitted at time of writing: the `evaluate_chunk_quality.py` rewrite and the judge output
directory in the bundle; the original project copy in `procurement-kg-rag/` is untouched.
