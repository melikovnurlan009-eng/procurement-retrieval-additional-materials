# Benchmark: 208 scenarios with requirement-level gold evidence

Two files, one JSON object per line:

- `scenarios_all_208.jsonl` — 208 scenarios: the query, its context, and its split.
- `gold_evidence_218.jsonl` — the evidence each scenario's answer must rest on, broken down
  into individually-checkable requirements.

The unit of evaluation is the **requirement**, not the query. A procurement question such as
"can we award without competition here?" is not answered by one relevant passage; it is
answered when the controlling statutory rule *and* the procedural guidance that operationalises
it are both in hand. Each scenario therefore decomposes into one or more requirements, each with
its own set of acceptable gold chunks, and a scenario is only fully satisfied when every
mandatory requirement is.

## Composition

| Suite | DEV | TEST | Total |
|---|---:|---:|---:|
| semantic / practitioner phrasing | 27 | 27 | 54 |
| exact_anchor (direct legal anchor) | 23 | 23 | 46 |
| procedural_multi_evidence | 10 | 10 | 20 |
| cross_reference_multi_instrument | 9 | 10 | 19 |
| applicability / transition | 8 | 8 | 16 |
| faq90 (practitioner FAQ derived) | 8 | 7 | 15 |
| expansion_v2_official_guidance | 6 | 6 | 12 |
| official_workflow_expansion_v2 | 4 | 5 | 9 |
| vocabulary_mismatch | 4 | 4 | 8 |
| compound / multi-requirement | 3 | 2 | 5 |
| authority / source-role sensitive | 2 | 2 | 4 |
| **Total** | **104** | **104** | **208** |

Legal regime: PA2023 115, PCR2015 61, OTHER_REGIME 15, mixed 12, and single-figure counts for
UCR2016, CCR2016, DSPCR2011 and PCR2015_POLICY.

Not all 208 scenarios carry mandatory requirements that resolve to chunks present in the corpus,
so the **scoreable** counts used throughout the report are 95 DEV and 94 TEST scenarios,
carrying 155 and 139 mandatory requirements respectively. Every script derives these counts from
the data rather than hardcoding them.

## Splits

`split` is `dev` or `test`, 104 each. The split is **stratified by suite and legal regime**:
scenarios were grouped by `(suite, regime_context)`, shuffled deterministically within each
group with seed 20260920, and each group divided as close to evenly as integer counts allow,
alternating the rounding direction between groups to land on an exact 104/104.

DEV is where every tunable was chosen. TEST was scored once, after freezing, and no parameter
was adjusted in response to a TEST result; `config/frozen_config.json` is the record of that,
and the analysis scripts refuse to run on TEST without it.

An earlier split by source is preserved per-scenario as `split_by_source`, and the split used
for the two experiments in Sections 6.2.2 and 6.2.3 as `source_split_150rebalance`. Nothing in
the pipeline hardcodes `split`, so any of the three can be selected with `--scenarios` and a
file emitted from these fields; `evaluation/make_benchmark150.py` does exactly that for the
150-scenario view.

## Scenario fields

The corpus was assembled over three collection rounds (`source_set`: `current60` 60, `faq90` 90,
`expansion_v2` 58), and later rounds added fields earlier ones did not have. Every field below
is present on all 208 scenarios unless marked otherwise.

| Field | Meaning |
|---|---|
| `scenario_id` | Stable identifier, e.g. `DEV004`, `EXP_GUID006`, `TEST003`. |
| `split` | `dev` or `test`. |
| `suite` | Query category, as tabulated above. |
| `topic` | Short subject label. |
| `scenario_text` | The practitioner situation in prose. |
| `query` | The string actually sent to the retriever. **This is the retrieval input.** |
| `regime_context` | Which legal regime governs, e.g. `PA2023`, `PCR2015`, `mixed`. |
| `retrieval_intents` | What the query is trying to find. |
| `evidence_requirements` | Prose statement of what a complete answer must cite. |
| `requires_multiple_evidence_items` | Whether one passage can suffice. |
| `notes` | Construction notes. |
| `source_set` | Collection round: `current60`, `faq90`, `expansion_v2`. |
| `is_mixed_evidence` | *(58)* Scenario needs both legislation and non-legislation evidence. Drives DualEvidenceCoverage. |
| `regime_confidence` | *(150)* Confidence that `regime_context` is right. |
| `split_by_source` | *(150)* The superseded source-based split, kept for provenance. |
| `difficulty`, `user_role`, `organisation_type` | *(150)* Scenario framing. |
| `answerability` | *(152)* Whether the corpus can answer it at all. |
| `reference_answer` | *(58)* A model answer, for qualitative inspection. Never used in scoring. |
| `legacy_requires_graph`, `graph_evaluation`, `regime_evaluation`, `temporal_evaluation` | Flags from earlier evaluation designs, retained but not used by any reported metric. |

## Gold evidence fields

```
{ "scenario_id": ..., "source_set": ..., "gold_audit": {...},
  "requirements": [
    { "requirement_id": "REQ1",
      "description": "General duty to publish a tender notice before ...",
      "mandatory": true,                       # absent on older records = mandatory
      "required_evidence_roles": ["controlling_rule", "procedural_guidance"],
      "essential_evidence": [
        { "citation": "Procurement Act 2023, s.21",
          "gold_target_id": "PROCUREMENT_ACT_2023_S_21",
          "target_type": "LEGAL_PROVISION",
          "acceptable_chunk_ids": ["...", "..."],
          "resolution": { "status": "MATCHED", "chunk_id": "UKPGA_2023_54__NODEV1__CH_00021", ... },
          "verification_note": "Confirmed live in corpus via sqlite3 -> chunk_id ..." } ],
      "strong_supporting_evidence": [ ... ],
      "acceptable_alternatives": [ ... ] } ] }
```

- **`resolution.chunk_id`** is the chunk this citation resolved to in the frozen corpus.
  `resolution.status` is `MATCHED` for 401 of 402 essential-evidence items; one is
  `NOT_IN_CORPUS` and is correctly excluded from scoring.
- **`acceptable_chunk_ids`** are alternate chunk representations of the *same* citation — the
  same provision reachable through a different chunking path. 191 of 402 items carry at least
  one, contributing 145 additional valid target chunks across the set. Treating them as
  acceptable is not leniency: missing them counts a correct retrieval as a failure purely
  because it surfaced a different chunk of the same section.
- **`required_evidence_roles`** drives the legislation/non-legislation split used by
  DualEvidenceCoverage. Legislation roles: `controlling_rule`, `implementing_detail`.
  Non-legislation roles: `explanatory_guidance`, `procedural_guidance`, `workflow_instruction`,
  `policy_rule`, `regulator_interpretation`, `transition_rule`.
- **`mandatory: false`** marks a requirement a complete answer need not satisfy. Nine such
  requirements exist and are excluded from scoring. Where the key is absent — older records
  predating the distinction — the requirement is treated as mandatory.
- **`strong_supporting_evidence`** and **`acceptable_alternatives`** are recorded but do not
  enter any reported metric, which is computed on `essential_evidence` only.

The single function that reads all of this is
`evaluation/ce_metrics_audit.py:mandatory_requirements_with_targets()`. It is the authoritative
evaluator; Section 6 of the technical appendix names which reported number comes from it and
which from its predecessor in `evaluation/common.py`.

## How the gold was built

For each scenario, the controlling authority was identified by hand from the legislation and
official guidance, written down as a citation (`Procurement Act 2023, s.21`), and then resolved
against the frozen corpus to a concrete `chunk_id`. Resolution was verified per item and the
verification recorded in `verification_note` — typically the exact SQLite lookup that confirmed
the chunk exists, with its authority class and regime.

Sixty scenarios from the first round carry a `gold_audit` block recording a later re-audit: its
`status`, a `confidence` rating, the `issues_fixed`, and the `verification_sources` consulted.
That audit is what surfaced the two evaluator corrections described above.

No retrieval output was used to decide what the gold should be. Gold was fixed from the legal
sources first, then resolved to chunks; a chunk was never selected because the system had
returned it.

## Record counts

`gold_evidence_218.jsonl` contains **218 records for 208 scenarios**: the 208 in the benchmark
plus ten `EXP_GRAPH*` records carried over from the frozen retrieval run, which covered 218.
The candidate cache covers the same 218.

Every evaluation script joins gold to scenarios by `scenario_id` and iterates over scenario
ids, so the ten extra records are never read. Both files are shipped unmodified, which is why
the gold file still hashes to the value recorded in `config/frozen_cache_manifest.json`. Part 2
of the reproduction checks this, and you can confirm it directly:

```bash
python - <<'PY'
import json
S = {json.loads(l)["scenario_id"] for l in open("benchmark/scenarios_all_208.jsonl")}
G = {json.loads(l)["scenario_id"] for l in open("benchmark/gold_evidence_218.jsonl")}
print(f"scenarios {len(S)}, gold records {len(G)}")
print("gold with no scenario:", sorted(G - S))
print("scenarios with no gold:", sorted(S - G))
PY
```
