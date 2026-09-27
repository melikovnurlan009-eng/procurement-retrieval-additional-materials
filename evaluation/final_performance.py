#!/usr/bin/env python3
"""The report's headline result: final top-25-per-lane performance on DEV and TEST.

This is Table 5 and Figure 7. It is computed here, in one place, so that the table and the
figure cannot drift apart: make_report_figures.py plots the CSV this script writes rather
than recomputing the same numbers a second time.

Every metric uses the corrected evaluator, ce_metrics_audit.mandatory_requirements_with_
targets() - acceptable_chunk_ids unioned, mandatory-only filter - and the post-cross-encoder
ranking at the frozen configuration. The two lanes are OR-pooled at 25 each, never merged
into one ranking.

    python evaluation/final_performance.py

Reads only the shipped candidate cache and benchmark. Writes
results/top25_per_lane_final_metrics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation.ce_metrics_audit import (
    LAW_ROLES,
    NONLAW_ROLES,
    load_jsonl,
    mandatory_requirements_with_targets,
    ranked_lane_post,
    req_satisfied,
)
from evaluation.paths import CANDIDATE_CACHE, GOLD_PATH, RESULTS_DIR, SCENARIOS_PATH

K_PER_LANE = 25
TEST_LABEL = "TEST (confirmation only, baseline CE config, not tuned)"


def split_metrics(cache, scen_by_id, targets, ce_scores, scen_ids, label) -> dict:
    """One row of Table 5, for one split."""
    sc_targets = {sid: targets[sid] for sid in scen_ids if sid in targets}
    n_req = sum(len(t) for t in sc_targets.values())

    # Mixed-evidence scenarios are those that genuinely need both sides: a mandatory
    # legislation-role requirement AND a mandatory non-legislation-role requirement.
    mixed = {}
    for sid, treqs in sc_targets.items():
        if not scen_by_id.get(sid, {}).get("is_mixed_evidence"):
            continue
        law = {r: t for r, t in treqs.items() if t["roles"] & LAW_ROLES}
        nonlaw = {r: t for r, t in treqs.items() if t["roles"] & NONLAW_ROLES}
        if law and nonlaw:
            mixed[sid] = (law, nonlaw)

    def ranked(sid):
        L = ranked_lane_post(cache, sid, "legislation", ce_scores.get((sid, "legislation")))
        O = ranked_lane_post(cache, sid, "other", ce_scores.get((sid, "other")))
        return L[:K_PER_LANE], O[:K_PER_LANE]

    req_num = cc_num = 0
    for sid, treqs in sc_targets.items():
        L, O = ranked(sid)
        satisfied_all = True
        for t in treqs.values():
            if req_satisfied(t["chunk_ids"], L, O, K_PER_LANE, K_PER_LANE):
                req_num += 1
            else:
                satisfied_all = False
        cc_num += int(satisfied_all)

    dual_num = leg_side = nonleg_side = 0
    for sid, (law, nonlaw) in mixed.items():
        L, O = ranked(sid)
        law_ok = all(req_satisfied(t["chunk_ids"], L, O, K_PER_LANE, K_PER_LANE)
                     for t in law.values())
        nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, K_PER_LANE, K_PER_LANE)
                        for t in nonlaw.values())
        leg_side += int(law_ok)
        nonleg_side += int(nonlaw_ok)
        dual_num += int(law_ok and nonlaw_ok)

    n_scen, n_mixed = len(sc_targets), len(mixed)
    return {
        "split": label,
        "n_scenarios": n_scen,
        "n_requirements": n_req,
        # Requirement-weighted, not a mean of per-scenario ratios: see the appendix, section 3.
        "RequirementRecall@25-per-lane": req_num / n_req if n_req else None,
        "CompleteCoverage@25-per-lane": cc_num / n_scen if n_scen else None,
        "n_mixed_evidence_scenarios": n_mixed,
        "DualEvidenceCoverage@25-per-lane": dual_num / n_mixed if n_mixed else None,
        "legislation_side_coverage": leg_side / n_mixed if n_mixed else None,
        "nonlegislation_side_coverage": nonleg_side / n_mixed if n_mixed else None,
    }


def main() -> None:
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_jsonl(SCENARIOS_PATH)
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    scen_by_id = {s["scenario_id"]: s for s in scenarios}

    targets = {}
    for sid in scen_by_id:
        t = mandatory_requirements_with_targets(gold.get(sid, {}))
        if t:
            targets[sid] = t

    ce_scores = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        ce_scores[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    def ids(split):
        return {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == split}

    rows = [
        split_metrics(cache, scen_by_id, targets, ce_scores, ids("dev"), "DEV"),
        split_metrics(cache, scen_by_id, targets, ce_scores, ids("test"), TEST_LABEL),
    ]

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "top25_per_lane_final_metrics.csv"
    df.to_csv(out, index=False)

    show = df.copy()
    show["split"] = ["DEV", "TEST"]
    for col in show.columns:
        if show[col].dtype == float:
            show[col] = show[col].round(4)
    print("Final top-25-per-lane performance (Table 5 / Figure 7)\n")
    print(show.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
