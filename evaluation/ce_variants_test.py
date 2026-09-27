#!/usr/bin/env python3
"""The cross-encoder variants re-measured on TEST (report Section 6.3.3).

The variants were selected on DEV; this runs the same comparison on the frozen TEST split so
the DEV choice can be checked against a split nothing was tuned on. No configuration is
selected here - the report notes that plain fusion at lambda=0.50 scores higher on TEST than
the DEV-selected metadata+fusion configuration and was still not re-selected, which is the
point of reporting it.

    python evaluation/ce_variants_test.py

Reads only the shipped candidate cache, benchmark and variant-2 TEST output.
Writes results/test_confirmation/comparison_TEST.csv.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from evaluation.ce_metrics_audit import load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, FAQ218, CACHE_PATH
from evaluation.common import requirement_recall_at_k, complete_coverage

LAW_ROLES = {"controlling_rule", "implementing_detail"}
NONLAW_ROLES = {"explanatory_guidance", "procedural_guidance", "workflow_instruction", "policy_rule", "regulator_interpretation", "transition_rule"}
from evaluation.paths import CE_OUTPUT_VARIANT2_TEST, RESULTS_DIR

RD = str(RESULTS_DIR / "test_confirmation")
RESULTS_DIR.joinpath("test_confirmation").mkdir(parents=True, exist_ok=True)


def ranked_lane_post(cache, sid, lane, ce_scores, fusion_lambda=None):
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty or not ce_scores:
        return sub.sort_values("pre_rerank_rank")["chunk_id"].tolist() if not sub.empty else []
    head = sub[sub.chunk_id.isin(ce_scores.keys())].copy()
    tail = sub[~sub.chunk_id.isin(ce_scores.keys())].sort_values("pre_rerank_rank")
    head["ce_score_v"] = head["chunk_id"].map(ce_scores)
    if fusion_lambda is None or fusion_lambda >= 1.0:
        head = head.sort_values("ce_score_v", ascending=False)
    else:
        ce_min, ce_max = head["ce_score_v"].min(), head["ce_score_v"].max()
        fs_min, fs_max = head["final_score"].min(), head["final_score"].max()
        ce_norm = (head["ce_score_v"] - ce_min) / (ce_max - ce_min) if ce_max > ce_min else 0.5
        fs_norm = (head["final_score"] - fs_min) / (fs_max - fs_min) if fs_max > fs_min else 0.5
        head["blend"] = fusion_lambda * ce_norm + (1 - fusion_lambda) * fs_norm
        head = head.sort_values("blend", ascending=False)
    return pd.concat([head, tail])["chunk_id"].tolist()


def compute(name, cache, scenarios, gold, ce_scores_fn, fusion_lambda=None):
    rows = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        g = gold.get(sid, {})
        targets_full = mandatory_requirements_with_targets(g)
        if not targets_full:
            continue
        targets = {rid: t["chunk_ids"] for rid, t in targets_full.items()}
        L_post = ranked_lane_post(cache, sid, "legislation", ce_scores_fn(sid, "legislation"), fusion_lambda)
        O_post = ranked_lane_post(cache, sid, "other", ce_scores_fn(sid, "other"), fusion_lambda)
        rr10 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
        cc10 = complete_coverage(L_post, O_post, targets, 5, 5)
        dual = None
        if sc.get("is_mixed_evidence"):
            law_reqs, nonlaw_reqs = [], []
            for rid, t in targets_full.items():
                roles = t["roles"]
                if roles & LAW_ROLES: law_reqs.append(rid)
                if roles & NONLAW_ROLES: nonlaw_reqs.append(rid)
            if law_reqs and nonlaw_reqs:
                top10 = set(L_post[:5]) | set(O_post[:5])
                law_ok = any(targets[r] & top10 for r in law_reqs)
                nonlaw_ok = any(targets[r] & top10 for r in nonlaw_reqs)
                dual = 1.0 if (law_ok and nonlaw_ok) else 0.0
        rows.append({"scenario_id": sid, "requirement_recall_10": rr10, "complete_coverage_10": cc10, "dual_evidence_coverage": dual})
    df = pd.DataFrame(rows)
    return {
        "variant": name, "n_scenarios": len(df),
        "requirement_recall_at_10": df.requirement_recall_10.mean(),
        "complete_coverage_at_10": df.complete_coverage_10.mean(),
        "dual_evidence_coverage_at_10": df.dual_evidence_coverage.dropna().mean() if df.dual_evidence_coverage.notna().any() else None,
        "n_dual_evidence_scenarios": int(df.dual_evidence_coverage.notna().sum()),
    }


if __name__ == "__main__":
    cache = pd.read_parquet(CACHE_PATH)
    scenarios = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    scenarios = [s for s in scenarios if str(s.get("split", "")).lower() == "test"]
    gold = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_218.jsonl")}
    want = {s["scenario_id"] for s in scenarios}
    cache = cache[cache.scenario_id.isin(want)]

    baseline_ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    v2 = json.load(open(CE_OUTPUT_VARIANT2_TEST))
    v2_ce = {}
    for sc in v2["scenarios"]:
        sid = sc["scenario_id"]
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            v2_ce[(sid, lane)] = {r["chunk_id"]: r["ce_score"] for r in sc.get(field, [])}

    summaries = []
    summaries.append(compute("1_baseline_raw_text_CE_TEST", cache, scenarios, gold, lambda sid, lane: baseline_ce.get((sid, lane))))
    for lam in [0.25, 0.50, 0.75, 1.00]:
        summaries.append(compute(f"3_CE_plus_fusion_lambda{lam}_TEST", cache, scenarios, gold, lambda sid, lane: baseline_ce.get((sid, lane)), fusion_lambda=lam))
    summaries.append(compute("2_metadata_enriched_CE_TEST", cache, scenarios, gold, lambda sid, lane: v2_ce.get((sid, lane))))
    for lam in [0.25, 0.50, 0.75, 1.00]:
        summaries.append(compute(f"4_metadata_enriched_CE_plus_fusion_lambda{lam}_TEST", cache, scenarios, gold, lambda sid, lane: v2_ce.get((sid, lane)), fusion_lambda=lam))

    df = pd.DataFrame(summaries)
    df.to_csv(f"{RD}/comparison_TEST.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {RD}/comparison_TEST.csv")
