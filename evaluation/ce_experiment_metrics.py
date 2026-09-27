#!/usr/bin/env python3
"""CE reranking experiment: compute the full required metric suite for each variant,
DEV split only, reusing the identical first-stage candidate pool throughout. No gold
features are used in any CE input - only corpus metadata available at real inference time.
"""
from __future__ import annotations
import sys, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from evaluation.common import load_scenarios, load_gold, essential_targets, all_gold_chunk_ids, requirement_recall_at_k, complete_coverage

from evaluation.paths import BENCHMARK_DIR
FAQ218 = str(BENCHMARK_DIR)
from evaluation.paths import CANDIDATE_CACHE, CORPUS_DB, RESULTS_DIR

DB = str(CORPUS_DB)
RD = str(RESULTS_DIR)

LAW_ROLES = {"controlling_rule", "implementing_detail"}
NONLAW_ROLES = {"explanatory_guidance", "procedural_guidance", "workflow_instruction", "policy_rule", "regulator_interpretation", "transition_rule"}


def load_variant_ce(path):
    """Returns {(scenario_id, lane): {chunk_id: ce_score}}"""
    d = json.load(open(path))
    out = {}
    for sc in d["scenarios"]:
        sid = sc["scenario_id"]
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            out[(sid, lane)] = {r["chunk_id"]: r["ce_score"] for r in sc.get(field, [])}
    return out


def ranked_lane(cache, sid, lane, ce_scores: dict | None, fusion_lambda: float | None = None):
    """ce_scores: {chunk_id: ce_score} for this (sid,lane)'s top-75, or None for pre-CE.
    fusion_lambda: if set, blend normalized ce_score with normalized first-stage final_score
    within the reranked head (lambda=1.0 = pure CE, lambda=0.0 = pure first-stage order)."""
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty:
        return []
    if ce_scores is None:
        return sub.sort_values("pre_rerank_rank")["chunk_id"].tolist()
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


def compute_metrics(variant_name, cache, scenarios, gold, ce_scores_fn):
    """ce_scores_fn(sid, lane) -> dict or None (pre-CE)."""
    rows = []
    rank_moves = []  # per essential-gold-chunk: pre_rank, post_rank, lane, authority_class
    for sc in scenarios:
        sid = sc["scenario_id"]
        g = gold.get(sid, {})
        targets = essential_targets(g)
        if not targets:
            continue
        gold_all = all_gold_chunk_ids(g)

        L_pre = ranked_lane(cache, sid, "legislation", None)
        O_pre = ranked_lane(cache, sid, "other", None)
        L_post = ranked_lane(cache, sid, "legislation", ce_scores_fn(sid, "legislation"))
        O_post = ranked_lane(cache, sid, "other", ce_scores_fn(sid, "other"))

        rr10 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
        cc10 = complete_coverage(L_post, O_post, targets, 5, 5)

        # essential gold coverage @5/10/25 (lane-independent OR-pool, per-lane k)
        cov5 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
        cov10 = requirement_recall_at_k(L_post, O_post, targets, 10, 10)
        cov25 = requirement_recall_at_k(L_post, O_post, targets, 25, 25)

        # DualEvidenceCoverage@10: only for is_mixed_evidence scenarios with both role types mandatory
        dual = None
        if sc.get("is_mixed_evidence"):
            law_reqs, nonlaw_reqs = [], []
            for req in g.get("requirements", []):
                if not req.get("mandatory"):
                    continue
                roles = set(req.get("required_evidence_roles", []))
                if roles & LAW_ROLES:
                    law_reqs.append(req["requirement_id"])
                if roles & NONLAW_ROLES:
                    nonlaw_reqs.append(req["requirement_id"])
            if law_reqs and nonlaw_reqs:
                top10 = set(L_post[:5]) | set(O_post[:5])
                law_ok = any(targets.get(r, set()) & top10 for r in law_reqs)
                nonlaw_ok = any(targets.get(r, set()) & top10 for r in nonlaw_reqs)
                dual = 1.0 if (law_ok and nonlaw_ok) else 0.0

        rows.append({"scenario_id": sid, "requirement_recall_10": rr10, "complete_coverage_10": cc10,
                     "coverage_5": cov5, "coverage_10": cov10, "coverage_25": cov25, "dual_evidence_coverage": dual})

        # per-chunk rank movement, essential gold only
        pre_rank_leg = {c: i + 1 for i, c in enumerate(L_pre)}
        pre_rank_oth = {c: i + 1 for i, c in enumerate(O_pre)}
        post_rank_leg = {c: i + 1 for i, c in enumerate(L_post)}
        post_rank_oth = {c: i + 1 for i, c in enumerate(O_post)}
        sub = cache[(cache.scenario_id == sid) & (cache.chunk_id.isin(gold_all))]
        for _, r in sub.iterrows():
            cid, lane, auth = r["chunk_id"], r["lane"], r["authority_class"]
            is_essential = any(cid in ids for ids in targets.values())
            if not is_essential:
                continue
            if lane == "legislation":
                pre_r, post_r = pre_rank_leg.get(cid), post_rank_leg.get(cid)
            else:
                pre_r, post_r = pre_rank_oth.get(cid), post_rank_oth.get(cid)
            if pre_r is None or post_r is None:
                continue
            rank_moves.append({"scenario_id": sid, "chunk_id": cid, "lane": lane, "authority_class": auth,
                                "pre_rank": pre_r, "post_rank": post_r, "movement": pre_r - post_r})

    df = pd.DataFrame(rows)
    moves = pd.DataFrame(rank_moves)

    n_into_25 = int(((moves.pre_rank > 25) & (moves.post_rank <= 25)).sum()) if len(moves) else 0
    n_out_25 = int(((moves.pre_rank <= 25) & (moves.post_rank > 25)).sum()) if len(moves) else 0
    was_top25 = moves[moves.pre_rank <= 25] if len(moves) else moves
    harmful_demotion_rate = float((was_top25.post_rank > 25).mean()) if len(was_top25) else None

    def cls_group(a):
        if a == "PRIMARY_LEGISLATION": return "primary_legislation"
        if a == "SECONDARY_LEGISLATION": return "secondary_legislation"
        return "guidance_workflow"
    if len(moves):
        moves["class_group"] = moves.authority_class.map(cls_group)

    summary = {
        "variant": variant_name,
        "n_scenarios": len(df),
        "essential_gold_coverage_at_5": df.coverage_5.mean(),
        "essential_gold_coverage_at_10": df.coverage_10.mean(),
        "essential_gold_coverage_at_25": df.coverage_25.mean(),
        "requirement_recall_at_10": df.requirement_recall_10.mean(),
        "complete_coverage_at_10": df.complete_coverage_10.mean(),
        "dual_evidence_coverage_at_10": df.dual_evidence_coverage.dropna().mean() if df.dual_evidence_coverage.notna().any() else None,
        "n_dual_evidence_scenarios": int(df.dual_evidence_coverage.notna().sum()),
        "mean_rank_movement": float(moves.movement.mean()) if len(moves) else None,
        "median_rank_movement": float(moves.movement.median()) if len(moves) else None,
        "n_essential_gold_moved_into_top25": n_into_25,
        "n_essential_gold_moved_out_of_top25": n_out_25,
        "harmful_demotion_rate_P_post_gt25_given_pre_le25": harmful_demotion_rate,
        "n_was_in_top25_pre": int(len(was_top25)),
    }
    by_class = moves.groupby("class_group").movement.agg(["count", "mean", "median"]).to_dict("index") if len(moves) else {}
    by_lane = moves.groupby("lane").movement.agg(["count", "mean", "median"]).to_dict("index") if len(moves) else {}
    summary["by_authority_class_group"] = by_class
    summary["by_lane"] = by_lane
    return summary, df, moves


if __name__ == "__main__":
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_scenarios(f"{FAQ218}/scenarios_all_208.jsonl", "DEV")
    gold = load_gold(f"{FAQ218}/gold_evidence_218.jsonl")
    want = {s["scenario_id"] for s in scenarios}
    cache = cache[cache.scenario_id.isin(want)]

    # variant 1: baseline (raw-text CE), already in the cache's own ce_score column
    baseline_ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    all_summaries = []
    all_dfs = {}

    summ, df, moves = compute_metrics("1_baseline_raw_text_CE", cache, scenarios, gold,
                                       lambda sid, lane: baseline_ce.get((sid, lane)))
    all_summaries.append(summ); all_dfs["1_baseline_raw_text_CE"] = (df, moves)
    print("V1 done:", summ["requirement_recall_at_10"], summ["essential_gold_coverage_at_25"])

    # variant 3: baseline CE + first-stage fusion, lambda sweep
    for lam in [0.25, 0.50, 0.75, 1.00]:
        name = f"3_CE_plus_fusion_lambda{lam}"
        summ, df, moves = compute_metrics(name, cache, scenarios, gold,
                                           lambda sid, lane: baseline_ce.get((sid, lane)))
        # need fusion_lambda passed through ranked_lane - patch by recomputing with lambda
        rows = []
        rank_moves = []
        for sc in scenarios:
            sid = sc["scenario_id"]
            g = gold.get(sid, {})
            targets = essential_targets(g)
            if not targets: continue
            gold_all = all_gold_chunk_ids(g)
            L_pre = ranked_lane(cache, sid, "legislation", None)
            O_pre = ranked_lane(cache, sid, "other", None)
            L_post = ranked_lane(cache, sid, "legislation", baseline_ce.get((sid, "legislation")), fusion_lambda=lam)
            O_post = ranked_lane(cache, sid, "other", baseline_ce.get((sid, "other")), fusion_lambda=lam)
            rr10 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
            cc10 = complete_coverage(L_post, O_post, targets, 5, 5)
            cov5 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
            cov10 = requirement_recall_at_k(L_post, O_post, targets, 10, 10)
            cov25 = requirement_recall_at_k(L_post, O_post, targets, 25, 25)
            dual = None
            if sc.get("is_mixed_evidence"):
                law_reqs, nonlaw_reqs = [], []
                for req in g.get("requirements", []):
                    if not req.get("mandatory"): continue
                    roles = set(req.get("required_evidence_roles", []))
                    if roles & LAW_ROLES: law_reqs.append(req["requirement_id"])
                    if roles & NONLAW_ROLES: nonlaw_reqs.append(req["requirement_id"])
                if law_reqs and nonlaw_reqs:
                    top10 = set(L_post[:5]) | set(O_post[:5])
                    law_ok = any(targets.get(r, set()) & top10 for r in law_reqs)
                    nonlaw_ok = any(targets.get(r, set()) & top10 for r in nonlaw_reqs)
                    dual = 1.0 if (law_ok and nonlaw_ok) else 0.0
            rows.append({"scenario_id": sid, "requirement_recall_10": rr10, "complete_coverage_10": cc10,
                         "coverage_5": cov5, "coverage_10": cov10, "coverage_25": cov25, "dual_evidence_coverage": dual})
            pre_rank_leg = {c: i+1 for i, c in enumerate(L_pre)}; pre_rank_oth = {c: i+1 for i, c in enumerate(O_pre)}
            post_rank_leg = {c: i+1 for i, c in enumerate(L_post)}; post_rank_oth = {c: i+1 for i, c in enumerate(O_post)}
            sub = cache[(cache.scenario_id == sid) & (cache.chunk_id.isin(gold_all))]
            for _, r in sub.iterrows():
                cid, lane, auth = r["chunk_id"], r["lane"], r["authority_class"]
                is_essential = any(cid in ids for ids in targets.values())
                if not is_essential: continue
                if lane == "legislation": pre_r, post_r = pre_rank_leg.get(cid), post_rank_leg.get(cid)
                else: pre_r, post_r = pre_rank_oth.get(cid), post_rank_oth.get(cid)
                if pre_r is None or post_r is None: continue
                rank_moves.append({"scenario_id": sid, "chunk_id": cid, "lane": lane, "authority_class": auth,
                                    "pre_rank": pre_r, "post_rank": post_r, "movement": pre_r - post_r})
        df2 = pd.DataFrame(rows); moves2 = pd.DataFrame(rank_moves)
        n_into_25 = int(((moves2.pre_rank > 25) & (moves2.post_rank <= 25)).sum()) if len(moves2) else 0
        n_out_25 = int(((moves2.pre_rank <= 25) & (moves2.post_rank > 25)).sum()) if len(moves2) else 0
        was_top25 = moves2[moves2.pre_rank <= 25] if len(moves2) else moves2
        hdr = float((was_top25.post_rank > 25).mean()) if len(was_top25) else None
        def cls_group(a):
            if a == "PRIMARY_LEGISLATION": return "primary_legislation"
            if a == "SECONDARY_LEGISLATION": return "secondary_legislation"
            return "guidance_workflow"
        if len(moves2): moves2["class_group"] = moves2.authority_class.map(cls_group)
        summ2 = {
            "variant": name, "n_scenarios": len(df2),
            "essential_gold_coverage_at_5": df2.coverage_5.mean(), "essential_gold_coverage_at_10": df2.coverage_10.mean(),
            "essential_gold_coverage_at_25": df2.coverage_25.mean(), "requirement_recall_at_10": df2.requirement_recall_10.mean(),
            "complete_coverage_at_10": df2.complete_coverage_10.mean(),
            "dual_evidence_coverage_at_10": df2.dual_evidence_coverage.dropna().mean() if df2.dual_evidence_coverage.notna().any() else None,
            "n_dual_evidence_scenarios": int(df2.dual_evidence_coverage.notna().sum()),
            "mean_rank_movement": float(moves2.movement.mean()) if len(moves2) else None,
            "median_rank_movement": float(moves2.movement.median()) if len(moves2) else None,
            "n_essential_gold_moved_into_top25": n_into_25, "n_essential_gold_moved_out_of_top25": n_out_25,
            "harmful_demotion_rate_P_post_gt25_given_pre_le25": hdr, "n_was_in_top25_pre": int(len(was_top25)),
            "by_authority_class_group": moves2.groupby("class_group").movement.agg(["count","mean","median"]).to_dict("index") if len(moves2) else {},
            "by_lane": moves2.groupby("lane").movement.agg(["count","mean","median"]).to_dict("index") if len(moves2) else {},
        }
        all_summaries.append(summ2); all_dfs[name] = (df2, moves2)
        print(f"V3 lambda={lam} done:", summ2["requirement_recall_at_10"], summ2["essential_gold_coverage_at_25"])

    json.dump(all_summaries, open(f"{RD}/summaries_v1_v3.json", "w"), indent=2, default=str)
    print(f"\nwrote {RD}/summaries_v1_v3.json (V2/V4 pending variant2 CE completion)")
