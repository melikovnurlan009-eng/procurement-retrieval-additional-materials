#!/usr/bin/env python3
"""Variants 2 (metadata-enriched CE), 4 (metadata-enriched CE + fusion), and 5
(truncation diagnostic), plus final comparison CSV + markdown report.
Reuses V1/V3 results from summaries_v1_v3.json. DEV only."""
from __future__ import annotations
import sys, json, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from evaluation.common import load_scenarios, load_gold, essential_targets, all_gold_chunk_ids, requirement_recall_at_k, complete_coverage
from evaluation.ce_experiment_metrics import ranked_lane, LAW_ROLES, NONLAW_ROLES

from evaluation.paths import BENCHMARK_DIR
FAQ218 = str(BENCHMARK_DIR)
from evaluation.paths import (CANDIDATE_CACHE, CE_OUTPUT_VARIANT2_DEV, CORPUS_DB,
                              RESULTS_DIR, require_corpus_db)

DB = str(CORPUS_DB)
RD = str(RESULTS_DIR)


def cls_group(a):
    if a == "PRIMARY_LEGISLATION": return "primary_legislation"
    if a == "SECONDARY_LEGISLATION": return "secondary_legislation"
    return "guidance_workflow"


def full_metrics(name, cache, scenarios, gold, ce_scores_v1, ce_scores_v2, fusion_lambda=None, use_v2=False):
    """ce_scores_v1/v2: dict[(sid,lane)] -> {chunk_id: score}. use_v2 selects which CE score set drives ranking."""
    src = ce_scores_v2 if use_v2 else ce_scores_v1
    rows, rank_moves = [], []
    for sc in scenarios:
        sid = sc["scenario_id"]
        g = gold.get(sid, {})
        targets = essential_targets(g)
        if not targets:
            continue
        gold_all = all_gold_chunk_ids(g)
        L_pre = ranked_lane(cache, sid, "legislation", None)
        O_pre = ranked_lane(cache, sid, "other", None)
        L_post = ranked_lane(cache, sid, "legislation", src.get((sid, "legislation")), fusion_lambda=fusion_lambda)
        O_post = ranked_lane(cache, sid, "other", src.get((sid, "other")), fusion_lambda=fusion_lambda)
        rr10 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
        cc10 = complete_coverage(L_post, O_post, targets, 5, 5)
        cov5 = requirement_recall_at_k(L_post, O_post, targets, 5, 5)
        cov10 = requirement_recall_at_k(L_post, O_post, targets, 10, 10)
        cov25 = requirement_recall_at_k(L_post, O_post, targets, 25, 25)
        dual = None
        if sc.get("is_mixed_evidence"):
            law_reqs, nonlaw_reqs = [], []
            for req in g.get("requirements", []):
                if not req.get("mandatory"):
                    continue
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
    df = pd.DataFrame(rows); moves = pd.DataFrame(rank_moves)
    n_into_25 = int(((moves.pre_rank > 25) & (moves.post_rank <= 25)).sum()) if len(moves) else 0
    n_out_25 = int(((moves.pre_rank <= 25) & (moves.post_rank > 25)).sum()) if len(moves) else 0
    was_top25 = moves[moves.pre_rank <= 25] if len(moves) else moves
    hdr = float((was_top25.post_rank > 25).mean()) if len(was_top25) else None
    if len(moves): moves["class_group"] = moves.authority_class.map(cls_group)
    return {
        "variant": name, "n_scenarios": len(df),
        "essential_gold_coverage_at_5": df.coverage_5.mean(), "essential_gold_coverage_at_10": df.coverage_10.mean(),
        "essential_gold_coverage_at_25": df.coverage_25.mean(), "requirement_recall_at_10": df.requirement_recall_10.mean(),
        "complete_coverage_at_10": df.complete_coverage_10.mean(),
        "dual_evidence_coverage_at_10": df.dual_evidence_coverage.dropna().mean() if df.dual_evidence_coverage.notna().any() else None,
        "n_dual_evidence_scenarios": int(df.dual_evidence_coverage.notna().sum()),
        "mean_rank_movement": float(moves.movement.mean()) if len(moves) else None,
        "median_rank_movement": float(moves.movement.median()) if len(moves) else None,
        "n_essential_gold_moved_into_top25": n_into_25, "n_essential_gold_moved_out_of_top25": n_out_25,
        "harmful_demotion_rate_P_post_gt25_given_pre_le25": hdr, "n_was_in_top25_pre": int(len(was_top25)),
        "by_authority_class_group": moves.groupby("class_group").movement.agg(["count", "mean", "median"]).to_dict("index") if len(moves) else {},
        "by_lane": moves.groupby("lane").movement.agg(["count", "mean", "median"]).to_dict("index") if len(moves) else {},
    }


def truncation_diagnostic(cache, scenarios, gold, ce_scores_v1):
    """How many essential-gold chunks that the cross-encoder left outside the top 25 would
    have exceeded its 512-token input window. Needs chunk text, so this is the one part of
    this script that requires the corpus database."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    con = sqlite3.connect(require_corpus_db())
    affected = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        query = sc.get("query") or sc.get("scenario_text", "")
        g = gold.get(sid, {})
        targets = essential_targets(g)
        if not targets: continue
        for lane in ("legislation", "other"):
            L_post = ranked_lane(cache, sid, lane, ce_scores_v1.get((sid, lane)))
            post_rank = {c: i + 1 for i, c in enumerate(L_post)}
            for req_id, ids in targets.items():
                for cid in ids:
                    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane) & (cache.chunk_id == cid)]
                    if sub.empty: continue
                    pr = post_rank.get(cid)
                    if pr is None or pr <= 25: continue
                    row = con.execute("SELECT text FROM chunks WHERE chunk_id=?", (cid,)).fetchone()
                    if not row: continue
                    text = row[0]
                    n_tok = len(tok.encode(query, text, truncation=False))
                    truncated = n_tok > 512
                    affected.append({"scenario_id": sid, "chunk_id": cid, "lane": lane, "ce_rank": pr,
                                      "n_tokens_untruncated": n_tok, "likely_truncated": truncated})
    # Sorted so a rerun produces a byte-identical file: requirement iteration order over a set
    # is not stable across runs, and without this two correct runs differ by row order alone.
    return (pd.DataFrame(affected)
            .sort_values(["scenario_id", "lane", "ce_rank", "chunk_id"])
            .reset_index(drop=True))


if __name__ == "__main__":
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_scenarios(f"{FAQ218}/scenarios_all_208.jsonl", "DEV")
    gold = load_gold(f"{FAQ218}/gold_evidence_208.jsonl")
    want = {s["scenario_id"] for s in scenarios}
    cache = cache[cache.scenario_id.isin(want)]

    baseline_ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    v2 = json.load(open(CE_OUTPUT_VARIANT2_DEV))
    v2_ce = {}
    for sc in v2["scenarios"]:
        sid = sc["scenario_id"]
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            v2_ce[(sid, lane)] = {r["chunk_id"]: r["ce_score"] for r in sc.get(field, [])}

    all_summaries = json.load(open(f"{RD}/summaries_v1_v3.json"))

    summ_v2 = full_metrics("2_metadata_enriched_CE", cache, scenarios, gold, baseline_ce, v2_ce, fusion_lambda=None, use_v2=True)
    all_summaries.append(summ_v2)
    print("V2 done:", summ_v2["requirement_recall_at_10"], summ_v2["essential_gold_coverage_at_25"])

    for lam in [0.25, 0.50, 0.75, 1.00]:
        name = f"4_metadata_enriched_CE_plus_fusion_lambda{lam}"
        s = full_metrics(name, cache, scenarios, gold, baseline_ce, v2_ce, fusion_lambda=lam, use_v2=True)
        all_summaries.append(s)
        print(f"V4 lambda={lam} done:", s["requirement_recall_at_10"], s["essential_gold_coverage_at_25"])

    json.dump(all_summaries, open(f"{RD}/summaries_all.json", "w"), indent=2, default=str)

    print("Running truncation diagnostic (variant 5)...")
    shipped = Path(RD) / "truncation_diagnostic.csv"
    if CORPUS_DB.exists():
        trunc_df = truncation_diagnostic(cache, scenarios, gold, baseline_ce)
        trunc_df.to_csv(shipped, index=False)
    elif shipped.exists():
        # The diagnostic needs chunk text, so it needs the corpus DB. Its output is shipped,
        # so report from that rather than silently overwriting it with an empty result.
        trunc_df = pd.read_csv(shipped)
        print(f"  corpus DB not available - reporting the shipped {shipped.name} unchanged "
              f"(set CORPUS_DB to recompute it)")
    else:
        print("  skipped: needs chunk text, so it needs the corpus database. Set CORPUS_DB.")
        trunc_df = None

    if trunc_df is not None:
        n_affected = len(trunc_df)
        n_truncated = int(trunc_df.likely_truncated.astype(bool).sum()) if n_affected else 0
        print(f"V5: {n_affected} essential-gold chunks left below rank 25, "
              f"{n_truncated} of them likely truncated (>512 tokens)")
        json.dump({"n_essential_gold_rank_gt25": n_affected, "n_likely_truncated": n_truncated,
                   "pct_truncated": (n_truncated / n_affected if n_affected else None)},
                  open(f"{RD}/truncation_summary.json", "w"), indent=2)

    # comparison CSV
    rows = []
    for s in all_summaries:
        rows.append({
            "variant": s["variant"], "n_scenarios": s["n_scenarios"],
            "coverage@5": round(s["essential_gold_coverage_at_5"], 4),
            "coverage@10": round(s["essential_gold_coverage_at_10"], 4),
            "coverage@25": round(s["essential_gold_coverage_at_25"], 4),
            "RequirementRecall@10": round(s["requirement_recall_at_10"], 4),
            "CompleteCoverage@10": round(s["complete_coverage_at_10"], 4),
            "DualEvidenceCoverage@10": round(s["dual_evidence_coverage_at_10"], 4) if s["dual_evidence_coverage_at_10"] is not None else None,
            "n_dual_evidence_scenarios": s["n_dual_evidence_scenarios"],
            "mean_rank_movement": round(s["mean_rank_movement"], 2) if s["mean_rank_movement"] is not None else None,
            "median_rank_movement": s["median_rank_movement"],
            "moved_into_top25": s["n_essential_gold_moved_into_top25"],
            "moved_out_of_top25": s["n_essential_gold_moved_out_of_top25"],
            "harmful_demotion_rate": round(s["harmful_demotion_rate_P_post_gt25_given_pre_le25"], 4) if s["harmful_demotion_rate_P_post_gt25_given_pre_le25"] is not None else None,
            "n_was_top25_pre": s["n_was_in_top25_pre"],
        })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(f"{RD}/table4_dev_ce_variants.csv", index=False)
    print(cdf.to_string(index=False))
    print(f"\nwrote {RD}/table4_dev_ce_variants.csv and {RD}/summaries_all.json")
