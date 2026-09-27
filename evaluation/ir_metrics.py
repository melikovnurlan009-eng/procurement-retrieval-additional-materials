#!/usr/bin/env python3
"""Secondary IR diagnostics: cumulative RequirementRecall@k, Precision@k and NDCG@k.

These are the report's Figure 10 (cumulative requirement recall by per-lane candidate depth,
before and after the cross-encoder, on DEV and TEST) and Figure 11 (post-CE NDCG@k by
evidence lane), and the "Secondary diagnostics" paragraph of the report's Appendix A.

k is per-lane candidate depth, evaluated at 10, 20, 30, 40, 50, 60, 70 and 75.

  - Cumulative RequirementRecall@k credits a mandatory requirement when at least one
    acceptable gold chunk for it appears within the first k candidates of EITHER lane -
    the same OR-pooled rule the headline metric uses.
  - Precision@k and NDCG@k are computed per lane, because each lane has its own ranked
    list and its own gold subset. NDCG uses binary relevance (acceptable gold = 1,
    otherwise 0), logarithmic rank discounting and per-query ideal-DCG normalisation.

Binary rather than graded relevance is deliberate: qrel_grade is populated for too small
and too biased a slice of the pool to support graded relevance safely.

No system parameter is changed and no depth was selected from TEST; these are descriptive.

    python evaluation/ir_metrics.py

Reads only the shipped candidate cache and benchmark. Writes results/ir_metrics/.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from evaluation.ce_metrics_audit import (
    load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, ranked_lane_post,
    req_satisfied, FAQ218, CACHE_PATH,
)

from evaluation.paths import RESULTS_DIR

RD = str(RESULTS_DIR / "ir_metrics")
RESULTS_DIR.joinpath("ir_metrics").mkdir(parents=True, exist_ok=True)

K_VALUES = [10, 20, 30, 40, 50, 60, 70, 75]


def precision_at_k(ranked, relevant, k):
    if k == 0:
        return None
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if c in relevant) / len(top)


def dcg_at_k(ranked, relevant, k):
    return sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if c in relevant)


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return None
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    if idcg == 0:
        return None
    return dcg_at_k(ranked, relevant, k) / idcg


def main():
    cache_full = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_218.jsonl")}
    scen_by_id = {s["scenario_id"]: s for s in scen_all}

    scen_targets_all = {}
    for sid, g in gold_all.items():
        t = mandatory_requirements_with_targets(g)
        if t:
            scen_targets_all[sid] = t

    def essential_ids_by_lane(sid, cache):
        """chunk_id -> lane, restricted to this scenario's essential-gold ids that actually
        appear in the candidate pool (so precision/nDCG denominators reflect retrievable gold)."""
        t = scen_targets_all.get(sid, {})
        all_ids = set().union(*[r["chunk_ids"] for r in t.values()]) if t else set()
        sub = cache[(cache.scenario_id == sid) & (cache.chunk_id.isin(all_ids))]
        out = {"legislation": set(), "other": set()}
        for _, row in sub.iterrows():
            out[row.lane].add(row.chunk_id)
        return out

    baseline_ce = {}
    for (sid, lane), grp in cache_full[cache_full.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    per_lane_rows = []   # Precision@K / nDCG@K, per scenario/lane/stage/K
    recall_rows = []     # cumulative RequirementRecall@K, per scenario/stage/K (two-lane OR-pool)

    for split_label in ["DEV", "TEST"]:
        split_ids = {s["scenario_id"] for s in scen_all if str(s.get("split", "")).lower() == split_label.lower()}
        cache = cache_full[cache_full.scenario_id.isin(split_ids)]
        scen_targets = {sid: t for sid, t in scen_targets_all.items() if sid in split_ids}
        print(f"{split_label}: {len(scen_targets)} scoreable scenarios", file=sys.stderr)

        for sid, targets in scen_targets.items():
            rel = essential_ids_by_lane(sid, cache)
            for stage, rank_fn in [
                ("pre_CE", lambda lane: ranked_lane_pre(cache, sid, lane)),
                ("post_CE", lambda lane: ranked_lane_post(cache, sid, lane, baseline_ce.get((sid, lane)))),
            ]:
                ranked_by_lane = {}
                for lane in ["legislation", "other"]:
                    ranked = rank_fn(lane)
                    ranked_by_lane[lane] = ranked
                    relevant = rel[lane]
                    if not relevant:
                        continue  # no essential gold in this lane for this scenario - skip P/nDCG rows (undefined)
                    for k in K_VALUES:
                        per_lane_rows.append({
                            "split": split_label, "stage": stage, "lane": lane, "k": k,
                            "scenario_id": sid, "n_relevant_in_pool": len(relevant),
                            "precision_at_k": precision_at_k(ranked, relevant, k),
                            "ndcg_at_k": ndcg_at_k(ranked, relevant, k),
                        })
                # cumulative RequirementRecall@K, two-lane OR-pool (k per lane, symmetric)
                for k in K_VALUES:
                    L, O = ranked_by_lane["legislation"][:k], ranked_by_lane["other"][:k]
                    sat = sum(1 for t in targets.values() if req_satisfied(t["chunk_ids"], L, O, k, k))
                    recall_rows.append({
                        "split": split_label, "stage": stage, "k": k, "scenario_id": sid,
                        "n_satisfied": sat, "n_total": len(targets),
                    })

    lane_df = pd.DataFrame(per_lane_rows)
    recall_df = pd.DataFrame(recall_rows)
    lane_df.to_csv(f"{RD}/precision_ndcg_per_scenario.csv", index=False)
    recall_df.to_csv(f"{RD}/cumulative_requirement_recall_per_scenario.csv", index=False)

    # ---------------- aggregate summary tables ----------------
    lane_summary = lane_df.groupby(["split", "stage", "lane", "k"]).agg(
        n_scenarios=("scenario_id", "nunique"),
        precision_at_k_mean=("precision_at_k", "mean"),
        ndcg_at_k_mean=("ndcg_at_k", "mean"),
    ).reset_index()
    lane_summary.to_csv(f"{RD}/precision_ndcg_summary.csv", index=False)

    recall_summary = recall_df.groupby(["split", "stage", "k"]).apply(
        lambda g: pd.Series({
            "n_scenarios": g.scenario_id.nunique(),
            "n_requirements": g.n_total.sum(),
            "requirement_recall_weighted": g.n_satisfied.sum() / g.n_total.sum(),
        })
    ).reset_index()
    recall_summary.to_csv(f"{RD}/cumulative_requirement_recall_summary.csv", index=False)

    print("\n=== Precision@K / nDCG@K summary (mean across scenarios, per lane) ===")
    print(lane_summary.to_string(index=False))
    print("\n=== Cumulative RequirementRecall@K (requirement-weighted, two-lane OR-pool) ===")
    print(recall_summary.to_string(index=False))

    # ---------------- figures ----------------



    print(f"\nwrote CSVs to {RD}/")


if __name__ == "__main__":
    main()
