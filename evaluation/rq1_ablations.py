#!/usr/bin/env python3
"""RQ1: minimum defensible ablation set isolating the effect of source-aware controls.

Configs 1-6 are CONVENTIONAL (non-two-lane) baselines: both lanes' candidates pooled into
one globally-ranked list, top-10 taken as a single budget - this is the correct way to
represent "what would a normal lexical/dense/hybrid retriever do" for a system that was
never told about the legislation/other split. Configs 7-9 are the actual two-lane system
at varying ablation settings, scored with the lane-independent OR-pool rule (5 legislation
+ 5 other, never merged/compared cross-lane), matching production semantics.

Config 9 (full + graph) uses a SEPARATE graph-off cache (real retrieval re-run with
use_graph=False), not an approximation - see build_candidate_cache.py --use-graph 0.

Configs:
 1. BM25 only        (pooled, beta=1.0, alpha=0.0, no jurisdiction)
 2. Dense only        (pooled, beta=0.0, alpha=0.0, no jurisdiction)
 3. BM25+Dense hybrid (pooled, beta=0.40, alpha=0.0, no jurisdiction)
 4. Hybrid+authority  (pooled, beta=0.40, alpha=0.30, no jurisdiction)
 5. Hybrid+jurisdiction (pooled, beta=0.40, alpha=0.0, WITH jurisdiction)
 6. Hybrid+authority+jurisdiction (pooled, beta=0.40, alpha=0.30, WITH jurisdiction)
 7. Two-lane hybrid, no authority (two-lane, beta=0.40, alpha=0.0, WITH jurisdiction, 5+5)
 8. Full source-aware two-lane (two-lane, beta=0.40, alpha=0.30, WITH jurisdiction, 5+5) - PRODUCTION
 9. Full config + graph (config 8's settings, graph-on cache vs graph-off cache, 5+5)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from evaluation.config import add_common_args, config_from_args, LEGISLATION_CLASSES
from evaluation.common import (
    load_scenarios, load_gold, essential_targets, all_gold_chunk_ids, wrong_regime_chunk_ids,
    requirement_recall_at_k, complete_coverage, global_requirement_recall_at_k, global_complete_coverage,
    weighted_retriever, refuse_pool, display_category,
)

CONFIGS = [
    {"id": 1, "name": "BM25 only",                      "mode": "pooled", "alpha": 0.0,  "beta": 1.0, "jurisdiction": False},
    {"id": 2, "name": "Dense only",                      "mode": "pooled", "alpha": 0.0,  "beta": 0.0, "jurisdiction": False},
    {"id": 3, "name": "BM25+Dense hybrid",                "mode": "pooled", "alpha": 0.0,  "beta": 0.40, "jurisdiction": False},
    {"id": 4, "name": "Hybrid + authority",               "mode": "pooled", "alpha": 0.30, "beta": 0.40, "jurisdiction": False},
    {"id": 5, "name": "Hybrid + jurisdiction",            "mode": "pooled", "alpha": 0.0,  "beta": 0.40, "jurisdiction": True},
    {"id": 6, "name": "Hybrid + authority + jurisdiction","mode": "pooled", "alpha": 0.30, "beta": 0.40, "jurisdiction": True},
    {"id": 7, "name": "Two-lane hybrid, no authority",    "mode": "two_lane", "alpha": 0.0,  "beta": 0.40, "jurisdiction": True},
    {"id": 8, "name": "Full source-aware two-lane (PRODUCTION, true scoring, graph ON)", "mode": "two_lane_true", "alpha": 0.30, "beta": 0.40, "jurisdiction": True},
    {"id": 9, "name": "Full config, graph OFF (true scoring)", "mode": "two_lane_graph_off", "alpha": 0.30, "beta": 0.40, "jurisdiction": True},
]


def rescore(cache: pd.DataFrame, sid: str, alpha: float, beta: float, use_jurisdiction: bool,
            db_path, collection, lane: str | None = None) -> pd.DataFrame:
    """Recompute final_score for scenario `sid`'s candidates (one lane, or pooled across
    both if lane=None) at the given alpha/beta, optionally zeroing jurisdiction weighting."""
    sub = cache[cache.scenario_id == sid]
    if lane is not None:
        sub = sub[sub.lane == lane]
    if sub.empty:
        return sub
    pool = set(sub.chunk_id)
    bm25_raw = dict(zip(sub.chunk_id, sub.bm25_raw))
    dense_raw = dict(zip(sub.chunk_id, sub.dense_raw))
    meta = {row.chunk_id: {"authority_class": row.authority_class, "jurisdiction": row.jurisdiction}
            for row in sub.itertuples()}
    with weighted_retriever(beta=beta, alpha=alpha, db_path=db_path, collection=collection) as r:
        if not use_jurisdiction:
            import chunk_retrieval as _cr
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta,
                                  jurisdiction_weights={k: 1.0 for k in _cr.JURISDICTION_WEIGHTS})
        else:
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta)
    out = sub.copy()
    out["rescored_final"] = out["chunk_id"].map(lambda c: scores[c]["final_score"])
    return out


def evaluate(cache: pd.DataFrame, scenarios: list[dict], gold: dict, cfg_spec: dict,
             db_path, collection, graph_off_cache: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    rows = []
    use_cache = graph_off_cache if cfg_spec["mode"] == "two_lane_graph_off" and graph_off_cache is not None else None
    for sc in scenarios:
        sid = sc["scenario_id"]
        g = gold.get(sid, {})
        targets = essential_targets(g)
        if not targets:
            continue
        gold_all = all_gold_chunk_ids(g)
        wrong_ids = wrong_regime_chunk_ids(g)

        if cfg_spec["mode"] == "pooled":
            df = rescore(cache, sid, cfg_spec["alpha"], cfg_spec["beta"], cfg_spec["jurisdiction"],
                         db_path, collection, lane=None)
            ranked = df.sort_values("rescored_final", ascending=False)["chunk_id"].tolist() if not df.empty else []
            rr = global_requirement_recall_at_k(ranked, targets, 10)
            cc = global_complete_coverage(ranked, targets, 10)
            top10 = set(ranked[:10])
            id_to_class = dict(zip(df.chunk_id, df.authority_class)) if not df.empty else {}
        elif cfg_spec["mode"] in ("two_lane_true", "two_lane_graph_off"):
            # TRUE production scoring: use the cache's OWN stored final_score directly (the
            # real scoring path used at candidate-cache build time, including graph's RRF
            # channel contribution) rather than the refuse_pool reconstruction. refuse_pool
            # only reconstructs bm25+dense+authority and silently drops graph's contribution -
            # using it for BOTH the graph-on and graph-off comparison points would make the
            # graph ablation vacuous (verified: an earlier version of this script did exactly
            # that and produced numerically identical results for "graph on" and "graph off").
            src = cache if cfg_spec["mode"] == "two_lane_true" else use_cache
            L = src[(src.scenario_id == sid) & (src.lane == "legislation")].sort_values("final_score", ascending=False)["chunk_id"].tolist()
            O = src[(src.scenario_id == sid) & (src.lane == "other")].sort_values("final_score", ascending=False)["chunk_id"].tolist()
            rr = requirement_recall_at_k(L, O, targets, 5, 5)
            cc = complete_coverage(L, O, targets, 5, 5)
            top10 = set(L[:5]) | set(O[:5])
            sub_leg = src[(src.scenario_id == sid) & (src.lane == "legislation")]
            sub_oth = src[(src.scenario_id == sid) & (src.lane == "other")]
            id_to_class = dict(zip(sub_leg.chunk_id, sub_leg.authority_class))
            id_to_class.update(dict(zip(sub_oth.chunk_id, sub_oth.authority_class)))
        else:
            src = cache if use_cache is None else use_cache
            L_df = rescore(src, sid, cfg_spec["alpha"], cfg_spec["beta"], cfg_spec["jurisdiction"],
                            db_path, collection, lane="legislation")
            O_df = rescore(src, sid, cfg_spec["alpha"], cfg_spec["beta"], cfg_spec["jurisdiction"],
                            db_path, collection, lane="other")
            L = L_df.sort_values("rescored_final", ascending=False)["chunk_id"].tolist() if not L_df.empty else []
            O = O_df.sort_values("rescored_final", ascending=False)["chunk_id"].tolist() if not O_df.empty else []
            rr = requirement_recall_at_k(L, O, targets, 5, 5)
            cc = complete_coverage(L, O, targets, 5, 5)
            top10 = set(L[:5]) | set(O[:5])
            id_to_class = dict(zip(L_df.chunk_id, L_df.authority_class)) if not L_df.empty else {}
            id_to_class.update(dict(zip(O_df.chunk_id, O_df.authority_class)) if not O_df.empty else {})

        n_primary = sum(1 for c in top10 if id_to_class.get(c) == "PRIMARY_LEGISLATION")
        n_secondary = sum(1 for c in top10 if id_to_class.get(c) == "SECONDARY_LEGISLATION")
        n_guidance = sum(1 for c in top10 if id_to_class.get(c) not in LEGISLATION_CLASSES)
        rows.append({
            "scenario_id": sid, "split": sc.get("split"), "suite": sc.get("suite"),
            "category": display_category(sc), "regime_context": sc.get("regime_context"),
            "requirement_recall_10": rr, "complete_coverage_10": cc,
            "wrong_regime_in_10": len(top10 & wrong_ids),
            "n_primary_in_10": n_primary, "n_secondary_in_10": n_secondary, "n_guidance_in_10": n_guidance,
            "candidate_recall_75": (len(set(ranked[:75] if cfg_spec['mode']=='pooled' else list(L[:75])+list(O[:75])) & gold_all) / len(gold_all)) if gold_all else None,
        })
    df = pd.DataFrame(rows)
    summary = {
        "config_id": cfg_spec["id"], "config_name": cfg_spec["name"], "mode": cfg_spec["mode"],
        "alpha": cfg_spec["alpha"], "beta": cfg_spec["beta"], "jurisdiction": cfg_spec["jurisdiction"],
        "n_scenarios": len(df),
        "requirement_recall_mean": df["requirement_recall_10"].mean() if len(df) else None,
        "complete_coverage_mean": df["complete_coverage_10"].mean() if len(df) else None,
        "wrong_regime_rate_mean": (df["wrong_regime_in_10"] / 10.0).mean() if len(df) else None,
        "mean_primary_in_10": df["n_primary_in_10"].mean() if len(df) else None,
        "mean_secondary_in_10": df["n_secondary_in_10"].mean() if len(df) else None,
        "mean_guidance_in_10": df["n_guidance_in_10"].mean() if len(df) else None,
        "candidate_recall_75_mean": df["candidate_recall_75"].mean() if len(df) else None,
    }
    return summary, df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--graph-off-cache", type=Path, default=None)
    args = ap.parse_args()
    cfg = config_from_args(args)
    cache_path = args.cache or (cfg.results_dir / "candidate_cache.parquet")
    cache = pd.read_parquet(cache_path)
    graph_off_cache = pd.read_parquet(args.graph_off_cache) if args.graph_off_cache else None
    scenarios = load_scenarios(cfg.scenarios_path, cfg.split)
    gold = load_gold(cfg.gold_path)
    if cfg.split != "ALL":
        want = {s["scenario_id"] for s in scenarios}
        cache = cache[cache.scenario_id.isin(want)]
        if graph_off_cache is not None:
            graph_off_cache = graph_off_cache[graph_off_cache.scenario_id.isin(want)]

    summaries, all_rows, cat_rows = [], [], []
    for c in CONFIGS:
        if c["mode"] == "two_lane_graph_off" and graph_off_cache is None:
            print(f"SKIPPING config {c['id']} ({c['name']}) - no --graph-off-cache provided")
            continue
        summ, df = evaluate(cache, scenarios, gold, c, cfg.db_path, cfg.collection, graph_off_cache)
        summaries.append(summ)
        df["config_id"] = c["id"]; df["config_name"] = c["name"]
        all_rows.append(df)
        print(f"[{c['id']}] {c['name']:38s} RR@10={summ['requirement_recall_mean']:.4f} "
              f"CC@10={summ['complete_coverage_mean']:.4f} "
              f"wrong_regime={summ['wrong_regime_rate_mean']:.4f} "
              f"primary={summ['mean_primary_in_10']:.2f} secondary={summ['mean_secondary_in_10']:.2f} "
              f"guidance={summ['mean_guidance_in_10']:.2f}")

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(cfg.results_dir / "rq1_ablation_summary.csv", index=False)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(cfg.results_dir / "rq1_ablation_scenario_level.csv", index=False)

    cat_summary = all_df.groupby(["config_id", "config_name", "category"]).agg(
        n=("scenario_id", "size"),
        requirement_recall_mean=("requirement_recall_10", "mean"),
        complete_coverage_mean=("complete_coverage_10", "mean"),
    ).reset_index()
    cat_summary.to_csv(cfg.results_dir / "rq1_ablation_by_category.csv", index=False)

    regime_summary = all_df.groupby(["config_id", "config_name", "regime_context"]).agg(
        n=("scenario_id", "size"),
        requirement_recall_mean=("requirement_recall_10", "mean"),
        complete_coverage_mean=("complete_coverage_10", "mean"),
        wrong_regime_mean=("wrong_regime_in_10", "mean"),
    ).reset_index()
    regime_summary.to_csv(cfg.results_dir / "rq1_ablation_by_regime.csv", index=False)

    print(f"\nwrote rq1_ablation_summary.csv, rq1_ablation_scenario_level.csv, "
          f"rq1_ablation_by_category.csv, rq1_ablation_by_regime.csv to {cfg.results_dir}")


if __name__ == "__main__":
    main()
