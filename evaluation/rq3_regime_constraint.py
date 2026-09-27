#!/usr/bin/env python3
"""RQ3: explicit regime-compatibility constraint applied AFTER cross-encoder reranking.

QUERY-TIME REGIME CLASSIFIER (the only "regime inference" this script performs - uses
ONLY the query's own text, never gold `regime_context`/`regime_confidence`/gold evidence):
a small explicit keyword list per regime is matched against the lower-cased query text;
whichever side has strictly more distinct keyword hits wins; a tie (including 0-0) yields
UNKNOWN, deliberately preferring "don't know" over a forced guess (per Section 18's
explicit instruction to allow UNKNOWN/MIXED). The classifier's accuracy against gold
regime_context is reported ONLY as a DEV-set diagnostic (standard practice for evaluating
any classifier against held-out labels) - it is NEVER an input to scoring.

INTERVENTION: for candidates in the CE-reranked top-75/lane pool, if the classifier is
confident (not UNKNOWN) and the candidate's OWN chunk-level `legal_regime` (corpus
metadata - every chunk's regime is intrinsic to which Act/instrument it comes from, not a
per-query gold label) is the OPPOSING regime to the classified query regime, its ce_score
is multiplied by a penalty factor theta < 1.0 and the reranked pool is re-sorted. Regime-
neutral candidates (legal_regime is None/guidance material - the majority of the corpus)
are NEVER penalized. theta is selected on DEV only (grid below), then frozen.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from evaluation.config import add_common_args, config_from_args
from evaluation.common import load_scenarios, load_gold, essential_targets, wrong_regime_chunk_ids, requirement_recall_at_k, complete_coverage

# ---------------------------------------------------------------------------- classifier
# Empirically informed by grepping actual DEV query text for candidate terms (query text
# only - no gold labels read at classification time; inspecting example TEXT to design
# keywords is ordinary feature engineering, not a gold-label leak). Finding: PA2023 is the
# largely UNMARKED default in this benchmark's natural practitioner phrasing (scenarios
# rarely name the Act), while PCR2015 scenarios are reliably marked by explicit legacy/
# contrast framing ("before the new Procurement Act rules came in"). This asymmetry is
# real, not a design flaw - documented and reported honestly rather than hidden.
PA2023_KEYWORDS = [
    "procurement act 2023", "pa 2023", "pa2023", "procurement regulations 2024", "pr 2024", "pr2024",
    "transparency notice", "competitive flexible procedure", "direct award notice", "direct award",
    "kpi notice", "payments compliance notice", "the new regime", "under the act now",
    "central digital platform", "assessment summary", "preliminary market engagement",
    "24 february 2025", "since february 2025", "after february 2025", "new procurement rules",
    "dynamic market", "mandatory standstill", "conflicts of interest register", "below-threshold",
    "contract change notice",
]
PCR2015_KEYWORDS = [
    "pcr 2015", "pcr2015", "public contracts regulations 2015", "the 2015 regulations",
    "ojeu", "regulation 84", "standstill letter", "most economically advantageous tender", " meat ",
    "before 24 february 2025", "before february 2025", "existing procurement", "already underway",
    "started before", "saved regulations", "old regime", "the old rules", "light touch regime",
    "old procurement regulations", "old regulations", "before the new procurement act",
    "before the new rules", "before the new act", "before the act came into force",
    "new procurement act rules came", "new procurement act came", "a couple of years before",
    "just before the new",
]


def classify_query_regime(query: str) -> tuple[str, int, int]:
    """Query-text-only classifier. Returns (label, n_pa_hits, n_pcr_hits).
    label in {"PA2023", "PCR2015", "UNKNOWN"}."""
    q = (query or "").lower()
    n_pa = sum(1 for kw in PA2023_KEYWORDS if kw in q)
    n_pcr = sum(1 for kw in PCR2015_KEYWORDS if kw in q)
    if n_pa > n_pcr:
        return "PA2023", n_pa, n_pcr
    if n_pcr > n_pa:
        return "PCR2015", n_pa, n_pcr
    return "UNKNOWN", n_pa, n_pcr


def incompatible(query_regime: str, candidate_regime: str | None) -> bool:
    if query_regime == "UNKNOWN" or candidate_regime is None or (isinstance(candidate_regime, float)):
        return False
    if query_regime == "PA2023":
        return candidate_regime == "PCR2015"
    if query_regime == "PCR2015":
        return candidate_regime in ("PA2023", "PR2024")
    return False


# ---------------------------------------------------------------------------- ranking
def ranked_lane_constrained(cache: pd.DataFrame, sid: str, lane: str, query_regime: str, theta: float) -> list[str]:
    """post-CE order (reranked head by ce_rank, tail by pre_rerank_rank - matches ce_analysis's
    construction), then a penalty applied to regime-incompatible candidates INSIDE the
    reranked head only (the constraint operates on the CE output, per Section 18: 'AFTER or
    DURING reranking'), re-sorted by the penalized score."""
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty:
        return []
    reranked = sub[sub.ce_rank.notna()].copy()
    tail = sub[sub.ce_rank.isna()].sort_values("pre_rerank_rank")
    if not reranked.empty:
        reranked["adj_score"] = reranked.apply(
            lambda r: r["ce_score"] * theta if incompatible(query_regime, r["legal_regime"]) else r["ce_score"],
            axis=1)
        reranked = reranked.sort_values("adj_score", ascending=False)
    return pd.concat([reranked, tail])["chunk_id"].tolist()


def ranked_lane_post_ce(cache: pd.DataFrame, sid: str, lane: str) -> list[str]:
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    reranked = sub[sub.ce_rank.notna()].sort_values("ce_rank")
    tail = sub[sub.ce_rank.isna()].sort_values("pre_rerank_rank")
    return pd.concat([reranked, tail])["chunk_id"].tolist()


def ranked_lane_pre_ce(cache: pd.DataFrame, sid: str, lane: str) -> list[str]:
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    return sub.sort_values("pre_rerank_rank")["chunk_id"].tolist()


# ---------------------------------------------------------------------------- eval
def evaluate(cache: pd.DataFrame, scenarios: list[dict], gold: dict, theta: float | None) -> pd.DataFrame:
    """theta=None means 'post-CE, no constraint' (baseline); a float applies the constraint."""
    rows = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        query = sc.get("query", "")
        g = gold.get(sid, {})
        targets = essential_targets(g)
        if not targets:
            continue
        wrong_ids = wrong_regime_chunk_ids(g)
        qregime, n_pa, n_pcr = classify_query_regime(query)
        if theta is None:
            L = ranked_lane_post_ce(cache, sid, "legislation")
            O = ranked_lane_post_ce(cache, sid, "other")
        else:
            L = ranked_lane_constrained(cache, sid, "legislation", qregime, theta)
            O = ranked_lane_constrained(cache, sid, "other", qregime, theta)
        top10 = set(L[:5]) | set(O[:5])
        sub = cache[(cache.scenario_id == sid) & (cache.chunk_id.isin(top10))]
        cand_regime = dict(zip(sub.chunk_id, sub.legal_regime))
        n_incompatible_in_10 = sum(1 for c in top10 if incompatible(qregime, cand_regime.get(c)))
        rows.append({
            "scenario_id": sid, "split": sc.get("split"), "regime_context_gold": sc.get("regime_context"),
            "classified_regime": qregime, "classifier_confident": qregime != "UNKNOWN",
            "requirement_recall_10": requirement_recall_at_k(L, O, targets, 5, 5),
            "complete_coverage_10": complete_coverage(L, O, targets, 5, 5),
            "wrong_regime_in_10_gold": len(top10 & wrong_ids),
            "n_regime_incompatible_in_10_broad": n_incompatible_in_10,
        })
    return pd.DataFrame(rows)


def classifier_accuracy(df: pd.DataFrame) -> dict:
    """DEV-only diagnostic: classifier label vs gold regime_context. NEVER used at scoring
    time - this is purely to characterize the classifier's reliability for the report."""
    known = df[df.regime_context_gold.isin(["PA2023", "PCR2015"])]
    confident = known[known.classified_regime != "UNKNOWN"]
    correct = (confident.classified_regime == confident.regime_context_gold).sum()
    return {
        "n_scenarios_with_known_gold_regime": len(known),
        "n_classifier_confident": len(confident),
        "n_classifier_unknown": len(known) - len(confident),
        "coverage": len(confident) / len(known) if len(known) else None,
        "accuracy_when_confident": correct / len(confident) if len(confident) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--mode", choices=["dev_grid", "test_frozen"], required=True)
    ap.add_argument("--theta", type=float, default=None, help="required for --mode test_frozen")
    args = ap.parse_args()
    cfg = config_from_args(args)
    cache_path = args.cache or (cfg.results_dir / "candidate_cache.parquet")
    cache = pd.read_parquet(cache_path)
    scenarios = load_scenarios(cfg.scenarios_path, cfg.split)
    gold = load_gold(cfg.gold_path)
    if cfg.split != "ALL":
        want = {s["scenario_id"] for s in scenarios}
        cache = cache[cache.scenario_id.isin(want)]
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "dev_grid":
        base_df = evaluate(cache, scenarios, gold, theta=None)
        acc = classifier_accuracy(base_df)
        print("=== classifier DEV diagnostic (evaluation only, not used at scoring time) ===")
        print(json.dumps(acc, indent=2))
        base_df.to_csv(cfg.results_dir / "rq3_classifier_diagnostic.csv", index=False)

        grid_rows = []
        THETAS = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
        for theta in THETAS:
            df = base_df if theta == 1.0 else evaluate(cache, scenarios, gold, theta=theta)
            grid_rows.append({
                "theta": theta,
                "requirement_recall_mean": df.requirement_recall_10.mean(),
                "complete_coverage_mean": df.complete_coverage_10.mean(),
                "wrong_regime_gold_mean": df.wrong_regime_in_10_gold.mean(),
                "n_regime_incompatible_broad_mean": df.n_regime_incompatible_in_10_broad.mean(),
                "n_scenarios_with_incompatible_gt0": (df.n_regime_incompatible_in_10_broad > 0).sum(),
            })
            print(f"theta={theta:.1f}: RR@10={grid_rows[-1]['requirement_recall_mean']:.4f} "
                  f"CC@10={grid_rows[-1]['complete_coverage_mean']:.4f} "
                  f"wrong_regime_gold={grid_rows[-1]['wrong_regime_gold_mean']:.4f} "
                  f"incompatible_broad={grid_rows[-1]['n_regime_incompatible_broad_mean']:.4f}")
        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_csv(cfg.results_dir / "rq3_theta_grid.csv", index=False)
        print(f"\nwrote rq3_classifier_diagnostic.csv, rq3_theta_grid.csv to {cfg.results_dir}")

    else:  # test_frozen
        assert args.theta is not None, "--theta required for --mode test_frozen"
        pre_ce_rows = []
        for sc in scenarios:
            sid = sc["scenario_id"]
            g = gold.get(sid, {})
            targets = essential_targets(g)
            if not targets:
                continue
            wrong_ids = wrong_regime_chunk_ids(g)
            L = ranked_lane_pre_ce(cache, sid, "legislation")
            O = ranked_lane_pre_ce(cache, sid, "other")
            top10 = set(L[:5]) | set(O[:5])
            pre_ce_rows.append({"scenario_id": sid,
                                 "requirement_recall_10": requirement_recall_at_k(L, O, targets, 5, 5),
                                 "complete_coverage_10": complete_coverage(L, O, targets, 5, 5),
                                 "wrong_regime_in_10_gold": len(top10 & wrong_ids)})
        pre_ce_df = pd.DataFrame(pre_ce_rows)
        post_ce_df = evaluate(cache, scenarios, gold, theta=None)
        constrained_df = evaluate(cache, scenarios, gold, theta=args.theta)

        summary = pd.DataFrame([
            {"stage": "1_pre_CE_source_aware", "requirement_recall_mean": pre_ce_df.requirement_recall_10.mean(),
             "complete_coverage_mean": pre_ce_df.complete_coverage_10.mean(),
             "wrong_regime_gold_mean": pre_ce_df.wrong_regime_in_10_gold.mean(), "n": len(pre_ce_df)},
            {"stage": "2_post_CE", "requirement_recall_mean": post_ce_df.requirement_recall_10.mean(),
             "complete_coverage_mean": post_ce_df.complete_coverage_10.mean(),
             "wrong_regime_gold_mean": post_ce_df.wrong_regime_in_10_gold.mean(), "n": len(post_ce_df)},
            {"stage": f"3_post_CE_plus_regime_constraint(theta={args.theta})",
             "requirement_recall_mean": constrained_df.requirement_recall_10.mean(),
             "complete_coverage_mean": constrained_df.complete_coverage_10.mean(),
             "wrong_regime_gold_mean": constrained_df.wrong_regime_in_10_gold.mean(), "n": len(constrained_df)},
        ])
        summary.to_csv(cfg.results_dir / "rq3_test_frozen_summary.csv", index=False)
        post_ce_df.to_csv(cfg.results_dir / "rq3_test_post_ce_scenario_level.csv", index=False)
        constrained_df.to_csv(cfg.results_dir / "rq3_test_constrained_scenario_level.csv", index=False)
        print(summary.to_string(index=False))
        print(f"\nwrote rq3_test_frozen_summary.csv + scenario-level CSVs to {cfg.results_dir}")


if __name__ == "__main__":
    main()
