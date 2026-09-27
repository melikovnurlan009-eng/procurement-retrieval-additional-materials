#!/usr/bin/env python3
"""RQ1 (same-budget, 50-result) and RQ3 (CE on/off at 25/lane) recomputed on the CURRENT
208-scenario benchmark (95 DEV / 94 TEST scoreable), frozen alpha=0.10/beta=0.40/graph-off,
using the CORRECTED evaluator (mandatory_requirements_with_targets: acceptable_chunk_ids
unioned, mandatory-only filter) - the same evaluator now cited as authoritative in the
dissertation's Appendix A. Fixes the budget mismatch in the current RQ1 draft (pooled top-10
vs two-lane 5+5): pooled configs now use top-50 (single global budget), two-lane configs use
25+25 (=50), so every comparison in this script is truly same-budget. Adds bootstrap CI,
permutation p, Cohen's d, and McNemar for the two headline comparisons: RQ1 (source-aware
two-lane vs conventional pooled hybrid+authority+jurisdiction, both @50) and RQ3 (post-CE vs
pre-CE two-lane, both @25-per-lane), TEST split (DEV included for reference).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats as sst

from evaluation.config import DEFAULT_DB, DEFAULT_COLLECTION
from evaluation.common import weighted_retriever, refuse_pool, display_category
from evaluation.ce_metrics_audit import (
    load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, ranked_lane_post,
    req_satisfied, FAQ218, CACHE_PATH,
)
from evaluation.paths import RESULTS_DIR

RD = str(RESULTS_DIR / "rq1_rq3_same_budget")
RESULTS_DIR.joinpath("rq1_rq3_same_budget").mkdir(parents=True, exist_ok=True)

POOLED_CONFIGS = [
    {"id": 1, "name": "BM25 only",                       "alpha": 0.0,  "beta": 1.0,  "jurisdiction": False},
    {"id": 2, "name": "Dense only",                       "alpha": 0.0,  "beta": 0.0,  "jurisdiction": False},
    {"id": 3, "name": "BM25+Dense hybrid",                "alpha": 0.0,  "beta": 0.40, "jurisdiction": False},
    {"id": 4, "name": "Hybrid + authority",                "alpha": 0.10, "beta": 0.40, "jurisdiction": False},
    {"id": 5, "name": "Hybrid + jurisdiction",             "alpha": 0.0,  "beta": 0.40, "jurisdiction": True},
    {"id": 6, "name": "Hybrid + authority + jurisdiction", "alpha": 0.10, "beta": 0.40, "jurisdiction": True},
]
POOLED_K = 50
LANE_K = 25


def pooled_rank(cache, sid, alpha, beta, use_jurisdiction):
    sub = cache[cache.scenario_id == sid]
    if sub.empty:
        return []
    pool = set(sub.chunk_id)
    bm25_raw = dict(zip(sub.chunk_id, sub.bm25_raw))
    dense_raw = dict(zip(sub.chunk_id, sub.dense_raw))
    meta = {row.chunk_id: {"authority_class": row.authority_class, "jurisdiction": row.jurisdiction}
            for row in sub.itertuples()}
    with weighted_retriever(beta=beta, alpha=alpha, db_path=DEFAULT_DB, collection=DEFAULT_COLLECTION) as r:
        if not use_jurisdiction:
            import chunk_retrieval as _cr
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta,
                                  jurisdiction_weights={k: 1.0 for k in _cr.JURISDICTION_WEIGHTS})
        else:
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta)
    ranked = sorted(pool, key=lambda c: -scores[c]["final_score"])
    return ranked


def two_lane_no_authority(cache, sid, lane):
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty:
        return []
    pool = set(sub.chunk_id)
    bm25_raw = dict(zip(sub.chunk_id, sub.bm25_raw))
    dense_raw = dict(zip(sub.chunk_id, sub.dense_raw))
    meta = {row.chunk_id: {"authority_class": row.authority_class, "jurisdiction": row.jurisdiction}
            for row in sub.itertuples()}
    with weighted_retriever(beta=0.40, alpha=0.0, db_path=DEFAULT_DB, collection=DEFAULT_COLLECTION) as r:
        scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta)
    return sorted(pool, key=lambda c: -scores[c]["final_score"])


def req_recall_counts(targets, top_set_or_lanes, is_pooled):
    """Returns (n_satisfied, n_total) - the raw counts needed for requirement-weighted
    aggregation (sum(satisfied)/sum(total) across scenarios), matching the dissertation's
    own stated convention ('requirement recall is requirement-weighted across the
    benchmark', Section 5.3) rather than a scenario-macro-average."""
    if is_pooled:
        top = top_set_or_lanes
        sat = sum(1 for t in targets.values() if t["chunk_ids"] & top)
    else:
        topL, topO = top_set_or_lanes
        sat = sum(1 for t in targets.values() if (t["chunk_ids"] & topL) or (t["chunk_ids"] & topO))
    return sat, len(targets)


def complete_cov(targets, top_set_or_lanes, is_pooled):
    if is_pooled:
        top = top_set_or_lanes
        return 1.0 if all(t["chunk_ids"] & top for t in targets.values()) else 0.0
    topL, topO = top_set_or_lanes
    return 1.0 if all((t["chunk_ids"] & topL) or (t["chunk_ids"] & topO) for t in targets.values()) else 0.0


def paired_bootstrap_ci(a, b, n_boot=10000, seed=20260920):
    """Scenario-macro-averaged metrics (e.g. CompleteCoverage): a/b are one value per
    scenario, resampled at the scenario level."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def paired_permutation_test(a, b, n_perm=10000, seed=20260920):
    rng = np.random.default_rng(seed)
    diffs = a - b
    obs = abs(diffs.mean())
    signs = rng.choice([-1, 1], size=(n_perm, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((perm_means >= obs).mean())


def cohens_d_paired(a, b):
    diffs = a - b
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def weighted_ratio(sat, tot):
    return sat.sum() / tot.sum() if tot.sum() else float("nan")


def cluster_bootstrap_ci_weighted(sat_a, tot_a, sat_b, tot_b, n_boot=10000, seed=20260920):
    """Requirement-weighted RequirementRecall (sum(satisfied)/sum(total)): the unit that
    can be independently resampled without breaking the ratio is the SCENARIO (a cluster of
    requirements), not the requirement itself (requirements within one scenario share the
    same retrieval run). Resamples scenarios with replacement, recomputing the pooled ratio
    for both configs in each resample."""
    rng = np.random.default_rng(seed)
    n = len(sat_a)
    obs_diff = weighted_ratio(sat_a, tot_a) - weighted_ratio(sat_b, tot_b)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = weighted_ratio(sat_a[idx], tot_a[idx]) - weighted_ratio(sat_b[idx], tot_b[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(obs_diff), "ci_low": float(lo), "ci_high": float(hi), "n_scenarios": int(n)}


def cluster_permutation_test_weighted(sat_a, tot_a, sat_b, tot_b, n_perm=10000, seed=20260920):
    """Sign-flip permutation on the same scenario-level clusters: for each scenario, swap
    which config's (satisfied,total) pair counts as A vs B, recompute the pooled ratio diff."""
    rng = np.random.default_rng(seed)
    n = len(sat_a)
    obs = abs(weighted_ratio(sat_a, tot_a) - weighted_ratio(sat_b, tot_b))
    count = 0
    for _ in range(n_perm):
        flip = rng.random(n) < 0.5
        pa_sat = np.where(flip, sat_b, sat_a); pa_tot = np.where(flip, tot_b, tot_a)
        pb_sat = np.where(flip, sat_a, sat_b); pb_tot = np.where(flip, tot_a, tot_b)
        d = abs(weighted_ratio(pa_sat, pa_tot) - weighted_ratio(pb_sat, pb_tot))
        if d >= obs:
            count += 1
    return count / n_perm


def cohens_d_weighted(sat_a, tot_a, sat_b, tot_b):
    """Cohen's d on the per-scenario ratio (satisfied/total per scenario), paired - the
    conventional per-scenario effect-size view, reported alongside the pooled weighted CI."""
    ra = np.divide(sat_a, tot_a, out=np.zeros_like(sat_a, dtype=float), where=tot_a > 0)
    rb = np.divide(sat_b, tot_b, out=np.zeros_like(sat_b, dtype=float), where=tot_b > 0)
    return cohens_d_paired(ra, rb)


def mcnemar_exact(pre_binary, post_binary):
    improved = int(((pre_binary == 0) & (post_binary == 1)).sum())
    worsened = int(((pre_binary == 1) & (post_binary == 0)).sum())
    n_disc = improved + worsened
    p = float(sst.binomtest(min(improved, worsened), n_disc, 0.5, alternative="two-sided").pvalue) if n_disc > 0 else 1.0
    return {"n_improved": improved, "n_worsened": worsened, "n_discordant": n_disc, "mcnemar_exact_p": p}


def main():
    cache_full = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_208.jsonl")}
    scen_by_id = {s["scenario_id"]: s for s in scen_all}

    scen_targets_all = {}
    for sid, g in gold_all.items():
        t = mandatory_requirements_with_targets(g)
        if t:
            scen_targets_all[sid] = t

    baseline_ce = {}
    for (sid, lane), grp in cache_full[cache_full.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    per_scenario_rows = []   # per-scenario, per-config metrics for stats
    per_category_rows = []

    for split_label in ["TEST", "DEV"]:
        split_ids = {s["scenario_id"] for s in scen_all if str(s.get("split", "")).lower() == split_label.lower()}
        cache = cache_full[cache_full.scenario_id.isin(split_ids)]
        scen_targets = {sid: t for sid, t in scen_targets_all.items() if sid in split_ids}
        print(f"\n=== {split_label}: {len(scen_targets)} scoreable scenarios ===")

        # -------- pooled configs @50 --------
        for cfgd in POOLED_CONFIGS:
            for sid, targets in scen_targets.items():
                ranked = pooled_rank(cache, sid, cfgd["alpha"], cfgd["beta"], cfgd["jurisdiction"])
                top = set(ranked[:POOLED_K])
                n_sat, n_tot = req_recall_counts(targets, top, True)
                cc = complete_cov(targets, top, True)
                per_scenario_rows.append({"split": split_label, "config": f"pooled:{cfgd['name']}",
                                           "config_id": cfgd["id"], "budget": "top-50 (pooled)",
                                           "scenario_id": sid, "suite": scen_by_id[sid].get("suite"),
                                           "category": display_category(scen_by_id[sid]),
                                           "n_satisfied": n_sat, "n_total": n_tot, "complete_coverage": cc})
            n = sum(1 for r in per_scenario_rows if r["split"] == split_label and r["config"] == f"pooled:{cfgd['name']}")
            print(f"  pooled@50 [{cfgd['id']}] {cfgd['name']:38s} n={n} done")

        # -------- two-lane, no authority @25+25 --------
        for sid, targets in scen_targets.items():
            L = two_lane_no_authority(cache, sid, "legislation")[:LANE_K]
            O = two_lane_no_authority(cache, sid, "other")[:LANE_K]
            n_sat, n_tot = req_recall_counts(targets, (set(L), set(O)), False)
            cc = complete_cov(targets, (set(L), set(O)), False)
            per_scenario_rows.append({"split": split_label, "config": "two_lane:no_authority",
                                       "config_id": 7, "budget": "25+25 (two-lane)",
                                       "scenario_id": sid, "suite": scen_by_id[sid].get("suite"),
                                       "category": display_category(scen_by_id[sid]),
                                       "n_satisfied": n_sat, "n_total": n_tot, "complete_coverage": cc})
        print(f"  two-lane no-authority @25+25 done")

        # -------- two-lane pre-CE (production first stage) @25+25, and post-CE (adopted baseline) @25+25 --------
        for sid, targets in scen_targets.items():
            L_pre = ranked_lane_pre(cache, sid, "legislation")[:LANE_K]
            O_pre = ranked_lane_pre(cache, sid, "other")[:LANE_K]
            n_sat_pre, n_tot_pre = req_recall_counts(targets, (set(L_pre), set(O_pre)), False)
            cc_pre = complete_cov(targets, (set(L_pre), set(O_pre)), False)
            per_scenario_rows.append({"split": split_label, "config": "two_lane:full_pre_CE",
                                       "config_id": 8, "budget": "25+25 (two-lane)",
                                       "scenario_id": sid, "suite": scen_by_id[sid].get("suite"),
                                       "category": display_category(scen_by_id[sid]),
                                       "n_satisfied": n_sat_pre, "n_total": n_tot_pre, "complete_coverage": cc_pre})

            L_post = ranked_lane_post(cache, sid, "legislation", baseline_ce.get((sid, "legislation")))[:LANE_K]
            O_post = ranked_lane_post(cache, sid, "other", baseline_ce.get((sid, "other")))[:LANE_K]
            n_sat_post, n_tot_post = req_recall_counts(targets, (set(L_post), set(O_post)), False)
            cc_post = complete_cov(targets, (set(L_post), set(O_post)), False)
            per_scenario_rows.append({"split": split_label, "config": "two_lane:full_post_CE",
                                       "config_id": 9, "budget": "25+25 (two-lane)",
                                       "scenario_id": sid, "suite": scen_by_id[sid].get("suite"),
                                       "category": display_category(scen_by_id[sid]),
                                       "n_satisfied": n_sat_post, "n_total": n_tot_post, "complete_coverage": cc_post})
        print(f"  two-lane pre-CE / post-CE @25+25 done")

    df = pd.DataFrame(per_scenario_rows)
    df.to_csv(f"{RD}/scenario_level_all_configs.csv", index=False)

    def weighted_agg(g):
        return pd.Series({
            "n_scenarios": len(g),
            "n_requirements": int(g["n_total"].sum()),
            "requirement_recall_weighted": g["n_satisfied"].sum() / g["n_total"].sum() if g["n_total"].sum() else None,
            "requirement_recall_scenario_macro_avg": (g["n_satisfied"] / g["n_total"]).mean(),
            "complete_coverage_mean": g["complete_coverage"].mean(),
        })

    # ---------------- summary table (same-budget 50-result baselines) ----------------
    summary = df.groupby(["split", "config", "config_id", "budget"]).apply(weighted_agg).reset_index()
    summary = summary.sort_values(["split", "config_id"])
    summary.to_csv(f"{RD}/rq1_same_budget_50_baselines.csv", index=False)
    print("\n=== Same-budget (50-result) RQ1 baselines (requirement-weighted) ===")
    print(summary.to_string(index=False))

    # ---------------- per-category table (RQ1: conventional pooled hybrid+auth+juris vs two-lane pre-CE, both @50) ----------------
    cat = df[df.config.isin(["pooled:Hybrid + authority + jurisdiction", "two_lane:full_pre_CE"])]
    cat_summary = cat.groupby(["split", "config", "category"]).apply(weighted_agg).reset_index()
    cat_summary.to_csv(f"{RD}/rq1_per_category.csv", index=False)
    print("\n=== RQ1 per-category (conventional pooled vs source-aware two-lane, both @50, requirement-weighted) ===")
    print(cat_summary.to_string(index=False))

    # ---------------- RQ3: CE on/off @25-per-lane, per-category too ----------------
    ce = df[df.config.isin(["two_lane:full_pre_CE", "two_lane:full_post_CE"])]
    ce_cat = ce.groupby(["split", "config", "category"]).apply(weighted_agg).reset_index()
    ce_cat.to_csv(f"{RD}/rq3_ce_onoff_per_category.csv", index=False)
    print("\n=== RQ3 CE on/off @25-per-lane by category (requirement-weighted) ===")
    print(ce_cat.to_string(index=False))

    # ---------------- statistical tests, TEST split ----------------
    stats_out = {}
    test_df = df[df.split == "TEST"]

    def get(config_name, col):
        sub = test_df[test_df.config == config_name].set_index("scenario_id")[col]
        return sub

    def rq_block(name_a, name_b, label_a, label_b):
        sat_a = get(name_a, "n_satisfied"); tot_a = get(name_a, "n_total")
        sat_b = get(name_b, "n_satisfied"); tot_b = get(name_b, "n_total")
        cc_a = get(name_a, "complete_coverage"); cc_b = get(name_b, "complete_coverage")
        common_ids = sorted(set(sat_a.index) & set(sat_b.index))
        sat_a, tot_a = sat_a.loc[common_ids].values, tot_a.loc[common_ids].values
        sat_b, tot_b = sat_b.loc[common_ids].values, tot_b.loc[common_ids].values
        cc_a, cc_b = cc_a.loc[common_ids].values, cc_b.loc[common_ids].values
        return {
            "n_scenarios": len(common_ids),
            "n_requirements": int(tot_a.sum()),
            "requirement_recall": {
                **cluster_bootstrap_ci_weighted(sat_a, tot_a, sat_b, tot_b),
                "permutation_p": cluster_permutation_test_weighted(sat_a, tot_a, sat_b, tot_b),
                "cohens_d_paired_per_scenario": cohens_d_weighted(sat_a, tot_a, sat_b, tot_b),
                f"mean_{label_a}_weighted": float(weighted_ratio(sat_a, tot_a)),
                f"mean_{label_b}_weighted": float(weighted_ratio(sat_b, tot_b)),
            },
            "complete_coverage": {
                **paired_bootstrap_ci(cc_a, cc_b), "permutation_p": paired_permutation_test(cc_a, cc_b),
                "cohens_d_paired": cohens_d_paired(cc_a, cc_b), **mcnemar_exact(cc_b, cc_a),
                f"mean_{label_a}": float(cc_a.mean()), f"mean_{label_b}": float(cc_b.mean()),
            },
        }

    # RQ1: source-aware two-lane pre-CE @25+25 (=50) vs conventional pooled hybrid+auth+juris @50, TEST
    stats_out["RQ1_source_aware_vs_conventional_TEST"] = rq_block(
        "two_lane:full_pre_CE", "pooled:Hybrid + authority + jurisdiction", "two_lane", "pooled")

    # RQ3: post-CE vs pre-CE two-lane @25-per-lane, TEST
    stats_out["RQ3_post_CE_vs_pre_CE_TEST_25perlane"] = rq_block(
        "two_lane:full_post_CE", "two_lane:full_pre_CE", "post_CE", "pre_CE")

    json.dump(stats_out, open(f"{RD}/statistical_tests_TEST.json", "w"), indent=2)
    print("\n=== Statistical tests (TEST) ===")
    print(json.dumps(stats_out, indent=2))
    print(f"\nwrote outputs to {RD}/")


if __name__ == "__main__":
    main()
