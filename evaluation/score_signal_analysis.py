#!/usr/bin/env python3
"""Final retrieval score / signal analysis, DEV (mechanism) + TEST (confirmation only).
No retrieval logic, benchmark, CE model, or production config is changed. Reuses the
already-frozen candidate_cache.parquet (alpha=0.10, beta=0.40, built once) and the
already-computed baseline/metadata-enriched CE outputs from the prior CE experiment.
Gold tags use the CORRECTED essential-target definition (acceptable_chunk_ids unioned,
mandatory filter applied) from evaluation/ce_metrics_audit.py, not the raw cache booleans.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import roc_auc_score
from evaluation.ce_metrics_audit import (
    load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, ranked_lane_post,
    ranked_lane_fused, req_satisfied, LAW_ROLES, NONLAW_ROLES, FAQ218, CACHE_PATH, V2_PATH,
)

from evaluation.paths import RESULTS_DIR

RD = str(RESULTS_DIR)
import os
os.makedirs(RD, exist_ok=True)

SOURCE_GROUP = {
    "PRIMARY_LEGISLATION": "primary_legislation", "SECONDARY_LEGISLATION": "secondary_legislation",
    "OFFICIAL_GOVERNMENT_GUIDANCE": "official_guidance", "OFFICIAL_TECHNICAL_GUIDANCE": "official_guidance",
    "OFFICIAL_PA23_TECHNICAL_GUIDANCE": "official_guidance", "OFFICIAL_SPECIALIST_GUIDANCE": "official_guidance",
    "OFFICIAL_PRACTICE_GUIDANCE": "official_guidance", "OFFICIAL_WORKFLOW": "workflow",
    "OFFICIAL_TRAINING": "workflow", "OFFICIAL_REGULATOR_GUIDANCE": "regulator_guidance",
    "PROCUREMENT_POLICY": "regulator_guidance", "PROFESSIONAL_INTERPRETATION": "professional_interpretation",
    "PROFESSIONAL_CASE_ANALYSIS": "professional_interpretation", "NON_AUTHORITATIVE_PROFESSIONAL": "professional_interpretation",
    "INDUSTRY_PRACTICE": "professional_interpretation",
}
AUTHORITY_WEIGHTS = {
    "PRIMARY_LEGISLATION": 1.00, "SECONDARY_LEGISLATION": 0.97, "OFFICIAL_TECHNICAL_GUIDANCE": 0.90,
    "OFFICIAL_GOVERNMENT_GUIDANCE": 0.88, "OFFICIAL_REGULATOR_GUIDANCE": 0.86, "PROCUREMENT_POLICY": 0.84,
    "OFFICIAL_WORKFLOW": 0.78, "OFFICIAL_TRAINING": 0.74, "PROFESSIONAL_INTERPRETATION": 0.62,
    "PROFESSIONAL_CASE_ANALYSIS": 0.62, "NON_AUTHORITATIVE_PROFESSIONAL": 0.62,
    "OFFICIAL_SPECIALIST_GUIDANCE": 0.86, "OFFICIAL_PA23_TECHNICAL_GUIDANCE": 0.90, "OFFICIAL_PRACTICE_GUIDANCE": 0.84,
    "INDUSTRY_PRACTICE": 0.55,
}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return None
    gt = sum((x > y) for x in a for y in b) if len(a) * len(b) < 2_000_000 else None
    if gt is None:
        # vectorized for larger sizes
        a_sorted = np.sort(a)
        cum = 0
        for y in b:
            cum += np.searchsorted(a_sorted, y, side='left')  # count a < y
        less = cum
        gt2 = 0
        for y in b:
            gt2 += len(a) - np.searchsorted(a_sorted, y, side='right')
        greater = gt2
        n = len(a) * len(b)
        return (greater - less) / n
    lt = sum((x < y) for x in a for y in b)
    n = len(a) * len(b)
    return (gt - lt) / n


def mw_cliffs(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n1": len(a), "n2": len(b), "U": None, "p": None, "cliffs_delta": None}
    U, p = sstats.mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    return {"n1": len(a), "n2": len(b), "U": float(U), "p": float(p), "cliffs_delta": round(d, 4) if d is not None else None}


def describe(vals, prefix=""):
    v = pd.Series(vals).dropna().astype(float)
    if len(v) == 0:
        return {}
    q = v.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        f"{prefix}n": len(v), f"{prefix}min": v.min(), f"{prefix}max": v.max(), f"{prefix}mean": v.mean(),
        f"{prefix}median": v.median(), f"{prefix}sd": v.std(), f"{prefix}iqr": q[0.75] - q[0.25],
        f"{prefix}q05": q[0.05], f"{prefix}q10": q[0.10], f"{prefix}q25": q[0.25], f"{prefix}q50": q[0.50],
        f"{prefix}q75": q[0.75], f"{prefix}q90": q[0.90], f"{prefix}q95": q[0.95],
    }


def main():
    cache_full = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_218.jsonl")}
    scen_by_id = {s["scenario_id"]: s for s in scen_all}

    dev_scen = [s for s in scen_all if str(s.get("split", "")).lower() == "dev"]
    dev_ids = {s["scenario_id"] for s in dev_scen}
    test_ids = {s["scenario_id"] for s in scen_all if str(s.get("split", "")).lower() == "test"}

    # corrected gold target sets, ALL scenarios (not just DEV) so TEST confirmation can reuse them
    scen_targets_all = {}
    for sid, g in gold_all.items():
        t = mandatory_requirements_with_targets(g)
        if t:
            scen_targets_all[sid] = t

    def essential_chunk_ids(sid):
        return set().union(*[t["chunk_ids"] for t in scen_targets_all.get(sid, {}).values()]) if sid in scen_targets_all else set()

    def gold_tag(row):
        sid, cid = row["scenario_id"], row["chunk_id"]
        if cid in essential_chunk_ids(sid):
            return "essential_gold"
        if bool(row.get("is_wrong_regime_gold")):
            return "wrong_regime"
        if bool(row.get("is_gold")):
            return "supporting_gold"
        return "non_gold"

    cache_full["gold_tag"] = cache_full.apply(gold_tag, axis=1)
    cache_full["source_group"] = cache_full["authority_class"].map(SOURCE_GROUP).fillna("other")

    cache = cache_full[cache_full.scenario_id.isin(dev_ids)].copy()
    print(f"DEV candidate rows: {len(cache)}  scenarios: {cache.scenario_id.nunique()}")
    print(cache.gold_tag.value_counts())

    # ---------------- weighted contributions (reconstruction) ----------------
    BM25_W, DENSE_W, ALPHA = 0.40, 0.60, 0.10
    cache["bm25_weighted"] = BM25_W * cache["bm25_norm"]
    cache["dense_weighted"] = DENSE_W * cache["dense_norm"]
    cache["fused_reconstructed"] = cache["bm25_weighted"] + cache["dense_weighted"]
    cache["authority_weighted"] = ALPHA * cache["authority_norm"]
    cache["pre_jurisdiction_score"] = cache["fused_reconstructed"] * (1 - ALPHA) + cache["authority_weighted"]
    cache["final_reconstructed"] = cache["pre_jurisdiction_score"] * cache["jurisdiction_weight"]
    recon_err = (cache["final_reconstructed"] - cache["final_score"]).abs()
    print(f"\nReconstruction error: mean={recon_err.mean():.2e} max={recon_err.max():.2e}")

    # ================================================================= 2. BM25 ANALYSIS
    rows = []
    d = describe(cache.bm25_raw, "raw_"); d.update({"signal": "bm25", "group": "ALL"}); rows.append(d)
    d = describe(cache.bm25_norm, "norm_"); d.update({"signal": "bm25_norm", "group": "ALL"}); rows.append(d)
    for grp in ["essential_gold", "supporting_gold", "non_gold", "wrong_regime"]:
        sub = cache[cache.gold_tag == grp]
        d = describe(sub.bm25_raw, "raw_"); d.update({"signal": "bm25", "group": grp}); rows.append(d)
    for grp in cache.source_group.unique():
        sub = cache[cache.source_group == grp]
        d = describe(sub.bm25_raw, "raw_"); d.update({"signal": "bm25", "group": f"source:{grp}"}); rows.append(d)
    # weighted / effective range
    d = describe(cache.bm25_weighted, "w_"); d.update({"signal": "bm25_weighted", "group": "ALL"}); rows.append(d)

    bm25_stats_rows = rows

    # rank correlations
    bm25_raw_rho = sstats.spearmanr(cache.bm25_raw, -cache.pre_rerank_rank, nan_policy="omit")
    bm25_norm_rho = sstats.spearmanr(cache.bm25_norm, -cache.pre_rerank_rank, nan_policy="omit")

    essential = cache[cache.gold_tag == "essential_gold"].bm25_raw
    nongold = cache[cache.gold_tag == "non_gold"].bm25_raw
    wrong_regime = cache[cache.gold_tag == "wrong_regime"].bm25_raw
    bm25_mw_gold_nongold = mw_cliffs(essential, nongold)
    try:
        y = (cache.gold_tag == "essential_gold").astype(int)
        bm25_auc = roc_auc_score(y, cache.bm25_raw.fillna(cache.bm25_raw.min()))
    except Exception:
        bm25_auc = None

    # effective weighted range (10th-90th pct spread)
    w = cache.bm25_weighted.dropna()
    bm25_eff_range = {"min": w.min(), "max": w.max(), "mean": w.mean(), "median": w.median(), "sd": w.std(),
                       "iqr": w.quantile(0.75) - w.quantile(0.25), "p10_p90_spread": w.quantile(0.90) - w.quantile(0.10)}

    # ================================================================= 3. DENSE ANALYSIS
    rows = []
    d = describe(cache.dense_raw, "raw_"); d.update({"signal": "dense", "group": "ALL"}); rows.append(d)
    d = describe(cache.dense_norm, "norm_"); d.update({"signal": "dense_norm", "group": "ALL"}); rows.append(d)
    for grp in ["essential_gold", "supporting_gold", "non_gold", "wrong_regime"]:
        sub = cache[cache.gold_tag == grp]
        d = describe(sub.dense_raw, "raw_"); d.update({"signal": "dense", "group": grp}); rows.append(d)
    d = describe(cache.dense_weighted, "w_"); d.update({"signal": "dense_weighted", "group": "ALL"}); rows.append(d)
    dense_stats_rows = rows

    dense_essential = cache[cache.gold_tag == "essential_gold"].dense_raw
    dense_nongold = cache[cache.gold_tag == "non_gold"].dense_raw
    dense_wrong = cache[cache.gold_tag == "wrong_regime"].dense_raw
    dense_mw_gold_nongold = mw_cliffs(dense_essential, dense_nongold)
    dense_mw_gold_wrongregime = mw_cliffs(dense_essential, dense_wrong)
    try:
        dense_auc = roc_auc_score((cache.gold_tag == "essential_gold").astype(int), cache.dense_raw.fillna(cache.dense_raw.min()))
    except Exception:
        dense_auc = None
    dense_gap_gold_nongold = dense_essential.mean() - dense_nongold.mean()
    dense_gap_gold_wrongregime = dense_essential.mean() - dense_wrong.mean() if len(dense_wrong) else None

    w = cache.dense_weighted.dropna()
    dense_eff_range = {"min": w.min(), "max": w.max(), "mean": w.mean(), "median": w.median(), "sd": w.std(),
                        "iqr": w.quantile(0.75) - w.quantile(0.25), "p10_p90_spread": w.quantile(0.90) - w.quantile(0.10)}

    # ================================================================= 4. AUTHORITY ANALYSIS
    auth_rows = []
    for cls, raw_w in AUTHORITY_WEIGHTS.items():
        sub = cache[cache.authority_class == cls]
        if sub.empty:
            continue
        auth_rows.append({
            "authority_class": cls, "n": len(sub), "raw_authority_weight": raw_w,
            "mean_authority_norm": sub.authority_norm.mean(), "median_authority_norm": sub.authority_norm.median(),
            "mean_authority_weighted": sub.authority_weighted.mean(),
        })
    auth_df = pd.DataFrame(auth_rows).sort_values("raw_authority_weight", ascending=False)

    prim = cache[cache.authority_class == "PRIMARY_LEGISLATION"]
    sec = cache[cache.authority_class == "SECONDARY_LEGISLATION"]
    raw_diff = 1.00 - 0.97
    norm_diff = prim.authority_norm.mean() - sec.authority_norm.mean() if len(prim) and len(sec) else None
    weighted_diff = prim.authority_weighted.mean() - sec.authority_weighted.mean() if len(prim) and len(sec) else None
    amplification = (norm_diff / raw_diff) if norm_diff is not None and raw_diff else None

    # authority rank inversions: for candidates with EQUAL/CLOSE fused (pre-authority) score, does authority flip primary<->secondary order?
    # pairwise comparisons of every (primary, secondary) candidate within the same
    # scenario+lane pool. Two DISTINCT, non-overlapping directions:
    #   sec_demoted_by_authority: fusion (bm25+dense only) favored SECONDARY, but authority
    #     flips the final order back to primary (secondary loses ground it had earned)
    #   prim_demoted_by_authority: fusion favored PRIMARY, but authority flips the final
    #     order to favor secondary instead (primary loses ground it had earned - the
    #     "concerning" direction, should be rare given authority's own primary>secondary bias)
    sec_demoted_by_authority = 0; prim_demoted_by_authority = 0; inv_total = 0
    for sid, grp in cache.groupby("scenario_id"):
        for lane, glane in grp.groupby("lane"):
            p = glane[glane.authority_class == "PRIMARY_LEGISLATION"]
            s = glane[glane.authority_class == "SECONDARY_LEGISLATION"]
            if p.empty or s.empty:
                continue
            for _, pr in p.iterrows():
                for _, sr in s.iterrows():
                    inv_total += 1
                    fused_p, fused_s = pr.fused_reconstructed, sr.fused_reconstructed
                    final_p, final_s = pr.final_score, sr.final_score
                    if fused_s > fused_p and final_s <= final_p:
                        sec_demoted_by_authority += 1
                    if fused_p > fused_s and final_p <= final_s:
                        prim_demoted_by_authority += 1
    inv_count = sec_demoted_by_authority + prim_demoted_by_authority
    # mean/median rank movement by source class: pre_rerank_rank vs a "fused-only" rank (no authority) within lane
    rank_move_rows = []
    for sid, grp in cache.groupby("scenario_id"):
        for lane, glane in grp.groupby("lane"):
            glane = glane.copy()
            glane["fused_rank"] = glane["fused_reconstructed"].rank(ascending=False, method="first")
            glane["final_rank"] = glane["final_score"].rank(ascending=False, method="first")
            glane["auth_rank_move"] = glane["fused_rank"] - glane["final_rank"]
            for _, r in glane.iterrows():
                rank_move_rows.append({"authority_class": r["authority_class"], "move": r["auth_rank_move"]})
    rm_df = pd.DataFrame(rank_move_rows)
    auth_rank_move = rm_df.groupby("authority_class")["move"].agg(["count", "mean", "median"]).reset_index()

    # ================================================================= 5. JURISDICTION
    juris_counts = cache.jurisdiction.value_counts()
    n_eu = (cache.jurisdiction == "EU").sum() if "EU" in cache.jurisdiction.values else 0
    juris_score_change = cache[cache.jurisdiction == "EU"].apply(
        lambda r: r.pre_jurisdiction_score - r.final_score, axis=1) if n_eu else pd.Series(dtype=float)
    eu_essential_gold = cache[(cache.jurisdiction == "EU") & (cache.gold_tag == "essential_gold")]

    # ================================================================= 8. SATURATION
    sat_rows = []
    for col in ["bm25_norm", "dense_norm", "authority_norm"]:
        v = cache[col].dropna()
        sat_rows.append({
            "signal": col, "pct_gt_090": (v > 0.90).mean(), "pct_gt_095": (v > 0.95).mean(),
            "pct_gt_099": (v > 0.99).mean(), "pct_lt_010": (v < 0.10).mean(), "n_unique": v.nunique(), "n": len(v),
        })
    sat_df = pd.DataFrame(sat_rows)

    # ================================================================= 10. SOURCE-TYPE
    src_rows = []
    for grp in sorted(cache.source_group.unique()):
        sub = cache[cache.source_group == grp]
        src_rows.append({
            "source_group": grp, "n": len(sub), "bm25_mean": sub.bm25_raw.mean(), "bm25_median": sub.bm25_raw.median(),
            "dense_mean": sub.dense_raw.mean(), "dense_median": sub.dense_raw.median(),
            "authority_norm_mean": sub.authority_norm.mean(), "final_score_mean": sub.final_score.mean(),
            "final_score_median": sub.final_score.median(), "gold_rate": (sub.gold_tag == "essential_gold").mean(),
            "avg_pre_rerank_rank": sub.pre_rerank_rank.mean(), "avg_ce_rank": sub.ce_rank.mean(),
        })
    src_df = pd.DataFrame(src_rows)

    # ================================================================= 12. CE SCORE ANALYSIS
    ce_rows = []
    ce_have = cache[cache.ce_score.notna()]
    for grp in ["essential_gold", "supporting_gold", "non_gold", "wrong_regime"]:
        sub = ce_have[ce_have.gold_tag == grp]
        d = describe(sub.ce_score, "ce_"); d.update({"group": grp}); ce_rows.append(d)
    for grp in sorted(ce_have.source_group.unique()):
        sub = ce_have[ce_have.source_group == grp]
        d = describe(sub.ce_score, "ce_"); d.update({"group": f"source:{grp}"}); ce_rows.append(d)
    ce_df = pd.DataFrame(ce_rows)
    try:
        ce_auc = roc_auc_score((ce_have.gold_tag == "essential_gold").astype(int), ce_have.ce_score)
    except Exception:
        ce_auc = None
    ce_mv = ce_have.assign(move=ce_have.pre_rerank_rank - ce_have.ce_rank)
    ce_gold_mv = ce_mv[ce_mv.gold_tag == "essential_gold"]
    n_into25 = int(((ce_gold_mv.pre_rerank_rank > 25) & (ce_gold_mv.ce_rank <= 25)).sum())
    n_out25 = int(((ce_gold_mv.pre_rerank_rank <= 25) & (ce_gold_mv.ce_rank > 25)).sum())
    was25 = ce_gold_mv[ce_gold_mv.pre_rerank_rank <= 25]
    harmful_rate = (was25.ce_rank > 25).mean() if len(was25) else None

    # ================================================================= 14. CORRELATION MATRIX
    corr_cols = ["bm25_raw", "bm25_norm", "dense_raw", "dense_norm", "authority_norm", "final_score",
                 "ce_score", "pre_rerank_rank", "ce_rank"]
    tmp = cache.copy()
    tmp["relevance_label"] = tmp.gold_tag.map({"essential_gold": 2, "supporting_gold": 1, "wrong_regime": -1, "non_gold": 0})
    corr_cols2 = corr_cols + ["relevance_label"]
    corr = tmp[corr_cols2].corr(method="spearman")

    # ================================================================= 13. TOP-25-PER-LANE FINAL METRICS
    def ranked_pre(sid, lane): return ranked_lane_pre(cache_full, sid, lane)
    baseline_ce = {}
    for (sid, lane), grp in cache_full[cache_full.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    def top25_metrics(scen_ids, label):
        sc_targets = {sid: scen_targets_all[sid] for sid in scen_ids if sid in scen_targets_all}
        n_req = sum(len(t) for t in sc_targets.values())
        req_num = 0
        cc_num = 0
        mixed = {}
        for sid in scen_ids:
            if sid not in sc_targets or not scen_by_id.get(sid, {}).get("is_mixed_evidence"):
                continue
            law_reqs = {r: t for r, t in sc_targets[sid].items() if t["roles"] & LAW_ROLES}
            nonlaw_reqs = {r: t for r, t in sc_targets[sid].items() if t["roles"] & NONLAW_ROLES}
            if law_reqs and nonlaw_reqs:
                mixed[sid] = (law_reqs, nonlaw_reqs)
        dual_num = 0
        leg_side = 0
        nonleg_side = 0
        for sid, treqs in sc_targets.items():
            L = ranked_lane_post(cache_full, sid, "legislation", baseline_ce.get((sid, "legislation")))[:25]
            O = ranked_lane_post(cache_full, sid, "other", baseline_ce.get((sid, "other")))[:25]
            sat_all = True
            for rid, t in treqs.items():
                ok = req_satisfied(t["chunk_ids"], L, O, 25, 25)
                if ok: req_num += 1
                else: sat_all = False
            if sat_all: cc_num += 1
        for sid, (law_reqs, nonlaw_reqs) in mixed.items():
            L = ranked_lane_post(cache_full, sid, "legislation", baseline_ce.get((sid, "legislation")))[:25]
            O = ranked_lane_post(cache_full, sid, "other", baseline_ce.get((sid, "other")))[:25]
            law_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in law_reqs.values())
            nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in nonlaw_reqs.values())
            if law_ok: leg_side += 1
            if nonlaw_ok: nonleg_side += 1
            if law_ok and nonlaw_ok: dual_num += 1
        return {
            "split": label, "n_scenarios": len(sc_targets), "n_requirements": n_req,
            "RequirementRecall@25-per-lane": req_num / n_req if n_req else None,
            "CompleteCoverage@25-per-lane": cc_num / len(sc_targets) if sc_targets else None,
            "n_mixed_evidence_scenarios": len(mixed),
            "DualEvidenceCoverage@25-per-lane": dual_num / len(mixed) if mixed else None,
            "legislation_side_coverage": leg_side / len(mixed) if mixed else None,
            "nonlegislation_side_coverage": nonleg_side / len(mixed) if mixed else None,
        }

    top25_dev = top25_metrics(dev_ids, "DEV")
    top25_test = top25_metrics(test_ids, "TEST (confirmation only, baseline CE config, not tuned)")

    # ---------------- write everything out ----------------
    pd.DataFrame(bm25_stats_rows + dense_stats_rows).to_csv(f"{RD}/score_distribution_summary.csv", index=False)

    eff_range_rows = [
        {"signal": "bm25", "nominal_coefficient": BM25_W, **bm25_eff_range,
         "rank_corr_with_final": float(sstats.spearmanr(cache.bm25_weighted, cache.final_score, nan_policy='omit').statistic)},
        {"signal": "dense", "nominal_coefficient": DENSE_W, **dense_eff_range,
         "rank_corr_with_final": float(sstats.spearmanr(cache.dense_weighted, cache.final_score, nan_policy='omit').statistic)},
        {"signal": "authority", "nominal_coefficient": ALPHA,
         "min": cache.authority_weighted.min(), "max": cache.authority_weighted.max(),
         "mean": cache.authority_weighted.mean(), "median": cache.authority_weighted.median(),
         "sd": cache.authority_weighted.std(), "iqr": cache.authority_weighted.quantile(0.75) - cache.authority_weighted.quantile(0.25),
         "p10_p90_spread": cache.authority_weighted.quantile(0.90) - cache.authority_weighted.quantile(0.10),
         "rank_corr_with_final": float(sstats.spearmanr(cache.authority_weighted, cache.final_score, nan_policy='omit').statistic)},
    ]
    eff_df = pd.DataFrame(eff_range_rows)
    eff_df["normalized_sd"] = [cache.bm25_norm.std(), cache.dense_norm.std(), cache.authority_norm.std()]
    eff_df["effective_range"] = eff_df["max"] - eff_df["min"]
    eff_df["coefficient_of_variation"] = eff_df["sd"] / eff_df["mean"].abs()
    total_weighted_sd = eff_df["sd"].sum()
    eff_df["relative_contribution_share"] = eff_df["sd"] / total_weighted_sd
    eff_df.to_csv(f"{RD}/signal_effective_range.csv", index=False)

    auth_df.to_csv(f"{RD}/authority_amplification_analysis.csv", index=False)
    auth_rank_move.to_csv(f"{RD}/authority_rank_movement_by_class.csv", index=False)
    src_df.to_csv(f"{RD}/source_type_score_analysis.csv", index=False)
    ce_df.to_csv(f"{RD}/ce_score_analysis.csv", index=False)
    corr.to_csv(f"{RD}/signal_correlation_matrix.csv")
    sat_df.to_csv(f"{RD}/normalization_saturation_analysis.csv", index=False)

    # query-category breakdown (final_score, ce_score by suite, min N=8)
    qc_rows = []
    for suite, grp in cache.groupby("suite"):
        if len(grp) < 30:  # candidate-row count floor; scenario N reported alongside
            continue
        n_scen = grp.scenario_id.nunique()
        qc_rows.append({
            "suite": suite, "n_scenarios": n_scen, "n_candidates": len(grp),
            "bm25_mean": grp.bm25_raw.mean(), "dense_mean": grp.dense_raw.mean(),
            "final_score_mean": grp.final_score.mean(),
            "ce_score_mean": grp.ce_score.mean(), "gold_rate": (grp.gold_tag == "essential_gold").mean(),
        })
    pd.DataFrame(qc_rows).to_csv(f"{RD}/query_category_score_analysis.csv", index=False)

    # Table 5 itself is written by evaluation/final_performance.py, which owns it, so the
    # headline table has exactly one producer. The values are still computed here because
    # the signal narrative refers to them.

    stats_summary = {
        "reconstruction_error_mean": float(recon_err.mean()), "reconstruction_error_max": float(recon_err.max()),
        "bm25_raw_spearman_vs_neg_pre_rerank_rank": float(bm25_raw_rho.statistic),
        "bm25_norm_spearman_vs_neg_pre_rerank_rank": float(bm25_norm_rho.statistic),
        "bm25_mw_essential_vs_nongold": bm25_mw_gold_nongold, "bm25_auc_essential_vs_rest": float(bm25_auc) if bm25_auc else None,
        "bm25_effective_weighted_range": {k: float(v) for k, v in bm25_eff_range.items()},
        "dense_mw_essential_vs_nongold": dense_mw_gold_nongold, "dense_mw_essential_vs_wrongregime": dense_mw_gold_wrongregime,
        "dense_auc_essential_vs_rest": float(dense_auc) if dense_auc else None,
        "dense_gap_essential_minus_nongold_raw": float(dense_gap_gold_nongold),
        "dense_gap_essential_minus_wrongregime_raw": float(dense_gap_gold_wrongregime) if dense_gap_gold_wrongregime is not None else None,
        "dense_effective_weighted_range": {k: float(v) for k, v in dense_eff_range.items()},
        "authority_primary_vs_secondary_raw_diff": float(raw_diff),
        "authority_primary_vs_secondary_norm_diff": float(norm_diff) if norm_diff is not None else None,
        "authority_primary_vs_secondary_weighted_diff": float(weighted_diff) if weighted_diff is not None else None,
        "authority_amplification_factor_norm_over_raw": float(amplification) if amplification is not None else None,
        "authority_pairwise_inversions_total": inv_count, "authority_pairwise_comparisons_total": inv_total,
        "authority_pairwise_inversion_rate": (inv_count / inv_total) if inv_total else None,
        "n_secondary_demoted_below_primary_by_authority": sec_demoted_by_authority,
        "n_primary_demoted_below_secondary_by_authority": prim_demoted_by_authority,
        "secondary_demoted_rate": (sec_demoted_by_authority / inv_total) if inv_total else None,
        "primary_demoted_rate": (prim_demoted_by_authority / inv_total) if inv_total else None,
        "jurisdiction_counts": juris_counts.to_dict(), "n_eu_candidates": int(n_eu),
        "jurisdiction_eu_mean_score_change": float(juris_score_change.mean()) if n_eu else None,
        "n_eu_essential_gold_candidates": len(eu_essential_gold),
        "ce_auc_essential_vs_rest": float(ce_auc) if ce_auc else None,
        "ce_n_essential_gold_moved_into_top25": n_into25, "ce_n_essential_gold_moved_out_of_top25": n_out25,
        "ce_harmful_demotion_rate": float(harmful_rate) if harmful_rate is not None else None,
        "ce_mean_rank_movement_essential_gold": float(ce_gold_mv.assign(mv=ce_gold_mv.pre_rerank_rank - ce_gold_mv.ce_rank).mv.mean()),
        "ce_median_rank_movement_essential_gold": float(ce_gold_mv.assign(mv=ce_gold_mv.pre_rerank_rank - ce_gold_mv.ce_rank).mv.median()),
        "top25_per_lane_DEV": top25_dev, "top25_per_lane_TEST": top25_test,
    }
    json.dump(stats_summary, open(f"{RD}/stats_summary.json", "w"), indent=2, default=str)
    print(f"\nwrote all CSVs + stats_summary.json to {RD}/")
    print(json.dumps({k: v for k, v in stats_summary.items() if not isinstance(v, dict)}, indent=2, default=str))


if __name__ == "__main__":
    main()
