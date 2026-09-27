#!/usr/bin/env python3
"""TEST-side confirmation checks, reporting only: nothing here selects a configuration.

Everything in this script was decided on DEV. It is re-measured on the frozen TEST split to
show whether the DEV-chosen behaviour holds where nothing was tuned. Three parts, matching
the report:

  1. Authority-alpha sensitivity on TEST across the DEV-chosen grid {0, 0.1, 0.2, 0.3, 0.4}
     (Section 6.2.1). Scores are recomputed from the cached raw values, not re-retrieved.

  2. Cross-encoder diagnostics on TEST (Section 6.3.1): the candidate ceiling at 75 per lane,
     the cross-encoder's AUC within that top-75 pool, essential-gold rank movement, how many
     essential-gold chunks move into and out of the top 25, and the harmful demotion rate
     P(post-CE rank > 25 | pre-CE rank <= 25).

  3. Category-boost reflection on TEST (Section 6.3.3): what the real query classifier
     achieves against what a perfect (oracle) category signal would achieve. The gap is the
     point - it separates "category information is useless" from "our classifier is weak".

    python evaluation/test_confirmation.py

Reads only the shipped candidate cache and benchmark. Writes results/test_confirmation/.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from evaluation.common import weighted_retriever, refuse_pool, requirement_recall_at_k, complete_coverage
from evaluation.config import DEFAULT_DB, DEFAULT_COLLECTION
from evaluation.ce_metrics_audit import load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, ranked_lane_post, FAQ218, CACHE_PATH
from evaluation.category_boost import classify_query


from evaluation.paths import RESULTS_DIR

RD = str(RESULTS_DIR / "test_confirmation")
RESULTS_DIR.joinpath("test_confirmation").mkdir(parents=True, exist_ok=True)


def main() -> None:
    cache_full = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_218.jsonl")}
    scen_by_id = {s["scenario_id"]: s for s in scen_all}
    test_ids = {s["scenario_id"] for s in scen_all if str(s.get("split", "")).lower() == "test"}
    cache = cache_full[cache_full.scenario_id.isin(test_ids)]

    scen_targets = {}
    for sid in test_ids:
        t = mandatory_requirements_with_targets(gold_all.get(sid, {}))
        if t:
            scen_targets[sid] = t
    print(f"TEST scoreable scenarios: {len(scen_targets)}", file=sys.stderr)

    # ============================================================ 1. AUTHORITY ALPHA SENSITIVITY (TEST, report-only)
    def pooled_rank(cache, sid, alpha, beta=0.40, use_jurisdiction=True):
        sub = cache[cache.scenario_id == sid]
        if sub.empty: return []
        pool = set(sub.chunk_id)
        bm25_raw = dict(zip(sub.chunk_id, sub.bm25_raw))
        dense_raw = dict(zip(sub.chunk_id, sub.dense_raw))
        meta = {row.chunk_id: {"authority_class": row.authority_class, "jurisdiction": row.jurisdiction} for row in sub.itertuples()}
        with weighted_retriever(beta=beta, alpha=alpha, db_path=DEFAULT_DB, collection=DEFAULT_COLLECTION) as r:
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta)
        return sorted(pool, key=lambda c: -scores[c]["final_score"])

    def two_lane_rank(cache, sid, lane, alpha, beta=0.40):
        sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
        if sub.empty: return []
        pool = set(sub.chunk_id)
        bm25_raw = dict(zip(sub.chunk_id, sub.bm25_raw))
        dense_raw = dict(zip(sub.chunk_id, sub.dense_raw))
        meta = {row.chunk_id: {"authority_class": row.authority_class, "jurisdiction": row.jurisdiction} for row in sub.itertuples()}
        with weighted_retriever(beta=beta, alpha=alpha, db_path=DEFAULT_DB, collection=DEFAULT_COLLECTION) as r:
            scores = refuse_pool(r, bm25_raw, dense_raw, pool, meta)
        return sorted(pool, key=lambda c: -scores[c]["final_score"])

    alpha_rows = []
    for alpha in [0.00, 0.10, 0.20, 0.30, 0.40]:
        n_sat = n_tot = 0
        cc_sum = 0
        for sid, targets in scen_targets.items():
            plain = {rid: t["chunk_ids"] for rid, t in targets.items()}
            L = two_lane_rank(cache, sid, "legislation", alpha)[:5]
            O = two_lane_rank(cache, sid, "other", alpha)[:5]
            rr = requirement_recall_at_k(L, O, plain, 5, 5)
            cc = complete_coverage(L, O, plain, 5, 5)
            n_sat += rr * len(targets); n_tot += len(targets); cc_sum += cc
        alpha_rows.append({"alpha": alpha, "beta": 0.40, "n_scenarios": len(scen_targets),
                            "n_requirements": n_tot, "requirement_recall_weighted": n_sat / n_tot,
                            "complete_coverage_mean": cc_sum / len(scen_targets)})
        print(f"  alpha={alpha}: RR={n_sat/n_tot:.4f} CC={cc_sum/len(scen_targets):.4f}", file=sys.stderr)
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(f"{RD}/test_authority_alpha_sensitivity.csv", index=False)

    # ============================================================ 2. TEST CE ceiling / AUC / rank movement / harmful demotion
    scen_targets_all_ids = {sid: set().union(*[t["chunk_ids"] for t in targets.values()]) for sid, targets in scen_targets.items()}

    def ess_ids(sid):
        return scen_targets_all_ids.get(sid, set())

    cache = cache.copy()
    cache["is_essential_gold_corrected"] = cache.apply(lambda r: r.chunk_id in ess_ids(r.scenario_id), axis=1)

    # ceiling @75
    num = den = 0
    for sid, targets in scen_targets.items():
        L = ranked_lane_pre(cache, sid, "legislation")[:75]
        O = ranked_lane_pre(cache, sid, "other")[:75]
        for t in targets.values():
            den += 1
            if (t["chunk_ids"] & set(L)) or (t["chunk_ids"] & set(O)):
                num += 1
    ceiling = num / den
    print(f"TEST CandidateRequirementRecall@75 (ceiling): {num}/{den} = {ceiling:.4f}", file=sys.stderr)

    ce_have = cache[cache.ce_score.notna()]
    try:
        ce_auc = roc_auc_score((ce_have.is_essential_gold_corrected).astype(int), ce_have.ce_score)
    except Exception as e:
        ce_auc = None
        print("AUC failed:", e, file=sys.stderr)

    gold_rows = ce_have[ce_have.is_essential_gold_corrected]
    moves = gold_rows.pre_rerank_rank - gold_rows.ce_rank
    n_into25 = int(((gold_rows.pre_rerank_rank > 25) & (gold_rows.ce_rank <= 25)).sum())
    n_out25 = int(((gold_rows.pre_rerank_rank <= 25) & (gold_rows.ce_rank > 25)).sum())
    was25 = gold_rows[gold_rows.pre_rerank_rank <= 25]
    harmful_rate = float((was25.ce_rank > 25).mean()) if len(was25) else None

    test_ce_summary = {
        "n_scenarios": len(scen_targets), "n_requirements": den,
        "candidate_requirement_recall_at_75_ceiling": ceiling,
        "ce_auc_essential_vs_rest": float(ce_auc) if ce_auc is not None else None,
        "n_essential_gold_rows_with_ce_score": len(gold_rows),
        "mean_rank_movement": float(moves.mean()), "median_rank_movement": float(moves.median()),
        "n_moved_into_top25": n_into25, "n_moved_out_of_top25": n_out25,
        "n_was_in_top25_pre": int(len(was25)), "harmful_demotion_rate": harmful_rate,
    }
    json.dump(test_ce_summary, open(f"{RD}/test_ce_diagnostics.json", "w"), indent=2)
    print(json.dumps(test_ce_summary, indent=2))

    # ============================================================ 3. TEST category-boost reflection
    baseline_ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))

    def other_lane_post_ranked(sid, boost_classes, factor):
        sub = cache[(cache.scenario_id == sid) & (cache.lane == "other")].copy()
        ces = baseline_ce.get((sid, "other"), {})
        if not ces:
            return sub.sort_values("pre_rerank_rank").chunk_id.tolist()
        head = sub[sub.chunk_id.isin(ces.keys())].copy()
        tail = sub[~sub.chunk_id.isin(ces.keys())].sort_values("pre_rerank_rank")
        head["ce_score_v"] = head.chunk_id.map(ces)
        head["boosted"] = head.apply(lambda r: r.ce_score_v * factor if (boost_classes and r.authority_class in boost_classes) else r.ce_score_v, axis=1)
        head = head.sort_values("boosted", ascending=False)
        return pd.concat([head, tail]).chunk_id.tolist()

    scenarios_with_other_gold = [sid for sid in scen_targets if cache[(cache.scenario_id == sid) & (cache.lane == "other") & (cache.is_essential_gold_corrected)].shape[0] > 0]

    def recall25_other_lane(rank_fn):
        num = den = 0
        for sid in scenarios_with_other_gold:
            gold_ids = set(cache[(cache.scenario_id == sid) & (cache.lane == "other") & (cache.is_essential_gold_corrected)].chunk_id)
            if not gold_ids: continue
            ranked = rank_fn(sid)
            top25 = set(ranked[:25])
            den += len(gold_ids); num += len(gold_ids & top25)
        return num, den

    results = []
    num, den = recall25_other_lane(lambda sid: other_lane_post_ranked(sid, set(), 1.0))
    results.append({"config": "baseline (no boost)", "numerator": num, "denominator": den, "recall@25_other_lane": num/den})

    query_pred = {sid: classify_query(scen_by_id[sid].get("query") or scen_by_id[sid].get("scenario_text", "")) for sid in test_ids}
    n_classified = sum(1 for v in query_pred.values() if v)
    coverage = n_classified / len(query_pred)
    for factor in [1.5, 2.0]:
        num, den = recall25_other_lane(lambda sid, factor=factor: other_lane_post_ranked(sid, query_pred.get(sid, set()), factor))
        results.append({"config": f"real classifier, boost={factor}x", "numerator": num, "denominator": den,
                         "recall@25_other_lane": num/den, "classifier_coverage": coverage})

    oracle_true_class = {sid: set(cache[(cache.scenario_id == sid) & (cache.lane == "other") & (cache.is_essential_gold_corrected)].authority_class.unique()) for sid in scen_targets}
    for factor in [1.5, 2.0]:
        num, den = recall25_other_lane(lambda sid, factor=factor: other_lane_post_ranked(sid, oracle_true_class.get(sid, set()), factor))
        results.append({"config": f"oracle (true gold class), boost={factor}x", "numerator": num, "denominator": den, "recall@25_other_lane": num/den})

    boost_df = pd.DataFrame(results)
    boost_df.to_csv(f"{RD}/test_category_boost_reflection.csv", index=False)
    print(boost_df.to_string(index=False))
    print(f"\nwrote test_authority_alpha_sensitivity.csv, test_ce_diagnostics.json, test_category_boost_reflection.csv to {RD}/")


if __name__ == "__main__":
    main()
