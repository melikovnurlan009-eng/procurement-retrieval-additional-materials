#!/usr/bin/env python3
"""Recompute every headline number in the report and check it against the published value.

This is the verification entry point for this package. It reads only artifacts shipped in
this repository (benchmark/, data/) and needs no corpus database, no Qdrant instance and no
GPU. Each check recomputes a number from the frozen candidate cache and the benchmark gold
file, then compares it with the value printed in the report.

    python evaluation/verify_reported_numbers.py

Exit code 0 means every reported number reproduced within tolerance; 1 means at least one
did not, and the offending row is marked FAIL.

Report section for each check is given in the "report" column of the output table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation.ce_metrics_audit import (
    load_jsonl,
    mandatory_requirements_with_targets,
    ranked_lane_pre,
    ranked_lane_post,
    req_satisfied,
    LAW_ROLES,
    NONLAW_ROLES,
)
from evaluation.paths import CANDIDATE_CACHE, GOLD_PATH, SCENARIOS_PATH, RESULTS_DIR

def tolerance_for(reported: float) -> float:
    """Half a unit in the last decimal place the report actually prints.

    A value the report gives as 23.76 is reproduced if the recomputation rounds to 23.76,
    i.e. lies within +/-0.005; one given as 0.813 must land within +/-0.0005. Comparing at
    a fixed tolerance would either fail correct results or pass wrong ones, depending on
    how many decimals that particular number was quoted to.
    """
    text = repr(float(reported))
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return 0.5 * (10 ** -decimals) if decimals else 0.5


def _load():
    cache = pd.read_parquet(CANDIDATE_CACHE)
    scenarios = load_jsonl(SCENARIOS_PATH)
    gold = {r["scenario_id"]: r for r in load_jsonl(GOLD_PATH)}
    targets = {}
    for sid, g in gold.items():
        t = mandatory_requirements_with_targets(g)
        if t:
            targets[sid] = t
    return cache, scenarios, gold, targets


def _split_ids(scenarios, split):
    return {s["scenario_id"] for s in scenarios if str(s.get("split", "")).lower() == split}


def _ce_scores(cache):
    out = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        out[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))
    return out


def main() -> int:
    cache_all, scenarios, gold, targets_all = _load()
    scen_by_id = {s["scenario_id"]: s for s in scenarios}
    ce_all = _ce_scores(cache_all)
    checks: list[dict] = []

    def check(name, report_section, reported, computed, exact=False):
        ok = (computed == reported) if exact else (abs(computed - reported) <= tolerance_for(reported))
        checks.append(
            {
                "check": name,
                "report": report_section,
                "reported": reported,
                "recomputed": round(computed, 4) if isinstance(computed, float) else computed,
                "status": "PASS" if ok else "FAIL",
            }
        )

    # ---------------------------------------------------------------- Table 5 / Figure 7
    # Final top-25-per-lane performance, post-cross-encoder, both splits.
    for split, n_scen_rep, n_req_rep, rr_rep, cc_rep, dual_rep, n_mixed_rep in [
        ("dev", 95, 155, 0.813, 0.747, 0.765, 17),
        ("test", 94, 139, 0.806, 0.755, 0.600, 15),
    ]:
        ids = _split_ids(scenarios, split)
        cache = cache_all[cache_all.scenario_id.isin(ids)]
        targets = {k: v for k, v in targets_all.items() if k in ids}

        n_sat = n_tot = cc_hits = 0
        dual_num = dual_den = 0
        for sid, treqs in targets.items():
            L = ranked_lane_post(cache, sid, "legislation", ce_all.get((sid, "legislation")))[:25]
            O = ranked_lane_post(cache, sid, "other", ce_all.get((sid, "other")))[:25]
            sat_all = True
            for t in treqs.values():
                n_tot += 1
                if req_satisfied(t["chunk_ids"], L, O, 25, 25):
                    n_sat += 1
                else:
                    sat_all = False
            cc_hits += int(sat_all)

            if scen_by_id[sid].get("is_mixed_evidence"):
                law = {r: t for r, t in treqs.items() if t["roles"] & LAW_ROLES}
                nonlaw = {r: t for r, t in treqs.items() if t["roles"] & NONLAW_ROLES}
                if law and nonlaw:
                    dual_den += 1
                    law_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in law.values())
                    nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in nonlaw.values())
                    dual_num += int(law_ok and nonlaw_ok)

        up = split.upper()
        check(f"{up} N scenarios (scoreable)", "Table 5", n_scen_rep, len(targets), exact=True)
        check(f"{up} N requirements (mandatory)", "Table 5", n_req_rep, n_tot, exact=True)
        check(f"{up} RequirementRecall@25-per-lane", "Table 5 / Fig 7", rr_rep, n_sat / n_tot)
        check(f"{up} CompleteCoverage@25-per-lane", "Table 5 / Fig 7", cc_rep, cc_hits / len(targets))
        check(f"{up} DualEvidenceCoverage@25-per-lane", "Table 5 / Fig 7", dual_rep, dual_num / dual_den)
        check(f"{up} N mixed-evidence scenarios", "Table 5", n_mixed_rep, dual_den, exact=True)

    # ---------------------------------------------------------------- Section 6.3 ceiling
    dev_ids = _split_ids(scenarios, "dev")
    cache_dev = cache_all[cache_all.scenario_id.isin(dev_ids)]
    targets_dev = {k: v for k, v in targets_all.items() if k in dev_ids}
    num = den = 0
    for sid, treqs in targets_dev.items():
        L = ranked_lane_pre(cache_dev, sid, "legislation")[:75]
        O = ranked_lane_pre(cache_dev, sid, "other")[:75]
        for t in treqs.values():
            den += 1
            num += int(req_satisfied(t["chunk_ids"], L, O, 75, 75))
    check("DEV CandidateRequirementRecall@75 (ceiling)", "Sec 6.3", 0.916, num / den)
    check("DEV ceiling numerator/denominator", "Sec 6.3", "142/155", f"{num}/{den}", exact=True)

    # ---------------------------------------------------------------- Figure 6: pre- vs post-CE dual evidence, DEV
    for label, use_ce, reported in [("pre-CE", False, 0.529), ("post-CE", True, 0.765)]:
        n_hit = n_mixed = 0
        for sid, treqs in targets_dev.items():
            if not scen_by_id[sid].get("is_mixed_evidence"):
                continue
            law = {r: t for r, t in treqs.items() if t["roles"] & LAW_ROLES}
            nonlaw = {r: t for r, t in treqs.items() if t["roles"] & NONLAW_ROLES}
            if not (law and nonlaw):
                continue
            n_mixed += 1
            if use_ce:
                L = ranked_lane_post(cache_dev, sid, "legislation", ce_all.get((sid, "legislation")))[:25]
                O = ranked_lane_post(cache_dev, sid, "other", ce_all.get((sid, "other")))[:25]
            else:
                L = ranked_lane_pre(cache_dev, sid, "legislation")[:25]
                O = ranked_lane_pre(cache_dev, sid, "other")[:25]
            law_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in law.values())
            nonlaw_ok = all(req_satisfied(t["chunk_ids"], L, O, 25, 25) for t in nonlaw.values())
            n_hit += int(law_ok and nonlaw_ok)
        check(f"DEV DualEvidenceCoverage {label}", "Fig 6 / Sec 6.3", reported, n_hit / n_mixed)

    # ---------------------------------------------------------------- Figure 4: first-stage AUC, DEV
    ess_by_scen = {
        sid: set().union(*[t["chunk_ids"] for t in treqs.values()]) for sid, treqs in targets_dev.items()
    }
    c = cache_dev.copy()
    c["is_essential"] = [
        row.chunk_id in ess_by_scen.get(row.scenario_id, ()) for row in c.itertuples()
    ]
    y = c["is_essential"].astype(int)
    check("DEV BM25 ROC AUC (essential gold vs rest)", "Fig 4 / Sec 6.1", 0.728,
          roc_auc_score(y, c.bm25_raw.fillna(c.bm25_raw.min())))
    check("DEV dense ROC AUC (essential gold vs rest)", "Fig 4 / Sec 6.1", 0.847,
          roc_auc_score(y, c.dense_raw.fillna(c.dense_raw.min())))
    check("DEV candidate rows analysed", "Sec 4.3 / 6.2", 105823, len(c), exact=True)

    # ---------------------------------------------------------------- Section 4.3: score reconstruction
    # final_score = (0.40*bm25_norm + 0.60*dense_norm)*0.90 + 0.10*authority_norm, times jurisdiction.
    fused = 0.40 * c.bm25_norm + 0.60 * c.dense_norm
    recon = (fused * 0.90 + 0.10 * c.authority_norm) * c.jurisdiction_weight
    check("Score reconstruction max abs error", "Sec 4.3", 0.0, float((recon - c.final_score).abs().max()))

    # ---------------------------------------------------------------- Section 6.2: authority amplification
    prim = c[c.authority_class == "PRIMARY_LEGISLATION"].authority_norm.mean()
    sec = c[c.authority_class == "SECONDARY_LEGISLATION"].authority_norm.mean()
    check("Primary legislation mean normalised authority", "Sec 6.2 / Fig 5", 0.751, float(prim))
    check("Secondary legislation mean normalised authority", "Sec 6.2 / Fig 5", 0.038, float(sec))
    check("Authority amplification factor (x)", "Sec 6.2 / Fig 5", 23.76, float((prim - sec) / 0.03), )

    # ---------------------------------------------------------------- Section 6.2: jurisdiction reach
    n_eu = int((c.jurisdiction == "EU").sum())
    check("DEV EU-jurisdiction candidate rows", "Sec 6.2", 5088, n_eu, exact=True)
    n_gold_eu = int(((c.jurisdiction == "EU") & c.is_essential).sum())
    check("EU-jurisdiction essential-gold rows", "Sec 6.2", 0, n_gold_eu, exact=True)
    check("DEV essential-gold candidate rows", "Sec 6.2", 187, int(c.is_essential.sum()), exact=True)

    # ---------------------------------------------------------------- Section 6.3: cross-encoder behaviour
    ce_rows = c[c.ce_score.notna()]
    check("DEV CE AUC within reranked top-75 pool", "Sec 6.3", 0.688,
          roc_auc_score(ce_rows.is_essential.astype(int), ce_rows.ce_score))
    gold_rows = ce_rows[ce_rows.is_essential]
    moves = gold_rows.pre_rerank_rank - gold_rows.ce_rank
    check("DEV essential-gold mean rank movement", "Sec 6.3", 3.59, float(moves.mean()))
    check("DEV essential-gold median rank movement", "Sec 6.3", 1.0, float(moves.median()))
    check("DEV essential gold moved INTO top-25", "Sec 6.3", 22,
          int(((gold_rows.pre_rerank_rank > 25) & (gold_rows.ce_rank <= 25)).sum()), exact=True)
    check("DEV essential gold moved OUT of top-25", "Sec 6.3", 12,
          int(((gold_rows.pre_rerank_rank <= 25) & (gold_rows.ce_rank > 25)).sum()), exact=True)
    was25 = gold_rows[gold_rows.pre_rerank_rank <= 25]
    check("DEV harmful demotion rate", "Sec 6.3", 0.091, float((was25.ce_rank > 25).mean()))

    # ---------------------------------------------------------------- report
    df = pd.DataFrame(checks)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 46)
    print(df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "verification_report.csv"
    df.to_csv(out, index=False)

    n_fail = int((df.status == "FAIL").sum())
    print(f"\n{len(df) - n_fail}/{len(df)} checks passed. Wrote {out}")
    if n_fail:
        print("FAILED checks:")
        print(df[df.status == "FAIL"].to_string(index=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
