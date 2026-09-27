#!/usr/bin/env python3
"""Strict DualEvidenceCoverage: ALL mandatory LAW-role reqs AND ALL mandatory NONLAW-role
reqs satisfied within cutoff, per scenario. Reuses ce_metrics_audit.py's corrected gold
targets (acceptable_chunk_ids unioned, mandatory filter applied) and ranking functions.
No new CE inference; no benchmark/results modified."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from evaluation.ce_metrics_audit import (
    load_jsonl, mandatory_requirements_with_targets, ranked_lane_pre, ranked_lane_post,
    ranked_lane_fused, req_satisfied, LAW_ROLES, NONLAW_ROLES, FAQ218, CACHE_PATH, V2_PATH, RD,
)

def main() -> None:

    cache = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    dev_scen = [s for s in scen_all if str(s.get("split", "")).lower() == "dev"]
    dev_ids = {s["scenario_id"] for s in dev_scen}
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_218.jsonl")}
    cache = cache[cache.scenario_id.isin(dev_ids)]
    scen_by_id = {s["scenario_id"]: s for s in dev_scen}

    scen_targets = {}
    for sid in dev_ids:
        t = mandatory_requirements_with_targets(gold_all.get(sid, {}))
        if t:
            scen_targets[sid] = t

    # mixed-evidence subgroup: is_mixed_evidence AND >=1 mandatory LAW-role req AND >=1 mandatory NONLAW-role req
    mixed = {}
    for sid, treqs in scen_targets.items():
        if not scen_by_id.get(sid, {}).get("is_mixed_evidence"):
            continue
        law_reqs = {rid: t for rid, t in treqs.items() if t["roles"] & LAW_ROLES}
        nonlaw_reqs = {rid: t for rid, t in treqs.items() if t["roles"] & NONLAW_ROLES}
        if law_reqs and nonlaw_reqs:
            mixed[sid] = {"law": law_reqs, "nonlaw": nonlaw_reqs}

    print(f"Mixed-evidence subgroup (mandatory law + mandatory non-law both present): N={len(mixed)}")
    print(f"scenario_ids: {sorted(mixed.keys())}\n")

    baseline_ce = {}
    for (sid, lane), grp in cache[cache.ce_rank.notna()].groupby(["scenario_id", "lane"]):
        baseline_ce[(sid, lane)] = dict(zip(grp.chunk_id, grp.ce_score))
    v2 = json.load(open(V2_PATH))
    v2_ce = {}
    for sc in v2["scenarios"]:
        sid = sc["scenario_id"]
        for lane, field in (("legislation", "legislation_lane_top75"), ("other", "other_lane_top75")):
            v2_ce[(sid, lane)] = {r["chunk_id"]: r["ce_score"] for r in sc.get(field, [])}

    VARIANTS = {
        "0_pre_CE_first_stage": lambda sid, lane: ranked_lane_pre(cache, sid, lane),
        "1_baseline_raw_text_CE": lambda sid, lane: ranked_lane_post(cache, sid, lane, baseline_ce.get((sid, lane))),
        "2_metadata_enriched_CE": lambda sid, lane: ranked_lane_post(cache, sid, lane, v2_ce.get((sid, lane))),
        "3_metadata_enriched_CE_fusion_lambda0.5": lambda sid, lane: ranked_lane_fused(cache, sid, lane, v2_ce.get((sid, lane)), 0.5),
    }
    _rcache = {}
    def get_ranked(variant, sid, lane):
        key = (variant, sid, lane)
        if key not in _rcache:
            _rcache[key] = VARIANTS[variant](sid, lane)
        return _rcache[key]

    def side_satisfied(reqs, L, O, k_leg, k_oth):
        return all(req_satisfied(t["chunk_ids"], L, O, k_leg, k_oth) for t in reqs.values())

    def dual_coverage(variant, k_leg, k_oth, label):
        num = 0
        leg_side_sat = 0
        other_side_sat = 0
        miss_leg_only, miss_other_only, miss_both = [], [], []
        for sid, sides in mixed.items():
            L = get_ranked(variant, sid, "legislation")[:k_leg]
            O = get_ranked(variant, sid, "other")[:k_oth]
            law_ok = side_satisfied(sides["law"], L, O, k_leg, k_oth)
            nonlaw_ok = side_satisfied(sides["nonlaw"], L, O, k_leg, k_oth)
            if law_ok:
                leg_side_sat += 1
            if nonlaw_ok:
                other_side_sat += 1
            if law_ok and nonlaw_ok:
                num += 1
            elif law_ok and not nonlaw_ok:
                miss_other_only.append(sid)
            elif nonlaw_ok and not law_ok:
                miss_leg_only.append(sid)
            else:
                miss_both.append(sid)
        den = len(mixed)
        print(f"=== {label} — variant: {variant} ===")
        print(f"  numerator={num}  denominator={den}  DualEvidenceCoverage={num/den:.4f}")
        print(f"  legislation-side coverage: {leg_side_sat}/{den} = {leg_side_sat/den:.4f}")
        print(f"  non-legislation-side coverage: {other_side_sat}/{den} = {other_side_sat/den:.4f}")
        print(f"  missing legislation only (n={len(miss_leg_only)}): {sorted(miss_leg_only)}")
        print(f"  missing non-legislation only (n={len(miss_other_only)}): {sorted(miss_other_only)}")
        print(f"  missing both (n={len(miss_both)}): {sorted(miss_both)}")
        print()
        return {"variant": variant, "cutoff_label": label, "k_leg": k_leg, "k_oth": k_oth,
                "numerator": num, "denominator": den, "value": round(num/den, 4),
                "legislation_side_coverage": round(leg_side_sat/den, 4),
                "nonlegislation_side_coverage": round(other_side_sat/den, 4),
                "n_missing_legislation_only": len(miss_leg_only), "missing_legislation_only_ids": sorted(miss_leg_only),
                "n_missing_nonlegislation_only": len(miss_other_only), "missing_nonlegislation_only_ids": sorted(miss_other_only),
                "n_missing_both": len(miss_both), "missing_both_ids": sorted(miss_both)}

    rows = []
    for variant in VARIANTS:
        rows.append(dual_coverage(variant, 25, 25, "DualEvidenceCoverage@25-per-lane"))
    for variant in VARIANTS:
        rows.append(dual_coverage(variant, 10, 10, "DualEvidenceCoverage@10-per-lane"))
    for variant in VARIANTS:
        rows.append(dual_coverage(variant, 5, 5, "DualEvidenceCoverage@10-budget-5+5"))

    json.dump(rows, open(f"{RD}/dual_evidence_strict.json", "w"), indent=2)
    pd.DataFrame(rows).drop(columns=["missing_legislation_only_ids","missing_nonlegislation_only_ids","missing_both_ids"]).to_csv(f"{RD}/dual_evidence_strict_summary.csv", index=False)
    print(f"wrote {RD}/dual_evidence_strict.json and dual_evidence_strict_summary.csv")


if __name__ == "__main__":
    main()
