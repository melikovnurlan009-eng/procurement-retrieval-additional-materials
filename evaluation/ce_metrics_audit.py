#!/usr/bin/env python3
"""From-scratch, denominator-audited recomputation of every CE evaluation metric.
Does NOT reuse any previously printed summary number. Does NOT reuse essential_targets()
from evaluation/common.py, because that function silently ignores `acceptable_chunk_ids`
(alternate chunk representations of the same citation) and does not filter `mandatory`.
Both are corrected here; both corrections are documented in the report.

DEV only. No new CE inference - reuses the existing candidate_cache.parquet (baseline CE,
top-75/lane, raw-text input) and the existing variant2_ce_output_DEV.json (metadata-enriched
CE, top-75/lane) from the prior experiment. No gold features were used to build either.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from evaluation.paths import (BENCHMARK_DIR, CANDIDATE_CACHE, CE_OUTPUT_VARIANT2_DEV, RESULTS_DIR)

FAQ218 = str(BENCHMARK_DIR)          # scenarios_all_208.jsonl / gold_evidence_208.jsonl live here
CACHE_PATH = str(CANDIDATE_CACHE)    # frozen first-stage candidate cache (shipped in data/)
V2_PATH = str(CE_OUTPUT_VARIANT2_DEV)
RD = str(RESULTS_DIR)
ACCEPTED_STATUSES = ("MATCHED", "FUZZY_MATCHED")
LAW_ROLES = {"controlling_rule", "implementing_detail"}
NONLAW_ROLES = {"explanatory_guidance", "procedural_guidance", "workflow_instruction",
                "policy_rule", "regulator_interpretation", "transition_rule"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cls_group(a):
    if a == "PRIMARY_LEGISLATION": return "primary_legislation"
    if a == "SECONDARY_LEGISLATION": return "secondary_legislation"
    if a == "OFFICIAL_GOVERNMENT_GUIDANCE": return "official_guidance"
    if a in ("REGULATOR_GUIDANCE", "REGULATOR_INTERPRETATION"): return "regulator_technical"
    return "guidance_workflow"


# --------------------------------------------------------------------- CORRECTED gold targets
def mandatory_requirements_with_targets(gold_record: dict) -> dict[str, dict]:
    """requirement_id -> {"chunk_ids": set, "roles": set}.
    Corrections applied vs evaluation/common.py:essential_targets():
      1. `mandatory` filter: a requirement is included only if mandatory is True OR the
         key is absent (absence = old-schema requirement = implicitly mandatory, since the
         old 150-scenario schema predates the mandatory/optional distinction entirely).
         mandatory == False is EXCLUDED (9 such requirements exist in the 208-set).
      2. `acceptable_chunk_ids` on each essential_evidence item are UNIONed into that
         requirement's target set alongside resolution.chunk_id, gated on the SAME item's
         resolution.status being MATCHED/FUZZY_MATCHED (never trusted on an unresolved item).
         The prior harness silently dropped these - 145 extra valid chunk ids across the
         208-scenario set (191/402 essential_evidence items carry at least one).
    """
    out = {}
    for req in gold_record.get("requirements", []):
        if req.get("mandatory") is False:
            continue
        ids = set()
        for it in req.get("essential_evidence", []) or []:
            res = it.get("resolution") or {}
            if res.get("status") not in ACCEPTED_STATUSES:
                continue
            if res.get("chunk_id"):
                ids.add(res["chunk_id"])
            for cid in it.get("acceptable_chunk_ids", []) or []:
                ids.add(cid)
        if ids:
            roles = set(req.get("required_evidence_roles", []) or [])
            out[req["requirement_id"]] = {"chunk_ids": ids, "roles": roles}
    return out


# --------------------------------------------------------------------- lane/rank lookups
def build_lane_rank_index(cache: pd.DataFrame):
    """(scenario_id, chunk_id) -> (lane, pre_rerank_rank). One row per (scenario,chunk) since
    a chunk lives in exactly one lane's candidate pool for a given scenario by construction."""
    idx = {}
    for r in cache[["scenario_id", "chunk_id", "lane", "pre_rerank_rank"]].itertuples(index=False):
        idx[(r.scenario_id, r.chunk_id)] = (r.lane, r.pre_rerank_rank)
    return idx


def ranked_lane_pre(cache: pd.DataFrame, sid: str, lane: str) -> list[str]:
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    return sub.sort_values("pre_rerank_rank")["chunk_id"].tolist()


def ranked_lane_post(cache: pd.DataFrame, sid: str, lane: str, ce_scores: dict) -> list[str]:
    """ce_scores: {chunk_id: score} for this (sid,lane)'s top-75 (None means use pre-CE order)."""
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty:
        return []
    if ce_scores is None:
        return sub.sort_values("pre_rerank_rank")["chunk_id"].tolist()
    head = sub[sub.chunk_id.isin(ce_scores.keys())].copy()
    tail = sub[~sub.chunk_id.isin(ce_scores.keys())].sort_values("pre_rerank_rank")
    head["ce_score_v"] = head["chunk_id"].map(ce_scores)
    head = head.sort_values("ce_score_v", ascending=False)
    return pd.concat([head, tail])["chunk_id"].tolist()


def ranked_lane_fused(cache: pd.DataFrame, sid: str, lane: str, ce_scores: dict, lam: float) -> list[str]:
    sub = cache[(cache.scenario_id == sid) & (cache.lane == lane)]
    if sub.empty or not ce_scores:
        return ranked_lane_post(cache, sid, lane, ce_scores)
    head = sub[sub.chunk_id.isin(ce_scores.keys())].copy()
    tail = sub[~sub.chunk_id.isin(ce_scores.keys())].sort_values("pre_rerank_rank")
    head["ce_score_v"] = head["chunk_id"].map(ce_scores)
    ce_min, ce_max = head["ce_score_v"].min(), head["ce_score_v"].max()
    fs_min, fs_max = head["final_score"].min(), head["final_score"].max()
    ce_norm = (head["ce_score_v"] - ce_min) / (ce_max - ce_min) if ce_max > ce_min else 0.5
    fs_norm = (head["final_score"] - fs_min) / (fs_max - fs_min) if fs_max > fs_min else 0.5
    head["blend"] = lam * ce_norm + (1 - lam) * fs_norm
    head = head.sort_values("blend", ascending=False)
    return pd.concat([head, tail])["chunk_id"].tolist()


# --------------------------------------------------------------------- metric primitives
def req_satisfied(req_targets: set[str], ranked_leg: list[str], ranked_oth: list[str], k_leg: int, k_oth: int) -> bool:
    topL, topO = set(ranked_leg[:k_leg]), set(ranked_oth[:k_oth])
    return bool(req_targets & (topL | topO))


def main():
    cache = pd.read_parquet(CACHE_PATH)
    scen_all = load_jsonl(f"{FAQ218}/scenarios_all_208.jsonl")
    dev_scen = [s for s in scen_all if str(s.get("split", "")).lower() == "dev"]
    dev_ids = {s["scenario_id"] for s in dev_scen}
    gold_all = {r["scenario_id"]: r for r in load_jsonl(f"{FAQ218}/gold_evidence_208.jsonl")}
    cache = cache[cache.scenario_id.isin(dev_ids)]
    lane_rank_idx = build_lane_rank_index(cache)
    scen_by_id = {s["scenario_id"]: s for s in dev_scen}

    # per-scenario corrected requirement targets
    scen_targets = {}
    for sid in dev_ids:
        g = gold_all.get(sid, {})
        t = mandatory_requirements_with_targets(g)
        if t:
            scen_targets[sid] = t

    n_scenarios = len(scen_targets)
    n_requirements = sum(len(t) for t in scen_targets.values())
    n_essential_chunk_instances = sum(len(r["chunk_ids"]) for t in scen_targets.values() for r in t.values())
    print(f"N scenarios (DEV, >=1 mandatory req w/ resolved gold): {n_scenarios}")
    print(f"N requirements (mandatory, resolved gold, requirement-level unit): {n_requirements}")
    print(f"N essential-gold chunk instances (chunk-level unit, incl. acceptable alternates): {n_essential_chunk_instances}")

    # CE score dicts
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
        "best_metadata_enriched_fusion_lambda0.5": lambda sid, lane: ranked_lane_fused(cache, sid, lane, v2_ce.get((sid, lane)), 0.5),
    }

    rows = []  # long-form metric table: metric,variant,numerator,denominator,value,unit,cutoff,lane_logic

    def add_row(metric, variant, num, den, unit, cutoff, lane_logic):
        val = (num / den) if den else None
        rows.append({"metric": metric, "variant": variant, "numerator": num, "denominator": den,
                     "value": round(val, 4) if val is not None else None, "unit": unit,
                     "cutoff": cutoff, "lane_logic": lane_logic})
        return val

    ranked_cache = {}  # (variant, sid, lane) -> ranked list

    def get_ranked(variant, sid, lane):
        key = (variant, sid, lane)
        if key not in ranked_cache:
            ranked_cache[key] = VARIANTS[variant](sid, lane)
        return ranked_cache[key]

    # ---------------- 1. Candidate RequirementRecall@75 (ceiling; pre-CE pool membership only)
    num = den = 0
    for sid, treqs in scen_targets.items():
        L = get_ranked("0_pre_CE_first_stage", sid, "legislation")[:75]
        O = get_ranked("0_pre_CE_first_stage", sid, "other")[:75]
        for rid, t in treqs.items():
            den += 1
            if req_satisfied(t["chunk_ids"], L, O, 75, 75):
                num += 1
    add_row("CandidateRequirementRecall@75 (ceiling)", "pre-CE pool", num, den, "requirement", "top-75 per lane (independent), OR across lanes", "per-lane top-75, OR-pooled")

    # ---------------- 2. EssentialGoldCoverage@k (CHUNK-level), k in 5,10,20,25,50,75
    for variant in ["1_baseline_raw_text_CE", "2_metadata_enriched_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        for k in [5, 10, 20, 25, 50, 75]:
            num = den = 0
            for sid, treqs in scen_targets.items():
                Lr = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "legislation"))}
                Or = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "other"))}
                for rid, t in treqs.items():
                    for cid in t["chunk_ids"]:
                        den += 1
                        loc = lane_rank_idx.get((sid, cid))
                        if loc is None:
                            continue
                        lane, _ = loc
                        rank = Lr.get(cid) if lane == "legislation" else Or.get(cid)
                        if rank is not None and rank <= k:
                            num += 1
            add_row(f"EssentialGoldCoverage@{k} (chunk-level)", variant, num, den, "chunk", f"top-{k} in chunk's own lane", "per-chunk, home-lane only")

    # ---------------- 3. RequirementRecall@k, k in 5,10,20,25 (requirement-weighted, micro-avg)
    for variant in ["0_pre_CE_first_stage", "1_baseline_raw_text_CE", "2_metadata_enriched_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        for k in [5, 10, 20, 25]:
            num = den = 0
            scenario_macro_vals = []
            for sid, treqs in scen_targets.items():
                L = get_ranked(variant, sid, "legislation")[:k]
                O = get_ranked(variant, sid, "other")[:k]
                sat = 0
                for rid, t in treqs.items():
                    den += 1
                    ok = req_satisfied(t["chunk_ids"], L, O, k, k)
                    if ok:
                        num += 1
                        sat += 1
                scenario_macro_vals.append(sat / len(treqs))
            add_row(f"RequirementRecall@{k} (requirement-weighted micro-avg)", variant, num, den, "requirement", f"top-{k} per lane (independent), OR across lanes", "per-lane top-k, OR-pooled")
            macro_val = sum(scenario_macro_vals) / len(scenario_macro_vals)
            rows.append({"metric": f"RequirementRecall@{k} (scenario-macro-avg, FOR COMPARISON ONLY)", "variant": variant,
                         "numerator": None, "denominator": len(scenario_macro_vals), "value": round(macro_val, 4),
                         "unit": "scenario (equal-weighted)", "cutoff": f"top-{k} per lane", "lane_logic": "per-lane top-k, OR-pooled"})

    # ---------------- 4. CompleteRequirementCoverage@k (scenario-level), k=10 budget-style(5+5) and 25
    for variant in ["1_baseline_raw_text_CE", "2_metadata_enriched_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        for label, k_leg, k_oth in [("10 (as 5+5 budget)", 5, 5), ("25 (25 per lane)", 25, 25)]:
            num = den = 0
            for sid, treqs in scen_targets.items():
                L = get_ranked(variant, sid, "legislation")
                O = get_ranked(variant, sid, "other")
                den += 1
                if all(req_satisfied(t["chunk_ids"], L, O, k_leg, k_oth) for t in treqs.values()):
                    num += 1
            add_row(f"CompleteRequirementCoverage@{label}", variant, num, den, "scenario", f"top-{k_leg} legislation + top-{k_oth} other (independent)", "per-lane, all-requirements-AND")

    # ---------------- 5. Production-budget metrics (real 5+5)
    for variant in ["1_baseline_raw_text_CE", "2_metadata_enriched_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        num = den = 0
        cc_num = cc_den = 0
        dual_num = dual_den = 0
        for sid, treqs in scen_targets.items():
            L = get_ranked(variant, sid, "legislation")[:5]
            O = get_ranked(variant, sid, "other")[:5]
            sat_all = True
            for rid, t in treqs.items():
                den += 1
                ok = req_satisfied(t["chunk_ids"], L, O, 5, 5)
                if ok:
                    num += 1
                else:
                    sat_all = False
            cc_den += 1
            if sat_all:
                cc_num += 1
            is_mixed = bool(scen_by_id.get(sid, {}).get("is_mixed_evidence"))
            if is_mixed:
                law_reqs = [rid for rid, t in treqs.items() if t["roles"] & LAW_ROLES]
                nonlaw_reqs = [rid for rid, t in treqs.items() if t["roles"] & NONLAW_ROLES]
                if law_reqs and nonlaw_reqs:
                    dual_den += 1
                    law_ok = any(req_satisfied(treqs[r]["chunk_ids"], L, O, 5, 5) for r in law_reqs)
                    nonlaw_ok = any(req_satisfied(treqs[r]["chunk_ids"], L, O, 5, 5) for r in nonlaw_reqs)
                    if law_ok and nonlaw_ok:
                        dual_num += 1
        add_row("RequirementRecall@10-budget (real 5+5)", variant, num, den, "requirement", "top-5 legislation + top-5 other (independent)", "per-lane top-5, OR-pooled")
        add_row("CompleteCoverage@10-budget (real 5+5)", variant, cc_num, cc_den, "scenario", "top-5 legislation + top-5 other (independent)", "per-lane, all-requirements-AND")
        add_row("DualEvidenceCoverage@10-budget (real 5+5)", variant, dual_num, dual_den, "scenario (mixed-evidence subgroup only)", "top-5 legislation + top-5 other (independent)", "per-lane top-5; needs >=1 satisfied LAW-role req AND >=1 satisfied NONLAW-role req")

    # ---------------- 6. Mixed-evidence subgroup detail (best variant + baseline)
    for variant in ["1_baseline_raw_text_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        law_num = law_den = 0
        nonlaw_num = nonlaw_den = 0
        n_mixed_scen = 0
        for sid, treqs in scen_targets.items():
            if not scen_by_id.get(sid, {}).get("is_mixed_evidence"):
                continue
            law_reqs = {rid: t for rid, t in treqs.items() if t["roles"] & LAW_ROLES}
            nonlaw_reqs = {rid: t for rid, t in treqs.items() if t["roles"] & NONLAW_ROLES}
            if not (law_reqs and nonlaw_reqs):
                continue
            n_mixed_scen += 1
            L = get_ranked(variant, sid, "legislation")[:5]
            O = get_ranked(variant, sid, "other")[:5]
            for rid, t in law_reqs.items():
                law_den += 1
                if req_satisfied(t["chunk_ids"], L, O, 5, 5):
                    law_num += 1
            for rid, t in nonlaw_reqs.items():
                nonlaw_den += 1
                if req_satisfied(t["chunk_ids"], L, O, 5, 5):
                    nonlaw_num += 1
        add_row(f"LawRequirementRecall@10-budget [mixed-evidence subgroup, N_scenarios={n_mixed_scen}]", variant, law_num, law_den, "requirement", "top-5+5 real budget", "per-lane top-5, OR-pooled, LAW-role reqs only")
        add_row(f"NonLawRequirementRecall@10-budget [mixed-evidence subgroup, N_scenarios={n_mixed_scen}]", variant, nonlaw_num, nonlaw_den, "requirement", "top-5+5 real budget", "per-lane top-5, OR-pooled, NONLAW-role reqs only")

    # ---------------- 7. Lane-specific chunk coverage (already computed in section 2, restated with explicit lane split)
    for variant in ["1_baseline_raw_text_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        for lane in ["legislation", "other"]:
            for k in [10, 25]:
                num = den = 0
                for sid, treqs in scen_targets.items():
                    Rr = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, lane))}
                    for rid, t in treqs.items():
                        for cid in t["chunk_ids"]:
                            loc = lane_rank_idx.get((sid, cid))
                            if loc is None or loc[0] != lane:
                                continue
                            den += 1
                            rank = Rr.get(cid)
                            if rank is not None and rank <= k:
                                num += 1
                add_row(f"LaneEssentialGoldCoverage@{k} [{lane} lane only]", variant, num, den, "chunk", f"top-{k} within {lane} lane", f"{lane} lane only, no OR with other lane")

    # ---------------- 8. Authority/source-type breakdown (chunk coverage@25, best variant vs baseline)
    for variant in ["1_baseline_raw_text_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        group_num = {}
        group_den = {}
        for sid, treqs in scen_targets.items():
            Lr = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "legislation"))}
            Or = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "other"))}
            for rid, t in treqs.items():
                for cid in t["chunk_ids"]:
                    row = cache[(cache.scenario_id == sid) & (cache.chunk_id == cid)]
                    if row.empty:
                        continue
                    auth = row.iloc[0]["authority_class"]
                    grp = cls_group(auth)
                    loc = lane_rank_idx.get((sid, cid))
                    if loc is None:
                        continue
                    lane, _ = loc
                    rank = Lr.get(cid) if lane == "legislation" else Or.get(cid)
                    group_den[grp] = group_den.get(grp, 0) + 1
                    if rank is not None and rank <= 25:
                        group_num[grp] = group_num.get(grp, 0) + 1
        for grp, den in group_den.items():
            num = group_num.get(grp, 0)
            add_row(f"EssentialGoldCoverage@25 by authority class [{grp}]", variant, num, den, "chunk", "top-25 in chunk's own lane", "per-chunk, home-lane only")

    # ---------------- 9. CE movement diagnostics + requirement-level rescued/harmed
    for variant in ["1_baseline_raw_text_CE", "2_metadata_enriched_CE", "best_metadata_enriched_fusion_lambda0.5"]:
        moves = []
        for sid, treqs in scen_targets.items():
            pre_L = {c: i + 1 for i, c in enumerate(get_ranked("0_pre_CE_first_stage", sid, "legislation"))}
            pre_O = {c: i + 1 for i, c in enumerate(get_ranked("0_pre_CE_first_stage", sid, "other"))}
            post_L = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "legislation"))}
            post_O = {c: i + 1 for i, c in enumerate(get_ranked(variant, sid, "other"))}
            seen = set()
            for rid, t in treqs.items():
                for cid in t["chunk_ids"]:
                    if cid in seen:
                        continue
                    seen.add(cid)
                    loc = lane_rank_idx.get((sid, cid))
                    if loc is None:
                        continue
                    lane, _ = loc
                    pre_r = (pre_L if lane == "legislation" else pre_O).get(cid)
                    post_r = (post_L if lane == "legislation" else post_O).get(cid)
                    if pre_r is None or post_r is None:
                        continue
                    moves.append({"sid": sid, "cid": cid, "pre": pre_r, "post": post_r})
        mv = pd.DataFrame(moves)
        n_into = int(((mv.pre > 25) & (mv.post <= 25)).sum()) if len(mv) else 0
        n_out = int(((mv.pre <= 25) & (mv.post > 25)).sum()) if len(mv) else 0
        was_top25 = mv[mv.pre <= 25] if len(mv) else mv
        hdr = float((was_top25.post > 25).mean()) if len(was_top25) else None
        mean_mv = float((mv.pre - mv.post).mean()) if len(mv) else None
        med_mv = float((mv.pre - mv.post).median()) if len(mv) else None
        rows.append({"metric": "CE chunk moved into top25", "variant": variant, "numerator": n_into,
                     "denominator": len(mv), "value": None, "unit": "chunk", "cutoff": "top-25", "lane_logic": "own-lane rank"})
        rows.append({"metric": "CE chunk moved out of top25", "variant": variant, "numerator": n_out,
                     "denominator": len(mv), "value": None, "unit": "chunk", "cutoff": "top-25", "lane_logic": "own-lane rank"})
        rows.append({"metric": "Harmful demotion rate P(post>25|pre<=25)", "variant": variant, "numerator": int((was_top25.post > 25).sum()) if len(was_top25) else 0,
                     "denominator": len(was_top25), "value": round(hdr, 4) if hdr is not None else None, "unit": "chunk", "cutoff": "top-25", "lane_logic": "own-lane rank"})
        rows.append({"metric": "Mean rank movement (pre-post)", "variant": variant, "numerator": None, "denominator": len(mv),
                     "value": round(mean_mv, 3) if mean_mv is not None else None, "unit": "chunk", "cutoff": "n/a", "lane_logic": "own-lane rank"})
        rows.append({"metric": "Median rank movement (pre-post)", "variant": variant, "numerator": None, "denominator": len(mv),
                     "value": med_mv, "unit": "chunk", "cutoff": "n/a", "lane_logic": "own-lane rank"})

        # requirement-level rescued vs harmed at @25
        rescued = harmed = 0
        for sid, treqs in scen_targets.items():
            pre_L = get_ranked("0_pre_CE_first_stage", sid, "legislation")[:25]
            pre_O = get_ranked("0_pre_CE_first_stage", sid, "other")[:25]
            post_L = get_ranked(variant, sid, "legislation")[:25]
            post_O = get_ranked(variant, sid, "other")[:25]
            for rid, t in treqs.items():
                pre_ok = req_satisfied(t["chunk_ids"], pre_L, pre_O, 25, 25)
                post_ok = req_satisfied(t["chunk_ids"], post_L, post_O, 25, 25)
                if not pre_ok and post_ok:
                    rescued += 1
                elif pre_ok and not post_ok:
                    harmed += 1
        rows.append({"metric": "Requirement-level rescued (unsat@25 pre -> sat@25 post)", "variant": variant,
                     "numerator": rescued, "denominator": n_requirements, "value": None, "unit": "requirement", "cutoff": "top-25", "lane_logic": "per-lane top-25, OR-pooled"})
        rows.append({"metric": "Requirement-level harmed (sat@25 pre -> unsat@25 post)", "variant": variant,
                     "numerator": harmed, "denominator": n_requirements, "value": None, "unit": "requirement", "cutoff": "top-25", "lane_logic": "per-lane top-25, OR-pooled"})

    df = pd.DataFrame(rows)
    df.to_csv(f"{RD}/metric_audit_table.csv", index=False)
    print(f"\nwrote {RD}/metric_audit_table.csv ({len(df)} rows)")

    with open(f"{RD}/audit_key_numbers.json", "w") as f:
        json.dump({"n_scenarios": n_scenarios, "n_requirements": n_requirements,
                    "n_essential_chunk_instances": n_essential_chunk_instances}, f, indent=2)

    return df, n_scenarios, n_requirements


if __name__ == "__main__":
    main()
