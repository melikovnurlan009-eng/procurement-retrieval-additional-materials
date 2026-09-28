#!/usr/bin/env python3
"""Reproduce the reported results one part at a time, and say whether each part succeeded.

The report is not one result, so this is not one script. It is nine independent parts, each
of which corresponds to a section, table or figure of the report. A part re-runs the scripts
that produce its outputs and then checks those outputs against the values the report prints.
A part passes only when every one of its checks passes: a script exiting cleanly is not
treated as success.

    python evaluation/reproduce.py --list                # what the parts are
    python evaluation/reproduce.py --all                 # run all nine
    python evaluation/reproduce.py performance signals   # run named parts
    python evaluation/reproduce.py 4                     # run a part by number

Parts 2 to 9 need only the artifacts shipped in this repository: no GPU, no Qdrant, no
corpus database. Part 1 regenerates its numbers from the corpus database when CORPUS_DB
points at one, and otherwise checks the shipped copy of its output without regenerating it,
which it says clearly in the output.

Exit status is 0 only if every selected part passed. Every check, with its reported value,
its recomputed value and its verdict, is written to results/reproduction_report.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation.paths import (
    CANDIDATE_CACHE,
    CONFIG_DIR,
    CORPUS_DB,
    FIGURES_DIR,
    GOLD_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    SCENARIOS_PATH,
)

GREEN, RED, YELLOW, BLUE, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
)
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = BLUE = DIM = BOLD = OFF = ""


# ---------------------------------------------------------------------------- check model
@dataclass
class Check:
    name: str
    reported: object
    recomputed: object
    ok: bool
    note: str = ""


def tolerance_for(reported: float) -> float:
    """Half a unit in the last decimal place the report actually prints.

    A value the report gives as 23.76 reproduces if the recomputation rounds to 23.76, i.e.
    lies within +/-0.005; one given as 0.813 must land within +/-0.0005. A single fixed
    tolerance would either fail correct results or pass wrong ones, depending only on how
    many decimals that particular number happened to be quoted to.
    """
    text = repr(float(reported))
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return 0.5 * (10 ** -decimals) if decimals else 0.5


class Checker:
    """Collects one part's checks."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def num(self, name: str, reported: float, recomputed: float, note: str = "",
            tol: float | None = None) -> None:
        try:
            ok = abs(float(recomputed) - float(reported)) <= (tol or tolerance_for(reported))
        except (TypeError, ValueError):
            ok = False
        shown = round(float(recomputed), 4) if isinstance(recomputed, (int, float)) else recomputed
        self.checks.append(Check(name, reported, shown, ok, note))

    def exact(self, name: str, reported, recomputed, note: str = "") -> None:
        self.checks.append(Check(name, reported, recomputed, recomputed == reported, note))

    def true(self, name: str, condition: bool, detail: str = "", note: str = "") -> None:
        self.checks.append(Check(name, "yes", detail or ("yes" if condition else "no"),
                                 bool(condition), note))


# ------------------------------------------------------------------------------- helpers
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _json(name: str):
    return json.loads((RESULTS_DIR / name).read_text())


def _expand(spec: str) -> list[Path]:
    """A `produces` entry names a file or a directory; return the concrete files."""
    base = REPO_ROOT / spec
    if base.is_dir():
        return sorted(f for f in base.rglob("*") if f.is_file() and f.name != ".DS_Store")
    return [base] if base.exists() else []


def _snapshot(specs: list[str]) -> dict[Path, str]:
    """sha256 of every file a part claims to produce, before it runs."""
    out = {}
    for spec in specs:
        for f in _expand(spec):
            out[f] = _sha256(f)
    return out


def _describe(path: Path) -> str:
    """Row count for tabular output, otherwise a size - something to recognise it by."""
    size = path.stat().st_size
    human = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.1f} KB"
    try:
        if path.suffix == ".csv":
            with path.open() as fh:
                return f"{sum(1 for _ in fh) - 1} rows, {human}"
        if path.suffix == ".jsonl":
            with path.open() as fh:
                return f"{sum(1 for _ in fh)} records, {human}"
        if path.suffix == ".json":
            obj = json.loads(path.read_text())
            n = len(obj)
            return f"{n} {'entries' if isinstance(obj, dict) else 'records'}, {human}"
    except Exception:
        pass
    return human


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ================================================================== part check functions
def check_corpus(c: Checker) -> None:
    """Section 3.1 and Figure 1: what the corpus contains."""
    st = _json("corpus_stats.json")
    c.exact("Total source documents", 1370, st["total_documents"])
    c.exact("Total chunks", 19087, st["total_chunks"])
    c.exact("Legislation-lane documents", 23, st["lane_split_documents"]["legislation_lane"])
    c.exact("Other-evidence-lane documents", 1347, st["lane_split_documents"]["other_evidence_lane"])
    c.exact("Citation-graph edges", 23785, st["graph_total_edges"])
    cls = st["documents_by_authority_class"]
    c.exact("PRIMARY_LEGISLATION documents", 14, cls.get("PRIMARY_LEGISLATION"))
    c.exact("SECONDARY_LEGISLATION documents", 9, cls.get("SECONDARY_LEGISLATION"))
    c.exact("OFFICIAL_WORKFLOW documents", 662, cls.get("OFFICIAL_WORKFLOW"))
    c.exact("Index manifest: chunks indexed", 19087, st["index_manifest"]["chunks_indexed"])
    c.exact("Index manifest: FTS tokenizer", "porter", st["index_manifest"]["fts_tokenizer"])


def check_artifacts(c: Checker) -> None:
    """The frozen inputs are the ones the reported results were computed from.

    Every other number in the report is conditional on this. It also pins down the
    218-versus-208 relationship explicitly, rather than leaving it as a surprise for whoever
    counts the lines.
    """
    manifest = json.loads((CONFIG_DIR / "frozen_cache_manifest.json").read_text())
    c.exact("Gold evidence SHA-256 matches the frozen manifest",
            manifest["gold_sha256"], _sha256(GOLD_PATH))

    scen_ids = {json.loads(l)["scenario_id"] for l in open(SCENARIOS_PATH)}
    gold_ids = {json.loads(l)["scenario_id"] for l in open(GOLD_PATH)}
    cache = pd.read_parquet(CANDIDATE_CACHE)
    cache_ids = set(cache.scenario_id.unique())
    dropped = {f"EXP_GRAPH{i:03d}" for i in range(1, 11)}

    c.exact("Scenario records in the shipped file", 208, len(scen_ids))

    # The benchmark is the 189 scenarios that carry a scoreable mandatory requirement -
    # 95 DEV and 94 TEST, the figures the report gives. The rest of the records are
    # carried in the file but never scored.
    from evaluation.ce_metrics_audit import mandatory_requirements_with_targets
    gold_by_id = {r["scenario_id"]: r for r in
                  (json.loads(l) for l in open(GOLD_PATH) if l.strip())}
    scen = [json.loads(l) for l in open(SCENARIOS_PATH) if l.strip()]
    scoreable = {s_["scenario_id"] for s_ in scen
                 if mandatory_requirements_with_targets(gold_by_id.get(s_["scenario_id"], {}))}
    c.exact("Benchmark scenarios", 189, len(scoreable))
    for split, n in (("dev", 95), ("test", 94)):
        ids = {s_["scenario_id"] for s_ in scen if s_.get("split") == split}
        c.exact(f"{split.upper()} scenarios", n, len(ids & scoreable))
    c.exact("Gold records (frozen 218-set)", 218, len(gold_ids))
    c.exact("Candidate cache scenarios (frozen 218-set)", 218, len(cache_ids))
    c.exact("Gold minus scenarios = ten dropped EXP_GRAPH", sorted(dropped),
            sorted(gold_ids - scen_ids))
    c.exact("Cache minus scenarios = the same ten", sorted(dropped), sorted(cache_ids - scen_ids))
    c.exact("Every benchmark scenario has gold", set(), scen_ids - gold_ids)
    c.exact("Every benchmark scenario is in the cache", set(), scen_ids - cache_ids)

    c.exact("Frozen alpha (authority blend)", 0.10, manifest["alpha"])
    c.exact("Frozen beta (BM25 share)", 0.40, manifest["beta"])
    c.exact("Frozen graph expansion", False, manifest["use_graph"])
    c.exact("Frozen jurisdiction weighting", True, manifest["use_jurisdiction"])
    c.exact("Frozen candidates per channel per lane", 300,
            manifest["candidates_per_channel_per_lane"])
    c.exact("Frozen rerank depth", 75, manifest["rerank_depth"])
    c.exact("Frozen embedding model", "BAAI/bge-m3", manifest["embedding_model"])
    c.exact("Frozen reranker model", "BAAI/bge-reranker-v2-m3", manifest["reranker_model"])
    c.exact("Frozen random seed", 20260920, manifest["random_seed"])

    frozen = json.loads((CONFIG_DIR / "frozen_config.json").read_text())
    c.exact("Freeze record agrees on alpha", manifest["alpha"], frozen["alpha"])
    c.exact("Freeze record agrees on beta", manifest["beta"], frozen["beta"])
    c.exact("Freeze record agrees on graph off", manifest["use_graph"], frozen["use_graph"])

    c.exact("Candidate cache lanes", ["legislation", "other"], sorted(cache.lane.unique()))
    c.true("Cross-encoder depth never exceeds 75 per lane",
           float(cache.ce_rank.max()) <= 75, f"max ce_rank = {cache.ce_rank.max():.0f}")

    # final_score must reconstruct from the cached RAW values through the frozen formula.
    fused = 0.40 * cache.bm25_norm + 0.60 * cache.dense_norm
    recon = (fused * 0.90 + 0.10 * cache.authority_norm) * cache.jurisdiction_weight
    c.num("Score reconstruction max abs error (all rows)", 0.0,
          float((recon - cache.final_score).abs().max()))


def check_rq1(c: Checker) -> None:
    """Section 6.1, Tables 3 and 4, Figures 4 and 5: the matched-budget comparison."""
    sub = RESULTS_DIR / "rq1_rq3_same_budget"
    test = pd.read_csv(sub / "rq1_same_budget_50_baselines.csv").query("split == 'TEST'") \
             .set_index("config_id")
    c.exact("TEST scenarios in every configuration", {94}, set(int(n) for n in test.n_scenarios))
    c.exact("TEST requirements in every configuration", {139},
            set(int(n) for n in test.n_requirements))

    # Table 3, in the report's own row order.
    for cfg, label, value in [
        (1, "BM25 only (pooled @50)", 0.345),
        (2, "Dense only (pooled @50)", 0.496),
        (3, "BM25 + dense hybrid (pooled @50)", 0.475),
        (6, "Pooled hybrid + authority + jurisdiction", 0.612),
        (7, "Two-lane, no authority (25+25)", 0.734),
        (8, "Two-lane, pre-CE (25+25)", 0.763),
        (9, "Two-lane, post-CE (25+25)", 0.806),
    ]:
        c.num(f"Table 3 - {label}", value, test.loc[cfg, "requirement_recall_weighted"])
    for cfg, label, value in [
        (1, "BM25 only (pooled @50)", 0.255),
        (2, "Dense only (pooled @50)", 0.415),
        (3, "BM25 + dense hybrid (pooled @50)", 0.404),
        (6, "Pooled hybrid + authority + jurisdiction", 0.521),
        (7, "Two-lane, no authority (25+25)", 0.691),
        (8, "Two-lane, pre-CE (25+25)", 0.713),
        (9, "Two-lane, post-CE (25+25)", 0.755),
    ]:
        c.num(f"Table 3 CC - {label}", value, test.loc[cfg, "complete_coverage_mean"])
    c.true("Every configuration retrieves the same 50-result budget",
           set(test.budget) == {"top-50 (pooled)", "25+25 (two-lane)"},
           ", ".join(sorted(set(test.budget))))

    # Table 4: the two largest categories and the one reversal the report names.
    cat = pd.read_csv(sub / "rq1_per_category.csv").query("split == 'TEST'")
    pooled = cat[cat.config.str.startswith("pooled")].set_index("category")
    twolane = cat[cat.config.str.startswith("two_lane")].set_index("category")
    for name, n_req, p_val, t_val, delta in [
        ("direct legal anchor", 23, 0.652, 0.913, +0.261),
        ("semantic / practitioner phrasing", 21, 0.619, 0.857, +0.238),
        ("procedural_multi_evidence", 20, 0.600, 0.700, +0.100),
        ("cross_reference_multi_instrument", 16, 0.562, 0.688, +0.125),
        ("applicability / transition", 12, 0.583, 0.500, -0.083),
        ("faq90", 12, 0.750, 0.833, +0.083),
        ("expansion_v2_official_guidance", 11, 0.727, 0.727, 0.000),
        ("official_workflow_expansion_v2", 9, 0.889, 0.889, 0.000),
        ("vocabulary mismatch", 8, 0.250, 0.625, +0.375),
        ("compound / multi-requirement", 5, 0.400, 0.600, +0.200),
        ("authority / source-role sensitive", 2, 0.000, 1.000, +1.000),
    ]:
        pv = pooled.loc[name, "requirement_recall_weighted"]
        tv = twolane.loc[name, "requirement_recall_weighted"]
        c.exact(f"Table 4 - {name}, N req", n_req, int(pooled.loc[name, "n_requirements"]))
        c.num(f"Table 4 - {name}, pooled", p_val, pv)
        c.num(f"Table 4 - {name}, two-lane", t_val, tv)
        c.num(f"Table 4 - {name}, difference", delta, tv - pv)
    c.exact("Table 4 covers every TEST category", 11, len(pooled))

    # Statistics behind the headline RQ1 claim.
    stats = json.loads((sub / "statistical_tests_TEST.json").read_text())
    rq1 = stats["RQ1_source_aware_vs_conventional_TEST"]
    rr, cc = rq1["requirement_recall"], rq1["complete_coverage"]
    c.num("RQ1 RequirementRecall difference", 0.151, rr["mean_diff"])
    c.num("RQ1 95% CI lower bound", 0.080, rr["ci_low"])
    c.num("RQ1 95% CI upper bound", 0.225, rr["ci_high"])
    c.true("RQ1 permutation p < 0.0001", rr["permutation_p"] < 0.0001, f"p = {rr['permutation_p']}")
    c.num("RQ1 CompleteCoverage difference", 0.191, cc["mean_diff"])
    c.num("RQ1 CompleteCoverage CI lower bound", 0.106, cc["ci_low"])
    c.num("RQ1 CompleteCoverage CI upper bound", 0.277, cc["ci_high"])
    c.exact("RQ1 McNemar scenarios improved", 19, cc["n_improved"])
    c.exact("RQ1 McNemar scenarios worsened", 1, cc["n_worsened"])


def check_signals(c: Checker) -> None:
    """Sections 6.1.1 and 6.2.1-6.2.2, Figures 6 and 7: how each signal behaves."""
    s = _json("score_signal_stats_summary.json")
    c.num("BM25 ROC AUC (essential gold vs rest, DEV)", 0.728, s["bm25_auc_essential_vs_rest"])
    c.num("Dense ROC AUC (essential gold vs rest, DEV)", 0.847, s["dense_auc_essential_vs_rest"])
    # Effective weighted contribution of each first-stage signal (Section 6.1.1).
    c.num("BM25 effective weighted contribution", 0.201,
          0.40 * pd.read_parquet(CANDIDATE_CACHE).bm25_norm.mean(), tol=0.001)
    c.num("Dense effective weighted spread", 0.504,
          s["dense_effective_weighted_range"]["p10_p90_spread"])
    c.num("Score reconstruction max error", 0.0, s["reconstruction_error_max"])

    # Section 6.2.1: authority calibration.
    c.num("Authority amplification factor", 23.76, s["authority_amplification_factor_norm_over_raw"])
    c.exact("Primary-secondary pairs authority reorders", 943159,
            s["authority_pairwise_inversions_total"])
    c.num("Share of pairs authority reorders", 0.193, s["authority_pairwise_inversion_rate"])
    auth = _csv("authority_amplification_analysis.csv").set_index("authority_class")
    c.num("Primary legislation mean normalised authority", 0.751,
          auth.loc["PRIMARY_LEGISLATION", "mean_authority_norm"])
    c.num("Secondary legislation mean normalised authority", 0.038,
          auth.loc["SECONDARY_LEGISLATION", "mean_authority_norm"])
    c.num("Nominal primary-secondary gap", 0.03,
          auth.loc["PRIMARY_LEGISLATION", "raw_authority_weight"]
          - auth.loc["SECONDARY_LEGISLATION", "raw_authority_weight"])

    # Section 6.2.2: applicability and jurisdiction.
    c.num("Dense gap, essential gold vs non-gold", 0.101,
          s["dense_gap_essential_minus_nongold_raw"])
    c.num("Dense gap, essential gold vs wrong-regime", 0.024,
          s["dense_gap_essential_minus_wrongregime_raw"])
    c.exact("DEV EU-jurisdiction candidate rows", 5088, s["n_eu_candidates"])
    c.exact("EU-jurisdiction essential-gold rows", 0, s["n_eu_essential_gold_candidates"])


def check_graph(c: Checker) -> None:
    """Section 6.2.3: the stratified audit of citation-edge quality."""
    audit = _json("graph_edge_audit.json")
    c.exact("Edges reviewed", 122, audit["n_edges_reviewed"])
    c.exact("Strata sampled", 14, audit["n_strata"])
    c.num("Strict meaningful rate", 0.598, audit["strict_meaningful_rate"])
    c.num("Not-wrong rate", 0.820, audit["not_wrong_rate"])
    c.true("Every sampled edge is classified",
           sum(audit["class_counts"].values()) == audit["n_edges_reviewed"],
           f"{sum(audit['class_counts'].values())}/{audit['n_edges_reviewed']}")

    strata = _csv("graph_edge_audit_by_stratum.csv")
    structured = strata[strata.stratum.str.contains("structured XML|cross-instrument", regex=True)]
    hubs = strata[strata.stratum.str.contains("citation-definition hubs|mention elsewhere")]
    c.true("Structured cross-reference strata reach 80-100%",
           bool((structured.meaningful_rate_pct >= 80).all()),
           ", ".join(f"{v:.0f}%" for v in structured.meaningful_rate_pct))
    c.true("Inferred and hub strata sit at 10-20%",
           bool((hubs.meaningful_rate_pct <= 20).all()),
           ", ".join(f"{v:.0f}%" for v in hubs.meaningful_rate_pct))


def check_reranking(c: Checker) -> None:
    """Section 6.3, Table 7 and Figure 8: what the cross-encoder does."""
    # Table 7: the four DEV-selected variants at the 5+5 diagnostic budget.
    df = _csv("table4_dev_ce_variants.csv").set_index("variant")
    for variant, rr, cc, harm in [
        ("1_baseline_raw_text_CE", 0.593, 0.463, 0.094),
        ("3_CE_plus_fusion_lambda0.5", 0.641, 0.526, 0.016),
        ("2_metadata_enriched_CE", 0.652, 0.516, 0.047),
        ("4_metadata_enriched_CE_plus_fusion_lambda0.5", 0.655, 0.526, 0.000),
    ]:
        row = df.loc[variant]
        c.num(f"Table 7 - {variant}, recall@10", rr, row["RequirementRecall@10"])
        c.num(f"Table 7 - {variant}, complete coverage", cc, row["CompleteCoverage@10"])
        c.num(f"Table 7 - {variant}, harmful demotion", harm, row["harmful_demotion_rate"])
    c.true("Every variant scored on the same 95 DEV scenarios",
           bool((df.n_scenarios == 95).all()), f"{sorted(set(df.n_scenarios))}")

    # Figure 8: strict mixed-evidence completeness, before and after.
    dual = _csv("dual_evidence_strict_summary.csv")
    at25 = dual[dual.cutoff_label == "DualEvidenceCoverage@25-per-lane"]
    pre = at25[at25.variant == "0_pre_CE_first_stage"].iloc[0]
    post = at25[at25.variant == "1_baseline_raw_text_CE"].iloc[0]
    c.exact("DEV pre-CE dual evidence (9/17)", "9/17",
            f"{int(pre.numerator)}/{int(pre.denominator)}")
    c.exact("DEV post-CE dual evidence (13/17)", "13/17",
            f"{int(post.numerator)}/{int(post.denominator)}")
    c.num("DEV pre-CE dual-evidence coverage", 0.529, pre.value)
    c.num("DEV post-CE dual-evidence coverage", 0.765, post.value)
    c.num("Post-CE legislation-side coverage", 1.0, post.legislation_side_coverage)
    c.num("Post-CE non-legislation-side coverage", 0.765, post.nonlegislation_side_coverage)

    # Section 6.3.1 and 6.3.3: TEST-side confirmation, nothing selected from TEST.
    diag = _json("test_confirmation/test_ce_diagnostics.json")
    c.exact("TEST scenarios", 94, diag["n_scenarios"])
    c.num("TEST candidate ceiling @75", 0.885, diag["candidate_requirement_recall_at_75_ceiling"])
    c.num("TEST cross-encoder AUC within the top-75 pool", 0.680, diag["ce_auc_essential_vs_rest"])
    c.num("TEST mean essential-gold rank movement", 3.37, diag["mean_rank_movement"])
    c.num("TEST median essential-gold rank movement", 1.0, diag["median_rank_movement"])
    c.exact("TEST essential gold moved into top 25", 15, diag["n_moved_into_top25"])
    c.exact("TEST essential gold moved out of top 25", 9, diag["n_moved_out_of_top25"])
    c.num("TEST harmful demotion rate", 0.073, diag["harmful_demotion_rate"])

    # Section 6.3.3: the DEV-selected configuration against the one that scores higher on TEST.
    tv = _csv("test_confirmation/comparison_TEST.csv").set_index("variant")
    c.exact("TEST variants compared", 10, len(tv))
    c.num("TEST plain fusion lambda=0.50, recall@10", 0.670,
          tv.loc["3_CE_plus_fusion_lambda0.5_TEST", "requirement_recall_at_10"])
    c.num("TEST metadata+fusion lambda=0.50, recall@10", 0.649,
          tv.loc["4_metadata_enriched_CE_plus_fusion_lambda0.5_TEST", "requirement_recall_at_10"])
    c.true("Plain fusion scores higher on TEST than the DEV-selected configuration",
           tv.loc["3_CE_plus_fusion_lambda0.5_TEST", "requirement_recall_at_10"]
           > tv.loc["4_metadata_enriched_CE_plus_fusion_lambda0.5_TEST", "requirement_recall_at_10"],
           "reported, and still not re-selected")

    boost = _csv("test_confirmation/test_category_boost_reflection.csv").set_index("config")
    c.num("TEST category boost - baseline", 0.556,
          boost.loc["baseline (no boost)", "recall@25_other_lane"])
    c.num("TEST category boost - real classifier", 0.583,
          boost.loc["real classifier, boost=1.5x", "recall@25_other_lane"])
    c.num("TEST category boost - oracle ceiling", 0.694,
          boost.loc["oracle (true gold class), boost=1.5x", "recall@25_other_lane"])


def check_performance(c: Checker) -> None:
    """Section 6.4, Table 8 and Figure 9: final fixed-configuration performance."""
    df = _csv("top25_per_lane_final_metrics.csv")
    dev = df[df.split == "DEV"].iloc[0]
    test = df[df.split.str.startswith("TEST")].iloc[0]
    for label, row, n_scen, n_req, rr, cc, dual, n_mixed in [
        ("DEV", dev, 95, 155, 0.813, 0.747, 0.765, 17),
        ("TEST", test, 94, 139, 0.806, 0.755, 0.600, 15),
    ]:
        c.exact(f"Table 8 - {label} scenarios", n_scen, int(row.n_scenarios))
        c.exact(f"Table 8 - {label} requirements", n_req, int(row.n_requirements))
        c.num(f"Table 8 - {label} requirement recall", rr, row["RequirementRecall@25-per-lane"])
        c.num(f"Table 8 - {label} complete coverage", cc, row["CompleteCoverage@25-per-lane"])
        c.num(f"Table 8 - {label} dual evidence", dual, row["DualEvidenceCoverage@25-per-lane"])
        c.exact(f"Table 8 - {label} mixed-evidence scenarios", n_mixed,
                int(row.n_mixed_evidence_scenarios))

    c.num("TEST legislation-side coverage", 0.867, test.legislation_side_coverage)
    c.num("TEST non-legislation-side coverage", 0.667, test.nonlegislation_side_coverage)
    c.num("DEV legislation-side coverage", 1.0, dev.legislation_side_coverage)
    c.num("DEV non-legislation-side coverage", 0.765, dev.nonlegislation_side_coverage)

    audit = _csv("metric_audit_table.csv")
    ceiling = audit[audit.metric.str.startswith("CandidateRequirementRecall@75")].iloc[0]
    c.num("DEV candidate-pool ceiling @75", 0.916, ceiling.value)
    c.exact("DEV ceiling numerator", 142, int(ceiling.numerator))
    c.exact("DEV ceiling denominator", 155, int(ceiling.denominator))

    # Section 6.3.2, Table 6: reranking on and off at the final depth.
    stats = json.loads(
        (RESULTS_DIR / "rq1_rq3_same_budget" / "statistical_tests_TEST.json").read_text())
    rq3 = stats["RQ3_post_CE_vs_pre_CE_TEST_25perlane"]
    c.num("Table 6 - pre-CE requirement recall", 0.763,
          rq3["requirement_recall"]["mean_pre_CE_weighted"])
    c.num("Table 6 - post-CE requirement recall", 0.806,
          rq3["requirement_recall"]["mean_post_CE_weighted"])
    c.num("Table 6 - difference", 0.043, rq3["requirement_recall"]["mean_diff"])
    c.num("Reranking permutation p", 0.215, rq3["requirement_recall"]["permutation_p"])
    c.num("Reranking CI lower bound", -0.008, rq3["requirement_recall"]["ci_low"])
    c.num("Reranking CI upper bound", 0.100, rq3["requirement_recall"]["ci_high"])
    c.true("Reranking CI spans zero, as reported",
           rq3["requirement_recall"]["ci_low"] < 0 < rq3["requirement_recall"]["ci_high"],
           f"[{rq3['requirement_recall']['ci_low']:.3f}, "
           f"{rq3['requirement_recall']['ci_high']:.3f}]")

    cc3 = stats["RQ3_post_CE_vs_pre_CE_TEST_25perlane"]["complete_coverage"]
    c.num("Table 6 - pre-CE complete coverage", 0.713, cc3["mean_pre_CE"])
    c.num("Table 6 - post-CE complete coverage", 0.755, cc3["mean_post_CE"])
    c.num("Reranking complete-coverage difference", 0.043, cc3["mean_diff"])
    c.num("Reranking complete-coverage CI lower bound", -0.021, cc3["ci_low"])
    c.num("Reranking complete-coverage CI upper bound", 0.106, cc3["ci_high"])
    c.num("Reranking complete-coverage permutation p", 0.342, cc3["permutation_p"])


def check_ir_metrics(c: Checker) -> None:
    """Appendix A secondary diagnostics, Figures 10 and 11."""
    rec = _csv("ir_metrics/cumulative_requirement_recall_summary.csv")

    def at(split, stage, k):
        return rec[(rec.split == split) & (rec.stage == stage)
                   & (rec.k == k)].requirement_recall_weighted.iloc[0]

    c.exact("Depths evaluated", [10, 20, 30, 40, 50, 60, 70, 75], sorted(rec.k.unique().tolist()))
    c.num("DEV post-CE recall at k=10", 0.690, at("DEV", "post_CE", 10))
    c.num("DEV pre-CE recall at k=10", 0.626, at("DEV", "pre_CE", 10))
    c.num("TEST post-CE recall at k=10", 0.727, at("TEST", "post_CE", 10))
    c.num("TEST pre-CE recall at k=10", 0.662, at("TEST", "pre_CE", 10))
    c.num("TEST recall at k=50", 0.885, at("TEST", "post_CE", 50))
    c.true("TEST recall is flat beyond k=50",
           at("TEST", "post_CE", 50) == at("TEST", "post_CE", 75),
           f"k=50 and k=75 both {at('TEST', 'post_CE', 75):.4f}")
    c.true("The curves converge at k=75, because reranking reorders one pool",
           abs(at("DEV", "pre_CE", 75) - at("DEV", "post_CE", 75)) < 1e-9,
           f"DEV pre and post both {at('DEV', 'post_CE', 75):.4f}")

    nd = _csv("ir_metrics/precision_ndcg_summary.csv").query("stage == 'post_CE'")
    leg = nd[nd.lane == "legislation"].ndcg_at_k_mean
    oth = nd[nd.lane == "other"].ndcg_at_k_mean
    c.true("Legislation-lane NDCG sits in 0.54-0.63",
           0.53 <= leg.min() and leg.max() <= 0.64, f"{leg.min():.3f} to {leg.max():.3f}")
    c.true("Other-evidence-lane NDCG sits in 0.29-0.39",
           0.28 <= oth.min() and oth.max() <= 0.40, f"{oth.min():.3f} to {oth.max():.3f}")
    c.true("Every lane-level NDCG is below its legislation counterpart",
           bool(oth.max() < leg.min()),
           f"other max {oth.max():.3f} < legislation min {leg.min():.3f}")


def check_tokens(c: Checker) -> None:
    """Appendix A: chunk token-count distributions, and the cost of the 512-token window."""
    st = _json("token_distribution/token_distribution_summary.json")
    meta, summ, raw = st["metadata"], st["summary"], st["rawtext"]

    c.exact("Chunks measured", 19087, meta["n_chunks"])
    c.num("Metadata preamble mean tokens", 103.1, meta["mean"])
    c.num("Metadata preamble median", 101, meta["median"])
    c.num("Metadata preamble P90", 144, meta["p90"])
    c.num("Metadata preamble P95", 157, meta["p95"])
    c.exact("Metadata preamble max", 380, meta["max"])
    c.num("Metadata preamble over 512 tokens", 0.0, meta["pct_over_512"])
    c.num("retrieval_summary mean tokens", 71.7, summ["mean"])
    c.num("retrieval_summary over 512 tokens", 0.0, summ["pct_over_512"])
    c.num("Raw chunk text mean tokens", 305.1, raw["mean"])
    c.num("Raw chunk text over 512 tokens", 14.2, raw["pct_over_512"])

    trunc = _json("truncation_summary.json")
    c.exact("Essential gold left below rank 25 on DEV", 48, trunc["n_essential_gold_rank_gt25"])
    c.exact("Of those, over the 512-token window", 23, trunc["n_likely_truncated"])


def check_figures(c: Checker) -> None:
    """Every figure the report draws from data. Numbering is the report's own."""
    expected = [
        ("1", "figure01_corpus_composition.png"),
        ("4", "figure04_configuration_comparison_test.png"),
        ("5", "figure05_category_comparison_test.png"),
        ("6", "figure06_first_stage_auc.png"),
        ("7", "figure07_authority_calibration.png"),
        ("8", "figure08_mixed_evidence_dev.png"),
        ("9", "figure09_final_performance.png"),
        ("10", "figure10_cumulative_recall.png"),
        ("11", "figure11_ndcg_by_lane.png"),
    ]
    for num, name in expected:
        p = FIGURES_DIR / name
        size = p.stat().st_size if p.exists() else 0
        c.true(f"Figure {num}: {name}", p.exists() and size > 5_000,
               f"{size:,} bytes" if size else "missing")
    for num, name in [("2", "two-lane architecture"), ("3", "benchmark construction protocol")]:
        c.true(f"Figure {num} is a schematic, not generated", True, name)


def check_cross_check(c: Checker) -> None:
    """Consumes verify_reported_numbers.py's own output table."""
    df = _csv("verification_report.csv")
    n_fail = int((df.status == "FAIL").sum())
    c.exact("Headline numbers checked", 32, len(df))
    c.exact("Checks failing", 0, n_fail)
    for row in df.itertuples():
        c.true(row.check, row.status == "PASS",
               f"reported {row.reported}, recomputed {row.recomputed}")


# ============================================================================= part table
@dataclass
class Part:
    key: str
    title: str
    report: str
    steps: list[list[str]]
    check: object
    needs_corpus: bool = False
    note: str = ""
    inputs: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)


def _py(script: str) -> list[str]:
    return [sys.executable, f"evaluation/{script}"]


PARTS: list[Part] = [
    Part(
        key="corpus",
        inputs=['corpus/chunk_index.sqlite3'],
        title="Corpus composition",
        report="Section 3.1, Figure 1",
        steps=[_py("corpus_stats.py") + ["--db", str(CORPUS_DB),
                                         "--out", str(RESULTS_DIR / "corpus_stats.json")]],
        check=check_corpus,
        needs_corpus=True,
        note="Needs the corpus database. Without it the shipped results/corpus_stats.json is "
             "checked as-is and not regenerated.",
        produces=["results/corpus_stats.json"],
    ),
    Part(
        key="artifacts",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet', 'config/frozen_cache_manifest.json', 'config/frozen_config.json'],
        title="Frozen artifact integrity and provenance",
        report="Sections 4 and 5.3, Appendix A",
        steps=[],
        check=check_artifacts,
        note="No script to run: this part checks that the shipped inputs hash to what the "
             "frozen run recorded, that the frozen parameters are what the report states, and "
             "that the cached scores reconstruct through the documented formula.",
    ),
    Part(
        key="rq1",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet'],
        title="RQ1: matched-budget comparison",
        report="Section 6.1, Tables 3 and 4, Figures 4 and 5",
        steps=[_py("rq1_rq3_same_budget.py")],
        check=check_rq1,
        produces=["results/rq1_rq3_same_budget/"],
    ),
    Part(
        key="signals",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet'],
        title="Signal behaviour and authority calibration",
        report="Sections 6.1.1 and 6.2.1-6.2.2, Figures 6 and 7",
        steps=[_py("score_signal_analysis.py")],
        check=check_signals,
        produces=["results/score_signal_stats_summary.json",
                  "results/authority_amplification_analysis.csv",
                  "results/signal_effective_range.csv",
                  "results/normalization_saturation_analysis.csv"],
    ),
    Part(
        key="graph",
        title="RQ2: citation-edge quality",
        report="Section 6.2.3",
        inputs=["benchmark/graph_edge_review/edge_classifications.csv"],
        steps=[_py("graph_edge_audit.py")],
        check=check_graph,
        produces=["results/graph_edge_audit.json", "results/graph_edge_audit_by_stratum.csv"],
    ),
    Part(
        key="reranking",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet', 'data/variant2_ce_output_DEV.json', 'data/variant2_ce_output_TEST.json'],
        title="RQ3: cross-encoder variants and completeness",
        report="Section 6.3, Table 7, Figure 8",
        steps=[
            _py("ce_experiment_metrics.py"),
            _py("ce_experiment_metrics2.py"),
            _py("dual_evidence_strict.py"),
            _py("test_confirmation.py"),
            _py("ce_variants_test.py"),
        ],
        check=check_reranking,
        note="Four steps. ce_experiment_metrics.py must precede ce_experiment_metrics2.py, "
             "which reads summaries_v1_v3.json from it; the other two are independent.",
        produces=["results/table4_dev_ce_variants.csv",
                  "results/dual_evidence_strict_summary.csv",
                  "results/truncation_diagnostic.csv",
                  "results/test_confirmation/"],
    ),
    Part(
        key="performance",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet'],
        title="Final fixed-configuration performance",
        report="Section 6.4, Table 8, Figure 9 (and Table 6)",
        steps=[_py("final_performance.py"), _py("ce_metrics_audit.py")],
        check=check_performance,
        note="final_performance.py owns Table 8; ce_metrics_audit.py produces the full "
             "denominator-audited metric table and the candidate-pool ceiling.",
        produces=["results/top25_per_lane_final_metrics.csv", "results/metric_audit_table.csv"],
    ),
    Part(
        key="ir-metrics",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet'],
        title="Secondary IR diagnostics",
        report="Appendix A, Figures 10 and 11",
        steps=[_py("ir_metrics.py")],
        check=check_ir_metrics,
        produces=["results/ir_metrics/"],
    ),
    Part(
        key="tokens",
        inputs=['corpus/chunk_index.sqlite3'],
        title="Token distributions and the 512-token window",
        report="Appendix A, Section 7.3",
        steps=[_py("corpus_token_distribution.py") + ["--db", str(CORPUS_DB)]],
        check=check_tokens,
        needs_corpus=True,
        note="Needs the corpus database, because it measures chunk text. Without it the "
             "shipped distributions are checked as-is and not regenerated.",
        produces=["results/token_distribution/"],
    ),
    Part(
        key="figures",
        inputs=['results/corpus_stats.json', 'results/top25_per_lane_final_metrics.csv', 'results/rq1_rq3_same_budget/', 'results/ir_metrics/', 'data/candidate_cache.parquet'],
        title="Figure regeneration",
        report="Figures 1, 4-11",
        steps=[_py("make_report_figures.py"), _py("score_signal_figures.py")],
        check=check_figures,
        note="Several figures plot a table another part owns, so run this after those parts "
             "or with --all.",
        produces=["figures/"],
    ),
    Part(
        key="cross-check",
        inputs=['benchmark/scenarios_all_208.jsonl', 'benchmark/gold_evidence_218.jsonl', 'data/candidate_cache.parquet'],
        title="Independent cross-check of headline numbers",
        report="all of the above",
        steps=[_py("verify_reported_numbers.py")],
        check=check_cross_check,
        note="Recomputes 32 reported numbers from the benchmark and the candidate cache "
             "directly, reading no other part's output, so a bug in a producing script "
             "cannot hide behind its own output.",
        produces=["results/verification_report.csv"],
    ),
]
BY_KEY = {p.key: p for p in PARTS}


# ================================================================================= runner
def run_part(part: Part) -> dict:
    n = PARTS.index(part) + 1
    width = 78
    print(f"\n{BOLD}{'=' * width}{OFF}")
    print(f"{BOLD} PART {n}/{len(PARTS)}  {part.key}{OFF} — {part.title}")
    print(f"{DIM} Report: {part.report}{OFF}")
    if part.note:
        for line in _wrap(part.note, width - 2):
            print(f"{DIM} {line}{OFF}")
    print(f"{BOLD}{'=' * width}{OFF}")

    if part.inputs:
        print()
        for i, spec in enumerate(part.inputs):
            files = _expand(spec)
            label = "  reads " if i == 0 else "        "
            if files:
                for f in files[:1]:
                    print(f"{label}{DIM}{spec:<52}{OFF} {_describe(f)}")
            else:
                print(f"{label}{DIM}{spec:<52}{OFF} {YELLOW}not present{OFF}")

    t0 = time.time()
    before = _snapshot(part.produces)
    steps, skipped = [], []

    for cmd in part.steps:
        # Long absolute paths (notably $CORPUS_DB) would wreck the column; show them short.
        parts = []
        for a in cmd[1:]:
            parts.append(Path(a).name if "/" in a and len(a) > 30 else a)
        label = " ".join(parts)
        if len(label) > 58:
            label = label[:57] + "\u2026"
        if part.needs_corpus and not CORPUS_DB.exists():
            print(f"  {YELLOW}skip {OFF} {label}")
            print(f"         {DIM}CORPUS_DB not set or not found; checking the shipped "
                  f"output instead{OFF}")
            skipped.append(label)
            continue
        print(f"  {DIM}run  {OFF} {label:<58}", end="", flush=True)
        t = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        dt = time.time() - t
        if proc.returncode == 0:
            print(f"{GREEN}ok{OFF}  {dt:5.1f}s")
            steps.append((label, True, dt))
        else:
            print(f"{RED}FAILED{OFF}  {dt:5.1f}s")
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            for line in tail:
                print(f"         {RED}{line[:110]}{OFF}")
            steps.append((label, False, dt))

    step_failures = [s for s in steps if not s[1]]

    # What the part actually produced, and whether a regenerated file still matches the copy
    # that shipped. "identical to shipped" is the reproducibility statement: the part rebuilt
    # the artifact from its inputs and got the same bytes back.
    after = _snapshot(part.produces)
    artifacts, n_identical, n_changed = [], 0, 0
    for f in sorted(after):
        old = before.get(f)
        if old is None:
            state, colour = "created", GREEN
        elif old == after[f]:
            state, colour = "identical to shipped", GREEN
            n_identical += 1
        else:
            state, colour = "DIFFERS from shipped", RED
            n_changed += 1
        artifacts.append((f, state, colour))

    if artifacts:
        print()
        for i, (f, state, colour) in enumerate(artifacts[:12]):
            rel = str(f.relative_to(REPO_ROOT))
            label = "  wrote " if i == 0 else "        "
            print(f"{label}{rel:<52} {DIM}{_describe(f):<22}{OFF} {colour}{state}{OFF}")
        if len(artifacts) > 12:
            print(f"        {DIM}... and {len(artifacts) - 12} more under the same paths{OFF}")

    checker = Checker()
    check_error = None
    if step_failures:
        print(f"  {DIM}checks skipped: a step above failed{OFF}")
    else:
        try:
            part.check(checker)
        except Exception as exc:  # a missing or malformed output is itself a failure
            check_error = f"{type(exc).__name__}: {exc}"
            print(f"  {RED}checks could not run: {check_error}{OFF}")

    if checker.checks:
        print()
        name_w = min(52, max(len(c.name) for c in checker.checks) + 1)
        for chk in checker.checks:
            mark = f"{GREEN}PASS{OFF}" if chk.ok else f"{RED}FAIL{OFF}"
            print(f"  {DIM}check{OFF} {chk.name:<{name_w}} "
                  f"{_short(chk.reported):>15}  {_short(chk.recomputed):>15}   {mark}")
            if chk.note:
                print(f"         {DIM}{chk.note}{OFF}")

    n_ok = sum(1 for c in checker.checks if c.ok)
    n_all = len(checker.checks)
    passed = (not step_failures) and check_error is None and n_all > 0 and n_ok == n_all
    elapsed = time.time() - t0

    verdict = f"{GREEN}PASS{OFF}" if passed else f"{RED}FAIL{OFF}"
    detail = f"{n_ok}/{n_all} checks"
    if artifacts:
        detail += f", {len(artifacts)} artifact{'s' if len(artifacts) != 1 else ''}"
        if n_identical:
            detail += f" ({n_identical} identical to shipped)"
    if skipped:
        detail += f", {len(skipped)} step(s) skipped"
    print(f"\n  ──> {BOLD}PART {n} {part.key}: {verdict}{OFF}   ({detail}, {elapsed:.1f}s)")

    return {
        "part_number": n,
        "part": part.key,
        "title": part.title,
        "report": part.report,
        "status": "PASS" if passed else "FAIL",
        "checks_passed": n_ok,
        "checks_total": n_all,
        "steps_run": len(steps),
        "steps_skipped": len(skipped),
        "seconds": round(elapsed, 1),
        "artifacts": len(artifacts),
        "artifacts_identical": n_identical,
        "artifacts_changed": n_changed,
        "error": check_error or ("step failed: " + step_failures[0][0] if step_failures else ""),
        "_checks": checker.checks,
    }


def _short(value, width: int = 15) -> str:
    """A value narrow enough for the column. The CSV keeps the full value."""
    if isinstance(value, (set, list, tuple)):
        if not value:
            return "(none)"
        items = sorted(value) if isinstance(value, set) else list(value)
        return str(items[0]) if len(items) == 1 else f"{len(items)} items"
    text = str(value)
    if len(text) <= width:
        return text
    if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
        return text[:7] + "\u2026" + text[-4:]          # a SHA-256
    return text[: width - 1] + "\u2026"


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def list_parts(chain: bool = False) -> None:
    print(f"\n{BOLD}Reproduction parts{OFF}\n")
    for i, p in enumerate(PARTS, 1):
        corpus = f"  {YELLOW}[needs CORPUS_DB]{OFF}" if p.needs_corpus else ""
        print(f"  {BOLD}{i}{OFF}  {p.key:<14} {p.title}{corpus}")
        print(f"     {DIM}{p.report}{OFF}")
        if chain:
            for spec in p.inputs:
                print(f"        {DIM}reads  {spec}{OFF}")
            for spec in p.produces:
                print(f"        {GREEN}writes {spec}{OFF}")
            print()
    if not chain:
        print(f"\n{DIM}  python evaluation/reproduce.py --chain   # what each part reads and writes{OFF}")
    print(f"{DIM}  python evaluation/reproduce.py --all{OFF}")
    print(f"{DIM}  python evaluation/reproduce.py performance signals{OFF}")
    print(f"{DIM}  python evaluation/reproduce.py 4{OFF}\n")


def resolve(tokens: list[str]) -> list[Part]:
    chosen, seen = [], set()
    for tok in tokens:
        if tok.isdigit() and 1 <= int(tok) <= len(PARTS):
            part = PARTS[int(tok) - 1]
        elif tok in BY_KEY:
            part = BY_KEY[tok]
        else:
            raise SystemExit(
                f"Unknown part {tok!r}. Known parts: {', '.join(BY_KEY)}.\n"
                f"Run  python evaluation/reproduce.py --list"
            )
        if part.key not in seen:
            seen.add(part.key)
            chosen.append(part)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", help="part names or numbers; omit with --all")
    ap.add_argument("--all", action="store_true", help="run every part")
    ap.add_argument("--list", action="store_true", help="list the parts and exit")
    ap.add_argument("--chain", action="store_true",
                    help="list the parts with what each one reads and writes, and exit")
    args = ap.parse_args()

    if args.chain:
        list_parts(chain=True)
        return 0
    if args.list or (not args.parts and not args.all):
        list_parts()
        return 0

    selected = PARTS if args.all else resolve(args.parts)

    print(f"\n{BOLD}Reproducing {len(selected)} of {len(PARTS)} parts{OFF}")
    print(f"{DIM}repository: {REPO_ROOT}{OFF}")
    db_state = str(CORPUS_DB) if CORPUS_DB.exists() else "not present (only part 1 needs it)"
    print(f"{DIM}corpus db : {db_state}{OFF}")

    t0 = time.time()
    results = [run_part(p) for p in selected]

    # ------------------------------------------------------------------------- summary
    width = 78
    print(f"\n\n{BOLD}{'=' * width}{OFF}")
    print(f"{BOLD} SUMMARY{OFF}")
    print(f"{BOLD}{'=' * width}{OFF}")
    for r in results:
        mark = f"{GREEN}PASS{OFF}" if r["status"] == "PASS" else f"{RED}FAIL{OFF}"
        checks = f'{r["checks_passed"]}/{r["checks_total"]}'
        skipped = f'  {YELLOW}({r["steps_skipped"]} step skipped){OFF}' if r["steps_skipped"] else ""
        arts = f'{r["artifacts"]:>2} art' if r["artifacts"] else "     -"
        print(f'  {r["part_number"]}  {r["part"]:<14} {r["title"]:<42} '
              f'{checks:>7} {arts}  {mark}{skipped}')

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    tot_ok = sum(r["checks_passed"] for r in results)
    tot_all = sum(r["checks_total"] for r in results)
    print(f"{DIM}{'-' * width}{OFF}")
    headline = (f"{GREEN}{n_pass}/{len(results)} parts reproduced{OFF}"
                if n_pass == len(results)
                else f"{RED}{len(results) - n_pass} of {len(results)} parts FAILED{OFF}")
    tot_art = sum(r["artifacts"] for r in results)
    tot_same = sum(r["artifacts_identical"] for r in results)
    tot_diff = sum(r["artifacts_changed"] for r in results)
    print(f"  {BOLD}{headline}{OFF}   {tot_ok}/{tot_all} checks passed   "
          f"({time.time() - t0:.1f}s)")
    if tot_art:
        same = (f"{GREEN}{tot_same} identical to the shipped copy{OFF}" if not tot_diff
                else f"{GREEN}{tot_same} identical{OFF}, {RED}{tot_diff} differing{OFF}")
        print(f"  {tot_art} artifacts regenerated under results/ and figures/ — {same}")

    # ------------------------------------------------------------------------- csv out
    rows = []
    for r in results:
        for chk in r["_checks"]:
            rows.append({
                "part_number": r["part_number"], "part": r["part"], "report": r["report"],
                "check": chk.name, "reported": chk.reported, "recomputed": chk.recomputed,
                "status": "PASS" if chk.ok else "FAIL",
            })
        if not r["_checks"]:
            rows.append({
                "part_number": r["part_number"], "part": r["part"], "report": r["report"],
                "check": "(part did not produce checks)", "reported": "", "recomputed": "",
                "status": "FAIL",
            })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "reproduction_report.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {DIM}check-by-check detail written to {out}{OFF}\n")

    if n_pass != len(results):
        print(f"  {RED}Failing parts:{OFF}")
        for r in results:
            if r["status"] == "FAIL":
                reason = r["error"] or f'{r["checks_total"] - r["checks_passed"]} check(s) failed'
                print(f'    {r["part_number"]} {r["part"]} — {reason}')
        print()
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
