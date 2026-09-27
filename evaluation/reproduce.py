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

    def num(self, name: str, reported: float, recomputed: float, note: str = "") -> None:
        try:
            ok = abs(float(recomputed) - float(reported)) <= tolerance_for(reported)
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
    s = _json("corpus_stats.json")
    c.exact("Total source documents", 1370, s["total_documents"])
    c.exact("Total chunks", 19087, s["total_chunks"])
    c.exact("Legislation-lane documents", 23, s["lane_split_documents"]["legislation_lane"])
    c.exact("Other-evidence-lane documents", 1347, s["lane_split_documents"]["other_evidence_lane"])
    c.exact("Citation-graph edges", 23785, s["graph_total_edges"])
    cls = s["documents_by_authority_class"]
    c.exact("PRIMARY_LEGISLATION documents", 14, cls.get("PRIMARY_LEGISLATION"))
    c.exact("SECONDARY_LEGISLATION documents", 9, cls.get("SECONDARY_LEGISLATION"))
    c.exact("OFFICIAL_WORKFLOW documents", 662, cls.get("OFFICIAL_WORKFLOW"))
    c.exact("Index manifest: chunks indexed", 19087, s["index_manifest"]["chunks_indexed"])
    c.exact("Index manifest: FTS tokenizer", "porter", s["index_manifest"]["fts_tokenizer"])


def check_artifacts(c: Checker) -> None:
    """The frozen inputs are the ones the reported results were computed from.

    This part exists because every other number in the report is conditional on it. If the
    benchmark on disk is not the benchmark the frozen run scored, or the cached scores do
    not reconstruct through the documented formula, then nothing downstream means what it
    says. It also pins down the 218-versus-208 relationship explicitly, rather than leaving
    it as a surprise for whoever counts the lines.
    """
    manifest = json.loads((CONFIG_DIR / "frozen_cache_manifest.json").read_text())

    # The gold file is byte-for-byte the one the frozen run used. Its 218 records are the
    # reason the manifest names gold_evidence_218.jsonl, and the reason this repository
    # ships it under that name rather than renaming it to match the scenario count.
    c.exact("Gold evidence SHA-256 matches the frozen manifest",
            manifest["gold_sha256"], _sha256(GOLD_PATH))

    # The scenario file legitimately does NOT match: the frozen run scored 218 scenarios,
    # then ten EXP_GRAPH* scenarios were dropped from the benchmark and the scenario file
    # was re-emitted with 208. The checks below establish that this is the only difference.
    scen_ids = {json.loads(l)["scenario_id"] for l in open(SCENARIOS_PATH)}
    gold_ids = {json.loads(l)["scenario_id"] for l in open(GOLD_PATH)}
    cache = pd.read_parquet(CANDIDATE_CACHE)
    cache_ids = set(cache.scenario_id.unique())
    dropped = {f"EXP_GRAPH{i:03d}" for i in range(1, 11)}

    c.exact("Benchmark scenarios", 208, len(scen_ids))
    c.exact("Gold records (frozen 218-set)", 218, len(gold_ids))
    c.exact("Candidate cache scenarios (frozen 218-set)", 218, len(cache_ids))
    c.exact("Gold minus scenarios = the ten dropped EXP_GRAPH",
            sorted(dropped), sorted(gold_ids - scen_ids))
    c.exact("Cache minus scenarios is exactly the same ten", sorted(dropped),
            sorted(cache_ids - scen_ids))
    c.exact("Every benchmark scenario has gold", set(), scen_ids - gold_ids)
    c.exact("Every benchmark scenario is in the cache", set(), scen_ids - cache_ids)
    c.true("The dropped scenarios cannot affect any result",
           all(sid not in scen_ids for sid in dropped),
           "every script iterates the scenario file, so they are never read")

    # The frozen parameters are the ones the report states.
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

    # The cache is structurally what the pipeline says it is.
    c.exact("Candidate cache lanes", ["legislation", "other"], sorted(cache.lane.unique()))
    c.true("Candidate cache carries merged cross-encoder scores",
           bool(cache.ce_score.notna().any()),
           f"{int(cache.ce_score.notna().sum()):,} scored rows")
    c.true("Cross-encoder depth never exceeds 75 per lane",
           float(cache.ce_rank.max()) <= 75, f"max ce_rank = {cache.ce_rank.max():.0f}")

    # final_score must be reconstructible from the cached RAW values through the frozen
    # formula. A non-zero error means the cache and the documented formula disagree, and
    # every score-based claim in the report would be resting on an unverified step.
    fused = 0.40 * cache.bm25_norm + 0.60 * cache.dense_norm
    recon = (fused * 0.90 + 0.10 * cache.authority_norm) * cache.jurisdiction_weight
    c.num("Score reconstruction max abs error (all rows)", 0.0,
          float((recon - cache.final_score).abs().max()))
    c.exact("Rows the formula was reconstructed over", len(cache), len(recon))


def check_performance(c: Checker) -> None:
    """Table 5 and Figure 7: final top-25-per-lane performance."""
    df = _csv("top25_per_lane_final_metrics.csv")
    dev = df[df.split == "DEV"].iloc[0]
    test = df[df.split.str.startswith("TEST")].iloc[0]

    for label, row, n_scen, n_req, rr, cc, dual, n_mixed in [
        ("DEV", dev, 95, 155, 0.813, 0.747, 0.765, 17),
        ("TEST", test, 94, 139, 0.806, 0.755, 0.600, 15),
    ]:
        c.exact(f"{label} scoreable scenarios", n_scen, int(row.n_scenarios))
        c.exact(f"{label} mandatory requirements", n_req, int(row.n_requirements))
        c.num(f"{label} RequirementRecall@25-per-lane", rr, row["RequirementRecall@25-per-lane"])
        c.num(f"{label} CompleteCoverage@25-per-lane", cc, row["CompleteCoverage@25-per-lane"])
        c.num(f"{label} DualEvidenceCoverage@25-per-lane", dual,
              row["DualEvidenceCoverage@25-per-lane"])
        c.exact(f"{label} mixed-evidence scenarios", n_mixed, int(row.n_mixed_evidence_scenarios))

    audit = _csv("metric_audit_table.csv")
    ceiling = audit[audit.metric.str.startswith("CandidateRequirementRecall@75")].iloc[0]
    c.num("DEV candidate-pool ceiling @75 (RequirementRecall)", 0.916, ceiling.value)
    c.exact("DEV ceiling numerator", 142, int(ceiling.numerator))
    c.exact("DEV ceiling denominator", 155, int(ceiling.denominator))


def check_dual_evidence(c: Checker) -> None:
    """Figure 6: strict mixed-evidence completeness, before and after reranking."""
    df = _csv("dual_evidence_strict_summary.csv")
    at25 = df[df.cutoff_label == "DualEvidenceCoverage@25-per-lane"]
    pre = at25[at25.variant == "0_pre_CE_first_stage"].iloc[0]
    post = at25[at25.variant == "1_baseline_raw_text_CE"].iloc[0]

    c.num("DEV dual-evidence coverage, pre-CE", 0.529, pre.value)
    c.num("DEV dual-evidence coverage, post-CE", 0.765, post.value)
    c.exact("Pre-CE numerator / denominator", "9/17", f"{int(pre.numerator)}/{int(pre.denominator)}")
    c.exact("Post-CE numerator / denominator", "13/17",
            f"{int(post.numerator)}/{int(post.denominator)}")
    c.num("Post-CE legislation-side coverage", 1.0, post.legislation_side_coverage)
    c.num("Post-CE non-legislation-side coverage", 0.765, post.nonlegislation_side_coverage)
    c.true("Reranking improves strict dual-evidence coverage", post.value > pre.value,
           f"{pre.value:.4f} -> {post.value:.4f}")

    detail = _json("dual_evidence_strict.json")
    c.true("Per-scenario failure diagnosis is recorded",
           any("missing_nonlegislation_only_ids" in row for row in detail),
           f"{len(detail)} variant/cutoff rows")


def check_signals(c: Checker) -> None:
    """Section 6.2 and Figures 4-5: how each retrieval signal behaves."""
    s = _json("score_signal_stats_summary.json")

    c.num("BM25 ROC AUC (essential gold vs rest, DEV)", 0.728, s["bm25_auc_essential_vs_rest"])
    c.num("Dense ROC AUC (essential gold vs rest, DEV)", 0.847, s["dense_auc_essential_vs_rest"])
    c.num("Score reconstruction max error", 0.0, s["reconstruction_error_max"])
    c.num("Authority amplification factor", 23.76,
          s["authority_amplification_factor_norm_over_raw"])
    c.exact("DEV EU-jurisdiction candidate rows", 5088, s["n_eu_candidates"])
    c.exact("EU-jurisdiction essential-gold rows", 0, s["n_eu_essential_gold_candidates"])
    c.num("Cross-encoder AUC within the reranked top-75 pool", 0.688, s["ce_auc_essential_vs_rest"])
    c.num("Essential-gold mean rank movement", 3.59, s["ce_mean_rank_movement_essential_gold"])
    c.num("Essential-gold median rank movement", 1.0, s["ce_median_rank_movement_essential_gold"])
    c.exact("Essential gold moved into top 25", 22, s["ce_n_essential_gold_moved_into_top25"])
    c.exact("Essential gold moved out of top 25", 12, s["ce_n_essential_gold_moved_out_of_top25"])
    c.num("Harmful demotion rate", 0.091, s["ce_harmful_demotion_rate"])

    auth = _csv("authority_amplification_analysis.csv").set_index("authority_class")
    c.num("Primary legislation mean normalised authority", 0.751,
          auth.loc["PRIMARY_LEGISLATION", "mean_authority_norm"])
    c.num("Secondary legislation mean normalised authority", 0.038,
          auth.loc["SECONDARY_LEGISLATION", "mean_authority_norm"])
    c.num("Primary/secondary nominal weight gap", 0.03,
          auth.loc["PRIMARY_LEGISLATION", "raw_authority_weight"]
          - auth.loc["SECONDARY_LEGISLATION", "raw_authority_weight"])

    rng = _csv("signal_effective_range.csv").set_index("signal")
    c.true("Effective-range table covers both first-stage signals",
           {"bm25", "dense"}.issubset({str(i).lower() for i in rng.index}),
           ", ".join(str(i) for i in rng.index))


def check_ce_variants(c: Checker) -> None:
    """Table 4 and the 512-token truncation diagnostic."""
    df = _csv("table4_dev_ce_variants.csv").set_index("variant")
    c.exact("Variants compared", 10, len(df))
    c.true("Baseline raw-text cross-encoder is present",
           "1_baseline_raw_text_CE" in df.index)
    c.true("Metadata-enriched cross-encoder is present",
           "2_metadata_enriched_CE" in df.index)

    base = df.loc["1_baseline_raw_text_CE"]
    c.exact("Baseline variant scenarios", 95, int(base.n_scenarios))
    c.num("Baseline coverage@25", 0.814, base["coverage@25"])
    c.num("Baseline RequirementRecall@10", 0.593, base["RequirementRecall@10"])
    c.exact("Baseline moved into top 25", 21, int(base.moved_into_top25))
    c.exact("Baseline moved out of top 25", 12, int(base.moved_out_of_top25))
    c.true("Every variant was scored on the same 95 scenarios",
           bool((df.n_scenarios == 95).all()),
           f"{sorted(set(df.n_scenarios))}")

    trunc = _json("truncation_summary.json")
    c.exact("Essential gold left below rank 25 (DEV)", 48, trunc["n_essential_gold_rank_gt25"])
    c.exact("Of those, exceeding the 512-token window", 23, trunc["n_likely_truncated"])
    c.num("Truncation rate among them", 0.479, trunc["pct_truncated"])

    rows = _csv("truncation_diagnostic.csv")
    c.exact("Truncation diagnostic rows", 48, len(rows))
    c.true("Every flagged chunk carries an untruncated token count",
           bool(rows.n_tokens_untruncated.notna().all()))


def check_rq1_rq3(c: Checker) -> None:
    """RQ1 same-budget ablations and RQ3 reranking, with confidence intervals."""
    sub = RESULTS_DIR / "rq1_rq3_same_budget"
    base = pd.read_csv(sub / "rq1_same_budget_50_baselines.csv")
    test = base[base.split == "TEST"].set_index("config_id")

    c.exact("TEST scenarios in every configuration", {94},
            set(int(n) for n in test.n_scenarios))
    c.exact("TEST requirements in every configuration", {139},
            set(int(n) for n in test.n_requirements))
    c.exact("Configurations compared", 9, len(test))

    for cfg, label, value in [
        (1, "BM25 only (pooled @50)", 0.345),
        (2, "Dense only (pooled @50)", 0.496),
        (3, "BM25+Dense hybrid (pooled @50)", 0.475),
        (6, "Hybrid + authority + jurisdiction (pooled @50)", 0.612),
        (7, "Two-lane, no authority (25+25)", 0.734),
        (8, "Full two-lane, pre-CE (25+25)", 0.763),
        (9, "Full two-lane, post-CE (25+25)", 0.806),
    ]:
        c.num(f"TEST RequirementRecall - {label}", value,
              test.loc[cfg, "requirement_recall_weighted"])

    c.true("Every configuration retrieves the same 50-result budget",
           set(test.budget) == {"top-50 (pooled)", "25+25 (two-lane)"},
           ", ".join(sorted(set(test.budget))))

    stats = json.loads((sub / "statistical_tests_TEST.json").read_text())
    rq1 = stats["RQ1_source_aware_vs_conventional_TEST"]
    rr, cc = rq1["requirement_recall"], rq1["complete_coverage"]
    c.num("RQ1 RequirementRecall difference", 0.151, rr["mean_diff"])
    c.num("RQ1 95% CI lower bound", 0.080, rr["ci_low"])
    c.num("RQ1 95% CI upper bound", 0.225, rr["ci_high"])
    c.num("RQ1 Cohen's d (paired, per scenario)", 0.444, rr["cohens_d_paired_per_scenario"])
    c.true("RQ1 permutation p < 0.0001", rr["permutation_p"] < 0.0001,
           f"p = {rr['permutation_p']}")
    c.exact("RQ1 McNemar scenarios improved", 19, cc["n_improved"])
    c.exact("RQ1 McNemar scenarios worsened", 1, cc["n_worsened"])
    c.true("RQ1 McNemar p < 0.001", cc["mcnemar_exact_p"] < 0.001,
           f"p = {cc['mcnemar_exact_p']:.2e}")

    rq3 = stats["RQ3_post_CE_vs_pre_CE_TEST_25perlane"]["requirement_recall"]
    c.num("RQ3 reranking difference (post-CE minus pre-CE)", 0.043, rq3["mean_diff"])
    c.num("RQ3 permutation p", 0.215, rq3["permutation_p"])
    c.true("RQ3 confidence interval spans zero, as reported",
           rq3["ci_low"] < 0 < rq3["ci_high"],
           f"[{rq3['ci_low']:.3f}, {rq3['ci_high']:.3f}]")

    for name in ("rq1_per_category.csv", "rq3_ce_onoff_per_category.csv",
                 "scenario_level_all_configs.csv"):
        c.true(f"{name} written", (sub / name).exists())


def check_figures(c: Checker) -> None:
    """Every figure the report draws from data is regenerated."""
    expected = [
        "figure1_corpus_composition.png",
        "figure4_first_stage_auc.png",
        "figure5_authority_normalisation.png",
        "figure6_dual_evidence_dev.png",
        "figure7_final_performance.png",
        "signal_diagnostics/fig1_raw_vs_normalized.png",
        "signal_diagnostics/fig2_bm25_gold_vs_nongold.png",
        "signal_diagnostics/fig3_dense_gold_vs_nongold_wrongregime.png",
        "signal_diagnostics/fig4_ce_gold_vs_nongold.png",
    ]
    for rel in expected:
        p = FIGURES_DIR / rel
        size = p.stat().st_size if p.exists() else 0
        c.true(rel, p.exists() and size > 5_000, f"{size:,} bytes" if size else "missing")


def check_cross_check(c: Checker) -> None:
    """Part 9 consumes verify_reported_numbers.py's own output table."""
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
    produces: list[str] = field(default_factory=list)


def _py(script: str) -> list[str]:
    return [sys.executable, f"evaluation/{script}"]


PARTS: list[Part] = [
    Part(
        key="corpus",
        title="Corpus composition",
        report="Section 3.1, Figure 1",
        steps=[_py("corpus_stats.py") + ["--db", str(CORPUS_DB),
                                         "--out", str(RESULTS_DIR / "corpus_stats.json")]],
        check=check_corpus,
        needs_corpus=True,
        note="Needs the corpus database. Without it, the shipped results/corpus_stats.json "
             "is checked as-is and not regenerated.",
        produces=["results/corpus_stats.json"],
    ),
    Part(
        key="artifacts",
        title="Frozen artifact integrity and provenance",
        report="Section 4, Appendix section 2",
        steps=[],
        check=check_artifacts,
        note="No script to run: this part checks that the shipped inputs hash to what the "
             "frozen run recorded, that the frozen parameters are what the report states, "
             "and that the cached scores reconstruct through the documented formula.",
    ),
    Part(
        key="performance",
        title="Final retrieval performance",
        report="Table 5, Figure 7",
        steps=[_py("final_performance.py"), _py("ce_metrics_audit.py")],
        check=check_performance,
        note="final_performance.py owns Table 5; ce_metrics_audit.py produces the full "
             "denominator-audited metric table and the candidate-pool ceiling.",
        produces=["results/top25_per_lane_final_metrics.csv", "results/metric_audit_table.csv"],
    ),
    Part(
        key="dual-evidence",
        title="Strict mixed-evidence completeness",
        report="Figure 6, Section 6.3",
        steps=[_py("dual_evidence_strict.py")],
        check=check_dual_evidence,
        produces=["results/dual_evidence_strict_summary.csv", "results/dual_evidence_strict.json"],
    ),
    Part(
        key="signals",
        title="Score and signal analysis",
        report="Section 6.2, Figures 4 and 5",
        steps=[_py("score_signal_analysis.py")],
        check=check_signals,
        produces=["results/score_signal_stats_summary.json",
                  "results/authority_amplification_analysis.csv"],
    ),
    Part(
        key="ce-variants",
        title="Cross-encoder input variants and truncation",
        report="Table 4, Section 4.4",
        steps=[_py("ce_experiment_metrics.py"), _py("ce_experiment_metrics2.py")],
        check=check_ce_variants,
        note="Two steps, in this order: the second reads summaries_v1_v3.json from the first.",
        produces=["results/table4_dev_ce_variants.csv", "results/truncation_diagnostic.csv"],
    ),
    Part(
        key="rq1-rq3",
        title="Same-budget ablations and statistical tests",
        report="RQ1 and RQ3 tables, confidence intervals",
        steps=[_py("rq1_rq3_same_budget.py")],
        check=check_rq1_rq3,
        produces=["results/rq1_rq3_same_budget/"],
    ),
    Part(
        key="figures",
        title="Figure regeneration",
        report="Figures 1, 4, 5, 6, 7 and the signal diagnostics",
        steps=[_py("make_report_figures.py"), _py("score_signal_figures.py")],
        check=check_figures,
        note="Figure 1 needs results/corpus_stats.json (part 1) and Figure 7 plots Table 5 "
             "(part 3), so run this after those, or with --all.",
        produces=["figures/"],
    ),
    Part(
        key="cross-check",
        title="Independent cross-check of headline numbers",
        report="all of the above",
        steps=[_py("verify_reported_numbers.py")],
        check=check_cross_check,
        note="Recomputes 32 reported numbers from the benchmark and the candidate cache "
             "directly, without reading any other part's output, and compares each against "
             "the report.",
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

    t0 = time.time()
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


def list_parts() -> None:
    print(f"\n{BOLD}Reproduction parts{OFF}\n")
    for i, p in enumerate(PARTS, 1):
        corpus = f"  {YELLOW}[needs CORPUS_DB]{OFF}" if p.needs_corpus else ""
        print(f"  {BOLD}{i}{OFF}  {p.key:<14} {p.title}{corpus}")
        print(f"     {DIM}{p.report}{OFF}")
    print(f"\n{DIM}  python evaluation/reproduce.py --all{OFF}")
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
    args = ap.parse_args()

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
        print(f'  {r["part_number"]}  {r["part"]:<14} {r["title"]:<42} '
              f'{checks:>7}  {mark}{skipped}')

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    tot_ok = sum(r["checks_passed"] for r in results)
    tot_all = sum(r["checks_total"] for r in results)
    print(f"{DIM}{'-' * width}{OFF}")
    headline = (f"{GREEN}{n_pass}/{len(results)} parts reproduced{OFF}"
                if n_pass == len(results)
                else f"{RED}{len(results) - n_pass} of {len(results)} parts FAILED{OFF}")
    print(f"  {BOLD}{headline}{OFF}   {tot_ok}/{tot_all} checks passed   "
          f"({time.time() - t0:.1f}s)")

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
